import os
import asyncio
import time
import aiohttp
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Callable, Any, Set

from telethon import functions, types
from telethon.errors import RPCError
from telethon.helpers import add_surrogate
from telethon.tl.functions.channels import EditPhotoRequest, EditTitleRequest
from telethon.utils import get_input_photo
from telethon.tl.types import (
    PeerChannel,
    MessageEntityTextUrl,
    MessageEntityBlockquote,
    MessageEntityBold,
    MessageEntityCode,
    MessageEntityUnderline,
    InputChatUploadedPhoto,
    Photo,
    InputFile,
    InputFileBig,
)

from constants import (
    ALBUM_LABEL_ALBUM,
    ALBUM_LABEL_SINGLE,
    DEFAULT_LOGO_PATH,
    DEFAULT_MESSAGE,
    DIRS,
    ERROR_CHANNEL_INFO_UPDATE,
    ERROR_CHANNEL_PHOTO_UPDATE,
    ERROR_CHANNEL_TITLE_UPDATE,
    ERROR_COVER_UPLOAD,
    ERROR_FALLBACK_RESET,
    ERROR_MEDIA_UPDATE,
    ERROR_TEXT_UPDATE,
    ERROR_UPDATE_LOOP,
    ERROR_UPDATING_MESSAGES,
    INFO_CHANNEL_DEFAULT_UPDATED,
    INFO_CHANNEL_UPDATED,
    INFO_COVER_DOWNLOADED_FALLBACK1,
    INFO_COVER_DOWNLOADED_FALLBACK2,
    INFO_COVER_DOWNLOADED_PRIMARY,
    INFO_FALLBACK_TRACK_INFO,
    INFO_MEDIA_DEFAULT_UPDATED,
    INFO_MEDIA_UPDATED,
    INFO_NO_TRACK_PLAYING,
    INFO_TRACK_PLAYING,
    LYRICS_ON_GENIUS,
    NOW_PLAYING_ON_SPOTIFY,
    REASON_TELEGRAM_UPDATE_FAILED,
    RELEASE_DATE_PREFIX,
)
from logger import _safe_log_info, _safe_log_error, _safe_log_warning, dev_logger
from utils.telethon_utils import safe_call_telethon
from utils.http_session import init_session
from lyrics.genius_search import search_lyrics_url
from lyrics.patterns.base import split_artist_text_by_feat_markers
from config import Settings
from utils.spotify_client import get_spg_cache_handler

_settings: Optional[Settings] = None

def set_settings(settings: Settings) -> None:
    global _settings
    _settings = settings

def get_settings() -> Settings:
    if _settings is None:
        raise RuntimeError("Settings are not initialized")
    return _settings

previous_track_id = None
was_playing = None
last_text_message_content = None
default_media_applied = False
track_info_cache = None
force_default_requested = False
updates_suspended = False
suspended_on_track_id = None
suspended_since = None
last_successful_cover_url = None

_CACHE_HANDLER = None
_NORMALIZE_FILES_FN: Optional[Callable] = None

@dataclass
class _ChannelCommitPayload:
    track_title: str
    is_playing: bool
    cover_url: Optional[str]
    track_id: Optional[str]
    artist_names: Optional[str]
    track_name: Optional[str]
    album_name: Optional[str]
    release_date: Optional[str]
    artist_urls: Optional[list[str]]
    track_url: Optional[str]
    album_url: Optional[str]
    current_track_raw: Optional[dict]
    genius_url: Optional[str] = None
    uploaded_photo: Any = None
    cover_media_unchanged: bool = False
    prepared_message_content: Optional[str] = None
    prepared_entities: Optional[list] = None

@dataclass
class _CoverPreuploadResult:
    uploaded_photo: Any
    source: str
    errors: tuple[str, ...] = ()

def _get_cache_handler():
    global _CACHE_HANDLER
    if _CACHE_HANDLER is None:
        cache_path = os.path.join(DIRS["SESSION"], "spotify", ".cache")
        _CACHE_HANDLER = get_spg_cache_handler(cache_path)
    return _CACHE_HANDLER

def _get_normalize_files_fn() -> Optional[Callable]:
    global _NORMALIZE_FILES_FN
    if _NORMALIZE_FILES_FN is not None:
        return _NORMALIZE_FILES_FN
    try:
        from lyrics.patterns.base import normalize_files
    except ImportError:
        return None
    if callable(normalize_files):
        _NORMALIZE_FILES_FN = normalize_files
        return _NORMALIZE_FILES_FN
    return None

_MAX_COVER_BYTES = 15 * 1024 * 1024
_MAX_TITLE_LENGTH = 128
_UPDATE_ERROR_STATE = {"ts": 0.0, "sig": ""}

def _is_allowed_cover_host(host: Optional[str]) -> bool:
    value = (host or "").lower()
    return (
        value == "scdn.co"
        or value.endswith(".scdn.co")
        or value == "spotifycdn.com"
        or value.endswith(".spotifycdn.com")
    )

async def _download_bytes(session, url: str) -> bytes:
    async with session.get(url) as resp:
        if not (200 <= resp.status < 300):
            raise RuntimeError(f"HTTP {resp.status}")
        if not _is_allowed_cover_host(getattr(resp.url, "host", None)):
            raise RuntimeError("unexpected cover host")
        content_type = (resp.headers.get("Content-Type") or "").lower()
        if content_type and not content_type.startswith("image/"):
            raise RuntimeError("unexpected cover content type")
        chunks = []
        total = 0
        async for chunk in resp.content.iter_chunked(65536):
            total += len(chunk)
            if total > _MAX_COVER_BYTES:
                raise RuntimeError("cover is too large")
            chunks.append(chunk)
        data = b"".join(chunks)
        if not data:
            raise RuntimeError("empty body")
        return data

def _utf16_len(value: str) -> int:
    return len(add_surrogate(value))

def _truncate_utf16(value: str, limit: int) -> str:
    if _utf16_len(value) <= limit:
        return value
    out: list[str] = []
    total = 0
    for char in value:
        char_len = _utf16_len(char)
        if total + char_len > limit - 1:
            break
        out.append(char)
        total += char_len
    return "".join(out) + "…"

async def _preupload_cover_photo_external(bot_client, cover_url: str, channel: PeerChannel) -> Optional[_CoverPreuploadResult]:
    try:
        input_photo = await asyncio.wait_for(
            safe_call_telethon(
                bot_client,
                functions.messages.UploadMediaRequest(
                    peer=channel,
                    media=types.InputMediaPhotoExternal(url=cover_url),
                ),
            ),
            timeout=5.0,
        )
        if input_photo and getattr(input_photo, "photo", None):
            return _CoverPreuploadResult(uploaded_photo=input_photo.photo, source="primary")
    except asyncio.CancelledError:
        raise
    except (aiohttp.ClientError, RPCError, ConnectionError, asyncio.TimeoutError, OSError) as e:
        error_text = str(e).strip() or e.__class__.__name__
        return _CoverPreuploadResult(uploaded_photo=None, source="primary", errors=(f"primary: {error_text}",))
    return None

async def _preupload_cover_photo_fallback(bot_client, cover_url: str) -> Optional[_CoverPreuploadResult]:
    session = init_session()
    cover_data: Optional[bytes] = None
    error_messages: list[str] = []
    try:
        cover_data = await asyncio.wait_for(_download_bytes(session, cover_url), timeout=5.0)
    except asyncio.CancelledError:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError) as e:
        error_text = str(e).strip() or e.__class__.__name__
        error_messages.append(f"download: {error_text}")

    if cover_data is not None:
        try:
            up_file = await asyncio.wait_for(
                bot_client.upload_file(cover_data, file_name="cover.jpg"),
                timeout=5.0,
            )
            return _CoverPreuploadResult(uploaded_photo=up_file, source="fallback1")
        except asyncio.CancelledError:
            raise
        except (asyncio.TimeoutError, OSError, RPCError, ConnectionError) as e:
            error_text = str(e).strip() or e.__class__.__name__
            error_messages.append(f"upload bytes: {error_text}")

    cover_path: Optional[str] = None
    try:
        if cover_data is None:
            cover_data = await asyncio.wait_for(_download_bytes(session, cover_url), timeout=5.0)
        with tempfile.NamedTemporaryFile("wb", suffix=".jpg", dir=DIRS["SESSION"], delete=False) as f:
            f.write(cover_data)
            cover_path = f.name
        up_file = await asyncio.wait_for(
            bot_client.upload_file(cover_path),
            timeout=5.0,
        )
        return _CoverPreuploadResult(uploaded_photo=up_file, source="fallback2")
    except asyncio.CancelledError:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError, RPCError, ConnectionError) as e:
        error_text = str(e).strip() or e.__class__.__name__
        error_messages.append(f"upload file: {error_text}")
        return _CoverPreuploadResult(uploaded_photo=None, source="fallback", errors=tuple(dict.fromkeys(error_messages)))
    finally:
        try:
            if cover_path and os.path.exists(cover_path):
                os.remove(cover_path)
        except OSError:
            pass

async def _preupload_cover_photo(bot_client, cover_url: str, channel: PeerChannel):
    external_task = asyncio.create_task(_preupload_cover_photo_external(bot_client, cover_url, channel))
    fallback_task = asyncio.create_task(_preupload_cover_photo_fallback(bot_client, cover_url))
    pending_tasks: Set[asyncio.Task] = {external_task, fallback_task}
    error_messages: list[str] = []
    try:
        while pending_tasks:
            done, pending = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
            pending_tasks = set(pending)
            for done_task in done:
                try:
                    result = await done_task
                except asyncio.CancelledError:
                    for task in pending_tasks:
                        task.cancel()
                    await asyncio.gather(*pending_tasks, return_exceptions=True)
                    raise
                except BaseException as e:
                    error_text = str(e).strip() or e.__class__.__name__
                    error_messages.append(f"preupload task: {error_text}")
                    continue
                if not isinstance(result, _CoverPreuploadResult):
                    continue
                if result.uploaded_photo is None:
                    if result.errors:
                        error_messages.extend(result.errors)
                    continue
                if result.source == "primary":
                    _safe_log_info(INFO_COVER_DOWNLOADED_PRIMARY)
                elif result.source == "fallback1":
                    _safe_log_info(INFO_COVER_DOWNLOADED_FALLBACK1)
                elif result.source == "fallback2":
                    _safe_log_info(INFO_COVER_DOWNLOADED_FALLBACK2)
                for task in pending_tasks:
                    task.cancel()
                await asyncio.gather(*pending_tasks, return_exceptions=True)
                return result.uploaded_photo
        if error_messages:
            _safe_log_error(ERROR_COVER_UPLOAD, "; ".join(dict.fromkeys(error_messages)))
        return None
    except asyncio.CancelledError:
        for task in pending_tasks:
            task.cancel()
        await asyncio.gather(*pending_tasks, return_exceptions=True)
        raise

def _normalize_track_info(data: Optional[dict]) -> dict:
    if not isinstance(data, dict):
        return {"track_id": None}
    return {
        "track_id": data.get("track_id"),
    }

_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

def _release_date_base(release_date: Optional[str], precision: Optional[str]) -> Optional[str]:
    if not isinstance(release_date, str) or not release_date:
        return None
    base = release_date
    if precision == "year":
        base = f"{release_date}-01-01"
    elif precision == "month":
        base = f"{release_date}-01"
    return base

def _album_total_tracks_from_item(item: Optional[dict]) -> Optional[int]:
    if not isinstance(item, dict):
        return None
    album = item.get("album") or {}
    total = album.get("total_tracks")
    if isinstance(total, int) and total > 0:
        return total
    return None

def _extract_spotify_track_meta(current_track_raw: Optional[dict]) -> tuple[Optional[list[dict]], Optional[int], Optional[str]]:
    if not isinstance(current_track_raw, dict):
        return None, None, None
    item = current_track_raw.get("item")
    if not isinstance(item, dict):
        return None, None, None
    spotify_artists_raw = item.get("artists") or None
    spotify_artists: Optional[list[dict]] = None
    if isinstance(spotify_artists_raw, list):
        spotify_artists = []
        for artist in spotify_artists_raw:
            if not isinstance(artist, dict):
                continue
            split_names = split_artist_text_by_feat_markers(artist.get("name"))
            if len(split_names) < 2:
                spotify_artists.append(artist)
                continue
            for name in split_names:
                clean_name = str(name).strip()
                if clean_name:
                    spotify_artists.append({"name": clean_name})
    album_total_tracks = _album_total_tracks_from_item(item)
    album_obj = item.get("album") if isinstance(item.get("album"), dict) else {}
    spotify_release_date_iso = _release_date_base(
        album_obj.get("release_date"),
        album_obj.get("release_date_precision"),
    )
    return spotify_artists, album_total_tracks, spotify_release_date_iso

def _select_spotify_cover_url(images: Any) -> Optional[str]:
    if not isinstance(images, list) or not images:
        return None
    candidates: list[tuple[int, int, int, str]] = []
    for idx, image in enumerate(images):
        if not isinstance(image, dict):
            continue
        url = image.get("url")
        if not isinstance(url, str) or not url:
            continue
        width = image.get("width")
        height = image.get("height")
        w = width if isinstance(width, int) and width > 0 else 0
        h = height if isinstance(height, int) and height > 0 else 0
        area = w * h
        candidates.append((area, idx, min(w, h), url))
    if not candidates:
        return None
    suitable = [item for item in candidates if item[2] >= 2000]
    if suitable:
        return min(suitable, key=lambda item: (item[0], item[1]))[3]
    return max(candidates, key=lambda item: (item[0], -item[1]))[3]

async def _collect_channel_commit_payload(
    bot_client,
    *,
    st: Settings,
    track_title: str,
    cover_url: Optional[str],
    track_id: Optional[str],
    is_playing: bool,
    artist_names,
    track_name,
    album_name,
    release_date,
    artist_urls,
    track_url,
    album_url,
    current_track_raw,
    track_changed: bool,
) -> _ChannelCommitPayload:
    payload = _ChannelCommitPayload(
        track_title=track_title,
        is_playing=is_playing,
        cover_url=cover_url,
        track_id=track_id,
        artist_names=artist_names,
        track_name=track_name,
        album_name=album_name,
        release_date=release_date,
        artist_urls=artist_urls,
        track_url=track_url,
        album_url=album_url,
        current_track_raw=current_track_raw,
    )

    if not (is_playing and track_changed):
        return payload

    lyrics_task: Optional[asyncio.Task] = None
    cover_task: Optional[asyncio.Task] = None
    phase1_tasks: list[asyncio.Task] = []

    if cover_url:
        channel = PeerChannel(st.tg_channel_id)
        if cover_url == last_successful_cover_url:
            payload.cover_media_unchanged = True
        else:
            cover_task = asyncio.create_task(_preupload_cover_photo(bot_client, cover_url, channel))
            phase1_tasks.append(cover_task)

    if st.enable_genius and st.genius_access_token and track_id:
        spotify_artists, album_total_tracks, spotify_release_date_iso = _extract_spotify_track_meta(current_track_raw)

        async def _lyrics_task():
            res = await search_lyrics_url(
                track_name,
                spotify_artists=spotify_artists,
                spotify_album_name=album_name,
                spotify_album_total_tracks=album_total_tracks,
                spotify_release_date=spotify_release_date_iso,
            )
            if isinstance(res, str) and res:
                return res
            return None

        lyrics_task = asyncio.create_task(_lyrics_task())
        phase1_tasks.append(lyrics_task)

    if phase1_tasks:
        try:
            phase1_results = await asyncio.gather(*phase1_tasks, return_exceptions=True)
        except asyncio.CancelledError:
            for task in phase1_tasks:
                try:
                    if isinstance(task, asyncio.Task) and not task.done():
                        task.cancel()
                except RuntimeError:
                    pass
            await asyncio.gather(*phase1_tasks, return_exceptions=True)
            raise

        phase1_results_by_task = {
            task: result
            for task, result in zip(phase1_tasks, phase1_results)
        }
        if lyrics_task is not None:
            lyrics_result = phase1_results_by_task.get(lyrics_task)
            if isinstance(lyrics_result, BaseException):
                _safe_log_warning(ERROR_UPDATE_LOOP, lyrics_result, logger=dev_logger)
            elif isinstance(lyrics_result, str):
                payload.genius_url = lyrics_result

        if cover_task is not None:
            cover_result = phase1_results_by_task.get(cover_task)
            if isinstance(cover_result, BaseException):
                _safe_log_warning(ERROR_UPDATE_LOOP, cover_result, logger=dev_logger)
            elif isinstance(cover_result, (Photo, InputFile, InputFileBig)):
                payload.uploaded_photo = cover_result

    payload.prepared_message_content, payload.prepared_entities = _build_playing_message_payload(
        artist_names,
        track_name,
        album_name,
        release_date,
        artist_urls,
        track_url,
        album_url,
        current_track=current_track_raw,
        genius_url=payload.genius_url,
    )
    return payload

async def _commit_channel_commit_payload(bot_client, payload: _ChannelCommitPayload, *, track_changed: bool) -> bool:
    global force_default_requested, last_successful_cover_url
    if not track_changed:
        return True

    commit_operations: list[Any] = []
    commit_ok = await update_messages(
        bot_client=bot_client,
        cover_url=payload.cover_url,
        is_playing=payload.is_playing,
        artist_names=payload.artist_names,
        track_name=payload.track_name,
        album_name=payload.album_name,
        release_date=payload.release_date,
        artist_urls=payload.artist_urls,
        track_url=payload.track_url,
        album_url=payload.album_url,
        track_changed=track_changed,
        current_track=payload.current_track_raw if payload.is_playing else None,
        genius_url=payload.genius_url,
        uploaded_photo=payload.uploaded_photo,
        cover_media_unchanged=payload.cover_media_unchanged,
        prepared_message_content=payload.prepared_message_content,
        prepared_entities=payload.prepared_entities,
        commit_operations=commit_operations,
    )
    commit_ok = bool(
        commit_ok and await update_channel_info(
            bot_client,
            payload.track_title,
            payload.is_playing,
            uploaded_photo=payload.uploaded_photo,
            skip_photo_update=payload.cover_media_unchanged,
            commit_operations=commit_operations,
        )
    )

    if commit_ok:
        commit_results = await asyncio.gather(*commit_operations, return_exceptions=True)
    else:
        for op in commit_operations:
            close_op = getattr(op, "close", None)
            if callable(close_op):
                close_op()
        commit_results = []

    for res in commit_results:
        if isinstance(res, BaseException):
            commit_ok = False
        elif res is not True:
            commit_ok = False

    if not commit_ok:
        force_default_requested = True
        last_successful_cover_url = None
    elif payload.is_playing and payload.cover_url and (payload.cover_media_unchanged or payload.uploaded_photo is not None):
        last_successful_cover_url = payload.cover_url
    elif not payload.is_playing:
        last_successful_cover_url = None
    return commit_ok and (not force_default_requested)

def _is_spotify_track_playing_payload(payload: Optional[dict]) -> tuple[bool, Optional[dict]]:
    if not isinstance(payload, dict) or not payload or not payload.get("is_playing"):
        return False, None
    item = payload.get("item")
    if not isinstance(item, dict) or item.get("type") != "track":
        return False, None
    return True, item

def read_track_info():
    global track_info_cache
    if track_info_cache is not None:
        return track_info_cache
    try:
        cached_track_id = _get_cache_handler().get_cached_track_id()
    except (OSError, ValueError, TypeError, RuntimeError):
        cached_track_id = None
    track_info_cache = {"track_id": cached_track_id}
    return track_info_cache

def write_track_info(track_id):
    global track_info_cache
    if isinstance(track_info_cache, dict):
        current = _normalize_track_info(track_info_cache)
    else:
        current = _normalize_track_info(None)

    if current.get("track_id") == track_id:
        return False

    track_info_cache = {"track_id": track_id}
    try:
        _get_cache_handler().save_track_id(track_id)
    except (OSError, ValueError, TypeError, RuntimeError) as e:
        _safe_log_warning(ERROR_UPDATE_LOOP, e)

    normalize_files = _get_normalize_files_fn()
    try:
        if callable(normalize_files):
            normalize_files()
    except (OSError, ValueError, TypeError) as e:
        _safe_log_warning(ERROR_UPDATE_LOOP, e)
    return True

async def get_current_track(spotify):
    global was_playing, previous_track_id
    current_track = await spotify.current_playback()
    if current_track is None:
        return None

    is_track_playing, item = _is_spotify_track_playing_payload(current_track)

    if is_track_playing:
        track_id = item.get("id")
        track_name = item.get("name")
        artists = item.get("artists") or []
        artist_names_list = []
        artist_urls = []
        for artist in artists:
            if not isinstance(artist, dict):
                continue
            name = artist.get("name")
            url = (artist.get("external_urls") or {}).get("spotify")
            if not name:
                continue
            artist_names_list.append(str(name))
            artist_urls.append(str(url) if url else None)
        track_url = (item.get("external_urls") or {}).get("spotify")
        album = item.get("album") or {}
        album_url = (album.get("external_urls") or {}).get("spotify")
        if track_id and track_name and artist_names_list:
            artist_names = ", ".join(artist_names_list)
            album_name = album.get("name") or ""
            release_date = album.get("release_date") or ""
            release_date_precision = album.get("release_date_precision") or ""
            formatted_release_date = release_date
            if release_date:
                base = _release_date_base(release_date, release_date_precision) or release_date
                try:
                    parsed_date = datetime.strptime(base, "%Y-%m-%d")
                    formatted_release_date = f"{parsed_date.day:02d} {_MONTH_NAMES[parsed_date.month - 1]} {parsed_date.year}"
                except (ValueError, TypeError, IndexError):
                    formatted_release_date = release_date
            cover_url = None
            cover_url = _select_spotify_cover_url(album.get("images"))

            if track_id != previous_track_id or not was_playing:
                _safe_log_info(INFO_TRACK_PLAYING, artist_names, track_name)

            previous_track_id = track_id
            was_playing = True

            return (
                f"{artist_names} – {track_name}",
                cover_url,
                track_id,
                artist_names,
                track_name,
                album_name,
                formatted_release_date,
                artist_urls,
                track_url,
                album_url,
                current_track
            )

    if was_playing:
        _safe_log_info(INFO_NO_TRACK_PLAYING)
        was_playing = False

    previous_track_id = None

    return DEFAULT_MESSAGE, None, None, None, None, None, None, None, None, None, None

def append_entity(entities, entity_type, offset, length, **kwargs):
    entity = entity_type(offset=offset, length=length, **kwargs)
    entities.append(entity)

def _split_artist_values(artist_names: Optional[str], artist_urls: Optional[list[str]]) -> tuple[list[str], list[str]]:
    urls_list = list(artist_urls or [])
    if not isinstance(artist_names, str) or not artist_names:
        return [], urls_list
    if len(urls_list) <= 1:
        return [artist_names], urls_list
    names_list = [n for n in artist_names.split(", ") if n]
    if len(names_list) != len(urls_list):
        return [artist_names], urls_list[:1]
    return names_list, urls_list

def _build_playing_message_payload(
    artist_names,
    track_name,
    album_name,
    release_date,
    artist_urls,
    track_url,
    album_url,
    current_track=None,
    genius_url=None,
):
    track_name = track_name or ""
    album_name = album_name or ""
    release_date = release_date or ""
    album_label = ALBUM_LABEL_ALBUM
    item = current_track.get("item") if isinstance(current_track, dict) else None
    if not (current_track and current_track.get("is_playing") and item and item.get("type") == "track"):
        _safe_log_info(INFO_FALLBACK_TRACK_INFO)
        artist_names_list, artist_urls_local = _split_artist_values(artist_names, artist_urls)
    else:
        artist_names_list = []
        artist_urls_local = []
        for a in (item.get("artists") or []):
            if not isinstance(a, dict):
                continue
            name = a.get("name")
            url = (a.get("external_urls") or {}).get("spotify")
            if not name:
                continue
            artist_names_list.append(str(name))
            artist_urls_local.append(str(url) if url else None)
        total_tracks = _album_total_tracks_from_item(item)
        if isinstance(total_tracks, int) and total_tracks > 0:
            album_label = ALBUM_LABEL_SINGLE if total_tracks <= 4 else ALBUM_LABEL_ALBUM

    if not artist_names_list:
        artist_names_list, artist_urls_local = _split_artist_values(artist_names, artist_urls)

    message_content = ", ".join(artist_names_list) + " – " + track_name + "\n\n" + \
                    f"{album_label}{album_name}\n" + \
                    f"{RELEASE_DATE_PREFIX}{release_date}"

    if genius_url:
        message_content += f"\n\n{LYRICS_ON_GENIUS}"
    surrogate_content = add_surrogate(message_content)
    entities = []
    current_offset = 0
    for i, name in enumerate(artist_names_list):
        name_len = _utf16_len(name)
        append_entity(entities, MessageEntityBold, current_offset, name_len)
        if i < len(artist_urls_local) and artist_urls_local[i]:
            append_entity(entities, MessageEntityTextUrl, current_offset, name_len, url=artist_urls_local[i])
        current_offset += name_len + (2 if i < len(artist_names_list) - 1 else 0)
    track_start = surrogate_content.index(" – ") + len(" – ")
    album_start = surrogate_content.index(album_label) + len(album_label)
    release_date_start = surrogate_content.index(RELEASE_DATE_PREFIX) + len(RELEASE_DATE_PREFIX)
    if track_name:
        track_len = _utf16_len(track_name)
        append_entity(entities, MessageEntityBold, track_start, track_len)
        if track_url:
            append_entity(entities, MessageEntityTextUrl, track_start, track_len, url=track_url)
    if album_name and album_url:
        append_entity(entities, MessageEntityTextUrl, album_start, _utf16_len(album_name), url=album_url)
    if release_date:
        append_entity(entities, types.MessageEntityItalic, release_date_start, _utf16_len(release_date))
    if genius_url:
        lyrics_start = surrogate_content.rindex(LYRICS_ON_GENIUS)
        lyrics_len = _utf16_len(LYRICS_ON_GENIUS)
        append_entity(entities, MessageEntityTextUrl, lyrics_start, lyrics_len, url=genius_url)
        append_entity(entities, MessageEntityUnderline, lyrics_start, lyrics_len)
    return message_content, entities

async def reset_to_default_state(bot_client, reason: str = ""):
    global last_text_message_content, default_media_applied, force_default_requested, updates_suspended, suspended_since, last_successful_cover_url
    reset_succeeded = True
    try:
        st = get_settings()
        force_default_requested = True
        updates_suspended = True
        suspended_since = datetime.now()
        last_successful_cover_url = None
        if reason:
            _safe_log_error(ERROR_FALLBACK_RESET, reason)

        try:
            is_conn = getattr(bot_client, "is_connected", None)
            if callable(is_conn) and not is_conn():
                return False
        except (AttributeError, TypeError, ValueError, RuntimeError, OSError):
            pass

        channel = PeerChannel(st.tg_channel_id)
        profile_picture = os.path.join(DIRS["SESSION"], DEFAULT_LOGO_PATH)
        if os.path.exists(profile_picture):
            try:
                file = await bot_client.upload_file(profile_picture)
                input_media = types.InputMediaUploadedPhoto(file)
                await safe_call_telethon(
                    bot_client,
                    functions.messages.EditMessageRequest(
                        peer=channel,
                        id=st.tg_media_message_id,
                        message=DEFAULT_MESSAGE,
                        media=input_media
                    )
                )
                _safe_log_info(INFO_MEDIA_DEFAULT_UPDATED)
            except (RPCError, ConnectionError, asyncio.TimeoutError, OSError) as e:
                _safe_log_error(ERROR_MEDIA_UPDATE, e)
                reset_succeeded = False
        else:
            try:
                await safe_call_telethon(
                    bot_client,
                    functions.messages.EditMessageRequest(
                        peer=channel,
                        id=st.tg_media_message_id,
                        message=DEFAULT_MESSAGE
                    )
                )
                _safe_log_info(INFO_MEDIA_DEFAULT_UPDATED)
            except (RPCError, ConnectionError, asyncio.TimeoutError, OSError) as e:
                _safe_log_error(ERROR_MEDIA_UPDATE, e)
                reset_succeeded = False
        try:
            await safe_call_telethon(
                bot_client,
                functions.messages.EditMessageRequest(
                    peer=channel,
                    id=st.tg_text_message_id,
                    message=DEFAULT_MESSAGE
                )
            )
            last_text_message_content = DEFAULT_MESSAGE
        except (RPCError, ConnectionError, asyncio.TimeoutError, OSError) as e:
            _safe_log_error(ERROR_TEXT_UPDATE, e)
            reset_succeeded = False
        try:
            channel_info_ok = await update_channel_info(bot_client, DEFAULT_MESSAGE, False)
            if not channel_info_ok:
                reset_succeeded = False
        except (RPCError, ConnectionError, asyncio.TimeoutError, OSError) as e:
            _safe_log_error(ERROR_CHANNEL_INFO_UPDATE, e)
            reset_succeeded = False
        if not reset_succeeded:
            return False
        try:
            write_track_info(None)
        except (OSError, ValueError, TypeError):
            return False

        default_media_applied = True
        return True
    except (RPCError, ConnectionError, asyncio.TimeoutError, OSError, ValueError, TypeError, RuntimeError) as e:
        _safe_log_error(ERROR_UPDATE_LOOP, e)
        return False

async def update_messages(bot_client, cover_url, is_playing, artist_names, track_name, album_name, release_date, artist_urls, track_url, album_url, track_changed=False, current_track=None, genius_url=None, uploaded_photo=None, cover_media_unchanged=False, prepared_message_content=None, prepared_entities=None, commit_operations=None):
    global default_media_applied, force_default_requested, last_successful_cover_url
    update_ok = True
    try:
        st = get_settings()
        channel = PeerChannel(st.tg_channel_id)
        if is_playing:
            if isinstance(prepared_message_content, str) and isinstance(prepared_entities, list):
                message_content = prepared_message_content
                entities = list(prepared_entities)
            else:
                message_content, entities = _build_playing_message_payload(
                    artist_names,
                    track_name,
                    album_name,
                    release_date,
                    artist_urls,
                    track_url,
                    album_url,
                    current_track=current_track,
                    genius_url=genius_url,
                )

            async def update_text_task():
                global last_text_message_content
                global force_default_requested
                nonlocal update_ok

                now_playing_text = NOW_PLAYING_ON_SPOTIFY
                if last_text_message_content == now_playing_text:
                    return True
                try:
                    text_entities = [
                        MessageEntityCode(offset=0, length=len(now_playing_text)),
                        MessageEntityBlockquote(offset=0, length=len(now_playing_text)),
                    ]
                    await safe_call_telethon(
                        bot_client,
                        functions.messages.EditMessageRequest(
                            peer=channel,
                            id=st.tg_text_message_id,
                            message=now_playing_text,
                            entities=text_entities
                        )
                    )
                    last_text_message_content = now_playing_text
                    return True
                except (RPCError, ConnectionError, asyncio.TimeoutError, OSError) as e:
                    _safe_log_error(ERROR_TEXT_UPDATE, e)
                    force_default_requested = True
                    update_ok = False
                    return False

            async def apply_media_message_task(uploaded_photo_param=None):
                global force_default_requested
                nonlocal update_ok
                try:
                    if cover_media_unchanged:
                        await safe_call_telethon(
                            bot_client,
                            functions.messages.EditMessageRequest(
                                peer=channel,
                                id=st.tg_media_message_id,
                                message=message_content,
                                entities=entities
                            )
                        )
                    elif cover_url:
                        if uploaded_photo_param is not None:
                            if isinstance(uploaded_photo_param, Photo):
                                media_obj = types.InputMediaPhoto(id=get_input_photo(uploaded_photo_param))
                            else:
                                media_obj = types.InputMediaUploadedPhoto(file=uploaded_photo_param)
                            await safe_call_telethon(
                                bot_client,
                                functions.messages.EditMessageRequest(
                                    peer=channel,
                                    id=st.tg_media_message_id,
                                    message=message_content,
                                    media=media_obj,
                                    entities=entities
                                )
                            )
                        else:
                            await safe_call_telethon(
                                bot_client,
                                functions.messages.EditMessageRequest(
                                    peer=channel,
                                    id=st.tg_media_message_id,
                                    message=message_content,
                                    media=types.InputMediaPhotoExternal(url=cover_url),
                                    entities=entities
                                )
                            )
                    else:
                        await safe_call_telethon(
                            bot_client,
                            functions.messages.EditMessageRequest(
                                peer=channel,
                                id=st.tg_media_message_id,
                                message=message_content,
                                entities=entities
                            )
                        )
                    _safe_log_info(INFO_MEDIA_UPDATED)
                    return True
                except (RPCError, ConnectionError, asyncio.TimeoutError, OSError) as e:
                    _safe_log_error(ERROR_MEDIA_UPDATE, e)
                    force_default_requested = True
                    update_ok = False
                    return False

            default_media_applied = False
            commit_ops = [update_text_task(), apply_media_message_task(uploaded_photo)]
            if commit_operations is not None:
                commit_operations.extend(commit_ops)
                return True
            await asyncio.gather(*commit_ops)
            return update_ok and (not force_default_requested)
        else:
            last_successful_cover_url = None
            profile_picture = os.path.join(DIRS["SESSION"], DEFAULT_LOGO_PATH)
            async def apply_default_media_task():
                global default_media_applied
                nonlocal update_ok
                if os.path.exists(profile_picture):
                    if not default_media_applied:
                        try:
                            file = await bot_client.upload_file(profile_picture)
                            input_media = types.InputMediaUploadedPhoto(file)
                            await safe_call_telethon(
                                bot_client,
                                functions.messages.EditMessageRequest(
                                    peer=channel,
                                    id=st.tg_media_message_id,
                                    message=DEFAULT_MESSAGE,
                                    media=input_media
                                )
                            )
                            _safe_log_info(INFO_MEDIA_DEFAULT_UPDATED)
                            default_media_applied = True
                        except (RPCError, ConnectionError, asyncio.TimeoutError, OSError) as e:
                            _safe_log_error(ERROR_MEDIA_UPDATE, e)
                            update_ok = False
                            return False
                return True

            async def update_default_text_task():
                global last_text_message_content
                nonlocal update_ok
                if last_text_message_content != DEFAULT_MESSAGE:
                    try:
                        await safe_call_telethon(
                            bot_client,
                            functions.messages.EditMessageRequest(
                                peer=channel,
                                id=st.tg_text_message_id,
                                message=DEFAULT_MESSAGE
                            )
                        )
                        last_text_message_content = DEFAULT_MESSAGE
                    except (RPCError, ConnectionError, asyncio.TimeoutError, OSError) as e:
                        _safe_log_error(ERROR_TEXT_UPDATE, e)
                        update_ok = False
                        return False
                return True

            commit_ops = [apply_default_media_task(), update_default_text_task()]
            if commit_operations is not None:
                commit_operations.extend(commit_ops)
                return True
            await asyncio.gather(*commit_ops)
            return update_ok
    except (RPCError, ConnectionError, asyncio.TimeoutError, OSError, ValueError, TypeError) as e:
        _safe_log_error(ERROR_UPDATING_MESSAGES, e)
        force_default_requested = True
        return False

async def update_channel_info(bot_client, track_title, is_playing, uploaded_photo=None, skip_photo_update=False, commit_operations=None):
    global force_default_requested, last_successful_cover_url
    update_ok = True
    try:
        st = get_settings()
        channel = PeerChannel(st.tg_channel_id)
        if not is_playing:
            last_successful_cover_url = None
            channel_name = DEFAULT_MESSAGE
            profile_picture = os.path.join(DIRS["SESSION"], DEFAULT_LOGO_PATH)

            async def update_title_task():
                nonlocal update_ok
                try:
                    await safe_call_telethon(bot_client, EditTitleRequest(channel, _truncate_utf16(channel_name, _MAX_TITLE_LENGTH)))
                    return True
                except (RPCError, ConnectionError, asyncio.TimeoutError, OSError) as e:
                    _safe_log_error(ERROR_CHANNEL_TITLE_UPDATE, e)
                    update_ok = False
                    return False

            async def update_photo_task():
                nonlocal update_ok
                if os.path.exists(profile_picture):
                    try:
                        file = await bot_client.upload_file(profile_picture)
                        input_photo = await safe_call_telethon(
                            bot_client,
                            functions.messages.UploadMediaRequest(
                                peer=channel,
                                media=types.InputMediaUploadedPhoto(file)
                            )
                        )
                        if input_photo is None or not getattr(input_photo, "photo", None):
                            _safe_log_error(ERROR_CHANNEL_PHOTO_UPDATE, "empty upload response")
                            update_ok = False
                            return False
                        await safe_call_telethon(bot_client, EditPhotoRequest(channel, input_photo.photo))
                        return True
                    except (RPCError, ConnectionError, asyncio.TimeoutError, OSError) as e:
                        _safe_log_error(ERROR_CHANNEL_PHOTO_UPDATE, e)
                        update_ok = False
                        return False
                return True

            commit_ops = [update_title_task(), update_photo_task()]
            if commit_operations is not None:
                commit_operations.extend(commit_ops)
                return True
            await asyncio.gather(*commit_ops, return_exceptions=True)
            if update_ok:
                _safe_log_info(INFO_CHANNEL_DEFAULT_UPDATED)
            return update_ok
        else:
            async def update_title_task():
                global force_default_requested
                nonlocal update_ok
                try:
                    await safe_call_telethon(bot_client, EditTitleRequest(channel, _truncate_utf16(track_title, _MAX_TITLE_LENGTH)))
                    return True
                except (RPCError, ConnectionError, asyncio.TimeoutError, OSError) as e:
                    _safe_log_error(ERROR_CHANNEL_TITLE_UPDATE, e)
                    force_default_requested = True
                    update_ok = False
                    return False

            async def update_photo_task():
                global force_default_requested
                nonlocal update_ok
                if skip_photo_update:
                    return True
                if uploaded_photo is not None:
                    try:
                        if isinstance(uploaded_photo, Photo):
                            await safe_call_telethon(bot_client, EditPhotoRequest(channel, uploaded_photo))
                        else:
                            await safe_call_telethon(bot_client, EditPhotoRequest(channel, InputChatUploadedPhoto(uploaded_photo)))
                        return True
                    except (RPCError, ConnectionError, asyncio.TimeoutError, OSError) as e:
                        _safe_log_error(ERROR_CHANNEL_PHOTO_UPDATE, e)
                        force_default_requested = True
                        update_ok = False
                        return False
                return True

            commit_ops = [update_title_task(), update_photo_task()]
            if commit_operations is not None:
                commit_operations.extend(commit_ops)
                return True
            await asyncio.gather(*commit_ops, return_exceptions=True)
            if update_ok and not force_default_requested:
                _safe_log_info(INFO_CHANNEL_UPDATED)
            return update_ok and (not force_default_requested)
    except (RPCError, ConnectionError, asyncio.TimeoutError, OSError, ValueError, TypeError, RuntimeError) as e:
        _safe_log_error(ERROR_CHANNEL_INFO_UPDATE, e)
        if is_playing:
            force_default_requested = True
        return False

async def update_channel(bot_client, spotify_client):
    track_info = read_track_info()
    cached_track_id = track_info.get("track_id") if isinstance(track_info, dict) else None
    try:
        global force_default_requested, updates_suspended, suspended_on_track_id, suspended_since
        st = get_settings()
        force_default_requested = False
        result = await get_current_track(spotify_client)
        if result is None:
            return cached_track_id
        track_title, cover_url, track_id, artist_names, track_name, album_name, release_date, artist_urls, track_url, album_url, current_track_raw = result
        is_playing = track_title != DEFAULT_MESSAGE and track_id is not None
        if updates_suspended:
            if not is_playing:
                updates_suspended = False
                suspended_on_track_id = None
                suspended_since = None
            else:
                if track_id and (suspended_on_track_id is None or track_id != suspended_on_track_id):
                    updates_suspended = False
                    suspended_on_track_id = None
                    suspended_since = None
                else:
                    retry_seconds = 30
                    if suspended_since is not None:
                        try:
                            if (datetime.now() - suspended_since).total_seconds() >= retry_seconds:
                                updates_suspended = False
                                suspended_on_track_id = None
                                suspended_since = None
                            else:
                                return cached_track_id
                        except (TypeError, ValueError):
                            return cached_track_id
                    else:
                        return cached_track_id

        track_changed = track_info["track_id"] != track_id
        if not is_playing and not track_changed:
            return None
        if is_playing and not track_changed:
            return track_id
        commit_payload = await _collect_channel_commit_payload(
            bot_client,
            st=st,
            track_title=track_title,
            cover_url=cover_url,
            track_id=track_id,
            is_playing=is_playing,
            artist_names=artist_names,
            track_name=track_name,
            album_name=album_name,
            release_date=release_date,
            artist_urls=artist_urls,
            track_url=track_url,
            album_url=album_url,
            current_track_raw=current_track_raw,
            track_changed=track_changed,
        )

        commit_ok = await _commit_channel_commit_payload(
            bot_client,
            commit_payload,
            track_changed=track_changed,
        )
        if track_changed and not commit_ok:
            if force_default_requested:
                suspended_on_track_id = track_id if is_playing else None
                try:
                    reset_ok = await reset_to_default_state(bot_client, REASON_TELEGRAM_UPDATE_FAILED)
                except (RPCError, ConnectionError, asyncio.TimeoutError, OSError, ValueError, TypeError, RuntimeError) as e:
                    _safe_log_error(ERROR_TEXT_UPDATE, e)
                    reset_ok = False
                return None if reset_ok else cached_track_id

        if force_default_requested:
            return None
        return track_id if is_playing else None
    except Exception as e:
        now = time.monotonic()
        try:
            err_text = str(e)
        except (ValueError, TypeError, UnicodeError):
            err_text = ""
        sig = f"{type(e).__name__}:{err_text}"
        if now - _UPDATE_ERROR_STATE["ts"] >= 60.0 or sig != _UPDATE_ERROR_STATE["sig"]:
            _UPDATE_ERROR_STATE["ts"] = now
            _UPDATE_ERROR_STATE["sig"] = sig
            _safe_log_error(ERROR_UPDATE_LOOP, e)
        return cached_track_id
