import os
import asyncio
import aiohttp
from datetime import datetime
from typing import Optional

from telethon import functions, types
from telethon.errors import RPCError
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
from logger import _safe_log_info, _safe_log_error
from utils.telethon_utils import safe_call_telethon
from utils.http_session import init_session
from lyrics.genius_search import search_lyrics_url
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

for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

_CACHE_HANDLER = None

def _get_cache_handler():
    global _CACHE_HANDLER
    if _CACHE_HANDLER is None:
        cache_path = os.path.join(DIRS["SESSION"], ".cache")
        _CACHE_HANDLER = get_spg_cache_handler(cache_path)
    return _CACHE_HANDLER

async def _download_bytes(session, url: str) -> bytes:
    async with session.get(url) as resp:
        if not (200 <= resp.status < 300):
            raise RuntimeError(f"HTTP {resp.status}")
        data = await resp.read()
        if not data:
            raise RuntimeError("empty body")
        return data

async def _preupload_cover_photo(bot_client, cover_url: str, channel: PeerChannel):
    try:
        input_photo = await asyncio.wait_for(
            safe_call_telethon(
                bot_client,
                functions.messages.UploadMediaRequest(
                    peer=channel,
                    media=types.InputMediaPhotoExternal(url=cover_url),
                ),
            ),
            timeout=10.0,
        )
        if input_photo and getattr(input_photo, "photo", None):
            _safe_log_info(INFO_COVER_DOWNLOADED_PRIMARY)
            return input_photo.photo
    except (RPCError, ConnectionError, asyncio.TimeoutError, OSError) as e:
        _safe_log_error(ERROR_COVER_UPLOAD, e)

    try:
        session = init_session()
        data = await asyncio.wait_for(_download_bytes(session, cover_url), timeout=10.0)
        up_file = await bot_client.upload_file(data, file_name="cover.jpg")
        _safe_log_info(INFO_COVER_DOWNLOADED_FALLBACK1)
        return up_file
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError, RPCError, ConnectionError) as e:
        _safe_log_error(ERROR_COVER_UPLOAD, e)

    cover_path = os.path.join(DIRS["SESSION"], "cover.jpg")
    try:
        session = init_session()
        data = await asyncio.wait_for(_download_bytes(session, cover_url), timeout=10.0)
        with open(cover_path, "wb") as f:
            f.write(data)
        up_file = await bot_client.upload_file(cover_path)
        _safe_log_info(INFO_COVER_DOWNLOADED_FALLBACK2)
        return up_file
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError, RPCError, ConnectionError) as e:
        _safe_log_error(ERROR_COVER_UPLOAD, e)
        return None
    finally:
        try:
            if os.path.exists(cover_path):
                os.remove(cover_path)
        except OSError:
            pass

def _normalize_track_info(data: Optional[dict]) -> dict:
    if not isinstance(data, dict):
        return {"track_id": None}
    return {
        "track_id": data.get("track_id"),
    }

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
    spotify_artists = item.get("artists") or None
    album_total_tracks = _album_total_tracks_from_item(item)
    album_obj = item.get("album") if isinstance(item.get("album"), dict) else {}
    spotify_release_date_iso = _release_date_base(
        album_obj.get("release_date"),
        album_obj.get("release_date_precision"),
    )
    return spotify_artists, album_total_tracks, spotify_release_date_iso

def read_track_info():
    global track_info_cache
    if track_info_cache is not None:
        return track_info_cache
    try:
        cached_track_id = _get_cache_handler().get_cached_track_id()
    except Exception:
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
    except Exception:
        pass

    try:
        from lyrics.patterns.base import normalize_files
    except ImportError:
        normalize_files = None
    try:
        if callable(normalize_files):
            normalize_files()
    except (OSError, ValueError, TypeError):
        pass
    return True

async def get_current_track(spotify):
    global was_playing, previous_track_id
    current_track = await spotify.current_playback()
    if current_track is None:
        return None

    item = current_track.get("item") if isinstance(current_track, dict) else None
    is_track_playing = (
        isinstance(current_track, dict)
        and bool(current_track)
        and bool(current_track.get("is_playing"))
        and isinstance(item, dict)
        and item.get("type") == "track"
    )
    if (not is_track_playing) and was_playing:
        retry_track = await spotify.current_playback()
        if retry_track is None:
            return None
        retry_item = retry_track.get("item") if isinstance(retry_track, dict) else None
        retry_is_playing = (
            isinstance(retry_track, dict)
            and bool(retry_track)
            and bool(retry_track.get("is_playing"))
            and isinstance(retry_item, dict)
            and retry_item.get("type") == "track"
        )
        if retry_is_playing:
            current_track = retry_track
            item = retry_item
            is_track_playing = True

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
            if not name or not url:
                continue
            artist_names_list.append(str(name))
            artist_urls.append(str(url))
        track_url = (item.get("external_urls") or {}).get("spotify")
        album = item.get("album") or {}
        album_url = (album.get("external_urls") or {}).get("spotify")
        if track_id and track_name and artist_names_list and track_url and album_url:
            artist_names = ", ".join(artist_names_list)
            album_name = album.get("name") or ""
            release_date = album.get("release_date") or ""
            release_date_precision = album.get("release_date_precision") or ""
            formatted_release_date = release_date
            if release_date:
                base = _release_date_base(release_date, release_date_precision) or release_date
                try:
                    formatted_release_date = datetime.strptime(base, "%Y-%m-%d").strftime("%d %B %Y")
                except (ValueError, TypeError):
                    formatted_release_date = release_date
            cover_url = None
            images = album.get("images") or []
            if isinstance(images, list) and images:
                first = images[0]
                if isinstance(first, dict):
                    cover_url = first.get("url")

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

async def reset_to_default_state(bot_client, reason: str = ""):
    global last_text_message_content, default_media_applied, force_default_requested, updates_suspended, suspended_since
    try:
        st = get_settings()
        force_default_requested = True
        updates_suspended = True
        suspended_since = datetime.now()
        if reason:
            _safe_log_error(ERROR_FALLBACK_RESET, reason)
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
        try:
            await update_channel_info(bot_client, DEFAULT_MESSAGE, False)
        except (RPCError, ConnectionError, asyncio.TimeoutError, OSError) as e:
            _safe_log_error(ERROR_CHANNEL_INFO_UPDATE, e)
        try:
            write_track_info(None)
        except (OSError, ValueError, TypeError):
            pass

        default_media_applied = True
    except (RPCError, ConnectionError, asyncio.TimeoutError, OSError, ValueError, TypeError, RuntimeError):
        pass

async def update_messages(bot_client, cover_url, is_playing, artist_names, track_name, album_name, release_date, artist_urls, track_url, album_url, track_changed=False, current_track=None, genius_url=None, uploaded_photo=None):
    global default_media_applied, last_text_message_content, force_default_requested
    try:
        st = get_settings()
        channel = PeerChannel(st.tg_channel_id)
        if is_playing:
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
            entities = []
            current_offset = 0
            for i, name in enumerate(artist_names_list):
                append_entity(entities, MessageEntityBold, current_offset, len(name))
                if i < len(artist_urls_local) and artist_urls_local[i]:
                    append_entity(entities, MessageEntityTextUrl, current_offset, len(name), url=artist_urls_local[i])
                current_offset += len(name) + (2 if i < len(artist_names_list) - 1 else 0)
            track_start = message_content.index(" – ") + len(" – ")
            album_start = message_content.index(album_label) + len(album_label)
            release_date_start = message_content.index(RELEASE_DATE_PREFIX) + len(RELEASE_DATE_PREFIX)
            if track_name:
                append_entity(entities, MessageEntityBold, track_start, len(track_name))
                if track_url:
                    append_entity(entities, MessageEntityTextUrl, track_start, len(track_name), url=track_url)
            if album_name and album_url:
                append_entity(entities, MessageEntityTextUrl, album_start, len(album_name), url=album_url)
            if release_date:
                append_entity(entities, types.MessageEntityItalic, release_date_start, len(release_date))
            if genius_url:
                lyrics_start = message_content.rindex(LYRICS_ON_GENIUS)
                append_entity(entities, MessageEntityTextUrl, lyrics_start, len(LYRICS_ON_GENIUS), url=genius_url)
                append_entity(entities, MessageEntityUnderline, lyrics_start, len(LYRICS_ON_GENIUS))

            async def update_text_task():
                global last_text_message_content
                global force_default_requested

                now_playing_text = NOW_PLAYING_ON_SPOTIFY
                if last_text_message_content == now_playing_text and not track_changed:
                    return
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
                except (RPCError, ConnectionError, asyncio.TimeoutError, OSError) as e:
                    _safe_log_error(ERROR_TEXT_UPDATE, e)
                    force_default_requested = True

            async def apply_media_message_task(uploaded_photo_param=None):
                global force_default_requested
                try:
                    if cover_url:
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
                except (RPCError, ConnectionError, asyncio.TimeoutError, OSError) as e:
                    _safe_log_error(ERROR_MEDIA_UPDATE, e)
                    force_default_requested = True

            default_media_applied = False
            await asyncio.gather(update_text_task(), apply_media_message_task(uploaded_photo))
            return uploaded_photo
        else:
            profile_picture = os.path.join(DIRS["SESSION"], DEFAULT_LOGO_PATH)
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
            global last_text_message_content
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
            return None
    except (RPCError, ConnectionError, asyncio.TimeoutError, OSError, ValueError, TypeError) as e:
        _safe_log_error(ERROR_UPDATING_MESSAGES, e)
        force_default_requested = True

async def update_channel_info(bot_client, track_title, is_playing, uploaded_photo=None):
    try:
        st = get_settings()
        channel = PeerChannel(st.tg_channel_id)
        if not is_playing:
            channel_name = DEFAULT_MESSAGE
            profile_picture = os.path.join(DIRS["SESSION"], DEFAULT_LOGO_PATH)

            async def update_title_task():
                try:
                    await safe_call_telethon(bot_client, EditTitleRequest(channel, channel_name))
                except (RPCError, ConnectionError, asyncio.TimeoutError, OSError) as e:
                    _safe_log_error(ERROR_CHANNEL_TITLE_UPDATE, e)

            async def update_photo_task():
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
                        if input_photo and getattr(input_photo, "photo", None):
                            await safe_call_telethon(bot_client, EditPhotoRequest(channel, input_photo.photo))
                    except (RPCError, ConnectionError, asyncio.TimeoutError, OSError) as e:
                        _safe_log_error(ERROR_CHANNEL_PHOTO_UPDATE, e)

            tasks = [update_title_task(), update_photo_task()]
            await asyncio.gather(*tasks, return_exceptions=True)
            _safe_log_info(INFO_CHANNEL_DEFAULT_UPDATED)
        else:
            async def update_title_task():
                global force_default_requested
                try:
                    await safe_call_telethon(bot_client, EditTitleRequest(channel, track_title))
                except (RPCError, ConnectionError, asyncio.TimeoutError, OSError) as e:
                    _safe_log_error(ERROR_CHANNEL_TITLE_UPDATE, e)
                    force_default_requested = True

            async def update_photo_task():
                global force_default_requested
                if uploaded_photo is not None:
                    try:
                        if isinstance(uploaded_photo, Photo):
                            await safe_call_telethon(bot_client, EditPhotoRequest(channel, uploaded_photo))
                        else:
                            await safe_call_telethon(bot_client, EditPhotoRequest(channel, InputChatUploadedPhoto(uploaded_photo)))
                    except (RPCError, ConnectionError, asyncio.TimeoutError, OSError) as e:
                        _safe_log_error(ERROR_CHANNEL_PHOTO_UPDATE, e)
                        force_default_requested = True

            tasks = [update_title_task(), update_photo_task()]
            await asyncio.gather(*tasks, return_exceptions=True)
            if not force_default_requested:
                _safe_log_info(INFO_CHANNEL_UPDATED)
    except (RPCError, ConnectionError, asyncio.TimeoutError, OSError) as e:
        _safe_log_error(ERROR_CHANNEL_INFO_UPDATE, e)

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
        track_info = read_track_info()
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
        genius_url = None
        preuploaded_photo = None
        phase1_tasks = []
        if st.enable_genius and st.genius_access_token and is_playing and track_changed and track_id:
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
            phase1_tasks.append(asyncio.create_task(_lyrics_task()))
        if is_playing and track_changed and cover_url:
            channel = PeerChannel(st.tg_channel_id)
            phase1_tasks.append(asyncio.create_task(_preupload_cover_photo(bot_client, cover_url, channel)))

        if phase1_tasks:
            try:
                phase1_results = await asyncio.gather(*phase1_tasks, return_exceptions=True)
            except asyncio.CancelledError:
                for t in phase1_tasks:
                    try:
                        if isinstance(t, asyncio.Task) and not t.done():
                            t.cancel()
                    except Exception:
                        pass
                await asyncio.gather(*phase1_tasks, return_exceptions=True)
                raise
            for res in phase1_results:
                if res is None or isinstance(res, BaseException):
                    continue
                if isinstance(res, (Photo, InputFile, InputFileBig)):
                    preuploaded_photo = res
                elif isinstance(res, str):
                    genius_url = res

        uploaded_photo_for_update = preuploaded_photo

        if track_changed:
            await asyncio.gather(
                update_messages(
                    bot_client=bot_client,
                    cover_url=cover_url,
                    is_playing=is_playing,
                    artist_names=artist_names,
                    track_name=track_name,
                    album_name=album_name,
                    release_date=release_date,
                    artist_urls=artist_urls,
                    track_url=track_url,
                    album_url=album_url,
                    track_changed=track_changed,
                    current_track=current_track_raw if is_playing else None,
                    genius_url=genius_url,
                    uploaded_photo=uploaded_photo_for_update,
                ),
                update_channel_info(
                    bot_client,
                    track_title,
                    is_playing,
                    uploaded_photo=uploaded_photo_for_update,
                ),
                return_exceptions=True,
            )
            if force_default_requested:
                suspended_on_track_id = track_id if is_playing else None
                try:
                    await reset_to_default_state(bot_client, REASON_TELEGRAM_UPDATE_FAILED)
                except (RPCError, ConnectionError, asyncio.TimeoutError, OSError, ValueError, TypeError, RuntimeError) as e:
                    _safe_log_error(ERROR_TEXT_UPDATE, e)
                return None

        if force_default_requested:
            return None
        return track_id if is_playing else None
    except (RPCError, ConnectionError, asyncio.TimeoutError, OSError, aiohttp.ClientError, ValueError, TypeError, RuntimeError):
        return cached_track_id
