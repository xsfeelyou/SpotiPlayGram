from __future__ import annotations

import asyncio
import copy
import json
import re
from typing import Any, Callable, Optional

import aiohttp

from constants import (
    ERROR_GENIUS_ACCESS_TOKEN_INVALID,
    ERROR_GENIUS_HTTP,
    ERROR_GENIUS_TIMEOUT,
    GENIUS_HTTP_MAX_RETRIES,
    GENIUS_HTTP_RETRY_DELAY_BASE,
    GENIUS_SEARCH_PER_PAGE,
    GENIUS_TIMEOUT_SECONDS,
    REASON_EMPTY_JSON_RESPONSE,
)
from logger import _safe_log_error, dev_logger
from utils.http_session import init_session

API_BASE = "https://api.genius.com"
_RE_FLOAT = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_GENIUS_SEARCH_INFLIGHT: dict[str, asyncio.Task] = {}
_GENIUS_SONG_INFLIGHT: dict[int, asyncio.Task] = {}

def _clone_cached_value(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except (TypeError, ValueError, RuntimeError):
        return value

class AsyncGeniusClient:
    def __init__(
        self,
        access_token_provider: Callable[[], Optional[str]],
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self._access_token_provider = access_token_provider
        self._session = session

    def _session_or_default(self) -> aiohttp.ClientSession:
        return self._session or init_session()

    @staticmethod
    def _format_error_detail(value: Any, max_len: int = 160) -> str:
        if isinstance(value, BaseException):
            text = str(value).strip() or value.__class__.__name__
        else:
            text = "" if value is None else str(value)
        text = text.strip()
        if not text:
            return REASON_EMPTY_JSON_RESPONSE
        title_match = re.search(r"<title[^>]*>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
        if title_match:
            text = title_match.group(1)
        else:
            text = re.sub(r"<[^>]+>", " ", text)
        text = " ".join(text.split())
        if not text:
            text = "non-json response"
        if len(text) > max_len:
            text = text[:max_len - 3].rstrip() + "..."
        return text

    @staticmethod
    async def _read_json_payload(resp: aiohttp.ClientResponse) -> Optional[dict[str, Any]]:
        txt = await resp.text(errors="ignore")
        t = txt.lstrip()
        if not t:
            _safe_log_error(ERROR_GENIUS_HTTP, "JSON", REASON_EMPTY_JSON_RESPONSE)
            return None
        try:
            data = json.loads(txt)
        except json.JSONDecodeError:
            _safe_log_error(
                ERROR_GENIUS_HTTP,
                "JSON",
                AsyncGeniusClient._format_error_detail(t),
            )
            return None
        if not isinstance(data, dict):
            _safe_log_error(ERROR_GENIUS_HTTP, "JSON", AsyncGeniusClient._format_error_detail(t))
            return None
        return data

    @staticmethod
    def _request_profile(path: str) -> tuple[int, float, float, float, float]:
        retries = _safe_int(GENIUS_HTTP_MAX_RETRIES, 0)
        if retries < 0:
            retries = 0

        base_delay = _safe_float(GENIUS_HTTP_RETRY_DELAY_BASE, 0.0)
        if base_delay < 0:
            base_delay = 0.0

        timeout_seconds = max(1.0, _safe_float(GENIUS_TIMEOUT_SECONDS, 30.0))
        retry_delay_cap = 30.0
        retry_after_cap = 60.0

        if path.startswith("/search"):
            retries = min(retries, 3)
            timeout_seconds = min(timeout_seconds, 4.0)
            retry_delay_cap = 4.0
            retry_after_cap = 10.0
        return retries, base_delay, timeout_seconds, retry_delay_cap, retry_after_cap

    async def get(self, path: str, params: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
        token = self._access_token_provider()
        if not token:
            return None
        url = f"{API_BASE}{path}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        session = self._session_or_default()
        retries, base_delay, timeout_seconds, retry_delay_cap, retry_after_cap = self._request_profile(path)
        max_attempts = 1 + retries

        def retry_after_seconds(headers_value: Any) -> Optional[float]:
            try:
                raw = headers_value.get("Retry-After")
            except (AttributeError, TypeError, ValueError):
                raw = None
            if raw is None:
                return None
            s = str(raw).strip()
            if not s:
                return None
            if s.isdigit():
                return float(int(s))
            if _RE_FLOAT.fullmatch(s):
                try:
                    return float(s)
                except ValueError:
                    return None
            return None

        def calc_backoff_seconds(attempt_num: int) -> float:
            if base_delay > 0:
                try:
                    return float(base_delay) * float(2 ** max(0, attempt_num - 1))
                except (OverflowError, ValueError, TypeError):
                    return float(base_delay) * float(attempt_num)
            return float(max(1, attempt_num))

        for attempt in range(1, max_attempts + 1):
            try:
                async with session.get(url, headers=headers, params=params, timeout=timeout_seconds) as resp:
                    status = resp.status
                    if 200 <= status < 300:
                        return await self._read_json_payload(resp)

                    if status == 429 and attempt < max_attempts:
                        ra = retry_after_seconds(getattr(resp, "headers", None) or {})
                        delay_s = ra if isinstance(ra, (int, float)) and ra > 0 else calc_backoff_seconds(attempt)
                        delay_s = min(max(float(delay_s), 0.5), retry_after_cap)
                        await asyncio.sleep(delay_s)
                        continue

                    if 500 <= status < 600 and attempt < max_attempts:
                        delay_s = calc_backoff_seconds(attempt)
                        delay_s = min(max(float(delay_s), 0.5), retry_delay_cap)
                        await asyncio.sleep(delay_s)
                        continue

                    if status in (401, 403):
                        _safe_log_error(ERROR_GENIUS_ACCESS_TOKEN_INVALID)
                        return None

                    txt = await resp.text(errors="ignore")
                    _safe_log_error(ERROR_GENIUS_HTTP, status, self._format_error_detail(txt))
                    return None
            except asyncio.TimeoutError:
                if attempt < max_attempts:
                    delay_s = calc_backoff_seconds(attempt)
                    delay_s = min(max(float(delay_s), 0.5), retry_delay_cap)
                    await asyncio.sleep(delay_s)
                    continue
                _safe_log_error(ERROR_GENIUS_TIMEOUT, timeout_seconds)
                return None
            except (aiohttp.ClientError, OSError) as e:
                if attempt < max_attempts:
                    delay_s = calc_backoff_seconds(attempt)
                    delay_s = min(max(float(delay_s), 0.5), retry_delay_cap)
                    await asyncio.sleep(delay_s)
                    continue
                _safe_log_error(ERROR_GENIUS_HTTP, "EXC", self._format_error_detail(e))
                return None
        return None

    async def search(self, query: str) -> list[dict[str, Any]]:
        per_page = _safe_int(GENIUS_SEARCH_PER_PAGE, 20)
        if per_page <= 0:
            per_page = 20
        query_text = str(query or "").strip()
        if not query_text:
            return []
        cache_key = f"{per_page}\x1f{query_text}"

        async def _load_search() -> list[dict[str, Any]]:
            payload = await self.get("/search", params={"q": query_text, "per_page": per_page})
            if not payload:
                return []
            hits: list[dict[str, Any]] = []
            response = payload.get("response")
            if isinstance(response, dict):
                raw_hits = response.get("hits")
                if isinstance(raw_hits, list):
                    hits = [
                        h.get("result")
                        for h in raw_hits
                        if isinstance(h, dict)
                        and h.get("type") == "song"
                        and isinstance(h.get("result"), dict)
                    ]
            return hits

        inflight = _GENIUS_SEARCH_INFLIGHT.get(cache_key)
        if inflight is None:
            inflight = asyncio.create_task(_load_search())
            _GENIUS_SEARCH_INFLIGHT[cache_key] = inflight

            def _clear_search_task(task, key=cache_key):
                if _GENIUS_SEARCH_INFLIGHT.get(key) is task:
                    _GENIUS_SEARCH_INFLIGHT.pop(key, None)
                if not task.cancelled():
                    task.exception()

            inflight.add_done_callback(_clear_search_task)
        try:
            hits = await asyncio.shield(inflight)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _safe_log_error(ERROR_GENIUS_HTTP, "EXC", self._format_error_detail(e), logger=dev_logger)
            return []
        cloned_hits = _clone_cached_value(hits)
        return cloned_hits if isinstance(cloned_hits, list) else []

    async def fetch_song(self, song_id: int) -> Optional[dict[str, Any]]:
        async def _load_song() -> Optional[dict[str, Any]]:
            data = await self.get(f"/songs/{song_id}", params={"text_format": "plain"})
            if not data:
                return None
            song = data.get("response", {}).get("song")
            if isinstance(song, dict):
                return song
            return None

        inflight = _GENIUS_SONG_INFLIGHT.get(song_id)
        if inflight is None:
            inflight = asyncio.create_task(_load_song())
            _GENIUS_SONG_INFLIGHT[song_id] = inflight

            def _clear_song_task(task, key=song_id):
                if _GENIUS_SONG_INFLIGHT.get(key) is task:
                    _GENIUS_SONG_INFLIGHT.pop(key, None)
                if not task.cancelled():
                    task.exception()

            inflight.add_done_callback(_clear_song_task)
        try:
            song = await asyncio.shield(inflight)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _safe_log_error(ERROR_GENIUS_HTTP, "EXC", self._format_error_detail(e), logger=dev_logger)
            return None
        cloned_song = _clone_cached_value(song)
        return cloned_song if isinstance(cloned_song, dict) or cloned_song is None else None

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default
