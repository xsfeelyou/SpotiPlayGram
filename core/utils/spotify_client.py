from __future__ import annotations

import asyncio
import json
import os
import threading
from typing import Any, Optional

import aiohttp
from spotipy import SpotifyOAuth
from spotipy.cache_handler import CacheHandler

from .http_session import init_session

class SPGCacheFileHandler(CacheHandler):
    TRACK_ID_KEY = "spg_track_id"

    def __init__(self, cache_path: str) -> None:
        self.cache_path = str(cache_path)
        self._lock = threading.Lock()

    def _read_cache_unlocked(self) -> dict[str, Any]:
        try:
            with open(self.cache_path, encoding="utf-8") as f:
                raw = f.read()
            if not raw:
                return {}
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            return {}

    def _write_cache_unlocked(self, data: dict[str, Any]) -> None:
        try:
            parent = os.path.dirname(self.cache_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp_path = self.cache_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(data))
            os.replace(tmp_path, self.cache_path)
            try:
                os.chmod(self.cache_path, 0o600)
            except OSError:
                pass
        except OSError:
            return

    def get_cached_token(self):
        with self._lock:
            data = self._read_cache_unlocked()
        if not isinstance(data, dict) or not data:
            return None
        token_info = dict(data)
        token_info.pop(self.TRACK_ID_KEY, None)
        return token_info or None

    def save_token_to_cache(self, token_info):
        if not isinstance(token_info, dict):
            return
        with self._lock:
            existing = self._read_cache_unlocked()
            track_id = existing.get(self.TRACK_ID_KEY)
            out = dict(token_info)
            if self.TRACK_ID_KEY in existing:
                out[self.TRACK_ID_KEY] = track_id
            self._write_cache_unlocked(out)

    def get_cached_track_id(self) -> Optional[str]:
        with self._lock:
            data = self._read_cache_unlocked()
        tid = data.get(self.TRACK_ID_KEY)
        if isinstance(tid, str) and tid:
            return tid
        return None

    def save_track_id(self, track_id: Optional[str]) -> None:
        with self._lock:
            data = self._read_cache_unlocked()
            if track_id is None:
                data.pop(self.TRACK_ID_KEY, None)
            else:
                data[self.TRACK_ID_KEY] = str(track_id)
            self._write_cache_unlocked(data)

_SPG_CACHE_HANDLERS: dict[str, SPGCacheFileHandler] = {}

def get_spg_cache_handler(cache_path: str) -> SPGCacheFileHandler:
    path = str(cache_path)
    handler = _SPG_CACHE_HANDLERS.get(path)
    if handler is None:
        handler = SPGCacheFileHandler(path)
        _SPG_CACHE_HANDLERS[path] = handler
    return handler

class AsyncSpotifyClient:
    API_BASE = "https://api.spotify.com/v1"

    def __init__(
        self,
        auth_manager: SpotifyOAuth,
        session: Optional[aiohttp.ClientSession] = None,
        request_timeout: float = 5.0,
        language: Optional[str] = None,
    ) -> None:
        self._auth = auth_manager
        self._session = session or init_session()
        self._timeout = aiohttp.ClientTimeout(total=float(request_timeout))
        self._language = language or "EN"

    async def _get_access_token(self) -> str:
        try:
            token = await asyncio.to_thread(self._auth.get_access_token, as_dict=False)
        except TypeError:
            token = await asyncio.to_thread(self._auth.get_access_token)
        if isinstance(token, dict):
            return token.get("access_token", "")
        return str(token)

    async def _safe_json(self, resp: aiohttp.ClientResponse) -> Optional[dict[str, Any]]:
        try:
            return await resp.json()
        except (aiohttp.ContentTypeError, ValueError):
            return None

    async def current_playback(self) -> Optional[dict[str, Any]]:
        url = f"{self.API_BASE}/me/player"
        token = await self._get_access_token()
        for attempt in range(2):
            try:
                async with self._session.get(
                    url,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/json",
                        "Accept-Language": self._language,
                    },
                    timeout=self._timeout,
                ) as resp:
                    if resp.status == 204:
                        return {}
                    if 200 <= resp.status < 300:
                        return await self._safe_json(resp)
                    if resp.status == 401 and attempt == 0:
                        token = await self._get_access_token()
                        continue
                    return None
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                return None
        return None
