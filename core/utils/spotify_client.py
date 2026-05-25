from __future__ import annotations

import asyncio
import json
import os
import threading
import time
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
        self._access_token = ""
        self._access_token_expires_at = 0.0
        self._token_lock = asyncio.Lock()
        self._token_refresh_margin = 60.0
        self._token_fallback_ttl = 300.0

    def _cached_token_valid(self, now: float) -> bool:
        return bool(self._access_token and self._access_token_expires_at > now + self._token_refresh_margin)

    def _extract_token_data(self, token: Any, now: float) -> tuple[str, float]:
        if token is None:
            return "", 0.0
        if isinstance(token, dict):
            value = token.get("access_token")
            if not value:
                return "", 0.0
            expires_at_raw = token.get("expires_at")
            expires_at = float(expires_at_raw) if isinstance(expires_at_raw, (int, float)) else 0.0
            if expires_at <= now:
                expires_in_raw = token.get("expires_in")
                expires_in = float(expires_in_raw) if isinstance(expires_in_raw, (int, float)) and expires_in_raw > 0 else self._token_fallback_ttl
                expires_at = now + expires_in
            return str(value), expires_at
        return str(token) if token else "", now + self._token_fallback_ttl

    async def _refresh_access_token(self, now: float) -> tuple[str, float]:
        cache_handler = getattr(self._auth, "cache_handler", None)
        get_cached_token = getattr(cache_handler, "get_cached_token", None)
        if not callable(get_cached_token):
            return "", 0.0
        try:
            cached_token = await asyncio.to_thread(get_cached_token)
        except Exception:
            return "", 0.0
        if not isinstance(cached_token, dict):
            return "", 0.0
        refresh_token = cached_token.get("refresh_token")
        if not refresh_token:
            return "", 0.0
        refresh_access_token = getattr(self._auth, "refresh_access_token", None)
        if not callable(refresh_access_token):
            return "", 0.0
        try:
            token = await asyncio.to_thread(refresh_access_token, refresh_token)
        except Exception:
            return "", 0.0
        return self._extract_token_data(token, now)

    async def _load_access_token(self, *, force_refresh: bool = False) -> tuple[str, float]:
        now = time.time()
        if force_refresh:
            token, expires_at = await self._refresh_access_token(now)
            if token:
                return token, expires_at
            return "", 0.0
        try:
            token = await asyncio.to_thread(self._auth.get_access_token, as_dict=True)
        except TypeError:
            token = await asyncio.to_thread(self._auth.get_access_token)
        return self._extract_token_data(token, now)

    async def _get_access_token(self, *, force_refresh: bool = False) -> str:
        now = time.time()
        if not force_refresh and self._cached_token_valid(now):
            return self._access_token
        async with self._token_lock:
            now = time.time()
            if not force_refresh and self._cached_token_valid(now):
                return self._access_token
            token, expires_at = await self._load_access_token(force_refresh=force_refresh)
            self._access_token = token
            self._access_token_expires_at = expires_at
            return token

    async def _safe_json(self, resp: aiohttp.ClientResponse) -> Optional[dict[str, Any]]:
        try:
            return await resp.json()
        except (aiohttp.ContentTypeError, ValueError):
            return None

    async def current_playback(self) -> Optional[dict[str, Any]]:
        url = f"{self.API_BASE}/me/player"
        token = await self._get_access_token()
        if not token:
            return None
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
                        token = await self._get_access_token(force_refresh=True)
                        if not token:
                            return None
                        continue
                    return None
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                if attempt == 0:
                    continue
                return None
        return None
