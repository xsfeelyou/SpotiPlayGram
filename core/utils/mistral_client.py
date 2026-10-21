from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

import aiohttp

from constants import ERROR_UNEXPECTED, MISTRAL_TIMEOUT_SECONDS
from logger import _safe_log_warning
from utils.http_session import init_session

try:
    from mistralai.client import Mistral as MistralClient
except ImportError:
    try:
        from mistralai import Mistral as MistralClient
    except ImportError:
        MistralClient = None

_mistral_client: Optional[Any] = None
_mistral_client_api_key: Optional[str] = None
_mistral_client_lock: Optional[asyncio.Lock] = None

def get_mistral_client_class() -> Optional[Any]:
    return MistralClient

def get_mistral_client_lock() -> asyncio.Lock:
    global _mistral_client_lock
    if _mistral_client_lock is None:
        _mistral_client_lock = asyncio.Lock()
    return _mistral_client_lock

async def close_mistral_client_unlocked() -> None:
    global _mistral_client, _mistral_client_api_key
    client = _mistral_client
    _mistral_client = None
    _mistral_client_api_key = None
    if client is None:
        return
    try:
        await client.__aexit__(None, None, None)
    except asyncio.CancelledError:
        raise
    except (RuntimeError, OSError, AttributeError, TypeError, ValueError) as e:
        _safe_log_warning(ERROR_UNEXPECTED, e)
    except Exception as e:
        _safe_log_warning(ERROR_UNEXPECTED, e)

async def get_or_create_mistral_client(api_key: str) -> Optional[Any]:
    global _mistral_client, _mistral_client_api_key
    if MistralClient is None:
        return None
    lock = get_mistral_client_lock()
    async with lock:
        if _mistral_client is not None and _mistral_client_api_key == api_key:
            return _mistral_client
        if _mistral_client is not None:
            await close_mistral_client_unlocked()
        try:
            _mistral_client = MistralClient(api_key=api_key)
        except (TypeError, ValueError, RuntimeError) as e:
            _safe_log_warning(ERROR_UNEXPECTED, e)
            _mistral_client = None
            _mistral_client_api_key = None
            return None
        _mistral_client_api_key = api_key
        return _mistral_client

async def close_mistral_client() -> None:
    lock = get_mistral_client_lock()
    async with lock:
        await close_mistral_client_unlocked()

async def prefetch_mistral_client(api_key: Optional[str]) -> None:
    if not api_key or MistralClient is None:
        return
    try:
        await get_or_create_mistral_client(api_key)
    except asyncio.CancelledError:
        raise
    except (RuntimeError, OSError, ValueError, TypeError) as e:
        _safe_log_warning(ERROR_UNEXPECTED, e)
    except Exception as e:
        _safe_log_warning(ERROR_UNEXPECTED, e)

async def complete_mistral_chat(api_key: str, chat_kwargs: dict[str, Any], timeout_seconds: float) -> Optional[Any]:
    mc = await get_or_create_mistral_client(api_key)
    if mc is None:
        return None
    return await asyncio.wait_for(
        mc.chat.complete_async(**chat_kwargs),
        timeout=timeout_seconds,
    )

async def probe_mistral_chat_payload(
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout_seconds: Optional[float] = None,
    max_retries: int = 3,
) -> tuple[bool, Any]:
    url = "https://api.mistral.ai/v1/chat/completions"
    session = init_session()
    timeout = aiohttp.ClientTimeout(total=float(timeout_seconds if timeout_seconds is not None else MISTRAL_TIMEOUT_SECONDS))
    for attempt in range(max_retries):
        try:
            async with session.post(url, headers=headers, json=payload, timeout=timeout) as resp:
                if resp.status == 200:
                    try:
                        data = await resp.json(content_type=None)
                    except (aiohttp.ContentTypeError, json.JSONDecodeError, ValueError):
                        data = await resp.text()
                    return True, data
                if resp.status in (400, 404, 422):
                    return False, None
                if resp.status == 429:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                if resp.status >= 500:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                return False, None
        except asyncio.TimeoutError:
            await asyncio.sleep(1.0 * (attempt + 1))
            continue
        except (aiohttp.ClientError, OSError):
            await asyncio.sleep(0.5)
            continue
    return False, None
