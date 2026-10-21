import asyncio
from typing import Any, Optional, Callable
import aiohttp
from telethon.errors import RPCError
from telethon.errors.rpcerrorlist import (
    FloodWaitError,
    SlowModeWaitError,
    FileReferenceExpiredError,
    MessageNotModifiedError,
    ChatNotModifiedError,
)
from constants import ERROR_FILE_REFERENCE_EXPIRED, WARNING_FLOOD_WAIT
from logger import _safe_log_warning, dev_logger

class _SafeCallSuccess:
    __slots__ = ()

    def __bool__(self) -> bool:
        return False

_SAFE_CALL_SUCCESS = _SafeCallSuccess()

async def safe_call_telethon(
    client,
    request: Any = None,
    *,
    call: Optional[Callable[[], Any]] = None,
    retries: int = 3,
    base_delay: float = 0.5,
    treat_none_as_success: bool = False,
    raise_on_error: bool = True,
    max_flood_wait: float = 300.0,
) -> Any:
    if call is None:
        if request is None:
            raise ValueError
        call = lambda: client(request)
    attempt = 0
    flood_attempts = 0
    delay = base_delay
    while True:
        try:
            result = await call()
            if result is None and treat_none_as_success:
                return _SAFE_CALL_SUCCESS
            return result
        except (MessageNotModifiedError, ChatNotModifiedError):
            if treat_none_as_success:
                return _SAFE_CALL_SUCCESS
            return None
        except (FloodWaitError, SlowModeWaitError) as e:
            flood_attempts += 1
            wait_seconds = float(e.seconds) + 1.0
            if wait_seconds > max_flood_wait or flood_attempts > 3:
                _safe_log_warning(WARNING_FLOOD_WAIT, int(wait_seconds), logger=dev_logger)
                if raise_on_error:
                    raise
                return None
            _safe_log_warning(WARNING_FLOOD_WAIT, int(wait_seconds), logger=dev_logger)
            await asyncio.sleep(wait_seconds)
        except FileReferenceExpiredError:
            _safe_log_warning(ERROR_FILE_REFERENCE_EXPIRED, logger=dev_logger)
            if raise_on_error:
                raise
            return None
        except (RPCError, ConnectionError, asyncio.TimeoutError, OSError, aiohttp.ClientError):
            if attempt >= retries:
                if raise_on_error:
                    raise
                return None
            await asyncio.sleep(delay)
            delay = min(delay * 2, 5.0)
            attempt += 1
