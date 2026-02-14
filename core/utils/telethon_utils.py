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
) -> Any:
    if call is None:
        if request is None:
            raise ValueError
        call = lambda: client(request)
    attempt = 0
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
            await asyncio.sleep(e.seconds + 1)
        except FileReferenceExpiredError:
            await asyncio.sleep(delay)
        except (RPCError, ConnectionError, asyncio.TimeoutError, OSError, aiohttp.ClientError):
            if attempt >= retries:
                if raise_on_error:
                    raise
                return None
            await asyncio.sleep(delay)
            delay = min(delay * 2, 5.0)
            attempt += 1
