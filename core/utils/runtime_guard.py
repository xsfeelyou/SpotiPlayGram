from __future__ import annotations

import asyncio
import hashlib
import os
import platform
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, BinaryIO, Optional

if TYPE_CHECKING:
    import aiohttp

from constants import (
    ERROR_DEVICE_LOCK,
    ERROR_SINGLE_INSTANCE_LOCK,
    PROMPT_PRESS_ENTER,
    WARNING_DEVICE_LOCK_LOST,
    WARNING_DEVICE_LOCK_UNAVAILABLE,
)
from logger import _safe_log_error, _safe_log_warning, dev_logger

_lock_handle: Optional[BinaryIO] = None

def _try_lock(handle: BinaryIO) -> bool:
    if os.name == "nt":
        import msvcrt
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True

def acquire_single_instance_lock(path: str) -> bool:
    global _lock_handle
    if _lock_handle is not None:
        return True
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
            if os.name != "nt":
                os.chmod(parent, 0o700)
    except OSError as e:
        _safe_log_error(ERROR_SINGLE_INSTANCE_LOCK, e)
        return False
    try:
        handle = open(path, "a+b")
    except OSError as e:
        _safe_log_error(ERROR_SINGLE_INSTANCE_LOCK, e)
        return False
    if not _try_lock(handle):
        try:
            handle.close()
        except OSError:
            pass
        return False
    _lock_handle = handle
    return True

def wait_for_enter() -> None:
    try:
        if sys.stdin is None or not sys.stdin.isatty():
            return
    except (AttributeError, OSError, ValueError):
        return
    try:
        if sys.stdout is not None:
            sys.stdout.write(PROMPT_PRESS_ENTER)
            sys.stdout.flush()
    except (AttributeError, OSError, ValueError):
        return
    try:
        sys.stdin.readline()
    except (EOFError, OSError, RuntimeError, ValueError, UnicodeError, KeyboardInterrupt):
        pass

def set_console_title(title: str) -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW(title)
    except (AttributeError, OSError, TypeError, ValueError):
        pass

LOCK_PREFIX = "SPG1|"
FIELD_SHORT = "short"
FIELD_MAIN = "main"
SHORT_KEY = "short_description"
MAIN_KEY = "description"
HEARTBEAT_INTERVAL = 30.0
HEARTBEAT_FIRST_DELAY = 5.0
LOCK_TTL = 90.0
SETTLE_SECONDS = 2.0
API_TIMEOUT = 10.0
HOSTNAME_LIMIT = 24

class DeviceLockBusy(RuntimeError):
    def __init__(self, host: str):
        super().__init__(host)
        self.host = host

class BotApiError(RuntimeError):
    pass

@dataclass
class LeaseRecord:
    instance_id: str
    device_id: str
    host: str
    heartbeat: float
    started: float

@dataclass
class FieldState:
    field: str
    text: str
    record: Optional[LeaseRecord]
    server_ts: float

def _read_text_file(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            raw = f.read(4096)
    except OSError:
        return None
    if not raw:
        return None
    try:
        text = raw.decode("utf-8", errors="ignore").strip()
    except (UnicodeError, ValueError):
        return None
    return text or None

def _run_text(cmd: list[str]) -> Optional[str]:
    resolved = list(cmd)
    if resolved and not os.path.isabs(resolved[0]):
        found = shutil.which(resolved[0])
        if found:
            resolved[0] = found
    try:
        proc = subprocess.run(
            resolved,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (proc.stdout or "").strip() or None

def _machine_id() -> str:
    system = platform.system()
    if os.name == "nt":
        try:
            import winreg
            access = winreg.KEY_READ | getattr(winreg, "KEY_WOW64_64KEY", 0)
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography",
                0,
                access,
            ) as key:
                value, _ = winreg.QueryValueEx(key, "MachineGuid")
                if isinstance(value, str) and value.strip():
                    return value.strip()
        except (ImportError, OSError, TypeError, ValueError):
            pass
    elif system == "Darwin":
        output = _run_text(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"])
        if output:
            match = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', output)
            if match:
                return match.group(1)
    else:
        for path in ("/etc/machine-id", "/var/lib/dbus/machine-id", "/etc/hostid"):
            value = _read_text_file(path)
            if value:
                return value
        if system in ("FreeBSD", "OpenBSD", "NetBSD", "DragonFly"):
            value = _run_text(["kenv", "-q", "smbios.system.uuid"])
            if value:
                return value
    try:
        return socket.gethostname() or "unknown"
    except OSError:
        return "unknown"

def _device_id(base_dir: str) -> str:
    path = os.path.normcase(os.path.abspath(base_dir))
    raw = "{0}|{1}".format(_machine_id(), path)
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:12]

def _hostname() -> str:
    try:
        name = socket.gethostname()
    except OSError:
        name = "unknown"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._-")
    if not name:
        name = "unknown"
    return hashlib.sha256(name.encode("utf-8", errors="replace")).hexdigest()[:HOSTNAME_LIMIT]

def _parse_lease(text: str) -> Optional[LeaseRecord]:
    if not text or not text.startswith(LOCK_PREFIX):
        return None
    parts = text[len(LOCK_PREFIX):].strip().split("|")
    if len(parts) != 5:
        return None
    instance_id, device_id, host, heartbeat_raw, started_raw = parts
    if not instance_id or not device_id:
        return None
    try:
        heartbeat = float(heartbeat_raw)
        started = float(started_raw)
    except (TypeError, ValueError):
        return None
    return LeaseRecord(instance_id, device_id, host, heartbeat, started)

def _format_lease(record: LeaseRecord) -> str:
    return "{0}{1}|{2}|{3}|{4}|{5}".format(
        LOCK_PREFIX,
        record.instance_id,
        record.device_id,
        record.host,
        int(record.heartbeat),
        int(record.started),
    )

def _redact(text: str, token: str) -> str:
    if token:
        return text.replace(token, "***")
    return text

def _server_timestamp(headers) -> float:
    value = None
    try:
        value = headers.get("Date")
    except (AttributeError, TypeError):
        value = None
    if isinstance(value, str) and value.strip():
        try:
            dt = parsedate_to_datetime(value)
            if dt is not None:
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
        except (TypeError, ValueError, OverflowError, IndexError):
            pass
    return time.time()

async def _bot_api(
    session: aiohttp.ClientSession,
    token: str,
    method: str,
    payload: Optional[dict] = None,
) -> tuple[dict, float]:
    import aiohttp
    url = "https://api.telegram.org/bot{0}/{1}".format(token, method)
    timeout = aiohttp.ClientTimeout(total=API_TIMEOUT)
    try:
        if payload is None:
            async with session.get(url, timeout=timeout) as response:
                data = await response.json(content_type=None)
                server_ts = _server_timestamp(response.headers)
        else:
            async with session.post(url, json=payload, timeout=timeout) as response:
                data = await response.json(content_type=None)
                server_ts = _server_timestamp(response.headers)
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError, TypeError) as e:
        raise BotApiError(_redact(str(e), token)) from e
    if not isinstance(data, dict) or not data.get("ok"):
        detail = data.get("description") if isinstance(data, dict) else data
        raise BotApiError(str(detail or "bot api error"))
    result = data.get("result")
    return result if isinstance(result, dict) else {}, server_ts

async def _read_fields(session: aiohttp.ClientSession, token: str) -> list[FieldState]:
    (short_result, short_ts), (main_result, main_ts) = await asyncio.gather(
        _bot_api(session, token, "getMyShortDescription"),
        _bot_api(session, token, "getMyDescription"),
    )
    short_text = short_result.get(SHORT_KEY) or ""
    main_text = main_result.get(MAIN_KEY) or ""
    if not isinstance(short_text, str):
        short_text = ""
    if not isinstance(main_text, str):
        main_text = ""
    return [
        FieldState(FIELD_SHORT, short_text, _parse_lease(short_text), short_ts),
        FieldState(FIELD_MAIN, main_text, _parse_lease(main_text), main_ts),
    ]

async def _write_field(
    session: aiohttp.ClientSession,
    token: str,
    field: str,
    text: str,
) -> None:
    if field == FIELD_SHORT:
        await _bot_api(session, token, "setMyShortDescription", {SHORT_KEY: text})
        return
    await _bot_api(session, token, "setMyDescription", {MAIN_KEY: text})

async def _clear_owned_field(
    session: aiohttp.ClientSession,
    token: str,
    field: str,
    instance_id: str,
) -> None:
    try:
        fields = await _read_fields(session, token)
    except BotApiError:
        return
    current = next((state for state in fields if state.field == field), None)
    if current is None or current.record is None:
        return
    if current.record.instance_id != instance_id:
        return
    try:
        await _write_field(session, token, field, "")
    except BotApiError:
        pass

def _active_foreign(state: FieldState, instance_id: str, now: float) -> bool:
    return (
        state.record is not None
        and state.record.instance_id != instance_id
        and now - state.record.heartbeat <= LOCK_TTL
    )

class DeviceLease:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        token: str,
        field: str,
        record: LeaseRecord,
    ):
        self._session = session
        self._token = token
        self._field = field
        self._record = record
        self._lost = False
        self._released = False
        self._lock = asyncio.Lock()

    @property
    def lost(self) -> bool:
        return self._lost

    async def _write_current(self, now: float) -> bool:
        previous = self._record.heartbeat
        self._record.heartbeat = now
        try:
            await _write_field(
                self._session,
                self._token,
                self._field,
                _format_lease(self._record),
            )
        except BotApiError as e:
            self._record.heartbeat = previous
            _safe_log_warning(ERROR_DEVICE_LOCK, e)
            return False
        return True

    async def _clear_own(self) -> None:
        await _clear_owned_field(
            self._session,
            self._token,
            self._field,
            self._record.instance_id,
        )

    async def _refresh(self) -> bool:
        async with self._lock:
            if self._released:
                return True
            try:
                fields = await _read_fields(self._session, self._token)
            except BotApiError:
                return True
            now = max((state.server_ts for state in fields), default=time.time())
            for state in fields:
                if _active_foreign(state, self._record.instance_id, now):
                    await self._clear_own()
                    self._lost = True
                    return False
            current = next((state for state in fields if state.field == self._field), None)
            if current is None:
                self._lost = True
                return False
            if current.record is not None and current.record.instance_id == self._record.instance_id:
                if await self._write_current(now):
                    return True
                return now - self._record.heartbeat <= LOCK_TTL
            if not current.text.strip():
                return await self._write_current(now)
            if current.record is not None and now - current.record.heartbeat > LOCK_TTL:
                return await self._write_current(now)
            await self._clear_own()
            self._lost = True
            return False

    async def heartbeat_loop(self, on_lost=None) -> None:
        import aiohttp
        first = True
        while not self._lost and not self._released:
            await asyncio.sleep(HEARTBEAT_FIRST_DELAY if first else HEARTBEAT_INTERVAL)
            first = False
            if self._released:
                return
            try:
                alive = await self._refresh()
            except asyncio.CancelledError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError, TypeError, RuntimeError) as e:
                _safe_log_warning(ERROR_DEVICE_LOCK, e)
                continue
            if not alive:
                self._lost = True
                _safe_log_warning(WARNING_DEVICE_LOCK_LOST)
                if on_lost is not None:
                    try:
                        await on_lost()
                    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError, AttributeError, TypeError, ValueError):
                        pass
                return

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        async with self._lock:
            await self._clear_own()

async def acquire_device_lock(token: str, base_dir: str) -> Optional[DeviceLease]:
    from utils.http_session import init_session
    if not token:
        _safe_log_warning(WARNING_DEVICE_LOCK_UNAVAILABLE, "bot token is empty")
        return None
    session = init_session()
    try:
        fields = await _read_fields(session, token)
    except BotApiError as e:
        _safe_log_warning(WARNING_DEVICE_LOCK_UNAVAILABLE, e, logger=dev_logger)
        return None
    now = max((state.server_ts for state in fields), default=time.time())
    device_id = _device_id(base_dir)
    host = _hostname()
    for state in fields:
        if (
            state.record is not None
            and state.record.device_id != device_id
            and now - state.record.heartbeat <= LOCK_TTL
        ):
            raise DeviceLockBusy(state.record.host or "unknown device")
    own_fields = [
        state for state in fields
        if state.record is not None and state.record.device_id == device_id
    ]
    available = [
        state for state in fields
        if not state.text.strip() or state.record is not None
    ]
    if own_fields:
        target = own_fields[0]
    elif available:
        target = next(
            (state for state in available if state.field == FIELD_SHORT),
            available[0],
        )
    else:
        _safe_log_warning(
            WARNING_DEVICE_LOCK_UNAVAILABLE,
            "bot profile description fields are already in use",
        )
        return None
    for state in fields:
        if state.field == target.field:
            continue
        if (
            state.record is not None
            and (
                state.record.device_id == device_id
                or now - state.record.heartbeat > LOCK_TTL
            )
        ):
            try:
                await _write_field(session, token, state.field, "")
            except BotApiError:
                pass
    record = LeaseRecord(secrets.token_hex(4), device_id, host, now, now)
    try:
        await _write_field(session, token, target.field, _format_lease(record))
    except BotApiError as e:
        _safe_log_warning(WARNING_DEVICE_LOCK_UNAVAILABLE, e, logger=dev_logger)
        return None
    await asyncio.sleep(SETTLE_SECONDS)
    try:
        fields_after = await _read_fields(session, token)
    except BotApiError as e:
        _safe_log_warning(ERROR_DEVICE_LOCK, e)
        return DeviceLease(session, token, target.field, record)
    now_after = max((state.server_ts for state in fields_after), default=time.time())
    foreign = next(
        (state for state in fields_after if _active_foreign(state, record.instance_id, now_after)),
        None,
    )
    if foreign is not None:
        await _clear_owned_field(session, token, target.field, record.instance_id)
        raise DeviceLockBusy(foreign.record.host or "unknown device")
    current = next(
        (state for state in fields_after if state.field == target.field),
        None,
    )
    if (
        current is not None
        and current.record is not None
        and current.record.instance_id == record.instance_id
    ):
        record.heartbeat = max(now_after, record.heartbeat)
        return DeviceLease(session, token, target.field, record)
    _safe_log_warning(WARNING_DEVICE_LOCK_UNAVAILABLE, "the lock was overwritten during startup")
    return None
