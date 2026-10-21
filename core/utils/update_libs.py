from __future__ import annotations

import ast
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from constants import (
    ERROR_DEP_UPDATES,
    INFO_DEP_UPDATES_DONE,
    INFO_DEP_UPDATES_FOUND,
    STATUS_IDLE_POLL_SECONDS,
    UPDATE_CHECK_TIME_UTC,
)
from logger import _safe_log_error, _safe_log_info, _safe_log_warning

_READ_TRACK_INFO_FN: Optional[Callable] = None
_PIP_TIMEOUT_SECONDS = 600.0

def _ast_str_value(node) -> str | None:
    if isinstance(node, str):
        return node
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if hasattr(ast, "Str") and isinstance(node, ast.Str):
        return node.s
    return None

def _read_required_pip_names_from_main() -> set[str]:
    main_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.py")
    try:
        with open(main_path, "r", encoding="utf-8") as f:
            source = f.read()
    except (OSError, UnicodeDecodeError) as e:
        _safe_log_warning(ERROR_DEP_UPDATES, e)
        return set()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        _safe_log_warning(ERROR_DEP_UPDATES, e)
        return set()

    pip_names: set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "install_libs":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    if not any(isinstance(t, ast.Name) and t.id == "libs" for t in stmt.targets):
                        continue
                    value = stmt.value
                    if isinstance(value, (ast.List, ast.Tuple)):
                        elements = value.elts
                    else:
                        continue

                    for el in elements:
                        if isinstance(el, (ast.Tuple, ast.List)) and len(el.elts) == 2:
                            pip_name_node = el.elts[1]
                            pip_name = _ast_str_value(pip_name_node)
                            if isinstance(pip_name, str) and pip_name:
                                pip_names.add(pip_name)
            break
    return pip_names

async def _run_pip(args: list[str]) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "pip",
        *args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=_PIP_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError("pip command timed out")
    except asyncio.CancelledError:
        proc.kill()
        await proc.wait()
        raise
    return (
        proc.returncode if proc.returncode is not None else -1,
        (stdout or b"").decode("utf-8", errors="replace"),
        (stderr or b"").decode("utf-8", errors="replace"),
    )

async def _pip_outdated_names() -> Optional[set[str]]:
    try:
        code, stdout, stderr = await _run_pip(["list", "--outdated", "--format=json"])
    except (OSError, asyncio.SubprocessError, RuntimeError) as e:
        _safe_log_warning(ERROR_DEP_UPDATES, e)
        return None
    if code != 0:
        _safe_log_warning(ERROR_DEP_UPDATES, stderr.strip()[:300])
        return None
    try:
        data = json.loads(stdout or "[]")
    except json.JSONDecodeError as e:
        _safe_log_warning(ERROR_DEP_UPDATES, e)
        return None
    return {
        item.get("name")
        for item in data
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }

async def _upgrade_packages(pip_names: list[str]) -> None:
    if not pip_names:
        return
    code, _, stderr = await _run_pip(["install", "--upgrade"] + pip_names)
    if code != 0:
        raise RuntimeError(stderr.strip()[:500] or f"pip exited with code {code}")

def _seconds_until_next_utc(hhmm: str) -> float:
    try:
        parts = hhmm.strip().split(":")
        hh = int(parts[0])
        mm = int(parts[1]) if len(parts) > 1 else 0
        if not (0 <= hh < 24 and 0 <= mm < 60):
            raise ValueError
    except (AttributeError, ValueError, TypeError, IndexError):
        hh, mm = 0, 0

    now_utc = datetime.now(timezone.utc)
    target = now_utc.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if target <= now_utc:
        target = target + timedelta(days=1)
    return (target - now_utc).total_seconds()

def _is_idle_status() -> bool:
    global _READ_TRACK_INFO_FN
    try:
        if _READ_TRACK_INFO_FN is None:
            from status.spotify_status import read_track_info
            if callable(read_track_info):
                _READ_TRACK_INFO_FN = read_track_info
        if not callable(_READ_TRACK_INFO_FN):
            return False
        info = _READ_TRACK_INFO_FN()
        return info.get("track_id") is None
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError, OSError) as e:
        _safe_log_warning(ERROR_DEP_UPDATES, e)
        return False

async def _wait_for_idle_status() -> None:
    while True:
        if _is_idle_status():
            return
        await asyncio.sleep(STATUS_IDLE_POLL_SECONDS)

def _start_script_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "start.py")

async def schedule_daily_update_check(
    shutdown_cb: Callable[[], asyncio.Future] | Callable[[], None] | None,
    *,
    enabled: bool = True,
) -> None:
    if not enabled:
        return
    while True:
        delay = _seconds_until_next_utc(UPDATE_CHECK_TIME_UTC)
        if delay > 0:
            await asyncio.sleep(delay)
        await _wait_for_idle_status()
        try:
            required = _read_required_pip_names_from_main()
            if not required:
                continue

            outdated = await _pip_outdated_names()
            if outdated is None:
                continue
            required_by_lower = {name.lower(): name for name in required}
            to_update = sorted(
                required_by_lower[name.lower()]
                for name in outdated
                if name.lower() in required_by_lower
            )
            if not to_update:
                continue

            _safe_log_info(INFO_DEP_UPDATES_FOUND, ", ".join(to_update))
            await _upgrade_packages(to_update)

            if shutdown_cb is not None:
                try:
                    maybe_coro = shutdown_cb()
                    if asyncio.iscoroutine(maybe_coro):
                        await maybe_coro
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    _safe_log_error(ERROR_DEP_UPDATES, e)

            _safe_log_info(INFO_DEP_UPDATES_DONE)
            start_script = _start_script_path()
            try:
                if os.path.exists(start_script):
                    os.execv(sys.executable, [sys.executable, start_script])
                os.execv(sys.executable, [sys.executable] + sys.argv)
            except OSError as e:
                _safe_log_error(ERROR_DEP_UPDATES, e)
                os._exit(1)
        except asyncio.CancelledError:
            raise
        except (OSError, RuntimeError, ValueError, TypeError, asyncio.TimeoutError, ConnectionError) as e:
            _safe_log_error(ERROR_DEP_UPDATES, e)
            continue
