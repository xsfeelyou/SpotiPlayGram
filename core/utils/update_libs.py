from __future__ import annotations

import ast
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from typing import Callable

from constants import (
    INFO_DEP_UPDATES_FOUND,
    INFO_DEP_UPDATES_DONE,
    ERROR_DEP_UPDATES,
    UPDATE_CHECK_TIME_UTC,
    STATUS_IDLE_POLL_SECONDS,
    DIRS,
)
from logger import _safe_log_info, _safe_log_error

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
    except (OSError, UnicodeDecodeError):
        return set()

    try:
        tree = ast.parse(source)
    except SyntaxError:
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

def _pip_outdated_names() -> set[str]:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return set()
        data = json.loads(proc.stdout or "[]")
        return {item.get("name") for item in data if isinstance(item, dict) and isinstance(item.get("name"), str)}
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError):
        return set()

def _upgrade_packages(pip_names: list[str]) -> None:
    if not pip_names:
        return
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + pip_names
    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

def _restart_process() -> None:
    os.execv(sys.executable, [sys.executable] + sys.argv)

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
    try:
        from status.spotify_status import read_track_info
        info = read_track_info()
        return info.get("track_id") is None
    except Exception:
        return True

async def _wait_for_idle_status() -> None:
    while True:
        if _is_idle_status():
            return
        await asyncio.sleep(STATUS_IDLE_POLL_SECONDS)

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

            outdated = _pip_outdated_names()
            if not outdated:
                continue
            to_update = sorted(required & outdated)
            if not to_update:
                continue

            _safe_log_info(INFO_DEP_UPDATES_FOUND, ", ".join(to_update))
            if shutdown_cb is not None:
                try:
                    maybe_coro = shutdown_cb()
                    if asyncio.iscoroutine(maybe_coro):
                        await maybe_coro
                except asyncio.CancelledError:
                    raise
                except (OSError, ValueError, TypeError, RuntimeError, asyncio.TimeoutError, ConnectionError):
                    pass

            _upgrade_packages(to_update)
            _safe_log_info(INFO_DEP_UPDATES_DONE)
            _restart_process()
        except asyncio.CancelledError:
            raise
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError, TypeError, RuntimeError, asyncio.TimeoutError, ConnectionError) as e:
            _safe_log_error(ERROR_DEP_UPDATES, e)
            continue
