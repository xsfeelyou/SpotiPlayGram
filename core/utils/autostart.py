from __future__ import annotations

import base64
import getpass
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import xml.sax.saxutils as saxutils
from typing import List, Optional

from constants import (
    APP_NAME,
    ENV_AUTOSTART_SUPERVISED,
    ERROR_AUTOSTART,
    ERROR_AUTOSTART_CRONTAB_READ,
    ERROR_AUTOSTART_CRONTAB_WRITE,
    ERROR_AUTOSTART_LINGER,
    INFO_AUTOSTART_ALREADY_OK,
    INFO_AUTOSTART_CRON_DETAIL,
    INFO_AUTOSTART_CRON_ENTRY,
    INFO_AUTOSTART_ENABLED,
    INFO_AUTOSTART_INSTALLED,
    INFO_AUTOSTART_LINGER_ENABLED,
    INFO_AUTOSTART_REMOVED,
    INFO_AUTOSTART_WINDOWS_REGISTRY,
    INFO_AUTOSTART_WINDOWS_SHORTCUT,
    INFO_AUTOSTART_XDG_DETAIL,
    WARNING_AUTOSTART_ALREADY_DISABLED,
    WARNING_AUTOSTART_DISABLED,
    WARNING_AUTOSTART_WINDOWS_FALLBACK,
)
from logger import _safe_log_error, _safe_log_info, _safe_log_warning

SERVICE_NAME = "spotiplaygram"
SERVICE_FILE = SERVICE_NAME + ".service"
LAUNCH_AGENT_LABEL = "com.spotisplaygram"
LAUNCH_AGENT_FILE = LAUNCH_AGENT_LABEL + ".plist"
DESKTOP_FILE = SERVICE_NAME + ".desktop"
REG_VALUE_NAME = APP_NAME
SHORTCUT_NAME = APP_NAME + ".lnk"
CRON_MARKER = "# " + APP_NAME
REGISTRY_RUN_PATH = r"Software\Microsoft\Windows\CurrentVersion\Run"

def _get_project_root(base_dir: Optional[str] = None) -> str:
    if base_dir:
        return os.path.abspath(base_dir)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))

def _get_start_script(base_dir: str) -> str:
    return os.path.join(base_dir, "start.py")

def _xdg_config_home() -> str:
    value = os.environ.get("XDG_CONFIG_HOME")
    if value and os.path.isabs(value):
        return value
    return os.path.join(os.path.expanduser("~"), ".config")

def _get_python_for_autostart(base_dir: str) -> str:
    if os.name == "nt":
        for candidate in (
            os.path.join(base_dir, "venv", "Scripts", "pythonw.exe"),
            os.path.join(base_dir, "venv", "Scripts", "python.exe"),
        ):
            if os.path.exists(candidate):
                return candidate
        if sys.executable:
            windowless = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
            if os.path.exists(windowless):
                return windowless
            return sys.executable
        return "pythonw.exe"
    for candidate in (
        os.path.join(base_dir, "venv", "bin", "python3"),
        os.path.join(base_dir, "venv", "bin", "python"),
    ):
        if os.path.exists(candidate):
            return candidate
    return sys.executable or "python3"

def _resolve_command(cmd: List[str]) -> List[str]:
    if not cmd:
        return cmd
    executable = cmd[0]
    if os.path.isabs(executable):
        return cmd
    resolved = shutil.which(executable)
    if not resolved:
        return cmd
    return [resolved] + cmd[1:]

def _run(cmd: List[str], timeout: float = 15.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        _resolve_command(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )

def _write_file_if_changed(path: str, content: str) -> bool:
    directory = os.path.dirname(path) or "."
    temp_path = None
    try:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    if f.read() == content:
                        return False
            except (OSError, UnicodeDecodeError):
                pass
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".spg-autostart-", dir=directory)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        os.replace(temp_path, path)
        temp_path = None
        return True
    except OSError:
        return False
    finally:
        if temp_path is not None:
            try:
                os.remove(temp_path)
            except OSError:
                pass

def _read_windows_run_value() -> Optional[str]:
    try:
        import winreg
    except ImportError:
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, REGISTRY_RUN_PATH, 0, winreg.KEY_QUERY_VALUE) as key:
            value, _ = winreg.QueryValueEx(key, REG_VALUE_NAME)
            return value if isinstance(value, str) else None
    except OSError:
        return None

def _windows_startup_dir() -> str:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        appdata = os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
    return os.path.join(appdata, "Microsoft", "Windows", "Start Menu", "Programs", "Startup")

def _windows_shortcut_path() -> str:
    return os.path.join(_windows_startup_dir(), SHORTCUT_NAME)

def _windows_launcher_path(base_dir: str) -> str:
    return os.path.join(base_dir, "venv", "Scripts", APP_NAME + ".exe")

def _files_match(first: str, second: str) -> bool:
    try:
        first_stat = os.stat(first)
        second_stat = os.stat(second)
    except OSError:
        return False
    return first_stat.st_size == second_stat.st_size and int(first_stat.st_mtime) == int(second_stat.st_mtime)

def _ensure_windows_launcher(base_dir: str) -> Optional[str]:
    launcher = _windows_launcher_path(base_dir)
    source = None
    for candidate in (
        os.path.join(base_dir, "venv", "Scripts", "pythonw.exe"),
        os.path.join(base_dir, "venv", "Scripts", "python.exe"),
    ):
        if os.path.exists(candidate):
            source = candidate
            break
    if source is None:
        return None
    if os.path.exists(launcher) and _files_match(launcher, source):
        return launcher
    try:
        shutil.copy2(source, launcher)
    except OSError as e:
        _safe_log_error(ERROR_AUTOSTART, e)
        return launcher if os.path.exists(launcher) else None
    return launcher if os.path.exists(launcher) else None

def _powershell_quote(value: str) -> str:
    return "'{0}'".format(value.replace("'", "''"))

def _run_powershell(script: str, timeout: float = 30.0) -> Optional[str]:
    executable = shutil.which("powershell") or shutil.which("powershell.exe") or shutil.which("pwsh")
    if not executable:
        system_root = os.environ.get("SystemRoot") or r"C:\Windows"
        candidate = os.path.join(
            system_root,
            "System32",
            "WindowsPowerShell",
            "v1.0",
            "powershell.exe",
        )
        if os.path.exists(candidate):
            executable = candidate
    if not executable:
        return None
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    try:
        proc = subprocess.run(
            [
                executable,
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-EncodedCommand",
                encoded,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout or ""

def _create_windows_shortcut(path: str, python_exe: str, start_py: str, base_dir: str) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError:
        return False
    script = (
        "$ErrorActionPreference = 'Stop'\n"
        "$shell = New-Object -ComObject WScript.Shell\n"
        "$link = $shell.CreateShortcut({0})\n"
        "$link.TargetPath = {1}\n"
        "$link.Arguments = {2}\n"
        "$link.WorkingDirectory = {3}\n"
        "$link.Description = {4}\n"
        "$link.WindowStyle = 7\n"
        "$link.Save()\n"
    ).format(
        _powershell_quote(path),
        _powershell_quote(python_exe),
        _powershell_quote('"{0}"'.format(start_py)),
        _powershell_quote(base_dir),
        _powershell_quote(APP_NAME),
    )
    return _run_powershell(script) is not None

def _read_windows_shortcut(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    script = (
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8\n"
        "$shell = New-Object -ComObject WScript.Shell\n"
        "$link = $shell.CreateShortcut({0})\n"
        "Write-Output $link.TargetPath\n"
        "Write-Output $link.Arguments\n"
        "Write-Output $link.WorkingDirectory\n"
    ).format(_powershell_quote(path))
    return _run_powershell(script)

def _windows_shortcut_matches(path: str, python_exe: str, start_py: str, base_dir: str) -> bool:
    output = _read_windows_shortcut(path)
    if output is None:
        return False
    lines = output.splitlines()
    if len(lines) < 3:
        return False
    target = os.path.normcase(os.path.abspath(lines[0].strip()))
    arguments = os.path.normcase(lines[1].strip())
    workdir = os.path.normcase(os.path.abspath(lines[2].strip()))
    return (
        target == os.path.normcase(os.path.abspath(python_exe))
        and arguments == os.path.normcase('"{0}"'.format(start_py))
        and workdir == os.path.normcase(os.path.abspath(base_dir))
    )

def _delete_windows_run_value() -> bool:
    try:
        import winreg
    except ImportError:
        return False
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, REGISTRY_RUN_PATH, 0, winreg.KEY_SET_VALUE) as key:
            try:
                winreg.DeleteValue(key, REG_VALUE_NAME)
            except FileNotFoundError:
                pass
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True

def _set_windows_run_value(command: str) -> bool:
    try:
        import winreg
    except ImportError:
        return False
    try:
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, REGISTRY_RUN_PATH, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, REG_VALUE_NAME, 0, winreg.REG_SZ, command)
    except OSError:
        return False
    return True

def _sync_windows(enabled: bool, base_dir: str) -> bool:
    shortcut_path = _windows_shortcut_path()
    location = "HKCU\\{0}\\{1}".format(REGISTRY_RUN_PATH, REG_VALUE_NAME)
    start_py = _get_start_script(base_dir)

    try:
        if enabled:
            python_exe = _ensure_windows_launcher(base_dir) or _get_python_for_autostart(base_dir)
            if _windows_shortcut_matches(shortcut_path, python_exe, start_py, base_dir):
                _delete_windows_run_value()
                _safe_log_info(INFO_AUTOSTART_ALREADY_OK)
                return True
            if _create_windows_shortcut(shortcut_path, python_exe, start_py, base_dir) and os.path.exists(shortcut_path):
                _delete_windows_run_value()
                _safe_log_info(INFO_AUTOSTART_ENABLED, INFO_AUTOSTART_WINDOWS_SHORTCUT)
                _safe_log_info(INFO_AUTOSTART_INSTALLED, shortcut_path)
                return True
            _safe_log_warning(WARNING_AUTOSTART_WINDOWS_FALLBACK, shortcut_path)
            try:
                if os.path.exists(shortcut_path):
                    os.remove(shortcut_path)
            except OSError:
                pass
            command = '"{0}" "{1}"'.format(python_exe, start_py)
            if not _set_windows_run_value(command):
                _safe_log_error(ERROR_AUTOSTART, location)
                return False
            _safe_log_info(INFO_AUTOSTART_ENABLED, INFO_AUTOSTART_WINDOWS_REGISTRY)
            _safe_log_info(INFO_AUTOSTART_INSTALLED, location)
            return True
        shortcut_existed = os.path.exists(shortcut_path)
        if shortcut_existed:
            try:
                os.remove(shortcut_path)
            except OSError as e:
                _safe_log_error(ERROR_AUTOSTART, e)
                return False
            if os.path.exists(shortcut_path):
                _safe_log_error(ERROR_AUTOSTART, shortcut_path)
                return False
        run_existed = _read_windows_run_value() is not None
        if run_existed and not _delete_windows_run_value():
            _safe_log_error(ERROR_AUTOSTART, location)
            return False
        if not shortcut_existed and not run_existed:
            _safe_log_info(WARNING_AUTOSTART_ALREADY_DISABLED)
            return True
        _safe_log_info(WARNING_AUTOSTART_DISABLED)
        _safe_log_info(INFO_AUTOSTART_REMOVED, shortcut_path if shortcut_existed else location)
        return True
    except OSError as e:
        _safe_log_error(ERROR_AUTOSTART, e)
        return False

def _systemd_path_escape(value: str) -> str:
    return value.replace("%", "%%")

def _systemd_exec_escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("%", "%%")
        .replace("$", "$$")
    )

def _systemd_unit_content(python_exe: str, start_py: str, base_dir: str) -> str:
    return (
        "[Unit]\n"
        "Description={0} - Spotify status in Telegram\n"
        "StartLimitIntervalSec=1800\n"
        "StartLimitBurst=40\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        "SyslogIdentifier={0}\n"
        "Environment={4}=1\n"
        'ExecStart="{1}" "{2}"\n'
        "WorkingDirectory={3}\n"
        "Restart=on-failure\n"
        "RestartSec=15\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    ).format(
        APP_NAME,
        _systemd_exec_escape(python_exe),
        _systemd_exec_escape(start_py),
        _systemd_path_escape(os.path.abspath(base_dir)),
        ENV_AUTOSTART_SUPERVISED,
    )

def _systemctl_available() -> bool:
    return shutil.which("systemctl") is not None

def _systemd_load_state() -> Optional[str]:
    if not _systemctl_available():
        return None
    try:
        proc = _run(["systemctl", "--user", "show", "-p", "LoadState", SERVICE_FILE])
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    for line in (proc.stdout or "").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "LoadState":
            state = value.strip()
            return state or None
    return None

def _systemd_unit_broken() -> bool:
    return _systemd_load_state() in ("bad-setting", "error")

def _remove_systemd_unit(unit_path: str) -> None:
    try:
        if os.path.exists(unit_path):
            os.remove(unit_path)
    except OSError:
        pass

def _remove_systemd_symlinks(unit_dir: str) -> None:
    try:
        entries = os.listdir(unit_dir)
    except OSError:
        return
    for entry in entries:
        if not entry.endswith(".wants"):
            continue
        link_path = os.path.join(unit_dir, entry, SERVICE_FILE)
        try:
            if os.path.islink(link_path) or os.path.exists(link_path):
                os.remove(link_path)
        except OSError:
            pass

def _enable_linger() -> None:
    if shutil.which("loginctl") is None:
        return
    try:
        user = getpass.getuser()
    except (KeyError, OSError):
        user = None
    cmd = ["loginctl", "enable-linger"]
    if user:
        cmd.append(user)
    try:
        proc = _run(cmd)
        if proc.returncode == 0:
            _safe_log_info(INFO_AUTOSTART_LINGER_ENABLED)
        else:
            reason = (proc.stderr or proc.stdout or "loginctl enable-linger failed").strip()
            _safe_log_error(ERROR_AUTOSTART_LINGER, reason)
    except (OSError, subprocess.SubprocessError) as e:
        _safe_log_error(ERROR_AUTOSTART_LINGER, e)

def _sync_linux_systemd(enabled: bool, base_dir: str) -> bool:
    start_py = _get_start_script(base_dir)
    python_exe = _get_python_for_autostart(base_dir)
    unit_dir = os.path.join(_xdg_config_home(), "systemd", "user")
    unit_path = os.path.join(unit_dir, SERVICE_FILE)

    if not _systemctl_available():
        if not enabled:
            _remove_systemd_unit(unit_path)
            _remove_systemd_symlinks(unit_dir)
        return _sync_xdg_and_cron(enabled, base_dir, reason="systemctl not found, using XDG/cron fallback")

    try:
        if enabled:
            content = _systemd_unit_content(python_exe, start_py, base_dir)
            unit_changed = _write_file_if_changed(unit_path, content)
            if not os.path.exists(unit_path):
                _safe_log_error(ERROR_AUTOSTART, unit_path)
                return _sync_xdg_and_cron(enabled, base_dir, reason="cannot write systemd unit")
            daemon = None
            if unit_changed:
                daemon = _run(["systemctl", "--user", "daemon-reload"])
            if daemon is not None and daemon.returncode != 0:
                reason = (daemon.stderr or daemon.stdout or "daemon-reload failed").strip()
                _safe_log_error(ERROR_AUTOSTART, reason)
                _remove_systemd_unit(unit_path)
                _remove_systemd_symlinks(unit_dir)
                return _sync_xdg_and_cron(enabled, base_dir, reason="systemctl --user unavailable")
            if _systemd_unit_broken():
                _safe_log_error(ERROR_AUTOSTART, "invalid systemd unit: " + unit_path)
                _remove_systemd_unit(unit_path)
                _remove_systemd_symlinks(unit_dir)
                _run(["systemctl", "--user", "daemon-reload"])
                return _sync_xdg_and_cron(enabled, base_dir, reason="invalid systemd unit")
            proc = _run(["systemctl", "--user", "enable", SERVICE_FILE])
            if proc.returncode != 0:
                reason = (proc.stderr or proc.stdout or "systemctl enable failed").strip()
                _safe_log_error(ERROR_AUTOSTART, reason)
                _remove_systemd_unit(unit_path)
                _remove_systemd_symlinks(unit_dir)
                return _sync_xdg_and_cron(enabled, base_dir, reason="systemctl enable failed")
            _enable_linger()
            if unit_changed:
                _safe_log_info(INFO_AUTOSTART_ENABLED, "systemd user service")
            else:
                _safe_log_info(INFO_AUTOSTART_ALREADY_OK)
            _cleanup_xdg_and_cron()
            _safe_log_info(INFO_AUTOSTART_INSTALLED, unit_path)
            return True
        proc = _run(["systemctl", "--user", "disable", SERVICE_FILE])
        _remove_systemd_unit(unit_path)
        _remove_systemd_symlinks(unit_dir)
        _run(["systemctl", "--user", "daemon-reload"])
        if os.path.exists(unit_path):
            _safe_log_error(ERROR_AUTOSTART, unit_path)
            return False
        _cleanup_xdg_and_cron()
        _safe_log_info(WARNING_AUTOSTART_DISABLED)
        _safe_log_info(INFO_AUTOSTART_REMOVED, unit_path)
        return proc.returncode == 0 or not os.path.exists(unit_path)
    except (OSError, subprocess.SubprocessError) as e:
        _safe_log_error(ERROR_AUTOSTART, e)
        return False

def _macos_launcher_path() -> str:
    return os.path.join(
        os.path.expanduser("~"),
        "Library",
        "Application Support",
        APP_NAME,
        APP_NAME,
    )

def _macos_launcher_content(python_exe: str, start_py: str) -> str:
    return "#!/bin/sh\nexec {0} {1}\n".format(shlex.quote(python_exe), shlex.quote(start_py))

def _write_macos_launcher(path: str, python_exe: str, start_py: str) -> Optional[bool]:
    content = _macos_launcher_content(python_exe, start_py)
    changed = _write_file_if_changed(path, content)
    try:
        os.chmod(path, 0o755)
        with open(path, "r", encoding="utf-8") as f:
            installed = f.read()
    except OSError as e:
        _safe_log_error(ERROR_AUTOSTART, e)
        return None
    if installed != content or not os.access(path, os.X_OK):
        _safe_log_error(ERROR_AUTOSTART, path)
        return None
    return changed

def _launchd_plist_content(executable: str, base_dir: str) -> str:
    log_dir = os.path.join(base_dir, "logs")
    out_log = os.path.join(log_dir, "autostart.log")
    err_log = os.path.join(log_dir, "autostart.error.log")
    esc = saxutils.escape
    args = "    <string>{0}</string>".format(esc(executable))
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        "<plist version=\"1.0\">\n"
        "<dict>\n"
        "  <key>Label</key>\n"
        "  <string>{0}</string>\n"
        "  <key>ProgramArguments</key>\n"
        "  <array>\n"
        "{1}\n"
        "  </array>\n"
        "  <key>WorkingDirectory</key>\n"
        "  <string>{2}</string>\n"
        "  <key>EnvironmentVariables</key>\n"
        "  <dict>\n"
        "    <key>{5}</key>\n"
        "    <string>1</string>\n"
        "  </dict>\n"
        "  <key>RunAtLoad</key>\n"
        "  <true/>\n"
        "  <key>KeepAlive</key>\n"
        "  <dict>\n"
        "    <key>SuccessfulExit</key>\n"
        "    <false/>\n"
        "  </dict>\n"
        "  <key>ThrottleInterval</key>\n"
        "  <integer>30</integer>\n"
        "  <key>StandardOutPath</key>\n"
        "  <string>{3}</string>\n"
        "  <key>StandardErrorPath</key>\n"
        "  <string>{4}</string>\n"
        "</dict>\n"
        "</plist>\n"
    ).format(
        esc(LAUNCH_AGENT_LABEL),
        args,
        esc(base_dir),
        esc(out_log),
        esc(err_log),
        esc(ENV_AUTOSTART_SUPERVISED),
    )

def _launchctl_gui_domain() -> Optional[str]:
    try:
        uid = str(os.getuid()) if hasattr(os, "getuid") else None
    except OSError:
        uid = None
    return "gui/{0}".format(uid) if uid is not None else None

def _sync_macos(enabled: bool, base_dir: str) -> bool:
    start_py = _get_start_script(base_dir)
    python_exe = _get_python_for_autostart(base_dir)
    launcher_path = _macos_launcher_path()
    plist_path = os.path.join(os.path.expanduser("~"), "Library", "LaunchAgents", LAUNCH_AGENT_FILE)
    try:
        if enabled:
            try:
                os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
            except OSError:
                pass
            launcher_changed = _write_macos_launcher(launcher_path, python_exe, start_py)
            if launcher_changed is None:
                return False
            content = _launchd_plist_content(launcher_path, base_dir)
            plist_changed = _write_file_if_changed(plist_path, content)
            if not os.path.exists(plist_path):
                _safe_log_error(ERROR_AUTOSTART, plist_path)
                return False
            domain = _launchctl_gui_domain()
            if shutil.which("launchctl") is not None and domain is not None:
                try:
                    target = "{0}/{1}".format(domain, LAUNCH_AGENT_LABEL)
                    if plist_changed or launcher_changed:
                        _run(["launchctl", "bootout", target])
                    enable_proc = _run(["launchctl", "enable", target])
                    bootstrap_proc = _run(["launchctl", "bootstrap", domain, plist_path])
                    if enable_proc.returncode != 0 or bootstrap_proc.returncode != 0:
                        detail = (bootstrap_proc.stderr or enable_proc.stderr or "").strip()
                        _safe_log_error(ERROR_AUTOSTART, detail[:300])
                        return False
                except (OSError, subprocess.SubprocessError) as e:
                    _safe_log_error(ERROR_AUTOSTART, e)
                    return False
            if plist_changed or launcher_changed:
                _safe_log_info(INFO_AUTOSTART_ENABLED, "macOS LaunchAgent")
            else:
                _safe_log_info(INFO_AUTOSTART_ALREADY_OK)
            _safe_log_info(INFO_AUTOSTART_INSTALLED, plist_path)
            return True
        plist_existed = os.path.exists(plist_path)
        launcher_existed = os.path.exists(launcher_path)
        domain = _launchctl_gui_domain()
        if shutil.which("launchctl") is not None and domain is not None:
            try:
                target = "{0}/{1}".format(domain, LAUNCH_AGENT_LABEL)
                _run(["launchctl", "bootout", target])
                disable_proc = _run(["launchctl", "disable", target])
                if disable_proc.returncode != 0:
                    _safe_log_error(ERROR_AUTOSTART, (disable_proc.stderr or "").strip()[:300])
            except (OSError, subprocess.SubprocessError) as e:
                _safe_log_error(ERROR_AUTOSTART, e)
        try:
            if plist_existed:
                os.remove(plist_path)
            if launcher_existed:
                os.remove(launcher_path)
        except OSError as e:
            _safe_log_error(ERROR_AUTOSTART, e)
            return False
        if os.path.exists(plist_path):
            _safe_log_error(ERROR_AUTOSTART, plist_path)
            return False
        if plist_existed or launcher_existed:
            _safe_log_info(WARNING_AUTOSTART_DISABLED)
            _safe_log_info(INFO_AUTOSTART_REMOVED, plist_path)
        else:
            _safe_log_info(WARNING_AUTOSTART_ALREADY_DISABLED)
        return True
    except (OSError, subprocess.SubprocessError) as e:
        _safe_log_error(ERROR_AUTOSTART, e)
        return False

def _desktop_exec_quote(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("`", "\\`")
        .replace("$", "\\$")
        .replace("%", "%%")
    )
    return '"{0}"'.format(escaped)

def _desktop_file_content(python_exe: str, start_py: str, base_dir: str) -> str:
    exec_line = "{0} {1}".format(
        _desktop_exec_quote(python_exe), _desktop_exec_quote(start_py)
    )
    return (
        "[Desktop Entry]\n"
        "Type=Application\n"
        "Name={0}\n"
        "Comment={0} - Spotify status in Telegram\n"
        "Exec={1}\n"
        "Path={2}\n"
        "Terminal=false\n"
        "X-GNOME-Autostart-enabled=true\n"
    ).format(APP_NAME, exec_line, base_dir)

def _cron_line(python_exe: str, start_py: str) -> str:
    return "@reboot {0} {1} {2}".format(
        shlex.quote(python_exe).replace("%", "\\%"),
        shlex.quote(start_py).replace("%", "\\%"),
        CRON_MARKER,
    )

def _read_crontab() -> Optional[List[str]]:
    if shutil.which("crontab") is None:
        return []
    try:
        proc = _run(["crontab", "-l"])
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").lower()
            if "no crontab" in detail:
                return []
            return None
        return (proc.stdout or "").splitlines()
    except (OSError, subprocess.SubprocessError):
        return None

def _write_crontab(lines: List[str]) -> bool:
    if shutil.which("crontab") is None:
        return False
    try:
        text = "\n".join(lines)
        if text:
            text += "\n"
        proc = subprocess.run(
            _resolve_command(["crontab", "-"]),
            input=text,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15.0,
            check=False,
        )
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False

def _ensure_cron_entry(python_exe: str, start_py: str) -> bool:
    if shutil.which("crontab") is None:
        return False
    lines = _read_crontab()
    if lines is None:
        _safe_log_error(ERROR_AUTOSTART_CRONTAB_READ, "crontab -l failed")
        return False
    wanted = _cron_line(python_exe, start_py)
    marker_lines = [ln for ln in lines if CRON_MARKER in ln]
    if marker_lines == [wanted]:
        return True
    kept = [ln for ln in lines if CRON_MARKER not in ln]
    kept.append(wanted)
    if not _write_crontab(kept):
        _safe_log_error(ERROR_AUTOSTART_CRONTAB_WRITE, "crontab - failed")
        return False
    return True

def _remove_cron_entry() -> bool:
    if shutil.which("crontab") is None:
        return True
    lines = _read_crontab()
    if lines is None:
        _safe_log_error(ERROR_AUTOSTART_CRONTAB_READ, "crontab -l failed")
        return False
    kept = [ln for ln in lines if CRON_MARKER not in ln]
    if kept == lines:
        return True
    if not _write_crontab(kept):
        _safe_log_error(ERROR_AUTOSTART_CRONTAB_WRITE, "crontab - failed")
        return False
    return True

def _remove_desktop_file() -> bool:
    desktop_path = os.path.join(_xdg_config_home(), "autostart", DESKTOP_FILE)
    try:
        if os.path.exists(desktop_path):
            os.remove(desktop_path)
    except OSError:
        return False
    return not os.path.exists(desktop_path)

def _has_gui_session() -> bool:
    if os.environ.get("SSH_CONNECTION"):
        return False
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

def _install_xdg_entry(
    python_exe: str,
    start_py: str,
    base_dir: str,
    desktop_path: str,
) -> Optional[bool]:
    content = _desktop_file_content(python_exe, start_py, base_dir)
    changed = _write_file_if_changed(desktop_path, content)
    if not os.path.exists(desktop_path):
        _safe_log_error(ERROR_AUTOSTART, desktop_path)
        return None
    return changed

def _log_xdg_install(changed: bool, desktop_path: str) -> None:
    if changed:
        _safe_log_info(INFO_AUTOSTART_ENABLED, INFO_AUTOSTART_XDG_DETAIL)
    else:
        _safe_log_info(INFO_AUTOSTART_ALREADY_OK)
    _safe_log_info(INFO_AUTOSTART_INSTALLED, desktop_path)

def _cleanup_xdg_and_cron() -> None:
    _remove_desktop_file()
    _remove_cron_entry()

def _sync_xdg_and_cron(enabled: bool, base_dir: str, reason: str = "") -> bool:
    start_py = _get_start_script(base_dir)
    python_exe = _get_python_for_autostart(base_dir)
    desktop_path = os.path.join(_xdg_config_home(), "autostart", DESKTOP_FILE)
    try:
        if enabled:
            if _has_gui_session():
                if not _remove_cron_entry():
                    _safe_log_error(ERROR_AUTOSTART_CRONTAB_WRITE, "cannot remove cron entry")
                    return False
                changed = _install_xdg_entry(python_exe, start_py, base_dir, desktop_path)
                if changed is not None:
                    _log_xdg_install(changed, desktop_path)
                    return True
                if _ensure_cron_entry(python_exe, start_py):
                    _safe_log_info(INFO_AUTOSTART_ENABLED, INFO_AUTOSTART_CRON_DETAIL)
                    _safe_log_info(INFO_AUTOSTART_INSTALLED, INFO_AUTOSTART_CRON_ENTRY)
                    return True
                _safe_log_error(ERROR_AUTOSTART, reason or "no autostart mechanism available")
                return False
            if _ensure_cron_entry(python_exe, start_py):
                _remove_desktop_file()
                _safe_log_info(INFO_AUTOSTART_ENABLED, INFO_AUTOSTART_CRON_DETAIL)
                _safe_log_info(INFO_AUTOSTART_INSTALLED, INFO_AUTOSTART_CRON_ENTRY)
                return True
            if shutil.which("crontab") is None:
                changed = _install_xdg_entry(python_exe, start_py, base_dir, desktop_path)
                if changed is not None:
                    _log_xdg_install(changed, desktop_path)
                    return True
            _safe_log_error(ERROR_AUTOSTART, reason or "no autostart mechanism available")
            return False
        if not _remove_desktop_file() and os.path.exists(desktop_path):
            _safe_log_error(ERROR_AUTOSTART, desktop_path)
            return False
        if not _remove_cron_entry():
            return False
        _safe_log_info(WARNING_AUTOSTART_DISABLED)
        _safe_log_info(INFO_AUTOSTART_REMOVED, desktop_path)
        return True
    except (OSError, subprocess.SubprocessError) as e:
        _safe_log_error(ERROR_AUTOSTART, e)
        return False

def _systemd_unit_path() -> str:
    return os.path.join(_xdg_config_home(), "systemd", "user", SERVICE_FILE)

def _desktop_autostart_path() -> str:
    return os.path.join(_xdg_config_home(), "autostart", DESKTOP_FILE)

def _launch_agent_path() -> str:
    return os.path.join(os.path.expanduser("~"), "Library", "LaunchAgents", LAUNCH_AGENT_FILE)

def _cron_entry_present() -> bool:
    lines = _read_crontab()
    if not lines:
        return False
    return any(CRON_MARKER in line for line in lines)

def repair_autostart() -> bool:
    try:
        if os.name == "nt" or platform.system() != "Linux":
            return False
        if not os.path.exists(_systemd_unit_path()):
            return False
        if not _systemd_unit_broken():
            return False
        return _sync_linux_systemd(True, _get_project_root())
    except (OSError, subprocess.SubprocessError):
        return False

def is_autostart_enabled() -> bool:
    try:
        system = platform.system()
        if os.name == "nt" or system == "Windows":
            return os.path.exists(_windows_shortcut_path()) or _read_windows_run_value() is not None
        if system == "Linux":
            unit_present = os.path.exists(_systemd_unit_path()) and not _systemd_unit_broken()
            return (
                unit_present
                or os.path.exists(_desktop_autostart_path())
                or _cron_entry_present()
            )
        if system == "Darwin":
            return os.path.exists(_launch_agent_path()) and os.path.exists(_macos_launcher_path())
        return os.path.exists(_desktop_autostart_path()) or _cron_entry_present()
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        return False

def sync_autostart(enabled: bool, base_dir: Optional[str] = None) -> bool:
    try:
        root = _get_project_root(base_dir)
        system = platform.system()
        if os.name == "nt" or system == "Windows":
            return _sync_windows(bool(enabled), root)
        if system == "Linux":
            return _sync_linux_systemd(bool(enabled), root)
        if system == "Darwin":
            return _sync_macos(bool(enabled), root)
        if system in ("FreeBSD", "OpenBSD", "NetBSD", "DragonFly"):
            return _sync_xdg_and_cron(bool(enabled), root, reason=system)
        return _sync_xdg_and_cron(bool(enabled), root, reason=system or "unknown OS")
    except Exception as e:
        try:
            _safe_log_error(ERROR_AUTOSTART, e)
        except (ValueError, TypeError, OSError):
            pass
        return False
