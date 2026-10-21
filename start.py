import os
import subprocess
import sys
import tempfile
import time
import importlib.util
from typing import Iterable, Optional

_START_DIR = os.path.dirname(os.path.abspath(__file__))
_CHDIR_ERROR = None
try:
    os.chdir(_START_DIR)
except OSError as e:
    _CHDIR_ERROR = e
_CORE = os.path.join(_START_DIR, "core")
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

_SUPERVISOR_LOG_DIR_SET = "SPG_LOG_DIR" not in os.environ
if _SUPERVISOR_LOG_DIR_SET:
    os.environ["SPG_LOG_DIR"] = os.path.join(_START_DIR, "logs", "supervisor")

from constants import (
    APP_NAME,
    DIRS,
    ENV_AUTOSTART_SUPERVISED,
    ENV_LOCAL_LOCK_HELD,
    ENV_NO_PAUSE,
    ERROR_START_RESTART_LIMIT,
    INFO_SINGLE_INSTANCE,
    INFO_START_RESTART,
    WARNING_CHDIR_FAILED,
)
from logger import _safe_log_error, _safe_log_info, _safe_log_warning
from utils.runtime_guard import acquire_single_instance_lock, set_console_title, wait_for_enter

if _CHDIR_ERROR is not None:
    _safe_log_warning(WARNING_CHDIR_FAILED, _CHDIR_ERROR)

START_RESTART_BASE_DELAY = 15
START_RESTART_MAX_DELAY = 300
START_RESTART_MAX_ATTEMPTS = 10
START_RESTART_STABLE_UPTIME = 600

INFO_VENV_CREATING = "Creating virtual environment..."
INFO_VENV_CREATED = "Virtual environment created successfully"
ERROR_VENV_CREATE = "Error creating virtual environment: {0}"
ERROR_PIP_UPGRADE = "Error upgrading pip: {0}"
ERROR_RUN_MAIN = "Error running main script: {0}"
ERROR_VENV_MODULE_MISSING = (
    "ERROR: venv module is not available. Please install python3-venv package."
)
ERROR_START_UNEXPECTED = "Unexpected error occurred: {0}"

class StartConfig:
    def __init__(
        self,
        venv_dir: str = "venv",
        entry_script: str = os.path.join("core", "main.py"),
        upgrade_pip: bool = True,
        use_system_python: bool = False,
        quiet_pip: bool = True,
    ):
        self.venv_dir = venv_dir
        self.entry_script = entry_script
        self.upgrade_pip = upgrade_pip
        self.use_system_python = use_system_python
        self.quiet_pip = quiet_pip

def _resolve_path(base_dir: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base_dir, path)

def _is_venv_active() -> bool:
    if hasattr(sys, "real_prefix"):
        return True
    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    return base_prefix != sys.prefix

def _venv_python_candidates(venv_dir: str) -> Iterable[str]:
    bin_dir = "Scripts" if os.name == "nt" else "bin"
    if os.name == "nt":
        candidates = ("python.exe", "python")
    else:
        candidates = ("python3", "python")
    return (os.path.join(venv_dir, bin_dir, name) for name in candidates)

def _find_venv_python(venv_dir: str) -> Optional[str]:
    for candidate in _venv_python_candidates(venv_dir):
        if os.path.exists(candidate):
            return candidate
    return None

def create_virtual_environment(venv_dir: str) -> None:
    _safe_log_info(INFO_VENV_CREATING)
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
        _safe_log_info(INFO_VENV_CREATED)
    except subprocess.CalledProcessError as e:
        _safe_log_error(ERROR_VENV_CREATE, e)
        raise

def upgrade_pip(python_executable: str, quiet: bool = True) -> None:
    args = [python_executable, "-m", "pip", "install", "--upgrade", "pip"]
    run_kwargs = {"check": True, "timeout": 180}
    if quiet:
        run_kwargs["stdout"] = subprocess.DEVNULL
        run_kwargs["stderr"] = subprocess.DEVNULL
    try:
        subprocess.run(args, **run_kwargs)
    except (subprocess.CalledProcessError, OSError, subprocess.SubprocessError) as e:
        _safe_log_error(ERROR_PIP_UPGRADE, e)
        raise

def _is_windows_background() -> bool:
    return os.name == "nt" and sys.stdout is None and sys.stderr is None

def _is_background_start() -> bool:
    if os.name == "nt":
        return _is_windows_background()
    try:
        return not sys.stdout.isatty()
    except (AttributeError, ValueError, OSError):
        return True

def _is_unsupervised_background() -> bool:
    if os.environ.get(ENV_AUTOSTART_SUPERVISED):
        return False
    return _is_background_start()

def _should_restart_on_failure() -> bool:
    return _is_unsupervised_background()

def _restart_delay(attempt: int) -> int:
    factor = 2 ** max(attempt - 1, 0)
    return min(START_RESTART_BASE_DELAY * factor, START_RESTART_MAX_DELAY)

def _run_main_once(python_executable: str, entry_script: str) -> None:
    run_kwargs = {}
    child_env = None
    error_log = None
    if _SUPERVISOR_LOG_DIR_SET:
        child_env = dict(os.environ)
        child_env.pop("SPG_LOG_DIR", None)
    if _is_windows_background():
        run_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    if _is_unsupervised_background():
        run_kwargs["stdin"] = subprocess.DEVNULL
        run_kwargs["stdout"] = subprocess.DEVNULL
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "autostart.error.log")
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            error_log = open(log_path, "a", encoding="utf-8")
        except OSError:
            fallback_path = os.path.join(tempfile.gettempdir(), "spotiplaygram-autostart.error.log")
            try:
                error_log = open(fallback_path, "a", encoding="utf-8")
            except OSError:
                error_log = None
        run_kwargs["stderr"] = error_log if error_log is not None else subprocess.DEVNULL
    if child_env is not None:
        run_kwargs["env"] = child_env
    try:
        subprocess.run([python_executable, entry_script], check=True, **run_kwargs)
    except subprocess.CalledProcessError as e:
        _safe_log_error(ERROR_RUN_MAIN, e)
        raise
    finally:
        if error_log is not None:
            try:
                error_log.close()
            except OSError:
                pass

def run_main_script(
    python_executable: str,
    entry_script: str,
    restart_on_failure: bool = False,
) -> None:
    if not restart_on_failure:
        _run_main_once(python_executable, entry_script)
        return

    attempts = 0
    while True:
        started_at = time.monotonic()
        try:
            _run_main_once(python_executable, entry_script)
            return
        except subprocess.CalledProcessError as e:
            if time.monotonic() - started_at >= START_RESTART_STABLE_UPTIME:
                attempts = 0
            if attempts >= START_RESTART_MAX_ATTEMPTS:
                _safe_log_error(ERROR_START_RESTART_LIMIT, START_RESTART_MAX_ATTEMPTS)
                raise
            attempts += 1
            delay = _restart_delay(attempts)
            _safe_log_info(
                INFO_START_RESTART,
                e.returncode,
                delay,
                attempts,
                START_RESTART_MAX_ATTEMPTS,
            )
            time.sleep(delay)

def _select_python_executable(config: StartConfig, base_dir: str) -> str:
    if config.use_system_python or _is_venv_active():
        return sys.executable
    venv_dir = _resolve_path(base_dir, config.venv_dir)
    python_executable = _find_venv_python(venv_dir)
    if python_executable:
        return python_executable
    create_virtual_environment(venv_dir)
    python_executable = _find_venv_python(venv_dir)
    return python_executable or sys.executable

def setup_and_run(
    config: Optional[StartConfig] = None,
    restart_on_failure: bool = False,
) -> None:
    config = config or StartConfig()
    base_dir = os.path.dirname(os.path.abspath(__file__))

    python_executable = _select_python_executable(config, base_dir)
    if config.upgrade_pip and not _is_background_start():
        try:
            upgrade_pip(python_executable, config.quiet_pip)
        except (subprocess.CalledProcessError, OSError, subprocess.SubprocessError):
            pass

    entry_script = _resolve_path(base_dir, config.entry_script)
    run_main_script(python_executable, entry_script, restart_on_failure=restart_on_failure)

def check_dependencies():
    if importlib.util.find_spec("venv") is not None:
        return True
    else:
        _safe_log_error(ERROR_VENV_MODULE_MISSING)
        return False

def main():
    set_console_title(APP_NAME)
    try:
        if not check_dependencies():
            wait_for_enter()
            sys.exit(1)

        lock_path = os.path.join(_START_DIR, DIRS["SESSION"], ".app.lock")
        if not acquire_single_instance_lock(lock_path):
            _safe_log_warning(INFO_SINGLE_INSTANCE)
            wait_for_enter()
            return

        os.environ[ENV_LOCAL_LOCK_HELD] = "1"
        os.environ[ENV_NO_PAUSE] = "1"
        setup_and_run(restart_on_failure=_should_restart_on_failure())

    except KeyboardInterrupt:
        time.sleep(1)
        sys.exit()
    except Exception as e:
        _safe_log_error(ERROR_START_UNEXPECTED, e)
        wait_for_enter()
        raise

if __name__ == "__main__":
    main()
