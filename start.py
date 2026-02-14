import os
import subprocess
import sys
import time
from typing import Iterable, Optional

from core.logger import _safe_log_error, _safe_log_info

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
        required_dirs: Optional[Iterable[str]] = None,
        upgrade_pip: bool = True,
        use_system_python: bool = False,
        quiet_pip: bool = True,
    ):
        self.venv_dir = venv_dir
        self.entry_script = entry_script
        self.required_dirs = list(required_dirs) if required_dirs is not None else ["session"]
        self.upgrade_pip = upgrade_pip
        self.use_system_python = use_system_python
        self.quiet_pip = quiet_pip

def _get_base_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))

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

def create_required_directories(config: StartConfig, base_dir: str) -> None:
    for dir_path in config.required_dirs:
        full_path = _resolve_path(base_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)

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
    run_kwargs = {"check": True}
    if quiet:
        run_kwargs["stdout"] = subprocess.DEVNULL
        run_kwargs["stderr"] = subprocess.DEVNULL
    try:
        subprocess.run(args, **run_kwargs)
    except subprocess.CalledProcessError as e:
        _safe_log_error(ERROR_PIP_UPGRADE, e)
        raise

def run_main_script(python_executable: str, entry_script: str) -> None:
    try:
        subprocess.run([python_executable, entry_script], check=True)
    except subprocess.CalledProcessError as e:
        _safe_log_error(ERROR_RUN_MAIN, e)
        raise

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

def setup_and_run(config: Optional[StartConfig] = None) -> None:
    config = config or StartConfig()
    base_dir = _get_base_dir()

    create_required_directories(config, base_dir)

    python_executable = _select_python_executable(config, base_dir)
    if config.upgrade_pip:
        upgrade_pip(python_executable, config.quiet_pip)

    entry_script = _resolve_path(base_dir, config.entry_script)
    run_main_script(python_executable, entry_script)

def check_dependencies():
    try:
        import venv  
        return True
    except Exception:
        _safe_log_error(ERROR_VENV_MODULE_MISSING)
        return False

def main():
    try:
        if not check_dependencies():
            sys.exit(1)

        setup_and_run()

    except KeyboardInterrupt:
        time.sleep(1)
        sys.exit()
    except Exception as e:
        _safe_log_error(ERROR_START_UNEXPECTED, e)
        input()
        raise

if __name__ == "__main__":
    main()
