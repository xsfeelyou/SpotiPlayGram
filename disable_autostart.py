import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
try:
    os.chdir(_ROOT)
except OSError:
    pass
_CORE = os.path.join(_ROOT, "core")
if _CORE not in sys.path:
    sys.path.insert(0, _CORE)

from constants import (
    ERROR_AUTOSTART_DISABLE,
    INFO_AUTOSTART_DISABLE_DONE,
)
from logger import _safe_log_error, _safe_log_info
from utils.autostart import sync_autostart
from utils.runtime_guard import wait_for_enter

def main() -> int:
    ok = sync_autostart(False, base_dir=_ROOT)
    if ok:
        _safe_log_info(INFO_AUTOSTART_DISABLE_DONE)
    else:
        _safe_log_error(ERROR_AUTOSTART_DISABLE)
    if "--no-pause" not in sys.argv[1:]:
        wait_for_enter()
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
