import ctypes
import logging
import logging.handlers
import os
import re
import sys
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional

from constants import ENV_LOG_DIR, ERROR_LOG_SETUP

_DEFAULT_EXCLUDED_DIRS = {
    "venv", ".venv", "env", ".env", "myvenv", "myenv",
    "virtualenv",
    "site-packages", "dist-packages",
}

_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")

_ENABLE_PROCESSED_OUTPUT = 0x0001
_ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
_STD_OUTPUT_HANDLE = 0xFFFFFFF5
_STD_ERROR_HANDLE = 0xFFFFFFF4
_INVALID_HANDLE_VALUE = (1 << (ctypes.sizeof(ctypes.c_void_p) * 8)) - 1

def _set_windows_console_mode(handle_id: int) -> bool:
    if os.name != "nt":
        return False
    try:
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
        kernel32.GetStdHandle.restype = wintypes.HANDLE
        kernel32.GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetConsoleMode.restype = wintypes.BOOL
        kernel32.SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetConsoleMode.restype = wintypes.BOOL
        handle = kernel32.GetStdHandle(handle_id)
        if not handle or handle == _INVALID_HANDLE_VALUE:
            return False
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        new_mode = mode.value | _ENABLE_PROCESSED_OUTPUT | _ENABLE_VIRTUAL_TERMINAL_PROCESSING
        if new_mode == mode.value:
            return True
        return bool(kernel32.SetConsoleMode(handle, new_mode))
    except (AttributeError, ImportError, OSError, TypeError, ValueError):
        return False

def _enable_windows_console_ansi() -> bool:
    if os.name != "nt":
        return False
    stdout_enabled = _set_windows_console_mode(_STD_OUTPUT_HANDLE)
    _set_windows_console_mode(_STD_ERROR_HANDLE)
    return stdout_enabled

def _stream_supports_ansi(stream: Any) -> bool:
    if stream is None:
        return False
    windows_ansi = _enable_windows_console_ansi() if os.name == "nt" else False
    force_color = os.environ.get("FORCE_COLOR", "")
    if force_color and force_color != "0":
        return True
    if "NO_COLOR" in os.environ:
        return False
    try:
        if not stream.isatty():
            return False
    except (AttributeError, ValueError, OSError):
        return False
    if os.name == "nt":
        return windows_ansi
    return os.environ.get("TERM", "").lower() != "dumb"

def _stream_supports_char(stream, char: str) -> bool:
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return True
    try:
        char.encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True

def _is_project_python_file(
    path: str,
    root_dir: str,
    excluded_dirs: Optional[Iterable[str]] = None,
    excluded_files: Optional[Iterable[str]] = None,
) -> bool:
    try:
        if not path or not root_dir:
            return False
        path_abs = os.path.normcase(os.path.abspath(path))
        root_abs = os.path.normcase(os.path.abspath(root_dir))
        if not path_abs.lower().endswith(".py"):
            return False
        try:
            if os.path.commonpath([path_abs, root_abs]) != root_abs:
                return False
        except ValueError:
            return False
        rel = os.path.relpath(path_abs, root_abs)
        if rel.startswith(os.pardir):
            return False
        segments = [seg.lower() for seg in rel.split(os.sep) if seg]
        if excluded_dirs is None:
            excluded_dirs = _DEFAULT_EXCLUDED_DIRS
        excluded_dirs_set = {seg.lower() for seg in excluded_dirs}
        if excluded_dirs_set and any(seg in excluded_dirs_set for seg in segments):
            return False
        if excluded_files:
            excluded_files_set = {os.path.basename(item).lower() for item in excluded_files}
            if os.path.basename(path_abs).lower() in excluded_files_set:
                return False
        return True
    except Exception:
        return False

class ProjectAwareFormatter(logging.Formatter):
    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
        root_dir: Optional[str] = None,
        excluded_dirs: Optional[Iterable[str]] = None,
        excluded_files: Optional[Iterable[str]] = None,
    ):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.root_dir = os.path.normcase(os.path.abspath(root_dir or os.getcwd()))
        self.excluded_dirs = {seg.lower() for seg in (excluded_dirs or _DEFAULT_EXCLUDED_DIRS)}
        if excluded_files is None:
            excluded_files = [__file__]
        self.excluded_files = {os.path.basename(item).lower() for item in (excluded_files or [])}

    def formatException(self, ei):
        if not ei:
            return ""
        exc_type, exc_value, exc_tb = ei
        if exc_tb is None:
            return "".join(traceback.format_exception_only(exc_type, exc_value))
        extracted = traceback.extract_tb(exc_tb)
        filtered = [
            frame for frame in extracted
            if _is_project_python_file(
                frame.filename,
                self.root_dir,
                self.excluded_dirs,
                self.excluded_files,
            )
        ]
        if filtered:
            tb_str = "Traceback (most recent call last):\n" + "".join(traceback.format_list(filtered))
            exc_only = "".join(traceback.format_exception_only(exc_type, exc_value))
            return tb_str + exc_only
        return "".join(traceback.format_exception_only(exc_type, exc_value))

class LevelEmojiFilter(logging.Filter):
    def __init__(self, level_emojis: Dict[int, str], use_emojis: bool) -> None:
        super().__init__()
        self.level_emojis = level_emojis
        self.use_emojis = use_emojis

    def filter(self, record: logging.LogRecord) -> bool:
        if self.use_emojis:
            record.level_emoji = self.level_emojis.get(record.levelno, record.levelname)
        else:
            record.level_emoji = record.levelname
        return True

class AnsiColorFormatter(logging.Formatter):
    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        level_colors: Optional[Dict[int, str]] = None,
        reset_code: str = "\033[0m",
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.level_colors = level_colors or {}
        self.reset_code = reset_code

    def format(self, record: logging.LogRecord) -> str:
        record_dict = record.__dict__.copy()
        level_emoji = record_dict.get("level_emoji", record.levelname)
        color = self.level_colors.get(record.levelno, "")
        if color:
            record_dict["level_emoji"] = f"{color}{level_emoji}{self.reset_code}"
        else:
            record_dict["level_emoji"] = level_emoji
        record_copy = logging.makeLogRecord(record_dict)
        return super().format(record_copy)

class AnsiStrippingFormatter(logging.Formatter):
    def __init__(self, base_formatter: logging.Formatter) -> None:
        super().__init__()
        self.base_formatter = base_formatter

    def format(self, record: logging.LogRecord) -> str:
        formatted = self.base_formatter.format(record)
        return _ANSI_ESCAPE_RE.sub("", formatted)

class TracebackOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return bool(record.exc_info or record.exc_text or record.stack_info)

class DevTracebackFormatter(logging.Formatter):
    def __init__(
        self,
        base_formatter: ProjectAwareFormatter,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.base_formatter = base_formatter

    def format(self, record: logging.LogRecord) -> str:
        exc_text = ""
        if record.exc_info:
            exc_text = self.base_formatter.formatException(record.exc_info)
        elif record.exc_text:
            exc_text = record.exc_text
        elif record.stack_info:
            exc_text = record.stack_info
        if not exc_text:
            exc_text = record.getMessage()
        record_dict = record.__dict__.copy()
        record_dict["msg"] = exc_text
        record_dict["args"] = ()
        record_dict["exc_info"] = None
        record_dict["exc_text"] = None
        record_dict["stack_info"] = None
        record_copy = logging.makeLogRecord(record_dict)
        return super().format(record_copy)

class LoggerConfig:
    def __init__(
        self,
        log_dir: Optional[str] = None,
        root_dir: Optional[str] = None,
        excluded_dirs: Optional[Iterable[str]] = None,
        excluded_files: Optional[Iterable[str]] = None,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        encoding: str = "utf-8",
        date_format: str = "%d.%m.%Y %H:%M:%S",
        console_level: int = logging.INFO,
        user_file_level: int = logging.INFO,
        dev_file_level: int = logging.ERROR,
        user_logger_name: str = "user_logger",
        dev_logger_name: str = "dev_logger",
        use_emojis: bool = True,
        use_color: Optional[bool] = None,
        install_excepthook: bool = True,
        duplicate_errors: bool = True,
    ):
        self.root_dir = os.path.normcase(os.path.abspath(root_dir or os.getcwd()))
        env_log_dir = os.environ.get(ENV_LOG_DIR)
        if env_log_dir:
            log_dir = env_log_dir
        if log_dir is None:
            log_dir = os.path.join(self.root_dir, "logs")
        self.log_dir = os.path.normcase(os.path.abspath(log_dir)) if log_dir else None
        self.file_logging_available = True
        self.excluded_dirs = {seg.lower() for seg in (excluded_dirs or _DEFAULT_EXCLUDED_DIRS)}
        if excluded_files is None:
            excluded_files = [__file__]
        self.excluded_files = {os.path.basename(item).lower() for item in (excluded_files or [])}

        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.encoding = encoding
        self.date_format = date_format
        self.console_level = console_level
        self.user_file_level = user_file_level
        self.dev_file_level = dev_file_level
        self.user_logger_name = user_logger_name
        self.dev_logger_name = dev_logger_name
        self.use_emojis = use_emojis
        self.use_color = None if use_color is None else bool(use_color)
        self.install_excepthook = install_excepthook
        self.duplicate_errors = duplicate_errors

        self.level_emojis = {
            logging.DEBUG: "⚙️",
            logging.INFO: "✅",
            logging.WARNING: "⚠️",
            logging.ERROR: "❌",
            logging.CRITICAL: "⛔️",
        }

        self.level_colors = {
            logging.DEBUG: "\033[36m",
            logging.INFO: "\033[32m",
            logging.WARNING: "\033[33m",
            logging.ERROR: "\033[31m",
            logging.CRITICAL: "\033[1;31m",
        }

        self.console_format_color = "%(asctime)s.%(msecs)03d | %(level_emoji)s | %(message)s"
        self.console_format_plain = "%(asctime)s.%(msecs)03d | %(level_emoji)s | %(message)s"
        self.user_file_format = "%(asctime)s.%(msecs)03d | %(level_emoji)s | %(message)s"
        self.dev_file_format = "%(asctime)s.%(msecs)03d | %(level_emoji)s | %(message)s"

_LOGGER_HANDLER_TAG = "logger_py_handler"
_SKIP_DUPLICATION_ATTR = "_skip_error_duplication"

class ErrorDuplicatorHandler(logging.Handler):
    def __init__(self, dev_logger: logging.Logger) -> None:
        super().__init__(level=logging.ERROR)
        self.dev_logger = dev_logger

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.ERROR or self.dev_logger is None:
            return
        if getattr(record, _SKIP_DUPLICATION_ATTR, False):
            return
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)
        source = ""
        if record.pathname:
            source = os.path.basename(record.pathname)
            if record.lineno:
                source = f"{source}:{record.lineno}"
        if record.funcName:
            source = f"{source} in {record.funcName}()" if source else f"{record.funcName}()"
        if source:
            message = f"{message} | Source: {source}"
        dev_record = self.dev_logger.makeRecord(
            name=self.dev_logger.name,
            level=record.levelno,
            fn=record.pathname or "<unknown>",
            lno=record.lineno or 0,
            msg=message,
            args=(),
            exc_info=record.exc_info,
            func=record.funcName,
            extra=None,
        )
        self.dev_logger.handle(dev_record)

class LoggerManager:
    def __init__(self, config: Optional[LoggerConfig] = None):
        self.config = config or LoggerConfig()
        self._setup_logging_environment()
        self.user_logger = self._create_user_logger()
        self.dev_logger = self._create_dev_logger()
        self._setup_error_duplication()
        self._setup_exception_handling()

    def _setup_logging_environment(self) -> None:
        if not self.config.log_dir:
            return
        try:
            os.makedirs(self.config.log_dir, exist_ok=True)
            if os.name != "nt":
                os.chmod(self.config.log_dir, 0o700)
        except OSError as e:
            self.config.file_logging_available = False
            try:
                if sys.__stderr__ is not None:
                    sys.__stderr__.write(ERROR_LOG_SETUP.format(e) + "\n")
            except (AttributeError, OSError, ValueError):
                pass

    def _reset_logger(self, logger: logging.Logger) -> None:
        for handler in list(logger.handlers):
            if getattr(handler, _LOGGER_HANDLER_TAG, False):
                logger.removeHandler(handler)
                handler.close()

    def _create_user_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.config.user_logger_name)
        self._reset_logger(logger)
        logger.propagate = False

        handlers: List[logging.Handler] = []
        console_handler = self._create_console_handler(self.config.console_level)
        if console_handler:
            handlers.append(console_handler)

        file_handler = self._create_file_handler("user.log", self.config.user_file_level)
        if file_handler:
            base_formatter = logging.Formatter(
                self.config.user_file_format,
                datefmt=self.config.date_format,
            )
            base_formatter.converter = time.localtime
            file_handler.setFormatter(AnsiStrippingFormatter(base_formatter))
            file_handler.addFilter(LevelEmojiFilter(self.config.level_emojis, self.config.use_emojis))
            handlers.append(file_handler)

        for handler in handlers:
            setattr(handler, _LOGGER_HANDLER_TAG, True)
            logger.addHandler(handler)

        logger.setLevel(min((handler.level for handler in handlers), default=logging.INFO))
        return logger

    def _create_dev_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.config.dev_logger_name)
        self._reset_logger(logger)
        logger.propagate = False

        file_handler = self._create_file_handler("dev.log", self.config.dev_file_level)
        if file_handler:
            base_formatter = ProjectAwareFormatter(
                fmt=self.config.dev_file_format,
                datefmt=self.config.date_format,
                root_dir=self.config.root_dir,
                excluded_dirs=self.config.excluded_dirs,
                excluded_files=self.config.excluded_files,
            )
            base_formatter.converter = time.localtime
            dev_formatter = DevTracebackFormatter(
                base_formatter=base_formatter,
                fmt=self.config.dev_file_format,
                datefmt=self.config.date_format,
            )
            dev_formatter.converter = time.localtime
            file_handler.setFormatter(AnsiStrippingFormatter(dev_formatter))
            file_handler.addFilter(LevelEmojiFilter(self.config.level_emojis, self.config.use_emojis))
            file_handler.addFilter(TracebackOnlyFilter())
            setattr(file_handler, _LOGGER_HANDLER_TAG, True)
            logger.addHandler(file_handler)

        logger.setLevel(self.config.dev_file_level)
        return logger

    def _create_console_handler(self, level: int) -> Optional[logging.StreamHandler]:
        if sys.stdout is None:
            return None
        try:
            sys.stdout.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            pass
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)

        use_color = False
        if self.config.use_color is not False:
            use_color = _stream_supports_ansi(sys.stdout)
        if use_color:
            formatter = AnsiColorFormatter(
                self.config.console_format_color,
                datefmt=self.config.date_format,
                level_colors=self.config.level_colors,
            )
        else:
            formatter = AnsiStrippingFormatter(
                logging.Formatter(
                    self.config.console_format_plain,
                    datefmt=self.config.date_format,
                )
            )
        formatter.converter = time.localtime
        handler.setFormatter(formatter)
        use_emojis = self.config.use_emojis and _stream_supports_char(sys.stdout, "✅")
        handler.addFilter(LevelEmojiFilter(self.config.level_emojis, use_emojis))

        return handler

    def _create_file_handler(self, filename: str, level: int) -> Optional[logging.handlers.RotatingFileHandler]:
        if not self.config.log_dir or not self.config.file_logging_available:
            return None
        path = os.path.join(self.config.log_dir, filename)
        try:
            handler = logging.handlers.RotatingFileHandler(
                path,
                mode="a",
                maxBytes=self.config.max_bytes,
                backupCount=self.config.backup_count,
                encoding=self.config.encoding,
            )
        except (OSError, ValueError) as e:
            try:
                if sys.__stderr__ is not None:
                    sys.__stderr__.write(ERROR_LOG_SETUP.format(e) + "\n")
            except (AttributeError, OSError, ValueError):
                pass
            return None
        if os.name != "nt":
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
        handler.setLevel(level)

        return handler

    def _setup_error_duplication(self) -> None:
        if not self.config.duplicate_errors:
            return
        error_duplicator = ErrorDuplicatorHandler(self.dev_logger)
        setattr(error_duplicator, _LOGGER_HANDLER_TAG, True)
        self.user_logger.addHandler(error_duplicator)

    def _setup_exception_handling(self) -> None:
        if not self.config.install_excepthook:
            return
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return

            _safe_log_error(
                "Uncaught exception:",
                logger=self.dev_logger,
                exc_info=(exc_type, exc_value, exc_traceback)
            )

        sys.excepthook = handle_exception

_logger_manager: Optional["LoggerManager"] = None

def configure_logging(config: Optional[LoggerConfig] = None) -> LoggerManager:
    global _logger_manager, user_logger, dev_logger
    _logger_manager = LoggerManager(config)
    user_logger = _logger_manager.user_logger
    dev_logger = _logger_manager.dev_logger
    return _logger_manager

def get_logger_manager() -> LoggerManager:
    global _logger_manager, user_logger, dev_logger
    if _logger_manager is None:
        _logger_manager = LoggerManager()
        user_logger = _logger_manager.user_logger
        dev_logger = _logger_manager.dev_logger
    return _logger_manager

def _normalize_log_arg(arg: Any) -> Any:
    if isinstance(arg, BaseException):
        text = str(arg).strip()
        if text:
            return text
        return arg.__class__.__name__
    return arg

def _safe_format_message(template: Any, args: tuple[Any, ...]) -> Optional[str]:
    try:
        message = str(template)
    except Exception:
        return None
    if args:
        normalized_args = tuple(_normalize_log_arg(arg) for arg in args)
        try:
            message = message.format(*normalized_args)
        except (KeyError, IndexError, ValueError, TypeError):
            try:
                safe_args = ", ".join(str(arg) for arg in normalized_args)
            except Exception:
                safe_args = ""
            if safe_args:
                message = f"{message} {safe_args}"
    return message

def _safe_log(
    logger: Optional[logging.Logger],
    level: str,
    template: Any,
    args: tuple[Any, ...],
    **kwargs,
) -> None:
    message = _safe_format_message(template, args)
    if not message:
        return
    try:
        manager = get_logger_manager()
        target_loggers = [logger] if logger is not None else [manager.user_logger, manager.dev_logger]
        stacklevel = kwargs.pop("stacklevel", 1)
        stacklevel = stacklevel + 2
        for target_logger in target_loggers:
            if target_logger is None:
                continue
            log_fn = getattr(target_logger, level, None)
            if not callable(log_fn):
                continue
            call_kwargs = dict(kwargs)
            call_kwargs["stacklevel"] = stacklevel
            if (
                logger is None
                and target_logger is manager.user_logger
                and manager.config.duplicate_errors
                and level in ("error", "exception", "critical")
            ):
                extra = call_kwargs.get("extra")
                if extra is None:
                    call_kwargs["extra"] = {_SKIP_DUPLICATION_ATTR: True}
                elif isinstance(extra, dict) and _SKIP_DUPLICATION_ATTR not in extra:
                    extra = dict(extra)
                    extra[_SKIP_DUPLICATION_ATTR] = True
                    call_kwargs["extra"] = extra
            log_fn(message, **call_kwargs)
    except Exception as e:
        try:
            if sys.__stderr__ is not None:
                sys.__stderr__.write(f"Logging failure: {e}; message: {message}\n")
        except (AttributeError, OSError, ValueError):
            pass

def _safe_log_info(template: Any, *args, logger: Optional[logging.Logger] = None, **kwargs) -> None:
    kwargs.setdefault("stacklevel", 2)
    _safe_log(logger, "info", template, args, **kwargs)

def _safe_log_warning(template: Any, *args, logger: Optional[logging.Logger] = None, **kwargs) -> None:
    kwargs.setdefault("stacklevel", 2)
    _safe_log(logger, "warning", template, args, **kwargs)

def _safe_log_error(template: Any, *args, logger: Optional[logging.Logger] = None, **kwargs) -> None:
    kwargs.setdefault("stacklevel", 2)
    if "exc_info" not in kwargs:
        for arg in args:
            if isinstance(arg, BaseException):
                kwargs["exc_info"] = arg
                break
    _safe_log(logger, "error", template, args, **kwargs)

_logger_manager = configure_logging()
user_logger = _logger_manager.user_logger
dev_logger = _logger_manager.dev_logger
