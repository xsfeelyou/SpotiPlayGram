import os
import sys
import asyncio
from subprocess import call

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_CHDIR_ERROR = None
try:
    os.chdir(_PROJECT_ROOT)
except OSError as e:
    _CHDIR_ERROR = e

from logger import _safe_log_info, _safe_log_error, _safe_log_warning, dev_logger

def install_libs():
    libs = [
        ("dotenv", "python-dotenv"),
        ("aiohttp", "aiohttp"),
        ("spotipy", "spotipy"),
        ("telethon", "telethon"),
        ("mistralai", "mistralai"),
        ("anyascii", "anyascii"),
    ]
    for pkg, inst in libs:
        try: __import__(pkg)
        except ImportError:
            _safe_log_info(f"Installing library: {pkg} ({inst})")
            call([sys.executable, "-m", "pip", "install", inst])

install_libs()

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from telethon import TelegramClient
from telethon.errors import RPCError
from spotipy import SpotifyOAuth
from spotipy.exceptions import SpotifyOauthError

import logging
logging.getLogger("telethon").setLevel(logging.ERROR)
from constants import (
    APP_NAME,
    DIRS,
    ERROR_SESSION_DIR,
    WARNING_CHDIR_FAILED,
    ERROR_AUTH_GENIUS,
    ERROR_AUTH_GENIUS_INVALID_TOKEN,
    ERROR_AUTH_MISTRAL,
    ERROR_AUTH_MISTRAL_API_KEY_MISSING,
    ERROR_AUTH_SPOTIFY,
    ERROR_AUTH_TG,
    ENV_LOCAL_LOCK_HELD,
    ENV_NO_PAUSE,
    ERROR_DEVICE_LOCK,
    ERROR_GENIUS_ACCESS_TOKEN_INVALID,
    ERROR_SPOTIFY_AUTH_INVALID,
    ERROR_TG_AUTH_INVALID,
    ERROR_UPDATE_LOOP,
    ERROR_UNEXPECTED,
    INFO_APP_SHUTDOWN,
    INFO_AUTH_GENIUS,
    INFO_AUTH_GENIUS_SUCCESS,
    INFO_AUTH_MISTRAL,
    INFO_AUTH_MISTRAL_SUCCESS,
    INFO_AUTH_RETRY,
    INFO_AUTH_SPOTIFY,
    INFO_AUTH_SPOTIFY_SUCCESS,
    INFO_AUTH_TG_BOT,
    INFO_AUTH_TG_BOT_SUCCESS,
    INFO_AUTH_TG_USER,
    INFO_AUTH_TG_USER_SUCCESS,
    INFO_ENTERING_MAIN_LOOP,
    INFO_GENIUS_DISABLED,
    INFO_MISTRAL_DISABLED,
    INFO_SINGLE_INSTANCE,
    INFO_SINGLE_INSTANCE_REMOTE,
    INFO_STARTUP_COMPLETE,
    LOG_FORMAT_GENIUS,
    LOG_FORMAT_MISTRAL,
    LOG_FORMAT_SPOTIFY,
    LOG_FORMAT_TELEGRAM,
)
from config import Settings, EnvValidationError

if _CHDIR_ERROR is not None:
    _safe_log_warning(WARNING_CHDIR_FAILED, _CHDIR_ERROR)

from utils.http_session import init_session, close_session
from utils.runtime_guard import (
    DeviceLockBusy,
    acquire_device_lock,
    acquire_single_instance_lock,
    set_console_title,
    wait_for_enter,
)
from utils.mistral_client import close_mistral_client, prefetch_mistral_client
from utils.spotify_client import AsyncSpotifyClient, get_spg_cache_handler
from utils.update_libs import schedule_daily_update_check
from aiohttp import ClientTimeout, ClientError

_BACKGROUND_TASKS: set[asyncio.Task] = set()

def _spawn_background_task(coro) -> None:
    task = asyncio.create_task(coro)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)

def _abs_under_root(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_PROJECT_ROOT, p)

def _secure_dir(path: str) -> None:
    if os.name == "nt":
        return
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass

def _secure_file(path: str) -> None:
    if os.name == "nt":
        return
    try:
        if os.path.exists(path):
            os.chmod(path, 0o600)
    except OSError:
        pass

def _ensure_runtime_dirs() -> None:
    session_failed = False
    for name, d in DIRS.items():
        path = _abs_under_root(d)
        try:
            os.makedirs(path, exist_ok=True)
            _secure_dir(path)
        except OSError as e:
            _safe_log_error(ERROR_SESSION_DIR, e)
            if name == "SESSION":
                session_failed = True
    session_dir_raw = DIRS.get("SESSION")
    session_dir = str(session_dir_raw).strip() if session_dir_raw else "session"
    session_dir = _abs_under_root(session_dir)
    for name in ("spotify", "genius", "mistral", "telegram"):
        path = os.path.join(session_dir, name)
        try:
            os.makedirs(path, exist_ok=True)
            _secure_dir(path)
        except OSError as e:
            _safe_log_error(ERROR_SESSION_DIR, e)
            session_failed = True
    _secure_file(os.path.join(_PROJECT_ROOT, ".env"))
    if session_failed:
        raise RuntimeError(ERROR_SESSION_DIR.format(session_dir))

def _is_telegram_auth_invalid(err: Exception) -> bool:
    try:
        low = str(err).lower()
    except (ValueError, TypeError, UnicodeError):
        return False
    needles = (
        "api_id_invalid",
        "api hash",
        "api_hash",
        "bot token",
        "token invalid",
        "unauthorized",
        "forbidden",
        "auth key",
    )
    return any(n in low for n in needles)

def _is_spotify_auth_invalid(err: Exception) -> bool:
    try:
        low = str(err).lower()
    except (ValueError, TypeError, UnicodeError):
        return False
    needles = (
        "invalid_client",
        "invalid client",
        "invalid_grant",
        "unauthorized",
        "forbidden",
    )
    return any(n in low for n in needles)

def _get_auth_retry_settings(settings: Settings) -> tuple[int, float]:
    retries = settings.auth_max_retries
    if retries < 0:
        retries = 0
    max_attempts = 1 + retries
    delay_base = settings.auth_retry_delay_base
    if delay_base < 0:
        delay_base = 0.0
    return max_attempts, delay_base

async def _sleep_auth_retry(prefix: str, delay_base: float, attempt: int, max_attempts: int) -> None:
    if attempt >= max_attempts:
        return
    delay = delay_base * attempt
    _safe_log_info(INFO_AUTH_RETRY, prefix, delay, attempt + 1, max_attempts)
    if delay > 0:
        await asyncio.sleep(delay)

async def _start_telegram_client(
    settings: Settings,
    *,
    session_name: str,
    start_log: str,
    success_log: str,
    bot_token=None,
) -> TelegramClient:
    _safe_log_info(start_log)
    client = TelegramClient(
        os.path.join(_abs_under_root(DIRS["SESSION"]), "telegram", session_name),
        settings.tg_api_id,
        settings.tg_api_hash,
        connection_retries=max(0, min(settings.auth_max_retries, 10)),
        retry_delay=max(0.5, min(settings.auth_retry_delay_base, 30.0)),
        auto_reconnect=True,
    )
    max_attempts, delay_base = _get_auth_retry_settings(settings)
    for attempt in range(1, max_attempts + 1):
        try:
            if bot_token is not None:
                await client.start(bot_token=bot_token)
            else:
                await client.start()
            _safe_log_info(success_log)
            return client
        except (RPCError, ConnectionError, asyncio.TimeoutError, OSError, ValueError, TypeError, RuntimeError, EOFError) as e:
            if isinstance(e, (ConnectionError, asyncio.TimeoutError, OSError)) and attempt < max_attempts:
                await _sleep_auth_retry(LOG_FORMAT_TELEGRAM, delay_base, attempt, max_attempts)
                continue
            if _is_telegram_auth_invalid(e):
                _safe_log_error(ERROR_TG_AUTH_INVALID)
            else:
                _safe_log_error(ERROR_AUTH_TG, e)
            raise

async def _start_spotify(settings: Settings) -> AsyncSpotifyClient:
    _safe_log_info(INFO_AUTH_SPOTIFY)
    cache_path = os.path.join(_abs_under_root(DIRS["SESSION"]), "spotify", ".cache")
    cache_handler = get_spg_cache_handler(cache_path)
    auth_manager = SpotifyOAuth(
        client_id=settings.spotify_client_id,
        client_secret=settings.spotify_client_secret,
        redirect_uri=settings.spotify_redirect_uri,
        scope="user-read-currently-playing user-read-playback-state",
        cache_handler=cache_handler,
    )
    spotify = AsyncSpotifyClient(
        auth_manager=auth_manager,
        session=init_session(),
        request_timeout=settings.spotify_request_timeout,
        language=settings.spotify_language,
    )
    max_attempts, delay_base = _get_auth_retry_settings(settings)
    for attempt in range(1, max_attempts + 1):
        try:
            token = await spotify._get_access_token()
            if not token:
                _safe_log_error(ERROR_SPOTIFY_AUTH_INVALID)
                raise ValueError("empty spotify access token")
            _safe_log_info(INFO_AUTH_SPOTIFY_SUCCESS)
            return spotify
        except (ClientError, ConnectionError, asyncio.TimeoutError, OSError, ValueError, TypeError, RuntimeError, EOFError, SpotifyOauthError) as e:
            if _is_spotify_auth_invalid(e):
                _safe_log_error(ERROR_SPOTIFY_AUTH_INVALID)
                raise
            if isinstance(e, (ClientError, ConnectionError, asyncio.TimeoutError, OSError)) and attempt < max_attempts:
                await _sleep_auth_retry(LOG_FORMAT_SPOTIFY, delay_base, attempt, max_attempts)
                continue
            _safe_log_error(ERROR_AUTH_SPOTIFY, e)
            raise

async def _check_genius_and_mistral(settings: Settings) -> bool:
    max_attempts, delay_base = _get_auth_retry_settings(settings)
    if not settings.enable_genius:
        _safe_log_info(INFO_GENIUS_DISABLED)
    else:
        token = settings.genius_access_token
        if not token:
            _safe_log_error(ERROR_AUTH_GENIUS_INVALID_TOKEN)
        else:
            _safe_log_info(INFO_AUTH_GENIUS)
            genius_ok = False
            genius_error_logged = False
            session = init_session()
            headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
            timeout = ClientTimeout(total=6.0)
            for attempt in range(1, max_attempts + 1):
                try:
                    async with session.get(
                        "https://api.genius.com/search",
                        params={"q": "ping"},
                        headers=headers,
                        timeout=timeout,
                    ) as r:
                        if 200 <= r.status < 300:
                            genius_ok = True
                            break
                        _ = await r.text()
                        if r.status in (401, 403):
                            _safe_log_error(ERROR_GENIUS_ACCESS_TOKEN_INVALID)
                            genius_error_logged = True
                            break
                        if (r.status == 429 or 500 <= r.status < 600) and attempt < max_attempts:
                            await _sleep_auth_retry(LOG_FORMAT_GENIUS, delay_base, attempt, max_attempts)
                            continue
                        _safe_log_error(ERROR_AUTH_GENIUS)
                        genius_error_logged = True
                        break
                except (ClientError, ConnectionError, asyncio.TimeoutError, OSError, RuntimeError):
                    _safe_log_error(ERROR_AUTH_GENIUS, logger=dev_logger, exc_info=True)
                    if attempt < max_attempts:
                        await _sleep_auth_retry(LOG_FORMAT_GENIUS, delay_base, attempt, max_attempts)
                        continue
                    break

            if genius_ok:
                _safe_log_info(INFO_AUTH_GENIUS_SUCCESS)
            elif not genius_error_logged:
                _safe_log_error(ERROR_AUTH_GENIUS)

    if not settings.enable_mistral:
        _safe_log_info(INFO_MISTRAL_DISABLED)
        return False

    api_key = settings.mistral_api_key
    if not api_key:
        _safe_log_error(ERROR_AUTH_MISTRAL_API_KEY_MISSING)
        return False

    _safe_log_info(INFO_AUTH_MISTRAL)
    m_session = init_session()
    m_headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    m_timeout = ClientTimeout(total=6.0)
    for attempt in range(1, max_attempts + 1):
        try:
            async with m_session.get(
                "https://api.mistral.ai/v1/models",
                headers=m_headers,
                timeout=m_timeout,
            ) as mr:
                if 200 <= mr.status < 300:
                    _safe_log_info(INFO_AUTH_MISTRAL_SUCCESS)
                    return True
                _ = await mr.text()
                if mr.status in (401, 403):
                    _safe_log_error(ERROR_AUTH_MISTRAL)
                    return False
                if (mr.status == 429 or 500 <= mr.status < 600) and attempt < max_attempts:
                    await _sleep_auth_retry(LOG_FORMAT_MISTRAL, delay_base, attempt, max_attempts)
                    continue
                _safe_log_error(ERROR_AUTH_MISTRAL)
                return False
        except (ClientError, ConnectionError, asyncio.TimeoutError, OSError, RuntimeError):
            if attempt < max_attempts:
                await _sleep_auth_retry(LOG_FORMAT_MISTRAL, delay_base, attempt, max_attempts)
                continue
            _safe_log_error(ERROR_AUTH_MISTRAL)
            return False
    return False

async def _prefetch_runtime_state(
    settings: Settings,
    *,
    genius_search_module,
    mistral_ai_module,
    mistral_valid: bool,
) -> None:
    runtime_tasks: list[asyncio.Task] = []
    if settings.enable_genius and settings.genius_access_token:
        try:
            prefetch_genius_runtime = getattr(genius_search_module, "prefetch_search_runtime", None)
            if callable(prefetch_genius_runtime):
                runtime_tasks.append(asyncio.create_task(prefetch_genius_runtime()))
        except (AttributeError, TypeError, RuntimeError, ValueError, OSError) as e:
            _safe_log_warning(ERROR_UNEXPECTED, e)

    if settings.enable_mistral and settings.mistral_api_key:
        try:
            read_mistral_model = getattr(mistral_ai_module, "read_mistral_model", None)
            if callable(read_mistral_model):
                read_mistral_model()
        except (AttributeError, TypeError, RuntimeError, ValueError, OSError) as e:
            _safe_log_warning(ERROR_UNEXPECTED, e)

        if mistral_valid:
            try:
                runtime_tasks.append(asyncio.create_task(prefetch_mistral_client(settings.mistral_api_key)))
            except (AttributeError, TypeError, RuntimeError, ValueError, OSError) as e:
                _safe_log_warning(ERROR_UNEXPECTED, e)
            try:
                from lyrics.mistral_models import prefetch_mistral_models_menu
                _spawn_background_task(prefetch_mistral_models_menu(settings.mistral_api_key))
            except (AttributeError, ImportError, TypeError, RuntimeError, ValueError, OSError) as e:
                _safe_log_warning(ERROR_UNEXPECTED, e)

    if runtime_tasks:
        results = await asyncio.gather(*runtime_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                _safe_log_warning(ERROR_UNEXPECTED, result)

async def main():
    set_console_title(APP_NAME)
    _ensure_runtime_dirs()

    if not os.environ.get(ENV_LOCAL_LOCK_HELD):
        lock_path = os.path.join(_abs_under_root(DIRS["SESSION"]), ".app.lock")
        if not acquire_single_instance_lock(lock_path):
            _safe_log_warning(INFO_SINGLE_INSTANCE)
            await asyncio.to_thread(wait_for_enter)
            return

    try:
        settings = Settings.from_env()
    except EnvValidationError as e:
        for msg in getattr(e, "messages", []) or []:
            try:
                _safe_log_error(msg)
            except (ValueError, TypeError, OSError):
                pass
        raise

    init_session()

    try:
        loop = asyncio.get_running_loop()

        def _loop_exception_handler(loop, context):
            try:
                msg = context.get("message") or ""
            except (AttributeError, TypeError, ValueError):
                msg = ""
            exc = context.get("exception")
            try:
                exc_text = str(exc) if exc is not None else ""
            except (ValueError, TypeError, UnicodeError):
                exc_text = ""
            try:
                low = (str(msg) + " " + exc_text).lower()
            except (ValueError, TypeError, UnicodeError):
                low = ""

            if "future exception was never retrieved" in low:
                if (
                    isinstance(exc, (ConnectionError, OSError, asyncio.TimeoutError))
                    or "connection to telegram failed" in low
                    or "automatic reconnection failed" in low
                ):
                    return

            loop.default_exception_handler(context)

        loop.set_exception_handler(_loop_exception_handler)
    except (RuntimeError, ValueError, TypeError, OSError):
        pass

    from status import bot as bot_module
    from status import spotify_status as spotify_status_module
    from lyrics import genius_search as genius_search_module
    from lyrics import mistral_ai as mistral_ai_module

    bot_module.set_settings(settings)
    spotify_status_module.set_settings(settings)
    genius_search_module.set_settings(settings)
    mistral_ai_module.set_settings(settings)

    lease = None
    lease_task = None
    user = None
    bot = None
    spotify = None
    update_task = None
    schedule_task = None
    cancelled_exc = None
    lease_clients = {}

    def _mark_shutting_down() -> None:
        try:
            marker = getattr(bot_module, "mark_shutting_down", None)
            if callable(marker):
                marker()
        except (AttributeError, TypeError, RuntimeError, ValueError, OSError):
            pass

    async def _on_lease_lost():
        _mark_shutting_down()
        for client in list(lease_clients.values()):
            if client is None:
                continue
            try:
                await client.disconnect()
            except (RPCError, ConnectionError, asyncio.TimeoutError, OSError, RuntimeError):
                pass

    async def _run_update_loop():
        delay = 1.0
        while True:
            try:
                await bot_module.patched_update_channel(user, bot, spotify)
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                _safe_log_error(ERROR_UPDATE_LOOP, e)
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, 60.0)

    try:
        lease = await acquire_device_lock(settings.tg_bot_token, _PROJECT_ROOT)
    except DeviceLockBusy as e:
        _safe_log_warning(INFO_SINGLE_INSTANCE_REMOTE, e.host)
        await close_session()
        await asyncio.to_thread(wait_for_enter)
        return
    except (ClientError, asyncio.TimeoutError, OSError, RuntimeError, ValueError, TypeError) as e:
        _safe_log_warning(ERROR_DEVICE_LOCK, e)
        lease = None
    if lease is not None:
        lease_task = asyncio.create_task(lease.heartbeat_loop(on_lost=_on_lease_lost))

    async def shutdown_cb():
        _mark_shutting_down()
        try:
            if update_task is not None and not update_task.done():
                update_task.cancel()
                try:
                    await update_task
                except asyncio.CancelledError:
                    pass
                except (RPCError, ClientError, ConnectionError, asyncio.TimeoutError, OSError, RuntimeError, ValueError, TypeError):
                    pass
        except (RPCError, ClientError, ConnectionError, asyncio.TimeoutError, OSError, RuntimeError, ValueError, TypeError):
            pass
        try:
            if bot is not None:
                await bot.disconnect()
        except (RPCError, ConnectionError, asyncio.TimeoutError, OSError, RuntimeError):
            pass
        try:
            if user is not None:
                await user.disconnect()
        except (RPCError, ConnectionError, asyncio.TimeoutError, OSError, RuntimeError):
            pass
        try:
            if lease is not None:
                await asyncio.wait_for(lease.release(), timeout=5.0)
        except (ClientError, asyncio.TimeoutError, OSError, RuntimeError, ValueError, TypeError) as e:
            _safe_log_warning(ERROR_DEVICE_LOCK, e)
        try:
            await close_mistral_client()
        except (ClientError, OSError, RuntimeError, AttributeError, TypeError, ValueError):
            pass
        try:
            await close_session()
        except (ClientError, OSError, RuntimeError):
            pass

    try:
        if lease is not None and lease.lost:
            await asyncio.to_thread(wait_for_enter)
            return
        user = await _start_telegram_client(
            settings,
            session_name="user",
            start_log=INFO_AUTH_TG_USER,
            success_log=INFO_AUTH_TG_USER_SUCCESS,
        )
        lease_clients["user"] = user
        bot = await _start_telegram_client(
            settings,
            session_name="bot",
            start_log=INFO_AUTH_TG_BOT,
            success_log=INFO_AUTH_TG_BOT_SUCCESS,
            bot_token=settings.tg_bot_token,
        )
        lease_clients["bot"] = bot
        if lease is not None and lease.lost:
            await asyncio.to_thread(wait_for_enter)
            return
        spotify = await _start_spotify(settings)
        mistral_valid = await _check_genius_and_mistral(settings)
        await _prefetch_runtime_state(
            settings,
            genius_search_module=genius_search_module,
            mistral_ai_module=mistral_ai_module,
            mistral_valid=mistral_valid,
        )
        bot_module.set_mistral_api_key_valid(mistral_valid)

        await bot_module.setup_bot_commands(bot, user)

        if lease is not None and lease.lost:
            await asyncio.to_thread(wait_for_enter)
            return

        _safe_log_info(INFO_STARTUP_COMPLETE)
        _safe_log_info(INFO_ENTERING_MAIN_LOOP)

        update_task = asyncio.create_task(_run_update_loop())
        schedule_task = asyncio.create_task(
            schedule_daily_update_check(shutdown_cb, enabled=settings.enable_dep_updates)
        )
        await bot.run_until_disconnected()
    except asyncio.CancelledError as e:
        cancelled_exc = e
    finally:
        _mark_shutting_down()
        if lease_task is not None:
            if not lease_task.done():
                lease_task.cancel()
            try:
                await lease_task
            except asyncio.CancelledError:
                pass
            except (ClientError, ConnectionError, asyncio.TimeoutError, OSError, RuntimeError, ValueError, TypeError):
                pass
        if schedule_task is not None and not schedule_task.done():
            schedule_task.cancel()
            try:
                await schedule_task
            except asyncio.CancelledError:
                cancelled_exc = cancelled_exc or asyncio.CancelledError()
            except (ClientError, ConnectionError, asyncio.TimeoutError, OSError, RuntimeError, ValueError, TypeError):
                pass
        try:
            await shutdown_cb()
        except asyncio.CancelledError:
            cancelled_exc = cancelled_exc or asyncio.CancelledError()
            try:
                await shutdown_cb()
            except asyncio.CancelledError:
                pass
            except (RPCError, ClientError, ConnectionError, asyncio.TimeoutError, OSError, RuntimeError, ValueError, TypeError):
                pass
        except (RPCError, ClientError, ConnectionError, asyncio.TimeoutError, OSError, RuntimeError, ValueError, TypeError):
            pass

    if cancelled_exc is None and lease is not None and lease.lost:
        await asyncio.to_thread(wait_for_enter)

    if cancelled_exc is not None:
        raise cancelled_exc

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
        _safe_log_info(INFO_APP_SHUTDOWN)
    except (ClientError, ConnectionError, asyncio.TimeoutError, OSError, RuntimeError, ValueError, TypeError) as e:
        _safe_log_error(ERROR_UNEXPECTED, e)
        if not os.environ.get(ENV_NO_PAUSE):
            wait_for_enter()
        raise
