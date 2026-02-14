import os
import sys
import asyncio
from subprocess import call

from logger import _safe_log_info, _safe_log_error, dev_logger

def install_libs():
    libs = [
        ("dotenv", "python-dotenv"),
        ("aiohttp", "aiohttp"),
        ("spotipy", "spotipy"),
        ("telethon", "telethon"),
        ("mistralai", "mistralai"),
    ]
    for pkg, inst in libs:
        try: __import__(pkg)
        except ImportError:
            _safe_log_info(f"Installing library: {pkg} ({inst})")
            call([sys.executable, "-m", "pip", "install", inst])

install_libs()

from dotenv import load_dotenv
load_dotenv()

from telethon import TelegramClient
from telethon.errors import RPCError
from spotipy import SpotifyOAuth

import logging
logging.getLogger("telethon").setLevel(logging.ERROR)
from constants import (
    DIRS,
    ERROR_AUTH_GENIUS,
    ERROR_AUTH_GENIUS_INVALID_TOKEN,
    ERROR_AUTH_MISTRAL,
    ERROR_AUTH_MISTRAL_API_KEY_MISSING,
    ERROR_AUTH_SPOTIFY,
    ERROR_AUTH_TG,
    ERROR_GENIUS_ACCESS_TOKEN_INVALID,
    ERROR_SPOTIFY_AUTH_INVALID,
    ERROR_TG_AUTH_INVALID,
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
    INFO_STARTUP_COMPLETE,
    LOG_FORMAT_GENIUS,
    LOG_FORMAT_MISTRAL,
    LOG_FORMAT_SPOTIFY,
    LOG_FORMAT_TELEGRAM,
)
from config import Settings, EnvValidationError

from utils.http_session import init_session, close_session
from utils.spotify_client import AsyncSpotifyClient, get_spg_cache_handler
from utils.update_libs import schedule_daily_update_check
from aiohttp import ClientTimeout, ClientError

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

async def _start_telegram_user(settings: Settings) -> TelegramClient:
    _safe_log_info(INFO_AUTH_TG_USER)
    client = TelegramClient(
        os.path.join(DIRS["SESSION"], "user"),
        settings.tg_api_id,
        settings.tg_api_hash,
    )
    max_attempts, delay_base = _get_auth_retry_settings(settings)
    for attempt in range(1, max_attempts + 1):
        try:
            await client.start()
            _safe_log_info(INFO_AUTH_TG_USER_SUCCESS)
            return client
        except (RPCError, ConnectionError, asyncio.TimeoutError, OSError, ValueError, TypeError, RuntimeError) as e:
            if isinstance(e, (ConnectionError, asyncio.TimeoutError, OSError)) and attempt < max_attempts:
                await _sleep_auth_retry(LOG_FORMAT_TELEGRAM, delay_base, attempt, max_attempts)
                continue
            try:
                if _is_telegram_auth_invalid(e):
                    _safe_log_error(ERROR_TG_AUTH_INVALID)
                    raise
            except (ValueError, TypeError, OSError):
                pass
            try:
                _safe_log_error(ERROR_AUTH_TG)
            except (ValueError, TypeError, OSError):
                pass
            raise

async def _start_telegram_bot(settings: Settings) -> TelegramClient:
    _safe_log_info(INFO_AUTH_TG_BOT)
    client = TelegramClient(
        os.path.join(DIRS["SESSION"], "bot"),
        settings.tg_api_id,
        settings.tg_api_hash,
    )
    max_attempts, delay_base = _get_auth_retry_settings(settings)
    for attempt in range(1, max_attempts + 1):
        try:
            await client.start(bot_token=settings.tg_bot_token)
            _safe_log_info(INFO_AUTH_TG_BOT_SUCCESS)
            return client
        except (RPCError, ConnectionError, asyncio.TimeoutError, OSError, ValueError, TypeError, RuntimeError) as e:
            if isinstance(e, (ConnectionError, asyncio.TimeoutError, OSError)) and attempt < max_attempts:
                await _sleep_auth_retry(LOG_FORMAT_TELEGRAM, delay_base, attempt, max_attempts)
                continue
            try:
                if _is_telegram_auth_invalid(e):
                    _safe_log_error(ERROR_TG_AUTH_INVALID)
                    raise
            except (ValueError, TypeError, OSError):
                pass
            try:
                _safe_log_error(ERROR_AUTH_TG)
            except (ValueError, TypeError, OSError):
                pass
            raise

async def _start_spotify(settings: Settings) -> AsyncSpotifyClient:
    _safe_log_info(INFO_AUTH_SPOTIFY)
    cache_path = os.path.join(DIRS["SESSION"], ".cache")
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
        except (ClientError, ConnectionError, asyncio.TimeoutError, OSError, ValueError, TypeError, RuntimeError) as e:
            try:
                if _is_spotify_auth_invalid(e):
                    _safe_log_error(ERROR_SPOTIFY_AUTH_INVALID)
                    raise
            except (KeyError, IndexError, ValueError, TypeError, OSError):
                pass
            if attempt < max_attempts:
                await _sleep_auth_retry(LOG_FORMAT_SPOTIFY, delay_base, attempt, max_attempts)
                continue
            try:
                _safe_log_error(ERROR_AUTH_SPOTIFY, e)
            except (KeyError, IndexError, ValueError, TypeError, OSError):
                pass
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
            for attempt in range(1, max_attempts + 1):
                try:
                    session = init_session()
                    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
                    timeout = ClientTimeout(total=6.0)
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
                            break
                        if (r.status == 429 or 500 <= r.status < 600) and attempt < max_attempts:
                            await _sleep_auth_retry(LOG_FORMAT_GENIUS, delay_base, attempt, max_attempts)
                            continue
                        _safe_log_error(ERROR_AUTH_GENIUS)
                        break
                except (ClientError, ConnectionError, asyncio.TimeoutError, OSError, RuntimeError):
                    _safe_log_error(ERROR_AUTH_GENIUS, logger=dev_logger, exc_info=True)
                    if attempt < max_attempts:
                        await _sleep_auth_retry(LOG_FORMAT_GENIUS, delay_base, attempt, max_attempts)
                        continue
                    genius_ok = True
                    break

            if genius_ok:
                _safe_log_info(INFO_AUTH_GENIUS_SUCCESS)

    if not settings.enable_mistral:
        _safe_log_info(INFO_MISTRAL_DISABLED)
        return False

    api_key = settings.mistral_api_key
    if not api_key:
        _safe_log_error(ERROR_AUTH_MISTRAL_API_KEY_MISSING)
        return False

    _safe_log_info(INFO_AUTH_MISTRAL)
    for attempt in range(1, max_attempts + 1):
        try:
            m_session = init_session()
            m_headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
            m_timeout = ClientTimeout(total=6.0)
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

async def main():
    try:
        settings = Settings.from_env()
    except EnvValidationError as e:
        for msg in getattr(e, "messages", []) or []:
            try:
                _safe_log_error(msg)
            except (ValueError, TypeError, OSError):
                pass
        raise

    from status import bot as bot_module
    from status import spotify_status as spotify_status_module
    from lyrics import genius_search as genius_search_module
    from lyrics import mistral_ai as mistral_ai_module

    bot_module.set_settings(settings)
    spotify_status_module.set_settings(settings)
    genius_search_module.set_settings(settings)
    mistral_ai_module.set_settings(settings)

    init_session()

    user = None
    bot = None
    spotify = None
    update_task = None
    schedule_task = None
    cancelled_exc = None

    async def shutdown_cb():
        nonlocal update_task
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
            await close_session()
        except (ClientError, OSError, RuntimeError):
            pass

    try:
        user = await _start_telegram_user(settings)
        bot = await _start_telegram_bot(settings)
        spotify = await _start_spotify(settings)
        mistral_valid = await _check_genius_and_mistral(settings)
        bot_module.set_mistral_api_key_valid(mistral_valid)

        await bot_module.setup_bot_commands(bot, user)

        _safe_log_info(INFO_STARTUP_COMPLETE)
        _safe_log_info(INFO_ENTERING_MAIN_LOOP)

        update_task = asyncio.create_task(bot_module.patched_update_channel(user, bot, spotify))
        schedule_task = asyncio.create_task(
            schedule_daily_update_check(shutdown_cb, enabled=settings.enable_dep_updates)
        )
        await bot.run_until_disconnected()
    except asyncio.CancelledError as e:
        cancelled_exc = e
    finally:
        if schedule_task is not None and not schedule_task.done():
            schedule_task.cancel()
            try:
                await schedule_task
            except asyncio.CancelledError:
                try:
                    cancelled_exc = cancelled_exc or asyncio.CancelledError()
                except Exception:
                    pass
            except (ClientError, ConnectionError, asyncio.TimeoutError, OSError, RuntimeError, ValueError, TypeError):
                pass
        try:
            await shutdown_cb()
        except asyncio.CancelledError:
            try:
                cancelled_exc = cancelled_exc or asyncio.CancelledError()
            except Exception:
                pass
            try:
                await shutdown_cb()
            except asyncio.CancelledError:
                pass
            except (RPCError, ClientError, ConnectionError, asyncio.TimeoutError, OSError, RuntimeError, ValueError, TypeError):
                pass
        except (RPCError, ClientError, ConnectionError, asyncio.TimeoutError, OSError, RuntimeError, ValueError, TypeError):
            pass

    if cancelled_exc is not None:
        raise cancelled_exc

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        _safe_log_info(INFO_APP_SHUTDOWN)
    except (ClientError, ConnectionError, asyncio.TimeoutError, OSError, RuntimeError, ValueError, TypeError) as e:
        _safe_log_error(ERROR_UNEXPECTED, e)
        raise
