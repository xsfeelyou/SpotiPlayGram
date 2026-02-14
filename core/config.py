from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os

from constants import (
    ERROR_AUTH_TG_API_ID_MISSING,
    ERROR_AUTH_TG_API_HASH_MISSING,
    ERROR_AUTH_TG_BOT_TOKEN_MISSING,
    ERROR_AUTH_TG_ADMIN_USER_ID_MISSING,
    ERROR_AUTH_TG_CHANNEL_ID_MISSING,
    ERROR_AUTH_TG_MEDIA_MESSAGE_ID_MISSING,
    ERROR_AUTH_TG_TEXT_MESSAGE_ID_MISSING,
    ERROR_AUTH_SPOTIFY_CLIENT_ID_MISSING,
    ERROR_AUTH_SPOTIFY_CLIENT_SECRET_MISSING,
    ERROR_AUTH_SPOTIFY_REDIRECT_URI_MISSING,
    ERROR_ENV_INVALID_INT,
    ERROR_ENV_INVALID_FLOAT,
)

class EnvValidationError(ValueError):
    def __init__(self, messages: list[str]):
        super().__init__("Invalid environment configuration")
        self.messages = messages

def _get_env_str(name: str) -> Optional[str]:
    val = os.getenv(name)
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None

def _get_env_bool(name: str, default: bool) -> bool:
    val = _get_env_str(name)
    if val is None:
        return default
    return val.lower() == "true"

def _get_env_int(name: str, *, errors: list[str], default: Optional[int] = None) -> Optional[int]:
    val = _get_env_str(name)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        errors.append(ERROR_ENV_INVALID_INT.format(name, val))
        return default

def _get_env_float(name: str, *, errors: list[str], default: Optional[float] = None) -> Optional[float]:
    val = _get_env_str(name)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        errors.append(ERROR_ENV_INVALID_FLOAT.format(name, val))
        return default

def _require_env_str(name: str, missing_msg: str, *, errors: list[str]) -> str:
    v = _get_env_str(name)
    if v is None:
        errors.append(missing_msg)
        return ""
    return v

def _require_env_int(name: str, missing_msg: str, *, errors: list[str]) -> int:
    v = _get_env_str(name)
    if v is None:
        errors.append(missing_msg)
        return 0
    try:
        return int(v)
    except (TypeError, ValueError):
        errors.append(ERROR_ENV_INVALID_INT.format(name, v))
        return 0

@dataclass(frozen=True)
class Settings:
    tg_api_id: int
    tg_api_hash: str
    tg_bot_token: str
    tg_admin_user_id: int
    tg_channel_id: int
    tg_media_message_id: int
    tg_text_message_id: int
    tg_auto_delete_messages: bool

    spotify_client_id: str
    spotify_client_secret: str
    spotify_redirect_uri: str
    spotify_language: str
    spotify_request_timeout: float

    genius_access_token: Optional[str]
    enable_genius: bool
    genius_detailed_log: bool

    mistral_api_key: Optional[str]
    enable_mistral: bool
    mistral_model: str

    auth_max_retries: int
    auth_retry_delay_base: float

    update_interval: int
    enable_dep_updates: bool

    @classmethod
    def from_env(cls) -> "Settings":
        errors: list[str] = []

        tg_api_id = _require_env_int("TG_API_ID", ERROR_AUTH_TG_API_ID_MISSING, errors=errors)
        tg_api_hash = _require_env_str("TG_API_HASH", ERROR_AUTH_TG_API_HASH_MISSING, errors=errors)
        tg_bot_token = _require_env_str("TG_BOT_TOKEN", ERROR_AUTH_TG_BOT_TOKEN_MISSING, errors=errors)
        tg_admin_user_id = _require_env_int("TG_ADMIN_USER_ID", ERROR_AUTH_TG_ADMIN_USER_ID_MISSING, errors=errors)
        tg_channel_id = _require_env_int("TG_CHANNEL_ID", ERROR_AUTH_TG_CHANNEL_ID_MISSING, errors=errors)
        tg_media_message_id = _require_env_int("TG_MEDIA_MESSAGE_ID", ERROR_AUTH_TG_MEDIA_MESSAGE_ID_MISSING, errors=errors)
        tg_text_message_id = _require_env_int("TG_TEXT_MESSAGE_ID", ERROR_AUTH_TG_TEXT_MESSAGE_ID_MISSING, errors=errors)

        tg_auto_delete_messages = _get_env_bool("TG_AUTO_DELETE_MESSAGES", True)
        update_interval_raw = _get_env_int("UPDATE_INTERVAL", errors=errors, default=1)
        update_interval = update_interval_raw if update_interval_raw is not None else 1

        spotify_client_id = _require_env_str("SPOTIFY_CLIENT_ID", ERROR_AUTH_SPOTIFY_CLIENT_ID_MISSING, errors=errors)
        spotify_client_secret = _require_env_str("SPOTIFY_CLIENT_SECRET", ERROR_AUTH_SPOTIFY_CLIENT_SECRET_MISSING, errors=errors)
        spotify_redirect_uri = _require_env_str("SPOTIFY_REDIRECT_URI", ERROR_AUTH_SPOTIFY_REDIRECT_URI_MISSING, errors=errors)
        spotify_language = _get_env_str("SPOTIFY_LANGUAGE") or "EN"
        spotify_request_timeout_raw = _get_env_float("SPOTIFY_REQUEST_TIMEOUT", errors=errors, default=5.0)
        spotify_request_timeout = spotify_request_timeout_raw if spotify_request_timeout_raw is not None else 5.0

        genius_access_token = _get_env_str("GENIUS_ACCESS_TOKEN")
        enable_genius = _get_env_bool("ENABLE_GENIUS", True)
        genius_detailed_log = _get_env_bool("GENIUS_DETAILED_LOG", True)

        mistral_api_key = _get_env_str("MISTRAL_API_KEY")
        enable_mistral = _get_env_bool("ENABLE_MISTRAL", True)
        mistral_model = _get_env_str("MISTRAL_MODEL") or "mistral-large-latest"

        auth_max_retries_raw = _get_env_int("AUTH_RETRY_MAX_RETRIES", errors=errors, default=2)
        auth_max_retries = auth_max_retries_raw if auth_max_retries_raw is not None else 2
        if auth_max_retries < 0:
            auth_max_retries = 0

        auth_retry_delay_base_raw = _get_env_float("AUTH_RETRY_DELAY_BASE", errors=errors, default=1.0)
        auth_retry_delay_base = auth_retry_delay_base_raw if auth_retry_delay_base_raw is not None else 1.0
        if auth_retry_delay_base < 0:
            auth_retry_delay_base = 0.0

        enable_dep_updates = _get_env_bool("ENABLE_DEP_UPDATES", True)

        if errors:
            raise EnvValidationError(errors)

        return cls(
            tg_api_id=tg_api_id,
            tg_api_hash=tg_api_hash,
            tg_bot_token=tg_bot_token,
            tg_admin_user_id=tg_admin_user_id,
            tg_channel_id=tg_channel_id,
            tg_media_message_id=tg_media_message_id,
            tg_text_message_id=tg_text_message_id,
            tg_auto_delete_messages=tg_auto_delete_messages,
            spotify_client_id=spotify_client_id,
            spotify_client_secret=spotify_client_secret,
            spotify_redirect_uri=spotify_redirect_uri,
            spotify_language=spotify_language,
            spotify_request_timeout=spotify_request_timeout,
            genius_access_token=genius_access_token,
            enable_genius=enable_genius,
            genius_detailed_log=genius_detailed_log,
            mistral_api_key=mistral_api_key,
            enable_mistral=enable_mistral,
            mistral_model=mistral_model,
            auth_max_retries=auth_max_retries,
            auth_retry_delay_base=auth_retry_delay_base,
            update_interval=update_interval,
            enable_dep_updates=enable_dep_updates,
        )
