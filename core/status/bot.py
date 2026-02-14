import re
import html
import aiohttp
from telethon import events, Button
from telethon.tl.types import PeerChannel
from telethon.errors import RPCError
from datetime import datetime
import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from logger import _safe_log_info, _safe_log_error
from constants import (
    ACCESS_DENIED_MESSAGE,
    BTN_BACK,
    BTN_DISABLE,
    BTN_ENABLE,
    BTN_MENU_FEAT_PATTERNS,
    BTN_MENU_GENIUS_SETTINGS,
    BTN_MENU_MISTRAL,
    BTN_MENU_STATUS,
    BTN_MENU_TAG_PATTERNS,
    BTN_PATTERNS_ADD,
    BTN_PATTERNS_CONFIRM_ADD,
    BTN_PATTERNS_CONFIRM_DELETE,
    BTN_PATTERNS_DELETE,
    BTN_REFRESH_MISTRAL,
    DEFAULT_MESSAGE,
    ERROR_MESSAGE_DELETE,
    ERROR_MISTRAL_FETCH_MODELS,
    ERROR_MISTRAL_MODEL_FORMAT,
    ERROR_MISTRAL_MODEL_INVALID,
    ERROR_PATTERNS_ALREADY_EXIST,
    ERROR_PATTERNS_DUPLICATES,
    ERROR_PATTERNS_EMPTY,
    ERROR_PATTERNS_EXISTS_IN_FEAT,
    ERROR_PATTERNS_EXISTS_IN_TAG,
    ERROR_PATTERNS_NOT_FOUND,
    ERROR_UPDATE_LOOP,
    INFO_AUTO_DELETE_DISABLED,
    INFO_AUTO_DELETE_ENABLED,
    INFO_BUTTON_DISABLE_PRESSED,
    INFO_BUTTON_ENABLE_PRESSED,
    INFO_DEP_UPDATES_DISABLED,
    INFO_DEP_UPDATES_ENABLED,
    INFO_GENIUS_SETTINGS_MENU,
    INFO_GENIUS_SETTINGS_MENU_NO_MISTRAL,
    INFO_MAIN_MENU,
    INFO_MESSAGE_AUTO_DELETED,
    INFO_MESSAGE_DELETED,
    INFO_MISTRAL_CURRENT_MODEL,
    INFO_MISTRAL_ENTER_MODEL,
    INFO_MISTRAL_FALLBACK_MODELS,
    INFO_MISTRAL_LOADING,
    INFO_MISTRAL_MENU_TITLE,
    INFO_MISTRAL_MODEL_SAVED,
    INFO_MISTRAL_SAVING,
    INFO_PATTERNS_APPLIED,
    INFO_PATTERNS_CONFIRM_ADD,
    INFO_PATTERNS_CONFIRM_DELETE,
    INFO_PATTERNS_ENTER_ADD,
    INFO_PATTERNS_ENTER_DELETE,
    INFO_PATTERNS_MENU_FEAT,
    INFO_PATTERNS_MENU_TAG,
    INFO_PATTERNS_SAVED_WILL_APPLY,
    INFO_STATUS_DISABLED,
    INFO_STATUS_DISABLING,
    INFO_STATUS_ENABLED,
    INFO_STATUS_ENABLING,
    PATTERNS_KIND_FEAT_LABEL,
    PATTERNS_KIND_GENERIC_LABEL,
    PATTERNS_KIND_TAG_LABEL,
    VALUE_NOT_SET,
)
from utils.http_session import init_session
from utils.telethon_utils import safe_call_telethon
from lyrics.mistral_ai import (
    write_mistral_model,
    read_mistral_model,
    ensure_selected_models_cached,
    _split_model_ids,
)
from lyrics.mistral_models import fetch_mistral_models_ids
from config import Settings

_settings: Optional[Settings] = None
_mistral_api_key_valid: bool = False

def set_settings(settings: Settings) -> None:
    global _settings
    _settings = settings

def set_mistral_api_key_valid(valid: bool) -> None:
    global _mistral_api_key_valid
    _mistral_api_key_valid = bool(valid)

def get_settings() -> Settings:
    if _settings is None:
        raise RuntimeError("Settings are not initialized")
    return _settings

def _is_mistral_ui_enabled(st: Settings) -> bool:
    return bool(st.enable_mistral and _mistral_api_key_valid)

async def _edit_or_respond(message, event, text: str, *, parse_mode: Optional[str] = None, buttons=None) -> None:
    if message:
        try:
            await message.edit(text, parse_mode=parse_mode, buttons=buttons)
            return
        except (RPCError, ConnectionError, asyncio.TimeoutError, OSError):
            pass
    await event.respond(text, parse_mode=parse_mode, buttons=buttons)

@dataclass
class BotState:
    status_enabled: bool = True
    auto_delete_enabled: bool = False
    manual_disable_in_progress: bool = False
    first_enable_log_needed: bool = False
    control_message_event: Optional[Any] = None
    disable_in_progress: bool = False
    enable_in_progress: bool = False
    update_task: Optional[asyncio.Task] = None
    last_spotify_keepalive: datetime = datetime.min
    last_genius_keepalive: datetime = datetime.min

    current_menu: str = "main"
    awaiting_mistral_model_input: bool = False
    mistral_busy: bool = False
    mistral_refresh_enabled: bool = False
    awaiting_patterns_input: bool = False
    awaiting_patterns_confirm: bool = False
    patterns_kind: Optional[str] = None
    patterns_action: Optional[str] = None
    patterns_pending: Optional[list[str]] = None

    def reset_menu(self) -> None:
        self.current_menu = "main"
        self.awaiting_mistral_model_input = False
        self.mistral_busy = False
        self.mistral_refresh_enabled = False
        self.awaiting_patterns_input = False
        self.awaiting_patterns_confirm = False
        self.patterns_kind = None
        self.patterns_action = None
        self.patterns_pending = None

    def set_menu(self, menu: str, *, patterns_kind: Optional[str] = None, awaiting_mistral: bool = False) -> None:
        self.current_menu = menu
        self.awaiting_mistral_model_input = awaiting_mistral
        self.mistral_busy = False
        self.mistral_refresh_enabled = False
        self.awaiting_patterns_input = False
        self.awaiting_patterns_confirm = False
        self.patterns_kind = patterns_kind
        self.patterns_action = None
        self.patterns_pending = None

_state = BotState()

def _cancel_update_task(state: BotState) -> None:
    task = getattr(state, "update_task", None)
    if task is not None and isinstance(task, asyncio.Task) and not task.done():
        try:
            task.cancel()
        except Exception:
            pass

async def delete_all_messages(user_client, channel_id):
    st = get_settings()
    channel = PeerChannel(channel_id)
    keep_message_ids = [1, st.tg_media_message_id, st.tg_text_message_id]
    try:
        messages = await safe_call_telethon(
            user_client,
            call=lambda: user_client.get_messages(channel, limit=100),
            retries=2,
        )
    except (RPCError, ConnectionError, asyncio.TimeoutError, OSError, aiohttp.ClientError) as e:
        _safe_log_error(ERROR_MESSAGE_DELETE, e)
        return
    if messages is None:
        _safe_log_error(ERROR_MESSAGE_DELETE, "")
        return
    ids_to_delete = [m.id for m in messages if m.id not in keep_message_ids]
    if not ids_to_delete:
        return
    try:
        deleted = await safe_call_telethon(
            user_client,
            call=lambda: user_client.delete_messages(channel, ids_to_delete),
            retries=2,
            treat_none_as_success=True,
        )
    except (RPCError, ConnectionError, asyncio.TimeoutError, OSError, aiohttp.ClientError) as e:
        _safe_log_error(ERROR_MESSAGE_DELETE, e)
        return
    if deleted is None:
        _safe_log_error(ERROR_MESSAGE_DELETE, "")
        return
    for mid in ids_to_delete:
        _safe_log_info(INFO_MESSAGE_DELETED, mid)
    return

async def _apply_default_state(
    *,
    bot_client,
    write_track_info,
    update_messages,
    update_channel_info,
    suppress_errors: bool,
) -> None:
    if suppress_errors:
        try:
            write_track_info(None)
        except (OSError, ValueError, TypeError):
            pass
        try:
            await update_messages(
                bot_client=bot_client,
                cover_url=None,
                is_playing=False,
                artist_names=None,
                track_name=None,
                album_name=None,
                release_date=None,
                artist_urls=None,
                track_url=None,
                album_url=None
            )
        except (RPCError, ConnectionError, asyncio.TimeoutError, OSError):
            pass
        try:
            await update_channel_info(bot_client, DEFAULT_MESSAGE, False)
        except (RPCError, ConnectionError, asyncio.TimeoutError, OSError):
            pass
        return

    write_track_info(None)
    await update_messages(
        bot_client=bot_client,
        cover_url=None,
        is_playing=False,
        artist_names=None,
        track_name=None,
        album_name=None,
        release_date=None,
        artist_urls=None,
        track_url=None,
        album_url=None
    )
    await update_channel_info(bot_client, DEFAULT_MESSAGE, False)

async def _apply_default_state_and_delete(
    *,
    bot_client,
    user_client,
    channel_id,
    write_track_info,
    update_messages,
    update_channel_info,
    suppress_errors: bool,
) -> None:
    await _apply_default_state(
        bot_client=bot_client,
        write_track_info=write_track_info,
        update_messages=update_messages,
        update_channel_info=update_channel_info,
        suppress_errors=suppress_errors,
    )
    if suppress_errors:
        try:
            await delete_all_messages(user_client, channel_id)
        except (RPCError, ConnectionError, asyncio.TimeoutError, OSError):
            pass
        return
    await delete_all_messages(user_client, channel_id)

async def setup_bot_commands(bot, user):
    st = get_settings()
    state = _state
    from .spotify_status import write_track_info, update_messages, update_channel_info
    state.auto_delete_enabled = st.tg_auto_delete_messages
    if state.auto_delete_enabled:
        _safe_log_info(INFO_AUTO_DELETE_ENABLED)
    else:
        _safe_log_info(INFO_AUTO_DELETE_DISABLED)
    if st.enable_dep_updates:
        _safe_log_info(INFO_DEP_UPDATES_ENABLED)
    else:
        _safe_log_info(INFO_DEP_UPDATES_DISABLED)
    api_key = st.mistral_api_key
    if api_key and st.enable_mistral:
        try:
            await ensure_selected_models_cached(api_key)
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError, RuntimeError, ValueError):
            pass

    @user.on(events.NewMessage(chats=st.tg_channel_id))
    async def auto_delete_handler(event):
        if not state.auto_delete_enabled:
            return

        keep_message_ids = [1, st.tg_media_message_id, st.tg_text_message_id]

        if event.id not in keep_message_ids:
            try:
                await event.delete()
                _safe_log_info(INFO_MESSAGE_AUTO_DELETED, event.id)
            except (RPCError, ConnectionError, asyncio.TimeoutError, OSError) as e:
                _safe_log_error(ERROR_MESSAGE_DELETE, e)

    @bot.on(events.NewMessage(pattern="/start"))
    async def start_handler(event):
        sender = await event.get_sender()
        admin_user_id = st.tg_admin_user_id

        if sender.id == admin_user_id:
            await send_main_menu(event, bot)
        else:
            await event.respond(ACCESS_DENIED_MESSAGE)

    async def show_mistral_menu(event):
        if not _is_mistral_ui_enabled(st):
            state.set_menu("genius")
            await show_genius_settings_menu(event)
            return
        try:
            state.mistral_busy = True
            state.mistral_refresh_enabled = False
            menu_buttons = [[
                Button.text(BTN_BACK, resize=True),
                Button.text(INFO_MAIN_MENU, resize=True),
            ]]
            menu_buttons_with_refresh = [[
                Button.text(BTN_REFRESH_MISTRAL, resize=True),
                Button.text(BTN_BACK, resize=True),
                Button.text(INFO_MAIN_MENU, resize=True),
            ]]
            current_model = read_mistral_model()
            _models_list = _split_model_ids(current_model)
            if not _models_list and st.mistral_model:
                _models_list = [st.mistral_model]
            primary_model = _models_list[0] if _models_list else st.mistral_model
            fallback_models = _models_list[1:] if len(_models_list) > 1 else []

            chat = await event.get_input_chat()
            loading_msg = await bot.send_message(chat, INFO_MISTRAL_LOADING, buttons=menu_buttons)

            ids, details_text, err = await fetch_mistral_models_ids(st.mistral_api_key)
            models_text = details_text or ""
            if err:
                models_text = ERROR_MISTRAL_FETCH_MODELS.format(err)

            if fallback_models:
                fallback_text = ", ".join(f"<code>{html.escape(m)}</code>" for m in fallback_models)
            else:
                fallback_text = VALUE_NOT_SET
            header = "\n\n".join([
                INFO_MISTRAL_MENU_TITLE,
                INFO_MISTRAL_CURRENT_MODEL.format(html.escape(primary_model)),
                INFO_MISTRAL_FALLBACK_MODELS.format(fallback_text),
            ])

            if models_text:
                lines = models_text.splitlines()
                max_len = 3800
                reserved_tail = 2 + len(INFO_MISTRAL_ENTER_MODEL)
                first_len = len(header) + 2 + reserved_tail
                i = 0
                first_chunk_lines = []
                while i < len(lines):
                    ln = lines[i]
                    if len(ln) > 700:
                        ln = ln[:700] + "…"
                    add_len = len(ln) + 1
                    if first_len + add_len > max_len and first_chunk_lines:
                        break
                    first_chunk_lines.append(ln)
                    first_len += add_len
                    i += 1
                first_text = header + "\n\n" + "\n".join(first_chunk_lines) + "\n\n" + INFO_MISTRAL_ENTER_MODEL
                await bot.send_message(chat, first_text, parse_mode="html", buttons=menu_buttons_with_refresh)
                state.mistral_refresh_enabled = True
                if i < len(lines):
                    chunk = []
                    chunk_len = 0
                    while i < len(lines):
                        ln = lines[i]
                        if len(ln) > 700:
                            ln = ln[:700] + "…"
                        add_len = len(ln) + 1
                        if chunk_len + add_len > max_len and chunk:
                            await bot.send_message(chat, "\n".join(chunk), parse_mode="html")
                            chunk = []
                            chunk_len = 0
                        chunk.append(ln)
                        chunk_len += add_len
                        i += 1
                    if chunk:
                        await bot.send_message(chat, "\n".join(chunk), parse_mode="html")
            else:
                await bot.send_message(
                    chat,
                    header + ("\n\n" + models_text if models_text else "") + "\n\n" + INFO_MISTRAL_ENTER_MODEL,
                    parse_mode="html",
                    buttons=menu_buttons_with_refresh,
                )
                state.mistral_refresh_enabled = True

            if loading_msg:
                try:
                    await loading_msg.delete()
                except (RPCError, ConnectionError, asyncio.TimeoutError, OSError):
                    pass
        except (RPCError, ConnectionError, asyncio.TimeoutError, OSError, aiohttp.ClientError, RuntimeError, ValueError, TypeError) as e:
            _safe_log_error(ERROR_MISTRAL_FETCH_MODELS, e)
            chat = await event.get_input_chat()
            await bot.send_message(
                chat,
                ERROR_MISTRAL_FETCH_MODELS.format(str(e)),
                buttons=[[Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]],
            )
        finally:
            state.mistral_busy = False

    async def show_genius_settings_menu(event):
        chat = await event.get_input_chat()
        buttons = []
        mistral_ui_enabled = _is_mistral_ui_enabled(st)
        if mistral_ui_enabled:
            buttons.append([Button.text(BTN_MENU_MISTRAL, resize=True)])
        buttons.append([Button.text(BTN_MENU_TAG_PATTERNS, resize=True), Button.text(BTN_MENU_FEAT_PATTERNS, resize=True)])
        buttons.append([Button.text(BTN_BACK, resize=True)])
        menu_text = INFO_GENIUS_SETTINGS_MENU if mistral_ui_enabled else INFO_GENIUS_SETTINGS_MENU_NO_MISTRAL
        await bot.send_message(
            chat,
            menu_text,
            parse_mode="html",
            buttons=buttons,
        )

    @bot.on(events.NewMessage)
    async def main_menu_router(event):
        state = _state
        try:
            sender = await event.get_sender()
            admin_user_id = st.tg_admin_user_id
            if sender.id != admin_user_id:
                return
            text = (event.raw_text or "").strip()
            if not text or text.startswith("/"):
                return
            if text == INFO_MAIN_MENU:
                state.reset_menu()
                await send_main_menu(event, bot)
                return
            if text == BTN_REFRESH_MISTRAL:
                if state.current_menu == "mistral" and state.mistral_refresh_enabled and not state.mistral_busy:
                    state.awaiting_mistral_model_input = True
                    await show_mistral_menu(event)
                return
            if text == BTN_BACK:
                if state.current_menu == "patterns" and state.patterns_kind in ("tag", "feat") and (state.awaiting_patterns_input or state.awaiting_patterns_confirm):
                    state.awaiting_mistral_model_input = False
                    state.awaiting_patterns_input = False
                    state.awaiting_patterns_confirm = False
                    state.patterns_action = None
                    state.patterns_pending = None
                    if state.patterns_kind == "tag":
                        await event.respond(
                            INFO_PATTERNS_MENU_TAG,
                            parse_mode="html",
                            buttons=[[Button.text(BTN_PATTERNS_ADD, resize=True), Button.text(BTN_PATTERNS_DELETE, resize=True)], [Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]],
                        )
                        return
                    if state.patterns_kind == "feat":
                        await event.respond(
                            INFO_PATTERNS_MENU_FEAT,
                            parse_mode="html",
                            buttons=[[Button.text(BTN_PATTERNS_ADD, resize=True), Button.text(BTN_PATTERNS_DELETE, resize=True)], [Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]],
                        )
                        return
                if state.current_menu in ("patterns", "mistral"):
                    state.set_menu("genius")
                    await show_genius_settings_menu(event)
                    return
                if state.current_menu in ("genius", "status"):
                    state.reset_menu()
                    await send_main_menu(event, bot)
                    return
                state.reset_menu()
                await send_main_menu(event, bot)
                return
            if text == BTN_MENU_STATUS:
                state.set_menu("status")
                await send_control_message(event, bot)
                return
            if text == BTN_MENU_GENIUS_SETTINGS:
                state.set_menu("genius")
                await show_genius_settings_menu(event)
                return

            if text == BTN_MENU_TAG_PATTERNS:
                state.set_menu("patterns", patterns_kind="tag")
                await event.respond(
                    INFO_PATTERNS_MENU_TAG,
                    parse_mode="html",
                    buttons=[[Button.text(BTN_PATTERNS_ADD, resize=True), Button.text(BTN_PATTERNS_DELETE, resize=True)], [Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]],
                )
                return

            if text == BTN_MENU_FEAT_PATTERNS:
                state.set_menu("patterns", patterns_kind="feat")
                await event.respond(
                    INFO_PATTERNS_MENU_FEAT,
                    parse_mode="html",
                    buttons=[[Button.text(BTN_PATTERNS_ADD, resize=True), Button.text(BTN_PATTERNS_DELETE, resize=True)], [Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]],
                )
                return
            if text == BTN_MENU_MISTRAL:
                if not _is_mistral_ui_enabled(st):
                    state.set_menu("genius")
                    await show_genius_settings_menu(event)
                    return
                state.set_menu("mistral", awaiting_mistral=True)
                await show_mistral_menu(event)
                return

            if state.current_menu == "patterns" and state.patterns_kind in ("tag", "feat") and text in (BTN_PATTERNS_ADD, BTN_PATTERNS_DELETE):
                state.awaiting_mistral_model_input = False
                state.awaiting_patterns_confirm = False
                state.patterns_pending = None
                state.patterns_action = "add" if text == BTN_PATTERNS_ADD else "delete"
                state.awaiting_patterns_input = True
                await event.respond(
                    INFO_PATTERNS_ENTER_ADD if state.patterns_action == "add" else INFO_PATTERNS_ENTER_DELETE,
                    parse_mode="html",
                    buttons=[[Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]],
                )
                return

            if state.awaiting_patterns_confirm and state.current_menu == "patterns" and state.patterns_kind in ("tag", "feat"):
                if (state.patterns_action == "add" and text == BTN_PATTERNS_CONFIRM_ADD) or (state.patterns_action == "delete" and text == BTN_PATTERNS_CONFIRM_DELETE):
                    try:
                        from lyrics.patterns.base import add_patterns, remove_patterns
                    except ImportError:
                        add_patterns = None
                        remove_patterns = None

                    try:
                        pending = list(state.patterns_pending or [])
                    except TypeError:
                        pending = []
                    if not pending:
                        state.awaiting_patterns_confirm = False
                        state.patterns_pending = None
                        await event.respond(ERROR_PATTERNS_EMPTY, buttons=[[Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]])
                        return

                    try:
                        if state.patterns_action == "add" and callable(add_patterns):
                            _ = add_patterns(state.patterns_kind, pending)
                        elif state.patterns_action == "delete" and callable(remove_patterns):
                            _ = remove_patterns(state.patterns_kind, pending)
                    except (OSError, ValueError, TypeError):
                        pass

                    state.awaiting_patterns_confirm = False
                    state.awaiting_patterns_input = False
                    state.patterns_action = None
                    state.patterns_pending = None
                    await event.respond(
                        INFO_PATTERNS_SAVED_WILL_APPLY,
                        parse_mode="html",
                        buttons=[[Button.text(BTN_PATTERNS_ADD, resize=True), Button.text(BTN_PATTERNS_DELETE, resize=True)], [Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]],
                    )
                    return
            if state.current_menu == "status":
                if text in (INFO_STATUS_ENABLING, INFO_STATUS_DISABLING):
                    return
                if text == BTN_ENABLE:
                    if state.enable_in_progress or state.disable_in_progress:
                        return
                    state.enable_in_progress = True
                    _safe_log_info(INFO_BUTTON_ENABLE_PRESSED)
                    _cancel_update_task(state)
                    state.status_enabled = True
                    state.first_enable_log_needed = True
                    try:
                        enabling_msg = await event.respond(
                            INFO_STATUS_ENABLING,
                            buttons=[[Button.text(BTN_BACK, resize=True)]]
                        )
                        state.control_message_event = enabling_msg
                    except (RPCError, ConnectionError, asyncio.TimeoutError, OSError):
                        state.control_message_event = None
                    try:
                        await delete_all_messages(user, st.tg_channel_id)
                    except (RPCError, ConnectionError, asyncio.TimeoutError, OSError):
                        pass
                    return
                if text == BTN_DISABLE:
                    if state.disable_in_progress or state.enable_in_progress:
                        return
                    state.disable_in_progress = True
                    _safe_log_info(INFO_BUTTON_DISABLE_PRESSED)
                    _cancel_update_task(state)
                    state.status_enabled = False
                    state.manual_disable_in_progress = True
                    state.control_message_event = None
                    try:
                        try:
                            await event.respond(
                                INFO_STATUS_DISABLING,
                                buttons=[[Button.text(BTN_BACK, resize=True)]]
                            )
                        except (RPCError, ConnectionError, asyncio.TimeoutError, OSError):
                            pass

                        try:
                            await _apply_default_state_and_delete(
                                bot_client=bot,
                                user_client=user,
                                channel_id=st.tg_channel_id,
                                write_track_info=write_track_info,
                                update_messages=update_messages,
                                update_channel_info=update_channel_info,
                                suppress_errors=True,
                            )
                        except (RPCError, ConnectionError, asyncio.TimeoutError, OSError, ValueError, TypeError):
                            pass

                        try:
                            await event.respond(
                                INFO_STATUS_DISABLED,
                                buttons=[[Button.text(BTN_ENABLE, resize=True)], [Button.text(BTN_BACK, resize=True)]]
                            )
                        except (RPCError, ConnectionError, asyncio.TimeoutError, OSError):
                            pass
                        _safe_log_info(INFO_STATUS_DISABLED)
                        return
                    finally:
                        state.manual_disable_in_progress = False
                        state.disable_in_progress = False

            if state.awaiting_patterns_input and state.current_menu == "patterns" and state.patterns_kind in ("tag", "feat") and text:
                raw = text
                parts = [p.strip() for p in raw.split("..")]
                parts = [p for p in parts if p]
                if not parts:
                    await event.respond(ERROR_PATTERNS_EMPTY, buttons=[[Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]])
                    return

                try:
                    from lyrics.patterns.base import pattern_exists, _normalize_item
                except ImportError:
                    pattern_exists = None
                    _normalize_item = None

                normalized = []
                lower_seen = set()
                has_dup = False
                for p in parts:
                    try:
                        pp = _normalize_item(p).lower() if callable(_normalize_item) else p.strip()
                    except (ValueError, TypeError):
                        pp = p.strip()
                    if not pp:
                        continue
                    key = pp.lower()
                    if key in lower_seen:
                        has_dup = True
                        break
                    lower_seen.add(key)
                    normalized.append(pp)

                if not normalized:
                    await event.respond(ERROR_PATTERNS_EMPTY, buttons=[[Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]])
                    return

                if has_dup:
                    await event.respond(ERROR_PATTERNS_DUPLICATES, buttons=[[Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]])
                    return

                can_check_patterns = callable(pattern_exists)
                other_kind = "tag" if state.patterns_kind == "feat" else "feat"

                if state.patterns_action == "add":
                    bad = False
                    conflict_other: str | None = None
                    conflict_patterns = []
                    for p in normalized:
                        try:
                            if can_check_patterns and pattern_exists(state.patterns_kind, p):
                                bad = True
                                break
                        except (ValueError, TypeError):
                            continue
                        try:
                            if can_check_patterns and pattern_exists(other_kind, p):
                                conflict_other = other_kind
                                conflict_patterns.append(p)
                        except (ValueError, TypeError):
                            continue
                    if bad:
                        await event.respond(ERROR_PATTERNS_ALREADY_EXIST, buttons=[[Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]])
                        return
                    if conflict_other and conflict_patterns:
                        formatted_conflicts = "\n".join(
                            f"<blockquote><code>{html.escape(p)}</code></blockquote>" for p in conflict_patterns
                        )
                        if conflict_other == "tag":
                            await event.respond(
                                ERROR_PATTERNS_EXISTS_IN_TAG.format(formatted_conflicts),
                                parse_mode="html",
                                buttons=[[Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]],
                            )
                            return
                        await event.respond(
                            ERROR_PATTERNS_EXISTS_IN_FEAT.format(formatted_conflicts),
                            parse_mode="html",
                            buttons=[[Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]],
                        )
                        return
                elif state.patterns_action == "delete":
                    bad = False
                    for p in normalized:
                        try:
                            if can_check_patterns and (not pattern_exists(state.patterns_kind, p)):
                                bad = True
                                break
                        except (ValueError, TypeError):
                            continue
                    if bad:
                        await event.respond(ERROR_PATTERNS_NOT_FOUND, buttons=[[Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]])
                        return

                state.awaiting_patterns_input = False
                state.awaiting_patterns_confirm = True
                state.patterns_pending = normalized

                formatted = "\n".join(
                    f"<blockquote><code>{html.escape(str(p))}</code></blockquote>" for p in normalized
                )

                await event.respond(
                    (INFO_PATTERNS_CONFIRM_ADD if state.patterns_action == "add" else INFO_PATTERNS_CONFIRM_DELETE).format(formatted),
                    parse_mode="html",
                    buttons=[[Button.text(BTN_PATTERNS_CONFIRM_ADD if state.patterns_action == "add" else BTN_PATTERNS_CONFIRM_DELETE, resize=True)], [Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]],
                )
                return
            if state.awaiting_mistral_model_input and state.current_menu == "mistral" and text:
                if state.mistral_busy:
                    return
                state.mistral_busy = True
                state.mistral_refresh_enabled = False
                try:
                    pattern = r"^[A-Za-z0-9._-]+(?:[\s,]+[A-Za-z0-9._-]+)*$"
                    if not re.fullmatch(pattern, text):
                        await event.respond(ERROR_MISTRAL_MODEL_FORMAT, parse_mode="html", buttons=[[Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]])
                        return
                except (re.error, TypeError):
                    bad_separators = (";", "|", "\n", "\t")
                    if any(sep in text for sep in bad_separators):
                        await event.respond(ERROR_MISTRAL_MODEL_FORMAT, parse_mode="html", buttons=[[Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]])
                        return

                try:
                    saving_msg = await event.respond(
                        INFO_MISTRAL_SAVING,
                        buttons=[[Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]]
                    )
                except (RPCError, ConnectionError, asyncio.TimeoutError, OSError):
                    saving_msg = None

                ids, _, err = await fetch_mistral_models_ids(st.mistral_api_key)
                if err:
                    await _edit_or_respond(
                        saving_msg,
                        event,
                        ERROR_MISTRAL_FETCH_MODELS.format(err),
                        buttons=[[Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]],
                    )
                    return
                parts = _split_model_ids(text)
                seen = set()
                normalized = []
                for p in parts:
                    if p not in seen:
                        seen.add(p)
                        normalized.append(p)
                if not normalized:
                    await _edit_or_respond(
                        saving_msg,
                        event,
                        ERROR_MISTRAL_MODEL_INVALID,
                        buttons=[[Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]],
                    )
                    return

                valid_ids = set(ids or [])
                for p in normalized:
                    if p not in valid_ids:
                        await _edit_or_respond(
                            saving_msg,
                            event,
                            ERROR_MISTRAL_MODEL_INVALID,
                            buttons=[[Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]],
                        )
                        return

                new_model_value = ", ".join(normalized)
                try:
                    _ = write_mistral_model(new_model_value)
                except (OSError, ValueError, TypeError) as e:
                    await _edit_or_respond(
                        saving_msg,
                        event,
                        ERROR_MISTRAL_FETCH_MODELS.format(e),
                        buttons=[[Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]],
                    )
                    return

                formatted = ", ".join(f"<code>{html.escape(str(m))}</code>" for m in normalized)

                await _edit_or_respond(
                    saving_msg,
                    event,
                    INFO_MISTRAL_MODEL_SAVED.format(formatted),
                    parse_mode="html",
                    buttons=[[Button.text(BTN_BACK, resize=True), Button.text(INFO_MAIN_MENU, resize=True)]],
                )

                await show_mistral_menu(event)
                return

        except (RPCError, ConnectionError, asyncio.TimeoutError, OSError, aiohttp.ClientError, RuntimeError, ValueError, TypeError):
            pass
        finally:
            state.mistral_busy = False

async def send_control_message(event, bot):
    state = _state
    message = INFO_STATUS_ENABLED if state.status_enabled else INFO_STATUS_DISABLED
    buttons = [
        [Button.text(BTN_DISABLE if state.status_enabled else BTN_ENABLE, resize=True)],
        [Button.text(BTN_BACK, resize=True)]
    ]
    await event.respond(message, buttons=buttons)

async def send_main_menu(event, bot):
    menu_buttons = [[Button.text(BTN_MENU_STATUS, resize=True)], [Button.text(BTN_MENU_GENIUS_SETTINGS, resize=True)]]
    chat = await event.get_input_chat()
    await bot.send_message(chat, INFO_MAIN_MENU, buttons=menu_buttons)

async def patched_update_channel(user_client, bot_client, spotify_client):
    from .spotify_status import update_channel, update_messages, update_channel_info, read_track_info, write_track_info
    state = _state
    st = get_settings()
    try:
        while True:
            sleep_seconds = st.update_interval
            try:
                if not state.status_enabled:
                    current_time = datetime.now()
                    if (current_time - state.last_spotify_keepalive).total_seconds() >= 30:
                        try:
                            await spotify_client.current_playback()
                            state.last_spotify_keepalive = current_time
                        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                            pass
                    if st.enable_genius and (current_time - state.last_genius_keepalive).total_seconds() >= 30:
                        try:
                            token = st.genius_access_token
                            if token:
                                session = init_session()
                                timeout = aiohttp.ClientTimeout(total=5.0)
                                headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
                                async with session.get("https://api.genius.com/search", params={"q": "ping"}, headers=headers, timeout=timeout) as r:
                                    if 200 <= r.status < 300:
                                        state.last_genius_keepalive = current_time
                        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                            pass
                    if state.manual_disable_in_progress:
                        await asyncio.sleep(st.update_interval)
                        continue
                    track_info = read_track_info()
                    if track_info["track_id"] is not None:
                        await _apply_default_state(
                            bot_client=bot_client,
                            write_track_info=write_track_info,
                            update_messages=update_messages,
                            update_channel_info=update_channel_info,
                            suppress_errors=False,
                        )
                else:
                    task = asyncio.create_task(update_channel(bot_client, spotify_client))
                    state.update_task = task
                    try:
                        track_id_result = await task
                    except asyncio.CancelledError:
                        if state.update_task is task:
                            state.update_task = None
                        continue
                    finally:
                        if state.update_task is task:
                            state.update_task = None
                    wrote = False
                    if state.status_enabled:
                        wrote = write_track_info(track_id_result)
                    if wrote:
                        await delete_all_messages(user_client, st.tg_channel_id)
                        try:
                            from lyrics.patterns import base as patterns_base
                        except ImportError:
                            patterns_base = None
                        try:
                            if (
                                patterns_base is not None
                                and callable(getattr(patterns_base, "apply_pending_reload", None))
                                and bool(patterns_base._PENDING_FEAT_RELOAD or patterns_base._PENDING_TAG_RELOAD)
                            ):
                                feat_applied, tag_applied = patterns_base.apply_pending_reload()
                                parts = []
                                if tag_applied:
                                    parts.append(PATTERNS_KIND_TAG_LABEL)
                                if feat_applied:
                                    parts.append(PATTERNS_KIND_FEAT_LABEL)
                                short = ", ".join(parts) if parts else PATTERNS_KIND_GENERIC_LABEL
                                _safe_log_info(INFO_PATTERNS_APPLIED, short)
                        except (OSError, ValueError, TypeError):
                            pass
                    if state.first_enable_log_needed:
                        _safe_log_info(INFO_STATUS_ENABLED)
                        state.first_enable_log_needed = False
                        if state.control_message_event:
                            try:
                                await state.control_message_event.edit(
                                    INFO_STATUS_ENABLED,
                                    buttons=[[Button.text(BTN_DISABLE, resize=True)], [Button.text(BTN_BACK, resize=True)]]
                                )
                            except (RPCError, ConnectionError, asyncio.TimeoutError, OSError):
                                try:
                                    await state.control_message_event.respond(
                                        INFO_STATUS_ENABLED,
                                        buttons=[[Button.text(BTN_DISABLE, resize=True)], [Button.text(BTN_BACK, resize=True)]]
                                    )
                                except (RPCError, ConnectionError, asyncio.TimeoutError, OSError):
                                    pass
                            state.control_message_event = None
                        state.enable_in_progress = False
            except (RPCError, ConnectionError, asyncio.TimeoutError, OSError, aiohttp.ClientError, RuntimeError, ValueError, TypeError) as e:
                _safe_log_error(ERROR_UPDATE_LOOP, e)
                try:
                    err_text = str(e)
                except (ValueError, TypeError, UnicodeError):
                    err_text = ""
                if isinstance(e, (ConnectionError, OSError)) or "Request was unsuccessful" in err_text:
                    sleep_seconds = max(sleep_seconds, 5)
                    try:
                        is_conn = getattr(user_client, "is_connected", None)
                        if callable(is_conn) and not is_conn():
                            await user_client.connect()
                    except (RPCError, ConnectionError, asyncio.TimeoutError, OSError):
                        pass
                    try:
                        is_conn = getattr(bot_client, "is_connected", None)
                        if callable(is_conn) and not is_conn():
                            await bot_client.connect()
                    except (RPCError, ConnectionError, asyncio.TimeoutError, OSError):
                        pass
            await asyncio.sleep(sleep_seconds)
    except asyncio.CancelledError:
        return
