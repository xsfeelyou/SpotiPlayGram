from __future__ import annotations
import os
import copy
import json
import asyncio
import re
import hashlib
from collections import OrderedDict
from typing import Optional, List, Dict, Any, Tuple, Literal, get_args, get_origin
from constants import (
    DIRS,
    ERROR_AUTH_MISTRAL_API_KEY_MISSING,
    ERROR_MISTRAL_API_KEY_INVALID,
    ERROR_MISTRAL_FINAL_SELECTION,
    ERROR_MISTRAL_MODEL_NOT_FOUND,
    ERROR_MISTRAL_REQUEST,
    ERROR_MISTRAL_SDK_MISSING,
    ERROR_MISTRAL_TIMEOUT,
    ERROR_UNEXPECTED,
    INFO_MISTRAL_FINAL_SELECTION_START,
    INFO_MISTRAL_MODEL_UPDATED,
    INFO_MISTRAL_SWITCH_MODEL,
    MISTRAL_TIMEOUT_SECONDS,
)
from logger import _safe_log_info, _safe_log_error, _safe_log_warning, dev_logger
from utils.mistral_client import (
    complete_mistral_chat,
    get_mistral_client_class,
    probe_mistral_chat_payload,
)
from config import Settings

MistralClient = get_mistral_client_class()

_MISTRAL_LOG_PATH = os.path.join(DIRS["LOGS"], "GENIUS_MISTRAL_FINAL_SELECTION.log")
_MISTRAL_MODELS_CACHE_PATH = os.path.join(DIRS["SESSION"], "mistral", "mistral_models_list.json")

_MISTRAL_FINAL_SELECTION_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "idx": {"type": "string"},
    },
    "required": ["idx"],
    "additionalProperties": False,
}

_MISTRAL_REASONING_PROBE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "ok": {"type": "string"},
    },
    "required": ["ok"],
    "additionalProperties": False,
}

_MISTRAL_CHOICE_CACHE_TTL_SECONDS: float = 3600.0
_MISTRAL_CHOICE_CACHE_MAX_SIZE: int = int(_MISTRAL_CHOICE_CACHE_TTL_SECONDS)
_MISTRAL_FINAL_SELECTION_MAX_TOKENS: int = 16
_MISTRAL_FINAL_SELECTION_REASONING_MAX_TOKENS: int = 1024
_MISTRAL_MODELS_CACHE_SCHEMA_VERSION: int = 2
_MISTRAL_MODEL_CAPABILITY_CONCURRENCY: int = 100
_MISTRAL_CHOICE_CACHE: "OrderedDict[str, Tuple[float, Optional[int], bool]]" = OrderedDict()
_MISTRAL_CHOICE_INFLIGHT: Dict[str, asyncio.Task] = {}

_models_cache_mem: Optional[Dict[str, Any]] = None
_models_cache_mem_mtime: Optional[float] = None

def _mistral_choice_cache_purge(now: float) -> None:
    while _MISTRAL_CHOICE_CACHE:
        k, v = next(iter(_MISTRAL_CHOICE_CACHE.items()))
        expires_at = v[0]
        if expires_at > now:
            break
        _MISTRAL_CHOICE_CACHE.popitem(last=False)

def _mistral_choice_cache_get(key: str, now: float) -> Optional[Tuple[Optional[int], bool]]:
    _mistral_choice_cache_purge(now)
    data = _MISTRAL_CHOICE_CACHE.get(key)
    if not data:
        return None
    expires_at, idx, explicit_none = data
    if expires_at <= now:
        _MISTRAL_CHOICE_CACHE.pop(key, None)
        return None
    return idx, explicit_none

def _mistral_choice_cache_set(key: str, now: float, idx: Optional[int], explicit_none: bool) -> None:
    _mistral_choice_cache_purge(now)
    expires_at = now + _MISTRAL_CHOICE_CACHE_TTL_SECONDS
    _MISTRAL_CHOICE_CACHE[key] = (expires_at, idx, bool(explicit_none))
    _MISTRAL_CHOICE_CACHE.move_to_end(key)
    while len(_MISTRAL_CHOICE_CACHE) > _MISTRAL_CHOICE_CACHE_MAX_SIZE:
        _MISTRAL_CHOICE_CACHE.popitem(last=False)

MistralChoiceSource = Literal["fresh", "cache", "inflight", "skipped"]

def build_mistral_payload_text(
    spotify_track_meta: Dict[str, str],
    genius_candidates_meta: List[Dict[str, Any]],
) -> Optional[str]:
    payload = {
        "spotify_track_meta": spotify_track_meta,
        "genius_candidates_meta": genius_candidates_meta,
    }
    try:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return None

def _format_mistral_payload_for_log(payload_json: str) -> str:
    try:
        parsed = json.loads(payload_json)
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except (TypeError, ValueError, json.JSONDecodeError):
        return payload_json

def build_mistral_choice_cache_key(
    chain: List[str],
    spotify_track_meta: Dict[str, str],
    genius_candidates_meta: List[Dict[str, Any]],
) -> Optional[str]:
    try:
        spotify_meta_norm: Dict[str, str] = {}
        if isinstance(spotify_track_meta, dict):
            for k, v in spotify_track_meta.items():
                spotify_meta_norm[str(k)] = "" if v is None else str(v)

        candidates_norm: List[Dict[str, str]] = []
        for c in (genius_candidates_meta or []):
            if not isinstance(c, dict):
                continue
            item: Dict[str, str] = {
                "idx": "" if c.get("idx") is None else str(c.get("idx")),
                "id": "" if (c.get("id") or c.get("song_id")) is None else str(c.get("id") or c.get("song_id")),
                "url": "" if c.get("url") is None else str(c.get("url")),
                "title": "" if c.get("title") is None else str(c.get("title")),
                "primary_artist": "" if c.get("primary_artist") is None else str(c.get("primary_artist")),
                "featured_artists": "" if c.get("featured_artists") is None else str(c.get("featured_artists")),
                "version_tag": "" if c.get("version_tag") is None else str(c.get("version_tag")),
            }
            candidates_norm.append(item)

        key_payload = {
            "models": [str(m) for m in (chain or [])],
            "spotify_track_meta": spotify_meta_norm,
            "genius_candidates_meta": candidates_norm,
        }
        key_text = json.dumps(key_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(key_text.encode("utf-8")).hexdigest()
    except (TypeError, ValueError, RuntimeError, UnicodeError):
        return None

_settings: Optional[Settings] = None

def set_settings(settings: Settings) -> None:
    global _settings
    _settings = settings

def get_settings() -> Settings:
    if _settings is None:
        raise RuntimeError("Settings are not initialized")
    return _settings

def _split_model_ids(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    text = str(value)
    return [m.strip() for m in re.split(r"[\s,]+", text) if m.strip()]

def _parse_model_spec_token(value: Any) -> Tuple[str, Optional[str]]:
    if isinstance(value, dict):
        model = str(value.get("id") or value.get("model") or "").strip()
        mode_raw = value.get("reasoning_effort")
        if mode_raw is None:
            mode_raw = value.get("reasoning_mode")
        mode = str(mode_raw).strip().lower() if mode_raw is not None else ""
        return model, mode or None
    text = str(value).strip()
    if "=" not in text:
        return text, None
    model, mode = text.split("=", 1)
    return model.strip(), (mode.strip().lower() or None)

def _format_model_spec(model_id: str, reasoning_mode: Optional[str] = None) -> str:
    model = str(model_id or "").strip()
    mode = str(reasoning_mode or "").strip().lower()
    if model and mode:
        return f"{model}={mode}"
    return model

def _split_model_specs(value: Any) -> List[Tuple[str, Optional[str]]]:
    if value is None:
        return []
    raw_items: List[Any]
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = _split_model_ids(value)
    specs: List[Tuple[str, Optional[str]]] = []
    for item in raw_items:
        model, mode = _parse_model_spec_token(item)
        if model:
            specs.append((model, mode))
    return specs

def _split_model_spec_strings(value: Any) -> List[str]:
    return [_format_model_spec(model, mode) for model, mode in _split_model_specs(value)]

def _get_model_info_from_models(models: Any, model_id: str) -> Optional[Dict[str, Any]]:
    model, _ = _parse_model_spec_token(model_id)
    if not isinstance(models, dict):
        return None
    if model in models and isinstance(models[model], dict):
        return models[model]
    for info in models.values():
        if not isinstance(info, dict):
            continue
        aliases = info.get("aliases", [])
        if isinstance(aliases, list) and model in aliases:
            return info
    return None

def _normalize_reasoning_effort_values(raw_modes: Any) -> List[str]:
    if not isinstance(raw_modes, list):
        return []
    modes: List[str] = []
    seen = set()
    for raw in raw_modes:
        mode = str(raw).strip().lower()
        if mode and mode not in seen:
            seen.add(mode)
            modes.append(mode)
    return modes

def _merge_reasoning_effort_values(*raw_values: Any) -> List[str]:
    modes: List[str] = []
    seen = set()
    for raw in raw_values:
        for mode in _normalize_reasoning_effort_values(raw):
            if mode not in seen:
                seen.add(mode)
                modes.append(mode)
    return modes

def _get_model_reasoning_modes_from_info(info: Optional[Dict[str, Any]]) -> List[str]:
    if not info:
        return []
    return _merge_reasoning_effort_values(info.get("reasoning_efforts"), info.get("reasoning_modes"))

def _merge_model_capabilities_with_known(
    capabilities: Dict[str, Any],
    previous_info: Optional[Dict[str, Any]],
    known_reasoning: Optional[bool],
    known_efforts: List[str],
) -> Dict[str, Any]:
    merged = dict(capabilities)
    previous_efforts = []
    if known_reasoning is not False or known_efforts:
        previous_efforts = _get_model_reasoning_modes_from_info(previous_info)
    efforts = _merge_reasoning_effort_values(
        merged.get("reasoning_efforts"),
        merged.get("reasoning_modes"),
        known_efforts,
        previous_efforts,
    )
    active_efforts = any(mode != "none" for mode in efforts)
    if efforts:
        merged["reasoning_efforts"] = efforts
    else:
        merged["reasoning_efforts"] = []
    if active_efforts:
        merged["reasoning"] = True
        merged["reasoning_type"] = "adjustable"
    elif known_reasoning is True or merged.get("reasoning") is True:
        merged["reasoning"] = True
        reasoning_type = str(merged.get("reasoning_type") or "").strip().lower()
        if not reasoning_type or reasoning_type == "none":
            merged["reasoning_type"] = "native"
    elif known_reasoning is False:
        merged["reasoning"] = bool(merged.get("reasoning", False))
    return merged

def _sanitize_model_specs_with_cache(value: Any, cache: Dict[str, Any]) -> List[str]:
    models = cache.get("available_models", {}) if isinstance(cache, dict) else {}
    specs: List[str] = []
    seen = set()
    for model, mode in _split_model_specs(value):
        info = _get_model_info_from_models(models, model)
        valid_modes = _get_model_reasoning_modes_from_info(info)
        normalized_mode = mode if mode and mode in valid_modes else None
        spec = _format_model_spec(model, normalized_mode)
        if spec and spec not in seen:
            seen.add(spec)
            specs.append(spec)
    return specs

def _is_genius_detailed_log() -> bool:
    st = get_settings()
    return bool(st.genius_detailed_log)

def _write_mistral_log(
    payload_json: str,
    response_text: str,
    model: str,
    usage: Optional[Dict[str, int]] = None,
    model_type: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    prompt_mode: Optional[str] = None,
) -> None:
    if not _is_genius_detailed_log():
        return
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("GENIUS MISTRAL FINAL SELECTION LOG")
    lines.append("=" * 80)
    lines.append("")

    lines.append(f"Model: {model}")
    if model_type:
        type_data = {"model_type": model_type}
        if reasoning_effort:
            type_data["reasoning_effort"] = reasoning_effort
        if prompt_mode:
            type_data["prompt_mode"] = prompt_mode
        type_json = json.dumps(type_data, ensure_ascii=False)
        lines.append(type_json)
    lines.append("")
    lines.append("-" * 40)
    lines.append("SENT TO MISTRAL")
    lines.append("-" * 40)
    lines.append("")
    lines.append(_format_mistral_payload_for_log(payload_json))
    lines.append("")
    lines.append("-" * 40)
    lines.append("RECEIVED FROM MISTRAL")
    lines.append("-" * 40)
    lines.append("")
    lines.append(response_text)
    lines.append("")
    lines.append("-" * 40)
    lines.append("TOKEN USAGE")
    lines.append("-" * 40)
    lines.append("")
    if usage:
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
        lines.append(f"Prompt: {prompt_tokens}")
        lines.append(f"Completion: {completion_tokens}")
        lines.append(f"Total: {total_tokens}")
    else:
        lines.append("No data")
    lines.append("")
    lines.append("=" * 80)

    try:
        os.makedirs(DIRS["LOGS"], exist_ok=True)
        tmp_path = _MISTRAL_LOG_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        os.replace(tmp_path, _MISTRAL_LOG_PATH)
    except OSError:
        return

def read_mistral_model() -> str:
    cache = _read_models_cache()
    models = _split_model_spec_strings(cache.get("selected_models"))
    if models:
        return ", ".join(models)
    st = get_settings()
    models = _split_model_spec_strings(st.mistral_model)
    if models:
        return ", ".join(models)
    return st.mistral_model

def write_mistral_model(new_model: str) -> bool:
    cache = _read_models_cache()
    nm = (new_model or "").strip()
    if not nm:
        return False
    new_list = _split_model_spec_strings(nm)
    if not new_list:
        return False
    cur_val = cache.get("selected_models", [])
    cur_list = _split_model_spec_strings(cur_val)

    if new_list == cur_list:
        return True

    cache["selected_models"] = new_list
    if not _write_models_cache(cache):
        return False
    _safe_log_info(INFO_MISTRAL_MODEL_UPDATED, ", ".join(new_list))
    return True

def get_mistral_models_chain() -> List[str]:
    cache = _read_models_cache()
    val = cache.get("selected_models")
    chain: List[str] = []
    parts = _split_model_spec_strings(val)
    seen = set()
    for p in parts:
        if p not in seen:
            seen.add(p)
            chain.append(p)

    if not chain:
        st = get_settings()
        chain = _split_model_spec_strings(st.mistral_model)
        if not chain and st.mistral_model:
            chain = [st.mistral_model]
    return chain

def _read_models_cache() -> Dict[str, Any]:
    global _models_cache_mem, _models_cache_mem_mtime
    try:
        mtime = os.path.getmtime(_MISTRAL_MODELS_CACHE_PATH)
    except OSError:
        _models_cache_mem = None
        _models_cache_mem_mtime = None
        return {}
    if _models_cache_mem is None or _models_cache_mem_mtime != mtime:
        try:
            with open(_MISTRAL_MODELS_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            _models_cache_mem = None
            _models_cache_mem_mtime = mtime
            return {}
        if not isinstance(data, dict):
            _models_cache_mem = None
            _models_cache_mem_mtime = mtime
            return {}
        _models_cache_mem = data
        _models_cache_mem_mtime = mtime
    return copy.deepcopy(_models_cache_mem)

def _write_models_cache(data: Dict[str, Any]) -> bool:
    global _models_cache_mem, _models_cache_mem_mtime
    try:
        parent = os.path.dirname(_MISTRAL_MODELS_CACHE_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = _MISTRAL_MODELS_CACHE_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, _MISTRAL_MODELS_CACHE_PATH)
    except OSError as e:
        _safe_log_error(ERROR_UNEXPECTED, e, logger=dev_logger)
        return False
    try:
        _models_cache_mem = copy.deepcopy(data) if isinstance(data, dict) else None
        _models_cache_mem_mtime = os.path.getmtime(_MISTRAL_MODELS_CACHE_PATH)
    except OSError:
        _models_cache_mem = None
        _models_cache_mem_mtime = None
    return True

def _coerce_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        chunk_type = str(value.get("type") or "").strip().lower()
        if chunk_type == "thinking":
            return ""
        t = value.get("text") or value.get("content") or value.get("data")
        return "" if t is None else str(t)
    if isinstance(value, list):
        parts: List[str] = []
        for el in value:
            if isinstance(el, str):
                parts.append(el)
            elif isinstance(el, dict):
                chunk_type = str(el.get("type") or "").strip().lower()
                if chunk_type == "thinking":
                    continue
                t = el.get("text") or el.get("content") or el.get("data")
                if t is not None:
                    parts.append(str(t))
            else:
                chunk_type = str(getattr(el, "type", "") or "").strip().lower()
                if chunk_type == "thinking":
                    continue
                t = getattr(el, "text", None)
                if t is None:
                    t = getattr(el, "content", None)
                if t is None:
                    t = getattr(el, "data", None)
                if t is not None:
                    parts.append(str(t))
                elif not chunk_type:
                    parts.append(str(el))
        return "".join(parts)
    chunk_type = str(getattr(value, "type", "") or "").strip().lower()
    if chunk_type == "thinking":
        return ""
    t = getattr(value, "text", None)
    if t is None:
        t = getattr(value, "content", None)
    if t is None:
        t = getattr(value, "data", None)
    if t is not None:
        return str(t)
    return str(value)

def _load_mistral_json_object(out_text: str) -> Any:
    try:
        return json.loads(out_text)
    except json.JSONDecodeError as original_error:
        decoder = json.JSONDecoder()
        parsed_idx_object: Optional[Dict[str, Any]] = None
        for pos, ch in enumerate(out_text):
            if ch != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(out_text[pos:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and "idx" in candidate:
                parsed_idx_object = candidate
        if parsed_idx_object is not None:
            return parsed_idx_object
        raise original_error

def _extract_literal_strings(type_obj: Any) -> List[str]:
    values: List[str] = []
    origin = get_origin(type_obj)
    if origin is Literal:
        for arg in get_args(type_obj):
            if isinstance(arg, str):
                values.append(arg)
        return values
    for arg in get_args(type_obj):
        values.extend(_extract_literal_strings(arg))
    return values

def get_available_reasoning_efforts() -> List[str]:
    reasoning_effort_type: Any = None
    try:
        from mistralai.client.models.reasoningeffort import ReasoningEffort
        reasoning_effort_type = ReasoningEffort
    except ImportError:
        try:
            from mistralai.models.reasoningeffort import ReasoningEffort
            reasoning_effort_type = ReasoningEffort
        except ImportError:
            reasoning_effort_type = None
    if reasoning_effort_type is None:
        return []
    values: List[str] = []
    seen = set()
    for value in _extract_literal_strings(reasoning_effort_type):
        normalized = value.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            values.append(normalized)
    return values

def _extract_mistral_content_text(resp: Any) -> str:
    raw: Any = None
    if isinstance(resp, dict):
        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            if isinstance(first_choice, dict):
                msg = first_choice.get("message") or {}
                if isinstance(msg, dict):
                    raw = msg.get("content")
    choices = getattr(resp, "choices", None)
    if raw is None and isinstance(choices, list) and choices:
        msg = getattr(choices[0], "message", None)
        raw = getattr(msg, "content", None) if msg is not None else None
    if raw is None:
        return ""
    return _coerce_to_text(raw).strip()

def _format_mistral_error_message(value: Any, max_len: int = 160) -> str:
    if isinstance(value, BaseException):
        text = str(value).strip() or value.__class__.__name__
    else:
        text = "" if value is None else str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split())
    if not text:
        text = "mistral request failed"
    if len(text) > max_len:
        text = text[:max_len - 3].rstrip() + "..."
    return text

def _mistral_response_has_thinking(resp: Any) -> bool:
    def walk(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, dict):
            if str(value.get("type") or "").strip().lower() == "thinking":
                return True
            return any(walk(v) for v in value.values())
        if isinstance(value, list):
            return any(walk(v) for v in value)
        if str(getattr(value, "type", "") or "").strip().lower() == "thinking":
            return True
        thinking = getattr(value, "thinking", None)
        if thinking is not None:
            return True
        return False

    return walk(resp)

def _parse_mistral_idx(out_text: str) -> tuple[Optional[int], bool, Optional[str]]:
    if not out_text:
        return None, False, "empty response"
    try:
        parsed = _load_mistral_json_object(out_text)
    except json.JSONDecodeError:
        return None, False, "invalid content: not json"
    if not isinstance(parsed, dict):
        return None, False, "invalid content: root is not object"
    idx_val = parsed.get("idx")
    if isinstance(idx_val, int) and not isinstance(idx_val, bool):
        if idx_val <= 0:
            return None, False, "invalid content: idx must be >= 1"
        return idx_val, False, None
    if not isinstance(idx_val, str) or not idx_val.strip():
        return None, False, "invalid content: idx is missing or not string"
    idx_str = idx_val.strip()
    if idx_str.lower() == "none":
        return None, True, None
    if not idx_str.isdigit():
        return None, False, "invalid content: idx is not numeric string"
    idx_int = int(idx_str)
    if idx_int <= 0:
        return None, False, "invalid content: idx must be >= 1"
    return idx_int, False, None

def _log_mistral_exception(message: str, model: str) -> None:
    message = _format_mistral_error_message(message)
    low = message.lower()
    if "401" in low or "unauthorized" in low or "forbidden" in low or "invalid api key" in low:
        _safe_log_error(ERROR_MISTRAL_API_KEY_INVALID)
    elif "404" in low or ("model" in low and "not found" in low):
        _safe_log_error(ERROR_MISTRAL_MODEL_NOT_FOUND, model)
    else:
        _safe_log_error(ERROR_MISTRAL_REQUEST, message)
    _safe_log_error(ERROR_MISTRAL_FINAL_SELECTION, message)

def _get_cached_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    cache = _read_models_cache()
    models = cache.get("available_models", {})
    return _get_model_info_from_models(models, model_id)

def is_model_reasoning(model_id: str) -> bool:
    info = _get_cached_model_info(model_id)
    if not info:
        return False
    modes = info.get("reasoning_efforts") or info.get("reasoning_modes") or []
    if isinstance(modes, list):
        for raw_mode in modes:
            mode = str(raw_mode).strip().lower()
            if mode and mode != "none":
                return True
    return bool(info.get("reasoning", False) or info.get("prompt_mode_reasoning", False))

def get_model_reasoning_type(model_id: str) -> str:
    info = _get_cached_model_info(model_id)
    if not info:
        return "none"
    reasoning_type = str(info.get("reasoning_type") or "").strip().lower()
    if reasoning_type:
        return reasoning_type
    modes = _get_model_reasoning_modes_from_info(info)
    if any(mode != "none" for mode in modes):
        return "adjustable"
    if info.get("prompt_mode_reasoning", False):
        return "prompt_mode"
    if info.get("reasoning", False):
        return "native"
    return "none"

def get_model_reasoning_modes(model_id: str) -> List[str]:
    info = _get_cached_model_info(model_id)
    return _get_model_reasoning_modes_from_info(info)

def is_model_reasoning_mode_valid(model_id: str, reasoning_mode: Optional[str]) -> bool:
    mode = str(reasoning_mode or "").strip().lower()
    if not mode:
        return True
    modes = get_model_reasoning_modes(model_id)
    return mode in modes

async def _probe_mistral_chat_payload(
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_seconds: Optional[float],
    max_retries: int,
) -> Tuple[bool, str, bool]:
    ok, data = await probe_mistral_chat_payload(
        headers,
        payload,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    if not ok:
        return False, "", False
    return True, _extract_mistral_content_text(data), _mistral_response_has_thinking(data)

def _is_probe_json_valid(out_text: str) -> bool:
    if not out_text:
        return False
    try:
        parsed = _load_mistral_json_object(out_text)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict)

async def check_model_reasoning_capabilities(
    model_id: str,
    api_key: str,
    max_retries: int = 3,
    known_reasoning: Optional[bool] = None,
    known_efforts: Optional[List[str]] = None,
    timeout_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json", "Content-Type": "application/json"}
    efforts: List[str] = []
    default_has_thinking = False
    prompt_mode_reasoning = False
    effort_candidates: List[str] = []
    seen_effort_candidates = set()
    raw_effort_candidates: List[str] = list(known_efforts or [])
    if known_reasoning is not False:
        raw_effort_candidates.extend(get_available_reasoning_efforts())
    for raw_effort in raw_effort_candidates:
        effort = str(raw_effort).strip().lower()
        if effort and effort not in seen_effort_candidates:
            seen_effort_candidates.add(effort)
            effort_candidates.append(effort)
    base_payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "Return only a JSON object."},
            {"role": "user", "content": "Return {\"ok\":\"yes\"}."},
        ],
        "max_tokens": _MISTRAL_FINAL_SELECTION_REASONING_MAX_TOKENS,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "reasoning_probe",
                "schema": _MISTRAL_REASONING_PROBE_JSON_SCHEMA,
            },
        },
    }

    async def probe_effort(effort: str) -> Optional[str]:
        payload = dict(base_payload)
        payload["reasoning_effort"] = effort
        ok, out_text, _ = await _probe_mistral_chat_payload(headers, payload, timeout_seconds, max_retries)
        if ok and _is_probe_json_valid(out_text):
            return effort
        return None

    async def probe_default() -> bool:
        ok, out_text, has_thinking = await _probe_mistral_chat_payload(
            headers, dict(base_payload), timeout_seconds, max_retries
        )
        return bool(ok and _is_probe_json_valid(out_text) and has_thinking)

    async def probe_prompt_mode() -> bool:
        payload = dict(base_payload)
        payload["prompt_mode"] = "reasoning"
        ok, out_text, _ = await _probe_mistral_chat_payload(headers, payload, timeout_seconds, max_retries)
        return bool(ok and _is_probe_json_valid(out_text))

    if known_reasoning is not False:
        default_has_thinking = await probe_default()
    if default_has_thinking:
        return {
            "reasoning": True,
            "reasoning_type": "native",
            "reasoning_efforts": [],
            "default_has_thinking": True,
            "prompt_mode_reasoning": False,
        }
    probe_results: List[Any] = []
    if known_reasoning is not False:
        probe_results = await asyncio.gather(
            *(probe_effort(effort) for effort in effort_candidates),
            probe_prompt_mode(),
        )
        efforts = [effort for effort in probe_results[:-1] if effort]
    has_active_reasoning_effort = any(effort != "none" for effort in efforts)
    if probe_results and not has_active_reasoning_effort:
        prompt_mode_reasoning = bool(probe_results[-1])
    if has_active_reasoning_effort:
        reasoning_type = "adjustable"
    elif known_reasoning is True:
        reasoning_type = "native"
    elif prompt_mode_reasoning:
        reasoning_type = "prompt_mode"
    else:
        reasoning_type = "none"
    return {
        "reasoning": reasoning_type != "none",
        "reasoning_type": reasoning_type,
        "reasoning_efforts": efforts,
        "default_has_thinking": default_has_thinking,
        "prompt_mode_reasoning": prompt_mode_reasoning,
    }

async def update_models_cache(
    models_data: List[Dict[str, Any]],
    api_key: str,
    force_recheck: bool = False,
    max_retries: int = 3,
    timeout_seconds: Optional[float] = None,
    concurrency: int = _MISTRAL_MODEL_CAPABILITY_CONCURRENCY,
    partial_update: bool = False,
) -> Dict[str, Any]:
    full_cache = _read_models_cache()
    existing_selected = _split_model_spec_strings(full_cache.get("selected_models", []))
    full_cached_models_raw = full_cache.get("available_models", {})
    full_cached_models = full_cached_models_raw if isinstance(full_cached_models_raw, dict) else {}

    cache = full_cache if not force_recheck else {}
    cached_models_raw = cache.get("available_models", {})
    cached_models = cached_models_raw if isinstance(cached_models_raw, dict) else {}

    new_models: Dict[str, Any] = {}
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    async def build_model_cache_item(item: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
        model_id = item.get("id")
        if not model_id:
            return None

        aliases = item.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []
        known_reasoning_raw = item.get("reasoning")
        known_reasoning = known_reasoning_raw if isinstance(known_reasoning_raw, bool) else None
        known_efforts_raw = item.get("reasoning_efforts") or item.get("reasoning_modes") or []
        current_known_efforts = _normalize_reasoning_effort_values(known_efforts_raw)
        known_efforts = current_known_efforts
        previous_info = _get_model_info_from_models(full_cached_models, model_id)
        if previous_info:
            previous_reasoning_raw = previous_info.get("reasoning")
            if known_reasoning is None and isinstance(previous_reasoning_raw, bool):
                known_reasoning = previous_reasoning_raw
            if known_reasoning is not False or current_known_efforts:
                known_efforts = _merge_reasoning_effort_values(
                    known_efforts,
                    previous_info.get("reasoning_efforts"),
                    previous_info.get("reasoning_modes"),
                )
        cached_entry = cached_models.get(model_id)
        if not isinstance(cached_entry, dict):
            cached_entry = previous_info if isinstance(previous_info, dict) else None
        async with semaphore:
            if cached_entry is not None and not force_recheck:
                cached_info = dict(cached_entry)
                cached_info["aliases"] = aliases
                if "reasoning_efforts" not in cached_info and "reasoning_modes" in cached_info:
                    cached_info["reasoning_efforts"] = cached_info.get("reasoning_modes", [])
                cached_info["reasoning_efforts"] = _merge_reasoning_effort_values(
                    cached_info.get("reasoning_efforts"),
                    cached_info.get("reasoning_modes"),
                    known_efforts,
                )
                needs_capabilities_check = (
                    "reasoning_type" not in cached_info
                    or "prompt_mode_reasoning" not in cached_info
                    or (known_reasoning is True and str(cached_info.get("reasoning_type") or "") == "none")
                )
                if needs_capabilities_check:
                    capabilities = await check_model_reasoning_capabilities(
                        model_id,
                        api_key,
                        max_retries=max_retries,
                        known_reasoning=known_reasoning,
                        known_efforts=known_efforts,
                        timeout_seconds=timeout_seconds,
                    )
                    cached_info.update(_merge_model_capabilities_with_known(
                        capabilities,
                        cached_info,
                        known_reasoning,
                        known_efforts,
                    ))
                elif any(mode != "none" for mode in cached_info["reasoning_efforts"]):
                    cached_info["reasoning"] = True
                    cached_info["reasoning_type"] = "adjustable"
                elif known_reasoning is True:
                    cached_info["reasoning"] = True
                return model_id, cached_info
            capabilities = await check_model_reasoning_capabilities(
                model_id,
                api_key,
                max_retries=max_retries,
                known_reasoning=known_reasoning,
                known_efforts=known_efforts,
                timeout_seconds=timeout_seconds,
            )
            capabilities = _merge_model_capabilities_with_known(
                capabilities,
                previous_info,
                known_reasoning,
                known_efforts,
            )
            return model_id, {
                "aliases": aliases,
                **capabilities,
            }

    results = await asyncio.gather(*(build_model_cache_item(item) for item in models_data))
    for result in results:
        if result is None:
            continue
        model_id, model_info = result
        new_models[model_id] = model_info
    if partial_update and not force_recheck:
        merged_models = dict(full_cached_models)
        merged_models.update(new_models)
        new_models = merged_models
    if not existing_selected:
        st = get_settings()
        existing_selected = _split_model_spec_strings(st.mistral_model)
    temp_cache = {"available_models": new_models}
    final_selected = _sanitize_model_specs_with_cache(existing_selected, temp_cache)
    new_cache = {
        "schema_version": _MISTRAL_MODELS_CACHE_SCHEMA_VERSION,
        "selected_models": final_selected,
        "available_models": new_models,
    }
    _write_models_cache(new_cache)

    return new_cache

async def ensure_selected_models_cached(api_key: str) -> None:
    st = get_settings()
    if not st.enable_mistral:
        return
    current_models_str = read_mistral_model()
    cache = _read_models_cache()
    cached_models = cache.get("available_models", {})
    cache_schema_outdated = cache.get("schema_version") != _MISTRAL_MODELS_CACHE_SCHEMA_VERSION
    models_list = _split_model_specs(current_models_str)
    models_to_check: List[Tuple[str, str]] = []
    seen_check_keys = set()
    for model_id, _ in models_list:
        cache_key: Optional[str] = model_id if model_id in cached_models else None
        if cache_key is None:
            for mid, info in cached_models.items():
                if not isinstance(info, dict):
                    continue
                aliases = info.get("aliases", [])
                if isinstance(aliases, list) and model_id in aliases:
                    cache_key = mid
                    break

        if cache_key is None:
            cache_key = model_id
            needs_check = True
        else:
            info = cached_models.get(cache_key)
            needs_check = (
                cache_schema_outdated
                or
                not isinstance(info, dict)
                or "reasoning_type" not in info
                or ("reasoning_efforts" not in info and "reasoning_modes" not in info)
                or "prompt_mode_reasoning" not in info
            )

        if needs_check and cache_key not in seen_check_keys:
            seen_check_keys.add(cache_key)
            models_to_check.append((cache_key, model_id))

    if models_to_check:
        for cache_key, model_id in models_to_check:
            cached_info = cached_models.get(cache_key)
            aliases = []
            known_reasoning: Optional[bool] = None
            known_efforts: List[str] = []
            if isinstance(cached_info, dict):
                aliases_raw = cached_info.get("aliases", [])
                if isinstance(aliases_raw, list):
                    aliases = aliases_raw
                if cache_key != model_id and model_id not in aliases:
                    aliases.append(model_id)
                known_reasoning_raw = cached_info.get("reasoning")
                if isinstance(known_reasoning_raw, bool):
                    known_reasoning = known_reasoning_raw
                known_efforts_raw = cached_info.get("reasoning_efforts") or cached_info.get("reasoning_modes") or []
                known_efforts = _normalize_reasoning_effort_values(known_efforts_raw)
            capabilities = await check_model_reasoning_capabilities(
                model_id,
                api_key,
                known_reasoning=known_reasoning,
                known_efforts=known_efforts,
            )
            capabilities = _merge_model_capabilities_with_known(
                capabilities,
                cached_info if isinstance(cached_info, dict) else None,
                known_reasoning,
                known_efforts,
            )
            cached_models[cache_key] = {
                "aliases": aliases,
                **capabilities,
            }
        cache["available_models"] = cached_models
    if models_to_check or cache_schema_outdated:
        cache["schema_version"] = _MISTRAL_MODELS_CACHE_SCHEMA_VERSION
        cache["available_models"] = cached_models
        sanitized_selected = _sanitize_model_specs_with_cache(cache.get("selected_models", []), cache)
        if sanitized_selected:
            cache["selected_models"] = sanitized_selected
        if not _write_models_cache(cache):
            _safe_log_error(ERROR_UNEXPECTED, "failed to save selected models", logger=dev_logger)

async def choose_genius_candidate_with_mistral(
    spotify_track_meta: Dict[str, str],
    genius_candidates_meta: List[Dict[str, Any]],
    *,
    cache_key: Optional[str] = None,
    payload_text: Optional[str] = None,
) -> Tuple[Optional[int], bool, MistralChoiceSource]:
    if not genius_candidates_meta:
        return None, False, "skipped"

    st = get_settings()
    api_key = st.mistral_api_key
    if MistralClient is None:
        _safe_log_error(ERROR_MISTRAL_SDK_MISSING)
        return None, False, "skipped"
    if not api_key:
        _safe_log_error(ERROR_AUTH_MISTRAL_API_KEY_MISSING)
        return None, False, "skipped"

    chain = get_mistral_models_chain()
    if cache_key is None:
        cache_key = build_mistral_choice_cache_key(chain, spotify_track_meta, genius_candidates_meta)
    if cache_key:
        now = asyncio.get_running_loop().time()
        cached = _mistral_choice_cache_get(cache_key, now)
        if cached is not None:
            idx_cached, explicit_none_cached = cached
            if explicit_none_cached:
                return None, True, "cache"
            return idx_cached, False, "cache"

        inflight = _MISTRAL_CHOICE_INFLIGHT.get(cache_key)
        if inflight is not None:
            try:
                idx_inflight, explicit_none_inflight = await inflight
                return idx_inflight, explicit_none_inflight, "inflight"
            except asyncio.CancelledError:
                raise
            except Exception as e:
                _safe_log_error(ERROR_UNEXPECTED, e, logger=dev_logger)

    if payload_text is None:
        payload = {
            "spotify_track_meta": spotify_track_meta,
            "genius_candidates_meta": genius_candidates_meta,
        }

        try:
            payload_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            _safe_log_error(ERROR_MISTRAL_FINAL_SELECTION, "failed to serialize payload")
            return None, False, "skipped"

    sys_content = (
        "You are a strict selector of the best Genius lyrics URL for a Spotify track.\n"
        "You receive a JSON object with two keys: \"spotify_track_meta\" and \"genius_candidates_meta\".\n"
        "\"spotify_track_meta\" describes the target Spotify track. "
        "\"genius_candidates_meta\" is an array of candidate Genius songs. "
        "Each candidate already has a ranking index field \"idx\" (string).\n"
        "Your task is to decide which candidate (by its existing idx) best matches the Spotify track, "
        "based ONLY on this metadata. Do not invent any new ids or modify the input JSON.\n"
        "\n"
        "Decision rules (VERY IMPORTANT):\n"
        "1) The main matching signal is the visual similarity between Spotify and Genius in these fields:\n"
        "   - primary_artist\n"
        "   - featured_artists\n"
        "   - title\n"
        "   - version_tag\n"
        "   Treat strings as visually similar if, after lowercasing and removing punctuation/spaces, "
        "they mostly look the same, even with small typos, doubled/missing letters or minor spelling variants.\n"
        "   Also treat cross-script transliterations as strong matches: if one side is written in a non-Latin script "
        "and the other side is an ASCII/Latin transliteration of the same text, consider them equivalent even when the "
        "spelling is not identical letter-by-letter. This includes artist/title/version text and ASCII/Latin slug forms in URLs.\n"
        "2) If spotify_track_meta.featured_artists is non-empty, treat overlap between these names and "
        "candidate.featured_artists as a VERY STRONG signal. When several candidates share the same primary_artist "
        "and base title, you MUST prefer the candidate whose featured_artists visually match the Spotify "
        "featured_artists over candidates without such artists, even if its title contains additional version "
        "descriptors.\n"
        "3) Compare version_tag as a SOFT version discriminator (a weak signal used mainly for tie-breaking). "
        "version_tag should matter slightly MORE than album/release_date, but it should matter LESS than artist/title/featured_artists.\n"
        "   - If spotify_track_meta.version_tag is non-empty AND at least one candidate.version_tag visually matches it, "
        "you SHOULD prefer those matching candidates over non-matching ones.\n"
        "   - If spotify_track_meta.version_tag is non-empty BUT no candidates match it (or candidates have empty/missing version_tag), "
        "DO NOT reject otherwise good matches and DO NOT output idx=\"none\" just because version_tag is missing or different.\n"
        "   - If spotify_track_meta.version_tag is empty, candidates with empty/absent version_tag are preferred when everything else is similar; "
        "do not let a non-empty version_tag win only because album/release_date look closer.\n"
        "4) You do NOT need exact equality of primary_artist / featured_artists / title / version_tag between Spotify and Genius. "
        "It is enough that, to a human reader, these fields clearly refer to the same song.\n"
        "5) When you are unsure / candidates seem similarly good, use the candidate url as an additional supporting signal:\n"
        "   - If candidate.url visually contains the same primary_artist, featured_artists (if any), title, and version_tag (if any), "
        "including transliterated ASCII/Latin slug forms of the Spotify text, "
        "treat this as a strong confirmation.\n"
        "6) Use album and release_date ONLY as LAST-RESORT supporting signals (they are often missing or inconsistent between Spotify and Genius).\n"
        "   - An exact release_date match can be a mild confirmation, but it is NOT decisive by itself.\n"
        "   - Never prefer a candidate with a worse version_tag match solely because album/release_date look closer.\n"
        "7) You must return idx=\"none\" ONLY when all candidates clearly refer to different songs, that is, "
        "both artist names and titles obviously do not correspond to the Spotify track even approximately.\n"
        "   - If at least one candidate has a reasonably matching primary_artist AND title (according to the flexible rules "
        "above), you MUST return the idx of the best such candidate and you MUST NOT output idx=\"none\".\n"
        "\n"
        "You MUST answer ONLY with a JSON object of the form {\"idx\": \"...\"} that satisfies the json_schema.\n"
        "If there is a clearly best match, return its existing idx value as a string.\n"
        "If none of the candidates are suitable, return {\"idx\": \"none\"}.\n"
        "Do not add explanations, comments, or extra fields."
    )

    user_content = (
        "Use the following JSON as the ONLY source of truth for your decision.\n"
        "Decide which candidate idx from \"genius_candidates_meta\" best matches \"spotify_track_meta\".\n"
        "Output idx=\"none\" ONLY if no candidate plausibly matches the same song by primary_artist + title.\n"
        "\n"
        "Input JSON:\n"
        f"{payload_text}\n"
    )

    async def _run_selection() -> Tuple[Optional[int], bool]:
        start_logged = False
        for idx_model, model_spec in enumerate(chain):
            model, selected_reasoning_effort = _parse_model_spec_token(model_spec)
            if not model:
                continue
            messages = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": user_content},
            ]
            model_is_reasoning = is_model_reasoning(model)
            model_reasoning_type = get_model_reasoning_type(model)
            prompt_mode_value: Optional[str] = None
            model_reasoning_modes = get_model_reasoning_modes(model)
            if selected_reasoning_effort and selected_reasoning_effort not in model_reasoning_modes:
                selected_reasoning_effort = None
            chat_kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "stream": False,
                "max_tokens": _MISTRAL_FINAL_SELECTION_MAX_TOKENS,
                "temperature": 0,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "genius_final_choice",
                        "schema": _MISTRAL_FINAL_SELECTION_JSON_SCHEMA,
                    },
                },
            }
            if selected_reasoning_effort:
                chat_kwargs["reasoning_effort"] = selected_reasoning_effort
            elif model_reasoning_type == "prompt_mode":
                prompt_mode_value = "reasoning"
            if prompt_mode_value:
                chat_kwargs["prompt_mode"] = "reasoning"
            if selected_reasoning_effort:
                current_model_type = "Reasoning (reasoning_effort)"
            elif prompt_mode_value:
                current_model_type = "Reasoning (prompt_mode)"
            elif model_reasoning_type == "adjustable":
                current_model_type = "Reasoning (adjustable, default)"
            elif model_reasoning_type == "native" or model_is_reasoning:
                current_model_type = "Reasoning (native)"
            else:
                current_model_type = "Non Reasoning"
            if current_model_type != "Non Reasoning":
                chat_kwargs["max_tokens"] = _MISTRAL_FINAL_SELECTION_REASONING_MAX_TOKENS

            if not start_logged:
                _safe_log_info(INFO_MISTRAL_FINAL_SELECTION_START, _format_model_spec(model, selected_reasoning_effort))
                start_logged = True

            try:
                resp = await complete_mistral_chat(
                    api_key,
                    chat_kwargs,
                    MISTRAL_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                _safe_log_error(ERROR_MISTRAL_TIMEOUT, model, MISTRAL_TIMEOUT_SECONDS)
                resp = None
            except asyncio.CancelledError:
                raise
            except Exception as e:
                msg = _format_mistral_error_message(e)
                _log_mistral_exception(msg, model)
                resp = None

            if resp is not None:
                out_text = _extract_mistral_content_text(resp)
                if not out_text and current_model_type == "Non Reasoning":
                    retry_kwargs = dict(chat_kwargs)
                    retry_kwargs["max_tokens"] = _MISTRAL_FINAL_SELECTION_REASONING_MAX_TOKENS
                    try:
                        retry_resp = await complete_mistral_chat(
                            api_key,
                            retry_kwargs,
                            MISTRAL_TIMEOUT_SECONDS,
                        )
                    except asyncio.TimeoutError:
                        retry_resp = None
                    except Exception as e:
                        _safe_log_warning(ERROR_MISTRAL_REQUEST, _format_mistral_error_message(e), logger=dev_logger)
                        retry_resp = None
                    if retry_resp is not None:
                        resp = retry_resp
                        out_text = _extract_mistral_content_text(resp)
                        current_model_type = "Reasoning (retry)"
                choices = getattr(resp, "choices", None)
                if isinstance(choices, list) and choices:
                    finish_reason = getattr(choices[0], "finish_reason", None)
                    if finish_reason and str(finish_reason) != "stop":
                        _safe_log_warning(ERROR_MISTRAL_REQUEST, f"finish_reason={finish_reason}", logger=dev_logger)
                usage = getattr(resp, "usage", None)
                usage_data: Optional[Dict[str, int]] = None
                if usage:
                    usage_data = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(usage, "completion_tokens", 0),
                        "total_tokens": getattr(usage, "total_tokens", 0),
                    }
                idx_int, explicit_none, err = _parse_mistral_idx(out_text)
                if explicit_none:
                    _write_mistral_log(
                        payload_json=payload_text,
                        response_text=out_text,
                        model=model,
                        usage=usage_data,
                        model_type=current_model_type,
                        reasoning_effort=selected_reasoning_effort,
                        prompt_mode=prompt_mode_value,
                    )
                    return None, True
                if err:
                    _safe_log_error(ERROR_MISTRAL_REQUEST, err)
                    _safe_log_error(ERROR_MISTRAL_FINAL_SELECTION, err)
                else:
                    _write_mistral_log(
                        payload_json=payload_text,
                        response_text=out_text,
                        model=model,
                        usage=usage_data,
                        model_type=current_model_type,
                        reasoning_effort=selected_reasoning_effort,
                        prompt_mode=prompt_mode_value,
                    )
                    return idx_int, False

            if idx_model + 1 < len(chain):
                if _is_genius_detailed_log():
                    _safe_log_info(INFO_MISTRAL_SWITCH_MODEL, chain[idx_model + 1])
                continue
            return None, False
        return None, False

    if cache_key:
        task = _MISTRAL_CHOICE_INFLIGHT.get(cache_key)
        if task is None:
            task = asyncio.create_task(_run_selection())
            _MISTRAL_CHOICE_INFLIGHT[cache_key] = task

            def _clear_choice_task(done_task, key=cache_key):
                if _MISTRAL_CHOICE_INFLIGHT.get(key) is done_task:
                    _MISTRAL_CHOICE_INFLIGHT.pop(key, None)
                if not done_task.cancelled():
                    done_task.exception()

            task.add_done_callback(_clear_choice_task)
            try:
                result = await asyncio.shield(task)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                _safe_log_error(ERROR_UNEXPECTED, e, logger=dev_logger)
                result = (None, False)

            if result[1] or result[0] is not None:
                now = asyncio.get_running_loop().time()
                _mistral_choice_cache_set(cache_key, now, result[0], result[1])
            return result[0], result[1], "fresh"
        try:
            idx_shared, explicit_none_shared = await asyncio.shield(task)
            return idx_shared, explicit_none_shared, "inflight"
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _safe_log_error(ERROR_UNEXPECTED, e, logger=dev_logger)
            return None, False, "skipped"

    task = asyncio.create_task(_run_selection())
    try:
        idx_direct, explicit_none_direct = await task
        return idx_direct, explicit_none_direct, "fresh"
    except asyncio.CancelledError:
        try:
            task.cancel()
        except RuntimeError:
            pass
        raise
    except Exception as e:
        _safe_log_error(ERROR_UNEXPECTED, e, logger=dev_logger)
        return None, False, "skipped"
