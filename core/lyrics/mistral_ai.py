from __future__ import annotations
import os
import json
import asyncio
import re
import hashlib
from collections import OrderedDict
import aiohttp
from typing import Optional, List, Dict, Any, Tuple
from constants import (
    DIRS,
    ERROR_AUTH_MISTRAL_API_KEY_MISSING,
    ERROR_MISTRAL_API_KEY_INVALID,
    ERROR_MISTRAL_FINAL_SELECTION,
    ERROR_MISTRAL_MODEL_NOT_FOUND,
    ERROR_MISTRAL_REQUEST,
    ERROR_MISTRAL_SDK_MISSING,
    ERROR_MISTRAL_TIMEOUT,
    INFO_MISTRAL_FINAL_SELECTION_START,
    INFO_MISTRAL_MODEL_UPDATED,
    INFO_MISTRAL_SWITCH_MODEL,
    MISTRAL_TIMEOUT_SECONDS,
)
from logger import _safe_log_info, _safe_log_error
from utils.http_session import init_session
from config import Settings

try:
    from mistralai import Mistral as MistralClient
except ImportError:
    MistralClient = None

_MISTRAL_LOG_PATH = os.path.join(DIRS["LOGS"], "GENIUS_MISTRAL_FINAL_SELECTION.log")
_MISTRAL_MODELS_CACHE_PATH = os.path.join(DIRS["SESSION"], "mistral_models_list.json")

_MISTRAL_FINAL_SELECTION_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "idx": {"type": "string"},
    },
    "required": ["idx"],
    "additionalProperties": False,
}

__all__ = [
    "get_mistral_models_chain",
    "read_mistral_model",
    "write_mistral_model",
    "choose_genius_candidate_with_mistral",
    "is_model_reasoning",
    "update_models_cache",
    "check_model_reasoning_type",
    "ensure_selected_models_cached",
]

_MISTRAL_CHOICE_CACHE_TTL_SECONDS: float = 3600.0
_MISTRAL_CHOICE_CACHE_MAX_SIZE: int = int(_MISTRAL_CHOICE_CACHE_TTL_SECONDS)
_MISTRAL_CHOICE_CACHE: "OrderedDict[str, Tuple[float, Optional[int], bool]]" = OrderedDict()
_MISTRAL_CHOICE_INFLIGHT: Dict[str, asyncio.Task] = {}

def _mistral_choice_cache_now() -> float:
    return asyncio.get_running_loop().time()

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

def _build_mistral_choice_cache_key(
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
    except Exception:
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

def _is_genius_detailed_log() -> bool:
    st = get_settings()
    return bool(st.genius_detailed_log)

def _write_mistral_log(
    payload_json: str,
    response_text: str,
    model: str,
    usage: Optional[Dict[str, int]] = None,
    model_type: Optional[str] = None,
) -> None:
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("GENIUS MISTRAL FINAL SELECTION LOG")
    lines.append("=" * 80)
    lines.append("")

    lines.append(f"Model: {model}")
    if model_type:
        type_json = json.dumps({"model_type": model_type}, ensure_ascii=False)
        lines.append(type_json)
    lines.append("")
    lines.append("-" * 40)
    lines.append("SENT TO MISTRAL")
    lines.append("-" * 40)
    lines.append("")
    lines.append(payload_json)
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
    models = _split_model_ids(cache.get("selected_models"))
    if models:
        return ", ".join(models)
    st = get_settings()
    return st.mistral_model

def write_mistral_model(new_model: str) -> bool:
    cache = _read_models_cache()
    nm = (new_model or "").strip()
    if not nm:
        return False
    new_list = _split_model_ids(nm)
    if not new_list:
        return False
    cur_val = cache.get("selected_models", [])
    cur_list = _split_model_ids(cur_val)

    if new_list == cur_list:
        return False

    cache["selected_models"] = new_list
    _write_models_cache(cache)
    _safe_log_info(INFO_MISTRAL_MODEL_UPDATED, ", ".join(new_list))
    return True

def get_mistral_models_chain() -> List[str]:
    cache = _read_models_cache()
    val = cache.get("selected_models")
    chain: List[str] = []
    parts = _split_model_ids(val)
    seen = set()
    for p in parts:
        if p not in seen:
            seen.add(p)
            chain.append(p)

    if not chain:
        st = get_settings()
        chain = [st.mistral_model]
    return chain

def _read_models_cache() -> Dict[str, Any]:
    try:
        with open(_MISTRAL_MODELS_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return {}

def _write_models_cache(data: Dict[str, Any]) -> None:
    try:
        os.makedirs(DIRS["SESSION"], exist_ok=True)
        tmp_path = _MISTRAL_MODELS_CACHE_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, _MISTRAL_MODELS_CACHE_PATH)
    except OSError:
        return

def _coerce_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for el in value:
            if isinstance(el, str):
                parts.append(el)
            elif isinstance(el, dict):
                t = el.get("text") or el.get("content") or el.get("data")
                if t is not None:
                    parts.append(str(t))
            else:
                parts.append(str(el))
        return "".join(parts)
    return str(value)

def _extract_mistral_content_text(resp: Any) -> str:
    raw: Any = None
    choices = getattr(resp, "choices", None)
    if isinstance(choices, list) and choices:
        msg = getattr(choices[0], "message", None)
        raw = getattr(msg, "content", None) if msg is not None else None
    if raw is None:
        raw = resp
    return _coerce_to_text(raw).strip()

def _parse_mistral_idx(out_text: str) -> tuple[Optional[int], bool, Optional[str]]:
    if not out_text:
        return None, False, "empty response"
    try:
        parsed = json.loads(out_text)
    except json.JSONDecodeError:
        return None, False, "invalid content: not json"
    if not isinstance(parsed, dict):
        return None, False, "invalid content: root is not object"
    idx_val = parsed.get("idx")
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
    low = message.lower()
    if "401" in low or "unauthorized" in low or "forbidden" in low or "invalid api key" in low:
        _safe_log_error(ERROR_MISTRAL_API_KEY_INVALID)
    elif "404" in low or ("model" in low and "not found" in low):
        _safe_log_error(ERROR_MISTRAL_MODEL_NOT_FOUND, model)
    else:
        _safe_log_error(ERROR_MISTRAL_REQUEST, message)
    _safe_log_error(ERROR_MISTRAL_FINAL_SELECTION, message)

def is_model_reasoning(model_id: str) -> bool:
    cache = _read_models_cache()
    models = cache.get("available_models", {})
    if model_id in models:
        return models[model_id].get("reasoning", False)
    for mid, info in models.items():
        aliases = info.get("aliases", [])
        if model_id in aliases:
            return info.get("reasoning", False)

    return False

async def check_model_reasoning_type(model_id: str, api_key: str, max_retries: int = 3) -> bool:
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json", "Content-Type": "application/json"}
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "1+1="}],
        "max_tokens": 1,
        "temperature": 0,
        "prompt_mode": "reasoning",
    }
    session = init_session()
    timeout = aiohttp.ClientTimeout(total=15.0)
    for attempt in range(max_retries):
        try:
            async with session.post(url, headers=headers, json=payload, timeout=timeout) as resp:
                if resp.status == 200:
                    return True
                elif resp.status == 400:
                    return False
                elif resp.status == 429:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                elif resp.status >= 500:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                else:
                    await asyncio.sleep(0.5)
                    continue
        except asyncio.TimeoutError:
            await asyncio.sleep(1.0 * (attempt + 1))
            continue
        except (aiohttp.ClientError, OSError):
            await asyncio.sleep(0.5)
            continue
    return False

async def update_models_cache(
    models_data: List[Dict[str, Any]],
    api_key: str,
    force_recheck: bool = False
) -> Dict[str, Any]:
    full_cache = _read_models_cache()
    existing_selected = _split_model_ids(full_cache.get("selected_models", []))

    cache = full_cache if not force_recheck else {}
    cached_models = cache.get("available_models", {})

    new_models: Dict[str, Any] = {}

    for item in models_data:
        model_id = item.get("id")
        if not model_id:
            continue

        aliases = item.get("aliases", [])
        if model_id in cached_models and not force_recheck:
            new_models[model_id] = cached_models[model_id]
            new_models[model_id]["aliases"] = aliases
        else:
            is_reasoning = await check_model_reasoning_type(model_id, api_key)
            new_models[model_id] = {
                "aliases": aliases,
                "reasoning": is_reasoning
            }
    if existing_selected:
        final_selected = existing_selected
    else:
        st = get_settings()
        final_selected = _split_model_ids(st.mistral_model)
    new_cache = {
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
    models_list = _split_model_ids(current_models_str)
    models_to_check = []
    for model_id in models_list:
        if model_id not in cached_models:
            found = False
            for mid, info in cached_models.items():
                if model_id in info.get("aliases", []):
                    found = True
                    break
            if not found:
                models_to_check.append(model_id)

    if models_to_check:
        for model_id in models_to_check:
            is_reasoning = await check_model_reasoning_type(model_id, api_key)
            cached_models[model_id] = {
                "aliases": [],
                "reasoning": is_reasoning
            }
        cache["available_models"] = cached_models
        _write_models_cache(cache)

async def choose_genius_candidate_with_mistral(
    spotify_track_meta: Dict[str, str],
    genius_candidates_meta: List[Dict[str, Any]],
) -> Tuple[Optional[int], bool]:
    if not genius_candidates_meta:
        return None, False

    st = get_settings()
    api_key = st.mistral_api_key
    if MistralClient is None:
        _safe_log_error(ERROR_MISTRAL_SDK_MISSING)
        return None, False
    if not api_key:
        _safe_log_error(ERROR_AUTH_MISTRAL_API_KEY_MISSING)
        return None, False

    chain = get_mistral_models_chain()
    cache_key = _build_mistral_choice_cache_key(chain, spotify_track_meta, genius_candidates_meta)
    if cache_key:
        now = _mistral_choice_cache_now()
        cached = _mistral_choice_cache_get(cache_key, now)
        if cached is not None:
            idx_cached, explicit_none_cached = cached
            if explicit_none_cached:
                return None, True
            return idx_cached, False

        inflight = _MISTRAL_CHOICE_INFLIGHT.get(cache_key)
        if inflight is not None:
            try:
                return await inflight
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

    payload = {
        "spotify_track_meta": spotify_track_meta,
        "genius_candidates_meta": genius_candidates_meta,
    }

    try:
        payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        _safe_log_error(ERROR_MISTRAL_FINAL_SELECTION, "failed to serialize payload")
        return None, False

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
        async with MistralClient(api_key=api_key) as mc:
            for idx_model, model in enumerate(chain):
                messages = [
                    {"role": "system", "content": sys_content},
                    {"role": "user", "content": user_content},
                ]
                model_is_reasoning = is_model_reasoning(model)
                current_model_type = "Reasoning" if model_is_reasoning else "Non Reasoning"
                chat_kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "max_tokens": 8,
                    "temperature": 0,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "genius_final_choice",
                            "schema": _MISTRAL_FINAL_SELECTION_JSON_SCHEMA,
                        },
                    },
                }
                if model_is_reasoning:
                    chat_kwargs["prompt_mode"] = "reasoning"

                if not start_logged:
                    _safe_log_info(INFO_MISTRAL_FINAL_SELECTION_START, model)
                    start_logged = True

                try:
                    resp = await asyncio.wait_for(
                        mc.chat.complete_async(**chat_kwargs),
                        timeout=MISTRAL_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    _safe_log_error(ERROR_MISTRAL_TIMEOUT, model, MISTRAL_TIMEOUT_SECONDS)
                    resp = None
                except (aiohttp.ClientError, OSError) as e:
                    msg = str(e) or "mistral request failed"
                    _log_mistral_exception(msg, model)
                    resp = None

                if resp is not None:
                    out_text = _extract_mistral_content_text(resp)
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
                        )
                        return idx_int, False

                if idx_model + 1 < len(chain):
                    if _is_genius_detailed_log():
                        _safe_log_info(INFO_MISTRAL_SWITCH_MODEL, chain[idx_model + 1])
                    continue
                return None, False

    if cache_key:
        task = _MISTRAL_CHOICE_INFLIGHT.get(cache_key)
        if task is None:
            task = asyncio.create_task(_run_selection())
            _MISTRAL_CHOICE_INFLIGHT[cache_key] = task
            try:
                result = await task
            except asyncio.CancelledError:
                try:
                    task.cancel()
                except Exception:
                    pass
                raise
            except Exception:
                result = (None, False)
            finally:
                if _MISTRAL_CHOICE_INFLIGHT.get(cache_key) is task:
                    _MISTRAL_CHOICE_INFLIGHT.pop(cache_key, None)

            if result[1] or result[0] is not None:
                now = _mistral_choice_cache_now()
                _mistral_choice_cache_set(cache_key, now, result[0], result[1])
            return result
        try:
            return await task
        except asyncio.CancelledError:
            raise
        except Exception:
            return None, False

    return await _run_selection()
