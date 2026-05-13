from __future__ import annotations

import asyncio
import html
import re
from typing import Optional

import aiohttp

from constants import (
    ERROR_AUTH_MISTRAL_API_KEY_MISSING,
    INFO_MISTRAL_MODELS_NON_REASONING,
    INFO_MISTRAL_MODELS_REASONING,
)
from utils.http_session import init_session
from .mistral_ai import update_models_cache

_MISTRAL_NUMERIC_VARIANT_RE = re.compile(r"-(\d+(?:[.-]\d+)*)$")
_MISTRAL_BUILD_VARIANT_RE = re.compile(r"-c\d+(?:-[a-z]\d+)*-\d+$", re.IGNORECASE)
_MISTRAL_MODEL_VALIDATION_CONCURRENCY = 100
_MISTRAL_MODEL_VALIDATION_TIMEOUT_SECONDS = 30.0
_MISTRAL_MODEL_CAPABILITIES_TIMEOUT_SECONDS = 30.0
_MISTRAL_MODEL_CAPABILITIES_MAX_RETRIES = 1
_MISTRAL_MODEL_CAPABILITIES_CONCURRENCY = 100

def _model_base_of(value: str) -> str:
    if value.endswith("-latest"):
        return value[:-7]
    m = _MISTRAL_BUILD_VARIANT_RE.search(value)
    if m:
        return value[: m.start()]
    m = _MISTRAL_NUMERIC_VARIANT_RE.search(value)
    if m:
        return value[: m.start()]
    return value

def _model_version_of(value: str) -> int:
    m = _MISTRAL_NUMERIC_VARIANT_RE.search(value)
    if m:
        try:
            return int("".join(part.zfill(3) for part in re.findall(r"\d+", m.group(1))))
        except (TypeError, ValueError):
            return -1
    return -1

def _model_display_sort_key(value: str) -> tuple[int, int, str]:
    return (0 if value.endswith("-latest") else 1, -_model_version_of(value), value.lower())

def _merge_model_reasoning_info(target: dict, item: dict) -> None:
    known_reasoning = item.get("reasoning")
    if known_reasoning is True:
        target["reasoning"] = True
    elif known_reasoning is False and target.get("reasoning") is not True:
        target["reasoning"] = False
    known_efforts = item.get("reasoning_efforts", [])
    if isinstance(known_efforts, list) and known_efforts:
        target_efforts = target.setdefault("reasoning_efforts", [])
        for mode in known_efforts:
            if mode not in target_efforts:
                target_efforts.append(mode)

def _format_model_groups(groups: dict[str, list[tuple[str, str]]]) -> str:
    lines: list[str] = []
    for family_key in sorted(groups, key=lambda value: value.lower()):
        if lines:
            lines.append("")
        lines.append(f"[{html.escape(family_key)}]")
        for _, line in sorted(groups[family_key], key=lambda item: _model_display_sort_key(item[0])):
            lines.append(line)
    return "\n".join(lines)

def _get_cached_model_info(cached_models: dict, model_id: str) -> Optional[dict]:
    cached = cached_models.get(model_id)
    if isinstance(cached, dict):
        return cached
    for info in cached_models.values():
        if not isinstance(info, dict):
            continue
        aliases = info.get("aliases", [])
        if isinstance(aliases, list) and model_id in aliases:
            return info
    return None

async def _validate_mistral_chat_models(
    session: aiohttp.ClientSession,
    headers: dict,
    display_models: dict[str, dict],
) -> dict[str, dict]:
    timeout = aiohttp.ClientTimeout(total=_MISTRAL_MODEL_VALIDATION_TIMEOUT_SECONDS)
    semaphore = asyncio.Semaphore(_MISTRAL_MODEL_VALIDATION_CONCURRENCY)
    url_chat = "https://api.mistral.ai/v1/chat/completions"

    async def validate_one(display_id: str, info: dict) -> Optional[tuple[str, dict]]:
        payload = {
            "model": info["id"],
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "temperature": 0,
        }
        async with semaphore:
            try:
                async with session.post(url_chat, headers=headers, json=payload, timeout=timeout) as r:
                    if r.status in (400, 404, 422):
                        return None
                    return display_id, info
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                return display_id, info

    tasks = [validate_one(display_id, info) for display_id, info in display_models.items()]
    results = await asyncio.gather(*tasks)
    return {display_id: info for result in results if result for display_id, info in [result]}

def _build_models_for_cache(items: list[dict], valid_models: dict[str, dict]) -> list[dict]:
    models_by_id: dict[str, dict] = {}
    for it in items:
        mid = it["id"]
        names: list[str] = []
        seen_names = set()
        for raw_name in [mid, *it["aliases"]]:
            name = str(raw_name).strip()
            if name and name in valid_models and name not in seen_names:
                seen_names.add(name)
                names.append(name)
        if not names:
            continue
        cache_id = mid if mid in valid_models else names[0]
        aliases = [name for name in names if name != cache_id]
        item = models_by_id.get(cache_id)
        if item is None:
            item = {"id": cache_id, "aliases": aliases}
            models_by_id[cache_id] = item
        else:
            existing_aliases = item.setdefault("aliases", [])
            for alias in aliases:
                if alias not in existing_aliases:
                    existing_aliases.append(alias)
        if isinstance(it.get("reasoning"), bool):
            item["reasoning"] = it["reasoning"]
        known_efforts = it.get("reasoning_efforts")
        if isinstance(known_efforts, list) and known_efforts:
            target_efforts = item.setdefault("reasoning_efforts", [])
            for mode in known_efforts:
                if mode not in target_efforts:
                    target_efforts.append(mode)
    return list(models_by_id.values())

def _model_known_reasoning(item: dict) -> Optional[bool]:
    capabilities = item.get("capabilities")
    if isinstance(capabilities, dict):
        reasoning = capabilities.get("reasoning")
        if isinstance(reasoning, bool):
            return reasoning
    reasoning = item.get("reasoning")
    if isinstance(reasoning, bool):
        return reasoning
    return None

def _model_known_reasoning_efforts(item: dict) -> list[str]:
    raw_values = []
    capabilities = item.get("capabilities")
    if isinstance(capabilities, dict):
        raw_values = capabilities.get("reasoning_efforts") or capabilities.get("reasoning_modes") or []
    if not raw_values:
        raw_values = item.get("reasoning_efforts") or item.get("reasoning_modes") or []
    if not isinstance(raw_values, list):
        return []
    values: list[str] = []
    seen = set()
    for raw in raw_values:
        mode = str(raw).strip().lower()
        if mode and mode not in seen:
            seen.add(mode)
            values.append(mode)
    return values

async def fetch_mistral_models_ids(
    api_key: Optional[str],
    *,
    include_details: bool = True,
    model_ids_to_cache: Optional[list[str]] = None,
) -> tuple[list[str], str, Optional[str]]:
    if not api_key:
        return [], "", ERROR_AUTH_MISTRAL_API_KEY_MISSING

    url = "https://api.mistral.ai/v1/models"
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    try:
        session = init_session()
        timeout = aiohttp.ClientTimeout(total=8.0)
        async with session.get(url, headers=headers, timeout=timeout) as resp:
            if resp.status < 200 or resp.status >= 300:
                txt = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {txt[:200]}")
            payload = await resp.json()
        if not isinstance(payload, dict):
            raise RuntimeError("invalid response payload")
        data = payload.get("data") or []
        items: list[dict] = []
        for item in data:
            mid = str(item.get("id")) if item.get("id") is not None else None
            if not mid:
                continue
            aliases = item.get("aliases") or item.get("alias") or []
            if not isinstance(aliases, (list, tuple)):
                aliases = []
            aliases = [str(a) for a in aliases if a]
            item_data = {"id": mid, "aliases": aliases}
            known_reasoning = _model_known_reasoning(item)
            if isinstance(known_reasoning, bool):
                item_data["reasoning"] = known_reasoning
            known_efforts = _model_known_reasoning_efforts(item)
            if known_efforts:
                item_data["reasoning_efforts"] = known_efforts
            items.append(item_data)

        display_models: dict[str, dict] = {}
        for it in items:
            mid = it["id"]
            aliases = it["aliases"]
            known_reasoning = it.get("reasoning")
            known_efforts = it.get("reasoning_efforts", [])
            names: list[str] = []
            seen_names = set()
            for raw_name in [mid, *aliases]:
                name = str(raw_name).strip()
                if name and name not in seen_names:
                    seen_names.add(name)
                    names.append(name)
            if not names:
                continue
            for display_id in names:
                prev = display_models.get(display_id)
                if prev is None:
                    display_models[display_id] = {
                        "id": display_id,
                        "aliases": set(name for name in names if name != display_id),
                        "family": _model_base_of(display_id),
                        "reasoning": known_reasoning,
                        "reasoning_efforts": list(known_efforts) if isinstance(known_efforts, list) else [],
                    }
                    continue
                prev["aliases"].update(name for name in names if name != display_id)
                _merge_model_reasoning_info(prev, it)
                if "family" not in prev:
                    prev["family"] = _model_base_of(display_id)

        requested_ids = {
            str(model_id).strip()
            for model_id in (model_ids_to_cache or [])
            if str(model_id).strip()
        }
        if include_details or not requested_ids:
            models_to_validate = display_models
        else:
            models_to_validate = {
                model_id: display_models[model_id]
                for model_id in requested_ids
                if model_id in display_models
            }
        valid_models = await _validate_mistral_chat_models(session, headers, models_to_validate)
        models_for_cache = _build_models_for_cache(items, valid_models)
        if models_for_cache:
            models_cache = await update_models_cache(
                models_for_cache,
                api_key,
                force_recheck=True,
                max_retries=_MISTRAL_MODEL_CAPABILITIES_MAX_RETRIES,
                timeout_seconds=_MISTRAL_MODEL_CAPABILITIES_TIMEOUT_SECONDS,
                concurrency=_MISTRAL_MODEL_CAPABILITIES_CONCURRENCY,
            )
            cached_models = models_cache.get("available_models", {})
        else:
            cached_models = {}

        if not include_details:
            return sorted(valid_models), "", None

        non_reasoning_groups: dict[str, list[tuple[str, str]]] = {}
        reasoning_groups: dict[str, list[tuple[str, str]]] = {}

        for info in valid_models.values():
            mid = info["id"]
            mid_html = f"<code>{html.escape(mid)}</code>"
            family_key = str(info.get("family") or _model_base_of(mid))

            is_reasoning = False
            modes: list[str] = []
            cached = _get_cached_model_info(cached_models, mid)
            if cached:
                is_reasoning = cached.get("reasoning", False)
                raw_modes = cached.get("reasoning_efforts") or cached.get("reasoning_modes") or []
                if isinstance(raw_modes, list):
                    seen_modes = set()
                    for raw_mode in raw_modes:
                        mode = str(raw_mode).strip().lower()
                        if mode and mode not in seen_modes:
                            seen_modes.add(mode)
                            modes.append(mode)

            if is_reasoning:
                visible_modes = [m for m in modes if m != "none"]
                if visible_modes:
                    modes_wrapped = ", ".join(f"<code>{html.escape(m)}</code>" for m in visible_modes)
                    line = f"- {mid_html} | reasoning_effort: {modes_wrapped}"
                else:
                    line = f"- {mid_html}"
                reasoning_groups.setdefault(family_key, []).append((mid, line))
            else:
                line = f"- {mid_html}"
                non_reasoning_groups.setdefault(family_key, []).append((mid, line))

        details_parts: list[str] = []
        if non_reasoning_groups:
            details_parts.append(INFO_MISTRAL_MODELS_NON_REASONING)
            details_parts.append(_format_model_groups(non_reasoning_groups))
        if reasoning_groups:
            if details_parts:
                details_parts.append("")
            details_parts.append(INFO_MISTRAL_MODELS_REASONING)
            details_parts.append(_format_model_groups(reasoning_groups))

        details_text = "\n".join(details_parts)

        valid_ids_set: set[str] = set()
        for info in valid_models.values():
            valid_ids_set.add(info["id"])

        return sorted(valid_ids_set), details_text, None
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, TypeError, ValueError, RuntimeError) as e:
        err_text = str(e).strip()
        if not err_text:
            err_text = e.__class__.__name__
        return [], "", err_text
