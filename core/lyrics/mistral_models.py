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

_MISTRAL_VERSION_RE = re.compile(r"-(\d{3,})$")

def _model_base_of(value: str) -> str:
    if value.endswith("-latest"):
        return value[:-7]
    m = _MISTRAL_VERSION_RE.search(value)
    if m:
        return value[: m.start()]
    return value

def _model_version_of(value: str) -> int:
    m = _MISTRAL_VERSION_RE.search(value)
    if m:
        try:
            return int(m.group(1))
        except (TypeError, ValueError):
            return -1
    return -1

async def fetch_mistral_models_ids(api_key: Optional[str]) -> tuple[list[str], str, Optional[str]]:
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
            items.append({"id": mid, "aliases": aliases})

        families: dict[str, dict] = {}
        for it in items:
            mid = it["id"]
            aliases = it["aliases"]
            bases = set()
            for al in aliases:
                if al.endswith("-latest"):
                    bases.add(al[:-7])
                elif not _MISTRAL_VERSION_RE.search(al):
                    bases.add(al)
            bmid = _model_base_of(mid)
            bases.add(bmid)
            v = _model_version_of(mid)
            has_latest_alias = any(a.endswith("-latest") for a in aliases)
            mid_is_latest = mid.endswith("-latest")
            score = (v, 1 if has_latest_alias else 0)
            for b in bases:
                prev = families.get(b)
                if prev is None or score > prev["score"]:
                    merged_aliases = set(aliases)
                    has_latest_id = mid_is_latest
                    if prev is not None:
                        merged_aliases |= prev.get("aliases", set())
                        has_latest_id = has_latest_id or prev.get("has_latest_id", False)
                    families[b] = {
                        "id": mid,
                        "aliases": merged_aliases,
                        "score": score,
                        "has_latest_id": has_latest_id,
                    }
                else:
                    prev["aliases"].update(aliases)
                    if mid_is_latest:
                        prev["has_latest_id"] = True

        valid_families: dict[str, dict] = {}
        try:
            session = init_session()
            timeout2 = aiohttp.ClientTimeout(total=7.0)
            url_chat = "https://api.mistral.ai/v1/chat/completions"
            for b, info in families.items():
                mid = info["id"]
                payload = {
                    "model": mid,
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                    "temperature": 0,
                }
                try:
                    async with session.post(url_chat, headers=headers, json=payload, timeout=timeout2) as r:
                        if r.status == 400:
                            continue
                        valid_families[b] = info
                except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                    valid_families[b] = info
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            valid_families = families

        models_for_cache: list[dict] = []
        for info in valid_families.values():
            mid = info["id"]
            aliases = list(info["aliases"])
            models_for_cache.append({"id": mid, "aliases": aliases})
        models_cache = await update_models_cache(models_for_cache, api_key, force_recheck=True)
        cached_models = models_cache.get("available_models", {})

        non_reasoning_pairs: list[tuple[str, str]] = []
        reasoning_pairs: list[tuple[str, str]] = []

        for info in valid_families.values():
            mid = info["id"]
            aliases = sorted(a for a in info["aliases"] if a != mid)
            fam_base = _model_base_of(mid)
            filtered_aliases_set = set()
            for a in aliases:
                if a.endswith("-latest") and _model_base_of(a) == fam_base:
                    filtered_aliases_set.add(a)
                elif (not _MISTRAL_VERSION_RE.search(a)) and _model_base_of(a) == fam_base:
                    filtered_aliases_set.add(a)
            if info.get("has_latest_id"):
                latest_alias_value = f"{fam_base}-latest"
                if latest_alias_value != mid:
                    filtered_aliases_set.add(latest_alias_value)
            filtered_aliases = sorted(filtered_aliases_set)
            latest_alias_value = f"{fam_base}-latest"
            display_list: list[str] = []
            if latest_alias_value in filtered_aliases:
                display_list.append(latest_alias_value)
            rest = [a for a in filtered_aliases if a != latest_alias_value]
            rest_space = 5 - len(display_list)
            if rest_space > 0:
                display_list.extend(rest[:rest_space])
            extra_count = max(0, len(rest) - rest_space)
            mid_html = f"<code>{html.escape(mid)}</code>"
            if display_list:
                aliases_wrapped = ", ".join(f"<code>{html.escape(a)}</code>" for a in display_list)
                if extra_count:
                    aliases_wrapped += f", … (+{extra_count})"
                line = f"- {mid_html} | aliases: {aliases_wrapped}"
            else:
                line = f"- {mid_html}"

            is_reasoning = False
            if mid in cached_models:
                is_reasoning = cached_models[mid].get("reasoning", False)

            if is_reasoning:
                reasoning_pairs.append((mid.lower(), line))
            else:
                non_reasoning_pairs.append((mid.lower(), line))

        non_reasoning_pairs.sort(key=lambda t: t[0])
        reasoning_pairs.sort(key=lambda t: t[0])

        details_parts: list[str] = []
        if non_reasoning_pairs:
            details_parts.append(INFO_MISTRAL_MODELS_NON_REASONING)
            details_parts.append("\n".join(line for _, line in non_reasoning_pairs))
        if reasoning_pairs:
            if details_parts:
                details_parts.append("")
            details_parts.append(INFO_MISTRAL_MODELS_REASONING)
            details_parts.append("\n".join(line for _, line in reasoning_pairs))

        details_text = "\n".join(details_parts)

        valid_ids_set: set[str] = set()
        for info in valid_families.values():
            mid = info["id"]
            valid_ids_set.add(mid)
            for a in info["aliases"]:
                valid_ids_set.add(a)
            fam_base = _model_base_of(mid)
            if info.get("has_latest_id"):
                valid_ids_set.add(f"{fam_base}-latest")

        return sorted(valid_ids_set), details_text, None
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError, TypeError, ValueError, RuntimeError) as e:
        return [], "", str(e)
