from __future__ import annotations
import re
import asyncio
import math
import threading
from collections import deque
from functools import lru_cache
from urllib.parse import urlsplit
from typing import Optional, Dict, Any, List, Tuple, Set, Iterator, Iterable, Callable

from logger import _safe_log_info, _safe_log_error, _safe_log_warning
from constants import (
    ERROR_AUTH_GENIUS_INVALID_TOKEN,
    ERROR_GENIUS_DETAILS_FAILED,
    ERROR_MISTRAL_FINAL_SELECTION,
    ERROR_UNEXPECTED,
    GENIUS_ALBUM_MIN_TRACKS,
    GENIUS_DETAILS_TOP_N,
    GENIUS_FUZZY_CHARS_PERCENT,
    GENIUS_FUZZY_TOKENS_PERCENT,
    GENIUS_PENALTY_ALBUM_MISMATCH,
    GENIUS_PENALTY_FEATURED_EXTRA_TOKEN_PER,
    GENIUS_PENALTY_FEATURED_MISSING_TOKEN_PER,
    GENIUS_PENALTY_TAG_MISSING,
    GENIUS_PENALTY_TAG_MISSING_TOKEN_PER,
    GENIUS_PENALTY_TITLE_EXTRA_TOKEN_PER,
    GENIUS_PENALTY_TITLE_MISSING_TOKEN_PER,
    GENIUS_PENALTY_TOKEN_CAP,
    GENIUS_SCORE_ALBUM_MATCH,
    GENIUS_SCORE_CONSEC_TOKENS_MAX,
    GENIUS_SCORE_CONSEC_TOKENS_PER,
    GENIUS_SCORE_FEATURED_IN_TITLE_MAX,
    GENIUS_SCORE_FEATURED_IN_TITLE_PER,
    GENIUS_SCORE_FEATURED_MATCH_MAX,
    GENIUS_SCORE_FEATURED_MATCH_PER,
    GENIUS_SCORE_RELATIONSHIP_TAG_TOKENS,
    GENIUS_SCORE_SLUG_TOKEN_PER,
    GENIUS_SCORE_TAG_PHRASE_IN_TITLE,
    GENIUS_SCORE_TAG_PHRASE_IN_TITLE_FEAT,
    GENIUS_SCORE_TAG_TOKENS_MAX,
    GENIUS_SCORE_TAG_TOKENS_PER,
    GENIUS_SCORE_TRACK_EXACT_TITLE_BONUS,
    GENIUS_SEARCH_CONCURRENCY,
    GENIUS_SLUG_TOKEN_MIN_LEN,
    GENIUS_TAG_FUZZY_CHARS_PERCENT,
    GENIUS_TAG_FUZZY_TOKENS_PERCENT,
    GENIUS_TOP1_BONUS_MAX,
    GENIUS_TOP1_BONUS_PER,
    INFO_GENIUS_CANDIDATE_REASON,
    INFO_GENIUS_CANDIDATE_SUMMARY,
    INFO_GENIUS_DETAILS_FETCHED,
    INFO_GENIUS_FEAT_REMOVED_FROM_TRACK,
    INFO_GENIUS_NO_RESULTS,
    INFO_GENIUS_SCORING_CHOSEN,
    INFO_GENIUS_SEARCH_QUERY,
    INFO_GENIUS_TAG_FOUND_AND_FEAT_REMOVED_FROM_TRACK,
    INFO_GENIUS_TAG_FOUND_FROM_TRACK,
    INFO_MISTRAL_FALLBACK,
    INFO_MISTRAL_FINAL_SELECTION_NONE,
    INFO_MISTRAL_FINAL_SELECTION_NONE_CACHED,
    INFO_MISTRAL_FINAL_SELECTION_NONE_INFLIGHT,
    INFO_TRANSLATION_REPLACED,
    SEARCH_VARIANTS_LIMIT,
)
from utils.genius_client import AsyncGeniusClient
from utils.http_session import init_session
from config import Settings
from utils.mistral_client import prefetch_mistral_client
from .mistral_ai import (
    build_mistral_choice_cache_key,
    build_mistral_payload_text,
    choose_genius_candidate_with_mistral,
    get_mistral_models_chain,
)
from .patterns.base import (
    normalize_letters,
    normalize_query_value,
    strip_spaced_separators,
    normalize_apostrophes,
    convert_separators_to_space,
    detect_feat_placeholders_bracketed,
    strip_feat_placeholders_bracketed,
    feat_placeholder_pattern,
    _normalize_marker_tokens,
    _best_multi_pattern,
    _remove_separators_with_set,
    _SECONDARY_SET,
    _is_feat_placeholder_inner,
    apply_symbol_letter_equiv,
    generate_anyascii_variants,
    generate_roman_l_i_variants,
    iter_symbol_letter_equiv_variants,
    get_feat_patterns,
    get_feat_patterns_version,
    get_tag_patterns,
    get_tag_patterns_version,
    split_artist_text_by_feat_markers,
)

_settings: Optional[Settings] = None

def set_settings(settings: Settings) -> None:
    global _settings
    _settings = settings

def get_settings() -> Settings:
    if _settings is None:
        raise RuntimeError("Settings are not initialized")
    return _settings

def _prefetch_search_runtime_sync() -> None:
    _ = get_feat_patterns()
    _ = get_tag_patterns()
    _ = get_feat_patterns_version()
    _ = get_tag_patterns_version()
    sample_track = "Warmup Track"
    sample_variants = [{
        "primary_artist": "Warmup Artist",
        "featured_artists": "Warmup Guest",
        "track": sample_track,
        "tag": None,
    }]
    prepared_variants, base_queries, seen_q, search_variants_limit = _prepare_base_queries_sync(
        sample_variants,
        sample_track,
        min(max(8, _safe_int(SEARCH_VARIANTS_LIMIT, 16)), 16),
    )
    variants_normalized = _prepare_normalized_variants_sync(prepared_variants).materialize()
    variants_symbolized = _prepare_symbolized_variants_sync(variants_normalized).materialize()
    variants_for_match = _prepare_anyascii_match_variants_sync(variants_symbolized).materialize()
    remaining_slots = max(0, search_variants_limit - len(base_queries))
    _ = _prepare_early_fallback_streams_sync(
        variants_normalized=variants_normalized,
        base_queries=base_queries,
        remaining_slots=remaining_slots,
        seen_keys=seen_q,
    )
    _ = _prepare_fallback_streams_sync(
        variants=prepared_variants,
        variants_normalized=variants_normalized,
        variants_symbolized=variants_symbolized,
        variants_for_match=variants_for_match,
        found_initial_tag_or_feat_placeholders=False,
        remaining_slots=remaining_slots,
        seen_keys=seen_q,
    )

async def prefetch_search_runtime() -> None:
    _ = init_session()
    try:
        await asyncio.to_thread(_prefetch_search_runtime_sync)
    except (RuntimeError, ValueError, TypeError, OSError):
        return

def _is_mistral_enabled() -> bool:
    st = get_settings()
    return bool(st.enable_mistral)

def _is_genius_detailed_log() -> bool:
    st = get_settings()
    return bool(st.genius_detailed_log)

def _collapse_ws(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())

def _log_no_results() -> None:
    _safe_log_info(INFO_GENIUS_NO_RESULTS)

_RE_CLEAR_TITLE_PARENS = re.compile(r"\s*\([^)]*\)\s*")
_RE_CLEAR_TITLE_TAIL_DASH = re.compile(r"\s+-\s*[^-]+$")
_RE_WORDS = re.compile(r"\w+")
_RE_NON_WORD = re.compile(r"\W+")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_APOSTROPHE = re.compile(r"'")
_RE_TOKSET_SEP = re.compile(r"[\s\-–—_/\\'\"\.,:;!\?]+")
_RE_NON_WORD_SPACE = re.compile(r"[^\w\s]")
_RE_TAG_DASH_BOTH = re.compile(r"\s[-–—]\s")
_RE_TAG_DASH_LEFT = re.compile(r"(^|\s)[-–—]\s")
_RE_TAG_DASH_RIGHT = re.compile(r"\s[-–—](\s|$)")
_RE_TRAILING_DASH = re.compile(r"\s*[-–—]+\s*$")
_RE_EXTRACT_TAG_BR = re.compile(r"\s*[\(\[]\s*([^\)\]]+?)\s*[\)\]]\s*$")
_RE_EXTRACT_TAG_DASH2 = re.compile(r"^(.*)\s+[-–—]+(?=\S)([^\(\)\[\]]*?\S)\s*[-–—]+\s*$")
_RE_EXTRACT_TAG_DASH1 = re.compile(r"^(.*)\s+[-–—]+\s+([^\(\)\[\]]+?)\s*$")

@lru_cache(maxsize=4096)
def _norm(s: Optional[str]) -> str:
    t = _safe_stripped(s)
    if not t:
        return ""
    t = normalize_apostrophes(t)
    t = t.replace("'", "")
    return _collapse_ws(t).lower()

@lru_cache(maxsize=2048)
def _spotify_text_variants(value: Optional[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for item in _anyascii_field_variants_cached(value):
        cleaned = _safe_stripped(item)
        if cleaned and cleaned not in out:
            out.append(cleaned)
    raw = _safe_stripped(value)
    if raw and raw not in out:
        out.insert(0, raw)
    return tuple(out)

@lru_cache(maxsize=8192)
def _normalized_field_variants_cached(value: Optional[str]) -> Tuple[Optional[str], ...]:
    if value is None:
        return (None,)
    original = str(value)
    normalized = normalize_letters(original)
    if not normalized or normalized == original:
        return (value,)
    return (value, normalized)

@lru_cache(maxsize=8192)
def _anyascii_field_variants_cached(value: Optional[str]) -> Tuple[Optional[str], ...]:
    if value is None:
        return (None,)
    generated = generate_anyascii_variants(value)
    if not generated:
        return (value,)
    out: List[Optional[str]] = []
    for item in generated:
        if item not in out:
            out.append(item)
    return tuple(out or [value])

@lru_cache(maxsize=8192)
def _symbol_field_variants_cached(value: Optional[str]) -> Tuple[Optional[str], ...]:
    if value is None:
        return (None,)
    original = str(value)
    replaced = apply_symbol_letter_equiv(original)
    if not replaced or replaced == original:
        return (value,)
    return (value, replaced)

@lru_cache(maxsize=4096)
def _token_display_pairs(source_value: Optional[str]) -> Tuple[Tuple[str, str], ...]:
    raw = _safe_stripped(source_value)
    if not raw:
        return ()
    prepared = normalize_apostrophes(raw)
    prepared = _RE_TOKSET_SEP.sub(" ", prepared)
    out: List[Tuple[str, str]] = []
    seen: Set[str] = set()
    for source_token in prepared.split():
        display_token = source_token.lower()
        normalized_token = _normalize_base_text(source_token)
        if not normalized_token:
            continue
        compare_tokens = [tok for tok in _RE_WORDS.findall(normalized_token) if tok]
        if not compare_tokens and normalized_token:
            compare_tokens = [normalized_token]
        for compare_token in compare_tokens:
            if compare_token and compare_token not in seen:
                seen.add(compare_token)
                out.append((compare_token, display_token))
    return tuple(out)

def _format_reason_tokens_for_display(tokens: List[str], source_value: Optional[str]) -> List[str]:
    if not tokens:
        return []
    display_map = dict(_token_display_pairs(source_value))
    out: List[str] = []
    seen_display: Set[str] = set()
    for token in tokens:
        display_token = display_map.get(token, token)
        if not display_token or display_token in seen_display:
            continue
        seen_display.add(display_token)
        out.append(display_token)
    return out

@lru_cache(maxsize=2048)
def _clear_title(s: str) -> str:
    t = _safe_stripped(s)
    if not t:
        return ""
    s1 = _RE_CLEAR_TITLE_PARENS.sub(" ", t)
    s1 = _RE_CLEAR_TITLE_TAIL_DASH.sub(" ", s1)
    s1 = _collapse_ws(s1)
    return s1

@lru_cache(maxsize=4096)
def _normalize_base_text(s: Optional[str], *, strip_separators: bool = False) -> str:
    t = _safe_stripped(s)
    if not t:
        return ""
    if strip_separators:
        t = strip_spaced_separators(t)
    t = normalize_letters(t)
    return _norm(t)

@lru_cache(maxsize=4096)
def _normalize_ws_tokens(s: Optional[str], *, strip_separators: bool = False) -> Tuple[str, ...]:
    base = _normalize_base_text(s, strip_separators=strip_separators)
    if not base:
        return ()
    return tuple(base.split())

@lru_cache(maxsize=4096)
def _norm_title_value(value: Optional[str], *, clear: bool = False) -> str:
    raw = _safe_stripped(value)
    if not raw:
        return ""
    if clear:
        raw = _clear_title(raw)
    return _normalize_base_text(raw)

def _remove_chars_for_transform(s: Optional[str]) -> str:
    t = _safe_stripped(s)
    if not t:
        return ""
    t = _remove_separators_with_set(t, _SECONDARY_SET)
    t = _collapse_ws(t)
    return t

def _concat_no_spaces(s: Optional[str]) -> str:
    t = _remove_chars_for_transform(s)
    if not t:
        return ""
    return "".join(t.split())

def _split_chars_spaced(s: Optional[str]) -> str:
    t = _remove_chars_for_transform(s)
    if not t:
        return ""
    return " ".join(" ".join(w) for w in t.split())

def _transform_by_mode(val: Optional[str], mode: str) -> str:
    if not val:
        return ""
    if mode == "norm":
        return _collapse_ws(strip_spaced_separators(val))
    if mode == "concat":
        return _concat_no_spaces(val)
    return _split_chars_spaced(val)

@lru_cache(maxsize=4096)
def _transform_by_mode_cached(val: Optional[str], mode: str) -> str:
    return _transform_by_mode(val, mode)

@lru_cache(maxsize=4096)
def _variant_mode_value_triplets_cached(
    primary_artist: Optional[str],
    featured_artists: Optional[str],
    track: Optional[str],
) -> Tuple[
    Tuple[str, str, str],
    Tuple[str, str, str],
    Tuple[str, str, str],
]:
    pa_raw = primary_artist or ""
    fea_raw = featured_artists if featured_artists else None
    tr_raw = track or ""
    _, base_track = _extract_trailing_tag(tr_raw)
    base_track = base_track or tr_raw
    return (
        (
            _transform_by_mode_cached(pa_raw, "norm"),
            _transform_by_mode_cached(pa_raw, "concat"),
            _transform_by_mode_cached(pa_raw, "spaced"),
        ),
        (
            _transform_by_mode_cached(fea_raw, "norm"),
            _transform_by_mode_cached(fea_raw, "concat"),
            _transform_by_mode_cached(fea_raw, "spaced"),
        ),
        (
            _transform_by_mode_cached(base_track, "norm"),
            _transform_by_mode_cached(base_track, "concat"),
            _transform_by_mode_cached(base_track, "spaced"),
        ),
    )

def _mode_triplet_value(values: Tuple[str, str, str], mode: Optional[str]) -> str:
    if mode == "concat":
        return values[1]
    if mode == "spaced":
        return values[2]
    return values[0]

@lru_cache(maxsize=4096)
def _build_mode_queries_cached(
    primary_artist: Optional[str],
    featured_artists: Optional[str],
    track: Optional[str],
    tag: Optional[str],
    found_initial_tag_or_feat_placeholders: bool,
    artist_modes: Tuple[str, ...],
    track_modes: Tuple[str, ...],
) -> Tuple[str, ...]:
    pa_raw = primary_artist or ""
    fea_raw = featured_artists if featured_artists else None
    tag_val = tag
    pa_values, fea_values, track_values = _variant_mode_value_triplets_cached(pa_raw, fea_raw, track or "")
    fea_modes: Tuple[Optional[str], ...] = artist_modes if fea_raw else (None,)
    out: List[str] = []
    for pa_mode in artist_modes:
        pa2 = _mode_triplet_value(pa_values, pa_mode)
        for fe_mode in fea_modes:
            fea2_val = _mode_triplet_value(fea_values, fe_mode) if fe_mode else ""
            for t_mode in track_modes:
                if pa_mode == "norm" and t_mode == "norm" and (fe_mode is None or fe_mode == "norm"):
                    continue
                tr2 = _mode_triplet_value(track_values, t_mode)
                skip_tag_feat_extract = (
                    (not found_initial_tag_or_feat_placeholders)
                    and (not tag_val)
                    and (t_mode in ("concat", "spaced"))
                )
                q2 = _build_query_from_variant_cached(
                    pa2,
                    fea2_val if fea2_val else None,
                    tr2,
                    tag_val,
                    skip_tag_feat_extract,
                )
                if q2:
                    out.append(q2)
    return tuple(out)

def _build_mode_queries(
    variant: Dict[str, Optional[str]],
    artist_modes: List[str],
    track_modes: List[str],
    found_initial_tag_or_feat_placeholders: bool,
) -> List[str]:
    if not isinstance(variant, dict):
        return []
    return list(
        _build_mode_queries_cached(
            variant.get("primary_artist"),
            variant.get("featured_artists"),
            variant.get("track"),
            variant.get("tag"),
            bool(found_initial_tag_or_feat_placeholders),
            tuple(artist_modes or []),
            tuple(track_modes or []),
        )
    )

def _longest_common_consecutive_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    n = len(b)
    dp = [0] * (n + 1)
    best = 0
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, n + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
                if dp[j] > best:
                    best = dp[j]
            else:
                dp[j] = 0
            prev = cur
    return best

@lru_cache(maxsize=4096)
def _tag_tokenize_for_fuzzy(s: Optional[str]) -> Tuple[str, ...]:
    base = _normalize_base_text(s, strip_separators=True)
    if not base:
        return ()
    return tuple(p for p in _RE_WORDS.findall(base) if p)

@lru_cache(maxsize=8192)
def _tag_token_similarity_consecutive(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b:
        return 0.0
    a1 = _normalize_base_text(a)
    b1 = _normalize_base_text(b)
    if not a1 or not b1:
        return 0.0
    a2 = _RE_NON_WORD.sub("", a1)
    b2 = _RE_NON_WORD.sub("", b1)
    if not a2 or not b2:
        return 0.0
    k = _longest_common_consecutive_len(list(a2), list(b2))
    denom = max(len(a2), len(b2))
    return float(k) / float(denom)

def _fuzzy_tag_match_in_text(tag_text: Optional[str], text: Optional[str], tokens_percent: float, chars_percent: float) -> bool:
    qtoks = _tag_tokenize_for_fuzzy(tag_text)
    if not qtoks:
        return False
    ctoks = _tag_tokenize_for_fuzzy(text)
    if not ctoks:
        return False

    p_tok = max(0.0, min(1.0, float(tokens_percent)))
    p_chr = max(0.0, min(1.0, float(chars_percent)))
    if p_tok <= 0 or p_chr <= 0:
        return False

    n = len(ctoks)
    dp = [0] * (n + 1)
    best = 0
    for i in range(1, len(qtoks) + 1):
        prev = 0
        for j in range(1, n + 1):
            cur = dp[j]
            ok = _tag_token_similarity_consecutive(qtoks[i - 1], ctoks[j - 1]) >= p_chr
            if ok:
                dp[j] = prev + 1
                if dp[j] > best:
                    best = dp[j]
            else:
                dp[j] = 0
            prev = cur

    ratio = float(best) / float(len(qtoks))
    return ratio >= p_tok

def _consec_title_tokens_score(spotify_title: Optional[str], genius_title: str, genius_title_with_featured: str) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    gtoks = _normalize_ws_tokens(genius_title, strip_separators=True)
    gftoks = _normalize_ws_tokens(genius_title_with_featured, strip_separators=True)
    k = 0
    for spotify_title_variant in _spotify_text_variants(spotify_title):
        stoks = _normalize_ws_tokens(spotify_title_variant, strip_separators=True)
        if not stoks:
            continue
        k = max(
            k,
            _longest_common_consecutive_len(stoks, gtoks),
            _longest_common_consecutive_len(stoks, gftoks),
        )
    if k > 0:
        add = min(GENIUS_SCORE_CONSEC_TOKENS_MAX, k * GENIUS_SCORE_CONSEC_TOKENS_PER)
        score += add
        reasons.append(f"consec_title_tokens +{add} ({k} tokens)")
    return score, reasons

def _fuzzy_tokens_match_core(qtoks: List[str], ctoks: List[str], tokens_percent: float, chars_percent: float) -> bool:
    if not qtoks or not ctoks:
        return False
    p_tok = max(0.0, min(1.0, float(tokens_percent)))
    p_chr = max(0.0, min(1.0, float(chars_percent)))
    n = len(qtoks)
    req_tokens = max(1, int(math.ceil(n * p_tok)))
    matched = 0
    for idx, q in enumerate(qtoks):
        L = len(q)
        if L <= 0:
            continue
        req_len = max(1, int(math.ceil(L * p_chr)))
        found = False
        if L >= req_len:
            for i in range(0, L - req_len + 1):
                sub = q[i:i + req_len]
                for ct in ctoks:
                    if sub in ct:
                        found = True
                        break
                if found:
                    break
        if found:
            matched += 1
            if matched >= req_tokens:
                return True
        remaining = n - (idx + 1)
        if matched + remaining < req_tokens:
            return False
    return matched >= req_tokens

def _fuzzy_tokens_match_impl(tokenizer, q_text: Optional[str], cand_text: Optional[str], tokens_percent: float, chars_percent: float) -> bool:
    qtoks = tokenizer(q_text)
    if not qtoks:
        return False
    ctoks = tokenizer(cand_text)
    if not ctoks:
        return False
    qtoks_sorted = qtoks if tokenizer is _tokens_with_charseqs else sorted(qtoks, key=lambda x: len(x), reverse=True)
    return _fuzzy_tokens_match_core(qtoks_sorted, ctoks, tokens_percent, chars_percent)

@lru_cache(maxsize=4096)
def _tokens_with_charseqs(s: Optional[str]) -> Tuple[str, ...]:
    base = _normalize_base_text(s, strip_separators=True)
    if not base:
        return ()
    tokens: List[str] = [p for p in _RE_WHITESPACE.split(base) if p]
    words = _RE_WORDS.findall(base)
    for w in words:
        if w and w not in tokens:
            tokens.append(w)
    joined_list: List[str] = []
    buf: List[str] = []
    for w in words:
        if len(w) == 1:
            buf.append(w)
        else:
            if buf:
                joined_list.append("".join(buf))
                buf = []
    if buf:
        joined_list.append("".join(buf))
    for j in joined_list:
        if j and j not in tokens:
            tokens.append(j)
    extra: List[str] = []
    for tok in tokens:
        t2 = _RE_APOSTROPHE.sub("", tok)
        if t2 and t2 != tok:
            extra.append(t2)
    for e in extra:
        if e not in tokens:
            tokens.append(e)
    sym_extra: List[str] = []
    for tok in tokens:
        repl = apply_symbol_letter_equiv(tok) or ""
        if repl and repl != tok:
            sym_extra.append(repl)
    for e in sym_extra:
        if e not in tokens:
            tokens.append(e)
    anyascii_extra: List[str] = []
    for tok in tokens:
        for repl in generate_anyascii_variants(tok):
            if repl and repl != tok:
                anyascii_extra.append(repl)
    for e in anyascii_extra:
        if e not in tokens:
            tokens.append(e)
    tokens_sorted = sorted(set(tokens), key=lambda x: len(x), reverse=True)
    return tuple(tokens_sorted)

def _safe_stripped(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()

def _safe_int(value: Any, default: int = 0) -> int:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return default
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if s and (s.isdigit() or ((s[0] in "+-") and s[1:].isdigit())):
            return int(s)
    return default

def _safe_int_or_none(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None

def _append_unique_artist_name(names: List[str], seen: Set[str], value: Any, *, skip_key: Optional[str] = None) -> None:
    s = _safe_stripped(value)
    if not s:
        return
    key = _norm(s)
    if not key or (skip_key and key == skip_key) or key in seen:
        return
    seen.add(key)
    names.append(s)

def _primary_artist_names(song: Dict[str, Any]) -> List[str]:
    if not isinstance(song, dict):
        return []
    names: List[str] = []
    seen: Set[str] = set()
    primary = (song.get("primary_artist") or {}).get("name")
    _append_unique_artist_name(names, seen, primary)
    for artist in song.get("primary_artists") or []:
        if isinstance(artist, dict):
            _append_unique_artist_name(names, seen, artist.get("name"))
    return names

def _song_matches_variant_fuzzy(song: Dict[str, Any], v: Dict[str, Optional[str]], tokens_percent: float, chars_percent: float) -> bool:
    if not isinstance(song, dict) or not isinstance(v, dict):
        return False
    title = song.get("title") or ""
    title_with_feat = song.get("title_with_featured") or ""
    titles = [title]
    if title_with_feat and title_with_feat != title:
        titles.append(title_with_feat)
    tr = v.get("track") or ""
    if tr and not any(_title_matches(tr, t) for t in titles):
        return False
    pa = v.get("primary_artist") or ""
    if pa:
        primary_names = _primary_artist_names(song)
        if primary_names:
            ok_primary = any(
                _fuzzy_tokens_match_impl(_tokens_with_charseqs, pa, gp, tokens_percent, chars_percent)
                or _fuzzy_tokens_match_impl(_tokens_with_charseqs, gp, pa, tokens_percent, chars_percent)
                for gp in primary_names
            )
            if not ok_primary:
                return False
    tg = v.get("tag") or ""
    if tg:
        t_tok = float(GENIUS_TAG_FUZZY_TOKENS_PERCENT)
        t_chr = float(GENIUS_TAG_FUZZY_CHARS_PERCENT)
        if not any(_fuzzy_tag_match_in_text(tg, t, t_tok, t_chr) for t in titles):
            return False
    return True

def _extract_featured_from_genius(song: Dict[str, Any]) -> List[str]:
    if not isinstance(song, dict):
        return []
    arr_feat = song.get("featured_artists") or []
    arr_prim = song.get("primary_artists") or []
    prim_name = ((song.get("primary_artist") or {}).get("name")) or None
    prim_key = _norm(prim_name) if prim_name else ""
    names: List[str] = []
    seen: Set[str] = set()
    for artist in arr_feat:
        if isinstance(artist, dict):
            _append_unique_artist_name(names, seen, artist.get("name"))
    for artist in arr_prim:
        if isinstance(artist, dict):
            _append_unique_artist_name(names, seen, artist.get("name"), skip_key=prim_key)
    return names

def _featured_match_score(featured_pref: Optional[List[str]], genius_song: Dict[str, Any], title_with_featured: str) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    if not featured_pref:
        return 0, reasons
    pref = [p for p in (featured_pref or []) if p]
    if not pref:
        return 0, reasons
    gfeat = _extract_featured_from_genius(genius_song)
    g_text = " ".join(n for n in gfeat if n) if gfeat else ""
    matched_cnt = 0
    for p in pref:
        if _fuzzy_tokens_match_impl(_tokens_with_charseqs, p, g_text, GENIUS_FUZZY_TOKENS_PERCENT, GENIUS_FUZZY_CHARS_PERCENT):
            matched_cnt += 1
    if matched_cnt:
        add = min(GENIUS_SCORE_FEATURED_MATCH_MAX, matched_cnt * GENIUS_SCORE_FEATURED_MATCH_PER)
        score += add
        reasons.append(f"featured_match +{add} ({matched_cnt} artists)")
    else:
        hits = 0
        for p in pref:
            if _fuzzy_tokens_match_impl(_tokens_with_charseqs, p, title_with_featured or "", GENIUS_FUZZY_TOKENS_PERCENT, GENIUS_FUZZY_CHARS_PERCENT):
                hits += 1
        if hits:
            add = min(GENIUS_SCORE_FEATURED_IN_TITLE_MAX, hits * GENIUS_SCORE_FEATURED_IN_TITLE_PER)
            score += add
            reasons.append(f"featured_in_title +{add} ({hits} names)")
    return score, reasons

def _album_match_score(spotify_album_name: Optional[str], spotify_album_total_tracks: Optional[int], genius_song: Dict[str, Any]) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    at = spotify_album_total_tracks
    if not (isinstance(at, int) and at >= GENIUS_ALBUM_MIN_TRACKS):
        return 0, reasons
    sp_name = _norm(spotify_album_name)
    g_name = _norm(((genius_song.get("album") or {}).get("name")) or "")
    if sp_name and g_name and _fuzzy_tokens_match_impl(_normalize_ws_tokens, sp_name, g_name, GENIUS_FUZZY_TOKENS_PERCENT, GENIUS_FUZZY_CHARS_PERCENT):
        score += GENIUS_SCORE_ALBUM_MATCH
        reasons.append(f"album_match +{GENIUS_SCORE_ALBUM_MATCH}")
    elif sp_name and g_name:
        score -= GENIUS_PENALTY_ALBUM_MISMATCH
        reasons.append(f"album_mismatch -{GENIUS_PENALTY_ALBUM_MISMATCH}")
    return score, reasons

def _title_matches(variant_track: str, genius_title: str) -> bool:
    vt_raw = _norm_title_value(variant_track)
    gt_raw = _norm_title_value(genius_title)
    vt_clr = _norm_title_value(variant_track, clear=True)
    gt_clr = _norm_title_value(genius_title, clear=True)

    vts = [vt_raw, vt_clr]
    gts = [gt_raw, gt_clr]
    for vt in vts:
        if not vt:
            continue
        for gt in gts:
            if not gt:
                continue
            if vt == gt:
                return True
            if len(vt) <= 3 or len(gt) <= 3:
                continue
            if vt in gt or gt in vt:
                return True
            if _fuzzy_tokens_match_impl(_tokens_with_charseqs, vt, gt, GENIUS_FUZZY_TOKENS_PERCENT, GENIUS_FUZZY_CHARS_PERCENT):
                return True
            if _fuzzy_tokens_match_impl(_tokens_with_charseqs, gt, vt, GENIUS_FUZZY_TOKENS_PERCENT, GENIUS_FUZZY_CHARS_PERCENT):
                return True
    return False

def _tag_match_score(tag: Optional[str], title: str, title_with_featured: str) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    t = _norm(tag)
    if not t:
        return 0, reasons
    title_n = _norm(title)
    titlef_n = _norm(title_with_featured)
    tokens = [w for w in _RE_WORDS.findall(t) if len(w) >= 2]
    phrase_in_title = False
    phrase_in_title_feat = False
    if t and t in title_n:
        phrase_in_title = True
    elif t and t in titlef_n:
        phrase_in_title_feat = True
    else:
        tt = float(GENIUS_TAG_FUZZY_TOKENS_PERCENT)
        tc = float(GENIUS_TAG_FUZZY_CHARS_PERCENT)
        raw_tag = tag or ""
        if raw_tag and _fuzzy_tag_match_in_text(raw_tag, title, tt, tc):
            phrase_in_title = True
        elif raw_tag and _fuzzy_tag_match_in_text(raw_tag, title_with_featured, tt, tc):
            phrase_in_title_feat = True

    if phrase_in_title:
        score += GENIUS_SCORE_TAG_PHRASE_IN_TITLE
        reasons.append(f"tag_phrase_title +{GENIUS_SCORE_TAG_PHRASE_IN_TITLE}")
    elif phrase_in_title_feat:
        score += GENIUS_SCORE_TAG_PHRASE_IN_TITLE_FEAT
        reasons.append(f"tag_phrase_title_feat +{GENIUS_SCORE_TAG_PHRASE_IN_TITLE_FEAT}")
    match_cnt = 0
    if tokens:
        match_cnt = sum(1 for tok in tokens if tok in title_n)
        if match_cnt:
            add = min(GENIUS_SCORE_TAG_TOKENS_MAX, match_cnt * GENIUS_SCORE_TAG_TOKENS_PER)
            score += add
            reasons.append(f"tag_tokens_title +{add} ({match_cnt} tokens)")
    has_any = phrase_in_title or phrase_in_title_feat or match_cnt > 0
    if not has_any:
        score -= GENIUS_PENALTY_TAG_MISSING
        reasons.append(f"tag_missing_penalty -{GENIUS_PENALTY_TAG_MISSING}")
    return score, reasons

def _relationship_tag_hint_score(tag: Optional[str], relationships: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    t = _norm(tag)
    if not t:
        return 0, reasons
    tokens = [w for w in _RE_WORDS.findall(t) if w]
    if not tokens:
        return 0, reasons
    found = False
    for rel in relationships or []:
        if not isinstance(rel, dict):
            continue
        for s in (rel.get("songs") or []):
            if not isinstance(s, dict):
                continue
            title = s.get("title") or ""
            title_feat = s.get("title_with_featured") or ""
            txt = f"{title} {title_feat}".strip()
            if not txt:
                continue
            n_txt = _norm(txt)
            if any(tok in n_txt for tok in tokens):
                score += GENIUS_SCORE_RELATIONSHIP_TAG_TOKENS
                reasons.append(f"tag_tokens_relationships +{GENIUS_SCORE_RELATIONSHIP_TAG_TOKENS}")
                found = True
                break
        if found:
            break
    return score, reasons

def _calc_title_featured_mismatch_penalty(spotify_track_name: Optional[str], title_with_featured: str, featured_pref: Optional[List[str]], genius_song: Dict[str, Any]) -> Tuple[int, List[str]]:
    def _tokset(s: Optional[str]) -> Set[str]:
        n = _normalize_base_text(s)
        if not n:
            return set()
        n = _RE_TOKSET_SEP.sub(" ", n)
        return {w for w in _RE_WORDS.findall(n) if w}

    def _strip_feat_from_title_for_penalty(s: Optional[str]) -> str:
        t = _safe_stripped(s)
        if not t:
            return ""
        return strip_feat_placeholders_bracketed(t)

    def _calc_penalty(track_value: Optional[str], featured_values: Optional[List[str]]) -> Tuple[int, List[str]]:
        penalty = 0
        reasons: List[str] = []
        tag_from_track, base_track = _extract_trailing_tag(track_value or "")
        base_tokens = _tokset(base_track)
        tag_tokens = _tokset(tag_from_track)
        baseline = base_tokens | tag_tokens
        gtokens = _tokset(_strip_feat_from_title_for_penalty(title_with_featured or ""))
        extra = sorted([t for t in gtokens if t not in baseline])
        missing_base = sorted([t for t in base_tokens if t not in gtokens])
        missing_tag = sorted([t for t in tag_tokens if t not in gtokens])
        missing_base_display = _format_reason_tokens_for_display(missing_base, base_track)
        missing_tag_display = _format_reason_tokens_for_display(missing_tag, tag_from_track)
        join_sep = ", "
        if extra:
            sub = min(GENIUS_PENALTY_TOKEN_CAP, len(extra) * GENIUS_PENALTY_TITLE_EXTRA_TOKEN_PER)
            penalty -= sub
            reasons.append(f"title_extra_tokens -{sub} ({len(extra)} tokens: {join_sep.join(extra)})")
        if missing_base:
            sub = min(GENIUS_PENALTY_TOKEN_CAP, len(missing_base) * GENIUS_PENALTY_TITLE_MISSING_TOKEN_PER)
            penalty -= sub
            reasons.append(
                f"title_missing_tokens -{sub} ({len(missing_base)} tokens: {join_sep.join(missing_base_display or missing_base)})"
            )
        if missing_tag:
            sub = min(GENIUS_PENALTY_TOKEN_CAP, len(missing_tag) * GENIUS_PENALTY_TAG_MISSING_TOKEN_PER)
            penalty -= sub
            reasons.append(
                f"tag_missing_tokens_title -{sub} ({len(missing_tag)} tokens: {join_sep.join(missing_tag_display or missing_tag)})"
            )

        sp_feat_tokens: Set[str] = set()
        for p in (featured_values or []):
            sp_feat_tokens |= _tokset(p)
        gfeat_names = _extract_featured_from_genius(genius_song)
        g_feat_tokens: Set[str] = set()
        for n in gfeat_names:
            g_feat_tokens |= _tokset(n)

        if sp_feat_tokens or g_feat_tokens:
            f_extra = sorted([t for t in g_feat_tokens if t not in sp_feat_tokens])
            f_missing = sorted([t for t in sp_feat_tokens if t not in g_feat_tokens])
            if f_extra:
                sub = min(GENIUS_PENALTY_TOKEN_CAP, len(f_extra) * GENIUS_PENALTY_FEATURED_EXTRA_TOKEN_PER)
                penalty -= sub
                reasons.append(f"featured_extra_tokens -{sub} ({len(f_extra)} tokens: {join_sep.join(f_extra)})")
            if f_missing:
                sub = min(GENIUS_PENALTY_TOKEN_CAP, len(f_missing) * GENIUS_PENALTY_FEATURED_MISSING_TOKEN_PER)
                penalty -= sub
                reasons.append(f"featured_missing_tokens -{sub} ({len(f_missing)} tokens: {join_sep.join(f_missing)})")
        return penalty, reasons

    featured_pref_raw = [_safe_stripped(p) for p in (featured_pref or []) if _safe_stripped(p)]
    featured_pref_anyascii = [
        variants[-1]
        for p in featured_pref_raw
        for variants in [_spotify_text_variants(p)]
        if variants
    ]
    best_penalty: Optional[int] = None
    best_reasons: List[str] = []
    for track_value, featured_values in [
        (_safe_stripped(spotify_track_name), featured_pref_raw),
        ((_spotify_text_variants(spotify_track_name) or [""])[-1], featured_pref_anyascii),
    ]:
        current_penalty, current_reasons = _calc_penalty(track_value, featured_values)
        if best_penalty is None or current_penalty > best_penalty:
            best_penalty = current_penalty
            best_reasons = current_reasons
    return best_penalty if best_penalty is not None else 0, best_reasons

def _slug_token_match_score(
    song: Dict[str, Any],
    variants: Iterable[Dict[str, Optional[str]]],
    spotify_track_name: Optional[str],
    precomputed_target_tokens: Optional[Set[str]] = None,
) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    slug = _song_slug(song)
    if not slug:
        return 0, reasons
    slug_tokens = [
        t for t in _tokens_with_charseqs(slug)
        if len(t) >= GENIUS_SLUG_TOKEN_MIN_LEN
    ]
    if not slug_tokens:
        return 0, reasons
    target_tokens = precomputed_target_tokens
    if target_tokens is None:
        target_tokens = _build_slug_target_tokens(variants, spotify_track_name)
    if not target_tokens:
        return 0, reasons
    matches = sorted({t for t in slug_tokens if t in target_tokens})
    if matches:
        add = len(matches) * GENIUS_SCORE_SLUG_TOKEN_PER
        score += add
        join_sep = ", "
        reasons.append(
            f"slug_tokens +{add} ({len(matches)} tokens: {join_sep.join(matches)})"
        )
    return score, reasons

def _build_slug_target_tokens(
    variants: Iterable[Dict[str, Optional[str]]],
    spotify_track_name: Optional[str],
) -> Set[str]:
    target_tokens: Set[str] = set()
    if spotify_track_name:
        target_tokens |= {
            t for t in _tokens_with_charseqs(spotify_track_name)
            if len(t) >= GENIUS_SLUG_TOKEN_MIN_LEN
        }
    for v in variants or []:
        if not isinstance(v, dict):
            continue
        for key in ("primary_artist", "featured_artists"):
            value = v.get(key)
            if not value:
                continue
            target_tokens |= {
                t for t in _tokens_with_charseqs(value)
                if len(t) >= GENIUS_SLUG_TOKEN_MIN_LEN
            }
    return target_tokens

def _build_query_from_variant(v: Dict[str, Optional[str]]) -> str:
    if not isinstance(v, dict):
        return ""
    parts: List[str] = []
    pa = str(v.get("primary_artist") or "")
    fea = str(v.get("featured_artists") or "")
    trk_raw = str(v.get("track") or "")
    tag = str(v.get("tag") or "")

    trk_src = trk_raw
    skip_track_tag_feat_extract = bool(v.get("_skip_track_tag_feat_extract"))
    if not skip_track_tag_feat_extract:
        tag_from_track, base_track = _extract_trailing_tag(trk_raw)
        if tag_from_track and base_track:
            trk_src = base_track
    trk_clean = normalize_query_value(trk_src) or ""
    pa = normalize_query_value(pa) or ""
    fea = normalize_query_value(fea) or ""
    tag_clean = normalize_query_value(tag) or ""

    if pa:
        parts.append(pa)
    if fea:
        parts.append(fea)
    if trk_clean:
        parts.append(trk_clean)

    if tag_clean:
        _nt = convert_separators_to_space(_norm(tag_clean))
        _ns = convert_separators_to_space(_norm(trk_clean))
        tag_tokens = set(_RE_WORDS.findall(_nt))
        track_tokens = set(_RE_WORDS.findall(_ns))
        pa_norm = _norm(pa)
        fea_norm = _norm(fea)
        pa_tokens = set(_RE_WORDS.findall(pa_norm))
        fea_tokens = set(_RE_WORDS.findall(fea_norm))
        artist_tokens = pa_tokens | fea_tokens
        single_word = len(tag_tokens) == 1
        has_punct = bool(_RE_NON_WORD_SPACE.search(tag_clean))
        if _nt and _nt not in _ns and not (tag_tokens & track_tokens) and not (tag_tokens & artist_tokens) and not (single_word and has_punct):
            parts.append(tag_clean)

    q = _collapse_ws(" ".join(parts))
    return normalize_query_value(q) or ""

@lru_cache(maxsize=4096)
def _build_query_from_variant_cached(
    primary_artist: Optional[str],
    featured_artists: Optional[str],
    track: Optional[str],
    tag: Optional[str],
    skip_track_tag_feat_extract: bool,
) -> str:
    return _build_query_from_variant(
        {
            "primary_artist": primary_artist,
            "featured_artists": featured_artists,
            "track": track,
            "tag": tag,
            "_skip_track_tag_feat_extract": skip_track_tag_feat_extract,
        }
    )

def _cached_query_from_variant(variant: Optional[Dict[str, Any]]) -> str:
    if not isinstance(variant, dict):
        return ""
    return _build_query_from_variant_cached(
        variant.get("primary_artist"),
        variant.get("featured_artists"),
        variant.get("track"),
        variant.get("tag"),
        bool(variant.get("_skip_track_tag_feat_extract")),
    )

def _query_dedupe_key(q: Optional[str]) -> str:
    if not q:
        return ""
    return _collapse_ws(_safe_stripped(q)).lower()

def _accept_stream_query(
    q: Optional[str],
    seen_local: Set[str],
    *,
    shared_seen_keys: Optional[Set[str]] = None,
) -> Optional[str]:
    qn = _query_dedupe_key(q)
    if not qn or qn in seen_local:
        return None
    seen_local.add(qn)
    if shared_seen_keys is not None:
        if qn in shared_seen_keys:
            return None
        shared_seen_keys.add(qn)
    return q

def _song_url(song: Dict[str, Any]) -> Optional[str]:
    if not isinstance(song, dict):
        return None
    url = song.get("url")
    if isinstance(url, str) and url:
        u = url.split("/q/", 1)[0]
        low = u.lower()
        if "genius.com/" in low and "/albums/" not in low and "/artists/" not in low:
            return u
    return None

def _song_slug(song: Dict[str, Any]) -> str:
    if not isinstance(song, dict):
        return ""
    slug_raw = song.get("path")
    slug = slug_raw.strip() if isinstance(slug_raw, str) else ""
    if not slug:
        url_raw = song.get("url")
        slug = url_raw.strip() if isinstance(url_raw, str) else ""
    if not slug:
        return ""
    if slug.startswith("http://") or slug.startswith("https://"):
        try:
            slug = urlsplit(slug).path or ""
        except ValueError:
            slug = slug.split("?", 1)[0]
    slug = slug.split("?", 1)[0].split("#", 1)[0].strip()
    if not slug:
        return ""
    if slug.startswith("/"):
        slug = slug[1:]
    if "/" in slug:
        slug = slug.strip("/")
        if "/" in slug:
            slug = slug.split("/")[-1]
    return slug

def _resolve_original_if_translation(song: Dict[str, Any], details_by_id: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(song, dict):
        return song
    for rel in song.get("song_relationships") or []:
        if not isinstance(rel, dict):
            continue
        rt = str(rel.get("relationship_type") or "").lower()
        if rt != "translation_of":
            continue
        for s in (rel.get("songs") or []):
            if not isinstance(s, dict):
                continue
            rid_int = _safe_int_or_none(s.get("id"))
            if rid_int is not None and details_by_id and rid_int in details_by_id:
                return details_by_id[rid_int]
            return s
    return song

def _resolve_song_url(song: Optional[Dict[str, Any]], details_by_id: Dict[int, Dict[str, Any]]) -> Optional[str]:
    if not song:
        return None
    url_before = _song_url(song)
    if _is_mistral_enabled():
        return url_before
    resolved = _resolve_original_if_translation(song, details_by_id)
    url_resolved = _song_url(resolved)
    url = url_resolved or url_before
    if url_resolved and url_before and url_resolved != url_before:
        if _is_genius_detailed_log():
            _safe_log_info(INFO_TRANSLATION_REPLACED, url_before, url_resolved)
    return url

def _score_song_detailed(song: Dict[str, Any], variants: Iterable[Dict[str, Optional[str]]], *, featured_pref: Optional[List[str]] = None, top1_counts: Optional[Dict[int, int]] = None, spotify_track_name: Optional[str] = None, spotify_album_name: Optional[str] = None, spotify_album_total_tracks: Optional[int] = None, slug_target_tokens: Optional[Set[str]] = None) -> Tuple[int, List[str]]:
    if not isinstance(song, dict):
        return 0, []
    score = 0
    reasons_all: List[str] = []
    title = song.get("title") or ""
    title_with_feat = song.get("title_with_featured") or ""
    relationships = song.get("song_relationships")
    if not isinstance(relationships, list):
        relationships = []
    variants_list = variants if isinstance(variants, list) else list(variants or [])
    title_exact_norm = _norm_title_value(title, clear=True)
    featured_score, featured_reasons = _featured_match_score(featured_pref, song, title_with_feat)

    best_var: Optional[int] = None
    best_var_reasons: List[str] = []
    for v in variants_list:
        if not isinstance(v, dict):
            continue
        v_add = 0
        v_reasons: List[str] = []
        tr = v.get("track") or ""
        if tr and (
            _fuzzy_tokens_match_impl(_tokens_with_charseqs, tr, title, GENIUS_FUZZY_TOKENS_PERCENT, GENIUS_FUZZY_CHARS_PERCENT)
            or _fuzzy_tokens_match_impl(_tokens_with_charseqs, tr, title_with_feat, GENIUS_FUZZY_TOKENS_PERCENT, GENIUS_FUZZY_CHARS_PERCENT)
        ):
            left = _norm_title_value(tr, clear=True)
            if left and left == title_exact_norm:
                v_add += GENIUS_SCORE_TRACK_EXACT_TITLE_BONUS
                v_reasons.append(f"exact_title +{GENIUS_SCORE_TRACK_EXACT_TITLE_BONUS}")
        ts, treasons = _tag_match_score(v.get("tag"), title, title_with_feat)
        v_add += ts
        v_reasons.extend(treasons)
        rs, rreasons = _relationship_tag_hint_score(v.get("tag"), relationships)
        v_add += rs
        v_reasons.extend(rreasons)
        v_add += featured_score
        v_reasons.extend(featured_reasons)
        if best_var is None or v_add > best_var:
            best_var = v_add
            best_var_reasons = v_reasons
    mp, mpreasons = _calc_title_featured_mismatch_penalty(spotify_track_name, title_with_feat, featured_pref, song)
    score += (best_var or 0) + mp
    reasons_all.extend(best_var_reasons)
    reasons_all.extend(mpreasons)

    cs, creasons = _consec_title_tokens_score(spotify_track_name, title, title_with_feat)
    score += cs
    reasons_all.extend(creasons)

    ss, sreasons = _slug_token_match_score(
        song,
        variants_list,
        spotify_track_name,
        precomputed_target_tokens=slug_target_tokens,
    )
    score += ss
    reasons_all.extend(sreasons)

    ascore, areasons = _album_match_score(spotify_album_name, spotify_album_total_tracks, song)
    score += ascore
    reasons_all.extend(areasons)
    sid = _safe_int_or_none(song.get("id"))
    top1_map = top1_counts if isinstance(top1_counts, dict) else None
    if sid is not None and top1_map and sid in top1_map:
        k = int(top1_map[sid])
        add_top = min(GENIUS_TOP1_BONUS_MAX, k * GENIUS_TOP1_BONUS_PER)
        if add_top:
            score += add_top
            reasons_all.append(f"top1_bonus +{add_top} ({k} hits)")
    return score, reasons_all

def _extract_artist_names_from_spotify(artists: Optional[List[Dict[str, Any]]]) -> List[str]:
    if not artists:
        return []
    names: List[str] = []
    seen: Set[str] = set()
    for artist in artists:
        if not isinstance(artist, dict):
            continue
        raw_name = str(artist.get("name") or "").strip()
        if not raw_name:
            continue
        split_names = split_artist_text_by_feat_markers(raw_name)
        if not split_names:
            split_names = [raw_name]
        for name in split_names:
            clean_name = _collapse_ws(normalize_apostrophes(name))
            if not clean_name:
                continue
            key = _norm(clean_name)
            if not key or key in seen:
                continue
            seen.add(key)
            names.append(clean_name)
    return names

def _build_artist_combinations(names: List[str]) -> List[Tuple[str, Optional[str]]]:
    out: List[Tuple[str, Optional[str]]] = []
    seen: Set[str] = set()

    def _add(seq: List[str]) -> None:
        seq2 = [s for s in (seq or []) if s]
        if not seq2:
            return
        key_parts = [_norm(s) for s in seq2]
        key = "|".join([k for k in key_parts if k])
        if not key or key in seen:
            return
        seen.add(key)
        pa = seq2[0]
        fea = ", ".join([s for s in seq2[1:] if s]) if len(seq2) > 1 else None
        out.append((pa, fea if fea else None))

    if not names:
        return []

    primary = names[0]
    featured = [n for n in names[1:] if n]

    _add([primary])
    _add([primary] + featured)
    if featured:
        _add(featured)

    for f in featured:
        _add([primary, f])
    for f in featured:
        _add([f])
    for i in range(len(featured)):
        for j in range(i + 1, len(featured)):
            _add([featured[i], featured[j]])
    return out

_REMIX_MARKERS_SINGLE: set[str] | None = None
_REMIX_MARKERS_MULTI: list[list[str]] | None = None
_REMIX_MARKERS_VERSION: int | None = None

def _get_remix_markers_norm() -> Tuple[set[str], list[list[str]]]:
    global _REMIX_MARKERS_SINGLE, _REMIX_MARKERS_MULTI, _REMIX_MARKERS_VERSION
    cur_version = get_tag_patterns_version()
    if (
        _REMIX_MARKERS_SINGLE is not None
        and _REMIX_MARKERS_MULTI is not None
        and _REMIX_MARKERS_VERSION == cur_version
    ):
        return _REMIX_MARKERS_SINGLE, _REMIX_MARKERS_MULTI
    single: set[str] = set()
    multi: list[list[str]] = []
    patterns = get_tag_patterns()
    for raw in patterns:
        toks = _normalize_marker_tokens(raw)
        if not toks:
            continue
        if len(toks) > 1:
            multi.append(toks)
        else:
            single.add(toks[0])
    _REMIX_MARKERS_SINGLE = single
    _REMIX_MARKERS_MULTI = multi
    _REMIX_MARKERS_VERSION = cur_version
    return single, multi

def _remix_pattern_in_tokens(tokens: List[str]) -> bool:
    if not tokens:
        return False
    single, multi = _get_remix_markers_norm()
    for t in tokens:
        if t in single:
            return True
    return _best_multi_pattern(tokens, multi) is not None

def _remix_pattern_in_tokens_match(tokens: List[str]) -> Optional[str]:
    if not tokens:
        return None
    single, multi = _get_remix_markers_norm()
    best_multi = _best_multi_pattern(tokens, multi)
    if best_multi is not None:
        return " ".join(best_multi).strip()

    for t in tokens:
        if t in single:
            return t
    return None

def _remix_tag_matched_pattern(tag: Optional[str]) -> Optional[str]:
    if not tag:
        return None
    tokens = _normalize_marker_tokens(tag)
    strict_pat = _remix_pattern_in_tokens_match(tokens)
    if strict_pat:
        return strict_pat

    tc = float(GENIUS_TAG_FUZZY_CHARS_PERCENT)
    tt = float(GENIUS_TAG_FUZZY_TOKENS_PERCENT)
    if tt <= 0 or tc <= 0:
        return None

    single, _ = _get_remix_markers_norm()
    best_pat: Optional[str] = None
    best_score: float = 0.0
    for q in tokens:
        for pat in single:
            sc = _tag_token_similarity_consecutive(q, pat)
            if sc >= tc and sc > best_score:
                best_score = sc
                best_pat = pat
    return best_pat

def _extract_inline_remix_tag(s: str) -> Tuple[Optional[str], Optional[str]]:
    t = _safe_stripped(s)
    if not t:
        return None, None
    t_clean = _collapse_ws(t)
    base_norm = normalize_letters(t)
    base_norm = _collapse_ws(base_norm)
    if not base_norm:
        return None, None
    tokens_orig = base_norm.split()
    tokens = [p.lower() for p in tokens_orig]
    single, multi = _get_remix_markers_norm()
    n = len(tokens)
    best_len = 0
    for pat in multi:
        m = len(pat)
        if m == 0 or m > n:
            continue
        if tokens[-m:] == pat and m > best_len:
            best_len = m
    if best_len > 0:
        start = n - best_len
        if start == 0:
            return None, t_clean
        base_tokens = tokens_orig[:start]
        tag_tokens = tokens_orig[start:]
        base = " ".join(base_tokens).strip()
        tag = " ".join(tag_tokens).strip()
        return tag, base
    if tokens[-1] in single:
        if len(tokens) == 1:
            return None, t_clean
        base_tokens = tokens_orig[:-1]
        tag_tokens = [tokens_orig[-1]]
        base = " ".join(base_tokens).strip()
        tag = " ".join(tag_tokens).strip()
        return tag, base
    return None, t_clean

def _remix_tag_match_type(tag: Optional[str]) -> int:
    if not tag:
        return 0
    tokens = _normalize_marker_tokens(tag)
    if _remix_pattern_in_tokens(tokens):
        return 2
    tc = float(GENIUS_TAG_FUZZY_CHARS_PERCENT)
    tt = float(GENIUS_TAG_FUZZY_TOKENS_PERCENT)
    if tt <= 0 or tc <= 0:
        return 0

    single, _ = _get_remix_markers_norm()
    for q in tokens:
        for pat in single:
            sc = _tag_token_similarity_consecutive(q, pat)
            if sc >= tc:
                return 1
    return 0

def _recompose_base_with_suffixes(base: str, suffixes: List[str]) -> str:
    parts = [base] if base else []
    parts.extend([s for s in (suffixes or []) if s])
    return _collapse_ws(" ".join(parts))

def _clean_base_for_extract(base: Optional[str], *, strip_trailing_dash: bool = False) -> str:
    base_clean = _collapse_ws(base or "")
    if base_clean:
        stripped = strip_feat_placeholders_bracketed(base_clean)
        if stripped:
            base_clean = stripped
        if strip_trailing_dash:
            base_clean = _RE_TRAILING_DASH.sub("", base_clean).strip()
    return base_clean

def _extract_trailing_tag(s: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not s:
        return None, None
    t0 = _collapse_ws(s)
    if not t0:
        return None, None

    t = t0
    suffixes_keep: List[str] = []
    best_fuzzy: Optional[Tuple[str, str]] = None
    saw_unknown_or_fuzzy_suffix = False
    removed_only_feat_suffix = True

    while True:
        m = _RE_EXTRACT_TAG_BR.search(t)
        mode = ""
        if m:
            mode = "br"
        else:
            m = _RE_EXTRACT_TAG_DASH2.search(t)
            if m:
                mode = "dash2"
            else:
                m = _RE_EXTRACT_TAG_DASH1.search(t)
                if m:
                    mode = "dash1"

        if not m:
            break

        if mode == "br":
            inner = (m.group(1) or "").strip()
            base = t[:m.start()].strip()
            suffix_raw = _collapse_ws(t[m.start():])
            base_clean = _clean_base_for_extract(base, strip_trailing_dash=True)
        else:
            inner = (m.group(2) or "").strip()
            base = (m.group(1) or "").strip()
            base_clean = _clean_base_for_extract(base)
            suffix_raw = _collapse_ws(t[m.span(1)[1]:])

        if _is_feat_placeholder_inner(inner):
            t = base_clean
            continue

        removed_only_feat_suffix = False

        mt = _remix_tag_match_type(inner)
        if mt == 2:
            return inner, _recompose_base_with_suffixes(base_clean, suffixes_keep)
        if mt == 1:
            saw_unknown_or_fuzzy_suffix = True
            if best_fuzzy is None:
                best_fuzzy = (inner, _recompose_base_with_suffixes(base_clean, suffixes_keep))
            if suffix_raw:
                suffixes_keep.insert(0, suffix_raw)
            t = base_clean
            continue
        saw_unknown_or_fuzzy_suffix = True
        if suffix_raw:
            suffixes_keep.insert(0, suffix_raw)
        t = base_clean

    if best_fuzzy is not None:
        return best_fuzzy[0], best_fuzzy[1]
    if saw_unknown_or_fuzzy_suffix:
        out_full = _recompose_base_with_suffixes(t, suffixes_keep)
        full_clean = strip_feat_placeholders_bracketed(out_full)
        if full_clean:
            return None, full_clean
        return None, out_full
    if removed_only_feat_suffix:
        tag_inline, base_inline = _extract_inline_remix_tag(t)
        if tag_inline:
            base_clean = _clean_base_for_extract(base_inline)
            return tag_inline, base_clean

    full_clean = strip_feat_placeholders_bracketed(_recompose_base_with_suffixes(t, suffixes_keep))
    if full_clean:
        return None, full_clean
    return None, _recompose_base_with_suffixes(t, suffixes_keep)

def _split_tag_phrase_variants(tag: Optional[str]) -> List[str]:
    if not tag:
        return []
    t = _safe_stripped(tag)
    if not t:
        return []
    t = _RE_TAG_DASH_BOTH.sub(" ", t)
    t = _RE_TAG_DASH_LEFT.sub(r"\1", t)
    t = _RE_TAG_DASH_RIGHT.sub(" ", t)
    t = _collapse_ws(t)
    if not t:
        return []
    parts = [p for p in _RE_WORDS.findall(t) if p]
    out: List[str] = []
    seen = set()
    if t and t.lower() not in seen:
        seen.add(t.lower())
        out.append(t)
    for p in parts:
        pl = p.strip()
        if pl and pl.lower() not in seen:
            seen.add(pl.lower())
            out.append(pl)
    n = len(parts)
    for i in range(n):
        for j in range(i + 1, n):
            pair = f"{parts[i]} {parts[j]}".strip()
            if pair and pair.lower() not in seen:
                seen.add(pair.lower())
                out.append(pair)
    return out

VariantFieldsKey = Tuple[Optional[str], Optional[str], Optional[str], Optional[str], bool]
VariantDedupKey = Tuple[str, str, str, str, bool]

def _variant_fields_key(variant: Optional[Dict[str, Any]]) -> VariantFieldsKey:
    if not isinstance(variant, dict):
        return (None, None, None, None, False)
    return (
        variant.get("primary_artist"),
        variant.get("featured_artists") or None,
        variant.get("track"),
        variant.get("tag") or None,
        bool(variant.get("_skip_track_tag_feat_extract")),
    )

def _variant_dedupe_key_from_fields(key: VariantFieldsKey) -> VariantDedupKey:
    pa, fea, tr, tg, skip_track_tag_feat_extract = key
    return (
        str(pa or "").lower(),
        str(fea or "").lower(),
        str(tr or "").lower(),
        str(tg or "").lower(),
        bool(skip_track_tag_feat_extract),
    )

def _variant_from_fields_key(key: VariantFieldsKey) -> Dict[str, Optional[str]]:
    pa, fea, tr, tg, skip_track_tag_feat_extract = key
    item: Dict[str, Optional[str]] = {
        "primary_artist": pa,
        "featured_artists": fea if fea else None,
        "track": tr,
        "tag": tg if tg else None,
    }
    if skip_track_tag_feat_extract:
        item["_skip_track_tag_feat_extract"] = True
    return item

def _expand_variant_fields_key(
    primary_artist: Optional[str],
    featured_artists: Optional[str],
    track: Optional[str],
    tag: Optional[str],
    skip_track_tag_feat_extract: bool,
    field_variants_builder: Callable[[Optional[str]], Tuple[Optional[str], ...]],
) -> Tuple[VariantFieldsKey, ...]:
    alt_fields: List[Tuple[str, Optional[str], Optional[str]]] = []
    for name, value in (
        ("primary_artist", primary_artist),
        ("featured_artists", featured_artists),
        ("track", track),
        ("tag", tag),
    ):
        value_variants = field_variants_builder(value)
        original = value_variants[0] if value_variants else value
        alternative = next((item for item in value_variants if item != original), original)
        if alternative != original:
            alt_fields.append((name, original, alternative))

    out: List[VariantFieldsKey] = []
    seen: Set[VariantDedupKey] = set()
    combo_count = 1 << len(alt_fields)
    for mask in sorted(range(combo_count), key=lambda current: (current.bit_count(), current)):
        pa = primary_artist
        fea = featured_artists
        tr = track
        tg = tag
        for idx, (name, original, alternative) in enumerate(alt_fields):
            value = alternative if ((mask >> idx) & 1) else original
            if name == "primary_artist":
                pa = value
            elif name == "featured_artists":
                fea = value
            elif name == "track":
                tr = value
            else:
                tg = value
        item_key = (pa, fea if fea else None, tr, tg if tg else None, bool(skip_track_tag_feat_extract))
        dedupe_key = _variant_dedupe_key_from_fields(item_key)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        out.append(item_key)
    return tuple(out)

@lru_cache(maxsize=4096)
def _expand_normalized_variant_fields_key_cached(
    primary_artist: Optional[str],
    featured_artists: Optional[str],
    track: Optional[str],
    tag: Optional[str],
    skip_track_tag_feat_extract: bool,
) -> Tuple[VariantFieldsKey, ...]:
    return _expand_variant_fields_key(
        primary_artist,
        featured_artists,
        track,
        tag,
        skip_track_tag_feat_extract,
        _normalized_field_variants_cached,
    )

@lru_cache(maxsize=4096)
def _expand_symbolized_variant_fields_key_cached(
    primary_artist: Optional[str],
    featured_artists: Optional[str],
    track: Optional[str],
    tag: Optional[str],
    skip_track_tag_feat_extract: bool,
) -> Tuple[VariantFieldsKey, ...]:
    return _expand_variant_fields_key(
        primary_artist,
        featured_artists,
        track,
        tag,
        skip_track_tag_feat_extract,
        _symbol_field_variants_cached,
    )

@lru_cache(maxsize=4096)
def _expand_anyascii_variant_fields_key_cached(
    primary_artist: Optional[str],
    featured_artists: Optional[str],
    track: Optional[str],
    tag: Optional[str],
    skip_track_tag_feat_extract: bool,
) -> Tuple[VariantFieldsKey, ...]:
    return _expand_variant_fields_key(
        primary_artist,
        featured_artists,
        track,
        tag,
        skip_track_tag_feat_extract,
        _anyascii_field_variants_cached,
    )

def _iter_variant_fields_keys_as_variants(items: Tuple[VariantFieldsKey, ...]) -> Iterator[Dict[str, Optional[str]]]:
    for item in items:
        yield _variant_from_fields_key(item)

def _iter_normalized_variant_expansions(variant: Optional[Dict[str, Any]]) -> Iterator[Dict[str, Optional[str]]]:
    if not isinstance(variant, dict):
        return iter(())
    return _iter_variant_fields_keys_as_variants(
        _expand_normalized_variant_fields_key_cached(*_variant_fields_key(variant))
    )

def _iter_symbolized_variant_expansions(variant: Optional[Dict[str, Any]]) -> Iterator[Dict[str, Optional[str]]]:
    if not isinstance(variant, dict):
        return iter(())
    return _iter_variant_fields_keys_as_variants(
        _expand_symbolized_variant_fields_key_cached(*_variant_fields_key(variant))
    )

def _iter_anyascii_variant_expansions(variant: Optional[Dict[str, Any]]) -> Iterator[Dict[str, Optional[str]]]:
    if not isinstance(variant, dict):
        return iter(())
    return _iter_variant_fields_keys_as_variants(
        _expand_anyascii_variant_fields_key_cached(*_variant_fields_key(variant))
    )

class _LazyExpandedVariants:
    def __init__(
        self,
        source_variants: Optional[Iterable[Dict[str, Optional[str]]]],
        expander: Callable[[Optional[Dict[str, Any]]], Iterable[Dict[str, Optional[str]]]],
    ) -> None:
        self._source_iter = iter(source_variants if source_variants is not None else [])
        self._expander = expander
        self._cache: List[Dict[str, Optional[str]]] = []
        self._current_iter: Optional[Iterator[Dict[str, Optional[str]]]] = None
        self._done = False
        self._lock = threading.Lock()
        self._seen: Set[VariantDedupKey] = set()

    def _advance_locked(self) -> bool:
        while True:
            if self._current_iter is None:
                try:
                    source_variant = next(self._source_iter)
                except StopIteration:
                    self._done = True
                    return False
                self._current_iter = iter(self._expander(source_variant))
            try:
                item = next(self._current_iter)
            except StopIteration:
                self._current_iter = None
                continue
            if not isinstance(item, dict):
                continue
            item_key = _variant_dedupe_key_from_fields(_variant_fields_key(item))
            if item_key in self._seen:
                continue
            self._seen.add(item_key)
            self._cache.append(item)
            return True

    def _get(self, index: int) -> Optional[Dict[str, Optional[str]]]:
        while True:
            with self._lock:
                if index < len(self._cache):
                    return self._cache[index]
                if self._done:
                    return None
                if not self._advance_locked() and self._done:
                    return None

    def __iter__(self) -> Iterator[Dict[str, Optional[str]]]:
        index = 0
        while True:
            item = self._get(index)
            if item is None:
                break
            yield item
            index += 1

    def materialize(self) -> List[Dict[str, Optional[str]]]:
        index = 0
        while self._get(index) is not None:
            index += 1
        with self._lock:
            return list(self._cache)

def _ensure_tag_tokenization(variants: List[Dict[str, Optional[str]]], original_track: Optional[str] = None) -> List[Dict[str, Optional[str]]]:
    out: List[Dict[str, Optional[str]]] = []
    seen = set()

    def _append_variant(pa: Optional[str], fea: Optional[str], tr: Optional[str], tg: Optional[str]) -> None:
        tag_val = tg if tg else None
        key = (str(pa or "").lower(), str(fea or "").lower(), str(tr or "").lower(), str(tag_val or "").lower())
        if key in seen:
            return
        seen.add(key)
        out.append({"primary_artist": pa, "featured_artists": fea, "track": tr, "tag": tag_val})

    for v in variants:
        if not isinstance(v, dict):
            continue
        pa = v.get("primary_artist")
        fea = v.get("featured_artists")
        tr = v.get("track")
        tg = v.get("tag") or ""
        _append_variant(pa, fea, tr, tg)
        if tg:
            for p in _split_tag_phrase_variants(tg):
                _append_variant(pa, fea, tr, p)
            continue
        source_for_tag = original_track or tr or ""
        if source_for_tag:
            tag_from_track, base_track = _extract_trailing_tag(source_for_tag)
            if tag_from_track:
                tr_clean = base_track
                _append_variant(pa, fea, tr_clean, None)
                for p in _split_tag_phrase_variants(tag_from_track):
                    _append_variant(pa, fea, tr_clean, p)
    return out

def _expand_variants_with_roman_l_i(variants: List[Dict[str, Optional[str]]]) -> List[Dict[str, Optional[str]]]:
    out: List[Dict[str, Optional[str]]] = []
    seen: Set[Tuple[str, str, str, str]] = set()

    def _append_variant(pa: Optional[str], fea: Optional[str], tr: Optional[str], tg: Optional[str]) -> None:
        key = (
            str(pa or "").lower(),
            str(fea or "").lower(),
            str(tr or "").lower(),
            str(tg or "").lower(),
        )
        if key in seen:
            return
        seen.add(key)
        out.append(
            {
                "primary_artist": pa,
                "featured_artists": fea,
                "track": tr,
                "tag": tg if tg else None,
            }
        )

    for v in variants:
        if not isinstance(v, dict):
            continue
        pa = v.get("primary_artist")
        fea = v.get("featured_artists")
        tr = v.get("track")
        tg = v.get("tag")
        track_variants = generate_roman_l_i_variants(tr)
        if not track_variants:
            track_variants = [tr]
        for tr_alt in track_variants:
            _append_variant(pa, fea, tr_alt, tg)
    return out

def _meta_str(val: Optional[str]) -> str:
    s = _safe_stripped(val)
    return s if s else ""

def _split_title_and_tag(value: Optional[str]) -> Tuple[str, str]:
    tag, base = _extract_trailing_tag(value or "")
    base_clean = _collapse_ws(base or "")
    tag_clean = _collapse_ws(tag) if tag else ""
    return base_clean, tag_clean

def _build_spotify_track_meta(
    primary: str,
    featured: Optional[str],
    track_name: Optional[str],
    spotify_album_name: Optional[str],
    spotify_release_date: Optional[str],
) -> Dict[str, str]:
    title, version_tag = _split_title_and_tag(track_name)

    return {
        "primary_artist": _meta_str(primary),
        "featured_artists": _meta_str(featured),
        "title": _meta_str(title),
        "version_tag": _meta_str(version_tag),
        "album": _meta_str(spotify_album_name),
        "release_date": _meta_str(spotify_release_date),
    }

def _build_genius_release_date(song: Dict[str, Any]) -> str:
    if not isinstance(song, dict):
        return ""
    release_date = song.get("release_date")
    if not release_date or not isinstance(release_date, str):
        return ""
    return release_date.strip()

def _build_genius_candidates_meta(
    scored_sorted: List[Tuple[Dict[str, Any], int, List[str]]],
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for idx, (song, _, _) in enumerate(scored_sorted, start=1):
        primary_artist_name = ""
        album_name = ""
        title_src = ""
        url_raw: Optional[str] = None
        if isinstance(song, dict):
            primary_artist = song.get("primary_artist")
            if isinstance(primary_artist, dict):
                primary_artist_name = primary_artist.get("name") or ""
            album = song.get("album")
            if isinstance(album, dict):
                album_name = album.get("name") or ""
            title_src = song.get("title_with_featured") or song.get("title") or ""
            url_raw = _song_url(song) or (song.get("url") if isinstance(song.get("url"), str) else None)
        featured_names = _extract_featured_from_genius(song)
        featured_text = ", ".join(n for n in featured_names if n) if featured_names else ""
        title_out, version_tag_out = _split_title_and_tag(title_src)

        rel_date = _build_genius_release_date(song)

        meta = {
            "idx": _meta_str(str(idx)),
            "url": _meta_str(url_raw),
            "primary_artist": _meta_str(primary_artist_name),
            "featured_artists": _meta_str(featured_text),
            "title": _meta_str(title_out),
            "version_tag": _meta_str(version_tag_out),
            "album": _meta_str(album_name),
            "release_date": _meta_str(rel_date),
        }
        out.append(meta)
    return out

class _VariantQueryStream:
    def __init__(
        self,
        variants_list: Iterable[Dict[str, Optional[str]]],
        limit: int,
        *,
        seen_keys: Optional[Set[str]] = None,
    ) -> None:
        self._variants_iter = iter(variants_list if variants_list is not None else [])
        self._limit = max(0, limit)
        self._emitted = 0
        self._seen_local: Set[str] = set(seen_keys or set())

    def take_batch(self, batch_size: int, *, shared_seen_keys: Optional[Set[str]] = None) -> List[str]:
        if self._limit <= 0 or batch_size <= 0:
            return []
        out: List[str] = []
        while len(out) < batch_size and self._emitted < self._limit:
            try:
                variant = next(self._variants_iter)
            except StopIteration:
                break
            q = _cached_query_from_variant(variant)
            if not q:
                continue
            accepted_query = _accept_stream_query(
                q,
                self._seen_local,
                shared_seen_keys=shared_seen_keys,
            )
            if not accepted_query:
                continue
            out.append(accepted_query)
            self._emitted += 1
        return out

class _AnyasciiQueryStream:
    def __init__(
        self,
        variants_list: Iterable[Dict[str, Optional[str]]],
        limit: int,
        *,
        seen_keys: Optional[Set[str]] = None,
    ) -> None:
        self._variants_iter = iter(variants_list if variants_list is not None else [])
        self._limit = max(0, limit)
        self._emitted = 0
        self._seen_local: Set[str] = set(seen_keys or set())
        self._current_iter: Optional[Iterator[str]] = None

    def _variant_iter(self, variant: Optional[Dict[str, Any]]) -> Iterator[str]:
        if not isinstance(variant, dict):
            return iter(())
        source_key = _variant_dedupe_key_from_fields(_variant_fields_key(variant))

        def _iter() -> Iterator[str]:
            for item in _iter_anyascii_variant_expansions(variant):
                item_key = _variant_dedupe_key_from_fields(_variant_fields_key(item))
                if item_key == source_key:
                    continue
                q = _cached_query_from_variant(item)
                if q:
                    yield q

        return _iter()

    def take_batch(self, batch_size: int, *, shared_seen_keys: Optional[Set[str]] = None) -> List[str]:
        if self._limit <= 0 or batch_size <= 0:
            return []
        out: List[str] = []
        while len(out) < batch_size and self._emitted < self._limit:
            if self._current_iter is None:
                try:
                    variant = next(self._variants_iter)
                except StopIteration:
                    break
                self._current_iter = self._variant_iter(variant)
            try:
                q = next(self._current_iter)
            except StopIteration:
                self._current_iter = None
                continue
            if not q:
                continue
            accepted_query = _accept_stream_query(
                q,
                self._seen_local,
                shared_seen_keys=shared_seen_keys,
            )
            if not accepted_query:
                continue
            out.append(accepted_query)
            self._emitted += 1
        return out

@lru_cache(maxsize=8192)
def _symbol_query_variants_cached(query: str, limit: int) -> Tuple[str, ...]:
    q = _safe_stripped(query)
    limit_int = _safe_int(limit, 0)
    if not q or limit_int <= 0:
        return ()
    return tuple(iter_symbol_letter_equiv_variants(q, limit_int))

class _SymbolQueryStream:
    def __init__(
        self,
        base_queries_snapshot: List[str],
        limit: int,
        *,
        seen_keys: Optional[Set[str]] = None,
    ) -> None:
        self._base_queries = list(base_queries_snapshot or [])
        self._limit = max(0, limit)
        self._query_index = 0
        self._emitted = 0
        self._seen_local: Set[str] = set(seen_keys or set())
        self._current_iter: Optional[Iterator[str]] = None

    def take_batch(self, batch_size: int, *, shared_seen_keys: Optional[Set[str]] = None) -> List[str]:
        if self._limit <= 0 or batch_size <= 0:
            return []
        out: List[str] = []
        while len(out) < batch_size and self._emitted < self._limit:
            if self._current_iter is None:
                if self._query_index >= len(self._base_queries):
                    break
                variant_limit = self._limit - self._emitted + len(self._seen_local) + 1
                self._current_iter = iter(
                    _symbol_query_variants_cached(
                        self._base_queries[self._query_index],
                        variant_limit,
                    )
                )
                self._query_index += 1
            try:
                q_sym = next(self._current_iter)
            except StopIteration:
                self._current_iter = None
                continue
            if not q_sym:
                continue
            accepted_query = _accept_stream_query(
                q_sym,
                self._seen_local,
                shared_seen_keys=shared_seen_keys,
            )
            if not accepted_query:
                continue
            out.append(accepted_query)
            self._emitted += 1
        return out

class _ModeQueryStream:
    def __init__(
        self,
        variants_groups: List[Iterable[Dict[str, Optional[str]]]],
        artist_modes: List[str],
        track_modes: List[str],
        found_initial_tag_or_feat_placeholders: bool,
        limit: int,
        *,
        seen_keys: Optional[Set[str]] = None,
    ) -> None:
        self._variants_groups = [iter(group if group is not None else []) for group in (variants_groups or [])]
        self._artist_modes = list(artist_modes or [])
        self._track_modes = list(track_modes or [])
        self._found_initial_tag_or_feat_placeholders = bool(found_initial_tag_or_feat_placeholders)
        self._limit = max(0, limit)
        self._group_index = 0
        self._emitted = 0
        self._seen_local: Set[str] = set(seen_keys or set())
        self._current_iter: Optional[Iterator[str]] = None

    def take_batch(self, batch_size: int, *, shared_seen_keys: Optional[Set[str]] = None) -> List[str]:
        if self._limit <= 0 or batch_size <= 0:
            return []
        out: List[str] = []
        while len(out) < batch_size and self._emitted < self._limit:
            if self._current_iter is None:
                while self._group_index < len(self._variants_groups):
                    current_variants = self._variants_groups[self._group_index]
                    try:
                        variant = next(current_variants)
                    except StopIteration:
                        self._group_index += 1
                        continue
                    self._current_iter = iter(
                        _build_mode_queries(
                            variant,
                            self._artist_modes,
                            self._track_modes,
                            self._found_initial_tag_or_feat_placeholders,
                        )
                    )
                    break
                if self._current_iter is None:
                    break
            try:
                q2 = next(self._current_iter)
            except StopIteration:
                self._current_iter = None
                continue
            if not q2:
                continue
            accepted_query = _accept_stream_query(
                q2,
                self._seen_local,
                shared_seen_keys=shared_seen_keys,
            )
            if not accepted_query:
                continue
            out.append(accepted_query)
            self._emitted += 1
        return out

def _prepare_base_queries_sync(
    initial_variants: List[Dict[str, Optional[str]]],
    track_name: Optional[str],
    search_variants_limit: int,
) -> Tuple[
    List[Dict[str, Optional[str]]],
    List[str],
    Set[str],
    int,
]:
    variants = _ensure_tag_tokenization(initial_variants, original_track=track_name)
    variants = _expand_variants_with_roman_l_i(variants)
    _svl = _safe_int(search_variants_limit, search_variants_limit if isinstance(search_variants_limit, int) else 100)
    if _svl <= 0:
        _svl = 1
    variants = variants[:_svl]

    seen_q: Set[str] = set()
    base_queries: List[str] = []
    base_queries_cap = max(1, int(_svl) - max(10, int(_svl) // 3))

    for v in variants:
        if len(base_queries) >= base_queries_cap:
            break
        q = _cached_query_from_variant(v)
        if not q:
            continue
        qn = _query_dedupe_key(q)
        if not qn or qn in seen_q:
            continue
        seen_q.add(qn)
        base_queries.append(q)

    return variants, base_queries, seen_q, _svl

def _prepare_normalized_variants_sync(
    variants: Iterable[Dict[str, Optional[str]]],
) -> _LazyExpandedVariants:
    return _LazyExpandedVariants(variants, _iter_normalized_variant_expansions)

def _prepare_symbolized_variants_sync(
    variants_normalized: Iterable[Dict[str, Optional[str]]],
) -> _LazyExpandedVariants:
    return _LazyExpandedVariants(variants_normalized, _iter_symbolized_variant_expansions)

def _prepare_anyascii_match_variants_sync(
    variants_symbolized: Iterable[Dict[str, Optional[str]]],
) -> _LazyExpandedVariants:
    return _LazyExpandedVariants(variants_symbolized, _iter_anyascii_variant_expansions)

def _fallback_phase_limit(phase_name: str, remaining_slots: int) -> int:
    if remaining_slots <= 0:
        return 0
    if phase_name == "spaced":
        return max(1, min(remaining_slots, 128))
    return remaining_slots

def _prepare_early_fallback_streams_sync(
    *,
    variants_normalized: Iterable[Dict[str, Optional[str]]],
    base_queries: List[str],
    remaining_slots: int,
    seen_keys: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    if remaining_slots <= 0:
        return {}
    return {
        "normalized": _VariantQueryStream(
            variants_normalized,
            remaining_slots,
            seen_keys=seen_keys,
        ),
        "symbols": _SymbolQueryStream(
            list(base_queries or []),
            remaining_slots,
            seen_keys=seen_keys,
        ),
    }

def _prepare_fallback_streams_sync(
    *,
    variants: Iterable[Dict[str, Optional[str]]],
    variants_normalized: Iterable[Dict[str, Optional[str]]],
    variants_symbolized: Iterable[Dict[str, Optional[str]]],
    variants_for_match: Iterable[Dict[str, Optional[str]]],
    found_initial_tag_or_feat_placeholders: bool,
    remaining_slots: int,
    seen_keys: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    if remaining_slots <= 0:
        return {}
    return {
        "anyascii": _AnyasciiQueryStream(
            variants_symbolized,
            remaining_slots,
            seen_keys=seen_keys,
        ),
        "concat": _ModeQueryStream(
            [variants or [], variants_normalized or [], variants_symbolized or [], variants_for_match or []],
            ["norm", "concat"],
            ["concat"],
            found_initial_tag_or_feat_placeholders,
            _fallback_phase_limit("concat", remaining_slots),
            seen_keys=seen_keys,
        ),
        "spaced": _ModeQueryStream(
            [variants or [], variants_normalized or [], variants_symbolized or [], variants_for_match or []],
            ["norm", "concat", "spaced"],
            ["spaced"],
            found_initial_tag_or_feat_placeholders,
            _fallback_phase_limit("spaced", remaining_slots),
            seen_keys=seen_keys,
        ),
    }

def _score_pool_best_sync(
    pool: List[Dict[str, Any]],
    variants_for_match: Iterable[Dict[str, Optional[str]]],
    *,
    featured_pref_list: Optional[List[str]],
    top1_counts: Optional[Dict[int, int]],
    song_priority_map: Optional[Dict[int, Tuple[int, int, int]]],
    default_priority: Tuple[int, int, int],
    track_name: Optional[str],
    spotify_album_name: Optional[str],
    spotify_album_total_tracks: Optional[int],
    slug_target_tokens: Optional[Set[str]],
) -> Optional[Dict[str, Any]]:
    best_song: Optional[Dict[str, Any]] = None
    best_score = -10**9
    best_priority = default_priority
    for s in pool:
        sc, _ = _score_song_detailed(
            s,
            variants_for_match,
            featured_pref=featured_pref_list,
            top1_counts=top1_counts,
            spotify_track_name=track_name,
            spotify_album_name=spotify_album_name,
            spotify_album_total_tracks=spotify_album_total_tracks,
            slug_target_tokens=slug_target_tokens,
        )
        sid = _safe_int_or_none(s.get("id")) if isinstance(s, dict) else None
        priority = song_priority_map.get(sid, default_priority) if (sid is not None and isinstance(song_priority_map, dict)) else default_priority
        if sc > best_score or (sc == best_score and priority < best_priority):
            best_score = sc
            best_song = s
            best_priority = priority
    return best_song

def _build_candidate_log_entries(
    scored_sorted: List[Tuple[Dict[str, Any], int, List[str]]],
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for idx, (s, sc, reasons) in enumerate(scored_sorted, start=1):
        if not isinstance(s, dict):
            continue
        primary_artist = s.get("primary_artist")
        artist = primary_artist.get("name") if isinstance(primary_artist, dict) else ""
        title = s.get("title_with_featured") or s.get("title") or ""
        name = f"{artist} – {title}".strip()
        album = (s.get("album") or {})
        album_name = album.get("name") if isinstance(album, dict) else ""
        date = s.get("release_date") or ""
        urlc = _song_url(s) or (s.get("url") if isinstance(s.get("url"), str) else None)
        entries.append({
            "idx": idx,
            "name": name,
            "album_name": album_name or "—",
            "date": date or "—",
            "score": sc,
            "url": urlc or "—",
            "reasons": list(reasons or []),
        })
    return entries

def _finalize_scored_candidates_sync(
    scored_details: List[Tuple[Dict[str, Any], int, List[str]]],
    *,
    song_priority_map: Optional[Dict[int, Tuple[int, int, int]]],
    default_priority: Tuple[int, int, int],
    detailed_log: bool,
    build_mistral_meta: bool,
    primary: str,
    featured: Optional[str],
    track_name: Optional[str],
    spotify_album_name: Optional[str],
    spotify_release_date: Optional[str],
    mistral_models_chain: Optional[List[str]] = None,
) -> Tuple[
    List[Tuple[Dict[str, Any], int, List[str]]],
    List[Dict[str, Any]],
    Optional[Dict[str, str]],
    Optional[List[Dict[str, str]]],
    Optional[str],
    Optional[str],
]:
    scored_sorted = sorted(
        scored_details,
        key=lambda x: (
            -x[1],
            song_priority_map.get(_safe_int_or_none(x[0].get("id")), default_priority)
            if isinstance(x[0], dict) and isinstance(song_priority_map, dict)
            else default_priority,
        ),
    )

    log_entries: List[Dict[str, Any]] = (
        _build_candidate_log_entries(scored_sorted) if detailed_log else []
    )

    spotify_track_meta: Optional[Dict[str, str]] = None
    genius_candidates_meta: Optional[List[Dict[str, str]]] = None
    mistral_payload_text: Optional[str] = None
    mistral_cache_key: Optional[str] = None
    if build_mistral_meta and scored_sorted:
        spotify_track_meta = _build_spotify_track_meta(
            primary=primary,
            featured=featured,
            track_name=track_name,
            spotify_album_name=spotify_album_name,
            spotify_release_date=spotify_release_date,
        )
        genius_candidates_meta = _build_genius_candidates_meta(scored_sorted)
        mistral_payload_text = build_mistral_payload_text(spotify_track_meta, genius_candidates_meta)
        if mistral_models_chain is not None:
            mistral_cache_key = build_mistral_choice_cache_key(
                mistral_models_chain,
                spotify_track_meta,
                genius_candidates_meta,
            )

    return (
        scored_sorted,
        log_entries,
        spotify_track_meta,
        genius_candidates_meta,
        mistral_payload_text,
        mistral_cache_key,
    )

def _score_candidates_and_build_meta_sync(
    candidates: List[Dict[str, Any]],
    variants_for_match: Iterable[Dict[str, Optional[str]]],
    *,
    featured_pref_list: Optional[List[str]],
    top1_counts: Optional[Dict[int, int]],
    song_priority_map: Optional[Dict[int, Tuple[int, int, int]]],
    default_priority: Tuple[int, int, int],
    track_name: Optional[str],
    spotify_album_name: Optional[str],
    spotify_album_total_tracks: Optional[int],
    slug_target_tokens: Optional[Set[str]],
    detailed_log: bool,
    build_mistral_meta: bool,
    primary: str,
    featured: Optional[str],
    spotify_release_date: Optional[str],
    mistral_models_chain: Optional[List[str]] = None,
) -> Tuple[
    List[Tuple[Dict[str, Any], int, List[str]]],
    List[Dict[str, Any]],
    Optional[Dict[str, str]],
    Optional[List[Dict[str, str]]],
    Optional[str],
    Optional[str],
]:
    scored_details: List[Tuple[Dict[str, Any], int, List[str]]] = []
    for s in candidates:
        sc, reasons = _score_song_detailed(
            s,
            variants_for_match,
            featured_pref=featured_pref_list,
            top1_counts=top1_counts,
            spotify_track_name=track_name,
            spotify_album_name=spotify_album_name,
            spotify_album_total_tracks=spotify_album_total_tracks,
            slug_target_tokens=slug_target_tokens,
        )
        scored_details.append((s, sc, reasons))
    return _finalize_scored_candidates_sync(
        scored_details,
        song_priority_map=song_priority_map,
        default_priority=default_priority,
        detailed_log=detailed_log,
        build_mistral_meta=build_mistral_meta,
        primary=primary,
        featured=featured,
        track_name=track_name,
        spotify_album_name=spotify_album_name,
        spotify_release_date=spotify_release_date,
        mistral_models_chain=mistral_models_chain,
    )

async def _score_candidates_and_build_meta_async(
    candidates: List[Dict[str, Any]],
    variants_for_match: Iterable[Dict[str, Optional[str]]],
    *,
    featured_pref_list: Optional[List[str]],
    top1_counts: Optional[Dict[int, int]],
    song_priority_map: Optional[Dict[int, Tuple[int, int, int]]],
    default_priority: Tuple[int, int, int],
    track_name: Optional[str],
    spotify_album_name: Optional[str],
    spotify_album_total_tracks: Optional[int],
    slug_target_tokens: Optional[Set[str]],
    detailed_log: bool,
    build_mistral_meta: bool,
    primary: str,
    featured: Optional[str],
    spotify_release_date: Optional[str],
    mistral_models_chain: Optional[List[str]] = None,
) -> Tuple[
    List[Tuple[Dict[str, Any], int, List[str]]],
    List[Dict[str, Any]],
    Optional[Dict[str, str]],
    Optional[List[Dict[str, str]]],
    Optional[str],
    Optional[str],
]:
    if len(candidates) <= 1:
        return await asyncio.to_thread(
            _score_candidates_and_build_meta_sync,
            candidates,
            variants_for_match,
            featured_pref_list=featured_pref_list,
            top1_counts=top1_counts,
            song_priority_map=song_priority_map,
            default_priority=default_priority,
            track_name=track_name,
            spotify_album_name=spotify_album_name,
            spotify_album_total_tracks=spotify_album_total_tracks,
            slug_target_tokens=slug_target_tokens,
            detailed_log=detailed_log,
            build_mistral_meta=build_mistral_meta,
            primary=primary,
            featured=featured,
            spotify_release_date=spotify_release_date,
            mistral_models_chain=mistral_models_chain,
        )

    async def _score_candidate(song: Dict[str, Any]) -> Tuple[Dict[str, Any], int, List[str]]:
        score, reasons = await asyncio.to_thread(
            _score_song_detailed,
            song,
            variants_for_match,
            featured_pref=featured_pref_list,
            top1_counts=top1_counts,
            spotify_track_name=track_name,
            spotify_album_name=spotify_album_name,
            spotify_album_total_tracks=spotify_album_total_tracks,
            slug_target_tokens=slug_target_tokens,
        )
        return song, score, reasons

    scored_details = await asyncio.gather(*(_score_candidate(song) for song in candidates))
    return await asyncio.to_thread(
        _finalize_scored_candidates_sync,
        scored_details,
        song_priority_map=song_priority_map,
        default_priority=default_priority,
        detailed_log=detailed_log,
        build_mistral_meta=build_mistral_meta,
        primary=primary,
        featured=featured,
        track_name=track_name,
        spotify_album_name=spotify_album_name,
        spotify_release_date=spotify_release_date,
        mistral_models_chain=mistral_models_chain,
    )

async def search_lyrics_url(track_name: str,
                            spotify_artists: Optional[List[Dict[str, Any]]] = None,
                            spotify_album_name: Optional[str] = None,
                            spotify_album_total_tracks: Optional[int] = None,
                            spotify_release_date: Optional[str] = None) -> Optional[str]:
    st = get_settings()
    token = st.genius_access_token
    if not token:
        _safe_log_error(ERROR_AUTH_GENIUS_INVALID_TOKEN)
        return None
    if not spotify_artists:
        return None
    mistral_enabled = _is_mistral_enabled()
    genius_client = AsyncGeniusClient(lambda: token, session=init_session())
    mistral_prefetch_task: Optional[asyncio.Task] = None
    detail_prefetch_cleanup_task: Optional[asyncio.Task] = None
    if mistral_enabled:
        mistral_api_key = getattr(st, "mistral_api_key", None)
        if mistral_api_key:
            mistral_prefetch_task = asyncio.create_task(prefetch_mistral_client(mistral_api_key))

    async def _finish_background_task(task: Optional[asyncio.Task], *, cancel: bool) -> None:
        if task is None:
            return
        if cancel and not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            return
        except Exception as e:
            _safe_log_warning(ERROR_UNEXPECTED, e)

    async def _finish_mistral_prefetch_task(*, cancel: bool) -> None:
        nonlocal mistral_prefetch_task
        task = mistral_prefetch_task
        mistral_prefetch_task = None
        await _finish_background_task(task, cancel=cancel)

    names_full = _extract_artist_names_from_spotify(spotify_artists)
    if not names_full:
        await _finish_mistral_prefetch_task(cancel=True)
        return None

    primary = names_full[0]
    rest_names = [n for n in names_full[1:] if n]
    featured = ", ".join(rest_names) if rest_names else None
    featured_pref_list = list(rest_names)
    found_initial_tag_or_feat_placeholders = False
    full_track = track_name or ""
    tag_from_track0, base0 = await asyncio.to_thread(_extract_trailing_tag, full_track)
    feats_before = detect_feat_placeholders_bracketed(full_track) or []
    feats_after = detect_feat_placeholders_bracketed(base0) or [] if base0 else []
    feats_removed = [f for f in feats_before if f not in feats_after]

    if tag_from_track0 or feats_removed:
        found_initial_tag_or_feat_placeholders = True
        base_log0 = normalize_query_value(base0) or base0
        tag_log0 = normalize_query_value(tag_from_track0) or tag_from_track0
        feat_log0 = ", ".join(
            (normalize_query_value(f) or f) for f in feats_removed
        ) if feats_removed else ""

        tag_pat0 = await asyncio.to_thread(_remix_tag_matched_pattern, tag_from_track0) if tag_from_track0 else None
        tag_pat_log0 = tag_pat0 or ""

        feat_pats: list[str] = []
        if feats_removed:
            for f in feats_removed:
                p = feat_placeholder_pattern(f)
                if p and p not in feat_pats:
                    feat_pats.append(p)
        feat_pat_log0 = ", ".join(feat_pats) if feat_pats else ""

        if not (_is_mistral_enabled() and not _is_genius_detailed_log()):
            if tag_from_track0 and feats_removed:
                _safe_log_info(
                    INFO_GENIUS_TAG_FOUND_AND_FEAT_REMOVED_FROM_TRACK,
                    base_log0,
                    tag_log0,
                    tag_pat_log0,
                    feat_log0,
                    feat_pat_log0,
                )
            elif tag_from_track0:
                _safe_log_info(
                    INFO_GENIUS_TAG_FOUND_FROM_TRACK,
                    base_log0,
                    tag_log0,
                    tag_pat_log0,
                )
            else:
                _safe_log_info(
                    INFO_GENIUS_FEAT_REMOVED_FROM_TRACK,
                    base_log0,
                    feat_log0,
                    feat_pat_log0,
                )
    artist_combos = _build_artist_combinations(names_full) or [(primary, None), (primary, featured)]

    variants: List[Dict[str, Optional[str]]] = []
    for pa2, fea2 in artist_combos:
        variants.append(
            {
                "primary_artist": pa2,
                "featured_artists": fea2,
                "track": track_name or "",
                "tag": None,
            }
        )

    (
        variants,
        base_queries,
        seen_q,
        _svl,
    ) = await asyncio.to_thread(
        _prepare_base_queries_sync,
        variants,
        track_name,
        SEARCH_VARIANTS_LIMIT,
    )
    remaining_slots = max(0, _svl - len(base_queries))
    concurrency = max(1, _safe_int(GENIUS_SEARCH_CONCURRENCY, 1))
    details_concurrency = max(1, min(concurrency, 128))
    limit_n = _safe_int(GENIUS_DETAILS_TOP_N, 10)
    if limit_n < 0:
        limit_n = 0
    sem = asyncio.BoundedSemaphore(concurrency)
    phase_order_map = {
        "base": 0,
        "normalized": 1,
        "symbols": 2,
        "anyascii": 3,
        "concat": 4,
        "spaced": 5,
    }
    variants_normalized: Iterable[Dict[str, Optional[str]]] = []
    variants_symbolized: Iterable[Dict[str, Optional[str]]] = []
    variants_for_match: Iterable[Dict[str, Optional[str]]] = list(variants)
    early_fallback_phase_streams: Dict[str, Any] = {}
    late_fallback_phase_streams: Dict[str, Any] = {}
    normalized_payload_task: Optional[asyncio.Task] = asyncio.create_task(asyncio.to_thread(
            _prepare_normalized_variants_sync,
            variants,
        ))
    normalized_payload_ready = False

    async def _build_symbolized_payload(normalized_task: asyncio.Task) -> _LazyExpandedVariants:
        prepared_variants_normalized = await asyncio.shield(normalized_task)
        return await asyncio.to_thread(_prepare_symbolized_variants_sync, prepared_variants_normalized)

    symbolized_payload_task: Optional[asyncio.Task] = (
        asyncio.create_task(_build_symbolized_payload(normalized_payload_task))
        if normalized_payload_task is not None
        else None
    )
    symbolized_payload_ready = False

    async def _build_match_payload(symbolized_task: asyncio.Task) -> _LazyExpandedVariants:
        prepared_variants_symbolized = await asyncio.shield(symbolized_task)
        return await asyncio.to_thread(_prepare_anyascii_match_variants_sync, prepared_variants_symbolized)

    match_payload_task: Optional[asyncio.Task] = (
        asyncio.create_task(_build_match_payload(symbolized_payload_task))
        if symbolized_payload_task is not None
        else None
    )
    match_payload_ready = False
    variants_for_match_snapshot: Optional[List[Dict[str, Optional[str]]]] = None

    def _materialize_match_variants_sync(
        variants_iterable: Iterable[Dict[str, Optional[str]]],
    ) -> List[Dict[str, Optional[str]]]:
        if isinstance(variants_iterable, _LazyExpandedVariants):
            return variants_iterable.materialize()
        return list(variants_iterable or [])

    async def _build_match_variants_snapshot(
        match_task: asyncio.Task,
    ) -> List[Dict[str, Optional[str]]]:
        prepared_variants_for_match = await asyncio.shield(match_task)
        return await asyncio.to_thread(_materialize_match_variants_sync, prepared_variants_for_match)

    match_variants_snapshot_task: Optional[asyncio.Task] = (
        asyncio.create_task(_build_match_variants_snapshot(match_payload_task))
        if match_payload_task is not None
        else None
    )

    async def _build_early_fallback_phase_streams(
        normalized_task: asyncio.Task,
    ) -> Dict[str, Any]:
        prepared_variants_normalized = await asyncio.shield(normalized_task)
        return _prepare_early_fallback_streams_sync(
            variants_normalized=prepared_variants_normalized,
            base_queries=base_queries,
            remaining_slots=remaining_slots,
            seen_keys=seen_q,
        )

    early_fallback_streams_task: Optional[asyncio.Task] = (
        asyncio.create_task(_build_early_fallback_phase_streams(normalized_payload_task))
        if remaining_slots > 0 and normalized_payload_task is not None
        else None
    )
    early_fallback_streams_ready = False

    async def _build_late_fallback_phase_streams(
        normalized_task: asyncio.Task,
        symbolized_task: asyncio.Task,
        match_task: asyncio.Task,
    ) -> Dict[str, Any]:
        prepared_variants_normalized = await asyncio.shield(normalized_task)
        prepared_variants_symbolized = await asyncio.shield(symbolized_task)
        prepared_variants_for_match = await asyncio.shield(match_task)
        return _prepare_fallback_streams_sync(
            variants=variants,
            variants_normalized=prepared_variants_normalized,
            variants_symbolized=prepared_variants_symbolized,
            variants_for_match=prepared_variants_for_match,
            found_initial_tag_or_feat_placeholders=found_initial_tag_or_feat_placeholders,
            remaining_slots=remaining_slots,
            seen_keys=seen_q,
        )

    late_fallback_streams_task: Optional[asyncio.Task] = (
        asyncio.create_task(
            _build_late_fallback_phase_streams(
                normalized_payload_task,
                symbolized_payload_task,
                match_payload_task,
            )
        )
        if (
            remaining_slots > 0
            and normalized_payload_task is not None
            and symbolized_payload_task is not None
            and match_payload_task is not None
        )
        else None
    )
    late_fallback_streams_ready = False

    async def _ensure_normalized_payload() -> None:
        nonlocal variants_normalized, normalized_payload_task, normalized_payload_ready
        if normalized_payload_ready:
            return
        if normalized_payload_task is not None:
            variants_normalized = await normalized_payload_task
            normalized_payload_task = None
        normalized_payload_ready = True

    async def _ensure_symbolized_payload() -> None:
        nonlocal variants_symbolized, symbolized_payload_task, symbolized_payload_ready
        if symbolized_payload_ready:
            return
        await _ensure_normalized_payload()
        if symbolized_payload_task is not None:
            variants_symbolized = await symbolized_payload_task
            symbolized_payload_task = None
        symbolized_payload_ready = True

    async def _ensure_match_payload() -> None:
        nonlocal variants_for_match, match_payload_task, match_payload_ready
        if match_payload_ready:
            return
        await _ensure_symbolized_payload()
        if match_payload_task is not None:
            variants_for_match = await match_payload_task
            match_payload_task = None
        match_payload_ready = True

    def _promote_match_variants_snapshot_if_ready() -> None:
        nonlocal variants_for_match, variants_for_match_snapshot, match_variants_snapshot_task
        if variants_for_match_snapshot is not None:
            if variants_for_match is not variants_for_match_snapshot:
                variants_for_match = variants_for_match_snapshot
            return
        if match_variants_snapshot_task is None or not match_variants_snapshot_task.done():
            return
        try:
            variants_for_match_snapshot = match_variants_snapshot_task.result()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            _safe_log_warning(ERROR_UNEXPECTED, e)
            match_variants_snapshot_task = None
            return
        variants_for_match = variants_for_match_snapshot
        match_variants_snapshot_task = None

    async def _ensure_match_variants_snapshot() -> List[Dict[str, Optional[str]]]:
        nonlocal variants_for_match, variants_for_match_snapshot, match_variants_snapshot_task
        _promote_match_variants_snapshot_if_ready()
        if variants_for_match_snapshot is not None:
            return variants_for_match_snapshot
        await _ensure_match_payload()
        if match_variants_snapshot_task is not None:
            try:
                variants_for_match_snapshot = await match_variants_snapshot_task
            except Exception:
                match_variants_snapshot_task = None
            else:
                variants_for_match = variants_for_match_snapshot
                match_variants_snapshot_task = None
                return variants_for_match_snapshot
        variants_for_match_snapshot = await asyncio.to_thread(_materialize_match_variants_sync, variants_for_match)
        variants_for_match = variants_for_match_snapshot
        return variants_for_match_snapshot

    async def _ensure_early_fallback_streams() -> None:
        nonlocal early_fallback_phase_streams, early_fallback_streams_task, early_fallback_streams_ready
        if early_fallback_streams_ready:
            return
        await _ensure_normalized_payload()
        if early_fallback_streams_task is not None:
            early_fallback_phase_streams = await early_fallback_streams_task
            early_fallback_streams_task = None
        else:
            early_fallback_phase_streams = _prepare_early_fallback_streams_sync(
                variants_normalized=variants_normalized,
                base_queries=base_queries,
                remaining_slots=remaining_slots,
                seen_keys=seen_q,
            )
        early_fallback_streams_ready = True

    async def _ensure_late_fallback_streams() -> None:
        nonlocal late_fallback_phase_streams, late_fallback_streams_task, late_fallback_streams_ready
        if late_fallback_streams_ready:
            return
        await _ensure_match_payload()
        if late_fallback_streams_task is not None:
            late_fallback_phase_streams = await late_fallback_streams_task
            late_fallback_streams_task = None
        else:
            late_fallback_phase_streams = _prepare_fallback_streams_sync(
                variants=variants,
                variants_normalized=variants_normalized,
                variants_symbolized=variants_symbolized,
                variants_for_match=variants_for_match,
                found_initial_tag_or_feat_placeholders=found_initial_tag_or_feat_placeholders,
                remaining_slots=remaining_slots,
                seen_keys=seen_q,
            )
        late_fallback_streams_ready = True

    async def _cancel_match_payload_tasks() -> None:
        nonlocal match_payload_task, symbolized_payload_task, normalized_payload_task, match_variants_snapshot_task
        await _finish_background_task(match_variants_snapshot_task, cancel=True)
        match_variants_snapshot_task = None
        await _finish_background_task(match_payload_task, cancel=True)
        match_payload_task = None
        await _finish_background_task(symbolized_payload_task, cancel=True)
        symbolized_payload_task = None
        await _finish_background_task(normalized_payload_task, cancel=True)
        normalized_payload_task = None

    fallback_search_logs_deferred = remaining_slots > 0
    fallback_search_log_buffer: List[Tuple[str, str]] = []

    def _log_search_query(variant_name: str, q: str) -> None:
        if not _is_genius_detailed_log():
            return
        if variant_name == "base" or not fallback_search_logs_deferred:
            _safe_log_info(INFO_GENIUS_SEARCH_QUERY, variant_name, q)
            return
        fallback_search_log_buffer.append((variant_name, q))

    def _flush_buffered_fallback_search_logs() -> None:
        nonlocal fallback_search_logs_deferred
        if not fallback_search_logs_deferred:
            return
        fallback_search_logs_deferred = False
        if not _is_genius_detailed_log():
            fallback_search_log_buffer.clear()
            return
        for buffered_variant_name, buffered_query in fallback_search_log_buffer:
            _safe_log_info(INFO_GENIUS_SEARCH_QUERY, buffered_variant_name, buffered_query)
        fallback_search_log_buffer.clear()

    def _discard_buffered_fallback_search_logs() -> None:
        nonlocal fallback_search_logs_deferred
        fallback_search_logs_deferred = False
        fallback_search_log_buffer.clear()

    async def run_search(variant_name: str, query_index: int, q: str) -> Tuple[str, int, List[Dict[str, Any]]]:
        async with sem:
            _log_search_query(variant_name, q)
            return variant_name, query_index, await genius_client.search(q)

    sem2 = asyncio.BoundedSemaphore(details_concurrency)

    async def guarded_fetch(sid: int) -> Tuple[int, Optional[Dict[str, Any]]]:
        async with sem2:
            song = await genius_client.fetch_song(sid)
            return sid, song
    detail_prefetch_tasks: Dict[int, asyncio.Task] = {}

    def _prime_detail_fetch(sid: Optional[int], *, prioritize: bool = False) -> None:
        if sid is None or limit_n <= 0:
            return
        if sid in detail_prefetch_tasks:
            return
        if len(detail_prefetch_tasks) >= limit_n:
            if not prioritize or len(detail_prefetch_tasks) >= limit_n * 4:
                return
        detail_prefetch_tasks[sid] = asyncio.create_task(guarded_fetch(sid))

    async def _cancel_detail_prefetches(*, keep_ids: Optional[Set[int]] = None) -> None:
        keep = set(keep_ids or set())
        tasks_to_wait: List[asyncio.Task] = []
        for sid, task in list(detail_prefetch_tasks.items()):
            if sid in keep:
                continue
            if not task.done():
                task.cancel()
            tasks_to_wait.append(task)
            detail_prefetch_tasks.pop(sid, None)
        if tasks_to_wait:
            await asyncio.gather(*tasks_to_wait, return_exceptions=True)

    async def _resolve_detail_prefetches(selected_song_ids: List[int]) -> List[Any]:
        tasks: List[asyncio.Task] = []
        for sid in selected_song_ids:
            task = detail_prefetch_tasks.get(sid)
            if task is None:
                task = asyncio.create_task(guarded_fetch(sid))
                detail_prefetch_tasks[sid] = task
            tasks.append(task)
        if not tasks:
            return []
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _schedule_detail_prefetch_cleanup() -> None:
        nonlocal detail_prefetch_cleanup_task
        if detail_prefetch_cleanup_task is not None or not detail_prefetch_tasks:
            return
        detail_prefetch_cleanup_task = asyncio.create_task(_cancel_detail_prefetches())

    async def _finish_detail_prefetch_cleanup_task() -> None:
        nonlocal detail_prefetch_cleanup_task
        task = detail_prefetch_cleanup_task
        detail_prefetch_cleanup_task = None
        await _finish_background_task(task, cancel=False)

    base_phase_songs: Dict[int, Dict[str, Any]] = {}
    base_phase_top1_counts: Dict[int, int] = {}
    base_phase_first_priority: Dict[int, Tuple[int, int, int]] = {}
    fallback_phase_songs: Dict[int, Dict[str, Any]] = {}
    fallback_phase_top1_counts: Dict[int, int] = {}
    fallback_phase_first_priority: Dict[int, Tuple[int, int, int]] = {}
    base_songs: Dict[int, Dict[str, Any]] = {}
    top1_counts: Dict[int, int] = {}
    song_first_priority: Dict[int, Tuple[int, int, int]] = {}
    matched_song_ids: Set[int] = set()
    rejected_song_ids: Set[int] = set()

    def _phase_order_value(variant_name: str) -> int:
        return phase_order_map.get(variant_name, len(phase_order_map))

    def _phase_result_maps(
        variant_name: str,
    ) -> Tuple[
        Dict[int, Dict[str, Any]],
        Dict[int, int],
        Dict[int, Tuple[int, int, int]],
    ]:
        if variant_name == "base":
            return base_phase_songs, base_phase_top1_counts, base_phase_first_priority
        return fallback_phase_songs, fallback_phase_top1_counts, fallback_phase_first_priority

    def _consume_search_result(variant_name: str, query_index: int, res: List[Dict[str, Any]]) -> bool:
        if not isinstance(res, list):
            return False
        _promote_match_variants_snapshot_if_ready()
        match_variants_iterable = variants_for_match_snapshot if variants_for_match_snapshot is not None else variants_for_match
        first_sid: Optional[int] = None
        phase_matched = False
        phase_order = _phase_order_value(variant_name)
        phase_songs, phase_top1_counts, phase_first_priority = _phase_result_maps(variant_name)
        for hit_index, song in enumerate(res):
            if not isinstance(song, dict):
                continue
            sid = _safe_int_or_none(song.get("id"))
            if sid is None:
                continue
            if sid in matched_song_ids:
                matched = True
            elif sid in rejected_song_ids:
                matched = False
            else:
                matched = any(
                    _song_matches_variant_fuzzy(song, v, GENIUS_FUZZY_TOKENS_PERCENT, GENIUS_FUZZY_CHARS_PERCENT)
                    for v in match_variants_iterable
                )
                if matched:
                    matched_song_ids.add(sid)
                else:
                    rejected_song_ids.add(sid)
            if not matched:
                continue
            phase_matched = True
            if first_sid is None:
                first_sid = sid
            priority = (phase_order, query_index, hit_index)
            current_priority = phase_first_priority.get(sid)
            if current_priority is None or priority < current_priority:
                phase_first_priority[sid] = priority
                phase_songs[sid] = song
            elif sid not in phase_songs:
                phase_songs[sid] = song
            _prime_detail_fetch(sid, prioritize=(phase_order == 0))

        if first_sid is not None:
            phase_top1_counts[first_sid] = phase_top1_counts.get(first_sid, 0) + 1
        return phase_matched

    async def _run_search_phase(
        variant_name: str,
        queries: List[str],
        *,
        on_phase_matched: Optional[Callable[[], None]] = None,
    ) -> bool:
        if not queries:
            return False
        search_tasks = [
            asyncio.create_task(run_search(variant_name, query_index, q))
            for query_index, q in enumerate(queries)
            if q
        ]
        if not search_tasks:
            return False
        phase_matched = False
        try:
            for done_task in asyncio.as_completed(search_tasks):
                try:
                    res = await done_task
                except asyncio.CancelledError:
                    for t in search_tasks:
                        try:
                            if not t.done():
                                t.cancel()
                        except RuntimeError:
                            pass
                    await asyncio.gather(*search_tasks, return_exceptions=True)
                    raise
                except Exception as e:
                    _safe_log_warning(ERROR_UNEXPECTED, e)
                    continue

                if not isinstance(res, tuple) or len(res) != 3:
                    continue
                await _ensure_match_payload()
                variant_name_done, query_index_done, result_items = res
                if _consume_search_result(variant_name_done, query_index_done, result_items):
                    if not phase_matched:
                        phase_matched = True
                        if on_phase_matched is not None:
                            try:
                                on_phase_matched()
                            except asyncio.CancelledError:
                                raise
                            except Exception as e:
                                _safe_log_warning(ERROR_UNEXPECTED, e)
        except asyncio.CancelledError:
            for t in search_tasks:
                try:
                    t.cancel()
                except RuntimeError:
                    pass
            await asyncio.gather(*search_tasks, return_exceptions=True)
            raise
        return phase_matched

    async def _run_fallback_full_compact() -> None:
        if remaining_slots <= 0:
            return
        early_phase_order = ["normalized", "symbols"]
        late_phase_order = ["anyascii", "concat", "spaced"]
        phase_order = early_phase_order + late_phase_order
        await _ensure_early_fallback_streams()
        phase_to_streams: Dict[str, Any] = {
            phase_name: early_fallback_phase_streams.get(phase_name)
            for phase_name in phase_order
        }
        active_phase_names = [
            phase_name
            for phase_name in early_phase_order
            if phase_to_streams.get(phase_name) is not None
        ]
        phase_positions: Dict[str, int] = {phase_name: 0 for phase_name in phase_order}
        ready_queries: deque[Tuple[str, int, str]] = deque()
        fallback_seen_keys = set(seen_q)
        fallback_queries_count = 0
        late_phase_names_added = False

        search_tasks: Set[asyncio.Task] = set()

        def _can_enqueue_more_queries() -> bool:
            return fallback_queries_count < remaining_slots

        def _activate_late_phase_queries() -> None:
            nonlocal late_phase_names_added
            if late_phase_names_added:
                return
            for phase_name in late_phase_order:
                phase_stream = late_fallback_phase_streams.get(phase_name)
                phase_to_streams[phase_name] = phase_stream
                if phase_stream is not None:
                    active_phase_names.append(phase_name)
            late_phase_names_added = True

        async def _maybe_activate_late_phase_queries() -> bool:
            if late_phase_names_added or not _can_enqueue_more_queries() or active_phase_names:
                return False
            await _ensure_late_fallback_streams()
            _activate_late_phase_queries()
            return True

        def _enqueue_round() -> bool:
            nonlocal active_phase_names, fallback_queries_count
            if not _can_enqueue_more_queries():
                active_phase_names = []
                return False
            if not active_phase_names:
                return False
            next_active_phase_names: List[str] = []
            advanced = False
            for phase_name in active_phase_names:
                if fallback_queries_count >= remaining_slots:
                    break
                stream = phase_to_streams.get(phase_name)
                if stream is None:
                    continue
                batch = stream.take_batch(1, shared_seen_keys=fallback_seen_keys)
                if not batch:
                    continue
                advanced = True
                next_active_phase_names.append(phase_name)
                for q in batch:
                    if fallback_queries_count >= remaining_slots:
                        break
                    query_index = phase_positions.get(phase_name, 0)
                    ready_queries.append((phase_name, query_index, q))
                    phase_positions[phase_name] = query_index + 1
                    fallback_queries_count += 1
            active_phase_names = next_active_phase_names
            return advanced

        async def _cancel_outstanding() -> None:
            tasks_to_cancel = list(search_tasks)
            for task in tasks_to_cancel:
                try:
                    if not task.done():
                        task.cancel()
                except RuntimeError:
                    pass
            if tasks_to_cancel:
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            search_tasks.clear()

        try:
            while True:
                while len(search_tasks) < concurrency:
                    if not ready_queries:
                        enqueued_round = _enqueue_round()
                        if not enqueued_round and await _maybe_activate_late_phase_queries():
                            enqueued_round = _enqueue_round()
                        if not enqueued_round:
                            break
                    if not ready_queries:
                        break
                    phase_name, query_index, q = ready_queries.popleft()
                    search_tasks.add(asyncio.create_task(run_search(phase_name, query_index, q)))
                    if not ready_queries and not active_phase_names:
                        await _maybe_activate_late_phase_queries()

                if not search_tasks:
                    if ready_queries:
                        continue
                    if await _maybe_activate_late_phase_queries():
                        continue
                    break

                done, _ = await asyncio.wait(set(search_tasks), return_when=asyncio.FIRST_COMPLETED)

                for done_task in done:
                    search_tasks.discard(done_task)
                    try:
                        res = await done_task
                    except asyncio.CancelledError:
                        await _cancel_outstanding()
                        raise
                    except Exception as e:
                        _safe_log_warning(ERROR_UNEXPECTED, e)
                        continue
                    if not isinstance(res, tuple) or len(res) != 3:
                        continue
                    await _ensure_match_payload()
                    variant_name_done, query_index_done, result_items = res
                    _ = _consume_search_result(variant_name_done, query_index_done, result_items)
                if not ready_queries and not active_phase_names:
                    await _maybe_activate_late_phase_queries()
        except asyncio.CancelledError:
            await _cancel_outstanding()
            raise

    fallback_search_task: Optional[asyncio.Task] = None
    keep_detail_prefetch_tasks = False
    keep_match_payload_tasks = False

    def _cancel_task(task: Optional[asyncio.Task]) -> None:
        if task is None:
            return
        try:
            if not task.done():
                task.cancel()
        except RuntimeError:
            pass

    def _cancel_fallback_background_work() -> None:
        _cancel_task(fallback_search_task)
        _cancel_task(early_fallback_streams_task)
        _cancel_task(late_fallback_streams_task)

    try:
        if remaining_slots > 0:
            fallback_search_task = asyncio.create_task(_run_fallback_full_compact())
        base_phase_matched = await _run_search_phase(
            "base",
            base_queries,
            on_phase_matched=_cancel_fallback_background_work if fallback_search_task is not None else None,
        )
        if fallback_search_task is not None:
            if base_phase_matched:
                await _finish_background_task(fallback_search_task, cancel=True)
                fallback_search_task = None
                _discard_buffered_fallback_search_logs()
                await _cancel_detail_prefetches(keep_ids=set(base_phase_songs))
            else:
                _flush_buffered_fallback_search_logs()
                await fallback_search_task
                fallback_search_task = None

        if base_phase_songs:
            base_songs = base_phase_songs
            top1_counts = base_phase_top1_counts
            song_first_priority = base_phase_first_priority
        else:
            base_songs = fallback_phase_songs
            top1_counts = fallback_phase_top1_counts
            song_first_priority = fallback_phase_first_priority

        keep_detail_prefetch_tasks = bool(base_songs and detail_prefetch_tasks)
        keep_match_payload_tasks = bool(base_songs)
    finally:
        await _finish_background_task(fallback_search_task, cancel=True)
        await _finish_background_task(early_fallback_streams_task, cancel=True)
        await _finish_background_task(late_fallback_streams_task, cancel=True)
        if not keep_match_payload_tasks:
            await _cancel_match_payload_tasks()
        if not keep_detail_prefetch_tasks:
            await _cancel_detail_prefetches()

    try:
        if not base_songs:
            await _finish_mistral_prefetch_task(cancel=True)
            _log_no_results()
            return None

        await _ensure_match_payload()
        variants_for_match_ready = await _ensure_match_variants_snapshot()
        slug_target_tokens = _build_slug_target_tokens(variants_for_match_ready, track_name)

        base_ids = list(base_songs.keys())
        default_song_priority = (len(phase_order_map), _svl + len(base_queries), 10**9)
        sorted_ids = sorted(
            base_ids,
            key=lambda s: (
                -top1_counts.get(s, 0),
                song_first_priority.get(s, default_song_priority),
            ),
        )
        selected_ids = sorted_ids[:limit_n]
        details_results: List[Any] = []
        try:
            details_results = await _resolve_detail_prefetches(selected_ids) if selected_ids else []
        finally:
            _schedule_detail_prefetch_cleanup()

        details_by_id: Dict[int, Dict[str, Any]] = {}
        for r in details_results:
            if isinstance(r, BaseException) or not isinstance(r, tuple) or len(r) != 2:
                continue
            sid, song = r
            if song:
                details_by_id[sid] = song
            else:
                _safe_log_error(ERROR_GENIUS_DETAILS_FAILED, sid)

        if _is_genius_detailed_log():
            _safe_log_info(INFO_GENIUS_DETAILS_FETCHED, len(details_by_id), len(base_ids))
        candidates: List[Dict[str, Any]] = []
        for sid in selected_ids:
            if sid in details_by_id:
                candidates.append(details_by_id[sid])
            elif sid in base_songs:
                candidates.append(base_songs[sid])

        if not candidates:
            pool = list(details_by_id.values()) if details_by_id else list(base_songs.values())
            if pool and not mistral_enabled:
                best_song_fb2 = await asyncio.to_thread(
                    _score_pool_best_sync,
                    pool,
                    variants_for_match_ready,
                    featured_pref_list=featured_pref_list,
                    top1_counts=top1_counts,
                    song_priority_map=song_first_priority,
                    default_priority=default_song_priority,
                    track_name=track_name,
                    spotify_album_name=spotify_album_name,
                    spotify_album_total_tracks=spotify_album_total_tracks,
                    slug_target_tokens=slug_target_tokens,
                )
                if best_song_fb2:
                    url_fb2 = _resolve_song_url(best_song_fb2, details_by_id)
                    if url_fb2:
                        await _finish_mistral_prefetch_task(cancel=True)
                        _safe_log_info(INFO_GENIUS_SCORING_CHOSEN, url_fb2)
                        return url_fb2
            await _finish_mistral_prefetch_task(cancel=True)
            _log_no_results()
            return None

        detailed_log = _is_genius_detailed_log()
        mistral_models_chain: Optional[List[str]] = None
        if mistral_enabled:
            try:
                mistral_models_chain = get_mistral_models_chain()
            except (RuntimeError, ValueError, TypeError):
                mistral_models_chain = None

        score_task = asyncio.create_task(_score_candidates_and_build_meta_async(
            candidates,
            variants_for_match_ready,
            featured_pref_list=featured_pref_list,
            top1_counts=top1_counts,
            song_priority_map=song_first_priority,
            default_priority=default_song_priority,
            track_name=track_name,
            spotify_album_name=spotify_album_name,
            spotify_album_total_tracks=spotify_album_total_tracks,
            slug_target_tokens=slug_target_tokens,
            detailed_log=detailed_log,
            build_mistral_meta=mistral_enabled,
            primary=primary,
            featured=featured,
            spotify_release_date=spotify_release_date,
            mistral_models_chain=mistral_models_chain,
        ))
        try:
            (
                scored_sorted,
                log_entries,
                spotify_track_meta,
                genius_candidates_meta,
                mistral_payload_text,
                mistral_cache_key,
            ) = await score_task
        except asyncio.CancelledError:
            await _finish_mistral_prefetch_task(cancel=True)
            raise
        if detailed_log:
            for entry in log_entries:
                _safe_log_info(
                    INFO_GENIUS_CANDIDATE_SUMMARY,
                    entry["idx"],
                    entry["name"],
                    entry["album_name"],
                    entry["date"],
                    entry["score"],
                    entry["url"],
                )
                for r in entry["reasons"]:
                    _safe_log_info(INFO_GENIUS_CANDIDATE_REASON, r)

        best_song = scored_sorted[0][0] if scored_sorted else None
        if not best_song:
            await _finish_mistral_prefetch_task(cancel=True)
            _log_no_results()
            return None

        final_song = best_song
        if mistral_enabled and not (scored_sorted and spotify_track_meta is not None and genius_candidates_meta is not None):
            _safe_log_info(INFO_MISTRAL_FALLBACK)
            mistral_enabled = False
        if mistral_enabled:
            idx_choice, explicit_none, choice_source = await choose_genius_candidate_with_mistral(
                spotify_track_meta,
                genius_candidates_meta,
                cache_key=mistral_cache_key,
                payload_text=mistral_payload_text,
            )

            if explicit_none:
                await _finish_mistral_prefetch_task(cancel=True)
                if choice_source == "cache":
                    _safe_log_info(INFO_MISTRAL_FINAL_SELECTION_NONE_CACHED)
                elif choice_source == "inflight":
                    _safe_log_info(INFO_MISTRAL_FINAL_SELECTION_NONE_INFLIGHT)
                else:
                    _safe_log_info(INFO_MISTRAL_FINAL_SELECTION_NONE)
                if not (_is_mistral_enabled() and not _is_genius_detailed_log()):
                    _safe_log_info(INFO_GENIUS_NO_RESULTS)
                return None

            if isinstance(idx_choice, int) and 1 <= idx_choice <= len(scored_sorted):
                final_song = scored_sorted[idx_choice - 1][0]
            elif idx_choice is not None:
                _safe_log_error(
                    ERROR_MISTRAL_FINAL_SELECTION,
                    f"idx out of range: {idx_choice} (candidates={len(scored_sorted)})",
                )
                _safe_log_info(INFO_MISTRAL_FALLBACK)
            else:
                _safe_log_info(INFO_MISTRAL_FALLBACK)

        url = _resolve_song_url(final_song, details_by_id)
        await _finish_mistral_prefetch_task(cancel=True)
        if url:
            _safe_log_info(INFO_GENIUS_SCORING_CHOSEN, url)
            return url
        return None
    except asyncio.CancelledError:
        await _cancel_match_payload_tasks()
        await _finish_mistral_prefetch_task(cancel=True)
        raise
    except Exception as e:
        _safe_log_error(ERROR_UNEXPECTED, e)
        await _cancel_match_payload_tasks()
        await _finish_mistral_prefetch_task(cancel=True)
        raise
    finally:
        _schedule_detail_prefetch_cleanup()
        await _finish_detail_prefetch_cleanup_task()
