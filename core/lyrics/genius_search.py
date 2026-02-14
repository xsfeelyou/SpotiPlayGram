from __future__ import annotations
import re
import asyncio
import math
import json
from urllib.parse import urlsplit
import aiohttp
from typing import Optional, Dict, Any, List, Tuple, Set

from logger import _safe_log_info, _safe_log_error
from constants import (
    ERROR_AUTH_GENIUS_INVALID_TOKEN,
    ERROR_GENIUS_ACCESS_TOKEN_INVALID,
    ERROR_GENIUS_DETAILS_FAILED,
    ERROR_GENIUS_HTTP,
    ERROR_MISTRAL_FINAL_SELECTION,
    GENIUS_ALBUM_MIN_TRACKS,
    GENIUS_DETAILS_TOP_N,
    GENIUS_FUZZY_CHARS_PERCENT,
    GENIUS_FUZZY_TOKENS_PERCENT,
    GENIUS_HTTP_MAX_RETRIES,
    GENIUS_HTTP_RETRY_DELAY_BASE,
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
    GENIUS_SEARCH_PER_PAGE,
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
    INFO_MISTRAL_FINAL_SELECTION_NONE,
    INFO_TRANSLATION_REPLACED,
    SEARCH_VARIANTS_LIMIT,
)
from utils.http_session import init_session
from config import Settings
from .mistral_ai import choose_genius_candidate_with_mistral
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
    generate_roman_l_i_variants,
    generate_symbol_letter_equiv_variants,
    get_tag_patterns,
    get_tag_patterns_version,
)

API_BASE = "https://api.genius.com"

_settings: Optional[Settings] = None

def set_settings(settings: Settings) -> None:
    global _settings
    _settings = settings

def get_settings() -> Settings:
    if _settings is None:
        raise RuntimeError("Settings are not initialized")
    return _settings

def _is_mistral_enabled() -> bool:
    st = get_settings()
    return bool(st.enable_mistral)

def _is_genius_detailed_log() -> bool:
    st = get_settings()
    return bool(st.genius_detailed_log)

async def _read_json_payload(resp) -> Optional[Dict[str, Any]]:
    txt = await resp.text(errors="ignore")
    t = txt.lstrip()
    if not t:
        _safe_log_error(ERROR_GENIUS_HTTP, "JSON", "")
        return None
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        _safe_log_error(
            ERROR_GENIUS_HTTP,
            "JSON",
            t[:200],
        )
        return None
    if not isinstance(data, dict):
        _safe_log_error(ERROR_GENIUS_HTTP, "JSON", t[:200])
        return None
    return data

def _collapse_ws(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())

def _log_no_results() -> None:
    if _is_mistral_enabled() and not _is_genius_detailed_log():
        _safe_log_info(INFO_MISTRAL_FINAL_SELECTION_NONE)
    else:
        _safe_log_info(INFO_GENIUS_NO_RESULTS)

__all__ = [
    "search_lyrics_url",
]

_RE_FLOAT = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

def _norm(s: Optional[str]) -> str:
    t = _safe_stripped(s)
    if not t:
        return ""
    t = normalize_apostrophes(t)
    t = t.replace("'", "")
    return _collapse_ws(t).lower()

def _clear_title(s: str) -> str:
    t = _safe_stripped(s)
    if not t:
        return ""
    s1 = re.sub(r"\s*\([^)]*\)\s*", " ", t)
    s1 = re.sub(r"\s+-\s*[^-]+$", " ", s1)
    s1 = _collapse_ws(s1)
    return s1

def _normalize_base_text(s: Optional[str], *, strip_separators: bool = False) -> str:
    t = _safe_stripped(s)
    if not t:
        return ""
    if strip_separators:
        t = strip_spaced_separators(t)
    t = normalize_letters(t)
    return _norm(t)

def _normalize_ws_tokens(s: Optional[str], *, strip_separators: bool = False) -> List[str]:
    base = _normalize_base_text(s, strip_separators=strip_separators)
    if not base:
        return []
    return base.split()

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

def _build_mode_queries(
    variant: Dict[str, Optional[str]],
    artist_modes: List[str],
    track_modes: List[str],
    found_initial_tag_or_feat_placeholders: bool,
) -> List[str]:
    if not isinstance(variant, dict):
        return []
    pa_raw = variant.get("primary_artist") or ""
    fea_raw_opt = variant.get("featured_artists")
    fea_raw = fea_raw_opt if fea_raw_opt else None
    tr_raw = variant.get("track") or ""
    tag_val = variant.get("tag")

    _, base_tr = _extract_trailing_tag(tr_raw)
    base_tr = base_tr or (tr_raw or "")

    fea_modes: List[Optional[str]] = artist_modes if fea_raw else [None]
    out: List[str] = []
    for pa_mode in artist_modes:
        for fe_mode in fea_modes:
            for t_mode in track_modes:
                if pa_mode == "norm" and t_mode == "norm" and (fe_mode is None or fe_mode == "norm"):
                    continue
                pa2 = _transform_by_mode(pa_raw, pa_mode)
                fea2_val = _transform_by_mode(fea_raw, fe_mode) if fe_mode else ""
                tr2 = _transform_by_mode(base_tr, t_mode)
                skip_tag_feat_extract = (
                    (not found_initial_tag_or_feat_placeholders)
                    and (not tag_val)
                    and (t_mode in ("concat", "spaced"))
                )
                v2 = {
                    "primary_artist": pa2,
                    "featured_artists": fea2_val if fea2_val else None,
                    "track": tr2,
                    "tag": tag_val,
                    "_skip_track_tag_feat_extract": skip_tag_feat_extract,
                }
                q2 = _build_query_from_variant(v2)
                if q2:
                    out.append(q2)
    return out

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

def _tag_tokenize_for_fuzzy(s: Optional[str]) -> List[str]:
    base = _normalize_base_text(s, strip_separators=True)
    if not base:
        return []
    return [p for p in re.findall(r"\w+", base) if p]

def _tag_token_similarity_consecutive(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b:
        return 0.0
    a1 = _normalize_base_text(a)
    b1 = _normalize_base_text(b)
    if not a1 or not b1:
        return 0.0
    a2 = re.sub(r"\W+", "", a1)
    b2 = re.sub(r"\W+", "", b1)
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
    stoks = _normalize_ws_tokens(spotify_title, strip_separators=True)
    gtoks = _normalize_ws_tokens(genius_title, strip_separators=True)
    gftoks = _normalize_ws_tokens(genius_title_with_featured, strip_separators=True)
    k = max(_longest_common_consecutive_len(stoks, gtoks), _longest_common_consecutive_len(stoks, gftoks))
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
    qtoks_sorted = sorted(qtoks, key=lambda x: len(x), reverse=True)
    return _fuzzy_tokens_match_core(qtoks_sorted, ctoks, tokens_percent, chars_percent)

def _tokens_with_charseqs(s: Optional[str]) -> List[str]:
    base = _normalize_base_text(s, strip_separators=True)
    if not base:
        return []
    tokens: List[str] = [p for p in re.split(r"\s+", base) if p]
    words = re.findall(r"\w+", base)
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
        t2 = re.sub(r"'", "", tok)
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
    tokens_sorted = sorted(set(tokens), key=lambda x: len(x), reverse=True)
    return tokens_sorted

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

def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        val = float(value)
        return val if math.isfinite(val) else default
    if isinstance(value, str):
        s = value.strip()
        if s and _RE_FLOAT.fullmatch(s):
            val = float(s)
            return val if math.isfinite(val) else default
    return default

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
    else:
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
    tokens = [w for w in re.findall(r"\w+", t) if len(w) >= 2]
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
    tokens = [w for w in re.findall(r"\w+", t) if w]
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
    penalty = 0
    reasons: List[str] = []
    def _tokset(s: Optional[str]) -> Set[str]:
        n = _normalize_base_text(s)
        if not n:
            return set()
        n = re.sub(r"[\s\-–—_/\\'\"\.,:;!\?]+", " ", n)
        return {w for w in re.findall(r"\w+", n) if w}

    def _strip_feat_from_title_for_penalty(s: Optional[str]) -> str:
        t = _safe_stripped(s)
        if not t:
            return ""
        return strip_feat_placeholders_bracketed(t)

    tag_from_track, base_track = _extract_trailing_tag(spotify_track_name or "")
    base_tokens = _tokset(base_track)
    tag_tokens = _tokset(tag_from_track)
    baseline = base_tokens | tag_tokens
    gtokens = _tokset(_strip_feat_from_title_for_penalty(title_with_featured or ""))
    extra = sorted([t for t in gtokens if t not in baseline])
    missing_base = sorted([t for t in base_tokens if t not in gtokens])
    missing_tag = sorted([t for t in tag_tokens if t not in gtokens])
    join_sep = ", "
    if extra:
        sub = min(GENIUS_PENALTY_TOKEN_CAP, len(extra) * GENIUS_PENALTY_TITLE_EXTRA_TOKEN_PER)
        penalty -= sub
        reasons.append(f"title_extra_tokens -{sub} ({len(extra)} tokens: {join_sep.join(extra)})")
    if missing_base:
        sub = min(GENIUS_PENALTY_TOKEN_CAP, len(missing_base) * GENIUS_PENALTY_TITLE_MISSING_TOKEN_PER)
        penalty -= sub
        reasons.append(f"title_missing_tokens -{sub} ({len(missing_base)} tokens: {join_sep.join(missing_base)})")
    if missing_tag:
        sub = min(GENIUS_PENALTY_TOKEN_CAP, len(missing_tag) * GENIUS_PENALTY_TAG_MISSING_TOKEN_PER)
        penalty -= sub
        reasons.append(f"tag_missing_tokens_title -{sub} ({len(missing_tag)} tokens: {join_sep.join(missing_tag)})")

    sp_feat_tokens: Set[str] = set()
    for p in (featured_pref or []):
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

def _slug_token_match_score(
    song: Dict[str, Any],
    variants: List[Dict[str, Optional[str]]],
    spotify_track_name: Optional[str],
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

async def _api_get(path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    st = get_settings()
    token = st.genius_access_token
    if not token:
        return None
    url = f"{API_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    session = init_session()
    retries = _safe_int(GENIUS_HTTP_MAX_RETRIES, 0)
    if retries < 0:
        retries = 0
    max_attempts = 1 + retries

    base_delay = _safe_float(GENIUS_HTTP_RETRY_DELAY_BASE, 0.0)
    if base_delay < 0:
        base_delay = 0.0

    def _retry_after_seconds(headers: Any) -> Optional[float]:
        try:
            raw = headers.get("Retry-After")
        except Exception:
            raw = None
        if raw is None:
            return None
        s = str(raw).strip()
        if not s:
            return None
        if s.isdigit():
            return float(int(s))
        if _RE_FLOAT.fullmatch(s):
            try:
                return float(s)
            except ValueError:
                return None
        return None

    def _calc_backoff_seconds(attempt_num: int) -> float:
        if base_delay > 0:
            try:
                return float(base_delay) * float(2 ** max(0, attempt_num - 1))
            except (OverflowError, ValueError, TypeError):
                return float(base_delay) * float(attempt_num)
        return float(max(1, attempt_num))

    for attempt in range(1, max_attempts + 1):
        try:
            async with session.get(url, headers=headers, params=params, timeout=30) as resp:
                status = resp.status
                if 200 <= status < 300:
                    return await _read_json_payload(resp)

                if status == 429 and attempt < max_attempts:
                    ra = _retry_after_seconds(getattr(resp, "headers", None) or {})
                    delay_s = ra if isinstance(ra, (int, float)) and ra > 0 else _calc_backoff_seconds(attempt)
                    delay_s = min(max(float(delay_s), 0.5), 60.0)
                    await asyncio.sleep(delay_s)
                    continue

                if 500 <= status < 600 and attempt < max_attempts:
                    delay_s = _calc_backoff_seconds(attempt)
                    delay_s = min(max(float(delay_s), 0.5), 30.0)
                    await asyncio.sleep(delay_s)
                    continue

                if status in (401, 403):
                    _safe_log_error(ERROR_GENIUS_ACCESS_TOKEN_INVALID)
                    return None

                txt = await resp.text(errors="ignore")

                _safe_log_error(
                    ERROR_GENIUS_HTTP,
                    status,
                    txt[:200],
                )
                return None
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as e:
            if attempt < max_attempts:
                delay_s = _calc_backoff_seconds(attempt)
                delay_s = min(max(float(delay_s), 0.5), 30.0)
                await asyncio.sleep(delay_s)
                continue
            _safe_log_error(ERROR_GENIUS_HTTP, "EXC", str(e))
            return None

async def _search_query(q: str) -> List[Dict[str, Any]]:
    per_page = _safe_int(GENIUS_SEARCH_PER_PAGE, 20)
    if per_page <= 0:
        per_page = 20
    payload = await _api_get("/search", params={"q": q, "per_page": per_page})
    if not payload:
        return []
    hits: List[Dict[str, Any]] = []
    response = payload.get("response")
    if isinstance(response, dict):
        for h in response.get("hits") or []:
            if isinstance(h, dict) and h.get("type") == "song":
                song = h.get("result") or {}
                if isinstance(song, dict):
                    hits.append(song)
    return hits

async def _fetch_song(id_: int) -> Optional[Dict[str, Any]]:
    data = await _api_get(f"/songs/{id_}", params={"text_format": "plain"})
    if not data:
        return None
    song = data.get("response", {}).get("song")
    if isinstance(song, dict):
        return song
    return None

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
        tag_tokens = set(re.findall(r"\w+", _nt))
        track_tokens = set(re.findall(r"\w+", _ns))
        pa_norm = _norm(pa)
        fea_norm = _norm(fea)
        pa_tokens = set(re.findall(r"\w+", pa_norm))
        fea_tokens = set(re.findall(r"\w+", fea_norm))
        artist_tokens = pa_tokens | fea_tokens
        single_word = len(tag_tokens) == 1
        has_punct = bool(re.search(r"[^\w\s]", tag_clean))
        if _nt and _nt not in _ns and not (tag_tokens & track_tokens) and not (tag_tokens & artist_tokens) and not (single_word and has_punct):
            parts.append(tag_clean)

    q = _collapse_ws(" ".join(parts))
    return normalize_query_value(q) or ""

def _song_url(song: Dict[str, Any]) -> Optional[str]:
    if not isinstance(song, dict):
        return None
    url = song.get("url")
    if isinstance(url, str) and url:
        if url.rstrip("/").lower().endswith("-lyrics"):
            return url
        u = url.split("/q/", 1)[0]
        if u.rstrip("/").lower().endswith("-lyrics"):
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

def _score_song_detailed(song: Dict[str, Any], variants: List[Dict[str, Optional[str]]], *, featured_pref: Optional[List[str]] = None, top1_counts: Optional[Dict[int, int]] = None, spotify_track_name: Optional[str] = None, spotify_album_name: Optional[str] = None, spotify_album_total_tracks: Optional[int] = None) -> Tuple[int, List[str]]:
    if not isinstance(song, dict):
        return 0, []
    score = 0
    reasons_all: List[str] = []
    title = song.get("title") or ""
    title_with_feat = song.get("title_with_featured") or ""
    relationships = song.get("song_relationships")
    if not isinstance(relationships, list):
        relationships = []
    title_exact_norm = _norm_title_value(title, clear=True)

    best_var = 0
    best_var_reasons: List[str] = []
    for v in variants:
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
        fs, freasons = _featured_match_score(featured_pref, song, title_with_feat)
        v_add += fs
        v_reasons.extend(freasons)
        if v_add > best_var:
            best_var = v_add
            best_var_reasons = v_reasons
    mp, mpreasons = _calc_title_featured_mismatch_penalty(spotify_track_name, title_with_feat, featured_pref, song)
    score += best_var + mp
    reasons_all.extend(best_var_reasons)
    reasons_all.extend(mpreasons)

    cs, creasons = _consec_title_tokens_score(spotify_track_name, title, title_with_feat)
    score += cs
    reasons_all.extend(creasons)

    ss, sreasons = _slug_token_match_score(song, variants, spotify_track_name)
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

def _score_song(song: Dict[str, Any], variants: List[Dict[str, Optional[str]]], *, featured_pref: Optional[List[str]] = None, top1_counts: Optional[Dict[int, int]] = None, spotify_track_name: Optional[str] = None, spotify_album_name: Optional[str] = None, spotify_album_total_tracks: Optional[int] = None) -> int:
    sc, _ = _score_song_detailed(song, variants, featured_pref=featured_pref, top1_counts=top1_counts, spotify_track_name=spotify_track_name, spotify_album_name=spotify_album_name, spotify_album_total_tracks=spotify_album_total_tracks)
    return sc

def _extract_artist_names_from_spotify(artists: Optional[List[Dict[str, Any]]]) -> List[str]:
    if not artists:
        return []
    names = [str(a.get("name") or "").strip() for a in artists if isinstance(a, dict)]
    names = [normalize_apostrophes(n) for n in names]
    return [_collapse_ws(n) for n in names if n]

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
            base_clean = re.sub(r"\s*[-–—]+\s*$", "", base_clean).strip()
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
        m = re.search(r"\s*[\(\[]\s*([^\)\]]+?)\s*[\)\]]\s*$", t)
        mode = ""
        if m:
            mode = "br"
        else:
            m = re.search(r"^(.*)\s+[-–—]+(?=\S)([^\(\)\[\]]*?\S)\s*[-–—]+\s*$", t)
            if m:
                mode = "dash2"
            else:
                m = re.search(r"^(.*)\s+[-–—]+\s+([^\(\)\[\]]+?)\s*$", t)
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
    t = re.sub(r"\s[-–—]\s", " ", t)
    t = re.sub(r"(^|\s)[-–—]\s", r"\1", t)
    t = re.sub(r"\s[-–—](\s|$)", r" ", t)
    t = _collapse_ws(t)
    if not t:
        return []
    parts = [p for p in re.findall(r"\w+", t) if p]
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
            "album": _meta_str(album_name or title_out),
            "release_date": _meta_str(rel_date),
        }
        out.append(meta)
    return out

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
    names_full = _extract_artist_names_from_spotify(spotify_artists)
    if not names_full:
        return None
    primary = names_full[0]
    rest_names = [n for n in names_full[1:] if n]
    featured = ", ".join(rest_names) if rest_names else None
    featured_pref_list = list(rest_names)
    found_initial_tag_or_feat_placeholders = False
    full_track = track_name or ""
    tag_from_track0, base0 = _extract_trailing_tag(full_track)
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

        tag_pat0 = _remix_tag_matched_pattern(tag_from_track0) if tag_from_track0 else None
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

    variants = _ensure_tag_tokenization(variants, original_track=track_name)
    variants = _expand_variants_with_roman_l_i(variants)
    variants = variants[:SEARCH_VARIANTS_LIMIT]
    base_queries: List[str] = []
    extra_symbol_queries: List[str] = []
    extra_mode_queries: List[str] = []
    seen_q: Set[str] = set()

    def _dedupe_key(q: Optional[str]) -> str:
        if not q:
            return ""
        return _collapse_ws(_safe_stripped(q)).lower()

    def _append_query(q: Optional[str], target: List[str], limit: int) -> None:
        if not q or len(target) >= limit:
            return
        qn = _dedupe_key(q)
        if qn and qn not in seen_q:
            seen_q.add(qn)
            target.append(q)

    _svl = _safe_int(SEARCH_VARIANTS_LIMIT, SEARCH_VARIANTS_LIMIT if isinstance(SEARCH_VARIANTS_LIMIT, int) else 100)
    if _svl <= 0:
        _svl = 1
    base_queries_cap = max(1, int(_svl) - max(10, int(_svl) // 3))
    for v in variants:
        q = _build_query_from_variant(v)
        _append_query(q, base_queries, base_queries_cap)
        if len(base_queries) >= base_queries_cap:
            break

    remaining_slots = max(0, _svl - len(base_queries))
    concurrency = max(1, _safe_int(GENIUS_SEARCH_CONCURRENCY, 1))
    sem = asyncio.BoundedSemaphore(concurrency)

    async def run_search(q: str) -> List[Dict[str, Any]]:
        async with sem:
            if _is_genius_detailed_log():
                _safe_log_info(INFO_GENIUS_SEARCH_QUERY, q)
            return await _search_query(q)

    sem2 = asyncio.BoundedSemaphore(concurrency)

    async def guarded_fetch(sid: int) -> Tuple[int, Optional[Dict[str, Any]]]:
        async with sem2:
            song = await _fetch_song(sid)
            return sid, song
    base_songs: Dict[int, Dict[str, Any]] = {}
    top1_counts: Dict[int, int] = {}

    async def _run_search_phase(queries: List[str]) -> None:
        if not queries:
            return
        search_tasks = [asyncio.create_task(run_search(q)) for q in queries]
        try:
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
        except asyncio.CancelledError:
            for t in search_tasks:
                try:
                    t.cancel()
                except Exception:
                    pass
            await asyncio.gather(*search_tasks, return_exceptions=True)
            raise
        for res in results:
            if isinstance(res, BaseException):
                continue
            if not isinstance(res, list):
                continue
            first_sid: Optional[int] = None
            for song in res:
                if not isinstance(song, dict):
                    continue
                sid = _safe_int_or_none(song.get("id"))
                if sid is None:
                    continue
                matched = any(
                    _song_matches_variant_fuzzy(song, v, GENIUS_FUZZY_TOKENS_PERCENT, GENIUS_FUZZY_CHARS_PERCENT)
                    for v in variants
                )
                if not matched:
                    continue
                if first_sid is None:
                    first_sid = sid
                if sid not in base_songs:
                    base_songs[sid] = song

            if first_sid is not None:
                top1_counts[first_sid] = top1_counts.get(first_sid, 0) + 1

    await _run_search_phase(base_queries)
    if not base_songs and remaining_slots > 0:
        for q in list(base_queries):
            if len(extra_symbol_queries) >= remaining_slots:
                break
            q_vars = generate_symbol_letter_equiv_variants(q)
            for q_sym in q_vars:
                _append_query(q_sym, extra_symbol_queries, remaining_slots)
                if len(extra_symbol_queries) >= remaining_slots:
                    break
        if extra_symbol_queries:
            await _run_search_phase(extra_symbol_queries)
    if not base_songs and remaining_slots > 0:
        remaining_for_mode = max(0, remaining_slots - len(extra_symbol_queries))
        if remaining_for_mode > 0:
            artist_modes = ["norm", "concat", "spaced"]
            track_modes = ["norm", "concat", "spaced"]
            for v in variants:
                for q2 in _build_mode_queries(
                    v,
                    artist_modes,
                    track_modes,
                    found_initial_tag_or_feat_placeholders,
                ):
                    _append_query(q2, extra_mode_queries, remaining_for_mode)
                    if len(extra_mode_queries) >= remaining_for_mode:
                        break
                if len(extra_mode_queries) >= remaining_for_mode:
                    break

            if extra_mode_queries:
                await _run_search_phase(extra_mode_queries)

    if not base_songs:
        _log_no_results()
        return None
    base_ids = list(base_songs.keys())
    sorted_ids = sorted(base_ids, key=lambda s: top1_counts.get(s, 0), reverse=True)
    limit_n = _safe_int(GENIUS_DETAILS_TOP_N, 10)
    if limit_n < 0:
        limit_n = 0
    selected_ids = sorted_ids[:limit_n]
    details_results = await asyncio.gather(*(guarded_fetch(s) for s in selected_ids), return_exceptions=True) if selected_ids else []

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
    candidates: List[Dict[str, Any]] = [details_by_id[sid] for sid in base_ids if sid in details_by_id]

    if not candidates:
        pool = list(details_by_id.values()) if details_by_id else list(base_songs.values())
        if pool:
            best_song_fb2 = None
            best_score_fb2 = -10**9
            for s in pool:
                sc = _score_song(
                    s,
                    variants,
                    featured_pref=featured_pref_list,
                    top1_counts=top1_counts,
                    spotify_track_name=track_name,
                    spotify_album_name=spotify_album_name,
                    spotify_album_total_tracks=spotify_album_total_tracks,
                )
                if sc > best_score_fb2:
                    best_score_fb2 = sc
                    best_song_fb2 = s
            if best_song_fb2:
                url_fb2 = _resolve_song_url(best_song_fb2, details_by_id)
                if url_fb2:
                    _safe_log_info(INFO_GENIUS_SCORING_CHOSEN, url_fb2)
                    return url_fb2
        _log_no_results()
        return None
    scored_details: List[Tuple[Dict[str, Any], int, List[str]]] = []
    for s in candidates:
        sc, reasons = _score_song_detailed(
            s,
            variants,
            featured_pref=featured_pref_list,
            top1_counts=top1_counts,
            spotify_track_name=track_name,
            spotify_album_name=spotify_album_name,
            spotify_album_total_tracks=spotify_album_total_tracks,
        )
        scored_details.append((s, sc, reasons))

    scored_sorted = sorted(scored_details, key=lambda x: x[1], reverse=True)
    if _is_genius_detailed_log():
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
            _safe_log_info(
                INFO_GENIUS_CANDIDATE_SUMMARY,
                idx,
                name,
                album_name or "—",
                date or "—",
                sc,
                urlc or "—",
            )
            for r in reasons:
                _safe_log_info(INFO_GENIUS_CANDIDATE_REASON, r)

    best_song = scored_sorted[0][0] if scored_sorted else None
    if not best_song:
        _log_no_results()
        return None

    final_song = best_song
    if _is_mistral_enabled() and scored_sorted:
        spotify_track_meta = _build_spotify_track_meta(
            primary=primary,
            featured=featured,
            track_name=track_name,
            spotify_album_name=spotify_album_name,
            spotify_release_date=spotify_release_date,
        )
        genius_candidates_meta = _build_genius_candidates_meta(scored_sorted)

        idx_choice, explicit_none = await choose_genius_candidate_with_mistral(
            spotify_track_meta,
            genius_candidates_meta,
        )

        if explicit_none:
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

    url = _resolve_song_url(final_song, details_by_id)
    if url:
        _safe_log_info(INFO_GENIUS_SCORING_CHOSEN, url)
        return url
    return None
