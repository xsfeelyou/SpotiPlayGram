import unicodedata
import re
import json
import os
import importlib
from typing import Optional, Iterable
from .symbols import (
    _CHAR_EQUIV_MAP,
    PRIMARY_SEPARATORS,
    SECONDARY_SEPARATORS,
    SYMBOL_LETTER_EQUIV,
    APOSTROPHE_CHARS,
)

_DIR = os.path.dirname(__file__)
_FEAT_PATH = os.path.join(_DIR, "feat.json")
_TAG_PATH = os.path.join(_DIR, "tag.json")

_FEAT_CACHE: list[str] | None = None
_TAG_CACHE: list[str] | None = None
_FEAT_VERSION: int = 0
_TAG_VERSION: int = 0
_PENDING_FEAT_RELOAD: bool = False
_PENDING_TAG_RELOAD: bool = False

_APOSTROPHE_TRANSLATION = {ord(ch): "'" for ch in APOSTROPHE_CHARS if ch != "'"}

PRIMARY_SEPARATORS = list(dict.fromkeys(PRIMARY_SEPARATORS + APOSTROPHE_CHARS))
SECONDARY_SEPARATORS = list(dict.fromkeys(SECONDARY_SEPARATORS + APOSTROPHE_CHARS))

_PRIMARY_SET = set(PRIMARY_SEPARATORS)
_SECONDARY_SET = set(SECONDARY_SEPARATORS)
_ALL_SEP_SET = _PRIMARY_SET | _SECONDARY_SET

_SEP_CLASS = "".join([re.escape(ch) for ch in SECONDARY_SEPARATORS])

_FEAT_MARKERS_CACHE: set[str] | None = None
_FEAT_MARKERS_MULTI_CACHE: list[list[str]] | None = None
_FEAT_MARKERS_VERSION: int | None = None

_RE_PUNCT_WS_LEFT = re.compile(rf"(?<=\s)[{_SEP_CLASS}]+")
_RE_PUNCT_WS_RIGHT = re.compile(rf"[{_SEP_CLASS}]+(?=\s)")
_RE_SEP_TO_SPACE = re.compile(rf"(?:\s|[{_SEP_CLASS}]|[\\/])+")

_ROMAN_BASE_CHARS = "IVXLCDM"
_RE_ROMAN_L_I_SEGMENT = re.compile(
    rf"(?<!\w)([{_ROMAN_BASE_CHARS}lI]*[lI][{_ROMAN_BASE_CHARS}lI]*)(?!\w)"
)

_FEAT_PLACEHOLDER_PATTERNS = [
    re.compile(r"[\(\[]\s*([^\)\]]+?)\s*[\)\]]"),
    re.compile(r"\s+[-–—]+(?=\S)([^\(\)\[\]]*?\S)\s*[-–—]+\s*$"),
    re.compile(r"\s+[-–—]+\s+([^\(\)\[\]]+?)\s*$"),
]

def _normalize_item(value: str) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())

def _normalize_items(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for raw in values:
        k = _normalize_item(raw).lower()
        if k:
            out.append(k)
    return out

def _unique_sorted(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in items:
        v = _normalize_item(raw).lower()
        if not v:
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    out.sort(key=lambda x: (len(x), x))
    return out

def _read_json_list_raw(path: str) -> list[str] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, list):
        return None
    return [str(x) for x in data if x is not None]

def _load_builtin_patterns(module_name: str, attr: str) -> list[str]:
    try:
        mod = importlib.import_module(module_name, package=__package__)
    except ImportError:
        return []
    value = getattr(mod, attr, None)
    return list(value) if isinstance(value, (list, tuple)) else []

def _read_json_list(path: str) -> list[str] | None:
    items = _read_json_list_raw(path)
    if items is None:
        return None
    return _unique_sorted(items)

def _write_json_list(path: str, items: Iterable[str]) -> None:
    out = _unique_sorted(items)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _ensure_initial_files() -> None:
    if not os.path.exists(_FEAT_PATH):
        src = _load_builtin_patterns(".feats", "FEAT_PATTERNS")
        _write_json_list(_FEAT_PATH, src)
    if not os.path.exists(_TAG_PATH):
        src = _load_builtin_patterns(".remixes", "REMIX_PATTERNS")
        _write_json_list(_TAG_PATH, src)

def get_feat_patterns_version() -> int:
    global _FEAT_CACHE
    if _FEAT_CACHE is None:
        get_feat_patterns()
    return _FEAT_VERSION

def get_tag_patterns_version() -> int:
    global _TAG_CACHE
    if _TAG_CACHE is None:
        get_tag_patterns()
    return _TAG_VERSION

def get_feat_patterns() -> list[str]:
    global _FEAT_CACHE, _FEAT_VERSION
    if _FEAT_CACHE is not None:
        return list(_FEAT_CACHE)
    _ensure_initial_files()
    data = _read_json_list(_FEAT_PATH) or []
    _FEAT_CACHE = data
    _FEAT_VERSION += 1
    return list(_FEAT_CACHE)

def get_tag_patterns() -> list[str]:
    global _TAG_CACHE, _TAG_VERSION
    if _TAG_CACHE is not None:
        return list(_TAG_CACHE)
    _ensure_initial_files()
    data = _read_json_list(_TAG_PATH) or []
    _TAG_CACHE = data
    _TAG_VERSION += 1
    return list(_TAG_CACHE)

def _update_patterns(kind: str, new_items: list[str]) -> None:
    global _PENDING_FEAT_RELOAD, _PENDING_TAG_RELOAD
    _ensure_initial_files()
    if kind == "feat":
        _write_json_list(_FEAT_PATH, new_items)
        _PENDING_FEAT_RELOAD = True
        return
    if kind == "tag":
        _write_json_list(_TAG_PATH, new_items)
        _PENDING_TAG_RELOAD = True
        return
    raise ValueError("unknown kind")

def _read_patterns_by_kind(kind: str) -> list[str]:
    _ensure_initial_files()
    if kind == "feat":
        return _read_json_list(_FEAT_PATH) or []
    if kind == "tag":
        return _read_json_list(_TAG_PATH) or []
    raise ValueError("unknown kind")

def apply_pending_reload() -> tuple[bool, bool]:
    global _PENDING_FEAT_RELOAD, _PENDING_TAG_RELOAD
    global _FEAT_CACHE, _TAG_CACHE, _FEAT_VERSION, _TAG_VERSION
    _ensure_initial_files()
    feat_applied = False
    tag_applied = False
    if _PENDING_FEAT_RELOAD:
        _FEAT_CACHE = _read_json_list(_FEAT_PATH) or []
        _FEAT_VERSION += 1
        _PENDING_FEAT_RELOAD = False
        feat_applied = True
    if _PENDING_TAG_RELOAD:
        _TAG_CACHE = _read_json_list(_TAG_PATH) or []
        _TAG_VERSION += 1
        _PENDING_TAG_RELOAD = False
        tag_applied = True
    return feat_applied, tag_applied

def normalize_files() -> tuple[bool, bool]:
    global _PENDING_FEAT_RELOAD, _PENDING_TAG_RELOAD
    _ensure_initial_files()

    raw_tag = _read_json_list_raw(_TAG_PATH) or []
    raw_feat = _read_json_list_raw(_FEAT_PATH) or []

    tag_norm = _unique_sorted(raw_tag)
    feat_norm = _unique_sorted(raw_feat)

    tag_set = set(tag_norm)
    feat_norm = [p for p in feat_norm if p not in tag_set]

    tag_changed = raw_tag != tag_norm
    feat_changed = raw_feat != feat_norm

    if tag_changed:
        _write_json_list(_TAG_PATH, tag_norm)
        _PENDING_TAG_RELOAD = True
    if feat_changed:
        _write_json_list(_FEAT_PATH, feat_norm)
        _PENDING_FEAT_RELOAD = True

    return feat_changed, tag_changed

def add_patterns(kind: str, patterns: Iterable[str]) -> list[str]:
    items = _normalize_items(patterns)
    cur = _read_patterns_by_kind(kind)
    existing = set(cur)
    merged = list(cur)
    for p in items:
        if p not in existing:
            existing.add(p)
            merged.append(p)
    merged = _unique_sorted(merged)
    _update_patterns(kind, merged)
    return merged

def remove_patterns(kind: str, patterns: Iterable[str]) -> list[str]:
    items = _normalize_items(patterns)
    cur = _read_patterns_by_kind(kind)
    to_remove = set(items)
    merged = [x for x in cur if x not in to_remove]
    merged = _unique_sorted(merged)
    _update_patterns(kind, merged)
    return merged

def pattern_exists(kind: str, pattern: str) -> bool:
    p = _normalize_item(pattern).lower()
    if not p:
        return False
    if kind not in ("feat", "tag"):
        return False
    cur = _read_patterns_by_kind(kind)
    return p in cur

def apply_symbol_letter_equiv(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    s = str(text)
    return "".join(SYMBOL_LETTER_EQUIV.get(ch, ch) for ch in s)

def generate_roman_l_i_variants(text: Optional[str]) -> list[str]:
    if text is None:
        return []
    s = str(text)
    if not s:
        return []

    matches = list(_RE_ROMAN_L_I_SEGMENT.finditer(s))
    if not matches:
        return [s]

    seg_original: list[tuple[int, int, str]] = []
    seg_swapped: list[str] = []

    for m in matches:
        start, end = m.span(1)
        seg = m.group(1)
        if not seg:
            continue
        seg_original.append((start, end, seg))
        out_chars: list[str] = []
        for ch in seg:
            if ch == "l":
                out_chars.append("I")
            elif ch == "I":
                out_chars.append("l")
            else:
                out_chars.append(ch)
        seg_swapped.append("".join(out_chars))

    if not seg_original:
        return [s]

    n = len(seg_original)
    variants: list[str] = []

    for mask in range(1 << n):
        res_parts: list[str] = []
        last_end = 0
        for i in range(n):
            start, end, seg = seg_original[i]
            if start > last_end:
                res_parts.append(s[last_end:start])
            if (mask >> i) & 1:
                res_parts.append(seg_swapped[i])
            else:
                res_parts.append(seg)
            last_end = end
        if last_end < len(s):
            res_parts.append(s[last_end:])
        v = "".join(res_parts)
        if v not in variants:
            variants.append(v)
    return variants or [s]

def generate_symbol_letter_equiv_variants(text: Optional[str]) -> list[str]:
    if text is None:
        return []
    s = str(text)
    if not s:
        return []

    positions: list[int] = []
    repl_chars: list[str] = []
    for idx, ch in enumerate(s):
        if ch in SYMBOL_LETTER_EQUIV:
            mapped = SYMBOL_LETTER_EQUIV.get(ch, ch)
            if mapped != ch:
                positions.append(idx)
                repl_chars.append(mapped)

    if not positions:
        return [s]

    n = len(positions)
    variants: list[str] = []

    for mask in range(1 << n):
        chars = list(s)
        for bit, pos in enumerate(positions):
            if (mask >> bit) & 1:
                chars[pos] = repl_chars[bit]
        v = "".join(chars)
        if v not in variants:
            variants.append(v)
    return variants or [s]

def normalize_letters(text: str) -> str:
    if not text:
        return text
    s = unicodedata.normalize("NFKC", text)
    decomp = unicodedata.normalize("NFKD", s)
    no_marks = "".join(ch for ch in decomp if not unicodedata.combining(ch))
    out_parts = []
    for ch in no_marks:
        out_parts.append(_CHAR_EQUIV_MAP.get(ch, ch))
    return "".join(out_parts)

def _normalize_marker_tokens(text: Optional[str]) -> list[str]:
    if text is None:
        return []
    s = normalize_letters(str(text)).lower()
    return s.split()

def _best_multi_pattern(tokens: list[str], multi: list[list[str]]) -> list[str] | None:
    if not tokens or not multi:
        return None
    n = len(tokens)
    best: list[str] | None = None
    for pat in multi:
        m = len(pat)
        if m == 0 or m > n:
            continue
        for i in range(0, n - m + 1):
            if tokens[i:i + m] == pat:
                if best is None or m > len(best):
                    best = pat
                break
    return best

def _get_feat_markers_norm() -> tuple[set[str], list[list[str]]]:
    global _FEAT_MARKERS_CACHE, _FEAT_MARKERS_MULTI_CACHE, _FEAT_MARKERS_VERSION
    cur_version = get_feat_patterns_version()
    if (
        _FEAT_MARKERS_CACHE is not None
        and _FEAT_MARKERS_MULTI_CACHE is not None
        and _FEAT_MARKERS_VERSION == cur_version
    ):
        return _FEAT_MARKERS_CACHE, _FEAT_MARKERS_MULTI_CACHE
    single: set[str] = set()
    multi: list[list[str]] = []
    patterns = get_feat_patterns()
    for raw in patterns:
        toks = _normalize_marker_tokens(raw)
        if not toks:
            continue
        if len(toks) > 1:
            multi.append(toks)
        else:
            single.add(toks[0])
    _FEAT_MARKERS_CACHE = single
    _FEAT_MARKERS_MULTI_CACHE = multi
    _FEAT_MARKERS_VERSION = cur_version
    return single, multi

def _feat_pattern_in_tokens(tokens: list[str]) -> bool:
    if not tokens:
        return False
    single, multi = _get_feat_markers_norm()
    for t in tokens:
        if t in single:
            return True
    return _best_multi_pattern(tokens, multi) is not None

def _feat_pattern_in_tokens_match(tokens: list[str]) -> Optional[str]:
    if not tokens:
        return None
    single, multi = _get_feat_markers_norm()
    best_multi = _best_multi_pattern(tokens, multi)
    if best_multi is not None:
        return " ".join(best_multi).strip()

    for t in tokens:
        if t in single:
            return t
    return None

def _is_feat_placeholder_inner(inner: Optional[str]) -> bool:
    tokens = _normalize_marker_tokens(inner)
    return _feat_pattern_in_tokens(tokens)

def feat_placeholder_pattern(text: Optional[str]) -> Optional[str]:
    tokens = _normalize_marker_tokens(text)
    return _feat_pattern_in_tokens_match(tokens)

def strip_feat_placeholders_bracketed(text: str) -> str:
    if not text:
        return text
    s = str(text)

    removed_any = {"v": False}

    def _repl(m: re.Match) -> str:
        inner = m.group(1)
        if _is_feat_placeholder_inner(inner):
            removed_any["v"] = True
            return " "
        return m.group(0)
    for pat in _FEAT_PLACEHOLDER_PATTERNS:
        s = pat.sub(_repl, s)
    if removed_any["v"]:
        s = re.sub(r"\s*[-–—]+\s*$", " ", s)
    return _normalize_item(s)

def detect_feat_placeholders_bracketed(text: Optional[str]) -> list[str]:
    if not text:
        return []
    s = str(text)
    found: list[str] = []
    seen: set[str] = set()
    for pat in _FEAT_PLACEHOLDER_PATTERNS:
        for m in pat.finditer(s):
            inner = m.group(1)
            if _is_feat_placeholder_inner(inner):
                val = inner.strip()
                if val and val not in seen:
                    seen.add(val)
                    found.append(val)
    return found

def normalize_apostrophes(text: str) -> str:
    if text is None:
        return text
    return str(text).translate(_APOSTROPHE_TRANSLATION)

def strip_spaced_separators(text: str) -> str:
    if text is None:
        return text
    s = str(text)
    s = _RE_PUNCT_WS_LEFT.sub("", s)
    s = _RE_PUNCT_WS_RIGHT.sub("", s)
    s = re.sub(rf"^[{_SEP_CLASS}]+", "", s)
    s = re.sub(rf"[{_SEP_CLASS}]+$", "", s)
    return _normalize_item(s)

def _remove_separators_with_set(text: str, sep_set: set[str]) -> str:
    if text is None:
        return text
    s = str(text)
    if not s:
        return s
    out_chars: list[str] = []
    n = len(s)
    for i, ch in enumerate(s):
        if ch in sep_set:
            left = s[i - 1] if i > 0 else None
            right = s[i + 1] if i + 1 < n else None
            left_ok = left is not None and not (left.isspace() or left in _ALL_SEP_SET)
            right_ok = right is not None and not (right.isspace() or right in _ALL_SEP_SET)
            if left_ok and right_ok:
                out_chars.append(ch)
            else:
                continue
        else:
            out_chars.append(ch)
    return "".join(out_chars)

def normalize_query_value(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    s = str(text)
    s = normalize_apostrophes(s)
    s = s.replace("'", "")
    s = strip_feat_placeholders_bracketed(s)
    s = strip_spaced_separators(s)
    s = _remove_separators_with_set(s, _PRIMARY_SET)
    return _normalize_item(s)

def convert_separators_to_space(text: str) -> str:
    if text is None:
        return text
    s = str(text)
    return _RE_SEP_TO_SPACE.sub(" ", s)
