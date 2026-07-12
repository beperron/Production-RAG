"""Text normalization & cleanup shared by the extraction lanes.

These are the unglamorous fixes that separate a usable transcription from a
noisy one — and exactly the failure modes that hurt retrieval: soft-hyphen line
breaks, ligatures/zero-width junk, running headers/footers/page numbers repeated
on every page, and *broken text layers* (born-digital PDFs whose embedded text
is garbage, where a frontier system would fall back to OCR and we must too).

All pure-Python, CPU-only, no dependencies.
"""

from __future__ import annotations

import re
import unicodedata

# Common ligatures NFKC misses or that we want explicit control over.
_LIGATURES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
    " ": " ", "​": "", "‌": "", "‍": "", "﻿": "",
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "—",
    "‘": "'", "’": "'", "“": '"', "”": '"', "…": "…",
}
_LIG_RE = re.compile("|".join(map(re.escape, _LIGATURES)))

# A word split across a line break: "informa-\ntion" → "information".
_HYPHEN_BREAK_RE = re.compile(r"([A-Za-z])-\s*\n\s*([a-z])")
# Page-number-only lines ("12", "- 12 -", "Page 12 of 30").
_PAGE_NUM_RE = re.compile(
    r"^\s*(?:[-–—]\s*)?(?:page\s+)?\d{1,4}(?:\s*(?:/|of)\s*\d{1,4})?\s*(?:[-–—])?\s*$",
    re.IGNORECASE,
)


def normalize_unicode(text: str) -> str:
    """NFKC + ligature/zero-width cleanup + whitespace tidy (keeps newlines)."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _LIG_RE.sub(lambda m: _LIGATURES[m.group(0)], text)
    # Collapse runs of spaces/tabs but preserve line structure.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n", "\n", text)
    return text.strip()


def dehyphenate(text: str) -> str:
    """Join words hyphenated across a line break (conservative: lower-after)."""
    return _HYPHEN_BREAK_RE.sub(r"\1\2", text)


def clean_text(text: str) -> str:
    """The standard cleanup applied to a transcribed page: normalize first (so
    ligatures like 'eﬃ-' become 'effi-'), then heal hyphenated line breaks."""
    return dehyphenate(normalize_unicode(text))


# --------------------------------------------------------------------------- #
# running header / footer / page-number stripping (cross-page)
# --------------------------------------------------------------------------- #
def strip_running_headers(pages: list[str], *, min_fraction: float = 0.6) -> list[str]:
    """Remove headers/footers/page numbers repeated across many pages.

    A running header is the same first (or last) non-empty line appearing on at
    least ``min_fraction`` of pages — journal titles, chapter names, page
    numbers. They add no information and pollute retrieval, so we drop them. Pure
    page-number lines are removed regardless of repetition.
    """
    if len(pages) < 3:
        # Too few pages to infer repetition reliably; only drop pure page numbers.
        return [_drop_page_numbers(p) for p in pages]

    from collections import Counter

    firsts: Counter = Counter()
    lasts: Counter = Counter()
    for p in pages:
        lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
        if not lines:
            continue
        firsts[lines[0]] += 1
        lasts[lines[-1]] += 1

    threshold = max(2, int(len(pages) * min_fraction))
    repeated = {ln for ln, c in firsts.items() if c >= threshold and not ln.startswith("#")}
    repeated |= {ln for ln, c in lasts.items() if c >= threshold and not ln.startswith("#")}

    out = []
    for p in pages:
        lines = p.splitlines()
        kept = []
        n = len(lines)
        # Only consider the first/last non-empty line of each page for removal.
        first_idx = next((i for i, ln in enumerate(lines) if ln.strip()), None)
        last_idx = next((i for i in range(n - 1, -1, -1) if lines[i].strip()), None)
        for i, ln in enumerate(lines):
            s = ln.strip()
            if i in (first_idx, last_idx) and s in repeated:
                continue
            kept.append(ln)
        out.append(_drop_page_numbers("\n".join(kept)))
    return out


def _drop_page_numbers(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not _PAGE_NUM_RE.match(ln)]
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------- #
# broken-text-layer detection
# --------------------------------------------------------------------------- #
def looks_garbled(text: str) -> bool:
    """Heuristic: does this embedded text layer look broken/unreliable?

    Catches the real failure mode where a born-digital PDF's text layer is
    mojibake (bad encoding, no spaces, replacement chars) — in which case the
    page should be OCR'd despite "having text". Tuned to be conservative: clean
    prose returns False.
    """
    t = text.strip()
    if len(t) < 40:
        return False  # too little to judge; let routing handle emptiness
    if t.count("�") / len(t) > 0.02:
        return True  # replacement characters — unambiguous corruption
    space_ratio = t.count(" ") / len(t)
    # No spaces over a long run → text layer lost word boundaries (mojibake).
    if len(t) > 200 and space_ratio < 0.02:
        return True
    letters = sum(c.isalpha() for c in t)
    letter_ratio = letters / len(t)
    # Letter-sparse is NOT garbled on its own — forms, tables, and fee schedules
    # are legitimately full of digits/checkboxes/pipes. Only treat it as broken
    # when sparseness co-occurs with abnormal spacing (true mojibake), or when
    # there are almost no letters at all.
    if letter_ratio < 0.5 and space_ratio < 0.06:
        return True
    if letter_ratio < 0.2:
        return True
    # Implausible mean token length (gibberish runs very long or 1-char),
    # ignoring table-cell separators so a normal table isn't misread. The 1-char
    # case (a broken text layer where every glyph is split) needs only a modest
    # run; the run-on case needs a substantial one to avoid false positives.
    words = [w for w in re.findall(r"[^\s|]+", t) if any(c.isalnum() for c in w)]
    if words:
        mean_len = sum(len(w) for w in words) / len(words)
        # Run-on gibberish (lost word boundaries): a high mean token length means
        # most tokens are 18+ chars — never true of normal prose, forms, or tables.
        if mean_len > 18:
            return True
        # Single-char gibberish needs a modest run so a tiny table of one-char
        # cells isn't misread as broken.
        if len(words) >= 12 and mean_len < 1.6:
            return True
    return False
