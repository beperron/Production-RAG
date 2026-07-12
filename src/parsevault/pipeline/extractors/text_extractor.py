"""Plain-text extraction — the ``.txt`` lane, zero-GPU and dependency-free.

Some authorities arrive already as text: fetched federal statutes/regulations
(Cornell LII) and case law (Wikisource), saved as ``.txt``. Like the DOCX XML
and the native PDF text layer, those characters *are* ground truth, so we read
them directly rather than rasterizing and OCR'ing.

The only wrinkle is scrape chrome: a leading ``Source:``/``Wikisource:`` line
and site navigation fragments (``prev``, ``next``, ``×`` …) that ride along when
a page is saved as text. We drop those conservatively (exact-match nav tokens
and punctuation-only lines) so they don't pollute retrieval. The leading source
line is left for the build driver to read as provenance before extraction.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from .base import BaseExtractor, ExtractionResult
from .textnorm import normalize_unicode

# Exact-match (lower-cased, stripped) nav/boilerplate lines seen in saved-as-text
# legal pages. Kept conservative — only fragments that are never real content.
_BOILERPLATE = {
    "please help us improve our site!", "no thank you", "quick search by citation:",
    "title", "section", "go!", "prev", "next", "share", "print", "download",
    "cite", "authorities (cfr)", "u.s. code", "us law", "us code", "cfr",
    "li / legal information institute", "×", "•",
}


# Wikisource case-law markup. Only applied when a wiki marker is present, so
# statute/regulation text (which never contains these) is left byte-for-byte.
_WIKI_CAT_RE = re.compile(r"^\s*\[\[Category:[^\]]*\]\]\s*$", re.IGNORECASE | re.MULTILINE)
_WIKI_EXTLINK_RE = re.compile(r"\[https?://\S+\s+([^\]]+)\]")  # [url label] -> label
_WIKI_BARELINK_RE = re.compile(r"\[https?://\S+\]")            # [url] -> (drop)
_WIKI_LINK_RE = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]|]+)\]\]")  # [[t|label]]/[[t]] -> label/t
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_APOSTROPHE_EMPH_RE = re.compile(r"'{2,}")  # ''italic'' / '''bold''' -> plain


def _strip_wiki_markup(text: str) -> str:
    if "[[" not in text and "<div" not in text and "[http" not in text:
        return text
    text = _WIKI_CAT_RE.sub("", text)
    text = _WIKI_EXTLINK_RE.sub(r"\1", text)
    text = _WIKI_BARELINK_RE.sub("", text)
    text = _WIKI_LINK_RE.sub(r"\1", text)
    text = _HTML_TAG_RE.sub("", text)
    text = _APOSTROPHE_EMPH_RE.sub("", text)
    return text


def _is_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return False  # blank lines are structural; collapsed later, not dropped
    low = s.lower()
    if low in _BOILERPLATE:
        return True
    # Punctuation/symbol-only fragments ("|", "×", "| next", separators).
    if not any(c.isalnum() for c in s):
        return True
    return False


def _collapse_blanks(lines: list[str]) -> str:
    out: list[str] = []
    blank = 0
    for ln in lines:
        if ln.strip():
            blank = 0
            out.append(ln.rstrip())
        else:
            blank += 1
            if blank <= 1:  # collapse runs of blank lines to a single separator
                out.append("")
    return "\n".join(out).strip()


class PlainTextExtractor(BaseExtractor):
    name = "text"

    def extract(self, file_path: Path, pages: list[int] | None = None) -> ExtractionResult:
        start = time.time()
        file_path = Path(file_path)
        raw = file_path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1", errors="replace")

        text = _strip_wiki_markup(normalize_unicode(text))
        kept: list[str] = []
        for ln in text.splitlines():
            # Drop the leading provenance line (Source:/Wikisource:) — the build
            # driver reads it for provenance; it isn't document content.
            if not kept and ln.strip().lower().startswith(("source:", "wikisource:")):
                continue
            if _is_noise(ln):
                continue
            kept.append(ln)

        markdown = _collapse_blanks(kept)
        return ExtractionResult(
            pdf_name=file_path.stem,
            markdown=markdown,
            page_count=1,  # plain text has no fixed page model; treat as one unit.
            elapsed_seconds=time.time() - start,
            extractor=self.name,
            page_routes=["text"],
        )
