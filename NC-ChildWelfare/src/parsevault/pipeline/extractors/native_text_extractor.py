"""Native text-layer extraction (PyMuPDF) — the zero-cost, zero-GPU lane.

For born-digital PDFs the embedded text layer *is* ground truth: it is the exact
character stream the author placed, so OCR can only lose fidelity. This extractor
reconstructs Markdown from that layer — paragraphs from line geometry, headings
from relative font size, **tables** from the vector grid, **lists** from bullet
markers, and a correct **multi-column reading order** — with no network, no model.

Because the local cascade routes structured-but-born-digital pages here whenever
no VLM is available, the quality of this lane is what carries CPU-only operation.

It is both a standalone ``BaseExtractor`` (whole-document) and a per-page
renderer the cascade composes with the OCR/VLM lanes.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .base import BaseExtractor, ExtractionResult, PageResult, assemble_anchored
from .textnorm import normalize_unicode

# Leading bullet/enumeration markers we promote to Markdown list items.
_BULLET_RE = re.compile(r"^\s*[•‣●▪⁃∙·◦\-\*]\s+(.*)")
_ENUM_RE = re.compile(r"^\s*(\d{1,3}|[a-zA-Z])[.)]\s+(.+)")
# Markers on lines we have ALREADY rendered (used to merge multi-block lists).
_BULLET_OUT_RE = re.compile(r"^- \S")
_ENUM_OUT_RE = re.compile(r"^\d{1,3}\. \S")

# PyMuPDF labels unlabeled table columns "Col1", "Col2", … — placeholders, not
# real headers; blank them so they don't pollute the Markdown or the text index.
_PLACEHOLDER_COL_RE = re.compile(r"^Col\d+$")
_mupdf_errors_silenced = False


def _silence_mupdf_errors() -> None:
    """Stop the MuPDF C layer from printing structure-tree warnings to stderr.

    Also silences ``page.find_tables()``'s stdout advisory in newer PyMuPDF
    builds by setting the message handler to a no-op. Idempotent — safe to
    call from any extraction path; replaces the per-call
    ``contextlib.redirect_stdout``/``redirect_stderr`` pattern, which was
    thread-unsafe (M2) since those routines monkey-patch the process-wide
    ``sys.stdout``/``sys.stderr``.

    L3 audit note: uses the module-level ``_mupdf_errors_silenced`` flag as
    a once-only sentinel; calling repeatedly is a no-op after the first
    successful invocation. Behavior is implicitly tested through every
    ``NativeTextExtractor`` invocation in ``tests/test_extractors_local.py``
    (running the cascade in-process re-uses the same MuPDF process state —
    if the silencing leaked or re-installed handlers, the test stderr would
    show the noise pattern). No dedicated unit test exists because the
    function has no observable return value, no exceptions raised, and
    third-party stderr capture is brittle across PyMuPDF versions.
    """
    global _mupdf_errors_silenced
    if _mupdf_errors_silenced:
        return
    try:
        import fitz

        fitz.TOOLS.mupdf_display_errors(False)
        # Newer PyMuPDF: route MuPDF messages through a no-op handler. Best-
        # effort — older builds don't expose this API and just rely on the
        # display-errors toggle above.
        set_msgs = getattr(fitz.TOOLS, "mupdf_warnings", None)
        if callable(set_msgs):
            try:
                set_msgs(reset=True)
            except Exception:
                pass
    except Exception:
        pass
    _mupdf_errors_silenced = True


def _blank_placeholder(cell: str) -> str:
    return "" if _PLACEHOLDER_COL_RE.match(cell.strip()) else cell


def _clean_table_markdown(md: str) -> str:
    """Blank PyMuPDF "ColN" placeholder headers in a Markdown table's first row."""
    lines = md.split("\n")
    if not lines or "|" not in lines[0]:
        return md
    cells = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    if any(_PLACEHOLDER_COL_RE.match(c) for c in cells):
        cleaned = [_blank_placeholder(c) for c in cells]
        lines[0] = "| " + " | ".join(cleaned) + " |"
    return "\n".join(lines)


def _join_paragraph(lines: list[str]) -> str:
    """Join wrapped lines into one paragraph, healing words hyphenated across a
    line break ("informa-" + "tion" → "information") instead of leaving "- "."""
    out = ""
    for ln in lines:
        if not out:
            out = ln
        elif len(out) >= 2 and out.endswith("-") and out[-2].isalpha() and ln[:1].islower():
            out = out[:-1] + ln
        else:
            out = f"{out} {ln}"
    return out


@dataclass
class _Item:
    """A page element (text block or table) with its bounding box."""

    x0: float
    y0: float
    x1: float
    y1: float
    kind: str  # "text" | "table"
    payload: object  # list[(text,size)] for text; markdown str for table

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0


class NativeTextExtractor(BaseExtractor):
    name = "native"

    def extract(self, file_path: Path, pages: list[int] | None = None) -> ExtractionResult:
        import fitz

        start = time.time()
        doc = fitz.open(file_path)
        page_count = len(doc)
        wanted = set(pages) if pages is not None else None
        page_objs: list[PageResult] = []
        for page in doc:
            if wanted is not None and page.number not in wanted:
                continue
            md = self.render_page(page)
            if md.strip():
                page_objs.append(
                    PageResult(page_number=page.number + 1, markdown=md, route=self.name)
                )
        doc.close()
        from ..quality import score_pages

        score_pages(page_objs)
        # P-2: a scanned (image-only) PDF yields no text-layer pages here.
        # Outside the cascade router, no caller would otherwise know that the
        # empty result was a misuse, not a genuinely empty document — emit a
        # degradation entry so the contract is loud.
        degradations: list[dict] = []
        if page_count > 0 and not page_objs:
            degradations.append({
                "kind": "native_text_empty_fallback",
                "detail": (
                    f"{file_path.name}: all {page_count} page(s) yielded no text "
                    "layer — likely scanned/image-only; escalate to OCR/VLM."
                ),
            })
        return ExtractionResult(
            pdf_name=file_path.stem,
            markdown=assemble_anchored(page_objs),
            page_count=page_count,
            elapsed_seconds=time.time() - start,
            extractor=self.name,
            page_routes=[p.route for p in page_objs],
            pages=page_objs,
            degradations=degradations,
        )

    def render_page(self, page) -> str:
        """Reconstruct Markdown for a single ``fitz.Page`` from its text layer."""
        tables = self._extract_tables(page)
        table_rects = [(t.x0, t.y0, t.x1, t.y1) for t in tables]

        data = page.get_text("dict")
        spans_sizes: list[float] = []
        text_items: list[_Item] = []

        for block in data.get("blocks", []):
            if block.get("type", 0) != 0:  # 0 == text block
                continue
            bx0, by0, bx1, by1 = block.get("bbox", (0, 0, 0, 0))
            if self._inside_any((bx0, by0, bx1, by1), table_rects):
                continue  # text belongs to a table we already captured

            lines: list[tuple[str, float]] = []
            for line in block.get("lines", []):
                parts_, max_size = [], 0.0
                for span in line.get("spans", []):
                    txt = span.get("text", "")
                    if txt:
                        parts_.append(txt)
                        size = float(span.get("size", 0))
                        max_size = max(max_size, size)
                        spans_sizes.append(size)
                line_text = "".join(parts_).strip()
                if line_text:
                    lines.append((line_text, max_size))
            if lines:
                text_items.append(_Item(bx0, by0, bx1, by1, "text", lines))

        items = text_items + tables
        if not items:
            return ""

        body_size = (
            Counter(round(s) for s in spans_sizes).most_common(1)[0][0]
            if spans_sizes
            else 1
        ) or 1

        ordered = self._reading_order(items, page.rect.width)
        blocks: list[str] = []
        for item in ordered:
            if item.kind == "table":
                rendered = item.payload
            else:
                rendered = self._render_text_item(item.payload, body_size)
            if not rendered:
                continue
            # Merge a list that spans multiple text blocks (one bullet per block
            # is common) into a single contiguous Markdown list.
            if blocks and self._is_list_block(rendered) and self._is_list_block(blocks[-1]):
                blocks[-1] = blocks[-1] + "\n" + rendered
            else:
                blocks.append(rendered)
        return normalize_unicode("\n\n".join(blocks))

    @staticmethod
    def _is_list_block(s: str) -> bool:
        lines = [ln for ln in s.split("\n") if ln.strip()]
        return bool(lines) and all(
            _BULLET_OUT_RE.match(ln) or _ENUM_OUT_RE.match(ln) for ln in lines
        )

    # -- tables ---------------------------------------------------------------
    def _extract_tables(self, page) -> list[_Item]:
        # Only the line-based ("lines_strict") strategy is used. The "text"
        # strategy recovers ruling-less form tables but, measured against the
        # LlamaParse gold, it invents tables from aligned PROSE and drops cells —
        # dropping prose content recall ~0.99 → 0.83, which hurts retrieval. Since
        # the GPU-free lanes already capture table DATA as prose (high content
        # recall, retrieval-neutral), recovering ruling-less table STRUCTURE is
        # left to the VLM lane (GPU). See docs/CONVERSION_QUALITY.md.
        #
        # M2: MuPDF noise (a one-line advisory to stdout from find_tables() and
        # structure-tree warnings to stderr) is quieted ONCE at first call by
        # ``_silence_mupdf_errors`` — no per-call ``redirect_stdout``/
        # ``redirect_stderr`` here. Those routines monkey-patch the process-wide
        # ``sys.stdout``/``sys.stderr`` and would race a concurrent caller in a
        # page-parallel future (the cascade is serial today, so the bug was
        # latent — but the redirect is the kind of thing that bites silently the
        # first time a driver parallelizes native extraction).
        _silence_mupdf_errors()
        try:
            found = page.find_tables()
        except Exception:
            return []
        items: list[_Item] = []
        for t in getattr(found, "tables", []) or []:
            md = self._table_to_markdown(t)
            if md:
                x0, y0, x1, y1 = t.bbox
                items.append(_Item(x0, y0, x1, y1, "table", md))
        return items

    @staticmethod
    def _table_to_markdown(table) -> str:
        # PyMuPDF tables can self-serialize; fall back to building from cells.
        try:
            md = table.to_markdown()
            if md and md.strip():
                return _clean_table_markdown(md.strip())
        except Exception:
            pass
        try:
            rows = table.extract()
        except Exception:
            return ""
        rows = [[("" if c is None else str(c)).replace("\n", " ").strip() for c in row] for row in rows]
        rows = [r for r in rows if any(cell for cell in r)]
        if not rows:
            return ""
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        header = [_blank_placeholder(c) for c in rows[0]]
        body = rows[1:] if len(rows) > 1 else []
        lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
        for r in body:
            lines.append("| " + " | ".join(r) + " |")
        return "\n".join(lines)

    # -- geometry / reading order --------------------------------------------
    @staticmethod
    def _inside_any(bbox, rects, tol: float = 2.0) -> bool:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        for x0, y0, x1, y1 in rects:
            if x0 - tol <= cx <= x1 + tol and y0 - tol <= cy <= y1 + tol:
                return True
        return False

    @staticmethod
    def _reading_order(items: list[_Item], page_width: float) -> list[_Item]:
        """Order items as a human reads them, including two-column layouts.

        Full-width items (titles, wide tables) stay in the left flow by vertical
        position; a genuine two-column body is emitted left column then right.
        Single-column (the common case) is a plain top-to-bottom, left-to-right
        sort — which also repairs PyMuPDF's occasionally out-of-order blocks.
        """
        if page_width <= 0 or len(items) < 4:
            return sorted(items, key=lambda it: (it.y0, it.x0))
        mid = page_width / 2.0
        full = [it for it in items if it.width > page_width * 0.6]
        narrow = [it for it in items if it.width <= page_width * 0.6]
        left = [it for it in narrow if it.center_x < mid]
        right = [it for it in narrow if it.center_x >= mid]
        if len(left) >= 2 and len(right) >= 2:
            left_col = sorted(full + left, key=lambda it: (it.y0, it.x0))
            right_col = sorted(right, key=lambda it: (it.y0, it.x0))
            return left_col + right_col
        return sorted(items, key=lambda it: (it.y0, it.x0))

    # -- text rendering -------------------------------------------------------
    def _render_text_item(self, lines: list[tuple[str, float]], body_size: float) -> str:
        # Each segment is a standalone block (heading / paragraph / list); a run
        # of list items is kept in ONE segment (single-newline) so it renders as
        # a single Markdown list rather than many loose one-item lists.
        segments: list[str] = []
        paragraph: list[str] = []
        list_buf: list[str] = []

        def flush_para():
            if paragraph:
                segments.append(_join_paragraph(paragraph))
                paragraph.clear()

        def flush_list():
            if list_buf:
                segments.append("\n".join(list_buf))
                list_buf.clear()

        for text, size in lines:
            level = self._heading_level(text, size, body_size)
            if level:
                flush_para()
                flush_list()
                segments.append(f"{'#' * level} {text}")
                continue
            bullet = _BULLET_RE.match(text)
            enum = _ENUM_RE.match(text)
            if bullet:
                flush_para()
                list_buf.append(f"- {bullet.group(1).strip()}")
            elif enum:
                flush_para()
                list_buf.append(f"{enum.group(1)}. {enum.group(2).strip()}")
            else:
                flush_list()
                paragraph.append(text)
        flush_para()
        flush_list()
        return "\n\n".join(segments)

    @staticmethod
    def _heading_level(text: str, size: float, body_size: float) -> int:
        """0 if body text, else a heading level 1–3 by relative font size.

        Headings are short lines set noticeably larger than the body. The length
        guard prevents large-font *paragraphs* (rare, but e.g. pull quotes) from
        becoming headings.
        """
        if body_size <= 0 or len(text) > 120:
            return 0
        ratio = size / body_size
        if ratio >= 1.6:
            return 1
        if ratio >= 1.3:
            return 2
        if ratio >= 1.15:
            return 3
        return 0
