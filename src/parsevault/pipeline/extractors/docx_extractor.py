"""DOCX text extraction — the Word-native lane, zero-GPU and dependency-free.

Word forms and templates ship as ``.docx`` (a zip of XML). Like the native PDF
text layer, that XML *is* ground truth — the exact authored characters — so we
read it directly rather than rasterizing and OCR'ing. Uses only the standard
library (``zipfile`` + ``xml.etree``): paragraphs from ``<w:p>`` (heading level
from the paragraph style), and tables from ``<w:tbl>`` rendered as Markdown.
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from .base import BaseExtractor, ExtractionResult
from .textnorm import normalize_unicode

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _para_text(p: ET.Element) -> str:
    """Concatenate the run text of a paragraph, honoring tabs and line breaks."""
    parts: list[str] = []
    for node in p.iter():
        tag = node.tag
        if tag == f"{_W}t":
            parts.append(node.text or "")
        elif tag == f"{_W}tab":
            parts.append("\t")
        elif tag in (f"{_W}br", f"{_W}cr"):
            parts.append("\n")
    return "".join(parts).strip()


def _heading_level(p: ET.Element) -> int:
    """Markdown heading level from the paragraph's style (Heading1 → 1), else 0."""
    style = p.find(f"{_W}pPr/{_W}pStyle")
    if style is None:
        return 0
    val = (style.get(f"{_W}val") or "").lower()
    if val.startswith("heading"):
        digits = "".join(c for c in val if c.isdigit())
        if digits:
            return min(int(digits), 6)
    if val in ("title",):
        return 1
    return 0


def _table_markdown(tbl: ET.Element) -> str:
    rows: list[list[str]] = []
    for tr in tbl.findall(f"{_W}tr"):
        cells = [
            " ".join(_para_text(p) for p in tc.findall(f"{_W}p")).strip()
            for tc in tr.findall(f"{_W}tc")
        ]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


class DocxExtractor(BaseExtractor):
    name = "docx"

    def extract(self, file_path: Path, pages: list[int] | None = None) -> ExtractionResult:
        start = time.time()
        file_path = Path(file_path)
        with zipfile.ZipFile(file_path) as z:
            xml = z.read("word/document.xml")
        body = ET.fromstring(xml).find(f"{_W}body")

        blocks: list[str] = []
        # Iterate top-level body children so paragraphs and tables stay in order.
        for el in list(body) if body is not None else []:
            if el.tag == f"{_W}p":
                text = _para_text(el)
                if not text:
                    continue
                level = _heading_level(el)
                blocks.append(f"{'#' * level} {text}" if level else text)
            elif el.tag == f"{_W}tbl":
                md = _table_markdown(el)
                if md:
                    blocks.append(md)

        markdown = normalize_unicode("\n\n".join(blocks)).strip()
        return ExtractionResult(
            pdf_name=file_path.stem,
            markdown=markdown,
            page_count=1,  # DOCX has no fixed page model; treat as a single unit.
            elapsed_seconds=time.time() - start,
            extractor=self.name,
        )
