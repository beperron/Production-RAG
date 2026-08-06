"""Base extractor interface and shared data structures."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

# Page boundaries are carried through the joined Markdown as HTML-comment anchors
# so the saved ``{doc_id}.md`` is navigable AND chunking can stamp page spans.
# This is the single definition of that contract (chunking matches PAGE_ANCHOR_RE).
PAGE_ANCHOR_FMT = "<!-- page: {n} -->"
PAGE_ANCHOR_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")


def page_anchor(n: int) -> str:
    return PAGE_ANCHOR_FMT.format(n=n)


@dataclass
class PageResult:
    """One physical page's transcription, with the lane that produced it.

    Carrying page boundaries out of extraction is what makes page-level citation
    possible downstream: the joined document Markdown loses page structure, so the
    per-page list is the artifact chunking uses to stamp page spans onto chunks.
    """
    page_number: int           # 1-based physical page
    markdown: str
    route: str = ""            # the lane that transcribed this page
    elapsed_seconds: float = 0.0
    # Per-page quality record (pipeline.quality.PageQuality.to_dict()); populated
    # by the cascade. Empty for lanes that don't measure quality. (R3)
    quality: dict = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result from a single document extraction."""
    pdf_name: str  # kept for backwards compatibility; holds any document stem
    markdown: str
    images: list[Path] = field(default_factory=list)
    page_count: int = 0
    elapsed_seconds: float = 0.0
    extractor: str = ""
    # Per-page provenance for the cascade: which lane transcribed each page
    # (e.g. ["native", "vlm_quality", "tesseract"]). Empty for single-lane
    # extractors. Lets the agent/benchmark see where OCR/VLM was actually used.
    page_routes: list[str] = field(default_factory=list)
    # Per-page transcription (1-based). The keystone for page-level traceability:
    # the joined ``markdown`` is reconstructable as the page-anchored join of
    # these. Empty only for legacy paths that don't yet emit pages.
    pages: list[PageResult] = field(default_factory=list)
    # Degradations that occurred during extraction (e.g. a VLM page that fell back
    # to Tesseract). Carried out so the index ledger records them (R4).
    degradations: list[dict] = field(default_factory=list)

    def save_markdown(self, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.markdown, encoding="utf-8")

    def save_images(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        # Images are already saved during extraction; this records the dir
        return output_dir


def assemble_anchored(pages: list[PageResult]) -> str:
    """Join page Markdown with page anchors — the canonical document Markdown.

    Empty pages contribute nothing (no anchor), so anchors always precede real
    content and chunking can attribute every line to a physical page.
    """
    return "\n\n".join(
        f"{page_anchor(p.page_number)}\n{p.markdown}".strip()
        for p in pages if p.markdown.strip()
    )


class BaseExtractor(ABC):
    """Abstract base for document extraction approaches."""

    name: str = "base"

    @abstractmethod
    def extract(self, file_path: Path) -> ExtractionResult:
        """Extract markdown + images from a document."""
        ...

    def extract_images_pymupdf(self, pdf_path: Path, output_dir: Path) -> list[Path]:
        """Extract embedded images from PDF using PyMuPDF."""
        import fitz

        output_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        doc = fitz.open(pdf_path)
        seen_xrefs = set()

        for page_num in range(len(doc)):
            page = doc[page_num]
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                base_image = doc.extract_image(xref)
                if base_image["image"] is None:
                    continue

                ext = base_image.get("ext", "png")
                img_path = output_dir / f"page{page_num + 1}_img{xref}.{ext}"
                img_path.write_bytes(base_image["image"])
                saved.append(img_path)

        doc.close()
        return saved
