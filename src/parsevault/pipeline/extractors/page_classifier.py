"""Per-page routing heuristics for the balanced local cascade.

The "balanced" policy means: do not pay for a VLM on a page a cheap text layer
already renders perfectly, and do not trust a text layer on a page that is
actually a scan or a complex table/figure where layout matters. This module
turns a PDF page into a small feature vector and a route label; the router
(``router.py``) maps the label onto an extractor, with fallbacks.

All features come from PyMuPDF (``fitz``) so classification is fast and local.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .textnorm import looks_garbled


class Route(str, Enum):
    NATIVE = "native"  # trust the embedded text layer (born-digital, simple)
    VLM_FAST = "vlm_fast"  # born-digital but structured (tables/math/columns)
    VLM_QUALITY = "vlm_quality"  # scanned or dense/complex — best model
    TESSERACT = "tesseract"  # OCR without a VLM (fallback lane)


@dataclass
class PageFeatures:
    index: int  # 0-based page number
    text_chars: int  # characters in the embedded text layer
    text_coverage: float  # fraction of page area inside text blocks (0..1)
    image_area_ratio: float  # fraction of page area covered by raster images
    drawing_count: int  # vector paths — a proxy for tables/figures/rules
    column_count: int  # estimated text columns (≥2 → multi-column layout)
    garbled: bool = False  # text layer present but broken (mojibake/no spaces)
    route: Route = Route.NATIVE

    @property
    def is_scanned(self) -> bool:
        """Image-dominant page with negligible real text → a scan."""
        return self.text_chars < 20 and self.image_area_ratio > 0.5

    @property
    def text_in_visuals(self) -> bool:
        """The real content is rendered as a raster image or vector paths that the
        embedded text layer does NOT capture — so the page needs OCR even though it
        has *some* text. Detected as visuals dominating while the text layer covers
        very little of the page. Validated against LlamaParse gold: these are the
        pages where the native lane loses the most content (recall 0.3–0.6 → ~1.0
        with OCR). See scripts/eval_conversion.py."""
        image_dominant = self.image_area_ratio > 0.35 and self.text_coverage < 0.20
        vector_dominant = self.drawing_count >= 90 and self.text_coverage < 0.60
        return image_dominant or vector_dominant

    @property
    def is_complex(self) -> bool:
        """Born-digital but with structure a flat text dump would mangle."""
        return self.column_count >= 2 or self.drawing_count >= 40


def _estimate_columns(text_block_boxes, page_width: float) -> int:
    """Estimate column count from the horizontal centers of text blocks.

    A genuinely two-column page has text-block centers clustered into a left
    band and a right band with a gap between. We bucket block centers into left
    / right halves and call it multi-column only when both halves carry
    substantial text *and* few blocks straddle the centerline (which would
    indicate full-width text instead).
    """
    if page_width <= 0 or len(text_block_boxes) < 4:
        return 1
    mid = page_width / 2.0
    left = right = straddle = 0
    for x0, _y0, x1, _y1 in text_block_boxes:
        center = (x0 + x1) / 2.0
        width = x1 - x0
        if width > page_width * 0.6:
            straddle += 1
        elif center < mid:
            left += 1
        else:
            right += 1
    if straddle > max(left, right):
        return 1
    return 2 if (left >= 2 and right >= 2) else 1


def classify_page(page, *, vlm_available: bool) -> PageFeatures:
    """Compute features for a single ``fitz.Page`` and assign a route.

    ``vlm_available`` lets the same heuristics degrade gracefully: when no VLM
    server is reachable, VLM routes collapse onto the Tesseract lane for scans
    and onto the native text layer for merely-structured pages.
    """
    rect = page.rect
    page_area = max(rect.width * rect.height, 1.0)

    text = page.get_text("text") or ""
    text_chars = len(text.strip())

    # Text-block boxes drive both coverage and column estimation.
    blocks = page.get_text("blocks") or []
    text_boxes = [(b[0], b[1], b[2], b[3]) for b in blocks if len(b) >= 5 and str(b[4]).strip()]
    text_area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in text_boxes)
    text_coverage = min(text_area / page_area, 1.0)

    image_area = 0.0
    try:
        for img in page.get_image_info():
            bbox = img.get("bbox")
            if bbox:
                image_area += abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    except Exception:
        # get_image_info is best-effort; absence just means "assume no images".
        pass
    image_area_ratio = min(image_area / page_area, 1.0)

    try:
        drawing_count = len(page.get_drawings())
    except Exception:
        drawing_count = 0

    column_count = _estimate_columns(text_boxes, rect.width)

    feats = PageFeatures(
        index=page.number,
        text_chars=text_chars,
        text_coverage=text_coverage,
        image_area_ratio=image_area_ratio,
        drawing_count=drawing_count,
        column_count=column_count,
        garbled=looks_garbled(text),
    )
    feats.route = _route_for(feats, vlm_available=vlm_available)
    return feats


def _route_for(f: PageFeatures, *, vlm_available: bool) -> Route:
    if f.is_scanned or f.garbled or f.text_in_visuals:
        # Scans, broken text layers, and pages whose text lives in images/vector
        # paths all need OCR. Prefer the VLM; else Tesseract.
        return Route.VLM_QUALITY if vlm_available else Route.TESSERACT

    if f.text_chars < 20:
        # No usable text and not clearly a big image (e.g. vector-only scan,
        # or an empty/near-empty page). Send to OCR to be safe.
        return Route.VLM_QUALITY if vlm_available else Route.TESSERACT

    if f.is_complex:
        # Real text exists but layout is structured; a VLM preserves it best.
        # Without a VLM we still trust the text layer (Tesseract rarely beats a
        # real text layer on born-digital pages, even multi-column ones).
        return Route.VLM_FAST if vlm_available else Route.NATIVE

    # Rich, simple, born-digital page: the embedded text layer is ground truth.
    return Route.NATIVE
