"""Tesseract OCR lane — the GPU-free, always-available fallback.

When no VLM server is reachable (model still loading, OOM, machine without the
GPU box), scanned pages still need *some* transcription. Tesseract is the
dependable floor: CPU-only, offline. We squeeze real accuracy out of it with
light image preprocessing (grayscale + autocontrast + upscale of small scans),
layout-aware reconstruction from ``image_to_data`` (paragraphs from Tesseract's
own block/paragraph segmentation, in a single pass that also yields confidence),
and a binarization retry when confidence is low.

Works on PDFs (rasterized per page) and on standalone image files.
"""

from __future__ import annotations

import time
from pathlib import Path

from .base import BaseExtractor, ExtractionResult
from .raster import DEFAULT_DPI, rasterize_page
from .textnorm import clean_text

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
# Below this width, upscaling materially improves recognition of small print.
_MIN_OCR_WIDTH = 1600
# Above this on the longest side, Tesseract's layout analysis (psm 3) can spend
# unbounded time on photographs/complex images for little gain. Cap to bound it.
_MAX_OCR_DIM = 4000


class TesseractExtractor(BaseExtractor):
    name = "tesseract"

    def __init__(
        self,
        lang: str = "eng",
        dpi: int = DEFAULT_DPI,
        *,
        psm: int = 3,
        preprocess: bool = True,
        retry_conf_threshold: float = 65.0,
        timeout: float = 30.0,
    ):
        import os

        # Pin Tesseract's OpenMP to one thread, once, process-wide (R10.2): the
        # cascade parallelizes at the page level, so per-call OpenMP threads only
        # oversubscribe. Set here (not toggled per extract) so concurrent
        # extractions in one process never race on the global.
        os.environ.setdefault("OMP_THREAD_LIMIT", "1")
        self.lang = lang
        self.dpi = dpi
        self.psm = psm
        self.preprocess = preprocess
        self.retry_conf_threshold = retry_conf_threshold
        # Per-page OCR wall-clock cap. A photograph or pathological scan can make
        # Tesseract spin for minutes; without this the whole pipeline stalls on
        # one image. On timeout the page degrades to empty rather than hanging.
        self.timeout = timeout

    def extract(self, file_path: Path, pages: list[int] | None = None) -> ExtractionResult:
        start = time.time()
        suffix = file_path.suffix.lower()

        if suffix in _IMAGE_SUFFIXES:
            from PIL import Image

            md = self.render_image(Image.open(file_path).convert("RGB"))
            return ExtractionResult(
                pdf_name=file_path.stem,
                markdown=md,
                page_count=1,
                elapsed_seconds=time.time() - start,
                extractor=self.name,
            )

        import fitz

        doc = fitz.open(file_path)
        page_count = len(doc)
        wanted = set(pages) if pages is not None else None
        parts: list[str] = []
        for page in doc:
            if wanted is not None and page.number not in wanted:
                continue
            parts.append(self.render_page(page))
        doc.close()
        return ExtractionResult(
            pdf_name=file_path.stem,
            markdown="\n\n".join(p for p in parts if p.strip()),
            page_count=page_count,
            elapsed_seconds=time.time() - start,
            extractor=self.name,
        )

    def render_page(self, page) -> str:
        return self.render_image(rasterize_page(page, self.dpi))

    def render_image(self, image) -> str:
        """OCR a PIL image to lightly-structured Markdown."""
        return self.render_image_scored(image)[0]

    def render_image_scored(self, image) -> tuple[str, float]:
        """OCR a PIL image to Markdown + mean per-word confidence (0–100).

        Reconstructs paragraphs from Tesseract's block/paragraph segmentation so
        prose reflows correctly. Headings/tables are intentionally NOT invented —
        that structure is unreliable from flat OCR and is the VLM lane's job. The
        confidence is the per-page quality signal for OCR'd pages (R3).
        """
        prepped = self._preprocess(image) if self.preprocess else image
        text, conf = self._ocr(prepped)
        if self.preprocess and conf < self.retry_conf_threshold:
            # Hard scan: a binarized variant often recovers low-contrast print.
            alt_text, alt_conf = self._ocr(self._binarize(image))
            if alt_conf > conf:
                text, conf = alt_text, alt_conf
        return clean_text(text), conf

    def mean_confidence(self, image) -> float:
        """Mean per-word OCR confidence (0–100); useful to flag bad scans."""
        prepped = self._preprocess(image) if self.preprocess else image
        return self._ocr(prepped)[1]

    # -- internals ------------------------------------------------------------
    def _ocr(self, image) -> tuple[str, float]:
        import pytesseract

        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.lang,
                config=f"--psm {self.psm}",
                output_type=pytesseract.Output.DICT,
                timeout=self.timeout,
            )
        except (RuntimeError, pytesseract.TesseractError):
            # Timeout (RuntimeError "Tesseract process timeout") or a hard
            # Tesseract failure: degrade this page to empty so the cascade can
            # move on (and route the doc through native/VLM) instead of stalling.
            return "", 0.0
        return self._reconstruct(data)

    @staticmethod
    def _reconstruct(data: dict) -> tuple[str, float]:
        """Group words into lines and paragraphs; return (markdown, mean conf)."""
        n = len(data.get("text", []))
        paragraphs: list[list[str]] = []  # each paragraph = list of line strings
        cur_line: list[str] = []
        confs: list[float] = []
        key = None  # (block, par)
        line_key = None  # (block, par, line)

        def flush_line():
            nonlocal cur_line
            if cur_line and paragraphs:
                paragraphs[-1].append(" ".join(cur_line))
            cur_line = []

        for i in range(n):
            word = (data["text"][i] or "").strip()
            try:
                c = float(data["conf"][i])
            except (ValueError, TypeError):
                c = -1.0
            pk = (data["block_num"][i], data["par_num"][i])
            lk = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            if pk != key:
                flush_line()
                paragraphs.append([])
                key = pk
                line_key = None
            if lk != line_key:
                flush_line()
                line_key = lk
            if word:
                cur_line.append(word)
                if c >= 0:
                    confs.append(c)
        flush_line()

        blocks = []
        for para in paragraphs:
            joined = " ".join(ln for ln in para if ln.strip()).strip()
            if joined:
                blocks.append(joined)
        text = "\n\n".join(blocks).strip()
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        return text, mean_conf

    # -- preprocessing --------------------------------------------------------
    @staticmethod
    def _fit_for_ocr(img):
        """Bound the working size: upscale small scans, downscale huge images.

        Small print benefits from upscaling to _MIN_OCR_WIDTH; conversely an
        oversized photo (longest side > _MAX_OCR_DIM) is downscaled so layout
        analysis stays bounded. Aspect ratio is preserved.
        """
        from PIL import Image

        w, h = img.width, img.height
        scale = 1.0
        if w < _MIN_OCR_WIDTH:
            scale = _MIN_OCR_WIDTH / w
        longest = max(w * scale, h * scale)
        if longest > _MAX_OCR_DIM:
            scale *= _MAX_OCR_DIM / longest
        if scale != 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        return img

    def _preprocess(self, image):
        """Grayscale + autocontrast, with the working size bounded for OCR."""
        from PIL import ImageOps

        img = ImageOps.grayscale(image)
        img = ImageOps.autocontrast(img)
        return self._fit_for_ocr(img)

    def _binarize(self, image):
        """Aggressive black/white threshold for low-contrast or noisy scans."""
        from PIL import ImageOps

        img = ImageOps.grayscale(image)
        img = ImageOps.autocontrast(img)
        img = self._fit_for_ocr(img)
        return img.point(lambda p: 255 if p > 160 else 0, mode="1").convert("L")
