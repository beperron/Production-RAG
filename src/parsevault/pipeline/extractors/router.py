"""The balanced local cascade — the default document extractor.

Policy (local-only, balanced):

  born-digital + simple   → native text layer        (instant, no GPU)
  born-digital + complex  → fast VLM                  (layout fidelity)
  scanned / image-only    → quality VLM               (best transcription)
  any OCR page, no VLM    → Tesseract                 (GPU-free fallback)

Every lane degrades rather than fails: a per-page VLM error falls back to
Tesseract (for scans) or the native text layer (for born-digital), and an empty
native render escalates to OCR. The server is health-probed once up front so
routing avoids the VLM entirely when it is down — no per-page timeouts.

No document ever leaves the machine: the only network call is to the local vLLM
server (default ``localhost``).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

from .base import BaseExtractor, ExtractionResult, PageResult, assemble_anchored
from .models import GPUPlan, recommend_plan
from .native_text_extractor import NativeTextExtractor
from .page_classifier import Route, classify_page
from .raster import DEFAULT_DPI, rasterize_page
from .tesseract_extractor import TesseractExtractor
from .textnorm import strip_running_headers
from .vlm_extractor import DEFAULT_BASE_URL, VLMExtractor, VLMUnavailable

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_TEXT_SUFFIXES = {".txt", ".text", ".md", ".markdown"}

_CONTENT_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def _content_token_count(md: str) -> int:
    """Count of content tokens (lowercased alphanumerics ≥2 chars), ignoring
    Markdown punctuation — a converter-agnostic proxy for how much text a
    transcription captured. Used to keep the richer of native vs OCR per page."""
    return len(_CONTENT_TOKEN_RE.findall(md.lower()))


@dataclass
class CascadeConfig:
    """How the cascade is wired. Sensible defaults target the 2×A6000 plan."""

    use_vlm: bool = True
    base_url: str = DEFAULT_BASE_URL
    api_key: str = "EMPTY"
    quality_model: str = ""  # "" → pick from the GPU plan
    fast_model: str = ""
    dpi: int = DEFAULT_DPI
    tesseract_lang: str = "eng"
    # Override routing for the whole document (e.g. force the quality VLM on
    # every page). None → per-page heuristic routing.
    force_route: Route | None = None
    # Parallelism for the OCR/VLM lanes across pages. Tesseract (subprocess) and
    # the VLM (HTTP) both release the GIL, so threads give a real multi-core
    # speedup on long scanned docs. 0 → auto (min(8, cpu_count)); 1 → sequential.
    max_workers: int = 0
    # Document converter. "local-cascade" (default) is fully local / no-egress.
    # "llamaparse" routes documents to the LlamaParse cloud API — for PUBLIC
    # data only; never enable it on the private/no-egress path.
    parse_provider: str = "local-cascade"
    llamaparse_api_key: str = ""
    llamaparse_base_url: str = "https://api.cloud.llamaindex.ai/api/v1"
    # Explicit acknowledgement that documents may leave the machine for the
    # LlamaParse cloud API. Default OFF — cloud conversion is refused without it
    # (the hardened egress gate, R2.5). Public data only.
    allow_cloud: bool = False
    # Per-page quality scoring (R3): OCR confidence on Tesseract pages and a
    # VLM↔OCR agreement cross-check on VLM pages (one extra CPU OCR pass per VLM
    # page). On by default; set False to skip the cross-check on hot paths.
    score_quality: bool = True
    # Raster archival (R3.4): keep the rasterized page image so a disputed
    # transcription can be re-examined against exactly what the model saw.
    # mode ∈ none | flagged (default) | all. Only OCR/VLM pages are rasterized.
    raster_archive_dir: str = ""
    raster_archive_mode: str = "flagged"

    @classmethod
    def for_gpu(cls, total_vram_gb: float, num_gpus: int = 1, **kw) -> "CascadeConfig":
        plan: GPUPlan = recommend_plan(total_vram_gb, num_gpus)
        return cls(quality_model=plan.quality, fast_model=plan.fast, **kw)


class LocalCascadeExtractor(BaseExtractor):
    name = "local-cascade"

    def __init__(self, config: CascadeConfig | None = None):
        self.config = config or CascadeConfig.for_gpu(96, 2)  # default: 2×A6000
        self.native = NativeTextExtractor()
        self.tesseract = TesseractExtractor(
            lang=self.config.tesseract_lang, dpi=self.config.dpi
        )
        self._fast: VLMExtractor | None = None
        self._quality: VLMExtractor | None = None
        if self.config.use_vlm:
            qm = self.config.quality_model or recommend_plan(96, 2).quality
            fm = self.config.fast_model or recommend_plan(96, 2).fast
            common = dict(
                base_url=self.config.base_url,
                api_key=self.config.api_key,
                dpi=self.config.dpi,
            )
            self._quality = VLMExtractor(qm, **common)
            # Reuse one client if fast and quality are the same served model.
            self._fast = self._quality if fm == qm else VLMExtractor(fm, **common)

    # -- availability (probed once per extract) ------------------------------
    def _vlm_ready(self) -> bool:
        return bool(self._quality) and self._quality.is_available()

    # -- main entry -----------------------------------------------------------
    def extract(self, file_path: Path, pages: list[int] | None = None) -> ExtractionResult:
        file_path = Path(file_path)
        start = time.time()
        vlm_ready = self.config.use_vlm and self._vlm_ready()

        if file_path.suffix.lower() in _TEXT_SUFFIXES:
            # Plain text is already ground truth — no rasterization/OCR/cloud.
            from .text_extractor import PlainTextExtractor

            r = PlainTextExtractor().extract(file_path)
            from ..quality import score_pages

            pages = score_pages([PageResult(page_number=1, markdown=r.markdown, route="text",
                                             elapsed_seconds=time.time() - start)])
            return ExtractionResult(
                pdf_name=r.pdf_name,
                markdown=assemble_anchored(pages),
                page_count=r.page_count,
                elapsed_seconds=time.time() - start,
                extractor=self.name,
                page_routes=["text"],
                pages=pages,
            )

        if file_path.suffix.lower() == ".docx":
            # Word's XML text layer is ground truth — no rasterization/OCR.
            from .docx_extractor import DocxExtractor

            r = DocxExtractor().extract(file_path)
            from ..quality import score_pages

            pages = score_pages([PageResult(page_number=1, markdown=r.markdown, route="docx",
                                             elapsed_seconds=time.time() - start)])
            return ExtractionResult(
                pdf_name=r.pdf_name,
                markdown=assemble_anchored(pages),
                page_count=r.page_count,
                elapsed_seconds=time.time() - start,
                extractor=self.name,
                page_routes=["docx"],
                pages=pages,
            )

        if file_path.suffix.lower() in _IMAGE_SUFFIXES:
            from PIL import Image

            img = Image.open(file_path).convert("RGB")
            route = self.config.force_route or (
                Route.VLM_QUALITY if vlm_ready else Route.TESSERACT
            )
            md, used, sig, page_elapsed = self._render_image_scored(img, route, vlm_ready)
            from ..quality import score_pages

            pages = score_pages(
                [PageResult(page_number=1, markdown=md, route=used, elapsed_seconds=page_elapsed)],
                {1: sig},
            )
            return ExtractionResult(
                pdf_name=file_path.stem,
                markdown=assemble_anchored(pages),
                page_count=1,
                elapsed_seconds=time.time() - start,
                extractor=self.name,
                page_routes=[used],
                pages=pages,
            )

        import fitz

        doc = fitz.open(file_path)
        page_count = len(doc)
        wanted = set(pages) if pages is not None else None

        # Pass 1 (main thread — PyMuPDF is not thread-safe): classify, render the
        # native pages now (fast, GIL-bound anyway), and rasterize the pages that
        # need OCR/VLM. Each slot is either ("done", md, label) or a pending
        # ("ocr", image, route) we render in parallel below.
        # CAL-5: pages that classified as NATIVE but had an empty text layer
        # and got ESCALATED to OCR are recorded as a degradation — the
        # extraction didn't fail, but it took a different lane than the
        # classifier asked for, and the ledger must reflect that (R4).
        plans: list[tuple] = []
        page_numbers: list[int] = []  # 1-based physical page per plan slot
        native_escalations: list[int] = []  # 1-based page numbers escalated
        for page in doc:
            if wanted is not None and page.number not in wanted:
                continue
            route = self.config.force_route or classify_page(
                page, vlm_available=vlm_ready
            ).route
            if route == Route.NATIVE:
                page_t0 = time.time()
                md = self.native.render_page(page)
                if md.strip():
                    plans.append(("done", md, "native", time.time() - page_t0))
                    page_numbers.append(page.number + 1)
                    continue
                # Empty text layer despite a NATIVE route → escalate to OCR.
                native_escalations.append(page.number + 1)
                route = Route.VLM_QUALITY if vlm_ready else Route.TESSERACT
            # OCR-bound: also capture the native text layer so we can keep whichever
            # is richer. Some pages have a partial layer + text baked into images or
            # vector paths (OCR recovers more); others have a complete layer that OCR
            # would only degrade. Comparing per page makes routing robust either way.
            native_md = self.native.render_page(page)
            plans.append(("ocr", rasterize_page(page, self.config.dpi), route, native_md))
            page_numbers.append(page.number + 1)
        doc.close()

        # Pass 2: render the OCR/VLM pages, in parallel when there is more than
        # one (threads — tesseract/VLM release the GIL). Page order preserved.
        ocr_idx = [i for i, p in enumerate(plans) if p[0] == "ocr"]
        rendered: dict[int, tuple[str, str, dict, float]] = {}
        workers = self._workers()
        if len(ocr_idx) > 1 and workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            # Tesseract is pinned to one OpenMP thread once at construction
            # (TesseractExtractor.__init__) so our page-level parallelism is the
            # only parallelism — no per-extract global env mutation to race on
            # when two extractions run concurrently in one process (R10.2).
            with ThreadPoolExecutor(max_workers=min(workers, len(ocr_idx))) as ex:
                futs = {
                    ex.submit(self._render_image_scored, plans[i][1], plans[i][2], vlm_ready): i
                    for i in ocr_idx
                }
                for fut, i in futs.items():
                    rendered[i] = fut.result()
        else:
            for i in ocr_idx:
                rendered[i] = self._render_image_scored(plans[i][1], plans[i][2], vlm_ready)

        page_md: list[str] = []
        routes: list[str] = []
        signals: list[dict] = []
        page_elapsed: list[float] = []
        for i, p in enumerate(plans):
            if p[0] == "done":
                md, used, sig, elapsed = p[1], p[2], {}, p[3]
            else:
                ocr_md, used, sig, elapsed = rendered[i]
                native_md = p[3]
                # Keep whichever transcription carries more content (R: conv-eval).
                # Guards against OCR'ing a page whose text layer was already complete
                # while still capturing pages whose text lives in images/vectors.
                if _content_token_count(native_md) > _content_token_count(ocr_md):
                    md, used, sig = native_md, "native", {}
                    # elapsed stays the OCR-lane time — that's the work actually
                    # done for this page, even though native won the content check.
                else:
                    md = ocr_md
            page_md.append(md)
            routes.append(used)
            signals.append(sig)
            page_elapsed.append(elapsed)

        # Remove headers/footers/page numbers repeated across pages (they add no
        # information and pollute retrieval), then drop now-empty pages. Page
        # numbers/routes stay aligned to the (post-strip) page Markdown so the
        # page anchors below point at the right physical page.
        page_md = strip_running_headers(page_md)
        sig_by_page = {n: s for n, s in zip(page_numbers, signals)}
        # A page routed to a VLM lane but transcribed by Tesseract fell back —
        # record it so the index ledger reflects the degradation (R4).
        degradations = [
            {"kind": "vlm_fallback_tesseract",
             "detail": f"page {n}: VLM unavailable, used Tesseract", "page": n}
            for n, plan, used in zip(page_numbers, plans, routes)
            if plan[0] == "ocr" and plan[2] in (Route.VLM_FAST, Route.VLM_QUALITY)
            and used == "tesseract"
        ]
        # CAL-5: native-OCR escalation — the classifier asked for the native
        # text layer but it came back empty. The page IS still extracted (via
        # the OCR/VLM lane), but the routing decision differed from the
        # initial plan; surface it in the ledger so "no silent degradation"
        # is enforced for this fallback path too (R4). One entry per page.
        degradations.extend(
            {"kind": "native_fallback_ocr",
             "detail": f"page {n}: empty text layer, escalated to OCR/VLM",
             "page": n}
            for n in native_escalations
        )
        page_objs = [
            PageResult(page_number=n, markdown=md, route=route, elapsed_seconds=elapsed)
            for n, md, route, elapsed in zip(page_numbers, page_md, routes, page_elapsed)
            if md.strip()
        ]
        from ..quality import score_pages

        score_pages(page_objs, sig_by_page)
        # Archive the raster for flagged (or, configurably, all) OCR/VLM pages so a
        # disputed transcription is re-examinable against the exact image (R3.4).
        if self.config.raster_archive_dir and self.config.raster_archive_mode != "none":
            raster_by_page = {
                page_numbers[i]: plans[i][1] for i in range(len(plans))
                if plans[i][0] == "ocr"
            }
            self._archive_rasters(file_path.stem, page_objs, raster_by_page)

        return ExtractionResult(
            pdf_name=file_path.stem,
            markdown=assemble_anchored(page_objs),
            page_count=page_count,
            elapsed_seconds=time.time() - start,
            extractor=self.name,
            page_routes=routes,
            pages=page_objs,
            degradations=degradations,
        )

    def _workers(self) -> int:
        import os

        w = self.config.max_workers
        return w if w and w > 0 else min(8, os.cpu_count() or 4)

    def _archive_rasters(self, stem: str, page_objs, raster_by_page: dict) -> None:
        from pathlib import Path

        out = Path(self.config.raster_archive_dir)
        keep_all = self.config.raster_archive_mode == "all"
        try:
            out.mkdir(parents=True, exist_ok=True)
            for p in page_objs:
                if (keep_all or p.quality.get("flagged")) and p.page_number in raster_by_page:
                    dest = out / f"{stem}-p{p.page_number}.png"
                    raster_by_page[p.page_number].save(dest)
                    p.raster_path = str(dest)
        except Exception:  # noqa: BLE001 — archival is best-effort, never fail extraction
            pass

    # -- per-image dispatch with fallback ------------------------------------
    def _render_image(self, img, route: Route, vlm_ready: bool) -> tuple[str, str]:
        md, used, _signals, _elapsed = self._render_image_scored(img, route, vlm_ready)
        return md, used

    def _render_image_scored(
        self, img, route: Route, vlm_ready: bool
    ) -> tuple[str, str, dict, float]:
        """Render an image, returning (markdown, lane, quality_signals, elapsed_seconds).

        Signals carry the per-page measurements (R3): Tesseract pages → mean OCR
        confidence; VLM pages → a VLM↔OCR token-overlap agreement from an
        independent Tesseract pass on the same raster (one extra CPU pass).

        ``elapsed_seconds`` covers the whole call — including a failed VLM attempt
        that falls back to Tesseract and the quality cross-check pass — since that
        is the real wall-clock cost of rendering this page via this lane."""
        from ..quality import jaccard_tokens

        page_t0 = time.time()
        if route in (Route.VLM_FAST, Route.VLM_QUALITY) and vlm_ready:
            client = self._fast if route == Route.VLM_FAST else self._quality
            try:
                md = client.render_image(img)
                signals: dict = {}
                if self.config.score_quality:
                    try:  # independent OCR cross-check — never fail extraction on it
                        ocr_md, ocr_conf = self.tesseract.render_image_scored(img)
                        signals = {"vlm_agreement": jaccard_tokens(md, ocr_md),
                                   "ocr_mean_confidence": ocr_conf}
                    except Exception:  # noqa: BLE001
                        signals = {}
                return md, route.value, signals, time.time() - page_t0
            except VLMUnavailable:
                pass
        md, conf = self.tesseract.render_image_scored(img)
        return md, "tesseract", {"ocr_mean_confidence": conf}, time.time() - page_t0

    # Convenience for callers that already hold a fitz page (e.g. benchmark).
    def rasterize(self, page):
        return rasterize_page(page, self.config.dpi)
