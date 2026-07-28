"""Per-page elapsed_seconds timing in the local cascade router.

Before this, ``PageResult`` had an ``elapsed_seconds`` field but router.py's
multi-page cascade never populated it — every page silently carried the
dataclass default of 0.0. These tests pin the fix: real per-page durations
must flow out of ``_render_image_scored`` (the OCR/VLM lane) and out of the
native "done" branch in ``extract()``.

``_render_image_scored`` is tested directly with fake native/VLM/tesseract
backends (bypassing ``__init__`` so no real Tesseract binary or vLLM server
is needed). The native branch is tested by generating a tiny real PDF with
PyMuPDF at test time — no checked-in fixture file required — and forcing
``Route.NATIVE`` so the classifier heuristics don't need to be reasoned about.
"""
from __future__ import annotations

import time

import fitz
import pytest

from parsevault.pipeline.extractors.page_classifier import Route
from parsevault.pipeline.extractors.router import CascadeConfig, LocalCascadeExtractor

_SLEEP = 0.02


class _FakeTesseract:
    def render_image_scored(self, img):
        time.sleep(_SLEEP)
        return "tesseract text", 0.87


class _FakeVLM:
    def render_image(self, img):
        time.sleep(_SLEEP)
        return "vlm text"


def _router(*, score_quality: bool = False) -> LocalCascadeExtractor:
    """A LocalCascadeExtractor with fake backends, built without __init__ so
    it needs neither a real Tesseract binary nor a reachable vLLM server."""
    router = LocalCascadeExtractor.__new__(LocalCascadeExtractor)
    router.config = CascadeConfig(score_quality=score_quality)
    router.tesseract = _FakeTesseract()
    router._quality = _FakeVLM()
    router._fast = router._quality
    return router


# --------------------------------------------------------------------- #
# _render_image_scored — the OCR/VLM timing engine
# --------------------------------------------------------------------- #

def test_vlm_lane_reports_real_elapsed():
    md, used, _signals, elapsed = _router()._render_image_scored(
        object(), Route.VLM_QUALITY, vlm_ready=True
    )
    assert (md, used) == ("vlm text", "vlm_quality")
    assert elapsed >= _SLEEP


def test_tesseract_lane_reports_real_elapsed():
    md, used, _signals, elapsed = _router()._render_image_scored(
        object(), Route.TESSERACT, vlm_ready=False
    )
    assert used == "tesseract"
    assert elapsed >= _SLEEP


def test_elapsed_covers_the_ocr_cross_check_when_score_quality_is_on():
    # score_quality=True runs an extra independent Tesseract pass on VLM pages
    # (the vlm_agreement cross-check) — elapsed must cover both calls, not
    # just the VLM call, since that's the true wall-clock cost of the lane.
    _md, _used, signals, elapsed = _router(score_quality=True)._render_image_scored(
        object(), Route.VLM_QUALITY, vlm_ready=True
    )
    assert "vlm_agreement" in signals
    assert elapsed >= _SLEEP * 2


# --------------------------------------------------------------------- #
# extract() — the native "done" branch, via a real (generated) PDF
# --------------------------------------------------------------------- #

@pytest.fixture
def two_page_pdf(tmp_path):
    doc = fitz.open()
    for i in range(2):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {i + 1} — hello world, this is real text.")
    path = tmp_path / "sample.pdf"
    doc.save(path)
    doc.close()
    return path


def test_native_pages_get_real_elapsed_not_the_dataclass_default(two_page_pdf):
    router = LocalCascadeExtractor(
        CascadeConfig(use_vlm=False, force_route=Route.NATIVE, score_quality=False)
    )
    result = router.extract(two_page_pdf)

    assert result.page_count == 2
    assert len(result.pages) == 2
    for page in result.pages:
        assert page.route == "native"
        # Regression check: previously this was always exactly 0.0 (the
        # PageResult dataclass default) because the native "done" branch in
        # extract()'s Pass 1 loop never timed the render call.
        assert page.elapsed_seconds > 0.0
