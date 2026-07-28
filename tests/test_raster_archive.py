"""Page-raster archival in the local cascade router.

CascadeConfig.raster_archive_dir/raster_archive_mode already existed for the
OCR-dispute-review use case (see docs on _archive_rasters), but nothing
previously recorded *where* a page's raster landed on PageResult itself —
the dashboard's hover-preview feature needs that path to know which pages
have an image to show. These tests pin PageResult.raster_path: set only for
OCR/VLM-lane pages that were actually archived, and pointing at a real file.

Uses a real (generated) PDF and the real Tesseract binary — no mocking of
the OCR lane — so this also confirms the raster save actually happens as a
side effect of a real extraction, not just of the archival bookkeeping.
"""
from __future__ import annotations

import fitz
import pytest

from parsevault.pipeline.extractors.page_classifier import Route
from parsevault.pipeline.extractors.router import CascadeConfig, LocalCascadeExtractor


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


def test_ocr_lane_pages_get_raster_path_and_a_real_file(tmp_path, two_page_pdf):
    archive_dir = tmp_path / "rasters"
    router = LocalCascadeExtractor(CascadeConfig(
        use_vlm=False, force_route=Route.TESSERACT, score_quality=False,
        raster_archive_dir=str(archive_dir), raster_archive_mode="all",
    ))
    result = router.extract(two_page_pdf)

    assert len(result.pages) == 2
    for page in result.pages:
        assert page.raster_path, "OCR-lane page should have a raster archived"
        raster = archive_dir / f"sample-p{page.page_number}.png"
        assert page.raster_path == str(raster)
        assert raster.is_file()
        assert raster.stat().st_size > 0


def test_native_lane_pages_get_no_raster_path(tmp_path, two_page_pdf):
    # Native pages never call rasterize_page() at all — even with archival on,
    # there's nothing to save, so raster_path must stay empty (the dashboard's
    # signal that a page has no hover-preview image).
    archive_dir = tmp_path / "rasters"
    router = LocalCascadeExtractor(CascadeConfig(
        use_vlm=False, force_route=Route.NATIVE, score_quality=False,
        raster_archive_dir=str(archive_dir), raster_archive_mode="all",
    ))
    result = router.extract(two_page_pdf)

    assert len(result.pages) == 2
    for page in result.pages:
        assert page.raster_path == ""
    # _archive_rasters always mkdirs the configured dir once it's called (a
    # pre-existing, harmless quirk); what matters here is no PNG was written.
    assert list(archive_dir.glob("*.png")) == []


def test_raster_archival_off_by_default_leaves_raster_path_empty(tmp_path, two_page_pdf):
    router = LocalCascadeExtractor(CascadeConfig(
        use_vlm=False, force_route=Route.TESSERACT, score_quality=False,
    ))
    result = router.extract(two_page_pdf)

    for page in result.pages:
        assert page.raster_path == ""
