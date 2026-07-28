"""Browser-level checks for the build dashboard's throbbing run-id pills.

Renders the real page with a headless Chromium (via Playwright) against the
throwaway test DB, so these exercise the actual DOM/CSS/JS the browser sees —
not just the HTML/JSON the server emits. Skips cleanly if Playwright or its
Chromium binary isn't installed (see conftest.py's chromium_browser fixture).
"""
from __future__ import annotations

import urllib.parse

from parsevault.pipeline.build_events import BuildEventLogger

import build_dashboard_server as dash


def _write_build(dsn, build_id, *, done=False):
    logger = BuildEventLogger(build_id, dsn=dsn)
    logger.event("build_start", total_todo=3, provider="local")
    if done:
        logger.event("build_done", docs=3, chunks=9, elapsed_s=1.2, report_path="r.json")
    logger.close()


def test_multiple_running_builds_render_multiple_pills(pg_test_dsn, dashboard_server, chromium_browser):
    _write_build(pg_test_dsn, "ui-running-a")
    _write_build(pg_test_dsn, "ui-running-b")
    _write_build(pg_test_dsn, "ui-done-c", done=True)

    page = chromium_browser.new_page()
    try:
        page.goto(dashboard_server)
        page.wait_for_selector("#run-pills .run-pill")
        pill_ids = page.eval_on_selector_all(
            "#run-pills .run-pill", "els => els.map(e => e.dataset.id)"
        )
        # The test DB is shared across the session, so other tests' running
        # builds may also show pills — assert on presence, not the full set.
        assert {"ui-running-a", "ui-running-b"} <= set(pill_ids)

        # The done build never gets a pill — only running ones do.
        assert "ui-done-c" not in pill_ids

        # Each pill pulses via a CSS animation on its .dot, not just a static color.
        animation_name = page.eval_on_selector(
            "#run-pills .run-pill .dot", "el => getComputedStyle(el).animationName"
        )
        assert animation_name != "none"
    finally:
        page.close()


def test_clicking_a_pill_switches_the_build_picker(pg_test_dsn, dashboard_server, chromium_browser):
    _write_build(pg_test_dsn, "ui-click-target")
    _write_build(pg_test_dsn, "ui-click-other")

    page = chromium_browser.new_page()
    try:
        page.goto(dashboard_server)
        page.wait_for_selector("#run-pills .run-pill")
        page.click('#run-pills .run-pill[data-id="ui-click-target"]')
        page.wait_for_function(
            "document.getElementById('build-picker').value === 'ui-click-target'"
        )
    finally:
        page.close()


def _write_build_with_pages(dsn, build_id, pages):
    logger = BuildEventLogger(build_id, dsn=dsn)
    logger.event("build_start", total_todo=1, provider="local")
    # doc_start must precede extract_done — buildCards() only pushes a card
    # into the render list from doc_start; without it, extract_done still
    # aggregates routeTime but the doc never gets a <details> card in the DOM.
    logger.event("doc_start", index=1, total=1, path="d1.pdf")
    logger.event(
        "extract_done", doc_id="d1", path="d1.pdf", extractor="local-cascade",
        page_count=len(pages), pages=pages, degradations=[],
    )
    logger.close()


def test_time_chart_renders_a_labeled_bar_per_route(pg_test_dsn, dashboard_server, chromium_browser):
    _write_build_with_pages(pg_test_dsn, "ui-timechart", pages=[
        {"page_number": 1, "route": "native", "chars": 500, "elapsed_seconds": 1.0,
         "flagged": False, "flag_reasons": [], "preview": ""},
        {"page_number": 2, "route": "vlm", "chars": 900, "elapsed_seconds": 4.0,
         "flagged": False, "flag_reasons": [], "preview": ""},
    ])

    page = chromium_browser.new_page()
    try:
        page.goto(f"{dashboard_server}/?build=ui-timechart")
        page.wait_for_selector("#timechart .timechart-row")
        rows = page.eval_on_selector_all(
            "#timechart .timechart-row",
            "els => els.map(e => ({"
            "label: e.querySelector('.timechart-label').textContent,"
            "val: e.querySelector('.timechart-val').textContent,"
            "width: e.querySelector('.timechart-fill').style.width"
            "}))",
        )
        by_label = {r["label"]: r for r in rows}
        assert "native text layer" in by_label
        assert "vision-language model" in by_label
        # vlm (4.0s) is the max, so it fills 100%; native (1.0s) is a quarter of that.
        assert by_label["vision-language model"]["width"] == "100%"
        assert by_label["native text layer"]["width"] == "25%"
        assert "4.0s" in by_label["vision-language model"]["val"]
        assert "1.0s" in by_label["native text layer"]["val"]
    finally:
        page.close()


def test_time_chart_shows_empty_state_before_any_pages(pg_test_dsn, dashboard_server, chromium_browser):
    _write_build(pg_test_dsn, "ui-timechart-empty")

    page = chromium_browser.new_page()
    try:
        page.goto(f"{dashboard_server}/?build=ui-timechart-empty")
        page.wait_for_selector("#timechart .timechart-empty")
    finally:
        page.close()


def _write_build_with_chunks(dsn, build_id, chunks, *, source_sha256="ab" * 32):
    logger = BuildEventLogger(build_id, dsn=dsn)
    logger.event("build_start", total_todo=1, provider="local")
    logger.event(
        "doc_start", index=1, total=1, path="d1.pdf",
    )
    logger.event(
        "extract_done", doc_id="d1", path="d1.pdf", source_sha256=source_sha256,
        extractor="local-cascade", page_count=1, pages=[], degradations=[],
    )
    logger.event("chunk_done", doc_id="d1", chunk_count=len(chunks), chunks=chunks)
    logger.close()


def _chunk(i, **overrides):
    base = {
        "chunk_id": f"d1:{i}", "heading_path": [f"Section {i}"],
        "char_len": 100 + i, "token_estimate": 25 + i,
        "page_start": 1, "page_end": 1, "extraction_routes": ["native"],
        "preview": f"chunk {i} text",
    }
    base.update(overrides)
    return base


def test_chunk_table_shows_size_and_source_hash(pg_test_dsn, dashboard_server, chromium_browser):
    full_hash = "0123456789abcdef" * 4  # 64 hex chars
    _write_build_with_chunks(
        pg_test_dsn, "ui-chunk-size", [_chunk(1, char_len=1500, token_estimate=375)],
        source_sha256=full_hash,
    )

    page = chromium_browser.new_page()
    try:
        page.goto(f"{dashboard_server}/?build=ui-chunk-size")
        page.click("details summary")
        page.wait_for_selector("table.chunktbl td.size")
        assert page.inner_text("table.chunktbl td.size") == "1500 ch / ~375 tok"

        chip = page.locator(".hash-chip")
        assert chip.get_attribute("title") == f"sha256:{full_hash}"
        assert full_hash[:12] in chip.inner_text()
    finally:
        page.close()


def test_chunk_table_caps_rows_and_expands_on_click(pg_test_dsn, dashboard_server, chromium_browser):
    chunks = [_chunk(i) for i in range(45)]
    _write_build_with_chunks(pg_test_dsn, "ui-chunk-cap", chunks)

    page = chromium_browser.new_page()
    try:
        page.goto(f"{dashboard_server}/?build=ui-chunk-cap")
        page.click("details summary")
        page.wait_for_selector("table.chunktbl tbody tr")
        # 40 shown + the "show more" row itself.
        assert page.locator("table.chunktbl tbody tr").count() == 41
        assert "Show 5 more chunks" in page.inner_text(".chunk-more-btn")

        page.click(".chunk-more-btn")
        page.wait_for_function(
            "document.querySelectorAll('table.chunktbl tbody tr').length === 46"
        )
        assert "Show fewer chunks" in page.inner_text(".chunk-more-btn")

        page.click(".chunk-more-btn")
        page.wait_for_function(
            "document.querySelectorAll('table.chunktbl tbody tr').length === 41"
        )
    finally:
        page.close()


# --------------------------------------------------------------------- #
# Page-badge hover preview + click-through
# --------------------------------------------------------------------- #

def test_page_badge_with_raster_shows_hover_preview_image(
    pg_test_dsn, dashboard_server, chromium_browser, tmp_path, monkeypatch
):
    monkeypatch.setattr(dash, "_SERVE_ROOTS", [tmp_path.resolve()])
    raster = tmp_path / "d1-p2.png"
    raster.write_bytes(b"\x89PNG\r\n\x1a\nfake-png-bytes")

    _write_build_with_pages(pg_test_dsn, "ui-hover-preview", pages=[
        {"page_number": 1, "route": "native", "chars": 500, "elapsed_seconds": 0.1,
         "flagged": False, "flag_reasons": [], "preview": "", "raster_path": None},
        {"page_number": 2, "route": "tesseract", "chars": 900, "elapsed_seconds": 2.0,
         "flagged": True, "flag_reasons": ["low confidence"], "preview": "",
         "raster_path": str(raster)},
    ])

    page = chromium_browser.new_page()
    try:
        page.goto(f"{dashboard_server}/?build=ui-hover-preview")
        page.click("details summary")
        page.wait_for_selector(".pg.has-preview")
        assert page.locator(".pg.has-preview").count() == 1

        page.hover(".pg.has-preview")
        page.wait_for_selector("#page-preview-popup img")
        page.wait_for_function(
            "getComputedStyle(document.getElementById('page-preview-popup')).display !== 'none'"
        )
        src = page.get_attribute("#page-preview-popup img", "src")
        assert urllib.parse.quote(str(raster), safe="") in src
    finally:
        page.close()


def test_page_badge_without_raster_has_no_preview_affordance(
    pg_test_dsn, dashboard_server, chromium_browser
):
    _write_build_with_pages(pg_test_dsn, "ui-no-preview", pages=[
        {"page_number": 1, "route": "native", "chars": 500, "elapsed_seconds": 0.1,
         "flagged": False, "flag_reasons": [], "preview": "", "raster_path": None},
    ])

    page = chromium_browser.new_page()
    try:
        page.goto(f"{dashboard_server}/?build=ui-no-preview")
        page.click("details summary")
        page.wait_for_selector(".pg")
        assert page.locator(".pg.has-preview").count() == 0

        page.hover(".pg")
        popup_display = page.eval_on_selector(
            "#page-preview-popup", "el => getComputedStyle(el).display"
        ) if page.locator("#page-preview-popup").count() else "none"
        assert popup_display == "none"
    finally:
        page.close()


def test_clicking_page_badge_opens_source_pdf_in_new_tab(
    pg_test_dsn, dashboard_server, chromium_browser
):
    _write_build_with_pages(pg_test_dsn, "ui-click-through", pages=[
        {"page_number": 3, "route": "native", "chars": 500, "elapsed_seconds": 0.1,
         "flagged": False, "flag_reasons": [], "preview": "", "raster_path": None},
    ])

    page = chromium_browser.new_page()
    try:
        page.goto(f"{dashboard_server}/?build=ui-click-through")
        page.click("details summary")
        page.wait_for_selector(".pg")
        with page.expect_popup() as popup_info:
            page.click(".pg")
        popup = popup_info.value
        assert popup.url == f"{dashboard_server}/api/source?path=d1.pdf#page=3"
        popup.close()
    finally:
        page.close()


# --------------------------------------------------------------------- #
# "Low confidence only" filter
# --------------------------------------------------------------------- #

def _write_build_with_two_docs(dsn, build_id, clean_pages, flagged_pages):
    logger = BuildEventLogger(build_id, dsn=dsn)
    logger.event("build_start", total_todo=2, provider="local")
    logger.event("doc_start", index=1, total=2, path="clean.pdf")
    logger.event(
        "extract_done", doc_id="clean", path="clean.pdf", extractor="local-cascade",
        page_count=len(clean_pages), pages=clean_pages, degradations=[],
    )
    logger.event("doc_start", index=2, total=2, path="flagged.pdf")
    logger.event(
        "extract_done", doc_id="flagged", path="flagged.pdf", extractor="local-cascade",
        page_count=len(flagged_pages), pages=flagged_pages, degradations=[],
    )
    logger.close()


def test_low_confidence_filter_takes_clean_documents_off_the_board(
    pg_test_dsn, dashboard_server, chromium_browser
):
    clean_pages = [
        {"page_number": 1, "route": "tesseract", "chars": 500, "elapsed_seconds": 1.0,
         "flagged": False, "flag_reasons": [], "garbled": False,
         "ocr_mean_confidence": 92.0, "vlm_agreement": None, "preview": ""},
    ]
    flagged_pages = [
        {"page_number": 1, "route": "tesseract", "chars": 500, "elapsed_seconds": 1.0,
         "flagged": True, "flag_reasons": ["low_ocr_confidence(40<65)"], "garbled": False,
         "ocr_mean_confidence": 40.0, "vlm_agreement": None, "preview": ""},
    ]
    _write_build_with_two_docs(pg_test_dsn, "ui-conf-filter", clean_pages, flagged_pages)

    page = chromium_browser.new_page()
    try:
        page.goto(f"{dashboard_server}/?build=ui-conf-filter")
        page.wait_for_selector('details.card[data-path="clean.pdf"]')
        page.wait_for_selector('details.card[data-path="flagged.pdf"]')

        page.click("#conf-flagged")
        page.wait_for_function(
            'document.querySelector(\'details.card[data-path="clean.pdf"]\') === null'
        )
        assert page.locator('details.card[data-path="flagged.pdf"]').count() == 1

        # Switching back to "All pages" brings the clean doc back.
        page.click("#conf-all")
        page.wait_for_selector('details.card[data-path="clean.pdf"]')
    finally:
        page.close()


def test_confidence_threshold_slider_reclassifies_a_borderline_page(
    pg_test_dsn, dashboard_server, chromium_browser
):
    # 70 sits between the two thresholds we'll try: not low-confidence at the
    # default 65 cutoff, but low-confidence once the slider is raised to 80.
    pages = [
        {"page_number": 1, "route": "tesseract", "chars": 500, "elapsed_seconds": 1.0,
         "flagged": False, "flag_reasons": [], "garbled": False,
         "ocr_mean_confidence": 70.0, "vlm_agreement": None, "preview": ""},
    ]
    _write_build_with_pages(pg_test_dsn, "ui-conf-threshold", pages=pages)

    page = chromium_browser.new_page()
    try:
        page.goto(f"{dashboard_server}/?build=ui-conf-threshold")
        page.wait_for_selector('details.card[data-path="d1.pdf"]')

        page.click("#conf-flagged")
        page.wait_for_function(
            'document.querySelector(\'details.card[data-path="d1.pdf"]\') === null'
        )

        page.evaluate(
            "() => { const el = document.getElementById('conf-threshold-range');"
            " el.value = 80; el.dispatchEvent(new Event('input')); }"
        )
        page.wait_for_selector('details.card[data-path="d1.pdf"]')
    finally:
        page.close()


def test_chunk_table_filters_to_chunks_touching_a_low_confidence_page(
    pg_test_dsn, dashboard_server, chromium_browser
):
    logger = BuildEventLogger("ui-conf-chunks", dsn=pg_test_dsn)
    logger.event("build_start", total_todo=1, provider="local")
    logger.event("doc_start", index=1, total=1, path="d1.pdf")
    logger.event(
        "extract_done", doc_id="d1", path="d1.pdf", extractor="local-cascade",
        page_count=2,
        pages=[
            {"page_number": 1, "route": "tesseract", "chars": 500, "elapsed_seconds": 1.0,
             "flagged": False, "flag_reasons": [], "garbled": False,
             "ocr_mean_confidence": 92.0, "vlm_agreement": None, "preview": ""},
            {"page_number": 2, "route": "tesseract", "chars": 500, "elapsed_seconds": 1.0,
             "flagged": True, "flag_reasons": ["low_ocr_confidence(40<65)"], "garbled": False,
             "ocr_mean_confidence": 40.0, "vlm_agreement": None, "preview": ""},
        ],
        degradations=[],
    )
    logger.event(
        "chunk_done", doc_id="d1", chunk_count=2,
        chunks=[
            _chunk(1, page_start=1, page_end=1),
            _chunk(2, page_start=2, page_end=2),
        ],
    )
    logger.close()

    page = chromium_browser.new_page()
    try:
        page.goto(f"{dashboard_server}/?build=ui-conf-chunks")
        page.click("details summary")
        page.wait_for_selector("table.chunktbl tbody tr")
        assert page.locator("table.chunktbl tbody tr").count() == 2

        page.click("#conf-flagged")
        page.wait_for_function(
            "document.querySelectorAll('table.chunktbl tbody tr').length === 1"
        )
        assert "p2-2" in page.inner_text("table.chunktbl")
        assert "p1-1" not in page.inner_text("table.chunktbl")
    finally:
        page.close()


# --------------------------------------------------------------------- #
# Extraction-route filter
# --------------------------------------------------------------------- #

def test_route_filter_checkbox_hides_pages_of_that_route(
    pg_test_dsn, dashboard_server, chromium_browser
):
    _write_build_with_pages(pg_test_dsn, "ui-route-filter", pages=[
        {"page_number": 1, "route": "native", "chars": 500, "elapsed_seconds": 0.1,
         "flagged": False, "flag_reasons": [], "preview": ""},
        {"page_number": 2, "route": "tesseract", "chars": 500, "elapsed_seconds": 0.1,
         "flagged": False, "flag_reasons": [], "preview": ""},
    ])

    page = chromium_browser.new_page()
    try:
        page.goto(f"{dashboard_server}/?build=ui-route-filter")
        page.click("details summary")
        page.wait_for_selector(".pg")
        assert page.locator(".pg").count() == 2

        page.click('.route-check[data-route="tesseract"]')
        page.wait_for_function("document.querySelectorAll('.pg').length === 1")
        assert page.locator(".pg.native").count() == 1
        assert page.locator(".pg.tesseract").count() == 0

        # Toggling it back on restores the page.
        page.click('.route-check[data-route="tesseract"]')
        page.wait_for_function("document.querySelectorAll('.pg').length === 2")
    finally:
        page.close()


def test_route_filter_takes_a_single_route_document_off_the_board(
    pg_test_dsn, dashboard_server, chromium_browser
):
    clean_pages = [
        {"page_number": 1, "route": "native", "chars": 500, "elapsed_seconds": 0.1,
         "flagged": False, "flag_reasons": [], "preview": ""},
    ]
    flagged_pages = [
        {"page_number": 1, "route": "tesseract", "chars": 500, "elapsed_seconds": 0.1,
         "flagged": False, "flag_reasons": [], "preview": ""},
    ]
    _write_build_with_two_docs(pg_test_dsn, "ui-route-board", clean_pages, flagged_pages)

    page = chromium_browser.new_page()
    try:
        page.goto(f"{dashboard_server}/?build=ui-route-board")
        page.wait_for_selector('details.card[data-path="clean.pdf"]')
        page.wait_for_selector('details.card[data-path="flagged.pdf"]')

        page.click('.route-check[data-route="native"]')
        page.wait_for_function(
            'document.querySelector(\'details.card[data-path="clean.pdf"]\') === null'
        )
        assert page.locator('details.card[data-path="flagged.pdf"]').count() == 1
    finally:
        page.close()


def test_route_filter_combines_with_confidence_filter(
    pg_test_dsn, dashboard_server, chromium_browser
):
    _write_build_with_pages(pg_test_dsn, "ui-route-conf-combo", pages=[
        {"page_number": 1, "route": "native", "chars": 500, "elapsed_seconds": 0.1,
         "flagged": True, "flag_reasons": ["garbled_text"], "garbled": True, "preview": ""},
        {"page_number": 2, "route": "tesseract", "chars": 500, "elapsed_seconds": 0.1,
         "flagged": True, "flag_reasons": ["low_ocr_confidence(40<65)"], "garbled": False,
         "ocr_mean_confidence": 40.0, "vlm_agreement": None, "preview": ""},
    ])

    page = chromium_browser.new_page()
    try:
        page.goto(f"{dashboard_server}/?build=ui-route-conf-combo")
        page.click("details summary")
        page.wait_for_selector(".pg")

        page.click("#conf-flagged")
        page.wait_for_function("document.querySelectorAll('.pg').length === 2")

        # Both pages are flagged, so both survive the confidence filter alone;
        # excluding the native route on top should leave just the tesseract one.
        page.click('.route-check[data-route="native"]')
        page.wait_for_function("document.querySelectorAll('.pg').length === 1")
        assert page.locator(".pg.tesseract").count() == 1
    finally:
        page.close()
