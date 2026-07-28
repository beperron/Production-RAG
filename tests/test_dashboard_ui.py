"""Browser-level checks for the build dashboard's throbbing run-id pills.

Renders the real page with a headless Chromium (via Playwright) against the
throwaway test DB, so these exercise the actual DOM/CSS/JS the browser sees —
not just the HTML/JSON the server emits. Skips cleanly if Playwright or its
Chromium binary isn't installed (see conftest.py's chromium_browser fixture).
"""
from __future__ import annotations

from parsevault.pipeline.build_events import BuildEventLogger


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
