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
