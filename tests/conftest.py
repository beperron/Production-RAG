"""Shared pytest fixtures.

DB-backed tests need the local carolina-policy Postgres stack running
(``cd postgres && docker compose up -d``, see postgres/compose.yml) — they
skip automatically if it's unreachable. Fixtures here always create/drop a
throwaway ``carolina_policy_pytest`` database rather than touching the real
``carolina_policy`` database: build_log is deliberately append-only (no
UPDATE/DELETE), so test rows written into the real database would be stuck
there forever.
"""
from __future__ import annotations

import os
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

try:
    import psycopg
except ImportError:  # pragma: no cover - the `build` extra pulls this in
    psycopg = None

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_SQL = REPO_ROOT / "postgres" / "init" / "001_schema.sql"
TEST_DB = "carolina_policy_pytest"


def _admin_dsn() -> str:
    password = os.environ.get("POSTGRES_PASSWORD", "carolina_policy_dev")
    return f"postgresql://carolina_policy:{password}@localhost:5433/carolina_policy"


@pytest.fixture(scope="session")
def pg_test_dsn():
    if psycopg is None:
        pytest.skip("psycopg not installed (pip install -e '.[build]')")
    try:
        admin = psycopg.connect(_admin_dsn(), autocommit=True, connect_timeout=2)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"local postgres (port 5433) not reachable: {e}")

    admin.execute(f"drop database if exists {TEST_DB} with (force)")
    admin.execute(f"create database {TEST_DB}")
    admin.close()

    test_dsn = _admin_dsn().rsplit("/", 1)[0] + f"/{TEST_DB}"
    conn = psycopg.connect(test_dsn, autocommit=True)
    conn.execute(SCHEMA_SQL.read_text())
    conn.close()

    yield test_dsn

    admin = psycopg.connect(_admin_dsn(), autocommit=True)
    admin.execute(f"drop database if exists {TEST_DB} with (force)")
    admin.close()


@pytest.fixture
def pg_conn(pg_test_dsn):
    conn = psycopg.connect(pg_test_dsn, autocommit=True)
    yield conn
    conn.close()


@pytest.fixture
def dashboard_server(pg_test_dsn, monkeypatch):
    """Real dashboard HTTP server, real socket, pointed at the throwaway test DB."""
    import build_dashboard_server as dash

    monkeypatch.setattr(dash, "_DSN", pg_test_dsn)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), dash.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    thread.join(timeout=5)
    httpd.server_close()


@pytest.fixture(scope="session")
def chromium_browser():
    """Headless Chromium via Playwright; skips cleanly if either isn't installed."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright not installed (pip install -e '.[dev]')")

    pw = sync_playwright().start()
    try:
        browser = pw.chromium.launch()
    except Exception as e:  # noqa: BLE001
        pw.stop()
        pytest.skip(f"chromium not available ({e}); run `playwright install chromium`")

    yield browser
    browser.close()
    pw.stop()
