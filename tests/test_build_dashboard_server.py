"""Tests for scripts/build_dashboard_server.py.

_summarize() is pure and tested without a database. Everything else reads
from build_log, so needs the local Postgres stack and skips automatically
if it's unreachable (see tests/conftest.py's pg_test_dsn fixture).

Imported as a top-level module (`import build_dashboard_server`) because
`scripts/` is on sys.path via the `pythonpath` pytest option in
pyproject.toml.
"""
from __future__ import annotations

import datetime
import json
import urllib.request

import build_dashboard_server as dash

from parsevault.pipeline.build_events import BuildEventLogger

# --------------------------------------------------------------------- #
# _summarize — pure function, no DB
# --------------------------------------------------------------------- #

def test_summarize_empty():
    s = dash._summarize([])
    assert s["status"] == "idle"
    assert s["total_todo"] == 0


def test_summarize_running_build():
    events = [
        {"stage": "build_start", "total_todo": 3},
        {"stage": "doc_start", "index": 1},
        {"stage": "extract_done", "degradations": [{"kind": "x"}]},
        {"stage": "index_done", "dense_missing": 0},
        {"stage": "doc_error", "path": "bad.pdf", "error": "boom"},
    ]
    s = dash._summarize(events)
    assert s["status"] == "running"
    assert s["total_todo"] == 3
    assert s["done"] == 1
    assert s["docs_indexed"] == 1
    assert s["doc_errors"] == 1
    assert s["degradations"] == 2  # 1 from extract_done + 1 from doc_error


def test_summarize_done_build():
    events = [
        {"stage": "build_start", "total_todo": 1},
        {"stage": "doc_start", "index": 1},
        {"stage": "index_done", "dense_missing": 2},
        {"stage": "build_done", "elapsed_s": 5.0, "chunks": 10, "docs": 1},
    ]
    s = dash._summarize(events)
    assert s["status"] == "done"
    assert s["degradations"] == 1  # dense_missing > 0
    assert s["elapsed_s"] == 5.0
    assert s["chunks"] == 10
    assert s["docs"] == 1


def test_summarize_done_build_with_nothing_new_to_index():
    # A fully-resumed build (everything already in the index) emits no
    # doc_start/index_done events at all — docs_indexed stays 0 even though
    # the index holds real documents, so "docs" must come from build_done.
    events = [
        {"stage": "build_start", "total_todo": 0},
        {"stage": "build_done", "elapsed_s": 7.0, "chunks": 7449, "docs": 28},
    ]
    s = dash._summarize(events)
    assert s["status"] == "done"
    assert s["docs_indexed"] == 0
    assert s["docs"] == 28


def test_summarize_aborted_build():
    # A build_aborted closing event (appended after a killed run — see
    # project_provenance_no_blockchain memory) is terminal, same as build_done,
    # and takes priority over staleness so it doesn't also get relabeled "stalled".
    events = [
        {"stage": "build_start", "total_todo": 5},
        {"stage": "doc_start", "index": 1},
        {"stage": "build_aborted", "reason": "killed manually"},
    ]
    ancient = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
    s = dash._summarize(events, ancient)
    assert s["status"] == "aborted"


def test_summarize_stalled_build():
    # No terminal event and no activity within _STALE_AFTER — a killed build
    # that never got a closing event shouldn't throb as "running" forever.
    events = [
        {"stage": "build_start", "total_todo": 5},
        {"stage": "doc_start", "index": 1},
    ]
    stale_last_seen = datetime.datetime.now(datetime.timezone.utc) - dash._STALE_AFTER - datetime.timedelta(seconds=1)
    s = dash._summarize(events, stale_last_seen)
    assert s["status"] == "stalled"


def test_summarize_recent_activity_is_running_not_stalled():
    events = [
        {"stage": "build_start", "total_todo": 5},
        {"stage": "doc_start", "index": 1},
    ]
    recent_last_seen = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=5)
    s = dash._summarize(events, recent_last_seen)
    assert s["status"] == "running"


# --------------------------------------------------------------------- #
# DB-backed reads
# --------------------------------------------------------------------- #

def test_discover_and_read_events(pg_test_dsn, pg_conn):
    logger = BuildEventLogger("dash-read-test", dsn=pg_test_dsn)
    logger.event("build_start", total_todo=1, provider="local")
    logger.event("build_done", docs=1, chunks=2, elapsed_s=0.5, report_path="r.json")
    logger.close()

    builds = dash._discover_builds(pg_conn)
    assert any(b["build_id"] == "dash-read-test" for b in builds)

    events = dash._read_events(pg_conn, "dash-read-test")
    assert [e["stage"] for e in events] == ["build_start", "build_done"]
    assert events[0]["total_todo"] == 1
    assert events[0]["provider"] == "local"
    assert "ts" in events[0]


def test_read_events_unknown_build_is_empty(pg_test_dsn, pg_conn):
    assert dash._read_events(pg_conn, "does-not-exist") == []


# --------------------------------------------------------------------- #
# HTTP layer — real server, real socket, pointed at the test DB
# --------------------------------------------------------------------- #

def test_health_endpoint(dashboard_server):
    with urllib.request.urlopen(f"{dashboard_server}/health") as resp:
        assert resp.status == 200
        assert resp.read() == b"ok"


def test_index_page_serves_html(dashboard_server):
    with urllib.request.urlopen(f"{dashboard_server}/") as resp:
        assert resp.status == 200
        body = resp.read().decode()
        assert "<title>Corpus Build Dashboard</title>" in body
        assert 'id=run-pills' in body  # throbbing-pill container for in-progress builds


def test_api_builds_and_events_roundtrip(pg_test_dsn, dashboard_server):
    logger = BuildEventLogger("http-roundtrip-test", dsn=pg_test_dsn)
    logger.event("build_start", total_todo=1)
    logger.event("build_done", docs=1, chunks=1, elapsed_s=0.1, report_path="r.json")
    logger.close()

    with urllib.request.urlopen(f"{dashboard_server}/api/builds") as resp:
        builds = json.loads(resp.read())["builds"]
    assert any(b["id"] == "http-roundtrip-test" and b["status"] == "done" for b in builds)

    with urllib.request.urlopen(f"{dashboard_server}/api/events?build=http-roundtrip-test") as resp:
        events = json.loads(resp.read())["events"]
    assert [e["stage"] for e in events] == ["build_start", "build_done"]


def test_page_elapsed_seconds_persists_through_postgres_roundtrip(pg_test_dsn, dashboard_server):
    """extract_done's per-page elapsed_seconds must survive the JSONB
    round-trip through build_log — that's what the dashboard's "Time by
    extraction route" chart reads from /api/events."""
    logger = BuildEventLogger("timing-roundtrip-test", dsn=pg_test_dsn)
    logger.event(
        "extract_done", doc_id="d1", path="d1.pdf", extractor="local-cascade", page_count=2,
        pages=[
            {"page_number": 1, "route": "native", "chars": 500, "elapsed_seconds": 0.012,
             "flagged": False, "flag_reasons": [], "preview": "hi"},
            {"page_number": 2, "route": "vlm", "chars": 900, "elapsed_seconds": 4.5,
             "flagged": False, "flag_reasons": [], "preview": "hi"},
        ],
        degradations=[],
    )
    logger.close()

    with urllib.request.urlopen(
        f"{dashboard_server}/api/events?build=timing-roundtrip-test"
    ) as resp:
        events = json.loads(resp.read())["events"]

    pages = events[0]["pages"]
    assert pages[0]["elapsed_seconds"] == 0.012
    assert pages[1]["elapsed_seconds"] == 4.5


def test_api_events_missing_build_param_is_empty(dashboard_server):
    with urllib.request.urlopen(f"{dashboard_server}/api/events") as resp:
        assert json.loads(resp.read())["events"] == []


def test_api_builds_reports_multiple_concurrent_running_builds(pg_test_dsn, dashboard_server):
    for build_id in ("concurrent-a", "concurrent-b"):
        logger = BuildEventLogger(build_id, dsn=pg_test_dsn)
        logger.event("build_start", total_todo=5, provider="local")
        logger.close()
    done_logger = BuildEventLogger("concurrent-done", dsn=pg_test_dsn)
    done_logger.event("build_start", total_todo=1, provider="local")
    done_logger.event("build_done", docs=1, chunks=1, elapsed_s=0.1, report_path="r.json")
    done_logger.close()

    with urllib.request.urlopen(f"{dashboard_server}/api/builds") as resp:
        builds = {b["id"]: b for b in json.loads(resp.read())["builds"]}

    assert builds["concurrent-a"]["status"] == "running"
    assert builds["concurrent-b"]["status"] == "running"
    assert builds["concurrent-done"]["status"] == "done"
