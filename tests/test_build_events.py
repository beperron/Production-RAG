"""Tests for parsevault.pipeline.build_events.BuildEventLogger.

Hash-chain computation and error handling are pure Python and tested
without a database. Insert/read-back and the append-only triggers need a
live Postgres and skip automatically if one isn't reachable (see
tests/conftest.py's pg_test_dsn fixture).
"""
from __future__ import annotations

import pytest

from parsevault.pipeline import build_events as be

# --------------------------------------------------------------------- #
# Pure logic — no DB required
# --------------------------------------------------------------------- #

def test_default_dsn_uses_env_override(monkeypatch):
    monkeypatch.setenv("BUILD_LOG_DSN", "postgresql://x:y@z/db")
    assert be.default_dsn() == "postgresql://x:y@z/db"


def test_default_dsn_uses_postgres_password_env(monkeypatch):
    monkeypatch.delenv("BUILD_LOG_DSN", raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", "supersecret")
    assert be.default_dsn() == "postgresql://carolina_policy:supersecret@localhost:5433/carolina_policy"


def test_default_dsn_falls_back_to_compose_default(monkeypatch):
    monkeypatch.delenv("BUILD_LOG_DSN", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    assert be.default_dsn() == "postgresql://carolina_policy:carolina_policy_dev@localhost:5433/carolina_policy"


def test_row_hash_is_deterministic():
    h1 = be._row_hash("prev", "build-1", 0, "2026-01-01T00:00:00", "build_start", {"a": 1})
    h2 = be._row_hash("prev", "build-1", 0, "2026-01-01T00:00:00", "build_start", {"a": 1})
    assert h1 == h2
    assert len(h1) == 64
    int(h1, 16)  # valid hex


def test_row_hash_changes_with_any_input():
    base = be._row_hash("prev", "build-1", 0, "ts", "stage", {"a": 1})
    assert base != be._row_hash("different-prev", "build-1", 0, "ts", "stage", {"a": 1})
    assert base != be._row_hash("prev", "build-2", 0, "ts", "stage", {"a": 1})
    assert base != be._row_hash("prev", "build-1", 1, "ts", "stage", {"a": 1})
    assert base != be._row_hash("prev", "build-1", 0, "ts", "other_stage", {"a": 1})
    assert base != be._row_hash("prev", "build-1", 0, "ts", "stage", {"a": 2})


def test_row_hash_field_order_does_not_matter():
    h1 = be._row_hash("prev", "b", 0, "ts", "s", {"a": 1, "b": 2})
    h2 = be._row_hash("prev", "b", 0, "ts", "s", {"b": 2, "a": 1})
    assert h1 == h2


def test_logger_disabled_when_psycopg_missing(monkeypatch):
    monkeypatch.setattr(be, "psycopg", None)
    logger = be.BuildEventLogger("test-build")
    logger.event("build_start", total_todo=1)  # must not raise
    logger.close()  # must not raise


def test_logger_disabled_on_unreachable_dsn():
    logger = be.BuildEventLogger("test-build", dsn="postgresql://nobody:nothing@127.0.0.1:1/nope")
    logger.event("build_start", total_todo=1)  # must not raise
    logger.close()


# --------------------------------------------------------------------- #
# DB-backed — needs the local Postgres stack (postgres/compose.yml)
# --------------------------------------------------------------------- #

def test_events_persist_with_valid_chain(pg_test_dsn, pg_conn):
    logger = be.BuildEventLogger("chain-test", dsn=pg_test_dsn)
    logger.event("build_start", total_todo=2)
    logger.event("doc_start", index=1, path="a.pdf")
    logger.close()

    rows = pg_conn.execute(
        "select seq, stage, prev_hash, row_hash from build_log "
        "where build_id = %s order by seq",
        ("chain-test",),
    ).fetchall()
    assert [r[1] for r in rows] == ["build_start", "doc_start"]
    assert rows[0][2] == be.GENESIS_HASH
    assert rows[1][2] == rows[0][3]  # each prev_hash == prior row's row_hash
    assert rows[0][3] != rows[1][3]


def test_build_log_rejects_update(pg_test_dsn, pg_conn):
    logger = be.BuildEventLogger("mutate-test", dsn=pg_test_dsn)
    logger.event("build_start", total_todo=1)
    logger.close()

    with pytest.raises(Exception, match="append-only"):
        pg_conn.execute(
            "update build_log set stage = 'tampered' where build_id = %s",
            ("mutate-test",),
        )


def test_build_log_rejects_delete(pg_test_dsn, pg_conn):
    logger = be.BuildEventLogger("delete-test", dsn=pg_test_dsn)
    logger.event("build_start", total_todo=1)
    logger.close()

    with pytest.raises(Exception, match="append-only"):
        pg_conn.execute("delete from build_log where build_id = %s", ("delete-test",))


def test_build_log_rejects_truncate(pg_test_dsn, pg_conn):
    """TRUNCATE doesn't fire row-level triggers, so this needs its own
    statement-level trigger — the UPDATE/DELETE ones don't cover it."""
    logger = be.BuildEventLogger("truncate-test", dsn=pg_test_dsn)
    logger.event("build_start", total_todo=1)
    logger.close()

    with pytest.raises(Exception, match="append-only"):
        pg_conn.execute("truncate build_log")


def test_failed_write_does_not_advance_chain(pg_test_dsn, pg_conn):
    """A write to a dead connection must not silently corrupt the in-memory
    chain state — the next successful event should still chain off the last
    row that actually made it to Postgres."""
    logger = be.BuildEventLogger("resilience-test", dsn=pg_test_dsn)
    logger.event("build_start", total_todo=1)
    logger._conn.close()  # simulate the connection dying mid-build
    logger.event("doc_start", index=1, path="a.pdf")  # must not raise

    rows = pg_conn.execute(
        "select stage from build_log where build_id = %s order by seq",
        ("resilience-test",),
    ).fetchall()
    assert [r[0] for r in rows] == ["build_start"]
