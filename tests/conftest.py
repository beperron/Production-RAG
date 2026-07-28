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
