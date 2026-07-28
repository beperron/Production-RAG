"""Build-time event emitter for the pedantic build dashboard.

Side-effect-only, same shape as the existing ``degradation_sink`` parameter
pattern used in ``build_document_metadata()`` (docindex.py). Writes each
event straight into the hash-chained, append-only ``build_log`` table in
Postgres (see postgres/init/001_schema.sql) — no intermediate JSONL file.
The hash chain (``prev_hash``/``row_hash``, Certificate-Transparency-log
style: each row commits to the previous row's hash) is computed here,
in-process, at insert time; Postgres only stores it and enforces
append-only via BEFORE UPDATE/DELETE triggers. See memory
provenance-no-blockchain for why this shape was chosen over a blockchain.

Never raises — a build must keep running even if Postgres is unreachable.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys

try:
    import psycopg
except ImportError:  # pragma: no cover - the `build` extra pulls this in
    psycopg = None

GENESIS_HASH = "0" * 64


def default_dsn() -> str:
    """Local carolina-policy Postgres (postgres/compose.yml, port 5433) —
    isolated from the practicum stack's shared postgres on 5432."""
    password = os.environ.get("POSTGRES_PASSWORD", "carolina_policy_dev")
    return os.environ.get(
        "BUILD_LOG_DSN",
        f"postgresql://carolina_policy:{password}@localhost:5433/carolina_policy",
    )


def _row_hash(prev_hash: str, build_id: str, seq: int, ts: str, stage: str, fields: dict) -> str:
    payload = json.dumps(
        {"prev_hash": prev_hash, "build_id": build_id, "seq": seq, "ts": ts,
         "stage": stage, "fields": fields},
        sort_keys=True, ensure_ascii=False, default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class BuildEventLogger:
    def __init__(self, build_id: str, dsn: str | None = None):
        self.build_id = build_id
        self._seq = 0
        self._prev_hash = GENESIS_HASH
        self._conn = None
        if psycopg is None:
            print("WARNING: build event log disabled (psycopg not installed)", file=sys.stderr)
            return
        dsn = dsn or default_dsn()
        try:
            self._conn = psycopg.connect(dsn, autocommit=True)
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: build event log disabled ({dsn}): {e}", file=sys.stderr)
            self._conn = None

    def event(self, stage: str, **fields) -> None:
        if self._conn is None:
            return
        ts = datetime.datetime.now().astimezone()
        ts_iso = ts.isoformat(timespec="seconds")
        seq = self._seq
        row_hash = _row_hash(self._prev_hash, self.build_id, seq, ts_iso, stage, fields)
        try:
            self._conn.execute(
                "insert into build_log (build_id, seq, ts, stage, fields, prev_hash, row_hash) "
                "values (%s, %s, %s, %s, %s, %s, %s)",
                (self.build_id, seq, ts, stage,
                 json.dumps(fields, ensure_ascii=False, default=str),
                 self._prev_hash, row_hash),
            )
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: build event write failed: {e}", file=sys.stderr)
            return
        self._seq += 1
        self._prev_hash = row_hash

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None
