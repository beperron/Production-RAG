#!/usr/bin/env python3
"""Query log: every search recorded, in the shape the Supabase schema expects.

    from querylog import QueryLog
    QueryLog().record(question, result, verify, latency_ms)

Local SQLite at 5_logs/queries.db, deliberately mirroring the mcr.answers /
answer_passages / answer_citations tables from supabase/0001_mcr_schema.sql --
so when credentials arrive the backlog ports with one INSERT SELECT per
table, and nothing about the application changes.

Two design points:

* The log records what the PROVENANCE model records: not just the query text
  but what was retrieved, by which route, and whether every emitted citation
  resolved and was grounded. A query log that keeps only the query cannot
  answer the question you will actually ask of it later -- "what did the
  system SAY to this user, and was it right?"

* Logging is disclosed in the interface footer. A court tool that silently
  records what judges type is a trust failure waiting to be discovered; the
  same page that promises falsifiability says plainly that searches are
  recorded on this machine.
"""
from __future__ import annotations

import json
import pathlib
import sqlite3
import threading
import time
import uuid

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB = ROOT / "5_logs" / "queries.db"

SCHEMA = """
create table if not exists answers (
  answer_id   text primary key,
  asked_at    text not null default (datetime('now')),
  question    text not null,
  answer      text,
  generator   text,
  refused     integer,
  latency_ms  integer,
  n_passages  integer,
  token_count integer
);
create table if not exists answer_passages (
  answer_id  text not null references answers(answer_id),
  chunk_id   text not null,
  rank       integer not null,
  score      real,
  route      text not null,
  citation   text,
  because_of text,
  primary key (answer_id, rank)
);
create table if not exists answer_citations (
  answer_id        text not null references answers(answer_id),
  citation         text not null,
  exists_in_corpus integer not null,
  was_retrieved    integer not null,
  primary key (answer_id, citation)
);
"""


class QueryLog:
    def __init__(self, path=DB):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript("pragma journal_mode=WAL;" + SCHEMA)
        for col in ("feedback integer", "feedback_at text"):
            try:
                self._conn.execute(f"alter table answers add column {col}")
            except sqlite3.OperationalError:
                pass                                    # already migrated
        self._lock = threading.Lock()

    def record(self, question, result, verify, latency_ms, refused):
        aid = uuid.uuid4().hex[:16]
        hits = result.get("hits", [])
        with self._lock:
            c = self._conn
            c.execute(
                "insert into answers (answer_id, question, answer, generator,"
                " refused, latency_ms, n_passages, token_count)"
                " values (?,?,?,?,?,?,?,?)",
                (aid, question, result.get("answer"), result.get("model"),
                 int(bool(refused)), int(latency_ms), len(hits),
                 sum(h.get("n_tokens", 0) for h in hits)))
            c.executemany(
                "insert into answer_passages (answer_id, chunk_id, rank,"
                " score, route, citation, because_of) values (?,?,?,?,?,?,?)",
                [(aid, h["chunk_id"], n + 1, float(h.get("score") or 0),
                  h["how"], h.get("citation"), h.get("because_of"))
                 for n, h in enumerate(hits)])
            c.executemany(
                "insert or ignore into answer_citations (answer_id, citation,"
                " exists_in_corpus, was_retrieved) values (?,?,?,?)",
                [(aid, r["citation"], int(r["exists"]), int(r["was_retrieved"]))
                 for r in (verify or {}).get("citations", [])])
            c.commit()
        return aid

    def set_feedback(self, answer_id, vote):
        """vote: +1 thumbs up, -1 thumbs down. Idempotent per answer."""
        if vote not in (1, -1):
            return False
        with self._lock:
            cur = self._conn.execute(
                "update answers set feedback=?, feedback_at=datetime('now')"
                " where answer_id=?", (vote, answer_id))
            self._conn.commit()
            return cur.rowcount == 1

    def stats(self):
        c = self._conn
        n = c.execute("select count(*) from answers").fetchone()[0]
        r = c.execute("select count(*) from answers where refused=1").fetchone()[0]
        up = c.execute("select count(*) from answers where feedback=1").fetchone()[0]
        dn = c.execute("select count(*) from answers where feedback=-1").fetchone()[0]
        return {"answers": n, "refusals": r, "thumbs_up": up, "thumbs_down": dn}


if __name__ == "__main__":
    q = QueryLog()
    print(json.dumps(q.stats()))
    for row in q._conn.execute(
            "select asked_at, refused, latency_ms, substr(question,1,60)"
            " from answers order by asked_at desc limit 10"):
        print(" ", row)
