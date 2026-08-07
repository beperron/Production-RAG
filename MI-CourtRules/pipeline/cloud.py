#!/usr/bin/env python3
"""Cloud adapters for the deployed (Vercel + Supabase) Court Rule Searcher.

CloudEngine keeps every measured behaviour of mcr_search.Engine -- the
citation router, graph expansion, token-budget context assembly, prompts --
and swaps only the two pieces a serverless host cannot run locally:

* query embedding  -> OpenRouter qwen/qwen3-embedding-4b, with the Qwen
  instruct template (raw queries embed ~0.91 cosine from local; templated
  0.996-0.9998 -- measured, not optional), truncated 2560 -> 2000 dims and
  renormalised (paired vs full width: p = 0.804, lossless);
* the dense scan   -> pgvector HNSW on mcr.chunks (ef_search 400: 18/20
  rank-1 agreement with the exact local scan on identical vectors).

QueryLogPG mirrors querylog.QueryLog's interface onto the mcr.* tables the
SQLite log was designed to port into.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
import uuid

from mcr_search import MODE, Engine

EMBED_MODEL = "qwen/qwen3-embedding-4b"
QUERY_TEMPLATE = ("Instruct: Given a web search query, retrieve relevant "
                  "passages that answer the query\nQuery: ")
DIMS = 2000


class _DB:
    """One psycopg connection per instance, lock-serialised, rebuilt on
    error -- the transaction pooler treats each transaction independently."""

    def __init__(self, dsn):
        self.dsn = dsn
        self._conn = None
        self._lock = threading.Lock()

    def run(self, fn):
        import psycopg
        with self._lock:
            for attempt in (1, 2):
                try:
                    if self._conn is None or self._conn.closed:
                        self._conn = psycopg.connect(
                            self.dsn, connect_timeout=15, autocommit=False)
                        # transaction-pooler safe: no server-side prepares
                        self._conn.prepare_threshold = None
                    with self._conn.transaction():
                        return fn(self._conn.cursor())
                except psycopg.OperationalError:
                    try:
                        self._conn.close()
                    except Exception:                    # noqa: BLE001
                        pass
                    self._conn = None
                    if attempt == 2:
                        raise


class CloudEngine(Engine):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.idx_of = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}
        self.db = _DB(os.environ["SUPABASE_DB_URL"])
        self._or_key = os.environ["OPENROUTER_API_KEY"]

    # -- query embedding via OpenRouter -----------------------------------
    def embed_query(self, query):
        body = json.dumps({"model": EMBED_MODEL,
                           "input": [QUERY_TEMPLATE + query]}).encode()
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/embeddings", body,
            {"Content-Type": "application/json",
             "Authorization": f"Bearer {self._or_key}"})
        last = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    v = json.load(r)["data"][0]["embedding"][:DIMS]
                n = sum(x * x for x in v) ** 0.5
                return [x / n for x in v]
            except Exception as exc:                     # noqa: BLE001
                last = exc
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"embedding unavailable: {last}")

    # -- dense scan via pgvector ------------------------------------------
    def dense_ranked(self, query, n=60):
        vec = "[" + ",".join(f"{x:.6f}" for x in self.embed_query(query)) + "]"

        def q(cur):
            cur.execute("set local hnsw.ef_search = 400")
            cur.execute(
                "select chunk_id, 1 - (embedding <=> %s::vector) as cos "
                "from mcr.chunks order by embedding <=> %s::vector limit %s",
                (vec, vec, n))
            return cur.fetchall()

        return [(self.idx_of[cid], float(cos)) for cid, cos in self.db.run(q)
                if cid in self.idx_of]

    def search(self, query, k=8, mode=MODE):
        # identical control flow to Engine.search's shipped dense_only path,
        # with the local encode+matmul replaced by the pgvector scan
        routed = self.route_citation(query) if mode != "dense_only" else None
        hits, seen = [], set()
        if routed:
            cit, idx = routed
            hits.append(self._hit(idx, 1.0, "citation-router", cit))
            seen.add(idx)
        for idx, cos in self.dense_ranked(query):
            if idx in seen:
                continue
            hits.append(self._hit(idx, cos, mode, None))
            seen.add(idx)
            if len(hits) >= k:
                break
        return hits[:k]


class QueryLogPG:
    """querylog.QueryLog's interface, writing straight to mcr.*."""

    def __init__(self, db):
        self.db = db

    def record(self, question, result, verify, latency_ms, refused):
        aid = uuid.uuid4().hex[:16]
        hits = result.get("hits", [])

        def w(cur):
            cur.execute(
                "insert into mcr.answers (answer_id, question, answer, "
                "generator, run_id, refused, latency_ms) "
                "values (%s,%s,%s,%s,'run-2026-08-05',%s,%s)",
                (aid, question, result.get("answer"), result.get("model"),
                 bool(refused), int(latency_ms)))
            cur.executemany(
                "insert into mcr.answer_passages (answer_id, chunk_id, rank, "
                "score, route, because_of) values (%s,%s,%s,%s,%s,%s)",
                [(aid, h["chunk_id"], n + 1, float(h.get("score") or 0),
                  h["how"], h.get("because_of")) for n, h in enumerate(hits)])
            cur.executemany(
                "insert into mcr.answer_citations (answer_id, citation, "
                "exists_in_corpus, was_retrieved) values (%s,%s,%s,%s) "
                "on conflict do nothing",
                [(aid, r["citation"], bool(r["exists"]),
                  bool(r["was_retrieved"]))
                 for r in (verify or {}).get("citations", [])])

        self.db.run(w)
        return aid

    def set_feedback(self, answer_id, vote):
        if vote not in (1, -1):
            return False

        def w(cur):
            cur.execute(
                "update mcr.answers set feedback=%s, feedback_at=now() "
                "where answer_id=%s", (vote, answer_id))
            return cur.rowcount == 1

        return bool(self.db.run(w))

    def stats(self):
        def q(cur):
            cur.execute(
                "select count(*), count(*) filter (where refused), "
                "count(*) filter (where feedback=1), "
                "count(*) filter (where feedback=-1) from mcr.answers")
            return cur.fetchone()

        n, r, up, dn = self.db.run(q)
        return {"answers": n, "refusals": r, "thumbs_up": up,
                "thumbs_down": dn}
