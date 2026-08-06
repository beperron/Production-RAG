#!/usr/bin/env python3
"""Ingest the Bench Book corpus into the mcr schema on Supabase.

    ingest_supabase.py [--wipe]

Idempotent: rows key on their natural ids and re-runs upsert. Vectors are the
cached Qwen3-Embedding-4B chunk embeddings truncated to 2000 dims and
re-normalised -- measured statistically identical to full width (0.9509 vs
0.9491, p = 0.804).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import psycopg

ROOT = pathlib.Path(__file__).resolve().parent.parent
ENV = dict(l.split("=", 1) for l in
           (pathlib.Path.home() / ".config/parsevault/lawsearch.env")
           .read_text().splitlines() if "=" in l)


def jl(p):
    return [json.loads(l) for l in open(ROOT / p) if l.strip()]


def main():
    wipe = "--wipe" in sys.argv
    conn = psycopg.connect(ENV["SUPABASE_DB_URL"].strip(), connect_timeout=20)
    conn.autocommit = False
    cur = conn.cursor()

    # feedback columns (added to the app after the schema file was written)
    for col in ("feedback integer", "feedback_at timestamptz"):
        try:
            cur.execute(f"alter table mcr.answers add column {col}")
            conn.commit()
        except psycopg.errors.DuplicateColumn:
            conn.rollback()

    if wipe:
        for t in ("answer_citations", "answer_passages", "answers",
                  "eval_queries", "xrefs", "chunk_blocks", "chunks",
                  "blocks", "parse_runs", "sources"):
            cur.execute(f"delete from mcr.{t}")
        conn.commit()
        print("wiped")

    # ---- source + parse run ---------------------------------------------
    pdf = ROOT / "0_source/michigan-court-rules.pdf"
    sha = hashlib.sha256(pdf.read_bytes()).hexdigest()
    cur.execute("""insert into mcr.sources
        (source_id, filename, sha256, page_count, edition, source_url)
        values (%s,%s,%s,%s,%s,%s) on conflict (source_id) do nothing""",
        ("mcr-2026-07-31", pdf.name, sha, 874,
         "Michigan Court Rules, as amended through July 31, 2026",
         "https://www.courts.michigan.gov/siteassets/rules-instructions-"
         "administrative-orders/michigan-court-rules/michigan-court-rules.pdf"))
    stats = json.load(open(ROOT / "1_parsed/parse_stats.json"))
    cur.execute("""insert into mcr.parse_runs
        (run_id, source_id, parser_commit, n_blocks, n_rules, n_citations,
         toc_reconciled, lossless_word_match, notes)
        values (%s,%s,%s,%s,%s,%s,true,true,%s)
        on conflict (run_id) do nothing""",
        ("run-2026-08-05", "mcr-2026-07-31", "mcr-poc@shipped",
         stats["blocks"], stats["rules"], 11860,
         "structure 625/625 vs printed contents; words 384,507=384,507"))
    conn.commit()
    print("source + parse run")

    # ---- blocks ----------------------------------------------------------
    blocks = jl("1_parsed/blocks.jsonl")
    cur.executemany("""insert into mcr.blocks
        (block_id, run_id, kind, chapter, subchapter, rule, subpath,
         citation, depth, pdf_page, printed_page, text)
        values (%s,'run-2026-08-05',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        on conflict (block_id) do nothing""",
        [(b["id"], b["kind"], b.get("chapter"), b.get("subchapter"),
          b.get("rule"), b.get("subpath"), b.get("citation"), b.get("depth"),
          b["page"], b["page"] - 18, b["text"]) for b in blocks])
    conn.commit()
    print(f"blocks: {len(blocks):,}")

    # ---- chunks + vectors ------------------------------------------------
    chunks = jl("3_chunks/v_rule256.jsonl")
    vecs = np.load(ROOT / "4_eval/cache/rule256.v_rule256.3580.npy")
    assert len(chunks) == vecs.shape[0], "vector/chunk mismatch"
    v2000 = vecs[:, :2000]
    v2000 = v2000 / np.linalg.norm(v2000, axis=1, keepdims=True)

    t0 = time.time()
    B = 100
    for i in range(0, len(chunks), B):
        rows = []
        for j in range(i, min(i + B, len(chunks))):
            c = chunks[j]
            rows.append((c["chunk_id"], c["rule"], c["rule_title"],
                         c["chapter"], c.get("subchapter"),
                         c.get("citation_first"), c["heading_path"],
                         c["text"], c["embed_text"], c["n_tokens"],
                         c["sha256"],
                         "[" + ",".join(f"{x:.6f}" for x in v2000[j]) + "]"))
        cur.executemany("""insert into mcr.chunks
            (chunk_id, run_id, rule, rule_title, chapter, subchapter,
             citation_first, heading_path, text, embed_text, n_tokens,
             sha256, embedding)
            values (%s,'run-2026-08-05',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s::vector)
            on conflict (chunk_id) do nothing""", rows)
        conn.commit()
        if (i // B) % 5 == 0:
            print(f"  chunks {i + len(rows):,}/{len(chunks):,} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    print(f"chunks: {len(chunks):,} with 2000-dim vectors "
          f"({time.time() - t0:.0f}s)")

    # ---- chunk_blocks ----------------------------------------------------
    pairs = [(c["chunk_id"], bid) for c in chunks for bid in c["block_ids"]]
    cur.executemany("""insert into mcr.chunk_blocks (chunk_id, block_id)
        values (%s,%s) on conflict do nothing""", pairs)
    conn.commit()
    print(f"chunk_blocks: {len(pairs):,}")

    # ---- xrefs -----------------------------------------------------------
    xr = jl("1_parsed/xrefs.jsonl")
    cur.executemany("""insert into mcr.xrefs
        (from_citation, to_citation, binding, context)
        values (%s,%s,%s,%s) on conflict do nothing""",
        [(e["from"], e["to"],
          e["relation"] in ("overrides", "excepts", "conditions"),
          f"[{e['relation']}] " + e["context"][:300]) for e in xr])
    conn.commit()
    print(f"xrefs: {len(xr):,}")

    # ---- eval set --------------------------------------------------------
    ev = jl("2_eval/mcr_eval_v1.jsonl")
    cur.executemany("""insert into mcr.eval_queries
        (query_id, query, query_type, arm, generator, gold,
         also_answered_by, status)
        values (%s,%s,%s,%s,%s,%s,%s,%s) on conflict (query_id) do nothing""",
        [(q["query_id"], q["query"], q["query_type"], q["arm"],
          q.get("generator"), q["gold"], q.get("also_answered_by") or [],
          q.get("status")) for q in ev])
    conn.commit()
    print(f"eval_queries: {len(ev):,}")

    # ---- query-log backlog (the INSERT SELECT the log was designed for) --
    import sqlite3
    db = ROOT / "5_logs/queries.db"
    if db.exists():
        lite = sqlite3.connect(str(db))
        a = lite.execute("""select answer_id, asked_at, question, answer,
            generator, refused, latency_ms, feedback, feedback_at
            from answers""").fetchall()
        cur.executemany("""insert into mcr.answers
            (answer_id, asked_at, question, answer, generator, refused,
             latency_ms, feedback, feedback_at)
            values (%s,%s::timestamptz,%s,%s,%s,%s::boolean,%s,%s,
                    %s::timestamptz)
            on conflict (answer_id) do nothing""",
            [(r[0], r[1] + "Z", r[2], r[3], r[4], bool(r[5]), r[6], r[7],
              (r[8] + "Z") if r[8] else None) for r in a])
        ap = lite.execute("""select answer_id, chunk_id, rank, score, route,
            citation, because_of from answer_passages""").fetchall()
        cur.executemany("""insert into mcr.answer_passages
            (answer_id, chunk_id, rank, score, route, because_of)
            values (%s,%s,%s,%s,%s,%s) on conflict do nothing""",
            [(r[0], r[1], r[2], r[3], r[4], r[6]) for r in ap
             if r[1] is not None])
        ac = lite.execute("""select answer_id, citation, exists_in_corpus,
            was_retrieved from answer_citations""").fetchall()
        cur.executemany("""insert into mcr.answer_citations
            (answer_id, citation, exists_in_corpus, was_retrieved)
            values (%s,%s,%s::boolean,%s::boolean)
            on conflict do nothing""",
            [(r[0], r[1], bool(r[2]), bool(r[3])) for r in ac])
        conn.commit()
        print(f"query-log backlog: {len(a)} answers, {len(ap)} passages, "
              f"{len(ac)} citations")

    # ---- verify ----------------------------------------------------------
    print("\n=== verification ===")
    for t, want in (("blocks", len(blocks)), ("chunks", len(chunks)),
                    ("chunk_blocks", None), ("xrefs", len(xr)),
                    ("eval_queries", len(ev)), ("answers", None)):
        n = cur.execute(f"select count(*) from mcr.{t}").fetchone()[0]
        flag = "" if want is None or n == want else f"  EXPECTED {want}"
        print(f"  {t:<14} {n:,}{flag}")
    conn.close()


if __name__ == "__main__":
    main()
