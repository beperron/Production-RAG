#!/usr/bin/env python3
"""Embed the public corpus with Jina v3 and load it into Supabase (pgvector).

Reads the committed docindex.json for each public collection, embeds every chunk
with jina-embeddings-v3 (task=retrieval.passage, 1024-dim), and upserts documents
+ chunks into Postgres. Public data only.

Env:
    JINA_API_KEY        Jina embeddings key
    SUPABASE_DB_URL     postgres connection string (service role / direct)
Usage:
    pip install psycopg[binary] requests
    python scripts/ingest_supabase.py            # both collections
    python scripts/ingest_supabase.py --limit 50 # smoke test
"""
import argparse, json, os, re, sys, time
from pathlib import Path
import requests
import psycopg
from psycopg.types.json import Json  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
COLLECTIONS = ["legal-authorities", "nc-child-welfare"]
JINA_URL = "https://api.jina.ai/v1/embeddings"
SEC_RE = re.compile(r"§?\s?\d+[A-Z]?-\d+(?:\.\d+)?")


def jina_embed(texts, key, task, batch=64):
    out = []
    for i in range(0, len(texts), batch):
        part = texts[i:i + batch]
        for attempt in range(5):
            r = requests.post(JINA_URL, timeout=120,
                              headers={"Authorization": f"Bearer {key}"},
                              json={"model": "jina-embeddings-v3", "task": task,
                                    "dimensions": 1024, "input": part})
            if r.status_code == 200:
                out.extend(d["embedding"] for d in r.json()["data"])
                break
            time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"Jina embed failed at batch {i}: {r.status_code} {r.text[:200]}")
        print(f"    embedded {min(i+batch, len(texts))}/{len(texts)}", end="\r", flush=True)
    print()
    return out


def source_url_of(d):
    for v in d.values():
        if isinstance(v, str) and v.startswith("http"):
            return v
    return d.get("source_path", "")


def section_of(chunk, doc_title):
    hp = chunk.get("heading_path") or []
    for cand in ([hp[-1]] if hp else []) + [doc_title or ""]:
        m = SEC_RE.search(cand or "")
        if m:
            return cand
    return (hp[-1] if hp else "") or (doc_title or "")


def load_rows(limit=None):
    docs, chunks = {}, []
    for coll in COLLECTIONS:
        idx = json.loads((ROOT / "knowledge-base" / coll / "docindex.json").read_text())
        durl = {d["doc_id"]: source_url_of(d) for d in idx["documents"]}
        dtitle = {d["doc_id"]: d.get("title", "") for d in idx["documents"]}
        for d in idx["documents"]:
            docs[d["doc_id"]] = (d["doc_id"], coll, d.get("title", ""),
                                 durl.get(d["doc_id"], ""), d.get("page_count"))
        cs = idx["chunks"][:limit] if limit else idx["chunks"]
        for c in cs:
            ps = c.get("page_start"); pe = c.get("page_end")
            span = (f"pp. {ps}–{pe}" if ps and pe and ps != pe
                    else f"p. {ps}" if ps else "")
            chunks.append({
                "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "collection": coll,
                "ordinal": c.get("ordinal"), "title": dtitle.get(c["doc_id"], ""),
                "section": section_of(c, dtitle.get(c["doc_id"], "")),
                "heading_path": " > ".join(c.get("heading_path") or []),
                "content": c.get("text", ""), "page_span": span,
                "source_url": durl.get(c["doc_id"], ""),
                "sha256": c.get("content_sha256") or c.get("source_sha256") or "",
            })
    return list(docs.values()), chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    key = os.environ.get("JINA_API_KEY", "").strip()
    dsn = os.environ.get("SUPABASE_DB_URL", "").strip()
    if not key or not dsn:
        sys.exit("set JINA_API_KEY and SUPABASE_DB_URL")
    docs, chunks = load_rows(a.limit)
    print(f"{len(docs)} documents, {len(chunks)} chunks to ingest")
    print("embedding chunks with jina-embeddings-v3 (retrieval.passage)…")
    vecs = jina_embed([c["content"][:8000] for c in chunks], key, "retrieval.passage")
    with psycopg.connect(dsn, autocommit=False) as conn, conn.cursor() as cur:
        cur.executemany(
            "insert into documents (doc_id,collection,title,source_url,page_count) "
            "values (%s,%s,%s,%s,%s) on conflict (doc_id) do nothing", docs)
        for c, v in zip(chunks, vecs):
            c["embedding"] = "[" + ",".join(f"{x:.6f}" for x in v) + "]"
        cur.executemany(
            "insert into chunks (chunk_id,doc_id,collection,ordinal,title,section,"
            "heading_path,content,page_span,source_url,sha256,embedding) values "
            "(%(chunk_id)s,%(doc_id)s,%(collection)s,%(ordinal)s,%(title)s,%(section)s,"
            "%(heading_path)s,%(content)s,%(page_span)s,%(source_url)s,%(sha256)s,%(embedding)s) "
            "on conflict (chunk_id) do update set embedding=excluded.embedding, content=excluded.content",
            chunks)
        conn.commit()
    print(f"ingested {len(docs)} docs + {len(chunks)} chunks into Supabase.")


if __name__ == "__main__":
    main()
