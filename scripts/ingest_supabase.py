#!/usr/bin/env python3
"""Embed the public corpus with Jina v3 and load it into Supabase via the REST
API (service_role key). Public data only.

Env:
    JINA_API_KEY            Jina embeddings key
    SUPABASE_URL            https://<ref>.supabase.co
    SUPABASE_SERVICE_KEY    service_role key (write; used locally only, never committed)
Usage:
    pip install requests
    python scripts/ingest_supabase.py --limit 40   # smoke test
    python scripts/ingest_supabase.py              # full corpus
"""
import argparse, json, os, re, sys, time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
COLLECTIONS = ["legal-authorities", "nc-child-welfare"]
JINA_URL = "https://api.jina.ai/v1/embeddings"
SEC_RE = re.compile(r"§?\s?\d+[A-Z]?-\d+(?:\.\d+)?")
# strip NUL + non-printable control chars Postgres text can't store (keep \n \t)
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def clean(s):
    return CTRL.sub("", s or "")


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
                out.extend(d["embedding"] for d in r.json()["data"]); break
            time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"Jina failed at {i}: {r.status_code} {r.text[:200]}")
        print(f"    embedded {min(i+batch, len(texts))}/{len(texts)}", end="\r", flush=True)
    print()
    return out


def source_url_of(d):
    for v in d.values():
        if isinstance(v, str) and v.startswith("http"):
            return v
    return d.get("source_path", "")


def section_of(hp, doc_title):
    for cand in ([hp[-1]] if hp else []) + [doc_title or ""]:
        if SEC_RE.search(cand or ""):
            return cand
    return (hp[-1] if hp else "") or (doc_title or "")


def load_rows(limit=None):
    docs, chunks = {}, []
    for coll in COLLECTIONS:
        idx = json.loads((ROOT / "knowledge-base" / coll / "docindex.json").read_text())
        durl = {d["doc_id"]: source_url_of(d) for d in idx["documents"]}
        dtitle = {d["doc_id"]: d.get("title", "") for d in idx["documents"]}
        for d in idx["documents"]:
            docs[d["doc_id"]] = {"doc_id": d["doc_id"], "collection": coll,
                                 "title": d.get("title", ""), "source_url": durl[d["doc_id"]],
                                 "page_count": d.get("page_count")}
        for c in (idx["chunks"][:limit] if limit else idx["chunks"]):
            hp = c.get("heading_path") or []
            ps, pe = c.get("page_start"), c.get("page_end")
            span = (f"pp. {ps}–{pe}" if ps and pe and ps != pe else f"p. {ps}" if ps else "")
            chunks.append({
                "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "collection": coll,
                "ordinal": c.get("ordinal"), "title": clean(dtitle.get(c["doc_id"], "")),
                "section": clean(section_of(hp, dtitle.get(c["doc_id"], ""))),
                "heading_path": clean(" > ".join(hp)), "content": clean(c.get("text", "")),
                "page_span": span, "source_url": durl.get(c["doc_id"], ""),
                "sha256": c.get("content_sha256") or c.get("source_sha256") or "",
            })
    return list(docs.values()), chunks


def _post(url, hdr, table, rows):
    """POST a batch; on statement-timeout (57014) split and recurse."""
    r = requests.post(f"{url}/rest/v1/{table}", headers=hdr,
                      data=json.dumps(rows), timeout=120)
    if r.status_code in (200, 201, 204):
        return len(rows)
    if ("57014" in r.text or r.status_code in (500, 504)) and len(rows) > 1:
        mid = len(rows) // 2
        return _post(url, hdr, table, rows[:mid]) + _post(url, hdr, table, rows[mid:])
    raise RuntimeError(f"upsert {table}: {r.status_code} {r.text[:300]}")


def upsert(url, key, table, rows, batch=100):
    hdr = {"apikey": key, "Authorization": f"Bearer {key}",
           "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates"}
    done = 0
    for i in range(0, len(rows), batch):
        done += _post(url, hdr, table, rows[i:i + batch])
        print(f"    upserted {done}/{len(rows)} into {table}", end="\r", flush=True)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    jk = os.environ.get("JINA_API_KEY", "").strip()
    url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
    if not (jk and url and key):
        sys.exit("set JINA_API_KEY, SUPABASE_URL, SUPABASE_SERVICE_KEY")
    docs, chunks = load_rows(a.limit)
    # resumable: skip chunks already loaded so re-runs only fill the gap
    have = set()
    hdr = {"apikey": key, "Authorization": f"Bearer {key}"}
    off = 0
    while True:
        r = requests.get(f"{url}/rest/v1/chunks?select=chunk_id&limit=1000&offset={off}", headers=hdr, timeout=60)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        have.update(x["chunk_id"] for x in batch); off += 1000
    missing = [c for c in chunks if c["chunk_id"] not in have]
    print(f"{len(docs)} documents, {len(chunks)} chunks ({len(have)} already loaded, {len(missing)} to add)")
    if missing:
        print("embedding missing chunks with jina-embeddings-v3 (retrieval.passage)…")
        vecs = jina_embed([c["content"][:8000] for c in missing], jk, "retrieval.passage")
        for c, v in zip(missing, vecs):
            c["embedding"] = "[" + ",".join(f"{x:.6f}" for x in v) + "]"
    print("upserting to Supabase…")
    upsert(url, key, "documents", docs)
    if missing:
        upsert(url, key, "chunks", missing)
    print(f"DONE — {len(docs)} docs + {len(have) + len(missing)} chunks in Supabase.")


if __name__ == "__main__":
    main()
