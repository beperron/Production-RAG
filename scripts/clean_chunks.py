#!/usr/bin/env python3
"""Re-clean the stored chunk text in Supabase with the deterministic cleaner.

Fetches every chunk, applies clean(), and upserts back only the rows whose text
changed (content/collection only — the embedding is left untouched; the fts
generated column recomputes automatically). Public data only.

Env: SUPABASE_URL, SUPABASE_SERVICE_KEY
"""
import os, sys
from pathlib import Path
import requests
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest_supabase import clean, upsert  # reuse cleaner + batched upsert

def main():
    url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
    if not (url and key):
        sys.exit("set SUPABASE_URL and SUPABASE_SERVICE_KEY")
    hdr = {"apikey": key, "Authorization": f"Bearer {key}"}
    rows, off = [], 0
    while True:
        r = requests.get(f"{url}/rest/v1/chunks?select=chunk_id,collection,content&limit=1000&offset={off}",
                         headers=hdr, timeout=90)
        r.raise_for_status()
        b = r.json()
        if not b:
            break
        rows += b
        off += 1000
        print(f"  fetched {len(rows)}", end="\r", flush=True)
    print(f"\nfetched {len(rows)} chunks")
    changed = []
    for c in rows:
        cc = clean(c["content"])
        if cc != c["content"]:
            changed.append({"chunk_id": c["chunk_id"], "collection": c["collection"], "content": cc})
    print(f"{len(changed)} chunks changed by cleaning — upserting…")
    if changed:
        upsert(url, key, "chunks", changed, batch=200)
    print("DONE")

if __name__ == "__main__":
    main()
