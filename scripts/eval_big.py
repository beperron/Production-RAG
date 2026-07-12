#!/usr/bin/env python3
"""Score the large synthetic benchmark against the LIVE system (Supabase hybrid
+ Jina rerank), the same path the app uses. Reports recall@k + MRR per corpus.

Env: JINA_API_KEY.  Usage: python scripts/eval_big.py [--limit N]
"""
import argparse, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
SB = "https://vfkllpkkorbqfshwhxex.supabase.co"
AK = "sb_publishable_29nC5rlDsZoV6KXwJuNB0A_iAzGbdN0"
H = {"apikey": AK, "Authorization": f"Bearer {AK}", "Content-Type": "application/json"}


def embed(q, key):
    return requests.post("https://api.jina.ai/v1/embeddings", timeout=60,
                         headers={"Authorization": f"Bearer {key}"},
                         json={"model": "jina-embeddings-v3", "task": "retrieval.query",
                               "dimensions": 1024, "input": [q]}).json()["data"][0]["embedding"]


def rerank(q, docs, key):
    r = requests.post("https://api.jina.ai/v1/rerank", timeout=60,
                      headers={"Authorization": f"Bearer {key}"},
                      json={"model": "jina-reranker-v2-base-multilingual", "query": q,
                            "documents": [d[:1200] for d in docs], "top_n": len(docs)}).json()
    s = [0.0] * len(docs)
    for it in r.get("results", []): s[it["index"]] = it.get("relevance_score", 0)
    return s


def search(q, coll, key):
    f = requests.post(f"{SB}/rest/v1/rpc/hybrid_search", headers=H, timeout=60,
                      json={"query_text": q, "query_embedding": embed(q, key),
                            "match_count": 40, "pool": 40, "coll": coll}).json()
    if not isinstance(f, list) or not f: return []
    sc = rerank(q, [(x.get("content") or "") for x in f], key)
    return [x for _, x in sorted(zip(sc, f), key=lambda t: -t[0])][:10]


def hit_ok(h, kind, keyv):
    if kind == "doc":
        return h.get("doc_id") == keyv
    return bool(re.search(re.escape(keyv), f"{h.get('section','')} {h.get('heading_path','')}"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=None); a = ap.parse_args()
    key = os.environ["JINA_API_KEY"].strip()
    gold = json.loads((ROOT / "evals/law_search/gold_big.json").read_text())
    if a.limit: gold = gold[:a.limit]
    ranks = {"legal-authorities": [], "nc-child-welfare": []}
    done = [0]

    def work(g):
        try:
            hits = search(g["q"], g["collection"], key)
            r = next((i + 1 for i, h in enumerate(hits) if hit_ok(h, g["gold_kind"], g["gold_key"])), 0)
        except Exception:
            r = 0
        done[0] += 1
        if done[0] % 50 == 0: print(f"  {done[0]}/{len(gold)}", flush=True)
        return g["collection"], r

    with ThreadPoolExecutor(max_workers=8) as ex:
        for coll, r in ex.map(work, gold):
            ranks[coll].append(r)
    print(f"\n=== LARGE BENCHMARK (n={len(gold)}, live system) ===")
    for coll, rs in ranks.items():
        if not rs: continue
        n = len(rs); r1 = sum(x == 1 for x in rs); r3 = sum(1 <= x <= 3 for x in rs)
        r5 = sum(1 <= x <= 5 for x in rs); r10 = sum(1 <= x <= 10 for x in rs)
        mrr = sum((1 / x if x else 0) for x in rs) / n
        print(f"  {coll:20s} n={n:4d}  R@1={100*r1//n:3d}%  R@3={100*r3//n:3d}%  "
              f"R@5={100*r5//n:3d}%  R@10={100*r10//n:3d}%  MRR={mrr:.3f}")


if __name__ == "__main__":
    main()
