#!/usr/bin/env python3
"""A/B: baseline vs Anthropic Contextual Retrieval, measured on the gold set.

For a collection, builds two representations of every chunk:
  baseline    = chunk text
  contextual  = GLM context blurb + chunk text
Each is embedded (Jina) AND lexically indexed (BM25). Retrieval = RRF(dense,
BM25) -> top-40 pool -> Jina rerank -> top-k. Reports R@1/R@3/R@5 for both.

Env: JINA_API_KEY.  Usage: python scripts/eval_contextual.py <collection> <gold.json>
"""
import json, math, os, re, sys, time
from collections import Counter
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
JINA = "https://api.jina.ai/v1/embeddings"; RR = "https://api.jina.ai/v1/rerank"


def embed(texts, key, task, batch=64, cache=None):
    if cache and Path(cache).exists():
        return json.loads(Path(cache).read_text())
    out = []
    for i in range(0, len(texts), batch):
        for att in range(5):
            r = requests.post(JINA, timeout=120, headers={"Authorization": f"Bearer {key}"},
                              json={"model": "jina-embeddings-v3", "task": task, "dimensions": 1024,
                                    "input": [t[:8000] for t in texts[i:i+batch]]})
            if r.status_code == 200:
                out.extend(d["embedding"] for d in r.json()["data"]); break
            time.sleep(2*(att+1))
        else:
            raise RuntimeError(f"embed fail @{i}: {r.status_code}")
        print(f"    embed {min(i+batch,len(texts))}/{len(texts)}", end="\r", flush=True)
    print()
    if cache: Path(cache).write_text(json.dumps(out))
    return out


def rerank(q, docs, key):
    r = requests.post(RR, timeout=60, headers={"Authorization": f"Bearer {key}"},
                      json={"model": "jina-reranker-v2-base-multilingual", "query": q,
                            "documents": [d[:1200] for d in docs], "top_n": len(docs)})
    s = [0.0]*len(docs)
    for it in r.json().get("results", []): s[it["index"]] = it.get("relevance_score", 0)
    return s


def cos(a, b):
    return sum(x*y for x, y in zip(a, b))  # jina vectors are L2-normalized


TOK = re.compile(r"[a-z0-9]+")
def toks(s): return TOK.findall(s.lower())

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [toks(d) for d in docs]; self.k1, self.b = k1, b
        self.dl = [len(d) for d in self.docs]; self.avg = sum(self.dl)/max(1, len(self.dl))
        self.df = Counter();
        for d in self.docs:
            for w in set(d): self.df[w] += 1
        self.N = len(self.docs)
        self.idf = {w: math.log(1 + (self.N - f + 0.5)/(f + 0.5)) for w, f in self.df.items()}
        self.tf = [Counter(d) for d in self.docs]
    def scores(self, q):
        qt = toks(q); out = [0.0]*self.N
        for i in range(self.N):
            s = 0.0
            for w in qt:
                if w in self.tf[i]:
                    f = self.tf[i][w]
                    s += self.idf.get(w, 0)*f*(self.k1+1)/(f + self.k1*(1-self.b+self.b*self.dl[i]/self.avg))
            out[i] = s
        return out


def rrf(ranks_lists, K=60):
    score = Counter()
    for ranks in ranks_lists:
        for rank, idx in enumerate(ranks):
            score[idx] += 1.0/(K + rank + 1)
    return [i for i, _ in score.most_common()]


def evaluate(name, texts, embs, bm25, gold, chunks, key, accept_field):
    ranks = []
    for g in gold:
        pats = g.get("accept") or [g["expect"]]
        qe = embed([g["q"]], key, "retrieval.query")[0]
        dense = sorted(range(len(embs)), key=lambda i: -cos(qe, embs[i]))[:40]
        bs = bm25.scores(g["q"]); lexr = sorted(range(len(texts)), key=lambda i: -bs[i])[:40]
        pool = rrf([dense, lexr])[:40]
        rs = rerank(g["q"], [texts[i] for i in pool], key)
        order = [pool[i] for i in sorted(range(len(pool)), key=lambda i: -rs[i])][:10]
        r = next((j+1 for j, idx in enumerate(order)
                  if any(re.search(p, accept_field(chunks[idx]), re.I) for p in pats)), 0)
        ranks.append(r)
    n = len(ranks); r1 = sum(x == 1 for x in ranks); r3 = sum(1 <= x <= 3 for x in ranks); r5 = sum(1 <= x <= 5 for x in ranks)
    print(f"  {name:12s} R@1={100*r1//n:3d}%  R@3={100*r3//n:3d}%  R@5={100*r5//n:3d}%  MRR={sum((1/x if x else 0) for x in ranks)/n:.3f}")


def main():
    coll, goldf = sys.argv[1], sys.argv[2]
    key = os.environ["JINA_API_KEY"].strip()
    idx = json.loads((ROOT/"knowledge-base"/coll/"docindex.json").read_text())
    ctx = {}
    cf = ROOT/"knowledge-base"/coll/"context.jsonl"
    for line in cf.read_text().splitlines():
        d = json.loads(line); ctx[d["chunk_id"]] = d["context"]
    chunks = [c for c in idx["chunks"] if c["chunk_id"] in ctx]
    print(f"{coll}: {len(chunks)} chunks with context")
    base_txt = [c["text"] for c in chunks]
    ctx_txt = [f"{ctx[c['chunk_id']]}\n\n{c['text']}" for c in chunks]
    def af(c): return f"{c.get('heading_path','')} {c['text'][:200]}" if isinstance(c.get('heading_path'),str) else f"{' '.join(c.get('heading_path') or [])} {c['text'][:200]}"
    gold = json.loads(Path(goldf).read_text())
    print("embedding baseline…");  be = embed(base_txt, key, "retrieval.passage", cache=str(ROOT/f"knowledge-base/{coll}/emb_base.json"))
    print("embedding contextual…"); ce = embed(ctx_txt, key, "retrieval.passage", cache=str(ROOT/f"knowledge-base/{coll}/emb_ctx.json"))
    bm_base = BM25(base_txt); bm_ctx = BM25(ctx_txt)
    print("=== A/B on gold ===")
    evaluate("BASELINE", base_txt, be, bm_base, gold, chunks, key, af)
    evaluate("CONTEXTUAL", ctx_txt, ce, bm_ctx, gold, chunks, key, af)


if __name__ == "__main__":
    main()
