#!/usr/bin/env python3
"""Score a chunking + retrieval configuration against mcr_eval_v1.

    eval_retrieval.py --chunks 3_chunks/chunks.jsonl --mode hybrid --tag base

The embedder is FIXED at Qwen3-Embedding-4B by decision, so every number here
is comparable and the only things varying are chunking, retrieval and
reranking.

Scoring rules that matter
-------------------------
A retrieval is correct when the returned CHUNK contains the gold citation.
Gold is a citation string, so this survives re-chunking -- the whole reason the
eval was built that way.

TWINS ARE CO-GOLD. 115 groups of byte-identical provisions span 244 citations
(MCR 2.222(E)(2) == 2.223(C)(2) == 2.225(C)(2) == 2.227(D)(2)). Returning one
when gold names another is not a miss: no retriever can tell them apart,
because they are the same bytes. Scoring them as misses would penalise every
configuration equally for something none of them can fix, and would understate
the whole benchmark.

UNANSWERABLE QUERIES ARE SCORED SEPARATELY. They have no gold, so they cannot
contribute to recall; they measure whether the system declines, which is a
generation property, not a retrieval one.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import re
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = ROOT / "4_eval" / "cache"
EMBEDDER = "Qwen/Qwen3-Embedding-4B"
RRF_K = 60


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def twin_groups(blocks):
    """Citations whose provision text is byte-identical."""
    h = collections.defaultdict(list)
    for b in blocks:
        if b["kind"] == "body" and b.get("citation") and len(b["text"].split()) > 8:
            h[hashlib.sha256(b["text"].strip().encode()).hexdigest()].append(b["citation"])
    eq = {}
    for group in h.values():
        if len(group) > 1:
            s = set(group)
            for c in group:
                eq.setdefault(c, set()).update(s)
    return eq


class Embedder:
    _model = None

    @classmethod
    def model(cls):
        if cls._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            cls._model = SentenceTransformer(
                EMBEDDER, device="mps",
                model_kwargs={"torch_dtype": torch.float16})
            # chunks top out near 1,300 tokens; the 40k default window costs
            # attention time for padding nothing needs
            cls._model.max_seq_length = 1536
        return cls._model

    @classmethod
    def encode(cls, texts, key, is_query=False, batch=32):
        CACHE.mkdir(parents=True, exist_ok=True)
        p = CACHE / f"{key}.npy"
        if p.exists():
            return np.load(p)
        kw = {"prompt_name": "query"} if is_query else {}
        t0 = time.time()
        v = cls.model().encode(texts, batch_size=batch, convert_to_numpy=True,
                               normalize_embeddings=True,
                               show_progress_bar=False, **kw)
        np.save(p, v)
        print(f"    embedded {len(texts):,} {'queries' if is_query else 'chunks'}"
              f" in {time.time()-t0:.0f}s -> {p.name}", flush=True)
        return v


# Practitioners type shorthand; the rules never do. BM25 is half the hybrid
# and cannot match "PPO" against "personal protection order" on its own.
ALIASES = {
    "ppo": "personal protection order", "sj": "summary disposition",
    "msj": "motion summary disposition", "foc": "friend of the court",
    "gal": "guardian ad litem", "tpr": "termination of parental rights",
    "coa": "court of appeals", "msc": "supreme court",
    "jnov": "judgment notwithstanding the verdict",
    "esi": "electronically stored information",
    "adr": "alternative dispute resolution", "pi": "personal injury",
    "dv": "domestic violence", "cps": "child protective services",
    "erpo": "extreme risk protection order", "psr": "presentence report",
    "roa": "register of actions", "sol": "statute of limitations",
    "pc": "probable cause", "poa": "power of attorney",
    "bindover": "bind over", "prelim": "preliminary examination",
    "interrog": "interrogatories", "depo": "deposition",
    "atty": "attorney", "ct": "court", "def": "defendant", "pl": "plaintiff",
}


def expand_aliases(q):
    out = []
    for w in re.findall(r"[A-Za-z0-9']+", q):
        out.append(w)
        a = ALIASES.get(w.lower())
        if a:
            out.append(a)
    return " ".join(out)


def build_bm25(chunks):
    from rank_bm25 import BM25Okapi
    return BM25Okapi([re.findall(r"[a-z0-9]+", c["embed_text"].lower())
                      for c in chunks])


RE_CITE = re.compile(r"\bMCR\s*(\d+\.\d+[A-Za-z]?)((?:\s*\([A-Za-z0-9]{1,4}\))*)", re.I)


def route(query, by_citation):
    m = RE_CITE.search(query)
    if not m:
        return None
    want = f"MCR {m.group(1)}{re.sub(r'\\s+', '', m.group(2) or '')}"
    if want in by_citation:
        return by_citation[want]
    bare = f"MCR {m.group(1)}"
    if bare in by_citation:
        return by_citation[bare]
    for cit, i in by_citation.items():
        if cit.startswith(bare + "("):
            return i
    return None


def evaluate(chunks_path, queries, eq, mode="hybrid", k=10, use_router=True,
             reranker=None, tag="", hyde=None, dedup_rule=False,
             w_dense=1.0, w_bm25=1.0, aliases=False):
    chunks = load_jsonl(chunks_path)
    by_citation = {}
    for i, c in enumerate(chunks):
        for cit in c["citations"]:
            by_citation.setdefault(cit, i)
    chunk_cits = [set(c["citations"]) for c in chunks]

    key = f"{tag}.{pathlib.Path(chunks_path).stem}.{len(chunks)}"
    vecs = Embedder.encode([c["embed_text"] for c in chunks], key)
    if hyde:
        # embed the hypothetical provision, not the question: rule-register
        # against rule-register, which is what closes the measured gap
        texts = [hyde.get(q["query_id"], q["query"]) for q in queries]
        qvecs = Embedder.encode(texts, f"hyde.{len(queries)}", is_query=False)
    else:
        qvecs = Embedder.encode([q["query"] for q in queries],
                                f"queries.{len(queries)}", is_query=True)
    bm = build_bm25(chunks) if mode in ("hybrid", "bm25") else None

    ranks = []
    for n, q in enumerate(queries):
        seen, order = set(), []
        if use_router:
            r = route(q["query"], by_citation)
            if r is not None:
                order.append(r)
                seen.add(r)
        lists = []
        if mode in ("hybrid", "bm25"):
            qt = expand_aliases(q["query"]) if aliases else q["query"]
            s = bm.get_scores(re.findall(r"[a-z0-9]+", qt.lower()))
            lists.append((np.argsort(-s)[:80], w_bm25))
        if mode in ("hybrid", "dense"):
            s = vecs @ qvecs[n]
            lists.append((np.argsort(-s)[:80], w_dense))
        fused = {}
        for lst, w in lists:
            for r_, i in enumerate(lst):
                fused[i] = fused.get(i, 0.0) + w / (RRF_K + r_ + 1)
        # One rule can occupy several of the top slots with adjacent chunks,
        # spending the budget on redundancy. Collapse to the best chunk per
        # rule so k slots hold k distinct rules.
        rules_used = set()
        for i, _ in sorted(fused.items(), key=lambda kv: -kv[1]):
            if i in seen:
                continue
            if dedup_rule:
                rl = chunks[i]["rule"]
                if rl in rules_used:
                    continue
                rules_used.add(rl)
            order.append(i)
            seen.add(i)
            if len(order) >= max(k, 40 if reranker else k):
                break

        if reranker is not None:
            pool = order[:40]
            pairs = [[q["query"], chunks[i]["embed_text"][:1800]] for i in pool]
            sc = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
            order = [pool[j] for j in np.argsort(-np.asarray(sc))] + order[40:]

        gold = set(q["gold"]) | set(q.get("also_answered_by") or [])
        for g in list(gold):
            gold |= eq.get(g, set())
        hit_rank = None
        for r_, i in enumerate(order[:k]):
            if chunk_cits[i] & gold:
                hit_rank = r_ + 1
                break
        ranks.append(hit_rank)
    return ranks


def metrics(ranks):
    n = len(ranks)
    at = lambda t: sum(1 for r in ranks if r and r <= t) / n
    mrr = sum(1.0 / r for r in ranks if r) / n
    return {"n": n, "R@1": round(at(1), 4), "R@3": round(at(3), 4),
            "R@5": round(at(5), 4), "R@10": round(at(10), 4),
            "MRR@10": round(mrr, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="3_chunks/chunks.jsonl")
    ap.add_argument("--eval", default="2_eval/mcr_eval_v1.jsonl")
    ap.add_argument("--blocks", default="1_parsed/blocks.jsonl")
    ap.add_argument("--mode", default="hybrid",
                    choices=["hybrid", "dense", "bm25"])
    ap.add_argument("--no-router", action="store_true")
    ap.add_argument("--rerank")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--out", default="4_eval/results.jsonl")
    args = ap.parse_args()

    qs = [q for q in load_jsonl(args.eval) if q["query_type"] != "unanswerable"]
    eq = twin_groups(load_jsonl(args.blocks))

    rr = None
    if args.rerank:
        from sentence_transformers import CrossEncoder
        rr = CrossEncoder(args.rerank, device="mps")

    t0 = time.time()
    ranks = evaluate(args.chunks, qs, eq, mode=args.mode,
                     use_router=not args.no_router, reranker=rr, tag=args.tag)
    m = metrics(ranks)
    m.update({"tag": args.tag, "chunks": args.chunks, "mode": args.mode,
              "router": not args.no_router, "rerank": args.rerank or None,
              "secs": round(time.time() - t0, 1)})

    # per-slice, because a headline number hides where a config actually fails
    for field in ("query_type", "arm"):
        for val in sorted({q[field] for q in qs}):
            sub = [r for r, q in zip(ranks, qs) if q[field] == val]
            if sub:
                m[f"R@1[{field}={val}]"] = round(
                    sum(1 for r in sub if r and r <= 1) / len(sub), 4)
    for lo, hi, name in ((0, 2, "shallow"), (3, 9, "deep")):
        sub = [r for r, q in zip(ranks, qs) if lo <= q.get("depth", 0) <= hi]
        if sub:
            m[f"R@1[{name}]"] = round(sum(1 for r in sub if r and r <= 1) / len(sub), 4)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as fh:
        fh.write(json.dumps(m) + "\n")
    print(json.dumps({k: v for k, v in m.items()
                      if not k.startswith("R@1[")}, indent=1))
    print("  slices:", {k[4:-1]: v for k, v in m.items() if k.startswith("R@1[")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
