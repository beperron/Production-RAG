#!/usr/bin/env python3
"""Systematic sweep: chunking x retrieval x reranking, one model load.

    run_sweep.py --stage chunking
    run_sweep.py --stage retrieval --chunks 3_chunks/v_rule512.jsonl
    run_sweep.py --stage rerank    --chunks 3_chunks/v_rule512.jsonl

Staged rather than full-factorial, and deliberately so. The CPS work already
established that chunking dominates (0.34 average R@1 swing against 0.05 for
the retriever), so the efficient design is to settle the dominant axis first
and carry its winner forward, rather than spend 150 runs re-deriving a ranking
that is already known. The embedder is fixed at Qwen3-Embedding-4B by decision.

Every configuration is scored on the same 1,092 queries, so the comparisons are
paired and a later significance test is possible.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import eval_retrieval as E                              # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

CHUNKING = [
    ("rule512", "3_chunks/v_rule512.jsonl"),
    ("rule256", "3_chunks/v_rule256.jsonl"),
    ("rule1024", "3_chunks/v_rule1024.jsonl"),
    ("provision", "3_chunks/v_provision.jsonl"),
    ("whole-rule", "3_chunks/v_whole.jsonl"),
    ("fixed512", "3_chunks/v_fixed512.jsonl"),
    ("fixed256", "3_chunks/v_fixed256.jsonl"),
    ("no-prefix", "3_chunks/v_noprefix.jsonl"),
    ("no-stem", "3_chunks/v_nostem.jsonl"),
]


def run(tag, chunks, queries, eq, mode, router=True, rr=None, out=None):
    t0 = time.time()
    ranks = E.evaluate(ROOT / chunks, queries, eq, mode=mode,
                       use_router=router, reranker=rr, tag=tag)
    m = E.metrics(ranks)
    m.update({"tag": tag, "chunks": chunks, "mode": mode, "router": router,
              "rerank": getattr(rr, "_name", None), "secs": round(time.time() - t0, 1)})
    for field in ("query_type",):
        for val in sorted({q[field] for q in queries}):
            sub = [r for r, q in zip(ranks, queries) if q[field] == val]
            if sub:
                m[f"R@1[{val}]"] = round(sum(1 for r in sub if r and r <= 1) / len(sub), 4)
    for name, lo, hi in (("shallow", 0, 2), ("deep", 3, 9)):
        sub = [r for r, q in zip(queries, queries) if lo <= q.get("depth", 0) <= hi]
    sub = [r for r, q in zip(ranks, queries) if q.get("depth", 0) >= 3]
    if sub:
        m["R@1[deep]"] = round(sum(1 for r in sub if r and r <= 1) / len(sub), 4)
    if out:
        with open(out, "a") as fh:
            fh.write(json.dumps(m) + "\n")
    print(f"  {tag:<14} {mode:<7} router={str(router):<5} "
          f"R@1 {m['R@1']:.3f}  R@5 {m['R@5']:.3f}  R@10 {m['R@10']:.3f}  "
          f"MRR {m['MRR@10']:.3f}  ({m['secs']:.0f}s)", flush=True)
    return m, ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["chunking", "retrieval", "rerank"])
    ap.add_argument("--chunks", default="3_chunks/v_rule512.jsonl")
    ap.add_argument("--eval", default="2_eval/mcr_eval_v1.jsonl")
    ap.add_argument("--blocks", default="1_parsed/blocks.jsonl")
    ap.add_argument("--out", default="4_eval/results.jsonl")
    args = ap.parse_args()

    qs = [q for q in E.load_jsonl(ROOT / args.eval)
          if q["query_type"] != "unanswerable"]
    eq = E.twin_groups(E.load_jsonl(ROOT / args.blocks))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"queries {len(qs):,} · twin groups {len(eq):,} citations · "
          f"embedder {E.EMBEDDER}\n", flush=True)

    rows = []
    if args.stage == "chunking":
        # hybrid + router held fixed while the dominant axis varies
        for tag, path in CHUNKING:
            if (ROOT / path).exists():
                rows.append(run(tag, path, qs, eq, "hybrid", True, None, out)[0])
    elif args.stage == "retrieval":
        for mode in ("dense", "bm25", "hybrid"):
            for router in (True, False):
                if mode == "bm25" and router is False:
                    pass
                rows.append(run(f"{pathlib.Path(args.chunks).stem}", args.chunks,
                                qs, eq, mode, router, None, out)[0])
    elif args.stage == "rerank":
        from sentence_transformers import CrossEncoder
        for name in ("BAAI/bge-reranker-base",):
            rr = CrossEncoder(name, device="mps")
            rr._name = name
            rows.append(run(f"{pathlib.Path(args.chunks).stem}+rerank",
                            args.chunks, qs, eq, "hybrid", True, rr, out)[0])

    if rows:
        best = max(rows, key=lambda r: r["R@1"])
        print(f"\n  best this stage: {best['tag']} {best['mode']} "
              f"R@1 {best['R@1']:.3f} MRR {best['MRR@10']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
