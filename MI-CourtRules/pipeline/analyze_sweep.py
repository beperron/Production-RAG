#!/usr/bin/env python3
"""Final table for the chunking sweep: both metrics, plus paired tests.

    analyze_sweep.py --baseline rule512

Recomputes every variant from the cached vectors -- no embedding, no model
load -- so the budget-normalised metric and the per-query ranks are available
for configurations that were scored before that metric existed.

Reports R@1 alongside R@2048tok deliberately. They disagree, and the
disagreement is the finding: R@1 is confounded with chunk granularity, because
a chunk holding N citations satisfies gold-containment for any of N
provisions. Showing only the corrected metric would hide why the naive reading
is wrong, and a court audience is owed the reason, not just the number.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import eval_retrieval as E                              # noqa: E402
import significance as S                                # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

VARIANTS = [
    ("provision", "v_provision"), ("rule256", "v_rule256"),
    ("rule512", "v_rule512"), ("rule1024", "v_rule1024"),
    ("whole-rule", "v_whole"), ("fixed256", "v_fixed256"),
    ("fixed512", "v_fixed512"), ("no-prefix", "v_noprefix"),
    ("no-stem", "v_nostem"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="rule512")
    ap.add_argument("--eval", default="2_eval/mcr_eval_v1.jsonl")
    ap.add_argument("--blocks", default="1_parsed/blocks.jsonl")
    ap.add_argument("--out", default="4_eval/sweep_final.json")
    args = ap.parse_args()

    qs = [q for q in E.load_jsonl(ROOT / args.eval)
          if q["query_type"] != "unanswerable"]
    eq = E.twin_groups(E.load_jsonl(ROOT / args.blocks))
    cached = {os.path.basename(f).split(".")[1]
              for f in glob.glob(str(ROOT / "4_eval/cache/*.npy"))
              if f.count(".") > 2}

    rows, ranks_by, bh_cache = [], {}, {}
    for tag, stem in VARIANTS:
        if stem not in cached:
            print(f"  (skipping {tag}: not embedded yet)")
            continue
        path = ROOT / "3_chunks" / f"{stem}.jsonl"
        chunks = E.load_jsonl(path)
        r, bh, sp = E.evaluate(path, qs, eq, mode="hybrid", tag=tag)
        m = E.metrics(r, bh, sp)
        m["tag"] = tag
        m["cits_per_chunk"] = round(
            sum(len(c["citations"]) for c in chunks) / len(chunks), 1)
        m["n_chunks"] = len(chunks)
        rows.append(m)
        ranks_by[tag] = r
        bh_cache[tag] = [1 if x else 0 for x in bh]

    if not rows:
        raise SystemExit("nothing cached yet")

    print(f"\n{'variant':<12} {'chunks':>7} {'cит/chk':>8} "
          f"{'R@1':>7} {'R@10':>7} {'MRR':>7} | {'R@2048tok':>10} {'tok':>6}")
    print("-" * 78)
    for m in sorted(rows, key=lambda r: -r["R@2048tok"]):
        print(f"{m['tag']:<12} {m['n_chunks']:>7,} {m['cits_per_chunk']:>8.1f} "
              f"{m['R@1']:>7.3f} {m['R@10']:>7.3f} {m['MRR@10']:>7.3f} | "
              f"{m['R@2048tok']:>10.3f} {m['median_tok_spent']:>6}")

    print(f"\nR@1 ordering:        "
          f"{' > '.join(m['tag'] for m in sorted(rows, key=lambda r: -r['R@1']))}")
    print(f"R@2048tok ordering:  "
          f"{' > '.join(m['tag'] for m in sorted(rows, key=lambda r: -r['R@2048tok']))}")

    # paired tests on the budget metric, against the best config
    best = max(rows, key=lambda r: r["R@2048tok"])["tag"]
    print(f"\npaired McNemar on R@2048tok against {best!r} (n = {len(qs):,})")
    print(f"  {'config':<12} {'R@2048':>8} {'delta':>8} {'wins':>6} "
          f"{'loses':>6} {'p':>10}")
    bh_by = bh_cache
    base = bh_by[best]
    n = len(base)
    for tag, h in sorted(bh_by.items(), key=lambda kv: -sum(kv[1])):
        if tag == best:
            continue
        a_only, b_only, p = S.mcnemar(base, h)
        print(f"  {tag:<12} {sum(h)/n:>8.3f} {sum(h)/n - sum(base)/n:>+8.3f} "
              f"{b_only:>6} {a_only:>6} {p:>10.2e}"
              f"{'  *' if p < 0.05 else ''}")
    print("\n  * p < 0.05; 'wins' = queries this config gets that the best misses")

    pathlib.Path(ROOT / args.out).write_text(json.dumps(
        {"rows": rows, "best_by_budget": best}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
