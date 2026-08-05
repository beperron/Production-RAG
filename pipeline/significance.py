#!/usr/bin/env python3
"""Paired significance tests between retrieval configurations.

    significance.py 4_eval/ranks.jsonl --baseline rule512 --against rule256 ...

Every configuration is scored on the SAME queries, so the comparisons are
paired and McNemar is far more powerful than reading two overlapping
confidence intervals -- which is the trap LESSONS-LEARNED names explicitly.

Why this file exists at all. With 1,060 queries the 95% interval on a
proportion near 0.5 is about +/- 3 points, so any two configs within ~6 points
of each other are indistinguishable from a leaderboard alone. The CPS
factorial found 53 of 154 configurations statistically tied with the winner.
Reporting a rank order without saying which differences are real would repeat
that mistake in a document going to a court.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import random
import sys


def load_ranks(path):
    """{tag: [rank or None per query]} plus the query ids they align to."""
    runs, ids = {}, None
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        runs[r["tag"]] = r["ranks"]
        ids = r.get("query_ids", ids)
    return runs, ids


def hit(ranks, k=1):
    return [1 if (r and r <= k) else 0 for r in ranks]


def mcnemar(a, b):
    """Exact binomial two-sided test on the discordant pairs."""
    b_only = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    a_only = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return a_only, b_only, min(1.0, p)


def boot_ci(a, b, iters=20000, seed=20260805):
    """Bootstrap 95% CI on the paired difference in hit rate."""
    rng = random.Random(seed)
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    means = []
    for _ in range(iters):
        s = 0
        for _ in range(n):
            s += d[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    return means[int(iters * 0.025)], means[int(iters * 0.975)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ranks", default="4_eval/ranks.jsonl", nargs="?")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--fast", action="store_true", help="skip the bootstrap")
    args = ap.parse_args()

    runs, _ = load_ranks(args.ranks)
    if args.baseline not in runs:
        raise SystemExit(f"no run tagged {args.baseline!r}; have "
                         f"{sorted(runs)}")
    base = hit(runs[args.baseline], args.k)
    n = len(base)
    print(f"paired comparison against {args.baseline!r} at R@{args.k} "
          f"(n = {n:,})")
    print(f"  {args.baseline:<22} {sum(base)/n:.4f}\n")
    print(f"  {'config':<22} {'R@'+str(args.k):>7} {'delta':>8} "
          f"{'wins':>6} {'loses':>6} {'p':>9}   95% CI")
    rows = []
    for tag, ranks in runs.items():
        if tag == args.baseline:
            continue
        h = hit(ranks, args.k)
        a_only, b_only, p = mcnemar(base, h)
        d = sum(h) / n - sum(base) / n
        ci = ("", "") if args.fast else boot_ci(base, h)
        rows.append((d, tag, sum(h) / n, a_only, b_only, p, ci))
    for d, tag, rate, a_only, b_only, p, ci in sorted(rows, key=lambda r: -r[0]):
        star = "*" if p < 0.05 else " "
        cis = "" if args.fast else f"  [{ci[0]:+.3f}, {ci[1]:+.3f}]"
        print(f"  {tag:<22} {rate:>7.4f} {d:>+8.4f} {b_only:>6} {a_only:>6} "
              f"{p:>9.2e}{star}{cis}")
    print("\n  * p < 0.05 (exact binomial on discordant pairs)")
    print("    'wins' = queries this config gets and the baseline misses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
