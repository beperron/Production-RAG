#!/usr/bin/env python3
"""Compare generator models on the answer layer, paired on the same queries.

    compare_generators.py 4_eval/gen_*.jsonl

Retrieval is held constant (same engine, same k, same passages), the judge is
held constant (kimi-k3, a family none of the contenders belong to), so any
difference is the generator. All comparisons are paired McNemar on the same
140 queries -- at this n, unpaired rate comparisons cannot separate models.
"""
from __future__ import annotations

import collections
import json
import math
import statistics as st
import sys


def mcnemar(a, b):
    b_only = sum(1 for x, y in zip(a, b) if not x and y)
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
    return a_only, b_only, min(1.0, p)


def load(path):
    recs = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            recs[r["query_id"]] = r
    return recs


def main():
    runs = {}
    for path in sys.argv[1:]:
        recs = load(path)
        name = next(iter(recs.values()))["generator"]
        runs[name] = recs
    ids = sorted(set.intersection(*[set(r) for r in runs.values()]))
    print(f"paired on {len(ids)} queries · judge kimi-k3 · retrieval identical\n")

    def rate(recs, f, sel=lambda r: True):
        v = [f(recs[i]) for i in ids if sel(recs[i])]
        return (sum(v) / len(v) if v else float("nan")), len(v)

    ans = lambda r: r["query_type"] != "unanswerable"
    neg = lambda r: r["query_type"] == "unanswerable"
    jd = lambda r: bool(r.get("judge"))
    METRICS = [
        ("citation valid",    lambda r: r["citation_valid"] if r["citation_valid"] is not None else 1.0, ans),
        ("citation grounded", lambda r: r["citation_grounded"] if r["citation_grounded"] is not None else 1.0, ans),
        ("cites gold",        lambda r: 1 if r["cites_gold"] else 0, ans),
        ("false refusal",     lambda r: 1 if r["refused"] else 0, ans),
        ("correct refusal",   lambda r: 1 if r["refused"] else 0, neg),
        ("judged supported",  lambda r: 1 if r["judge"].get("supported") else 0, jd),
        ("judged overreach",  lambda r: 1 if r["judge"].get("overreach") else 0, jd),
        ("cites match",       lambda r: 1 if r["judge"].get("citations_match") else 0, jd),
    ]
    names = list(runs)
    hdr = "  " + f"{'metric':<20}" + "".join(f"{n[:22]:>24}" for n in names)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, f, sel in METRICS:
        row = f"  {label:<20}"
        for n in names:
            v, cnt = rate(runs[n], f, sel)
            row += f"{v:>18.3f} (n={cnt:<3})"
        print(row)
    lat = {n: [runs[n][i]["latency_ms"] for i in ids if "latency_ms" in runs[n][i]]
           for n in names}
    row = f"  {'latency p50 / p90 s':<20}"
    for n in names:
        L = sorted(lat[n])
        row += (f"{L[len(L)//2]/1000:>10.1f} /{L[int(len(L)*.9)]/1000:>6.1f}"
                + " " * 6) if L else f"{'—':>24}"
    print(row)

    print("\n  paired McNemar (row model vs column model)")
    for metric, f, sel in (("cites gold", lambda r: 1 if r["cites_gold"] else 0, ans),
                           ("judged supported", lambda r: 1 if (r.get("judge") or {}).get("supported") else 0, jd)):
        print(f"\n  [{metric}]")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                common = [q for q in ids if sel(runs[a][q]) and sel(runs[b][q])]
                va = [f(runs[a][q]) for q in common]
                vb = [f(runs[b][q]) for q in common]
                ao, bo, p = mcnemar(va, vb)
                da = sum(va) / len(va) - sum(vb) / len(vb)
                lead = a if da > 0 else b
                print(f"    {a[:20]:<22} vs {b[:20]:<22} "
                      f"delta {abs(da):+.3f} to {lead[:18]:<18} p={p:.3g}"
                      f"{'  *' if p < 0.05 else ''}")
    # fabricated citations, the disqualifier
    print("\n  invalid citations emitted (fabrication check)")
    for n in names:
        bad = [c for i in ids for c in runs[n][i].get("invalid_citations", [])]
        print(f"    {n[:30]:<32} {len(bad)}"
              + (f"  e.g. {bad[:3]}" if bad else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
