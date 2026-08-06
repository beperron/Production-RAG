#!/usr/bin/env python3
"""Materialise the cross-reference graph at parse time.

    build_graph.py 1_parsed/blocks.jsonl -o 1_parsed/xrefs.jsonl

Until now the graph existed only transiently: regex over retrieved passage
text at answer time, outbound only, first-three matches. That misses the
entire INBOUND direction -- 63 rules have exceptions or overrides living in
OTHER rules ("Notwithstanding MCR 5.158(A)...", written in MCR 5.740), and an
answer drawn from the target rule alone states law that another rule has
modified.

One pass over every provision, every edge resolved against the real citation
set, classified by relation and ranked by legal force:

  overrides     "notwithstanding X", "shall not ... under X"   strongest
  excepts       "except as provided in X"
  conditions    "subject to X", "in accordance with X",
                "governed by X", "pursuant to X", "under X" (imperative)
  defines       "as defined in X", "as used in X"
  refers        bare citation in running text                  weakest

The edge list serves three consumers: answer-time expansion (both
directions), the provenance UI ("this rule is modified by..."), and the
Supabase mcr.xrefs table when credentials arrive.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

RX_CITE = re.compile(r"\bMCR\s+(\d+\.\d+[A-Za-z]?)((?:\s*\([A-Za-z0-9]{1,4}\))*)")

# relation patterns tested against the ~90 chars before the citation
RELATIONS = [
    ("overrides", re.compile(
        r"(notwithstanding|((shall|may)\s+not[^.]{0,50}(under|as)))\s*$", re.I)),
    ("excepts", re.compile(
        r"except\s+as\s+(otherwise\s+)?(provided|specified|set\s+forth)"
        r"[^.]{0,40}$", re.I)),
    ("conditions", re.compile(
        r"(subject\s+to|in\s+accordance\s+with|as\s+provided\s+(in|by)|"
        r"governed\s+by|pursuant\s+to|in\s+the\s+manner\s+provided"
        r"[^.]{0,30}|unless\s+otherwise\s+provided[^.]{0,30})\s*$", re.I)),
    ("defines", re.compile(
        r"as\s+(defined|used)\s+in[^.]{0,30}$|same\s+meaning[^.]{0,40}$", re.I)),
]
FORCE = {"overrides": 0, "excepts": 1, "conditions": 2, "defines": 3,
         "refers": 4}


def classify(pre_text):
    for name, rx in RELATIONS:
        if rx.search(pre_text):
            return name
    return "refers"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blocks")
    ap.add_argument("-o", "--out", default="1_parsed/xrefs.jsonl")
    args = ap.parse_args()

    blocks = [json.loads(l) for l in open(args.blocks) if l.strip()]
    body = [b for b in blocks if b["kind"] == "body" and b.get("citation")]
    valid = {b["citation"] for b in body}
    valid_rules = {b["rule"] for b in body if b.get("rule")}

    edges = []
    for b in body:
        for m in RX_CITE.finditer(b["text"]):
            rule = m.group(1)
            subs = re.sub(r"\s+", "", m.group(2) or "")
            if rule not in valid_rules or rule == b.get("rule"):
                continue
            # resolve to the deepest citation the corpus actually has
            target = None
            groups = re.findall(r"\([A-Za-z0-9]{1,4}\)", subs)
            for n in range(len(groups), -1, -1):
                cand = f"MCR {rule}{''.join(groups[:n])}"
                if cand in valid:
                    target = cand
                    break
            if target is None:
                continue
            rel = classify(b["text"][max(0, m.start() - 90):m.start()])
            edges.append({
                "from": b["citation"], "from_rule": b["rule"],
                "to": target, "to_rule": rule,
                "relation": rel, "force": FORCE[rel],
                "context": b["text"][max(0, m.start() - 70):m.end() + 40]
                           .replace("\n", " ").strip(),
            })

    # dedupe identical (from,to,relation)
    seen, uniq = set(), []
    for e in edges:
        k = (e["from"], e["to"], e["relation"])
        if k not in seen:
            seen.add(k)
            uniq.append(e)

    out = pathlib.Path(args.out)
    with open(out, "w") as fh:
        for e in sorted(uniq, key=lambda e: (e["from"], e["force"])):
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    by_rel = collections.Counter(e["relation"] for e in uniq)
    inbound_mod = collections.Counter(
        e["to_rule"] for e in uniq if e["relation"] in ("overrides", "excepts"))
    print(f"wrote {out}  ({len(uniq):,} edges from {len(edges):,} mentions)")
    print(f"  by relation : {dict(by_rel.most_common())}")
    print(f"  binding (overrides+excepts+conditions): "
          f"{sum(v for k, v in by_rel.items() if k in ('overrides','excepts','conditions')):,}")
    print(f"  rules modified from OUTSIDE (inbound overrides/excepts): "
          f"{len(inbound_mod)}")
    for r, n in inbound_mod.most_common(5):
        print(f"    MCR {r}: {n}")


if __name__ == "__main__":
    sys.exit(main())
