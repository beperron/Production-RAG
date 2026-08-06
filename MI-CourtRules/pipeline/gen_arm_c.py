#!/usr/bin/env python3
"""Arm C: verbatim citation lookups, generated deterministically.

    gen_arm_c.py 2_eval/targets.jsonl -o 2_eval/arm_c.jsonl

No model is involved, so these queries cannot drift, hallucinate, or echo. They
exist for two reasons:

* they exercise the citation router, which should answer them exactly and at
  zero inference cost -- dense embedders are weak at exact identifier matching,
  so this is the one query class that should NOT go to the vector index;
* they are a regression canary. If a future change breaks citation-path
  construction, Arm C collapses immediately and unmistakably, while the
  natural-language arms would only sag a little and could be mistaken for
  noise.

Phrasings vary deterministically so the set is not 125 copies of one template,
but every one contains the citation verbatim -- that is the point.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

# how a judge, clerk or attorney actually types a citation lookup
TEMPLATES = [
    "{c}",
    "{c}?",
    "What does {c} say?",
    "text of {c}",
    "{c} full text",
    "pull up {c}",
    "What is required under {c}?",
    "Show me {c}",
    "{c} -- what does it provide?",
    "I need the language of {c}",
    "quote {c}",
    "What are the requirements in {c}?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets")
    ap.add_argument("-o", "--out", default="2_eval/arm_c.jsonl")
    ap.add_argument("--seed", type=int, default=20260805)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    targets = [json.loads(l) for l in open(args.targets) if l.strip()]
    picked = [t for t in targets if t["arm"] == "C"]

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for i, t in enumerate(picked):
            tmpl = TEMPLATES[i % len(TEMPLATES)]
            fh.write(json.dumps({
                "query_id": f"mcr-q-C-{t['citation'].replace(' ', '').replace('/', '')}",
                "arm": "C", "generator": "deterministic",
                "query": tmpl.format(c=t["citation"]),
                "gold": [t["citation"]],
                "also_answered_by": [],
                "query_type": "citation_lookup",
                "citation": t["citation"], "rule": t["rule"],
                "chapter": t["chapter"], "depth": t["depth"],
                "prominence": t["prominence"], "size_band": t["size_band"],
                "block_id": t["block_id"],
                "generator_reasoning": "verbatim citation lookup; routed, not embedded",
                "generator_confidence": "high",
                "status": "accepted",
            }, ensure_ascii=False) + "\n")

    print(f"wrote {out}  ({len(picked)} citation-lookup queries)")
    print(f"  distinct templates used: {min(len(TEMPLATES), len(picked))}")
    print(f"  every query contains its gold citation verbatim: "
          f"{all(t['citation'] in TEMPLATES[i % len(TEMPLATES)].format(c=t['citation']) for i, t in enumerate(picked))}")


if __name__ == "__main__":
    sys.exit(main())
