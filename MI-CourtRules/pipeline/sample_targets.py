#!/usr/bin/env python3
"""Choose which provisions the evaluation set will ask about.

    sample_targets.py 1_parsed/blocks.jsonl -o 2_eval/targets.jsonl -n 1250

Sampling is deterministic under --seed so the target list is reproducible and
reviewable BEFORE any model writes a question against it. Getting the strata
right matters more than the wording: a set that over-samples short, heavily
cross-referenced rules will report a retrieval score that does not survive
contact with the long tail.

Strata, measured from the corpus rather than assumed
---------------------------------------------------
chapter      all 9, allocated by body-block mass with a floor, so Chapter 5
             (Probate) cannot be crowded out by Chapter 2 (Civil Procedure)

prominence   the internal citation graph has 1,629 inbound references over 325
             rules, and 300 rules are never referenced at all. The top 50 rules
             absorb 55% of references. Real questions concentrate there, but a
             benchmark that ignores the tail cannot detect a retriever that has
             simply memorised the popular rules -- so the tail is deliberately
             over-sampled relative to its reference share.

depth        citation depth 0-6. Deep provisions ((C)(10)(a)(ii)) are where a
             chunker either preserves the path or loses it, so depth >= 3 is
             held at roughly a third of the set.

size         rule char-length quartile. Long rules must be split by any chunker,
             so they exercise the split policy; short rules test whether a
             standalone chunk carries enough meaning.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random
import re
import statistics as st
import sys

RX_REF = re.compile(r"MCR\s+(\d+\.\d+[A-Za-z]?)")

# query types and their share of the set
MIX = [
    ("known_item", 0.40, "restate the provision as a practitioner would ask it"),
    ("procedural", 0.25, "a real-world how/when/what-must-I-do question"),
    ("citation_lookup", 0.10, "verbatim citation, answered by the router"),
    ("cross_reference", 0.10, "requires following a reference between rules"),
    ("disambiguation", 0.10, "a term or duty addressed in more than one rule"),
    ("unanswerable", 0.05, "plausible but NOT answered anywhere in the MCR"),
]


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def build_index(blocks):
    rules = {}
    for b in blocks:
        if b["kind"] == "rule" and b["rule"]:
            rules[b["rule"]] = {"rule": b["rule"], "title": b["text"],
                                "chapter": b["chapter"], "page": b["page"],
                                "chars": 0, "blocks": []}
    for b in blocks:
        if b["kind"] == "body" and b["rule"] in rules:
            rules[b["rule"]]["chars"] += len(b["text"])
            rules[b["rule"]]["blocks"].append(b)

    inbound = collections.Counter()
    for b in blocks:
        if b["kind"] != "body" or not b["rule"]:
            continue
        for m in RX_REF.finditer(b["text"]):
            t = m.group(1)
            if t in rules and t != b["rule"]:
                inbound[t] += 1
    for r, meta in rules.items():
        meta["inbound"] = inbound.get(r, 0)
    return rules


def prominence(meta, top50):
    if meta["rule"] in top50:
        return "top50"
    return "referenced" if meta["inbound"] > 0 else "tail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blocks")
    ap.add_argument("-o", "--out", default="2_eval/targets.jsonl")
    ap.add_argument("-n", type=int, default=1250)
    ap.add_argument("--seed", type=int, default=20260805)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    blocks = load(args.blocks)
    rules = build_index(blocks)
    top50 = {r for r, _ in collections.Counter(
        {k: v["inbound"] for k, v in rules.items()}).most_common(50)}

    sizes = sorted(m["chars"] for m in rules.values())
    q = [sizes[int(len(sizes) * f)] for f in (0.25, 0.5, 0.75)]

    def size_band(c):
        return ("q1" if c <= q[0] else "q2" if c <= q[1]
                else "q3" if c <= q[2] else "q4")

    # every citable body block is a candidate target
    cands = []
    for b in blocks:
        if b["kind"] != "body" or not b["rule"] or not b["citation"]:
            continue
        if len(b["text"].split()) < 12:      # too short to ask a fair question
            continue
        meta = rules[b["rule"]]
        cands.append({
            "citation": b["citation"], "block_id": b["id"], "rule": b["rule"],
            "rule_title": meta["title"], "chapter": b["chapter"],
            "subchapter": b["subchapter"], "page": b["page"],
            "depth": b["subpath"].count("("), "text": b["text"],
            "prominence": prominence(meta, top50),
            "size_band": size_band(meta["chars"]),
            "rule_chars": meta["chars"], "inbound": meta["inbound"],
        })

    # ---- allocation ------------------------------------------------------
    # Pass 1 guarantees EVERY rule is represented. Sampling blocks alone
    # over-samples long rules (they simply contain more blocks) and leaves
    # hundreds of short rules untested -- the eval would then report a score
    # for the rules that happen to be verbose.
    by_rule = collections.defaultdict(list)
    for c in cands:
        by_rule[c["rule"]].append(c)

    # A handful of rules are a single short sentence, so the 12-word filter
    # leaves them with no candidate at all. Comprehensive means 625 of 625:
    # readmit their longest block rather than leave the rule untested.
    short = {b["rule"] for b in blocks if b["kind"] == "rule"} - set(by_rule)
    for b in blocks:
        if b["kind"] == "body" and b["rule"] in short and b.get("citation"):
            meta = rules[b["rule"]]
            by_rule[b["rule"]].append({
                "citation": b["citation"], "block_id": b["id"], "rule": b["rule"],
                "rule_title": meta["title"], "chapter": b["chapter"],
                "subchapter": b["subchapter"], "page": b["page"],
                "depth": b["subpath"].count("("), "text": b["text"],
                "prominence": prominence(meta, top50),
                "size_band": size_band(meta["chars"]),
                "rule_chars": meta["chars"], "inbound": meta["inbound"],
            })

    def representative(pool):
        """Prefer a substantive, moderately deep provision over a bare stem."""
        return sorted(pool, key=lambda c: (
            0 if 1 <= c["depth"] <= 3 else 1,
            -min(len(c["text"].split()), 120),
        ))[0]

    picked, seen = [], set()
    for r in sorted(by_rule):
        c = representative(by_rule[r])
        seen.add(c["citation"])
        picked.append(c)

    # Pass 2 spends what is left where questions actually concentrate, while
    # holding deep provisions at roughly a third: depth is where a chunker
    # either preserves the citation path or loses it.
    remaining = args.n - len(picked)
    pool = [c for c in cands if c["citation"] not in seen]
    weight = lambda c: (1.0
                        + 0.8 * (c["prominence"] == "top50")
                        + 0.4 * (c["prominence"] == "referenced")
                        + 0.6 * (c["depth"] >= 3))
    rng.shuffle(pool)
    pool.sort(key=lambda c: -weight(c) * rng.random())
    for c in pool:
        if remaining <= 0:
            break
        if c["citation"] in seen:
            continue
        seen.add(c["citation"])
        picked.append(c)
        remaining -= 1

    rng.shuffle(picked)
    picked = picked[:args.n]

    # assign query types across the picked targets, deterministically
    order = []
    for name, share, _ in MIX:
        order += [name] * round(args.n * share)
    order = (order + ["known_item"] * args.n)[:len(picked)]
    rng.shuffle(order)
    for c, t in zip(picked, order):
        c["query_type"] = t
        c["arm"] = "C" if t == "citation_lookup" else None

    # split the model-generated targets evenly between the two arms
    model_targets = [c for c in picked if c["arm"] is None]
    for i, c in enumerate(model_targets):
        c["arm"] = "A" if i % 2 == 0 else "B"

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for c in picked:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    # ---- report ----------------------------------------------------------
    def dist(key):
        return dict(sorted(collections.Counter(c[key] for c in picked).items()))
    print(f"wrote {out}  ({len(picked)} targets, seed {args.seed})")
    print(f"  candidates available : {len(cands):,}")
    print(f"  distinct citations   : {len({c['citation'] for c in picked}):,}")
    print(f"  distinct rules        : {len({c['rule'] for c in picked}):,} of {len(rules)}")
    print(f"  arm      : {dist('arm')}")
    print(f"  type     : {dist('query_type')}")
    print(f"  chapter  : {dist('chapter')}")
    print(f"  prominence: {dist('prominence')}")
    print(f"  depth    : {dist('depth')}")
    print(f"  size band: {dist('size_band')}")


if __name__ == "__main__":
    sys.exit(main())
