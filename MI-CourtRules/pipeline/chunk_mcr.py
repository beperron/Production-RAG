#!/usr/bin/env python3
"""Pack the parsed blocks into retrieval chunks.

    chunk_mcr.py 1_parsed/blocks.jsonl -o 3_chunks/chunks.jsonl

This is PACKING, not splitting. The parse already segmented the document into
11,593 citable provisions, so chunking groups siblings under a shared rule up
to a budget and never crosses a rule boundary.

Why these choices, from measured evidence rather than convention
---------------------------------------------------------------
NEVER CROSS A RULE.  The CPS factorial put chunking at a 0.34 average R@1
swing, dominating embedder (0.08) and retriever (0.05); heading-aware beat
fixed-512 by +0.32 (p < 0.0001). And it only helps where it reaches: the
bench-book replication measured +0.230 (p = 4e-12) on queries whose answer sat
in a chunk that crossed a heading, and +0.000 (p = 1.00) where it did not. At
the rule boundary 40.2% of fixed-512 chunks in this document cross -- the same
regime as CPS (46.6%) and 10-40x the flattened bench books (1.1-3.8%). So the
lever is fully available here.

512 TOKENS, NOT 1024.  Size is a *significant tie* in the factorial (head-1024
vs head-256, p = 1.0), so this is not a quality choice. It is an operational
one: bge-large and mpnet truncate at 512, which is what actually caused the
"bge-large collapse at head-1024" in the replication. 512 keeps every embedder
on the table at no measured cost.

CITATION PATH AS PREFIX.  The single highest-value free signal -- worth ~32
points in the CPS work (0.55 with the heading-path prefix vs 0.23 body-only).
MCR's path is richer than anything there: Chapter > Subchapter > Rule > (C)(10).

CARRY THE PARENT STEM.  Adopted from carolina-policy-search, whose retrieval
uses neighbor_window=1 to keep "a section's enumerated body attached to its
heading". The eval validators independently found the same failure repeatedly:
a bare list item is unanswerable without the stem that holds its operative
verb -- "the court must:" lives in the parent, "(1) advise the respondent" in
the child. A chunk holding only the child cannot answer anything.

NO OVERLAP.  head-512+overlap scored *below* plain head-512 (0.636 vs 0.646).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import re
import sys

BUDGET_TOKENS = 512
TOK_PER_WORD = 1.32          # calibrated for legal prose against cl100k
MIN_TAIL = 40                # never emit a chunk smaller than this many tokens


def toks(text):
    return int(len(text.split()) * TOK_PER_WORD) + 1


def parent_of(subpath):
    """(C)(10)(a) -> (C)(10).  Returns '' at the top."""
    parts = re.findall(r"\([^)]*\)", subpath or "")
    return "".join(parts[:-1])


def build(blocks):
    rules = {}
    order = []
    for b in blocks:
        if b["kind"] == "rule" and b["rule"]:
            rules[b["rule"]] = {"rule": b["rule"], "title": b["text"],
                                "chapter": b["chapter"],
                                "subchapter": b["subchapter"],
                                "page": b["page"], "body": []}
            order.append(b["rule"])
    for b in blocks:
        if b["kind"] == "body" and b["rule"] in rules:
            rules[b["rule"]]["body"].append(b)
    return [rules[r] for r in order]


def chunk_fixed(rules, budget):
    """Structure-blind control: pack provisions by size across rule
    boundaries. This is the arm the CPS factorial measured at 0.37 R@1 against
    0.65 for heading-aware, and the one this corpus's 40.2% crossing rate
    predicts should lose badly here too."""
    flat = [(r, b) for r in rules for b in r["body"]]
    out, cur, n = [], [], 0
    for r, b in flat:
        t = toks(b["text"])
        if cur and n + t > budget:
            out.append(cur); cur, n = [], 0
        cur.append((r, b)); n += t
    if cur:
        out.append(cur)
    return out


def chunk_rule(rule):
    """Pack one rule's provisions into <=BUDGET chunks, splitting only at the
    shallowest available boundary so a subrule is never cut in half.

    The budget is measured against the FINAL embedded string, so the citation
    path and any parent stem have to be reserved up front. Checking the body
    alone and prepending afterwards pushed a third of chunks past 512, which
    bge-large and mpnet silently truncate -- the defect behind the head-1024
    collapse in the replication."""
    body = rule["body"]
    if not body:
        return []

    path_t = toks(f"Chapter {rule['chapter']} > {rule.get('subchapter') or ''} "
                  f"> {rule['title']} > (X)(99)")
    stem_t = max((toks(b["text"]) for b in body
                  if b["subpath"].count("(") <= 1), default=0)
    budget = max(160, BUDGET_TOKENS - path_t - min(stem_t, 180))

    stem_for = {}
    for b in body:
        stem_for[b["subpath"]] = b

    out, cur, n = [], [], 0
    for b in body:
        t = toks(b["text"])
        # A single provision over budget still ships whole: splitting mid-rule
        # is worse than an oversize chunk, and 4 rules in the corpus need it.
        if cur and n + t > budget:
            depth = b["subpath"].count("(")
            shallow = min((x["subpath"].count("(") for x in cur), default=0)
            if depth <= shallow or n >= budget - MIN_TAIL:
                out.append(cur)
                cur, n = [], 0
        cur.append(b)
        n += t
    if cur:
        out.append(cur)
    return out


def render(rule, group, with_prefix=True):
    """The string that gets embedded."""
    path = f"Chapter {rule['chapter']}"
    if rule.get("subchapter"):
        path += f" > {rule['subchapter']}"
    path += f" > {rule['title']}"
    first = group[0]["subpath"]
    if first:
        path += f" > {first}"

    lines = []
    # the parent stem, when this chunk starts below the top of the rule
    par = parent_of(first)
    if par and globals().get("CARRY_STEM", True):
        stem = next((b["text"] for b in rule["body"]
                     if b["subpath"] == par), None)
        if stem and stem not in (g["text"] for g in group):
            lines.append(stem)
    lines += [g["text"] for g in group]
    body = "\n\n".join(lines)
    return (f"{path}\n\n{body}" if with_prefix else body), path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("blocks")
    ap.add_argument("-o", "--out", default="3_chunks/chunks.jsonl")
    ap.add_argument("--no-prefix", action="store_true",
                    help="ablation: embed the body without the citation path")
    ap.add_argument("--no-stem", action="store_true",
                    help="ablation: do not carry the parent stem")
    ap.add_argument("--budget", type=int, default=BUDGET_TOKENS)
    ap.add_argument("--strategy", default="rule",
                    choices=["rule", "provision", "whole", "fixed"])
    args = ap.parse_args()
    globals()["BUDGET_TOKENS"] = args.budget
    globals()["CARRY_STEM"] = not args.no_stem

    blocks = [json.loads(l) for l in open(args.blocks) if l.strip()]
    rules = build(blocks)

    out, n = [], 0
    if args.strategy == "fixed":
        groups = []
        for grp in chunk_fixed(rules, args.budget):
            # a structure-blind chunk has no single owning rule; attribute it
            # to the rule of its first provision, as the control does
            groups.append((grp[0][0], [b for _, b in grp]))
    elif args.strategy == "provision":
        groups = [(r, [b]) for r in rules for b in r["body"]]
    elif args.strategy == "whole":
        groups = [(r, r["body"]) for r in rules if r["body"]]
    else:
        groups = [(r, g) for r in rules for g in chunk_rule(r)]

    for r, g in groups:
        if True:
            embed_text, path = render(r, g, not args.no_prefix)
            cits = [b["citation"] for b in g if b.get("citation")]
            out.append({
                "chunk_id": f"mcr#C{n:05d}",
                "rule": r["rule"], "rule_title": r["title"],
                "chapter": r["chapter"], "subchapter": r["subchapter"],
                "citations": cits,
                "citation_first": cits[0] if cits else None,
                "heading_path": path,
                "block_ids": [b["id"] for b in g],
                "pages": sorted({b["page"] for b in g}),
                "depth_min": min(b["subpath"].count("(") for b in g),
                "depth_max": max(b["subpath"].count("(") for b in g),
                "text": "\n\n".join(b["text"] for b in g),
                "embed_text": embed_text,
                "n_tokens": toks(embed_text),
                "oversize": toks(embed_text) > BUDGET_TOKENS,
                "sha256": hashlib.sha256(embed_text.encode()).hexdigest()[:16],
            })
            n += 1

    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        for c in out:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    tk = sorted(c["n_tokens"] for c in out)
    cov = {c for ch in out for c in ch["citations"]}
    allc = {b["citation"] for b in blocks
            if b["kind"] == "body" and b.get("citation")}
    print(f"wrote {p}  ({len(out):,} chunks from {len(rules)} rules)")
    print(f"  tokens   : min {tk[0]} · median {tk[len(tk)//2]} · "
          f"p90 {tk[int(len(tk)*.9)]} · max {tk[-1]}")
    print(f"  oversize : {sum(c['oversize'] for c in out)} "
          f"(a lone provision over budget, kept whole)")
    print(f"  chunks/rule: median {sorted(collections.Counter(c['rule'] for c in out).values())[len(rules)//2]}")
    print(f"  citation coverage: {len(cov & allc):,} / {len(allc):,} provisions reachable")
    missing = allc - cov
    if missing:
        print(f"  MISSING {len(missing)}: {sorted(missing)[:5]}")


if __name__ == "__main__":
    sys.exit(main())
