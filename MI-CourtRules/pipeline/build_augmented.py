#!/usr/bin/env python3
"""Fold generated text into a chunk set's embedded string.

    build_augmented.py 3_chunks/v_rule512.jsonl \
        --context 3_chunks/context.jsonl -o 3_chunks/v_ctx.jsonl
    build_augmented.py 3_chunks/v_rule512.jsonl \
        --questions 3_chunks/doc2query.jsonl -o 3_chunks/v_d2q.jsonl

Only `embed_text` changes. `text` stays the verbatim provision, so what a user
is shown and what a generator cites remain exactly what the document says --
no generated sentence can reach an answer or a citation. The generated
material exists solely to give the retriever a second surface to match
against, and it is regenerable from the corpus, so it never becomes a source
of truth.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks")
    ap.add_argument("--context")
    ap.add_argument("--questions")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--questions-only", action="store_true",
                    help="ablation: embed ONLY the generated questions")
    args = ap.parse_args()

    chunks = [json.loads(l) for l in open(args.chunks) if l.strip()]
    ctx = {}
    if args.context:
        for l in open(args.context):
            if l.strip():
                r = json.loads(l)
                ctx[r["chunk_id"]] = r["context"]
    qs = {}
    if args.questions:
        for l in open(args.questions):
            if l.strip():
                r = json.loads(l)
                qs[r["chunk_id"]] = r["questions"]

    n_ctx = n_q = 0
    for c in chunks:
        parts = []
        if c["chunk_id"] in ctx:
            parts.append(ctx[c["chunk_id"]])
            n_ctx += 1
        if c["chunk_id"] in qs:
            parts.append("Questions this answers: "
                         + " ".join(qs[c["chunk_id"]]))
            n_q += 1
        if not parts:
            continue
        if args.questions_only and c["chunk_id"] in qs:
            c["embed_text"] = (c["heading_path"] + "\n\n"
                               + " ".join(qs[c["chunk_id"]]))
        else:
            c["embed_text"] = "\n\n".join(parts + [c["embed_text"]])
        c["n_tokens"] = int(len(c["embed_text"].split()) * 1.32) + 1

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    t = sorted(c["n_tokens"] for c in chunks)
    print(f"wrote {out}  ({len(chunks):,} chunks)")
    print(f"  context added   : {n_ctx:,}")
    print(f"  questions added : {n_q:,}")
    print(f"  tokens: median {t[len(t)//2]} · p90 {t[int(len(t)*.9)]} · max {t[-1]}")


if __name__ == "__main__":
    sys.exit(main())
