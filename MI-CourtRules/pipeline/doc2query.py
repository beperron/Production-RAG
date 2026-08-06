#!/usr/bin/env python3
"""Generate, for each chunk, the questions a practitioner would ask it.

    doc2query.py 3_chunks/v_rule512.jsonl -o 3_chunks/doc2query.jsonl

Why this and not more context. The measured difficulty in this corpus is a
REGISTER gap, not a coverage gap: an evaluation query shares only ~18% of its
content words with the provision that answers it, because practitioners
describe situations ("opposing counsel is demanding we post a bond") while
rules state conditions on actors ("security for costs ... shall not be
required"). Recall is already 0.83 at k=10, so the answer is usually in the
pool; it is the top of the ranking that suffers.

A context blurb describes the passage in the RULE's register, which is the
register retrieval already has. Generated questions add the register it is
missing. They are embedded alongside the provision, so a query matches a
question that resembles it rather than having to cross the gap unaided.

Runs on qwen3.6:27b locally.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HOST = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen3.6:27b-q4_K_M"
LOCK = threading.Lock()
STATS = collections.Counter()

PROMPT = """You are indexing the Michigan Court Rules for a search engine used
by judges, clerks and attorneys.

RULE: {rule_title}  (Chapter {chapter})
PROVISION: {citation}

{text}

Write the 4 questions a practitioner would ask that THIS provision answers.

Rules for the questions:
- Use the words a lawyer with a file open would use, not the words the
  provision uses. If the provision says "security for costs", a practitioner
  might say "post a bond".
- Be concrete: name the posture, the actor, the deadline, the document.
- Do NOT include any rule number or citation.
- Each on its own line, no numbering, no preamble.

Four questions:"""


def gen(prompt, timeout=180, attempts=3):
    body = {"model": MODEL, "prompt": prompt, "stream": False, "think": False,
            "options": {"temperature": 0.4, "num_predict": 220}}
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(HOST, json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            t = (d.get("response") or "").strip()
            if t:
                with LOCK:
                    STATS["ok"] += 1
                return t
            raise ValueError("empty")
        except Exception as exc:                        # noqa: BLE001
            last = exc
            with LOCK:
                STATS["retry"] += 1
            time.sleep(min(2 ** i, 8))
    with LOCK:
        STATS["fail"] += 1
    raise RuntimeError(str(last))


def clean(raw):
    out = []
    for line in raw.splitlines():
        s = line.strip().lstrip("-*0123456789.) ").strip()
        if len(s.split()) < 4 or "MCR" in s:
            continue
        out.append(s)
    return out[:4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks")
    ap.add_argument("-o", "--out", default="3_chunks/doc2query.jsonl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    chunks = [json.loads(l) for l in open(args.chunks) if l.strip()]
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for l in open(out_path):
            if l.strip():
                done.add(json.loads(l)["chunk_id"])
    todo = [c for c in chunks if c["chunk_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{MODEL} · {len(todo):,} chunks ({len(done):,} done)", flush=True)

    fh = open(out_path, "a")
    t0 = time.time()

    def work(c):
        try:
            qs = clean(gen(PROMPT.format(
                rule_title=c["rule_title"], chapter=c["chapter"],
                citation=c["citation_first"] or c["rule"],
                text=c["text"][:2600])))
        except Exception as exc:                        # noqa: BLE001
            print(f"  FAIL {c['chunk_id']}: {exc}", flush=True)
            return
        if not qs:
            return
        with LOCK:
            fh.write(json.dumps({"chunk_id": c["chunk_id"], "questions": qs},
                                ensure_ascii=False) + "\n")
            fh.flush()
            n = STATS["ok"]
            if n and n % 100 == 0:
                el = time.time() - t0
                print(f"  {n:,}/{len(todo):,} · {el/60:.1f} min · "
                      f"eta {(len(todo)-n)*el/n/60:.0f} min", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    print(f"\ndone in {(time.time()-t0)/60:.1f} min · {dict(STATS)}")


if __name__ == "__main__":
    sys.exit(main())
