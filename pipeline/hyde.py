#!/usr/bin/env python3
"""HyDE: embed a hypothetical provision instead of the question.

    hyde.py 2_eval/mcr_eval_v1.jsonl -o 4_eval/hyde.jsonl

The measured problem is a register gap -- an evaluation query shares ~18% of
its content words with the provision that answers it. A bi-encoder is being
asked to place a practitioner's situation ("opposing counsel is demanding we
post a bond") next to a rule's conditions on actors ("security for costs ...
shall not be required"). HyDE removes the asymmetry by writing what the rule
would plausibly say and matching rule-text against rule-text.

The hypothetical does not need to be CORRECT. It needs to be in the right
register and about the right subject; the embedding is what gets used, never
the content. That is the property that makes this safe for a legal corpus --
nothing generated here is ever shown to a user or cited.

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

PROMPT = """Write the text of a Michigan Court Rule provision that would answer
this question.

QUESTION: {query}

Write it the way the Michigan Court Rules are actually written: a short
provision stating who must do what, by when, and subject to what condition.
Use the formal register of a procedural rule -- "must", "shall", "may",
"within N days", "on motion", "the court", "a party".

Do not include a rule number. Do not explain. Do not hedge. If you are unsure
of the specifics, still write a plausible provision on the right subject --
only the wording style and subject matter matter here.

At most 70 words. The provision:"""


def gen(prompt, timeout=180, attempts=3):
    body = {"model": MODEL, "prompt": prompt, "stream": False, "think": False,
            "options": {"temperature": 0.3, "num_predict": 130}}
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
                return " ".join(t.split())
            raise ValueError("empty")
        except Exception as exc:                        # noqa: BLE001
            last = exc
            with LOCK:
                STATS["retry"] += 1
            time.sleep(min(2 ** i, 8))
    with LOCK:
        STATS["fail"] += 1
    raise RuntimeError(str(last))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries")
    ap.add_argument("-o", "--out", default="4_eval/hyde.jsonl")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    qs = [q for q in (json.loads(l) for l in open(args.queries) if l.strip())
          if q["query_type"] != "unanswerable"]
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for l in open(out_path):
            if l.strip():
                done.add(json.loads(l)["query_id"])
    todo = [q for q in qs if q["query_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{MODEL} · {len(todo):,} queries ({len(done):,} done)", flush=True)

    fh = open(out_path, "a")
    t0 = time.time()

    def work(q):
        try:
            h = gen(PROMPT.format(query=q["query"]))
        except Exception as exc:                        # noqa: BLE001
            print(f"  FAIL {q['query_id']}: {exc}", flush=True)
            return
        with LOCK:
            fh.write(json.dumps({"query_id": q["query_id"], "hyde": h},
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
