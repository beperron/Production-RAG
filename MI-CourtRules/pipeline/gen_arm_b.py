#!/usr/bin/env python3
"""Arm B: generate evaluation queries with GLM 5.2 via the Ollama Cloud API.

    gen_arm_b.py 2_eval/targets.jsonl -o 2_eval/arm_b.jsonl [--limit N]

Resumable: already-generated citations are skipped, so an interrupted run
continues where it stopped. Concurrency is modest by default -- the point is a
clean set, not a fast one.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_prompts import build_prompt, SCHEMA          # noqa: E402

KEY_PATH = pathlib.Path(os.path.expanduser("~/.config/ollama/cloud.key"))
URL = "https://ollama.com/api/generate"
LOCK = threading.Lock()
STATS = collections.Counter()

FENCE = __import__("re").compile(r"^\s*```(?:json)?\s*|\s*```\s*$", __import__("re").M)


def parse_loose(txt):
    """The cloud endpoint does not apply the JSON schema for every model, so
    the reply may arrive fenced and with the model's own key names. Recover
    the object rather than discarding a good question over its wrapper."""
    body = FENCE.sub("", txt).strip()
    if not body.startswith("{"):
        i, j = body.find("{"), body.rfind("}")
        if i == -1 or j <= i:
            raise ValueError(f"no JSON object in reply: {txt[:120]!r}")
        body = body[i:j + 1]
    d = json.loads(body)
    if "query" not in d:
        for alias in ("question", "q", "text"):
            if alias in d:
                d["query"] = d.pop(alias)
                break
    if not d.get("query"):
        raise ValueError(f"no query field in {list(d)}")
    d.setdefault("reasoning", "")
    d.setdefault("confidence", "")
    d.setdefault("also_answered_by", [])
    if isinstance(d["also_answered_by"], str):
        d["also_answered_by"] = [d["also_answered_by"]]
    return d


def call(model, prompt, key, temperature=0.8, attempts=4, timeout=240):
    body = {"model": model, "prompt": prompt, "stream": False,
            "format": SCHEMA, "think": False,
            "options": {"temperature": temperature}}
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(
                URL, json.dumps(body).encode(),
                {"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            txt = (d.get("response") or "").strip()
            if not txt:
                raise ValueError(f"empty response, done_reason={d.get('done_reason')}")
            out = parse_loose(txt)
            with LOCK:
                STATS["ok"] += 1
                STATS["tok_out"] += int(d.get("eval_count") or 0)
            return out
        except Exception as exc:                       # noqa: BLE001
            last = exc
            with LOCK:
                STATS["retry"] += 1
            time.sleep(min(2 ** i, 15))
    with LOCK:
        STATS["fail"] += 1
    raise RuntimeError(f"generation failed after {attempts}: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets")
    ap.add_argument("-o", "--out", default="2_eval/arm_b.jsonl")
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--arm", default="B")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()

    key = KEY_PATH.read_text().strip()
    targets = [json.loads(l) for l in open(args.targets) if l.strip()]

    # A co-gold must be a citation this corpus can actually return. Models
    # reach for the State Bar rules and the evidence rules, which are real
    # authorities but not in this document.
    valid = set()
    bp = pathlib.Path(args.targets).parent.parent / "1_parsed" / "blocks.jsonl"
    if bp.exists():
        for l in open(bp):
            c = json.loads(l).get("citation")
            if c:
                valid.add(c)
    print(f"  {len(valid):,} citations available as co-gold")

    # siblings give the model the context to say what is distinctive here
    by_rule = collections.defaultdict(list)
    for t in targets:
        by_rule[t["rule"]].append(t["citation"])

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for l in open(out_path):
            if l.strip():
                done.add(json.loads(l)["citation"])

    todo = [t for t in targets
            if t["arm"] == args.arm and t["citation"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"arm {args.arm} · {args.model} · {len(todo)} to generate "
          f"({len(done)} already done)")

    fh = open(out_path, "a")

    def work(t):
        try:
            r = call(args.model, build_prompt(t, by_rule[t["rule"]]), key)
        except Exception as exc:                       # noqa: BLE001
            print(f"  FAIL {t['citation']}: {exc}", flush=True)
            return
        rec = {
            "query_id": f"mcr-q-{args.arm}-{t['citation'].replace(' ', '').replace('/', '')}",
            "arm": args.arm, "generator": args.model,
            "query": r["query"].strip(),
            "gold": [t["citation"]] if t["query_type"] != "unanswerable" else [],
            "also_answered_by": [c for c in
                                 (c.strip() for c in r.get("also_answered_by", []))
                                 if c in valid],
            "also_answered_rejected": [c for c in
                                       (c.strip() for c in r.get("also_answered_by", []))
                                       if c and c not in valid],
            "query_type": t["query_type"],
            "citation": t["citation"], "rule": t["rule"], "chapter": t["chapter"],
            "depth": t["depth"], "prominence": t["prominence"],
            "size_band": t["size_band"], "block_id": t["block_id"],
            "generator_reasoning": r.get("reasoning", ""),
            "generator_confidence": r.get("confidence", ""),
            "status": "draft",
        }
        with LOCK:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n = STATS["ok"]
            if n and n % 25 == 0:
                print(f"  {n} generated · {STATS['retry']} retries · "
                      f"{STATS['fail']} failed", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(work, todo))
    fh.close()
    dt = time.time() - t0
    print(f"\ndone in {dt/60:.1f} min · {dict(STATS)}")
    if STATS["ok"]:
        print(f"  {dt/STATS['ok']:.1f}s per query")


if __name__ == "__main__":
    sys.exit(main())
