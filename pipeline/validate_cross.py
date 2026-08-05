#!/usr/bin/env python3
"""Cross-validation: one arm's generator adversarially checks the other's items.

    validate_cross.py 2_eval/arm_a.jsonl 1_parsed/blocks.jsonl \
        -o 2_eval/arm_a.validated.jsonl --model glm-5.2

The checker never wrote the item it is checking. That is the whole point: a
model asked to grade its own questions will confirm them, because the question
made sense to it when it wrote it. Arm A is checked by GLM 5.2 and Arm B by
Claude, so every item is seen by a family that had no hand in it.

Three things get decided here, and only the first two need a model:

  answerable    does the gold provision actually answer the question?
  co-gold       does some OTHER provision answer it as well or better? An
                unmarked co-valid answer scores a correct retrieval as a
                failure -- your CPS miss analysis found exactly this artifact.
  echo          computed arithmetically, not judged: the share of the query's
                content words that also appear in the source provision. A high
                value means the question would be found by string overlap
                rather than by understanding.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

KEY_PATH = pathlib.Path(os.path.expanduser("~/.config/ollama/cloud.key"))
URL = "https://ollama.com/api/generate"
LOCK = threading.Lock()
STATS = collections.Counter()
FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.M)

STOP = set("""the a an and or of to in for on with that this which is are be been
by as at from any all not no if when what who how does do must shall may can
could would should there their them its it his her he she we you your our""".split())

ECHO_FAIL = 0.55          # above this, the question is findable by overlap alone


def content_words(t):
    return {w for w in re.findall(r"[a-z']+", t.lower())
            if len(w) > 3 and w not in STOP}


def parse_loose(txt):
    body = FENCE.sub("", txt).strip()
    if not body.startswith("{"):
        i, j = body.find("{"), body.rfind("}")
        if i == -1 or j <= i:
            raise ValueError(f"no JSON in {txt[:120]!r}")
        body = body[i:j + 1]
    return json.loads(body)


PROMPT = """\
You are adversarially reviewing an item in a retrieval benchmark over the
Michigan Court Rules. You did NOT write this item. It will be shown to the
Michigan court system, so a wrong gold label is worse than a rejected item.

THE QUESTION
  {query}

THE PROVISION IT IS LABELLED TO
  citation: {citation}
  rule    : {rule_title}

  text:
{text}

DECIDE, honestly and independently:

1. answerable -- can a competent reader answer that question from THIS
   provision alone? If the provision only touches the subject, or answers a
   neighbouring question, say false.

2. better_answer_exists -- is there some OTHER Michigan Court Rule provision
   that answers the question as well or better? If so, name it as an MCR
   citation. This matters: an unmarked co-valid answer scores a correct
   retrieval as a failure. Name only citations you are confident exist.

3. realistic -- would a judge, clerk or attorney plausibly ask this, in these
   words? Reject stilted textbook phrasing and questions that are obviously
   reverse-engineered from the text.

4. reveals_answer -- does the question quote or closely paraphrase the
   provision's own distinctive wording, such that it could be found by string
   matching without understanding anything?

RETURN EXACTLY THIS JSON SHAPE:

{{
  "answerable": true | false,
  "better_answer_exists": ["<MCR citation>", "..."],
  "realistic": true | false,
  "reveals_answer": true | false,
  "verdict": "accept" | "revise" | "reject",
  "note": "<one sentence; required when verdict is not accept>"
}}

Output the JSON object and nothing else -- no prose, no code fence."""


def call(model, prompt, key, attempts=4, timeout=240):
    body = {"model": model, "prompt": prompt, "stream": False,
            "think": False, "options": {"temperature": 0.2}}
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(
                URL, json.dumps(body).encode(),
                {"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                d = json.load(r)
            out = parse_loose((d.get("response") or "").strip())
            with LOCK:
                STATS["ok"] += 1
            return out
        except Exception as exc:                       # noqa: BLE001
            last = exc
            with LOCK:
                STATS["retry"] += 1
            time.sleep(min(2 ** i, 15))
    with LOCK:
        STATS["fail"] += 1
    raise RuntimeError(f"validation failed after {attempts}: {last}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queries")
    ap.add_argument("blocks")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    key = KEY_PATH.read_text().strip()
    queries = [json.loads(l) for l in open(args.queries) if l.strip()]
    blocks = [json.loads(l) for l in open(args.blocks) if l.strip()]
    by_cit = {}
    for b in blocks:
        if b.get("citation") and b["citation"] not in by_cit:
            by_cit[b["citation"]] = b
    titles = {b["rule"]: b["text"] for b in blocks if b["kind"] == "rule"}

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for l in open(out_path):
            if l.strip():
                done.add(json.loads(l)["query_id"])
    todo = [q for q in queries if q["query_id"] not in done
            and q["query_type"] != "citation_lookup"]
    if args.limit:
        todo = todo[:args.limit]
    print(f"validating {len(todo)} items with {args.model} "
          f"({len(done)} already done)")

    fh = open(out_path, "a")

    def work(q):
        src = by_cit.get(q["citation"])
        if q["query_type"] == "unanswerable":
            # nothing to check against; the model judged out-of-scope at write
            # time and the human pass will sample these
            rec = dict(q, validation={"skipped": "unanswerable"},
                       echo=0.0, status="draft")
            with LOCK:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            return
        if not src:
            return
        try:
            v = call(args.model, PROMPT.format(
                query=q["query"], citation=q["citation"],
                rule_title=titles.get(q["rule"], ""), text=src["text"]), key)
        except Exception as exc:                       # noqa: BLE001
            print(f"  FAIL {q['query_id']}: {exc}", flush=True)
            return

        qw, pw = content_words(q["query"]), content_words(src["text"])
        echo = len(qw & pw) / len(qw) if qw else 0.0

        extra = [c.strip() for c in v.get("better_answer_exists", [])
                 if c.strip() in by_cit and c.strip() != q["citation"]]
        invented = [c.strip() for c in v.get("better_answer_exists", [])
                    if c.strip() and c.strip() not in by_cit]

        verdict = v.get("verdict", "revise")
        if not v.get("answerable"):
            verdict = "reject"
        elif echo >= ECHO_FAIL or v.get("reveals_answer"):
            verdict = "revise"
        elif not v.get("realistic"):
            verdict = "revise"

        rec = dict(q)
        rec["also_answered_by"] = sorted(set(rec.get("also_answered_by", [])) | set(extra))
        rec["echo"] = round(echo, 3)
        rec["validation"] = {"checker": args.model, **v,
                             "co_gold_added": extra,
                             "co_gold_invented": invented}
        rec["status"] = {"accept": "accepted", "revise": "needs_revision",
                         "reject": "rejected"}.get(verdict, "needs_revision")
        with LOCK:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
            n = STATS["ok"]
            if n and n % 50 == 0:
                print(f"  {n} validated", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(work, todo))
    fh.close()

    recs = [json.loads(l) for l in open(out_path) if l.strip()]
    print(f"\ndone in {(time.time()-t0)/60:.1f} min · {dict(STATS)}")
    print("  status:", dict(collections.Counter(r["status"] for r in recs)))
    ge = [r["echo"] for r in recs if "echo" in r]
    if ge:
        ge.sort()
        print(f"  echo median {ge[len(ge)//2]:.2f} · over {ECHO_FAIL}: "
              f"{sum(1 for e in ge if e >= ECHO_FAIL)}")
    print(f"  co-gold added: {sum(len(r.get('validation',{}).get('co_gold_added',[])) for r in recs)}"
          f" · invented and rejected: {sum(len(r.get('validation',{}).get('co_gold_invented',[])) for r in recs)}")


if __name__ == "__main__":
    sys.exit(main())
