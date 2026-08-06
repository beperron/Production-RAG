#!/usr/bin/env python3
"""Score the ANSWER layer, not retrieval.

    eval_answers.py --n 150 --out 4_eval/answers.jsonl

Retrieval numbers do not predict answer quality, and for a corpus going in
front of a court the failure that matters most is a confident wrong citation.
Everything measured before this point was retrieval; nothing here has been
looked at.

Four metrics, and three of them need no judge at all
----------------------------------------------------
citation_valid    every "MCR x.xxx" the answer emits must EXIST in the corpus.
                  A citation that resolves to nothing is a fabrication, and it
                  is checkable against the 11,293 parsed citations.
citation_grounded every cited provision must have been among the passages the
                  model was shown. Citing something real but unretrieved means
                  the model answered from training data and dressed it in a
                  citation -- the most dangerous failure mode here, because it
                  looks identical to a correct answer.
cites_gold        does the answer cite the provision the question was written
                  from (or a co-gold)?
refusal           on the 32 unanswerable queries the system must decline; on
                  answerable ones it must not. Both directions are errors.

Faithfulness -- is every claim actually supported by the passages -- does need
a judge, and it runs on GLM 5.2 rather than the generator's own family so the
model is not grading its own work.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import random
import re
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mcr_search import Engine, RE_CITE, resolve_cite    # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
KEY_PATH = pathlib.Path(os.path.expanduser("~/.config/ollama/cloud.key"))
JUDGE = "mistral-large-3:675b"   # neutral family: not glm, deepseek, nemotron, or qwen
LOCK = threading.Lock()

REFUSAL = re.compile(
    r"do(es)? not (answer|address|contain|provide)|not answered|no provision|"
    r"cannot be answered|do not (support|establish)|passages do not|"
    r"is not (governed|addressed) by", re.I)

JUDGE_PROMPT = """You are auditing an answer generated from the Michigan Court
Rules for a benchmark. Decide only what the evidence supports.

QUESTION
{question}

PASSAGES THE MODEL WAS GIVEN
{passages}

THE ANSWER
{answer}

Judge:
1. supported -- is EVERY factual claim in the answer supported by the passages
   above? false if any claim goes beyond them, however plausible it sounds.
2. citations_match -- does each inline citation actually correspond to the
   passage that supports the claim it is attached to?
3. overreach -- does the answer state something as settled that the passages
   only partly establish?

RETURN EXACTLY:
{{"supported": true|false, "citations_match": true|false,
  "overreach": true|false, "note": "<one sentence if anything is false>"}}

JSON only, no fence."""


def judge(prompt, timeout=180, attempts=3):
    key = KEY_PATH.read_text().strip()
    body = {"model": JUDGE, "prompt": prompt, "stream": False, "think": False,
            "options": {"temperature": 0.1}}
    for i in range(attempts):
        try:
            req = urllib.request.Request(
                "https://ollama.com/api/generate", json.dumps(body).encode(),
                {"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                t = (json.load(r).get("response") or "").strip()
            t = re.sub(r"^\s*```(?:json)?|```\s*$", "", t, flags=re.M).strip()
            i0, j0 = t.find("{"), t.rfind("}")
            return json.loads(t[i0:j0 + 1])
        except Exception:                               # noqa: BLE001
            time.sleep(min(2 ** i, 8))
    return {}


def cites_in(text, known):
    out = []
    for m in RE_CITE.finditer(text):
        out.append(resolve_cite(m.group(1), m.group(2), known))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="2_eval/mcr_eval_v1.jsonl")
    ap.add_argument("--blocks", default="1_parsed/blocks.jsonl")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge", default=None)
    ap.add_argument("--no-expand", action="store_true")
    ap.add_argument("--out", default="4_eval/answers.jsonl")
    ap.add_argument("--seed", type=int, default=20260805)
    args = ap.parse_args()
    global JUDGE
    if args.judge:
        JUDGE = args.judge

    qs = [json.loads(l) for l in open(ROOT / args.eval) if l.strip()]
    valid = {json.loads(l)["citation"] for l in open(ROOT / args.blocks)
             if l.strip() and json.loads(l).get("citation")}

    rng = random.Random(args.seed)
    answerable = [q for q in qs if q["query_type"] not in
                  ("unanswerable", "citation_lookup")]
    negatives = [q for q in qs if q["query_type"] == "unanswerable"]
    rng.shuffle(answerable)
    sample = answerable[:max(0, args.n - len(negatives))] + negatives
    rng.shuffle(sample)

    eng = Engine(quiet=True)
    _ = eng.vecs
    # Pre-warm the MODEL, not just the vectors. The vector property hits cache
    # and returns without ever touching the encoder, so the first query encode
    # happens inside the thread pool and N workers race to load a 4B model at
    # once. Loading it here makes that a single serial cost.
    _ = eng.model.encode(["warm"], normalize_embeddings=True,
                         prompt_name="query", show_progress_bar=False)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for l in open(out_path):
            if l.strip():
                done.add(json.loads(l)["query_id"])
    todo = [q for q in sample if q["query_id"] not in done]
    print(f"answering {len(todo)} queries "
          f"({len(negatives)} of them unanswerable) · expand="
          f"{not args.no_expand} · judge={'off' if args.no_judge else JUDGE}",
          flush=True)

    fh = open(out_path, "a")
    n_done = [0]

    def work(q):
        t_ans = time.time()
        try:
            r = eng.answer(q["query"], k=8, expand=not args.no_expand)
        except Exception as exc:                        # noqa: BLE001
            print(f"  FAIL {q['query_id']}: {exc}", flush=True)
            return
        ans = r["answer"]
        shown = {c for h in r["hits"] for c in h["citations"]}
        emitted = cites_in(ans, valid)
        gold = set(q["gold"]) | set(q.get("also_answered_by") or [])

        rec = {
            "latency_ms": int((time.time() - t_ans) * 1000),
            "generator": os.environ.get("MCR_GEN_MODEL", "deepseek-v4-flash:0731"),
            "query_id": q["query_id"], "query": q["query"],
            "query_type": q["query_type"], "gold": q["gold"],
            "answer": ans,
            "n_citations": len(emitted),
            "citation_valid": (sum(1 for c in emitted if c in valid)
                               / len(emitted)) if emitted else None,
            "citation_grounded": (sum(1 for c in emitted if c in shown)
                                  / len(emitted)) if emitted else None,
            "invalid_citations": [c for c in emitted if c not in valid],
            "ungrounded_citations": [c for c in emitted
                                     if c in valid and c not in shown],
            "cites_gold": bool(set(emitted) & gold) if gold else None,
            "gold_retrieved": bool(shown & gold) if gold else None,
            "refused": bool(REFUSAL.search(ans)),
        }
        if not args.no_judge and not rec["refused"]:
            passages = "\n\n".join(
                f"[{n+1}] {h['citation']}\n{h['text'][:1400]}"
                for n, h in enumerate(r["hits"]))
            rec["judge"] = judge(JUDGE_PROMPT.format(
                question=q["query"], passages=passages, answer=ans))
        with LOCK:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n_done[0] += 1
            if n_done[0] % 20 == 0:
                print(f"  {n_done[0]}/{len(todo)}", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))
    fh.close()

    recs = [json.loads(l) for l in open(out_path) if l.strip()]
    ansd = [r for r in recs if r["query_type"] != "unanswerable"]
    negs = [r for r in recs if r["query_type"] == "unanswerable"]
    f = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
    print(f"\n=== answer layer, n={len(recs)} · {(time.time()-t0)/60:.1f} min ===")
    print(f"  citations emitted per answer : "
          f"{f([r['n_citations'] for r in ansd]):.2f}")
    print(f"  citation VALID (exists)      : "
          f"{f([r['citation_valid'] for r in ansd if r['citation_valid'] is not None]):.4f}")
    print(f"  citation GROUNDED (retrieved): "
          f"{f([r['citation_grounded'] for r in ansd if r['citation_grounded'] is not None]):.4f}")
    print(f"  answer cites the gold        : "
          f"{f([1 if r['cites_gold'] else 0 for r in ansd]):.4f}")
    print(f"  gold was retrieved at all    : "
          f"{f([1 if r['gold_retrieved'] else 0 for r in ansd]):.4f}")
    print(f"  FALSE refusal (answerable)   : "
          f"{f([1 if r['refused'] else 0 for r in ansd]):.4f}")
    if negs:
        print(f"  correct refusal (negatives)  : "
              f"{f([1 if r['refused'] else 0 for r in negs]):.4f}  (n={len(negs)})")
    j = [r["judge"] for r in recs if r.get("judge")]
    if j:
        print(f"  judged supported             : "
              f"{f([1 if x.get('supported') else 0 for x in j]):.4f}  (n={len(j)})")
        print(f"  judged citations_match       : "
              f"{f([1 if x.get('citations_match') else 0 for x in j]):.4f}")
        print(f"  judged overreach             : "
              f"{f([1 if x.get('overreach') else 0 for x in j]):.4f}")
    bad = [c for r in ansd for c in r["invalid_citations"]]
    if bad:
        print(f"\n  FABRICATED citations: {len(bad)} — "
              f"{collections.Counter(bad).most_common(6)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
