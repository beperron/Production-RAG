#!/usr/bin/env python3
"""Faithfulness judge: does each grounded answer's claims follow from its sources?

Samples questions, gets the live grounded answer + its sources, and asks GLM-5.2
to judge whether every claim in the answer is supported by the retrieved sources.
Reports grounded / refused / faithful rates.

Env: OLLAMA_KEY.  Usage: python scripts/eval_faithfulness.py [--n 40]
"""
import argparse, json, os, re, sys, threading, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://nc-policy.vercel.app"
OLLAMA = "https://ollama.com/api/chat"

JUDGE = """You are auditing a legal-research assistant for faithfulness.

QUESTION: {q}

ANSWER GIVEN:
{answer}

SOURCE PASSAGES the answer was supposed to rely on:
{sources}

Is EVERY factual claim in the ANSWER directly supported by the SOURCE PASSAGES
(no outside facts, no contradictions)? Reply on one line: "YES" or "NO" followed
by a short reason (<=10 words)."""


def get_answer(q):
    url = f"{BASE}/api/answer?" + urllib.parse.urlencode({"q": q})
    raw = urllib.request.urlopen(url, timeout=75).read().decode()
    nl = raw.find("\n")
    hits = json.loads(raw[:nl]).get("hits", []) if nl > 0 else []
    return raw[nl + 1:], hits


def judge(q, answer, hits, key):
    src = "\n\n".join(f"[{i+1}] {(h.get('section') or h.get('title') or '')}: {(h.get('content') or '')[:900]}"
                      for i, h in enumerate(hits[:6]))
    r = requests.post(OLLAMA, timeout=90, headers={"Authorization": f"Bearer {key}"},
                      json={"model": "glm-5.2", "stream": False, "options": {"temperature": 0},
                            "messages": [{"role": "user", "content": JUDGE.format(q=q, answer=answer[:2500], sources=src)}]})
    return (r.json().get("message") or {}).get("content", "").strip()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40); a = ap.parse_args()
    key = os.environ["OLLAMA_KEY"].strip()
    gold = json.loads((ROOT / "evals/law_search/gold_big.json").read_text())
    law = [g for g in gold if g["collection"] == "legal-authorities"][:a.n // 2]
    pol = [g for g in gold if g["collection"] == "nc-child-welfare"][:a.n // 2]
    sample = law + pol
    res = {"grounded": 0, "refused": 0, "faithful": 0, "n": 0}
    lock = threading.Lock()

    def work(g):
        try:
            answer, hits = get_answer(g["q"])
        except Exception:
            return
        grounded = bool(re.search(r"\[\d+\]", answer)) and "do not contain enough" not in answer
        refused = "do not contain enough" in answer
        faithful = False
        if grounded:
            try:
                v = judge(g["q"], answer, hits, key)
                faithful = v.strip().upper().startswith("YES")
            except Exception:
                v = "ERR"
        with lock:
            res["n"] += 1; res["grounded"] += grounded; res["refused"] += refused; res["faithful"] += faithful
            print(f"  {'G' if grounded else ('R' if refused else '.')} {'F' if faithful else ' '}  {g['q'][:60]}", flush=True)

    with ThreadPoolExecutor(max_workers=5) as ex:
        list(ex.map(work, sample))
    n = res["n"]
    print(f"\n=== FAITHFULNESS (n={n}) ===")
    print(f"  grounded (cited answer):     {100*res['grounded']//n}%")
    print(f"  refused (declined):          {100*res['refused']//n}%")
    print(f"  faithful | grounded:         {100*res['faithful']//max(1,res['grounded'])}%  (GLM judge)")


if __name__ == "__main__":
    main()
