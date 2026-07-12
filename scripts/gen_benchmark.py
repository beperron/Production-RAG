#!/usr/bin/env python3
"""Generate a large labeled retrieval benchmark with GLM-5.2 (Ollama Cloud).

For sampled substantive passages, GLM writes ONE realistic question a practitioner
would ask that the passage answers (without quoting it). The passage's source is
the known-correct label. Output: evals/law_search/gold_big.json.

Sampling is stratified by gold key so coverage is broad, not clustered.
Env: OLLAMA_KEY.  Usage: python scripts/gen_benchmark.py [--per 250]
"""
import argparse, json, os, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://ollama.com/api/chat"; MODEL = "glm-5.2"
SEC = re.compile(r"\d+[A-Z]?-\d+(?:\.\d+)?")
ROLE = {"legal-authorities": "attorney or guardian ad litem",
        "nc-child-welfare": "county child-welfare caseworker or supervisor"}
KIND = {"legal-authorities": "North Carolina statute or regulation",
        "nc-child-welfare": "North Carolina DHHS child-welfare policy"}

PROMPT = """Below is a passage from {kind}.

<passage>
{passage}
</passage>

Write ONE realistic, self-contained question that a {role} might ask, which this
passage directly answers. Do NOT quote the passage or reuse its distinctive
wording — phrase it naturally, as a real person would type it into a search box.
Do not mention section numbers. Output ONLY the question."""


def glm(prompt, key):
    r = requests.post(URL, timeout=90, headers={"Authorization": f"Bearer {key}"},
                      json={"model": MODEL, "stream": False, "options": {"temperature": 0.4},
                            "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return (r.json().get("message") or {}).get("content", "").strip()


def gold_key(coll, c):
    hp = c.get("heading_path") or []
    m = SEC.search(" ".join(hp)) if coll == "legal-authorities" else None
    if m:
        return ("section", m.group(0))
    return ("doc", c["doc_id"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per", type=int, default=250)
    ap.add_argument("--workers", type=int, default=20); a = ap.parse_args()
    key = os.environ.get("OLLAMA_KEY", "").strip()
    if not key: sys.exit("set OLLAMA_KEY")
    tasks = []
    for coll in ["legal-authorities", "nc-child-welfare"]:
        idx = json.loads((ROOT / "knowledge-base" / coll / "docindex.json").read_text())
        subst = [c for c in idx["chunks"] if len(c.get("text", "")) >= 450]
        by_key = {}
        for c in subst:
            by_key.setdefault(gold_key(coll, c), []).append(c)
        keys = list(by_key)
        # round-robin one chunk per key until we hit the target (broad coverage)
        picks, i = [], 0
        while len(picks) < a.per and any(by_key.values()):
            k = keys[i % len(keys)]; i += 1
            if by_key[k]:
                picks.append((coll, k, by_key[k].pop()))
            if i > a.per * 20: break
        tasks += picks[:a.per]
    print(f"generating {len(tasks)} questions…", flush=True)
    out, lock, n = [], threading.Lock(), [0]

    def work(t):
        coll, (kind, keyv), c = t
        try:
            q = glm(PROMPT.format(kind=KIND[coll], role=ROLE[coll], passage=c["text"][:2500]), key)
        except Exception:
            q = ""
        q = q.replace("\n", " ").strip().strip('"')
        if 8 <= len(q) <= 240 and "?" in q:
            with lock:
                out.append({"q": q, "collection": coll, "gold_kind": kind, "gold_key": keyv})
        with lock:
            n[0] += 1
            if n[0] % 100 == 0: print(f"  {n[0]}/{len(tasks)}", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, tasks))
    p = ROOT / "evals/law_search/gold_big.json"
    p.write_text(json.dumps(out, indent=1))
    bycoll = {}
    for g in out: bycoll[g["collection"]] = bycoll.get(g["collection"], 0) + 1
    print(f"DONE — {len(out)} questions -> {p}  ({bycoll})", flush=True)


if __name__ == "__main__":
    main()
