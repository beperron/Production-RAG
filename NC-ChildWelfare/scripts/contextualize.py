#!/usr/bin/env python3
"""Anthropic Contextual Retrieval — step 2: generate a per-chunk context blurb.

For each chunk, GLM-5.2 (Ollama Cloud) writes a short context situating the chunk
within its document, following Anthropic's prompt. The "document" scope passed is
the chunk's statutory section / heading-section (whole-chapter is infeasible
without prompt caching, and the section is the semantically relevant unit).

Resumable JSONL output: {chunk_id, context}. Threaded.

Env: OLLAMA_KEY
Usage: python scripts/contextualize.py <collection> [--workers 16] [--limit N]
"""
import argparse, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://ollama.com/api/chat"
MODEL = "glm-5.2"
SEC_CAP = 6000  # cap the section context passed to the model

PROMPT = """<document>
Document: {title}
Statutory section: {head}

{section}
</document>
Here is a chunk taken from the section above:
<chunk>
{chunk}
</chunk>
Give a short, succinct context (ONE sentence, <=40 words) situating this chunk \
within the document — its chapter/section, subsection number if any, and the \
specific topic — to improve search retrieval. Answer with ONLY the context."""


def glm(prompt, key, timeout=90):
    r = requests.post(URL, timeout=timeout,
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                      json={"model": MODEL, "stream": False, "options": {"temperature": 0},
                            "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return (r.json().get("message") or {}).get("content", "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("collection")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    key = os.environ.get("OLLAMA_KEY", "").strip()
    if not key:
        sys.exit("set OLLAMA_KEY")
    idx = json.loads((ROOT / "knowledge-base" / a.collection / "docindex.json").read_text())
    title = {d["doc_id"]: d.get("title", "") for d in idx["documents"]}
    chunks = idx["chunks"][:a.limit] if a.limit else idx["chunks"]
    # per-doc ordinal-sorted chunks, so each chunk can see its local neighborhood
    by_doc = {}
    for c in idx["chunks"]:
        by_doc.setdefault(c["doc_id"], []).append(c)
    for lst in by_doc.values():
        lst.sort(key=lambda c: c.get("ordinal", 0))

    def local_context(c):
        lst = by_doc[c["doc_id"]]
        i = next((j for j, x in enumerate(lst) if x["chunk_id"] == c["chunk_id"]), 0)
        window = lst[max(0, i - 4): i + 5]
        return " ".join(x["text"] for x in window)[:SEC_CAP]

    out_path = ROOT / "knowledge-base" / a.collection / "context.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try: done.add(json.loads(line)["chunk_id"])
            except Exception: pass
    todo = [c for c in chunks if c["chunk_id"] not in done]
    print(f"{a.collection}: {len(chunks)} chunks, {len(done)} done, {len(todo)} to do", flush=True)
    lock = threading.Lock()
    fh = out_path.open("a")
    n = [0]

    def work(c):
        head = " > ".join(c.get("heading_path") or []) or "(document body)"
        p = PROMPT.format(title=title.get(c["doc_id"], ""), head=head,
                          section=local_context(c), chunk=c["text"])
        try:
            ctx = glm(p, key)
        except Exception as e:
            ctx = ""
        ctx = ctx.replace("\n", " ").strip().strip('"')[:400]
        with lock:
            fh.write(json.dumps({"chunk_id": c["chunk_id"], "context": ctx}) + "\n"); fh.flush()
            n[0] += 1
            if n[0] % 100 == 0:
                print(f"  {n[0]}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, todo))
    fh.close()
    print(f"DONE — contexts in {out_path}", flush=True)


if __name__ == "__main__":
    main()
