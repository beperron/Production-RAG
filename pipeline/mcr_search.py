#!/usr/bin/env python3
"""Retrieval over the Michigan Court Rules, with grounded cited answers.

    from mcr_search import Engine
    e = Engine()
    e.search("how long to answer a complaint")
    e.answer("can I serve a defendant in a psychiatric facility directly?")

Design follows carolina-policy-search: BM25 + dense, reciprocal-rank fused,
answers drawn only from retrieved passages and declining when they do not
support one. Two things are specific to this corpus.

THE CITATION ROUTER.  Judges and clerks type "MCR 2.116(C)(10)" verbatim, and
dense embedders are weak at exact identifier matching. That query class is
answered by exact lookup over the 11,293 parsed citation strings -- 100%
precision at no inference cost -- and never reaches the vector index. It is a
router, not fusion, so it does not reintroduce the BM25 noise the CPS hybrid
sweep ruled out.

HYBRID, NOT DENSE-ONLY.  The CPS bake-off found dense alone optimal and hybrid
merely noisy, but that was a policy manual. Carolina measured 66% R@1 on
statutes against 89% on policy with the same engine -- statutory text leans
harder on lexical matching, because defined terms and cross-references ARE the
signal. MCR is statute-like, so both are wired and the eval decides.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
# The winning configuration, settled by paired tests over 1,060 queries:
# rule-scoped 256-token chunks, dense only, citation router.
#   R@1 0.660 · R@10 0.947 · R@2048tok 0.944 · citation lookups 1.000
# Everything else measured worse. BM25 fused into dense costs 15.6 points
# (p=7.7e-36); bge-reranker-base costs 8.9 (p=1.3e-16); the citation-path
# prefix and the parent stem are worth 0.000; rule-level dedup costs 12-14.
CHUNKS = ROOT / "3_chunks" / "v_rule256.jsonl"
VECTORS = ROOT / "4_eval" / "cache" / "rule256.v_rule256.3580.npy"
KEY_PATH = pathlib.Path(os.path.expanduser("~/.config/ollama/cloud.key"))

EMBEDDER = os.environ.get("MCR_EMBEDDER", "Qwen/Qwen3-Embedding-4B")
GEN_MODEL = os.environ.get("MCR_GEN_MODEL", "deepseek-v4-flash:0731")
RRF_K = 60
MODE = os.environ.get("MCR_MODE", "dense")

RE_CITE = re.compile(r"\bMCR\s*(\d+\.\d+[A-Za-z]?)((?:\s*\([A-Za-z0-9]{1,4}\))*)",
                     re.I)

# A cross-reference is LOAD-BEARING when the sentence makes the target a
# condition on the source. 690 of the corpus's 1,629 edges read this way, and
# an answer that quotes the source without the target is not merely
# incomplete -- for a court it can be wrong. MCR 3.717 opens "Except as
# specified in MCR 3.718(B), MCR 3.718(D), and MCR 3.720...".
RE_BINDING = re.compile(
    r"(except as (?:otherwise )?(?:provided|specified|set forth)|subject to|"
    r"in accordance with|as (?:provided|defined) in|governed by|pursuant to|"
    r"unless otherwise provided)[^.]{0,80}$", re.I)

SYSTEM = """\
You answer questions about the Michigan Court Rules for judges, court clerks \
and attorneys.

Answer ONLY from the numbered passages provided. Every factual sentence must \
carry an inline citation to the rule it came from, written as the citation \
shown in the passage header, e.g. (MCR 2.116(C)(10)).

If the passages do not answer the question, say so plainly and name what would \
answer it. Do NOT use outside knowledge of Michigan practice, do not guess, and \
do not soften a gap with a general statement. A wrong citation is worse than no \
answer.

Some passages are marked as supplied because an earlier passage makes them a \
condition on itself. Use them: an answer that states a rule without the \
exception it is expressly subject to is wrong, not merely incomplete.

Be brief. Judges are busy. Lead with the answer, then the condition or \
exception if one matters."""


def _norm_cite(rule, subs):
    subs = re.sub(r"\s+", "", subs or "")
    return f"MCR {rule}{subs}"


def resolve_cite(rule, subs, known):
    """Trim a greedily-matched citation back to one the corpus recognises.

    The marker pattern cannot tell "MCR 3.205(D)(3)" followed by a
    parenthetical clause from "MCR 3.205(D)(3)(4)", so it over-reaches and the
    result was being counted as a fabricated citation. The corpus is the
    authority: drop trailing groups until the citation resolves.
    """
    groups = re.findall(r"\([A-Za-z0-9]{1,4}\)", subs or "")
    for n in range(len(groups), -1, -1):
        cand = f"MCR {rule}{''.join(groups[:n])}"
        if cand in known:
            return cand
    return f"MCR {rule}{''.join(groups)}"


class Engine:
    def __init__(self, chunks_path=CHUNKS, embedder=EMBEDDER, quiet=False):
        self.quiet = quiet
        self.chunks = [json.loads(l) for l in open(chunks_path) if l.strip()]
        self.by_citation = {}
        for i, c in enumerate(self.chunks):
            for cit in c["citations"]:
                self.by_citation.setdefault(cit, i)
        self._bm25 = None
        self._model = None
        self._vecs = None
        self.embedder_name = embedder

    # -- lexical ----------------------------------------------------------
    @property
    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            corpus = [re.findall(r"[a-z0-9]+", c["embed_text"].lower())
                      for c in self.chunks]
            self._bm25 = BM25Okapi(corpus)
        return self._bm25

    # -- dense ------------------------------------------------------------
    @property
    def model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self.embedder_name, device="mps",
                model_kwargs={"torch_dtype": torch.float16})
            self._model.max_seq_length = 1536
        return self._model

    @property
    def vecs(self):
        if self._vecs is None:
            tag = VECTORS if VECTORS.exists() else VECTORS.with_name(
                f"vectors.{self.embedder_name.split('/')[-1]}.npy")
            if tag.exists():
                self._vecs = np.load(tag)
            else:
                if not self.quiet:
                    print(f"embedding {len(self.chunks):,} chunks with "
                          f"{self.embedder_name} (one time)...", flush=True)
                v = self.model.encode([c["embed_text"] for c in self.chunks],
                                      batch_size=32, convert_to_numpy=True,
                                      normalize_embeddings=True,
                                      show_progress_bar=not self.quiet)
                np.save(tag, v)
                self._vecs = v
        return self._vecs

    # -- routing ----------------------------------------------------------
    def route_citation(self, query):
        """Exact-citation queries never reach the vector index."""
        m = RE_CITE.search(query)
        if not m:
            return None
        want = _norm_cite(m.group(1), m.group(2))
        if want in self.by_citation:
            return want, self.by_citation[want]
        # a bare rule number should land on the top of that rule
        bare = f"MCR {m.group(1)}"
        if bare in self.by_citation:
            return bare, self.by_citation[bare]
        for cit, i in self.by_citation.items():
            if cit.startswith(bare + "("):
                return cit, i
        return None

    # -- search -----------------------------------------------------------
    def search(self, query, k=8, mode=MODE):
        routed = self.route_citation(query) if mode != "dense_only" else None
        hits, seen = [], set()
        if routed:
            cit, idx = routed
            hits.append(self._hit(idx, 1.0, "citation-router", cit))
            seen.add(idx)

        ranks = []
        if mode in ("hybrid", "bm25"):
            s = self.bm25.get_scores(re.findall(r"[a-z0-9]+", query.lower()))
            ranks.append(list(np.argsort(-s)[:60]))
        if mode in ("hybrid", "dense", "dense_only"):
            q = self.model.encode([query], convert_to_numpy=True,
                                  normalize_embeddings=True,
                                  prompt_name="query")
            s = self.vecs @ q[0]
            ranks.append(list(np.argsort(-s)[:60]))

        fused = {}
        for lst in ranks:
            for r, i in enumerate(lst):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
        for i, sc in sorted(fused.items(), key=lambda kv: -kv[1]):
            if i in seen:
                continue
            hits.append(self._hit(i, sc, mode, None))
            seen.add(i)
            if len(hits) >= k:
                break
        return hits[:k]

    def _hit(self, idx, score, how, routed_citation):
        c = self.chunks[idx]
        return {"chunk_id": c["chunk_id"], "score": round(float(score), 5),
                "how": how, "citation": routed_citation or c["citation_first"],
                "citations": c["citations"], "rule": c["rule"],
                "rule_title": c["rule_title"], "heading_path": c["heading_path"],
                "pages": c["pages"], "text": c["text"],
                "n_tokens": c["n_tokens"]}

    # -- graph ------------------------------------------------------------
    def binding_refs(self, text, limit=3):
        """Citations this passage makes a condition on itself."""
        out = []
        for m in RE_CITE.finditer(text):
            before = text[:m.start()]
            if not RE_BINDING.search(before[-140:]):
                continue
            cit = _norm_cite(m.group(1), m.group(2))
            idx = self.by_citation.get(cit)
            if idx is None:
                bare = f"MCR {m.group(1)}"
                idx = self.by_citation.get(bare)
                cit = bare if idx is not None else cit
            if idx is not None and cit not in [o[0] for o in out]:
                out.append((cit, idx))
            if len(out) >= limit:
                break
        return out

    def expand(self, hits, limit=3):
        """Attach the provisions the retrieved passages depend on."""
        have = {h["chunk_id"] for h in hits}
        extra = []
        for h in hits:
            for cit, idx in self.binding_refs(h["text"], limit):
                c = self.chunks[idx]
                if c["chunk_id"] in have:
                    continue
                have.add(c["chunk_id"])
                e = self._hit(idx, 0.0, "cross-reference", cit)
                e["because_of"] = h["citation"]
                extra.append(e)
                if len(extra) >= limit:
                    return extra
        return extra

    # -- generation -------------------------------------------------------
    def answer(self, question, k=8, mode=MODE, timeout=180, expand=True):
        hits = self.search(question, k=k, mode=mode)
        if expand:
            hits = hits + self.expand(hits)
        def fmt(n, h):
            # Say WHY a cross-referenced passage is present. Unlabelled, the
            # model treats it as another search result and reports the
            # exception as "not detailed in the provided passages" while
            # holding the text that details it.
            why = (f"    [supplied because {h['because_of']} makes this a "
                   f"condition on itself]\n" if h.get("because_of") else "")
            return (f"[{n+1}] {h['heading_path']}\n"
                    f"    citation: {h['citation']}  "
                    f"(page {h['pages'][0] + 1 - 18})\n{why}"
                    f"{h['text'][:2400]}")
        passages = "\n\n".join(fmt(n, h) for n, h in enumerate(hits))
        user = (f"QUESTION\n{question}\n\n"
                f"PASSAGES FROM THE MICHIGAN COURT RULES\n{passages}\n\n"
                f"Answer the question using only these passages, with inline "
                f"citations. If they do not answer it, say so.")
        try:
            text = self._generate(SYSTEM, user, timeout)
        except Exception as exc:                        # noqa: BLE001
            text = f"(generation unavailable: {exc})"
        return {"question": question, "answer": text, "hits": hits,
                "model": GEN_MODEL}

    def _generate(self, system, user, timeout):
        key = KEY_PATH.read_text().strip()
        body = {"model": GEN_MODEL, "prompt": user, "system": system,
                "stream": False, "think": False,
                "options": {"temperature": 0.1}}
        req = urllib.request.Request(
            "https://ollama.com/api/generate", json.dumps(body).encode(),
            {"Content-Type": "application/json",
             "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return (json.load(r).get("response") or "").strip()


if __name__ == "__main__":
    import sys
    e = Engine()
    q = " ".join(sys.argv[1:]) or "how long does a defendant have to answer a complaint"
    if q.startswith("--answer "):
        r = e.answer(q[9:])
        print(r["answer"], "\n")
        for h in r["hits"]:
            print(f"  [{h['how']:<16}] {h['citation']:<24} {h['rule_title'][:50]}")
    else:
        for h in e.search(q):
            print(f"  [{h['how']:<16}] {h['score']:.4f}  {h['citation']:<24} "
                  f"{h['rule_title'][:52]}")
