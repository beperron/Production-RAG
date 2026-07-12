"""Public North-Carolina law / policy semantic search — PUBLIC DATA ONLY.

Loads the public knowledge-base indexes (``legal-authorities`` = NC Admin Code +
General Statutes; ``nc-child-welfare`` = policy / forms / guidance), runs hybrid
BM25F + GTE-base semantic retrieval, and optional grounded, cited RAG answers.

HARD SAFETY INVARIANT: this engine refuses any confidential / ``private/``
workspace. It exists so the NC-law search surface can be built on the retained
RAG engine **without ever touching any confidential / private workspace**.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from . import rag
from .config import embedder_from_env, generator_from_env
from .pipeline.docindex import DocIndex

# The only workspaces this surface may ever open. Public, non-confidential.
PUBLIC_COLLECTIONS: dict[str, str] = {
    "legal-authorities": "Laws — NC Admin Code 10A·70 + General Statutes",
    "nc-child-welfare": "Policies, forms & guidance",
}
_KB_ROOT = Path("knowledge-base")

# ---------------------------------------------------------------------------
# Optional cloud enhancement stack — PUBLIC data only. Egress is acceptable here
# because this surface never touches confidential material. All three stages are
# env-gated and degrade to the local hybrid engine when a key/service is absent.
# ---------------------------------------------------------------------------
_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_OPENROUTER_MODEL = os.environ.get("OPENROUTER_QUERY_MODEL", "qwen/qwen3-8b")
_JINA_URL = "https://api.jina.ai/v1/rerank"
_JINA_MODEL = os.environ.get("JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual")
_RRF_K = int(os.environ.get("LAWSEARCH_RRF_K", "60"))
_POOL = int(os.environ.get("LAWSEARCH_POOL", "40"))
# Weight of the (normalized) Jina score vs the 1/(rrf_rank) prior when blending
# the reranker into the fused order. 0.5 measured best (R@3=96%, R@1 preserved).
_RERANK_BLEND = float(os.environ.get("LAWSEARCH_RERANK_BLEND", "0.5"))


def _rewrite_query(query: str, timeout: float = 12.0) -> str:
    """Expand/clarify the query with OpenRouter Qwen-4B for better recall.
    Returns the original query unchanged if no key or the call fails."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        return query
    import requests
    sysp = ("You rewrite a legal-research query about North Carolina child-welfare "
            "law and policy into a single, keyword-rich search query (statutes, "
            "regulations, defined terms). Reply with ONLY the rewritten query.")
    try:
        r = requests.post(_OPENROUTER_URL, timeout=timeout,
                          headers={"Authorization": f"Bearer {key}"},
                          json={"model": _OPENROUTER_MODEL, "temperature": 0,
                                "max_tokens": 120,
                                "messages": [{"role": "system", "content": sysp},
                                             {"role": "user", "content": query}]})
        r.raise_for_status()
        out = (r.json()["choices"][0]["message"].get("content") or "").strip()
        return out or query
    except Exception:
        return query


def _jina_reranker():
    """A pipeline Reranker backed by the Jina API, for grounding the generative
    answer on the reranked pool (so the right statute reaches the model). None
    without a key."""
    if not os.environ.get("JINA_API_KEY", "").strip():
        return None
    from .pipeline.reranker import Reranker

    class _JinaRR(Reranker):
        name = "jina"

        def scores(self, query: str, documents: list[str]) -> list[float]:
            key = os.environ.get("JINA_API_KEY", "").strip()
            if not key or not documents:
                return [0.0] * len(documents)
            import requests
            try:
                r = requests.post(_JINA_URL, timeout=30,
                                  headers={"Authorization": f"Bearer {key}"},
                                  json={"model": _JINA_MODEL, "query": query,
                                        "documents": documents, "top_n": len(documents)})
                r.raise_for_status()
                sc = [0.0] * len(documents)
                for it in r.json().get("results", []):
                    sc[it["index"]] = float(it.get("relevance_score", 0.0))
                return sc
            except Exception:
                return [0.0] * len(documents)

    return _JinaRR()


_OPENROUTER_GEN_MODEL = os.environ.get("OPENROUTER_GEN_MODEL", "deepseek/deepseek-v4-flash")


def _openrouter_generator():
    """A ChatGenerator (``.complete(system, user)``) backed by an OpenRouter
    model (default deepseek-v4-flash) for grounded answers over PUBLIC law. None
    when no key — the caller falls back to the local Qwen generator."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        return None
    model = _OPENROUTER_GEN_MODEL

    class _Gen:
        def __init__(self):
            self.model = model

        def complete(self, system: str, user: str) -> str:
            import requests
            r = requests.post(_OPENROUTER_URL, timeout=90,
                              headers={"Authorization": f"Bearer {key}"},
                              json={"model": model, "temperature": 0, "max_tokens": 1200,
                                    "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": user}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"].get("content", "") or ""

    return _Gen()


def _rrf_fuse(lists, K: int = _RRF_K):
    """Reciprocal-rank fusion across ranked candidate lists. Each list element is
    ``(collection, SearchHit)``; a hit's contribution is ``1/(K + rank)``. Within
    each list the engine already RRF-fused BM25F + dense; this fuses the
    per-collection hybrid rankings into one ordering."""
    scores: dict = {}
    items: dict = {}
    for lst in lists:
        for rank, (name, h) in enumerate(lst):
            chunk = getattr(h, "chunk", None)
            key = (name, getattr(chunk, "chunk_id", None) or id(h))
            scores[key] = scores.get(key, 0.0) + 1.0 / (K + rank + 1)
            items[key] = (name, h)
    return [items[k] for k in sorted(scores, key=lambda k: -scores[k])]


def _jina_rerank(query: str, docs: list[str], top_n: int, timeout: float = 20.0):
    """Return reranked indices (into ``docs``) via the Jina reranker, or None to
    signal 'no rerank' (missing key / failure) so the caller keeps RRF order."""
    key = os.environ.get("JINA_API_KEY", "").strip()
    if not key or not docs:
        return None
    import requests
    try:
        r = requests.post(_JINA_URL, timeout=timeout,
                          headers={"Authorization": f"Bearer {key}"},
                          json={"model": _JINA_MODEL, "query": query,
                                "documents": docs, "top_n": min(top_n, len(docs))})
        r.raise_for_status()
        return [it["index"] for it in r.json().get("results", [])]
    except Exception:
        return None


def _assert_public(path: Path) -> None:
    """Refuse anything under a private/confidential subtree — belt and braces."""
    parts = {p.lower() for p in path.resolve().parts}
    if "private" in parts or "case-file" in parts:
        raise PermissionError(
            f"law search is PUBLIC only — refusing confidential workspace {path}"
        )


@dataclass
class Hit:
    collection: str
    score: float
    title: str
    section: str
    snippet: str
    source_url: str
    source_domain: str
    sha256: str          # integrity hash (tamper-evidence) for the source doc
    doc_id: str
    chunk_id: str
    citation: str        # full source-traceable citation line
    lane: str            # how it was found: bm25 / dense / hybrid / regex
    page_span: str       # 'p. 14' / 'pp. 14–15' / ''


class LawSearch:
    """Public-only semantic search over the NC law/policy knowledge bases."""

    def __init__(self, collections: list[str] | None = None):
        self.embedder = embedder_from_env()
        self.indexes: dict[str, DocIndex] = {}
        self.labels = dict(PUBLIC_COLLECTIONS)
        for name in (collections or list(PUBLIC_COLLECTIONS)):
            if name not in PUBLIC_COLLECTIONS:
                raise PermissionError(f"{name!r} is not a public collection")
            idx_path = _KB_ROOT / name / "docindex.json"
            _assert_public(idx_path)
            if idx_path.is_file():
                self.indexes[name] = DocIndex.load(idx_path, embedder=self.embedder)

    def collections(self) -> list[str]:
        return list(self.indexes)

    def _to_hit(self, name: str, h) -> Hit:
        chunk = getattr(h, "chunk", None)
        sha = (getattr(chunk, "source_sha256", "") or
               getattr(chunk, "content_sha256", "")) if chunk else ""
        return Hit(
            collection=name,
            score=float(getattr(h, "score", 0.0) or 0.0),
            title=getattr(h, "title", "") or "",
            section=(h.statutory_section() or getattr(h, "section", "") or ""),
            snippet=(getattr(h, "snippet", "") or "")[:600],
            source_url=getattr(h, "source_url", "") or "",
            source_domain=getattr(h, "source_domain", "") or "",
            sha256=sha,
            doc_id=getattr(h, "doc_id", "") or "",
            chunk_id=getattr(chunk, "chunk_id", "") if chunk else "",
            citation=h.citation(),
            lane=getattr(h, "lane", "") or "",
            page_span=h.page_span(),
        )

    def enhancement_status(self) -> dict[str, bool]:
        return {"query_rewrite": bool(os.environ.get("OPENROUTER_API_KEY", "").strip()),
                "jina_rerank": bool(os.environ.get("JINA_API_KEY", "").strip())}

    def search(self, query: str, k: int = 10, collection: str | None = None) -> list[Hit]:
        # NOTE: query rewrite is deliberately NOT applied to search — measured to
        # HURT (over-expands short keyword queries: R@3 85%->71%). It is used only
        # on the generative-answer path (verbose questions). See docs/law eval.
        names = [collection] if collection in self.indexes else list(self.indexes)
        # 1) per-collection hybrid retrieval (each index already RRF-fuses BM25F+dense)
        lists: list[list[tuple[str, object]]] = []
        for name in names:
            lists.append([(name, h) for h in
                          self.indexes[name].search(query, k=_POOL, dedup=False)])
        # 2) reciprocal-rank fusion across the collection lists
        fused = _rrf_fuse(lists)
        # 3) Jina rerank as a BLEND with the RRF rank. A full Jina reorder demotes
        #    strong RRF #1s (R@1 75%->71%); blending recovers R@1 AND lifts R@3 to
        #    96% (goal). alpha weights the (normalized) Jina score vs 1/(rrf_rank).
        rr = _jina_reranker()
        if rr is not None and fused:
            docs = [(getattr(h, "chunk", None) and getattr(h.chunk, "text", "")) or
                    getattr(h, "snippet", "") for _, h in fused]
            js = rr.scores(query, docs)
            jmax = max(js) or 1.0
            a = _RERANK_BLEND
            blended = [(a * (js[i] / jmax) + (1 - a) * (1.0 / (i + 1)), fused[i])
                       for i in range(len(fused))]
            fused = [x for _, x in sorted(blended, key=lambda t: -t[0])]
        return [self._to_hit(name, h) for name, h in fused[:k]]

    def answer(self, question: str, k: int = 10, collection: str | None = None):
        """Grounded, cited RAG answer within one public collection (defaults to
        the first available). Returns (collection, RagAnswer)."""
        name = collection if collection in self.indexes else next(iter(self.indexes))
        # Prefer the OpenRouter cloud model (deepseek-v4-flash) for public-law
        # answers; fall back to the local Qwen generator when no key.
        gen = _openrouter_generator() or generator_from_env()
        # Rewrite the (often verbose) question to keyword form FIRST — measured to
        # move the governing statute to rank 1 for retrieval — then let the index
        # rerank + window it. neighbor_window=1 keeps a section's enumerated body
        # attached to its heading so the generator has the substance to cite.
        q2 = _rewrite_query(question)
        res = rag.answer(self.indexes[name], q2, generator=gen, k=k, dedup=False,
                         reranker=_jina_reranker(), rerank_pool=40,
                         neighbor_window=1, max_passage_chars=2200)
        return name, res
