"""Re-ranking — a precision stage on top of hybrid retrieval.

Lexical+dense fusion is high-recall but coarse. A cross-encoder re-ranker reads
the query and each candidate *together* and scores true relevance, so the top-k
shown to a user (or fed to an LLM) is far more precise. Retrieve a wide pool,
re-rank it, return the best.

Backends (all local / no-egress when pointed at a local server):

* ``Qwen3Reranker`` — the recommended cross-encoder. Point it at a server that
  exposes a ``/rerank`` endpoint for ``Qwen/Qwen3-Reranker-*`` (e.g. HuggingFace
  TEI or Infinity). Falls back gracefully if unreachable.
* ``LLMJudgeReranker`` — uses a local chat model (Ollama) to score query↔passage
  relevance. Needs no extra service; works with the models already installed.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod

_log = logging.getLogger(__name__)


class Reranker(ABC):
    name = "reranker"

    @abstractmethod
    def scores(self, query: str, documents: list[str]) -> list[float]:
        """Return one relevance score per document (higher = more relevant)."""

    def rerank(self, query: str, documents: list[str], *, top_k: int | None = None
               ) -> list[tuple[int, float]]:
        """Rank document indices by relevance; returns (index, score) descending."""
        if not documents:
            return []
        scored = list(enumerate(self.scores(query, documents)))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k] if top_k else scored


class Qwen3Reranker(Reranker):
    """Cross-encoder re-ranker over an HTTP ``/rerank`` endpoint (TEI/Infinity).

    Request:  ``{"query": q, "texts"|"documents": [...]}``
    Response: a list of ``{"index", "score"|"relevance_score"}`` (TEI/Jina/Cohere
    shapes are all accepted).
    """

    def __init__(self, model: str = "Qwen/Qwen3-Reranker-0.6B", *,
                 base_url: str = "http://localhost:8002", api_key: str = "EMPTY",
                 timeout: float = 60.0, batch_size: int = 32):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.batch_size = max(1, batch_size)
        self.name = f"qwen3-reranker:{model}"

    def is_available(self) -> bool:
        import requests

        try:
            return requests.get(f"{self.base_url}/health", timeout=5).status_code < 500
        except requests.RequestException:
            return False

    def scores(self, query: str, documents: list[str]) -> list[float]:
        out = [0.0] * len(documents)
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            for local_idx, score in self._rerank_batch(query, batch):
                out[i + local_idx] = score
        return out

    def _rerank_batch(self, query: str, batch: list[str]) -> list[tuple[int, float]]:
        import requests

        payload = {"model": self.model, "query": query,
                   "texts": batch, "documents": batch}
        r = requests.post(
            f"{self.base_url}/rerank", json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"}, timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        results = data.get("results", data) if isinstance(data, dict) else data
        out = []
        for item in results:
            idx = item.get("index")
            score = item.get("score", item.get("relevance_score", 0.0))
            if idx is not None:
                out.append((idx, float(score)))
        return out


class CrossEncoderReranker(Reranker):
    """Local sentence-transformers ``CrossEncoder`` (e.g. a BGE/MS-MARCO or
    served Qwen3 reranker checkpoint). Fast, no server, runs on CPU/MPS."""

    def __init__(self, model: str = "BAAI/bge-reranker-base", *, device: str | None = None):
        from sentence_transformers import CrossEncoder  # lazy, optional dep

        self._ce = CrossEncoder(model, device=device)
        self.name = f"cross-encoder:{model}"

    def scores(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        pairs = [[query, d] for d in documents]
        return [float(s) for s in self._ce.predict(pairs)]


class LLMJudgeReranker(Reranker):
    """Score relevance with a local chat model (Ollama) — no extra service.

    Prompts the model for an integer relevance score 0–10 per passage. Slower
    than a cross-encoder but works with already-installed models and is a useful
    baseline/fallback. Use a Qwen3 chat model to keep it a Qwen-based reranker.
    """

    _SCORE_RE = re.compile(r"-?\d+(?:\.\d+)?")

    def __init__(self, model: str = "qwen3.6:latest", *,
                 base_url: str = "http://localhost:11434/v1", api_key: str = "ollama",
                 timeout: float = 30.0, max_chars: int = 1200):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_chars = max_chars
        self.name = f"llm-judge:{model}"
        # True after any batch where at least one judge call failed and its
        # score degraded to 0.0 — callers that surface reranker health (e.g.
        # the dashboard) can read this instead of mistaking failures for
        # "all passages irrelevant".
        self.degraded = False

    def scores(self, query: str, documents: list[str]) -> list[float]:
        out: list[float] = []
        failures = 0
        last_err: Exception | None = None
        for d in documents:
            try:
                out.append(self._judge(query, d))
            except Exception as e:  # noqa: BLE001 — degrade per-doc, warn per-batch
                failures += 1
                last_err = e
                out.append(0.0)
        self.degraded = failures > 0
        if failures:
            # A 0.0 score from a dead judge is indistinguishable from "all
            # irrelevant" — warn ONCE per batch (not per document) so the
            # degradation is visible without flooding the log.
            _log.warning(
                "LLM-judge reranker degraded: %d/%d judge call(s) failed "
                "(last error %s: %s) — failed passages scored 0.0",
                failures, len(documents),
                type(last_err).__name__, last_err,
            )
        return out

    def _judge(self, query: str, doc: str) -> float:
        """Score one passage. Raises on transport/parse failure — ``scores``
        converts failures into 0.0 with a per-batch warning + degraded flag."""
        import requests

        prompt = (
            "Rate how well the PASSAGE answers the QUERY on a 0-10 scale "
            "(10 = directly answers, 0 = unrelated). Reply with only the number.\n\n"
            f"QUERY: {query}\n\nPASSAGE: {doc[:self.max_chars]}\n\nScore:"
        )
        r = requests.post(
            f"{self.base_url}/chat/completions",
            json={"model": self.model, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0, "max_tokens": 8},
            headers={"Authorization": f"Bearer {self.api_key}"}, timeout=self.timeout,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"].get("content") or ""
        m = self._SCORE_RE.search(text)
        return float(m.group(0)) if m else 0.0
