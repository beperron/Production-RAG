"""Dense embedding backends — the GPU semantic lane (with a CPU fallback).

Lexical retrieval (BM25F + char-ngram) is strong and free, but it cannot match
*meaning* — "car" vs "automobile", "due date" vs "deadline". That is where a
dense embedding model earns its keep, and where the GPU comes in. This module
provides one ``Embedder`` interface with three backends, chosen by what is
available:

* ``ServerEmbedder`` — the production path: a local embedding server (vLLM or
  Text-Embeddings-Inference, OpenAI-compatible ``/v1/embeddings``) on the GPU
  box. Best quality, nothing leaves the machine.
* ``SentenceTransformerEmbedder`` — if ``sentence-transformers`` is installed
  (CPU or GPU). Convenient for a single box.
* ``HashingEmbedder`` — pure-Python, dependency-free, deterministic. Not
  semantic (it is lexical features in dense form), but it keeps the dense
  pathway alive and testable when no model/GPU is present.

The retriever fuses dense cosine with the lexical signals via RRF, so dense is
*additive*: it can only help recall, and the system degrades gracefully to
lexical-only when no embedder is configured. See ``DocIndex(embedder=...)``.
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .retrieval import analyze


# PERF-01 hardening — MPS thrashes under concurrent encode; per-embedder lock
# serializes calls into ``model.encode`` so the GPU isn't fought over by many
# threads at once. Under load this *raises* total throughput (no device
# thrashing). ``EMBED_MAX_PARALLEL`` is documented but currently hard-capped to
# 1 (single-flight); the env var is reserved for future Semaphore-based fan-out
# once we can prove a given backend benefits from >1 in-flight encode.
def _embed_max_parallel() -> int:
    try:
        v = int(os.environ.get("EMBED_MAX_PARALLEL", "1"))
    except ValueError:
        v = 1
    return max(1, v)


# --------------------------------------------------------------------------- #
# interface
# --------------------------------------------------------------------------- #
class Embedder(ABC):
    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        """Return one unit-normalized vector per input text."""
        ...

    def embed_one(self, text: str, *, is_query: bool = False) -> list[float]:
        return self.embed([text], is_query=is_query)[0]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))  # vectors are unit-normalized


def _l2_normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm else v


# --------------------------------------------------------------------------- #
# CPU fallback — deterministic feature hashing (no deps, no model)
# --------------------------------------------------------------------------- #
class HashingEmbedder(Embedder):
    """Hash analyzed tokens (+char trigrams) into a fixed-dim unit vector.

    Honestly lexical, not semantic — but deterministic, instant, dependency-free,
    and enough to exercise/serve the dense pathway when there is no GPU. Swap in
    a real model (server/ST) for genuine semantic matching.
    """

    name = "hashing"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        toks = analyze(text)
        # token trigrams give a little fuzziness on top of the token features
        feats = list(toks)
        for tok in toks:
            s = f" {tok} "
            feats += [s[i : i + 3] for i in range(len(s) - 2)]
        for f in feats:
            h = int.from_bytes(hashlib.md5(f.encode()).digest()[:4], "little")
            sign = 1.0 if (h >> 31) & 1 else -1.0
            v[h % self.dim] += sign
        return _l2_normalize(v)


# --------------------------------------------------------------------------- #
# optional local model via sentence-transformers
# --------------------------------------------------------------------------- #
class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", device: str | None = None,
                 query_prefix: str = "", passage_prefix: str = ""):
        from sentence_transformers import SentenceTransformer  # lazy, optional dep

        self._model = SentenceTransformer(model, device=device)
        self.dim = self._model.get_sentence_embedding_dimension()
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.name = f"st:{model}"
        # PERF-01: per-instance lock serializes ``encode`` calls. Under
        # concurrent dashboard requests this prevents MPS thrashing — the
        # device is finite, so fighting for it produces 7-30× slowdowns;
        # serialization actually *raises* aggregate throughput.
        self._encode_lock = threading.Lock()
        # Reserved for future Semaphore-based fan-out (currently == 1).
        self._max_parallel = _embed_max_parallel()

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        prefix = self.query_prefix if is_query else self.passage_prefix
        inputs = [prefix + t for t in texts] if prefix else texts
        # PERF-01: serialize the GPU/MPS encode call. The lock is held for the
        # duration of the SentenceTransformer call only — Python-side prep and
        # post-processing run unlocked.
        with self._encode_lock:
            vecs = self._model.encode(inputs, normalize_embeddings=True)
        return [list(map(float, v)) for v in vecs]


# --------------------------------------------------------------------------- #
# production GPU path — local embedding server (OpenAI-compatible)
# --------------------------------------------------------------------------- #
class EmbeddingUnavailable(RuntimeError):
    """Raised when the embedding server is unreachable."""


class ServerEmbedder(Embedder):
    """Calls a local OpenAI-compatible ``/v1/embeddings`` endpoint (vLLM/TEI).

    This is the GPU lane: serve e.g. ``BAAI/bge-large-en-v1.5`` with vLLM
    (``--task embed``) or HuggingFace TEI on the A6000 box, point ``base_url``
    at it, and embeddings are computed on-GPU, locally.
    """

    name = "server"

    def __init__(self, model: str, *, base_url: str = "http://localhost:8001/v1",
                 api_key: str = "EMPTY", dim: int = 0, timeout: float = 60.0,
                 query_prefix: str = "", passage_prefix: str = "", batch_size: int = 16,
                 max_input_chars: int = 4000):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.dim = dim
        self.timeout = timeout
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        # Embed in batches: some servers (notably Ollama) drop the connection on
        # very large single requests, so we chunk the inputs.
        self.batch_size = max(1, batch_size)
        # Truncate each input to stay within the model's context window — an
        # oversized chunk (e.g. a giant table) otherwise returns HTTP 400/EOF.
        self.max_input_chars = max_input_chars
        self.name = f"server:{model}"

    def is_available(self) -> bool:
        import requests

        try:
            r = requests.get(f"{self.base_url}/models",
                             headers={"Authorization": f"Bearer {self.api_key}"}, timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        prefix = self.query_prefix if is_query else self.passage_prefix
        inputs = [(prefix + t)[: self.max_input_chars] for t in texts]
        out: list[list[float]] = []
        for i in range(0, len(inputs), self.batch_size):
            out.extend(self._embed_resilient(inputs[i:i + self.batch_size]))
        return [_l2_normalize(v) for v in out]

    def _embed_resilient(self, inputs: list[str], _retries: int = 3) -> list[list[float]]:
        """Embed a batch, retrying transient failures and splitting on hard ones.

        Local embedding servers (notably Ollama) occasionally drop a request with
        an EOF/HTTP 400. We retry with backoff; if a multi-item batch still
        fails, we split it to isolate a genuinely-bad input rather than aborting
        the whole build."""
        import time

        for attempt in range(_retries):
            try:
                return self._embed_batch(inputs)
            except EmbeddingUnavailable:
                if attempt < _retries - 1:
                    time.sleep(0.5 * (attempt + 1))
        if len(inputs) > 1:  # isolate the offending input(s)
            mid = len(inputs) // 2
            return self._embed_resilient(inputs[:mid]) + self._embed_resilient(inputs[mid:])
        # A single input still fails (e.g. token-dense table beyond the model's
        # context). Degrade to an empty/zero vector so the chunk stays lexically
        # searchable instead of aborting the whole index build. The DenseStore
        # treats this as a missing vector and records it (R4.3).
        #
        # EMPTY-VECTOR CONTRACT (H5): two equivalent "no vector" markers may be
        # returned by this resilient path —
        #   * ``[0.0] * self.dim`` when the embedder dim is known (a zero
        #     vector of the right shape — DenseStore.add() detects all-zero
        #     and routes it to ``_no_vector`` → the chunk is marked missing).
        #   * ``[]`` (empty list) when ``self.dim`` is still 0 (no batch has
        #     succeeded yet) and the truncated probe also failed — DenseStore
        #     treats an empty vector as missing too.
        # Both paths are valid "missing" signals; callers MUST NOT assume the
        # returned vector has a fixed dimension.
        if self.dim:
            return [[0.0] * self.dim]
        try:  # dim unknown yet: one last truncated probe; never abort the build (F12)
            return [self._embed_batch([inputs[0][:512]])[0]]
        except EmbeddingUnavailable:
            return [[]]  # empty marker → "no vector"

    def _embed_batch(self, inputs: list[str]) -> list[list[float]]:
        import requests

        try:
            r = requests.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json={"model": self.model, "input": inputs},
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise EmbeddingUnavailable(f"embedding request failed: {e}") from e
        if r.status_code != 200:
            raise EmbeddingUnavailable(f"embedding server HTTP {r.status_code}: {r.text[:200]}")
        data = r.json().get("data", [])
        vecs = [item["embedding"] for item in sorted(data, key=lambda d: d.get("index", 0))]
        if vecs and not self.dim:
            self.dim = len(vecs[0])
        return vecs


# --------------------------------------------------------------------------- #
# embedding model registry (GPU server), keyed to VRAM
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EmbedModel:
    key: str
    hf_repo: str
    dim: int
    max_tokens: int
    min_vram_gb: float
    query_prefix: str = ""
    passage_prefix: str = ""
    notes: str = ""


EMBED_REGISTRY: dict[str, EmbedModel] = {
    "bge-large-en": EmbedModel(
        "bge-large-en", "BAAI/bge-large-en-v1.5", 1024, 512, 4.0,
        query_prefix="Represent this sentence for searching relevant passages: ",
        notes="Strong general English retrieval embedder. Tiny VRAM; high throughput.",
    ),
    "e5-large-v2": EmbedModel(
        "e5-large-v2", "intfloat/e5-large-v2", 1024, 512, 4.0,
        query_prefix="query: ", passage_prefix="passage: ",
        notes="Robust retrieval; requires query:/passage: prefixes.",
    ),
    "gte-large": EmbedModel(
        "gte-large", "thenlper/gte-large", 1024, 512, 4.0,
        notes="Competitive general embedder, no prefix needed.",
    ),
    "nomic-embed-v1.5": EmbedModel(
        "nomic-embed-v1.5", "nomic-ai/nomic-embed-text-v1.5", 768, 8192, 4.0,
        query_prefix="search_query: ", passage_prefix="search_document: ",
        notes="8k context — good for long chunks. Needs trust-remote-code.",
    ),
    "bge-m3": EmbedModel(
        "bge-m3", "BAAI/bge-m3", 1024, 8192, 6.0,
        notes="Multilingual + long context; strong all-rounder.",
    ),
}


def recommend_embed_model(total_vram_gb: float) -> str:
    """Embedders are tiny; on any real GPU pick a long-context multilingual one,
    else a compact high-quality English model."""
    return "bge-m3" if total_vram_gb >= 8 else "bge-large-en"


def server_embedder_for(key: str, **kw) -> ServerEmbedder:
    m = EMBED_REGISTRY[key]
    return ServerEmbedder(
        m.hf_repo, dim=m.dim, query_prefix=m.query_prefix,
        passage_prefix=m.passage_prefix, **kw,
    )


# --------------------------------------------------------------------------- #
# dense store — chunk vectors + cosine search
# --------------------------------------------------------------------------- #
def _no_vector(v: list[float] | None) -> bool:
    """A degraded/empty embedding (missing or all-zero) carries no signal."""
    return not v or not any(v)


@dataclass
class DenseStore:
    embedder: Embedder
    vectors: dict[str, list[float]] = field(default_factory=dict)
    groups: dict[str, str] = field(default_factory=dict)
    # Chunk ids whose embedding failed → no dense vector (lexical-only). Tracked
    # so the build can report exactly which chunks lack semantic coverage (R4.3).
    missing: set[str] = field(default_factory=set)
    # Optional callback(chunk_id, group) invoked on each zero/empty vector, so the
    # index can record a dense_zero_vector degradation against the chunk.
    on_degrade: object = None

    def _store(self, item_id: str, vec: list[float], group: str) -> None:
        self.groups[item_id] = group
        if _no_vector(vec):
            self.missing.add(item_id)
            if callable(self.on_degrade):
                self.on_degrade(item_id, group)
        else:
            self.vectors[item_id] = vec
            self.missing.discard(item_id)

    def add(self, item_id: str, text: str, *, group: str = "") -> None:
        self._store(item_id, self.embedder.embed_one(text, is_query=False), group)

    def add_many(self, items: list[tuple[str, str, str]]) -> None:
        """Batch-embed (id, text, group) triples — one server round-trip."""
        if not items:
            return
        vecs = self.embedder.embed([t for _i, t, _g in items], is_query=False)
        for (iid, _t, g), v in zip(items, vecs):
            self._store(iid, v, g)

    def remove_group(self, group: str) -> None:
        for iid in [i for i, g in self.groups.items() if g == group]:
            self.vectors.pop(iid, None)
            self.groups.pop(iid, None)
            # Removed ids must also leave ``missing`` — otherwise the
            # query-time degradation ratio (len(missing)/len(chunks)) can
            # fire a false "dense lane degraded — re-embed" warning after a
            # doc removal, violating the no-dishonest-degradation invariant.
            self.missing.discard(iid)

    def search(self, query: str, k: int = 10, *, group: str | None = None) -> list[tuple[str, float]]:
        ids = [i for i in self.vectors if group is None or self.groups.get(i) == group]
        if not ids:
            return []
        qv = self.embedder.embed_one(query, is_query=True)
        scored = [(i, cosine(qv, self.vectors[i])) for i in ids]
        scored = [(i, s) for i, s in scored if s > 0]
        scored.sort(key=lambda x: -x[1])
        return scored[:k]
