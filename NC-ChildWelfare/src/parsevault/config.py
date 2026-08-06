"""Read the extraction-cascade configuration from the environment.

The CPU lanes (native text + Tesseract) need no configuration. The vision-LLM
lane points at any OpenAI-compatible server via OCR_* env vars — on a Mac that
is typically Ollama at http://localhost:11434/v1. With nothing set, the defaults
keep the cascade pointed at the (Linux/A6000) vLLM server on :8000 and the VLM
lane simply stays dormant if no server is reachable.
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

from .pipeline.extractors.router import CascadeConfig

# CLI-2: loopback-only egress invariant for LOCAL LLM endpoints. The
# generator / embedder / reranker helpers below all build clients that
# point at an OpenAI-compatible base URL — if that URL is not loopback,
# we egress confidential prompts off-machine. The check is centralised
# here so every caller (RAG, qoc_review, dashboard) is covered uniformly.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})


def _allow_remote_llm() -> bool:
    """``PARSEVAULT_ALLOW_REMOTE_LLM=1`` opts out of the loopback check.

    This is the documented escape hatch for the rare legitimate case of a
    remote dev server on a controlled tailnet / SSH-forwarded port. The
    default (env unset) refuses any non-loopback URL.
    """
    return os.environ.get("PARSEVAULT_ALLOW_REMOTE_LLM", "").lower() in (
        "1", "true", "yes",
    )


def _assert_loopback_base_url(base_url: str, *, label: str) -> None:
    """Raise ``RuntimeError`` if ``base_url`` is not a loopback address.

    ``label`` is the human-readable name of the helper (e.g. ``"RAG generator"``,
    ``"embedder"``) so the error tells the operator *which* knob is wrong.
    Bypass with ``PARSEVAULT_ALLOW_REMOTE_LLM=1`` — that path emits a
    ``logging.warning`` so the bypass shows up in any centralised log.
    """
    parsed = urlparse(base_url)
    host = (parsed.hostname or "").lower()
    if host in _LOOPBACK_HOSTS:
        return
    if _allow_remote_llm():
        logging.getLogger(__name__).warning(
            "PARSEVAULT_ALLOW_REMOTE_LLM=1: %s base_url %r is NOT loopback "
            "(host=%r) — confidential prompts may egress. This must be an "
            "intentional opt-in on a controlled network.",
            label, base_url, host,
        )
        return
    raise RuntimeError(
        f"refusing non-loopback URL for {label}: {base_url!r} (host={host!r}). "
        "Local LLM endpoints must be loopback (localhost/127.0.0.1/::1). "
        "Set PARSEVAULT_ALLOW_REMOTE_LLM=1 to opt in on a controlled network."
    )


def strict_mode() -> bool:
    """Return True iff ``PARSEVAULT_STRICT`` selects refuse-on-degrade behavior.

    M15: single source of truth — the same env-var check used to live in
    ``pipeline/docindex.py::_strict`` and inline in ``embedder_from_env``.
    Centralized here so the two never drift on accepted values.
    """
    return os.environ.get("PARSEVAULT_STRICT", "").lower() in ("1", "true", "yes")


def private_config() -> CascadeConfig:
    """A no-egress extraction config for the PRIVATE system.

    Confidential records (e.g. case files) must never leave the machine, so this
    forces the local cascade and refuses cloud conversion **regardless of the
    environment** — even if ``PARSE_PROVIDER=llamaparse`` or ``PARSE_PUBLIC_DATA_ACK``
    is set. Use this for any confidential workspace; the cloud lane is reachable
    only via the public path with an explicit acknowledgement.
    """
    # HF Hub offline (egress hygiene): sentence-transformers pings the
    # HuggingFace Hub on EVERY embedder load unless told not to — an
    # off-message network call for a "no egress" workspace and an offline-demo
    # stall risk. ``setdefault`` so an operator's explicit HF_HUB_OFFLINE=0
    # still wins. NOTE: the embedding models are cached locally, so offline
    # load works — a fresh machine needs one online pull first.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    cfg = cascade_config_from_env()
    cfg.parse_provider = "local-cascade"
    cfg.allow_cloud = False
    return cfg


def cascade_config_from_env() -> CascadeConfig:
    """Build the extraction cascade config from OCR_* env vars (Mac/Ollama-friendly)."""
    use_vlm = os.environ.get("OCR_USE_VLM", "1").lower() not in ("0", "false", "no")
    vlm_base_url = os.environ.get("OCR_VLM_BASE_URL", "http://localhost:8000/v1")
    if use_vlm:
        # CLI-2 parity: the VLM OCR lane posts base64 PAGE IMAGES to this URL —
        # a remote address egresses document content exactly like a remote
        # generator/embedder/reranker would egress prompts. ``private_config()``
        # forces the local cascade but previously left this URL unchecked.
        # Same escape hatch as the other guards: PARSEVAULT_ALLOW_REMOTE_LLM=1.
        # Skipped when OCR_USE_VLM=0 — the URL is dormant then, and a stale
        # remote value in the env must not break a CPU-only build.
        _assert_loopback_base_url(vlm_base_url, label="VLM OCR")
    return CascadeConfig(
        use_vlm=use_vlm,
        base_url=vlm_base_url,
        api_key=os.environ.get("OCR_VLM_API_KEY", "ollama"),
        quality_model=os.environ.get("OCR_QUALITY_MODEL", ""),
        fast_model=os.environ.get("OCR_FAST_MODEL", ""),
        dpi=int(os.environ.get("OCR_DPI", "200")),
        tesseract_lang=os.environ.get("OCR_TESSERACT_LANG", "eng"),
        max_workers=int(os.environ.get("OCR_MAX_WORKERS", "0")),
        parse_provider=os.environ.get("PARSE_PROVIDER", "local-cascade"),
        llamaparse_api_key=os.environ.get("LLAMAPARSE_API_KEY", ""),
        llamaparse_base_url=os.environ.get(
            "LLAMAPARSE_BASE_URL", "https://api.cloud.llamaindex.ai/api/v1"
        ),
        # Cloud-egress acknowledgement (R2.5): off unless explicitly set. The CLI
        # --public-data-acknowledged flag sets this directly.
        allow_cloud=os.environ.get("PARSE_PUBLIC_DATA_ACK", "").lower() in ("1", "true", "yes"),
        # Raster archival of flagged pages (R3.4).
        raster_archive_dir=os.environ.get("OCR_RASTER_ARCHIVE_DIR", ""),
        raster_archive_mode=os.environ.get("OCR_RASTER_ARCHIVE_MODE", "flagged"),
    )


# GTE-base is the chosen default dense model: best ranking among base models in
# our evaluation (evals/RESULTS.md), tiny/fast, and fine-tunable. Override with
# EMBED_MODEL; set EMBED_MODEL=none to force lexical-only.
DEFAULT_EMBED_MODEL = "thenlper/gte-base"


def embedder_from_env():
    """Build the dense embedder from EMBED_* env vars (defaults to GTE-base).

    Defaults to the local GTE-base sentence-transformers model. Override::

        EMBED_MODEL=qwen3-embedding:0.6b EMBED_BACKEND=server  # Ollama /v1
        EMBED_MODEL=none                                       # lexical-only

    ``EMBED_BACKEND`` auto-detects when unset: a HuggingFace repo / local path →
    sentence-transformers; an Ollama model tag → the OpenAI-compatible server.
    Falls back to lexical-only (returns ``None``) if the embedder can't load, so
    retrieval always works.

    Offline behavior: the sentence-transformers lane honors the standard
    ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` env vars (passed through
    untouched here). ``private_config()`` sets both to ``1`` (via
    ``setdefault``) so private workspaces never ping the HuggingFace Hub;
    public workspaces keep the library default unless the operator sets them.
    Either way the models load from the local cache — a fresh machine needs
    one online pull first.
    """
    model = os.environ.get("EMBED_MODEL", DEFAULT_EMBED_MODEL).strip()
    if not model or model.lower() in ("none", "off", "lexical"):
        return None
    backend = os.environ.get("EMBED_BACKEND", "").lower()
    if not backend:  # auto: HF repo / local path → st; ollama tag → server
        backend = "st" if ("/" in model or model.startswith(".") or "gte" in model.lower()) else "server"
    try:
        if backend in ("st", "sentence-transformers", "sentence_transformers"):
            from .pipeline.embeddings import SentenceTransformerEmbedder

            return SentenceTransformerEmbedder(
                model,
                query_prefix=os.environ.get("EMBED_QUERY_PREFIX", ""),
                passage_prefix=os.environ.get("EMBED_PASSAGE_PREFIX", ""),
            )
        from .pipeline.embeddings import ServerEmbedder

        embed_base_url = os.environ.get("EMBED_BASE_URL", "http://localhost:11434/v1")
        _assert_loopback_base_url(embed_base_url, label="embedder")
        return ServerEmbedder(
            model,
            base_url=embed_base_url,
            api_key=os.environ.get("EMBED_API_KEY", "ollama"),
            query_prefix=os.environ.get("EMBED_QUERY_PREFIX", ""),
            passage_prefix=os.environ.get("EMBED_PASSAGE_PREFIX", ""),
        )
    except Exception as e:  # noqa: BLE001
        # "Configured but failed" is different from "not configured" (R4.2): a
        # misconfigured embedder must not silently become lexical-only in
        # production. Log it, and under strict mode refuse rather than degrade.
        import logging

        logging.getLogger(__name__).error("embedder %r failed to load: %s", model, e)
        if strict_mode():
            raise RuntimeError(
                f"EMBED_MODEL={model!r} is configured but failed to load ({e}); "
                "refusing to degrade to lexical-only under PARSEVAULT_STRICT."
            ) from e
        return None  # graceful: lexical-only if the embedder is unavailable


def generator_from_env():
    """Build the local RAG chat generator from RAG_* env vars (Ollama by default).

    Returns a ``rag.ChatGenerator`` pointed at a local OpenAI-compatible server,
    so generation never egresses. The model defaults to ``RAG_MODEL`` (or a common
    local instruct model); if it is not actually pulled/reachable, generation
    degrades to retrieval-only at call time with a note — it never fails the UI::

        RAG_MODEL=qwen3.6:latest          # any local Qwen3.6-family chat model you've pulled
        RAG_BASE_URL=http://localhost:11434/v1
    """
    from .rag import ChatGenerator

    # Default to qwen3.6:latest (the non-MLX, full-precision 35B). Higher
    # fidelity than the MLX-quantized variant at the cost of latency — the
    # explicit user choice for legal-grade work where output correctness
    # outranks UX speed. Qwen3.6 is a REASONING model and needs a large
    # token budget (it thinks before answering; too small → empty content).
    # Earlier history (kept for context): qwen3.6:35b-mlx beat qwen3-vl:8b
    # by ~0.16 grounded / ~0.16 faithfulness in docs/RAG_EVAL.md; the
    # non-quantized 35B sits above that.
    model = os.environ.get("RAG_MODEL", "qwen3.6:latest")
    base_url = os.environ.get("RAG_BASE_URL", "http://localhost:11434/v1")
    _assert_loopback_base_url(base_url, label="RAG generator")
    return ChatGenerator(
        model,
        base_url=base_url,
        api_key=os.environ.get("RAG_API_KEY", "ollama"),
        timeout=float(os.environ.get("RAG_TIMEOUT", "600")),
        # Bumped from 4000 to 8000 after attorney-eval found Qwen3.6:latest
        # reasoning-model consumes the prior budget inside ``<think>`` tags and
        # never reaches the answer (Deep Research mode scored 7.50/20 with 5 of
        # 6 sampled runs aborting "model returned only reasoning, no answer").
        # ``rag._strip_reasoning`` removes the ``<think>`` tags from the output
        # before parsing, but the model still NEEDS the tokens to think before
        # answering — so the budget covers ~4000 reasoning + ~4000 answer.
        max_tokens=int(os.environ.get("RAG_MAX_TOKENS", "8000")),
    )


def reranker_from_env():
    """Build the re-ranker from RERANK_* env vars, or return None (no reranking).

    ``RERANK_MODEL`` enables it. ``RERANK_BACKEND`` selects the implementation::

        RERANK_BACKEND=qwen3       # cross-encoder over an HTTP /rerank server
        RERANK_BASE_URL=http://localhost:8002
        RERANK_MODEL=Qwen/Qwen3-Reranker-0.6B

        RERANK_BACKEND=llm-judge   # local chat model scores relevance (Ollama)
        RERANK_MODEL=qwen3.6:latest
    """
    model = os.environ.get("RERANK_MODEL", "").strip()
    if not model:
        return None
    backend = os.environ.get("RERANK_BACKEND", "qwen3").lower()
    if backend in ("cross-encoder", "ce", "st"):
        from .pipeline.reranker import CrossEncoderReranker

        return CrossEncoderReranker(model)
    if backend in ("llm-judge", "llm", "judge"):
        from .pipeline.reranker import LLMJudgeReranker

        judge_url = os.environ.get("RERANK_BASE_URL", "http://localhost:11434/v1")
        _assert_loopback_base_url(judge_url, label="LLM-judge reranker")
        return LLMJudgeReranker(
            model,
            base_url=judge_url,
            api_key=os.environ.get("RERANK_API_KEY", "ollama"),
        )
    from .pipeline.reranker import Qwen3Reranker

    qwen_url = os.environ.get("RERANK_BASE_URL", "http://localhost:8002")
    _assert_loopback_base_url(qwen_url, label="Qwen3 reranker")
    return Qwen3Reranker(
        model,
        base_url=qwen_url,
        api_key=os.environ.get("RERANK_API_KEY", "EMPTY"),
    )
