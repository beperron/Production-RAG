"""LLM query rewriter — addresses the categorical regression we measured on
the QoC sweep, where hybrid retrieval *lost* on questions like "monthly
face-to-face visits" because the docs use the phrase "monthly contact log"
instead. A small ``qwen3.6:latest`` rewriter generates N paraphrases per
question, unions the candidate pools, and returns a fused result list.

Two public entry points:

  * ``QueryRewriter.rewrite(q, n)`` → list of N paraphrases (incl. original)
  * ``multi_search(index, q, rewriter, k, ...)`` → union-then-fuse SearchHit list

Local-only (Ollama). No egress. ``<think>`` tokens emitted by Qwen 3.6 are
stripped before the variants are parsed — see ``rag._strip_reasoning`` for
the shared regex. Variant parsing is defensive: the prompt asks for a JSON
list, but if the model returns numbered lines / bullets / a mix, we fall
back to a line-splitter. The original question is always included as
variant 0 so we never produce a worse pool than baseline.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

from parsevault.rag import ChatGenerator, _strip_reasoning

log = logging.getLogger(__name__)

# M7: bound the rewriter cache so a long-lived dashboard process can't grow
# without limit. 256 entries covers the typical session (a few dozen queries
# with N=5 variants each) with headroom.
_CACHE_MAX = 256


_SYSTEM = """You are a search-query specialist for a North Carolina
child-welfare case-record corpus (DSS notes, court orders, clinical
assessments, treatment plans, medication records, agency forms, IEPs,
incident reports). Given a research question, produce DIVERSE search-query
variants — different vocabulary, different framings — so a hybrid lexical+
dense retriever can find every relevant document.

Coverage instructions:
- Include CLINICAL vernacular (e.g. "no-show" / "did not attend" / "DNA"
  for missed therapy; "step-down" / "discharge plan" for transitions).
- Include CASE-MANAGEMENT phrasing (e.g. "monthly contact log",
  "face-to-face visit", "30-day visit", "Title IV-E monthly contact").
- Include LEGAL phrasing where relevant (statute names, court-order types,
  "permanency hearing", "TPR", "GAL").
- Include CONCRETE artifact phrases the records would actually use
  (e.g. "Plan of Care", "Person-Centered Plan", "Service Plan").
- Names and dates from the original question must appear in every variant
  unchanged.
- Each variant must be CONCISE (under 25 words).

Return ONLY a JSON array of strings, like ["…", "…", "…"]. No commentary,
no markdown fences, no numbering."""


@dataclass(frozen=True)
class RewriteResult:
    original: str
    variants: list[str]      # always includes original at index 0
    raw_response: str = ""   # for trace


class QueryRewriter:
    """Wraps a local ``ChatGenerator`` (Ollama) to produce paraphrase variants."""

    def __init__(self, generator: ChatGenerator, *, default_n: int = 5,
                 max_tokens: int = 600, cache_max: int = _CACHE_MAX):
        self.generator = generator
        self.default_n = default_n
        self.max_tokens = max_tokens
        # M7: bounded LRU cache + lock. The dashboard reuses one rewriter
        # across requests (per-process), so the cache must (a) cap memory and
        # (b) tolerate concurrent reads/writes from worker threads/coroutines.
        self._cache_max = cache_max
        self._cache: OrderedDict[tuple[str, int], RewriteResult] = OrderedDict()
        self._cache_lock = threading.Lock()

    def _cache_get(self, key: tuple[str, int]) -> RewriteResult | None:
        with self._cache_lock:
            result = self._cache.get(key)
            if result is not None:
                # Move-to-end → LRU order.
                self._cache.move_to_end(key)
            return result

    def _cache_set(self, key: tuple[str, int], value: RewriteResult) -> None:
        with self._cache_lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_max:
                self._cache.popitem(last=False)  # evict least-recently-used

    def rewrite(self, question: str, n: int | None = None) -> RewriteResult:
        n = n or self.default_n
        # M-9: empty/whitespace input short-circuits before any LLM round-trip
        # (the rewriter has nothing to paraphrase, and an attorney could
        # accidentally fire a deep-research run on a blank query box).
        stripped = (question or "").strip()
        if not stripped:
            return RewriteResult(original=question, variants=[question],
                                 raw_response="")
        key = (stripped, n)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        user = (f"Question: {question}\n\n"
                f"Produce {n - 1} ADDITIONAL search variants (the original "
                "is variant 0 and you do NOT need to repeat it). Return a "
                "JSON array of the variant strings only.")
        try:
            raw = self.generator.complete(_SYSTEM, user)
        except Exception as e:  # noqa: BLE001 — never let rewriter failure block search
            log.warning("query rewriter failed (%s) — falling back to original",
                        type(e).__name__)
            # Do NOT cache the failure fallback: a transient LLM outage would
            # otherwise pin the original-only result for the process lifetime,
            # permanently disabling rewrites for this query even after the
            # generator recovers. Only successful rewrites are cached.
            return RewriteResult(original=question, variants=[question],
                                 raw_response="")
        cleaned = _strip_reasoning(raw)
        variants = _parse_variants(cleaned, want=n - 1)
        # Always lead with the original — guarantees rewriter is never worse.
        out = [question]
        for v in variants:
            if v and v not in out:
                out.append(v)
            if len(out) >= n:
                break
        result = RewriteResult(original=question, variants=out, raw_response=raw)
        self._cache_set(key, result)
        return result


# Defensive variant parsing: prefer a clean JSON array, but tolerate numbered
# lists / bullets / mixed output. Names + dates in the original survive because
# the model is instructed to preserve them; we don't try to enforce that here.
_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*?\]")
_BULLET_LINE_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*(.+)$")


def _parse_variants(text: str, want: int) -> list[str]:
    text = text.strip()
    # Strip code fences if any leaked through.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    # 1. clean JSON array
    m = _JSON_ARRAY_RE.search(text)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()][:want]
        except Exception:
            pass
    # 2. bulleted / numbered list per line
    out: list[str] = []
    for line in text.splitlines():
        m = _BULLET_LINE_RE.match(line)
        if m:
            out.append(m.group(1).strip().strip('"').strip("'"))
        if len(out) >= want:
            break
    if out:
        return out
    # 3. last resort: non-empty lines, capped
    return [ln.strip().strip('"').strip("'")
            for ln in text.splitlines()
            if ln.strip() and not ln.strip().startswith("```")][:want]


# ---------- multi-query search ---------------------------------------------------

def _dedup_hits(hits: list, *, threshold: float = 0.9) -> list:
    """Drop near-duplicates from a ranked hit list using char-trigram cosine.

    Mirrors the in-search ``dedup=True`` collapse (docindex._cosine + char_ngrams)
    but operates *cross-variant* — after the fused list is built — to remove
    duplicates that two different paraphrases retrieved as different chunks
    of substantively identical text. Hits without recoverable text fall through.
    M-4 audit.
    """
    from .retrieval import _cosine, char_ngrams

    kept: list = []
    kept_vecs: list = []
    for h in hits:
        # Best-effort recover the chunk text from either a real SearchHit
        # (h.chunk.text) or a test-double exposing a top-level ``text``.
        text = (getattr(getattr(h, "chunk", None), "text", None)
                or getattr(h, "text", None) or "")
        if not text:
            kept.append(h)
            continue
        vec = char_ngrams(text)
        if any(_cosine(vec, kv) >= threshold for kv in kept_vecs):
            continue
        kept.append(h)
        kept_vecs.append(vec)
    return kept


def multi_search(index, question: str, *, rewriter: QueryRewriter, k: int = 10,
                 n_variants: int | None = None, fuse: str = "max",
                 dedup: bool = True, **search_kw) -> list:
    """Search across N rewriter variants, union the candidate pools, fuse scores.

    ``fuse`` options:
      * ``"max"``  — keep each chunk's BEST score across variants
      * ``"sum"``  — sum scores across variants the chunk appears in (boosts
                     chunks that match multiple framings)

    Returns up to ``k`` SearchHit-shaped objects sorted by fused score. The
    original question is always one of the variants, so this method NEVER
    returns fewer hits than ``index.search(question, k)``.
    """
    rr = rewriter.rewrite(question, n=n_variants)
    by_chunk: dict[str, tuple[float, object]] = {}
    for variant in rr.variants:
        try:
            hits = index.search(variant, k=k, **search_kw)
        except Exception as e:  # noqa: BLE001
            # L-4 audit: log the exception (with traceback) — earlier this
            # only logged the variant string, so a downstream crash was
            # invisible to anyone debugging multi_search behavior.
            log.warning(
                "variant search failed (%s: %s) on %r",
                type(e).__name__, e, variant[:80], exc_info=True,
            )
            continue
        for h in hits:
            # Real ``SearchHit`` (pipeline.docindex) exposes the chunk id only
            # via ``hit.chunk.chunk_id``; a few internal/fake hit objects (and
            # the test doubles in this module's tests) expose ``chunk_id``
            # directly at the top level. Read both — without this fallback,
            # ``multi_search`` silently returned zero hits in production while
            # the unit-test fakes (which expose top-level ``chunk_id``) passed.
            # See audit F-2.
            cid = (
                getattr(getattr(h, "chunk", None), "chunk_id", None)
                or getattr(h, "chunk_id", None)
                or getattr(h, "id", "")
                or ""
            )
            if not cid:
                continue
            score = float(getattr(h, "score", 0.0))
            if cid in by_chunk:
                prev_score, prev_hit = by_chunk[cid]
                new_score = max(prev_score, score) if fuse == "max" else prev_score + score
                # Keep the hit object from whichever variant scored higher so the
                # snippet matches the strongest framing.
                if score > prev_score:
                    by_chunk[cid] = (new_score, h)
                else:
                    by_chunk[cid] = (new_score, prev_hit)
            else:
                by_chunk[cid] = (score, h)
    # Stable sort by fused score descending.
    ranked = sorted(by_chunk.values(), key=lambda x: -x[0])
    out = []
    for fused_score, hit in ranked:
        # Best-effort: stamp the fused score back onto the hit if it has a score
        # attribute; otherwise return the hit untouched.
        try:
            hit.score = fused_score
        except (AttributeError, TypeError):
            pass
        out.append(hit)
    # Cross-variant near-dup collapse (M-4): a paraphrase often returns the
    # same legal definition copied across documents under different chunk IDs.
    # Per-variant ``dedup=True`` removes within-pool dupes, but the FUSED list
    # can still carry duplicates from different paraphrases. Apply the same
    # char-trigram cosine collapse here, then truncate to k.
    if dedup:
        out = _dedup_hits(out)
    return out[:k]


def collect_variants(rewriter: QueryRewriter, questions: Iterable[str],
                     n_variants: int | None = None) -> dict[str, list[str]]:
    """Pre-warm the rewriter cache for a batch of questions. Useful when you
    want to commit the generated variants before the actual search run (so the
    variants are reviewable / reproducible)."""
    return {q: rewriter.rewrite(q, n=n_variants).variants for q in questions}
