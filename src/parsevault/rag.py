"""Retrieval-augmented generation — a *grounded* answer mode over the index.

Retrieval finds the passages; this adds an optional generation step that answers
the question **using only those passages**, citing each one inline. Built for
legal work, so the guarantees matter more than fluency:

* **Local only / no egress.** Generation calls a local OpenAI-compatible chat
  server (Ollama by default), the same lane the LLM-judge reranker uses. No
  document or question ever leaves the machine.
* **Grounded, not free-form.** The model is instructed to use *only* the numbered
  sources, cite every statement inline as ``[n]``, and explicitly say when the
  sources don't answer the question. Every source's **full passage** and citation
  travel with the answer, so the attorney verifies the grounding directly — the
  answer is a reading aid over cited primary text, never a substitute for it.
* **Fails safe.** If the local model is unreachable or returns nothing, the
  answer degrades to retrieval-only (the sources are still returned) with a clear
  note — never a silent or fabricated answer.

This is an *option* (a toggle): retrieval works exactly as before with generation
off; turning it on adds the synthesized, cited answer on top of the same sources.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Phrases the model is told to emit when the sources are insufficient — detected
# so the UI can mark the answer "not grounded" rather than implying confidence.
# The system prompt asks for an exact canonical sentence, but Qwen3.6 reasoning
# models routinely rephrase. ``_REFUSAL`` is the canonical sentence we ask the
# model to emit (we still embed this exact phrase in ``_SYSTEM``);
# ``_REFUSAL_PHRASES`` is the wider set used at detection time so paraphrases
# don't flip the badge to "not grounded" when the model meant "I can't answer
# from these sources" (audit S-4).
#
# Rationale for each new addition (pass-2 audit L-14 follow-up):
#
# * ``"unable to find"`` / ``"i'm unable to"`` — Claude-style refusal:
#   "I'm unable to find information in the provided sources …".
# * ``"no relevant information"`` — common GPT-style hedge:
#   "There is no relevant information in the sources to answer this".
# * ``"not in the provided"`` — generic "the answer is not in the provided
#   context" lane that L-14 explicitly called out.
#
# The first-sentence-OR-zero-cite gate around this list (see ``_is_refusal``)
# prevents these short phrases from false-positiving inside a substantive
# cited answer that happens to use the same wording incidentally.
_REFUSAL = "do not contain enough information"
_REFUSAL_PHRASES: tuple[str, ...] = (
    "do not contain enough",
    "don't contain enough",
    "do not have enough",
    "don't have enough",
    "insufficient information",
    "cannot answer",
    "can't answer",
    "no information about",
    "sources do not address",
    "sources don't address",
    "not enough information",
    # L-14 paraphrases (pass-2 audit):
    "unable to find",
    "i'm unable to",
    "no relevant information",
    "not in the provided",
)


# CAL-3: refusal detection used to be a bare substring sweep over the whole
# answer. The S-4 widening (10+ paraphrases) lifted the false-positive rate to
# ~40-60 % on substantive legal answers — phrases like "insufficient
# information to establish a prima facie case" or "cannot answer for the
# conduct of private parties" are real legal-art terms that appear INSIDE a
# fully-grounded answer. The audit
# (`archive/audit-reports/2026-06-28-confidence.md` §3) showed 5 of 8 plausible probes
# flipping to refused.
#
# Tightened semantics: a matched phrase only counts as a refusal when EITHER
#   (a) it appears in the FIRST SENTENCE of the answer — the model is
#       prompted to lead with the canonical refusal, so a genuine refusal
#       always opens with one of these phrases, OR
#   (b) the answer contains ZERO ``[n]`` citations — a real refusal cites
#       nothing.
#
# Incidental wording deeper in a substantive (cited) answer is ignored.
# M-4 (CAL-3 follow-up): the original implementation looked for ``". "`` /
# ``".\n"`` literal pairs. Real model output drops the inter-sentence space
# more often than expected — e.g. ``"Sentence one.Sentence two."`` would parse
# as ONE sentence, so a refusal phrase embedded after a substantive lead would
# trip the first-sentence rule. The regex below treats ``.``/``!``/``?`` as a
# sentence boundary when followed by whitespace, an uppercase letter, or
# end-of-string, AND treats a bare newline as a boundary. That matches the
# common no-space-after-period case without splitting on every decimal or
# abbreviation in mid-sentence (``Mr.`` / ``§ 7B-1111.`` followed by lowercase
# are NOT boundaries because the lookahead requires uppercase/whitespace/EOS).
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|[A-Za-z]|$)|\n")


def _first_sentence(text: str) -> str:
    """Return the leading sentence (everything up to and including the first
    sentence-ending punctuation, or the whole stripped text if none).

    Boundary: ``.``/``!``/``?`` followed by whitespace, an uppercase letter, or
    end-of-string; OR a literal newline. The uppercase-letter lookahead is
    what catches the no-space-after-period case (``"Cannot answer.Source [1]
    confirms…"`` splits at the ``.`` before ``Source``) without false-splitting
    on decimals or mid-sentence section numbers (``§ 7B-1111.`` followed by a
    lowercase continuation stays in one sentence).

    Used for CAL-3's "is the refusal phrase in the LEAD position?" check.
    """
    stripped = text.strip()
    if not stripped:
        return ""
    m = _SENTENCE_END_RE.search(stripped)
    if not m:
        return stripped
    end = m.end()
    # For newline matches the newline itself is the boundary char and is
    # included; for punctuation matches the punctuation is the last char.
    return stripped[:end].rstrip()


_CITATION_PROBE_RE = re.compile(r"\[\d+")


def _is_refusal(text: str) -> bool:
    """True when the (lowercased) model output declares the sources insufficient.

    Covers the canonical sentence in ``_SYSTEM`` plus common paraphrases. CAL-3:
    a matched phrase only triggers refusal when EITHER the match lies in the
    FIRST sentence of the answer OR the answer carries zero ``[n]`` citations.
    A phrase appearing incidentally deeper inside a cited answer is ignored —
    that was the substring lane's failure mode. ``[1,2]`` / ``[1-3]`` count as
    citations here (a loose probe, not the strict CAL-4 parser); we only need
    to know whether the answer cites *anything*."""
    if not text:
        return False
    low = text.lower()
    # First-sentence rule: a genuine refusal leads with one of the phrases
    # (the system prompt asks for the exact canonical sentence to open the
    # answer; paraphrases keep that lead position).
    lead = _first_sentence(low)
    if any(p in lead for p in _REFUSAL_PHRASES):
        return True
    # Citation-presence rule: a substantive grounded answer cites SOMETHING.
    # When the answer cites nothing AND a refusal phrase appears anywhere,
    # the phrase is the refusal — not incidental wording.
    has_citation = bool(_CITATION_PROBE_RE.search(text))
    if not has_citation and any(p in low for p in _REFUSAL_PHRASES):
        return True
    return False

# Qwen3.6 (and similar reasoning models) emit chain-of-thought inside
# ``<think>...</think>`` blocks before the actual answer. Those tokens are not
# part of the answer the attorney sees, but they will derail both the
# single-turn citation parser (``[n]`` extraction) and the ReAct ``Action:``
# parser if left in place. ``_strip_reasoning`` removes them as a defensive
# post-processing step — non-destructive (the raw output is preserved for trace)
# and a NO-OP for outputs from non-reasoning generators.
_THINK_CLOSED_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_THINK_UNCLOSED_RE = re.compile(r"<think\b[^>]*>.*\Z", re.IGNORECASE | re.DOTALL)
# Markers that look like real answer content emitted by the generator. If we
# see any of these after a *still-open* ``<think>`` tag, the unclosed-block
# wipe is unsafe (it would eat real answer text); keep the prefix + the tail
# in that case so the citation parser still sees ``[n]`` markers. M-1.
_ANSWER_MARKERS = (
    re.compile(r"\[\d+\]"),                # inline [n] citation
    re.compile(r"\b[Aa]nswer\s*:"),        # explicit "Answer:" lead-in
    re.compile(r"</think\s*>", re.IGNORECASE),
)


def _strip_reasoning(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks from a model output.

    Handles three Qwen3.6-family cases:

    1. **Closed blocks** — ``<think>...</think>`` (the well-formed case): all
       such blocks are removed, including nested/repeated occurrences.
    2. **Unclosed block** — ``<think>`` with no closing tag (the model ran out
       of budget while reasoning): everything from the opening ``<think>`` to
       end-of-string is dropped UNLESS the post-tag region contains answer
       markers (``[n]`` or an ``Answer:`` lead-in); in that case the tag is
       dropped but the surrounding text is preserved so the parser can still
       find the citations (M-1).
    3. **No reasoning markup** — a plain answer from a non-reasoning generator
       passes through untouched (NO-OP), so existing single-turn RAG tests
       against ``qwen3-vl:8b`` / fake generators are unaffected.

    Matching is case-insensitive (``<THINK>``, ``<Think>``) and multi-line
    (``re.DOTALL``). The raw response is preserved by the caller for trace /
    debug; only the parser sees the cleaned string.
    """
    if not text or "<think" not in text.lower():
        return text
    cleaned = _THINK_CLOSED_RE.sub("", text)
    # Look for an unclosed <think> tag in what remains. If the tail has answer
    # markers, treat this as "reasoning leaked": log + strip just the tag,
    # keeping the surrounding content (M-1 audit). Otherwise wipe the tail.
    m = re.search(r"<think\b[^>]*>", cleaned, re.IGNORECASE)
    if m:
        tail = cleaned[m.end():]
        if any(p.search(tail) for p in _ANSWER_MARKERS):
            import logging
            logging.getLogger(__name__).warning(
                "reasoning leaked: unclosed <think> with answer content after — "
                "stripping tag only, keeping prefix and tail"
            )
            cleaned = cleaned[:m.start()] + tail
        else:
            cleaned = _THINK_UNCLOSED_RE.sub("", cleaned)
    return cleaned

# CAL-4: citation parsing. The prompt asks for ``[1]`` / ``[1][2]`` but
# generators routinely emit ``[1,2]``, ``[1, 2]``, or ranges ``[1-3]``. The
# narrow legacy ``\[(\d+)\]`` regex silently dropped those, making the answer
# look ungrounded. ``_CITE_RE`` matches a single bracketed group of one or
# more integers separated by commas, hyphens, en-dashes, or whitespace;
# ``_parse_citation_markers`` expands each group with a bound so a wild
# ``[1-1000]`` doesn't explode the cited list.
# M-5 (CAL-4 follow-up): accept a trailing comma / dash / whitespace inside the
# brackets — ``[1,2,]`` is what Claude and GPT-4o emit occasionally and the
# stricter regex (no trailing separator) silently returned ``[]``. Allow a
# trailing run of comma/dash/space after the last digit; the inner parser
# already ignores empty pieces, so ``[1,2,]`` → ``[1,2]``.
_CITE_RE = re.compile(r"\[(\d+(?:[\s,\-–—]+\d+)*[\s,\-–—]*)\]")
# Inside one matched group, find each integer.
_CITE_INT_RE = re.compile(r"\d+")
# A range marker between two integers (hyphen / en-dash / em-dash).
_CITE_RANGE_SPLIT_RE = re.compile(r"[\-–—]")


def _parse_citation_markers(text: str, *, max_index: int) -> list[int]:
    """Parse all citation indices from ``text``.

    Accepts the prompted styles and common alternatives:

    * ``[1]``, ``[10]``                — single index (legacy)
    * ``[1][2][3]``                    — back-to-back single indices (legacy)
    * ``[1,2]``, ``[1, 2, 3]``         — comma list (CAL-4)
    * ``[1-3]``, ``[1–3]``             — inclusive range (CAL-4)
    * ``[1, 2-4, 6]``                  — mixed comma list with embedded range

    Each integer is expanded once; duplicates are returned in document order
    (the caller already de-duplicates with ``sorted(set(...))``). The range
    expansion is CAPPED at ``max_index`` (and at ``len(sources)``) so a
    malformed ``[1-1000]`` produces at most ``max_index`` ints, not 1000.
    Anything that isn't an integer/comma/hyphen/dash/whitespace inside the
    brackets (e.g. ``[1,2,banana]``) fails the outer regex and is silently
    skipped — the legacy strictness around junk content is preserved.
    """
    indices: list[int] = []
    # Defensive bound — never expand a single range to more than max_index
    # entries even if the model emits ``[1-99999]``. The cited list will
    # still be filtered to ``1..max_index`` downstream, but bounding here
    # avoids materializing a huge intermediate list.
    cap = max(max_index, 1)
    for match in _CITE_RE.finditer(text):
        group = match.group(1)
        # Split on commas first; each piece is either a single int or a range.
        for piece in group.split(","):
            piece = piece.strip()
            if not piece:
                continue
            # Try a range (hyphen or dash between two numbers).
            parts = _CITE_RANGE_SPLIT_RE.split(piece)
            nums = [int(n) for n in parts if n.strip().isdigit()]
            if len(nums) == 2 and nums[0] <= nums[1]:
                lo, hi = nums[0], nums[1]
                # Cap the expansion. We still record the lo end so an
                # out-of-range probe (M-3 note) can name what the model emitted.
                stop = min(hi, lo + cap - 1)
                indices.extend(range(lo, stop + 1))
            else:
                # Single int or any other shape we can parse one integer from.
                for n_str in _CITE_INT_RE.findall(piece):
                    indices.append(int(n_str))
                    break  # one integer per piece
    return indices


# Public aliases — deep-search (``agent_search``) reuses the EXACT same
# citation-marker parser and refusal detector so single-turn RAG and the
# multi-turn ReAct loop can never drift on what counts as "grounded"
# (agent_search previously carried a narrow ``\[(\d+)\]`` regex that missed
# ``[1,2]`` / ``[1-3]`` and matched only one refusal phrasing).
parse_citation_markers = _parse_citation_markers
is_refusal = _is_refusal
REFUSAL_PHRASES = _REFUSAL_PHRASES


# NOTE (measured): an aggressive "cite after EVERY sentence" variant raised the
# guard's [n] count but LOWERED judge-faithfulness (citation spam — the model
# staples [n] onto unsupported sentences). This wording cites each statement
# drawn from a source and names the section number, without forcing every
# sentence, and scored best on faithfulness (see docs/RAG_EVAL.md).
_SYSTEM = (
    "You are a careful legal research assistant for North Carolina child-welfare "
    "attorneys. Answer the QUESTION using ONLY the numbered SOURCES provided. "
    "Cite the supporting source for each statement you draw from a source, inline "
    "with its bracketed number, e.g. [1] or [2][3]; do not cite a source for a "
    "statement it does not support. When you rely on a statute or regulation, name "
    "its section number (e.g. § 7B-1111). Quote statutory or regulatory language "
    "verbatim where it is dispositive. Do NOT use any knowledge outside the sources "
    "and do NOT speculate. If the sources do not contain enough information to "
    f"answer, reply exactly: 'The provided sources {_REFUSAL} to answer this.' "
    "Be precise and concise; this will be checked against the cited text."
)


@dataclass
class Source:
    """One retrieved, citable passage offered to (and cited by) the generator.

    ``lane`` (L-2 audit): which retrieval lane surfaced this chunk —
    ``"bm25"`` (lexical-only), ``"dense"`` (dense-only), ``"hybrid"``
    (both lanes ranked it), ``"regex"`` (exact-pattern lane), or ``""``
    when the lane is not known (e.g. constructed from a tool's evidence
    pool without retrieval). The attorney trace surfaces this so a reader
    can tell *how* a passage was found, which matters when only one lane
    is operational.
    """

    n: int
    doc_id: str
    title: str
    citation: str
    passage: str            # the FULL chunk text — the grounding the attorney verifies
    source_url: str = ""
    page: str = ""
    section: str = ""
    score: float = 0.0
    lane: str = ""          # "bm25" | "dense" | "hybrid" | "regex" | ""


@dataclass
class RagAnswer:
    question: str
    answer: str                              # generated text with inline [n] citations
    sources: list[Source] = field(default_factory=list)
    model: str = ""
    grounded: bool = False                   # cites ≥1 source and is not a refusal
    cited: list[int] = field(default_factory=list)
    note: str = ""                           # fallback / status note for the UI
    # Raw model output BEFORE ``_strip_reasoning`` removed any ``<think>`` blocks
    # — preserved so the trace / debug surface can show what the reasoning model
    # actually emitted (citation parsing operates on the stripped ``answer``).
    raw_response: str = ""

    @property
    def uncited_sources(self) -> list[Source]:
        """Sources retrieved but not cited by the model.

        L-5 audit: when ``cited`` is empty, this returns *all* sources — which
        is correct but ambiguous in the UI: it cannot distinguish between
        "model refused" (a refusal answer with no citations) and "no
        citations parsed" (e.g. the model used a different bracket style).
        Callers that need to disambiguate should check ``self.grounded``
        and the ``note`` field (refusals set ``note=""`` while the grounding
        guard sets a non-empty note like "not grounded — verify manually"
        or "model cited indices outside [1..k]").
        """
        return [s for s in self.sources if s.n not in set(self.cited)]


class ChatGenerator:
    """Minimal local OpenAI-compatible chat client (Ollama by default).

    ``seed`` (M-11 audit): pass an integer for reproducible auditing — Ollama
    forwards it to the sampler, so an eval pinned to seed=N gives the same
    answer text on every run. Default ``None`` preserves today's behavior
    (no seed sent → Ollama's own scheduling determines variability)."""

    def __init__(self, model: str, *, base_url: str = "http://localhost:11434/v1",
                 api_key: str = "ollama", timeout: float = 120.0, max_tokens: int = 700,
                 seed: int | None = None):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.seed = seed

    def complete(self, system: str, user: str) -> str:
        import requests

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        # M-11: forward ``seed`` only when set, so the on-the-wire request is
        # unchanged for callers that don't opt into reproducibility.
        if self.seed is not None:
            body["seed"] = int(self.seed)
        r = requests.post(
            f"{self.base_url}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        if not content and msg.get("reasoning"):
            # A reasoning model spent the whole budget thinking and emitted no
            # answer — make it explainable rather than a silent blank.
            raise RuntimeError(
                "model returned only reasoning, no answer — increase RAG_MAX_TOKENS")
        return content


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def _to_sources(hits, max_passage_chars: int) -> list[Source]:
    out: list[Source] = []
    for i, h in enumerate(hits, 1):
        out.append(Source(
            n=i, doc_id=h.chunk.doc_id, title=h.title, citation=h.citation(),
            passage=_clip(h.chunk.text, max_passage_chars), source_url=h.source_url,
            page=h.page_span(), section=h.statutory_section(), score=round(h.score, 4),
            # L-2: carry the retrieval lane forward so the attorney trace can
            # show whether the passage came from lexical, dense, or both.
            lane=getattr(h, "lane", ""),
        ))
    return out


def _windowed_sources(index, hits, *, window: int, max_passage_chars: int) -> list[Source]:
    """Expand each hit with its neighboring chunks (same document, adjacent
    ordinals) and merge contiguous runs into one passage.

    A statute's section is often split across chunks — the heading ("§ 7B-1111 …
    upon a finding of one or more of the following:") in one, the enumerated
    grounds in the next. Retrieving a single chunk leaves the answer grounded in a
    fragment; pulling the neighbors puts the whole section in the window. Overlap
    across hits is de-duplicated so the same text never appears twice.
    """
    by_doc: dict[str, list] = {}
    for c in index.chunks.values():
        by_doc.setdefault(c.doc_id, []).append(c)
    for v in by_doc.values():
        v.sort(key=lambda c: c.ordinal)

    used: set[tuple[str, int]] = set()
    sources: list[Source] = []
    for h in hits:
        doc = h.chunk.doc_id
        chunks = by_doc.get(doc, [h.chunk])
        ords = [c.ordinal for c in chunks]
        i = ords.index(h.chunk.ordinal) if h.chunk.ordinal in ords else 0
        lo, hi = max(0, i - window), min(len(chunks), i + window + 1)
        sel = [c for c in chunks[lo:hi] if (doc, c.ordinal) not in used]
        if not sel:
            continue  # this section already covered by an earlier hit
        for c in sel:
            used.add((doc, c.ordinal))
        text = _clip("\n".join(c.text.strip() for c in sel), max_passage_chars)
        sources.append(Source(
            n=len(sources) + 1, doc_id=doc, title=h.title, citation=h.citation(),
            passage=text, source_url=h.source_url, page=h.page_span(),
            section=h.statutory_section(), score=round(h.score, 4),
            lane=getattr(h, "lane", ""),
        ))
    return sources


def build_user_prompt(question: str, sources: list[Source]) -> str:
    blocks = [f"[{s.n}] {s.citation}\n{s.passage}" for s in sources]
    return f"QUESTION: {question}\n\nSOURCES:\n\n" + "\n\n".join(blocks)


def answer(
    index, question: str, *, generator: ChatGenerator | None, k: int = 10,
    group_by_doc: bool = False, max_passage_chars: int = 1400, reranker=None,
    rerank_pool: int | None = None, neighbor_window: int = 0,
    dedup: bool | None = None, **search_kw,
) -> RagAnswer:
    """Retrieve, then (if a generator is given) synthesize a grounded, cited answer.

    With ``generator=None`` this returns the retrieved sources and an empty answer
    — i.e. retrieval-only, the toggle's "off" state. With a generator, it builds a
    sources-only prompt and returns the model's cited answer; any failure degrades
    to sources-only with a note (never a fabricated answer).

    Retrieval is chunk-level (``group_by_doc=False``) so several passages of the
    *same* document can ground the answer — e.g. both the heading and the
    enumerated body of a statute section — which one-chunk-per-document would split
    apart and leave the answer incompletely grounded.

    ``dedup`` (S-6 lever): when ``None`` (default), dedup is ON when not grouping
    by doc — preserves prior behavior. Set ``dedup=False`` explicitly for
    statutory text, where two adjacent chunks of the same section can be
    char-trigram-near-identical (shared heading + boilerplate) and dedup can
    drop a gold passage.
    """
    # A reranker (cross-encoder, local) reorders the candidate pool so the most
    # relevant passages land in the small RAG context window — the highest-
    # leverage place to improve grounding before the model ever sees the sources.
    dedup_on = (not group_by_doc) if dedup is None else dedup
    hits = index.search(question, k=k, group_by_doc=group_by_doc,
                        dedup=dedup_on, reranker=reranker,
                        rerank_pool=rerank_pool, **search_kw)
    # Section-window expansion (optional): pull each hit's neighbors so a full
    # statute section grounds the answer. MEASURED at k=10 to slightly HURT answer
    # quality (context dilution — more facts reach the window but the model
    # extracts fewer; see docs/RAG_EVAL.md), so default 0. Useful at small k where
    # a single chunk would otherwise fragment a section.
    sources = (_windowed_sources(index, hits, window=neighbor_window,
                                 max_passage_chars=max(max_passage_chars, 2200))
               if neighbor_window > 0 else _to_sources(hits, max_passage_chars))
    if not sources:
        return RagAnswer(question, "", [], note="No relevant sources found in this collection.")
    if generator is None:
        return RagAnswer(question, "", sources,
                         note="Generation off — showing retrieved sources only.")
    try:
        raw_text = generator.complete(_SYSTEM, build_user_prompt(question, sources))
    except Exception as e:  # noqa: BLE001 — never let generation failure hide the sources
        return RagAnswer(question, "", sources, model=generator.model,
                         note=f"Generation unavailable ({type(e).__name__}); showing sources only.")
    # Qwen3.6 is a reasoning model: it emits ``<think>...</think>`` blocks
    # before the answer. Strip them BEFORE citation extraction (otherwise the
    # parser sees thousands of reasoning tokens with no ``[n]`` markers and
    # the answer reads as ungrounded). The raw response is preserved on the
    # result for trace / debug.
    text = _strip_reasoning(raw_text)
    # M-3: split valid [n] (in-range) from out-of-range markers so an off-by-one
    # generator that cites [0]…[k-1] (or [99] when k=10) surfaces an explainable
    # note instead of looking like a generic "not grounded" answer.
    # CAL-4: accept comma-list and range citation styles too. Some generators
    # default to ``[1,2]`` or ``[1-3]`` instead of the prompted ``[1][2]`` —
    # the narrow ``\[(\d+)\]`` regex silently dropped those (audit
    # 2026-06-28-confidence.md §8). The expansion is bounded to the number of
    # available sources so a wild ``[1-1000]`` doesn't explode the cite list.
    raw_cites = _parse_citation_markers(text, max_index=len(sources))
    cited = sorted({n for n in raw_cites if 1 <= n <= len(sources)})
    out_of_range = sorted({n for n in raw_cites if not (1 <= n <= len(sources))})
    refused = _is_refusal(text)
    grounded = bool(cited) and not refused
    if grounded:
        note = ""
    elif refused:
        note = "The model reported the sources do not answer this question."
    elif raw_cites and not cited:
        # Citations were emitted but ALL fell outside [1..k] — the model is
        # mis-indexed against this source list. Surface the misalignment
        # explicitly (M-3) so the attorney can see WHY the answer is not
        # grounded.
        note = (
            f"Model cited indices outside [1..{len(sources)}] "
            f"({out_of_range}); refusing to ground."
        )
    else:
        note = "Answer is not grounded in the sources — verify manually."
    return RagAnswer(question, text, sources, model=generator.model,
                     grounded=grounded, cited=cited, note=note,
                     raw_response=raw_text)
