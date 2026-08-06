"""Deep Research — a multi-turn ReAct loop over the local index.

The single-turn ``rag.answer`` is fast but committed to one retrieval pass: it
fires the user's question at the hybrid retriever, takes k chunks, and asks the
model to synthesize. That works for direct questions; it under-serves a
*compound* legal question ("what are NC's TPR grounds AND the federal floor AND
what counts as 'reasonable efforts' under each?") where the right sources live
under different vocabularies and need to be assembled in steps.

This module wires the same LOCAL Qwen3.6 model into a ReAct loop:

1.  **Query rewrite** — the model decomposes the question into 2-3 sub-queries
    (keyword/semantic/citation forms) before any tool fires.
2.  **Tool turns** — the model emits ``Action: tool(args)`` lines; the harness
    runs the named tool against the index and feeds back an ``Observation``.
    Available tools cover BM25, dense, hybrid, regex (when agent A's module
    lands), per-chunk reads, and ``done`` to finalize.
3.  **Reflection** — every two tool turns the model is prompted to decide
    whether the evidence is sufficient; if yes it calls ``done`` early.

Hard caps:

*   ``max_turns`` bounds tool-turn iterations (default 6).
*   ``tool_budget`` bounds total tool calls across the loop (default 12).
*   Each tool runs under a per-call timeout (30s); failures surface as
    Observations, never silent.
*   Two consecutive malformed Actions abort with ``parse_failure``.
*   Ollama unreachable / empty content aborts with ``generator_unavailable``,
    and the harness returns the best partial answer it has assembled.

Same grounding contract as ``rag.answer`` — only the numbered ``Source [n]``
passages may ground the final answer; inline ``[n]`` citations are tracked.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# Reuse the single-turn module's machinery: same prompt language, same citation
# convention, same numbered-source contract — the deep loop is an extension, not
# a replacement. ``parse_citation_markers`` / ``is_refusal`` are the shared
# grounding primitives (comma-list / range citations + widened refusal
# detection) so single-turn and multi-turn grounding never drift.
from .rag import Source, _clip, _strip_reasoning, is_refusal, parse_citation_markers

# Match the rag module's canonical refusal sentence (embedded in the ReAct
# system prompt below) so a malformed/unsupported final answer is flagged
# consistently for the attorney UI.
_REFUSAL = "do not contain enough information"

# Hard ceiling on a single Observation block so a careless k=50 search can't
# blow the model's context window; per-source clip kept tight for the loop.
_OBSERVATION_PASSAGE_CHARS = 600
_PER_TOOL_TIMEOUT_S = 30.0


# --- public dataclasses -----------------------------------------------------
@dataclass
class ToolCall:
    """One tool invocation the harness ran on the model's behalf."""

    turn: int
    name: str
    args: dict
    ok: bool
    latency_s: float
    result_preview: str = ""        # short human-readable summary for the trace
    error: str = ""


@dataclass
class Turn:
    """One pass through the loop: model output + parsed action (if any)."""

    n: int
    kind: str                       # rewrite | act | reflect | final
    raw_output: str
    action: str = ""                # tool name, or "done" / "" when none parsed
    action_input: dict = field(default_factory=dict)


@dataclass
class Citation:
    """An inline citation extracted from the final answer."""

    n: int
    doc_id: str
    title: str
    citation: str
    page: str = ""
    section: str = ""


@dataclass
class DeepSearchResult:
    question: str
    final_answer: str
    citations: list[Citation] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)  # the full evidence pool
    turns: list[Turn] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    aborted: bool = False
    reason: str = ""                # "" | parse_failure | generator_unavailable | budget
    grounded: bool = False
    # FUNC-07: True when ``citations`` were *derived from gathered sources*
    # because the loop aborted before the model could call ``done``. The UI
    # uses this flag to caveat the citations block (the attorney sees that the
    # cites are evidence-pool entries, not model-attributed).
    citations_from_abort: bool = False
    note: str = ""


# --- tool dispatch ----------------------------------------------------------
class _ToolBox:
    """The set of retrieval operations the agent may invoke.

    The dispatcher is dict-based so the prompt's tool grammar and the runtime
    map are obviously in sync; tests assert on the same names.
    """

    def __init__(self, index):
        self.index = index
        # Saved sources keyed by their assigned [n] across the whole loop, so
        # later observations can refer to existing numbers and the final answer
        # can cite anything the model has seen.
        self.evidence: list[Source] = []
        self._seen_chunks: set[str] = set()
        # Optional regex tool — wired against the index's workspace dir (set by
        # ``DocIndex.load`` as ``_workspace``). RegexRetriever's constructor
        # expects a workspace **path**, not the DocIndex; the previous call
        # passed the index itself, which raised TypeError and was swallowed by
        # a bare ``except Exception`` — leaving the tool silently disabled the
        # whole time (audit F-4).
        self._regex = None
        workspace = getattr(index, "_workspace", None)
        if workspace is not None:
            try:
                from .pipeline.regex_retriever import RegexRetriever
                self._regex = RegexRetriever(workspace)
            except ImportError as e:
                # Module legitimately missing — log so it's visible, don't swallow.
                import logging
                logging.getLogger(__name__).warning(
                    "regex_search unavailable: %s", e)
            except Exception as e:  # noqa: BLE001
                # Any other constructor failure — surface it loudly rather than
                # silently disabling the tool (R4: no silent degradation).
                import logging
                logging.getLogger(__name__).warning(
                    "regex_search constructor failed (%s): %s",
                    type(e).__name__, e)

    @property
    def regex_available(self) -> bool:
        return self._regex is not None

    def names(self) -> list[str]:
        base = ["bm25_search", "dense_search", "hybrid_search",
                "read_chunk", "done"]
        if self.regex_available:
            base.insert(3, "regex_search")
        return base

    # -- internals ----------------------------------------------------------
    def _hits_to_sources(self, hits) -> list[Source]:
        """Append unseen hits to the running evidence; return the additions."""
        added: list[Source] = []
        for h in hits:
            cid = h.chunk.chunk_id
            if cid in self._seen_chunks:
                continue
            self._seen_chunks.add(cid)
            n = len(self.evidence) + len(added) + 1
            added.append(Source(
                n=n, doc_id=h.chunk.doc_id, title=h.title, citation=h.citation(),
                passage=_clip(h.chunk.text, _OBSERVATION_PASSAGE_CHARS),
                source_url=h.source_url, page=h.page_span(),
                section=h.statutory_section(), score=round(h.score, 4),
                # L-2: carry retrieval lane forward through the agent trace.
                lane=getattr(h, "lane", ""),
            ))
        self.evidence.extend(added)
        return added

    def _search(self, query: str, k: int, *, mode: str) -> list[Source]:
        # Mode dispatch: pure BM25 means temporarily detach the dense lane so
        # the index's existing search() rebranches into lexical-only; dense
        # alone we can't get from search() (it always fuses), so for "dense"
        # we use a heavy dense weight + low pool — a measured approximation
        # rather than a separate code path. Hybrid is the default.
        idx = self.index
        if mode == "bm25":
            saved = idx._dense
            idx._dense = None
            try:
                hits = idx.search(query, k=k)
            finally:
                idx._dense = saved
        elif mode == "dense":
            if getattr(idx, "_dense", None) is None:
                # Honest signal: no dense lane → fall back to hybrid and note it
                # in the result preview rather than silently returning lexical.
                hits = idx.search(query, k=k)
            else:
                saved_w = idx.dense_weight
                idx.dense_weight = max(saved_w, 10.0)
                try:
                    hits = idx.search(query, k=k)
                finally:
                    idx.dense_weight = saved_w
        else:  # hybrid
            hits = idx.search(query, k=k)
        return self._hits_to_sources(hits)

    # -- public tool methods -----------------------------------------------
    def bm25_search(self, query: str, k: int = 5) -> list[Source]:
        return self._search(query, max(1, min(k, 20)), mode="bm25")

    def dense_search(self, query: str, k: int = 5) -> list[Source]:
        return self._search(query, max(1, min(k, 20)), mode="dense")

    def hybrid_search(self, query: str, k: int = 5) -> list[Source]:
        return self._search(query, max(1, min(k, 20)), mode="hybrid")

    def regex_search(self, pattern: str, k: int = 5) -> list[Source]:
        if self._regex is None:
            raise RuntimeError("regex_search is not available (regex_retriever not installed)")
        hits = self._regex.search(pattern, k=max(1, min(k, 20)))
        # L-2: regex hits don't carry a ``lane`` attribute — stamp it so the
        # downstream Source reflects how the passage was found.
        added = self._hits_to_sources(hits)
        for s in added:
            if not s.lane:
                s.lane = "regex"
        return added

    def read_chunk(self, doc_id: str, chunk_id: str) -> Source:
        # Lets the model zoom in on a specific cited chunk it saw a snippet of —
        # essential for statutes where the heading lands in one chunk and the
        # operative list in the next.
        ch = self.index.chunks.get(chunk_id)
        if ch is None or ch.doc_id != doc_id:
            raise KeyError(f"unknown chunk {chunk_id!r} for doc {doc_id!r}")
        # M-7: the legacy short-circuit ("chunk_id.endswith(s.doc_id) is False"
        # was always True because chunk_id is "{doc_id}:{ordinal}" — so the
        # whole guard collapsed to a loose passage-prefix check that let the
        # same chunk be re-added under a fresh [n]). Use the authoritative
        # ``_seen_chunks`` set so a re-read returns the existing Source and
        # preserves its [n] across the evidence pool.
        if chunk_id in self._seen_chunks:
            for s in self.evidence:
                # Same chunk_id is the only correctness key. (Multiple Sources
                # can share doc_id legitimately when different chunks of the
                # same statute were retrieved.)
                if s.doc_id == doc_id and s.passage and s.passage[:60] in ch.text:
                    return s
        meta = self.index.documents.get(doc_id)
        title = meta.title if meta else doc_id
        # L-3 audit: previously the citation field was just the title and
        # ``page``/``section`` were empty — the evidence-pool render in the
        # Deep Research trace then listed ``[n] {title}`` with no locator.
        # Mirror what ``SearchHit.citation()`` / ``page_span`` /
        # ``statutory_section`` produce so a read_chunk-sourced Source carries
        # the same provenance fields a hit-sourced Source does.
        page = ""
        if ch.page_start:
            page = (f"pp. {ch.page_start}–{ch.page_end}"
                    if ch.page_end and ch.page_end != ch.page_start
                    else f"p. {ch.page_start}")
        # Statutory section: leading heading when it looks like a § / NCAC /
        # USC / CFR locator (mirror SearchHit.statutory_section()).
        from .pipeline.docindex import _LEGAL_SECTION_RE
        section = ""
        if ch.heading_path and _LEGAL_SECTION_RE.match(ch.heading_path[0]):
            section = ch.heading_path[0]
        # Citation line: prefer "§ 7B-1111 — title" when statutory, else
        # "title › heading › path" so the evidence-pool [n] line carries the
        # locator.
        if section:
            citation_parts = [section, title]
        else:
            citation_parts = [title]
            if ch.heading_path:
                citation_parts.append(" › ".join(ch.heading_path))
        if page:
            citation_parts.append(page)
        if meta:
            where = meta.source_url or meta.source_domain
            if where:
                when = f", retrieved {meta.retrieved_at}" if meta.retrieved_at else ""
                citation_parts.append(f"{where}{when}")
        citation = " — ".join(citation_parts)
        source_url = meta.source_url if meta else ""
        n = len(self.evidence) + 1
        s = Source(n=n, doc_id=doc_id, title=title,
                   citation=citation,
                   passage=_clip(ch.text, _OBSERVATION_PASSAGE_CHARS * 2),
                   source_url=source_url, page=page, section=section,
                   score=0.0)
        self.evidence.append(s)
        self._seen_chunks.add(chunk_id)
        return s


# --- prompts ----------------------------------------------------------------
_REWRITE_SYSTEM = (
    "You decompose a complex legal research question into 2-3 SHORT search "
    "queries for a North Carolina child-welfare retrieval system. The output "
    "MUST be a JSON array of strings — no prose, no markdown. Prefer one "
    "keyword-style query, one natural-language paraphrase, and (when relevant) "
    "one with a statute or section number. Each query under 12 words."
)


def _react_system(tool_names: list[str]) -> str:
    """ReAct system prompt: tool grammar + grounding contract + hard caps."""
    tools = ", ".join(tool_names)
    return (
        "You are a careful legal research assistant running a multi-turn search "
        "loop against a LOCAL index of North Carolina child-welfare statutes, "
        "regulations, and policy. You investigate by issuing one TOOL CALL per "
        "turn and reading the OBSERVATION returned to you.\n\n"
        "AVAILABLE TOOLS:\n"
        f"  {tools}\n\n"
        "TOOL GRAMMAR — produce EXACTLY these two lines per turn, nothing else:\n"
        "  Action: <tool_name>\n"
        "  Action-Input: {\"<arg>\": <value>, ...}\n\n"
        "Examples:\n"
        "  Action: hybrid_search\n"
        "  Action-Input: {\"query\": \"grounds for termination of parental rights\", \"k\": 6}\n"
        "  Action: read_chunk\n"
        "  Action-Input: {\"doc_id\": \"d3\", \"chunk_id\": \"d3:14\"}\n"
        "  Action: done\n"
        "  Action-Input: {\"answer\": \"The court may terminate ... [1][2].\", "
        "\"citations\": [1, 2]}\n\n"
        "RULES:\n"
        "  - Use Action-Input EXACTLY (valid JSON, double-quoted keys).\n"
        "  - Cite only the [n] numbers that appear in your gathered Sources.\n"
        "  - When you have enough evidence, call `done` with the final answer.\n"
        "  - The final answer must cite each statement drawn from a source as "
        "[n]; name section numbers (e.g. § 7B-1111) when present.\n"
        "  - If the sources do not answer the question, in `done` reply exactly: "
        f"'The provided sources {_REFUSAL} to answer this.'\n"
        "  - Do NOT use outside knowledge."
    )


_REFLECT_NUDGE = (
    "REFLECT: review the Sources gathered so far. If they cover the question "
    "(grounds, definitions, section numbers as needed), call `done` with the "
    "final answer this turn. Otherwise issue one more focused search."
)


# --- parsing ---------------------------------------------------------------
_ACTION_RE = re.compile(r"^\s*Action\s*:\s*(\w+)\s*$", re.MULTILINE)
# Action-Input may span multiple lines; capture from the colon to the next
# Action: line or end of output.
_INPUT_RE = re.compile(
    r"Action-?Input\s*:\s*(.*?)(?=\n\s*Action\s*:|\Z)", re.DOTALL | re.IGNORECASE,
)


def _parse_action(text: str) -> tuple[str, dict] | None:
    """Extract (tool_name, args) from a model turn. None on parse failure."""
    am = _ACTION_RE.search(text)
    if not am:
        return None
    name = am.group(1).strip()
    im = _INPUT_RE.search(text, am.end())
    if not im:
        return (name, {})
    payload = im.group(1).strip()
    # Trim a leading code fence if the model wrapped JSON in ```json … ```.
    payload = re.sub(r"^```(?:json)?\s*", "", payload)
    payload = re.sub(r"\s*```\s*$", "", payload).strip()
    # The model sometimes appends a stray "Observation:" line — strip it.
    payload = re.split(r"\n\s*Observation\s*:", payload, maxsplit=1)[0].strip()
    if not payload:
        return (name, {})
    import json as _json
    try:
        args = _json.loads(payload)
    except Exception:  # noqa: BLE001
        # Last-ditch: try to repair simple single-quote JSON. Otherwise we
        # surface the parse failure to the caller — no silent guess.
        try:
            args = _json.loads(payload.replace("'", '"'))
        except Exception:  # noqa: BLE001
            return None
    if not isinstance(args, dict):
        return None
    return (name, args)


def _parse_rewrite(text: str) -> list[str]:
    """The rewrite turn returns a JSON list of sub-queries; fail soft."""
    import json as _json

    # Strip code fences if present.
    payload = re.sub(r"^```(?:json)?\s*", "", text.strip())
    payload = re.sub(r"\s*```\s*$", "", payload).strip()
    try:
        arr = _json.loads(payload)
    except Exception:  # noqa: BLE001
        # Try to find a bracketed list inside the text — reasoning models often
        # wrap the array in a sentence ("Here are the queries: [...]").
        m = re.search(r"\[(.*?)\]", payload, re.DOTALL)
        if not m:
            return []
        try:
            arr = _json.loads("[" + m.group(1) + "]")
        except Exception:  # noqa: BLE001
            return []
    if not isinstance(arr, list):
        return []
    return [str(x).strip() for x in arr if str(x).strip()][:3]


# --- observation rendering --------------------------------------------------
def _render_observation(added: list[Source], tool: str) -> str:
    """Format a tool's freshly-added Sources for the next model turn."""
    if not added:
        return f"Observation ({tool}): no new sources matched."
    lines = [f"Observation ({tool}): {len(added)} new source(s)."]
    for s in added:
        loc = s.section or s.page
        head = f"[{s.n}] {s.title}{(' ' + loc) if loc else ''}  ({s.score:.3f})"
        lines.append(head)
        lines.append(f"    {s.passage}")
    return "\n".join(lines)


def _render_evidence_pool(sources: list[Source]) -> str:
    """The compact running list of every Source the model can cite as [n]."""
    if not sources:
        return "SOURCES gathered so far: (none yet)."
    lines = ["SOURCES gathered so far:"]
    for s in sources:
        loc = s.section or s.page
        lines.append(f"[{s.n}] {s.citation}{(' ' + loc) if loc else ''}")
    return "\n".join(lines)


# --- the loop ---------------------------------------------------------------
def deep_search(
    index, question: str, *, generator=None, max_turns: int = 6,
    tool_budget: int = 12, max_rewrites: int = 1,
    now: Callable[[], float] = time.monotonic,
) -> DeepSearchResult:
    """Run the ReAct loop and return a fully traced ``DeepSearchResult``.

    ``generator`` is the local chat client (Ollama-compatible). When None,
    no rewrite/loop is run — the harness aborts with ``generator_unavailable``,
    consistent with the single-turn module's "no silent fabrication" rule.
    """
    res = DeepSearchResult(question=question, final_answer="")
    # M-9: an empty / whitespace question must not fire an LLM round-trip on
    # the rewrite step (the dashboard short-circuits, but CLI/script callers
    # could otherwise burn a generator call on a blank box). Return cleanly
    # so the caller sees an explainable empty result.
    if not (question or "").strip():
        res.aborted = True
        res.reason = "empty_query"
        res.note = "Empty question — nothing to search."
        return res
    if generator is None:
        res.aborted = True
        res.reason = "generator_unavailable"
        res.note = "No generator configured — deep_search needs a local chat model."
        return res

    box = _ToolBox(index)
    react_system = _react_system(box.names())

    # 1) Query-rewrite turn — decompose the question first.
    sub_queries: list[str] = []
    for _ in range(max(1, max_rewrites)):
        try:
            raw = generator.complete(_REWRITE_SYSTEM, f"QUESTION: {question}")
        except Exception as e:  # noqa: BLE001
            res.aborted = True
            res.reason = "generator_unavailable"
            res.note = f"Generator failed on rewrite ({type(e).__name__}: {e})."
            return res
        # Qwen3.6 emits ``<think>…</think>`` reasoning before the JSON; strip it
        # so ``_parse_rewrite`` sees the actual sub-query array, not 3,000
        # tokens of prose prelude. ``raw`` is preserved on the Turn for trace.
        res.turns.append(Turn(n=len(res.turns) + 1, kind="rewrite", raw_output=raw))
        sub_queries = _parse_rewrite(_strip_reasoning(raw))
        if sub_queries:
            break
    if not sub_queries:
        # If the model can't even produce sub-queries, fall back to the
        # original question — degraded but not aborted.
        sub_queries = [question]
        res.note = "Rewrite produced no sub-queries; using original question."
    res.sub_queries = sub_queries

    # 2) Seed evidence with one hybrid pass per sub-query so the model has
    # something concrete to react to on its first tool turn. This is the same
    # work it would otherwise do as its first 1-3 actions; doing it eagerly
    # saves turns and gives the loop a sane floor.
    seeded_total = 0
    for q in sub_queries:
        try:
            added = box.hybrid_search(q, k=4)
        except Exception as e:  # noqa: BLE001
            res.tool_calls.append(ToolCall(turn=0, name="hybrid_search",
                                           args={"query": q, "k": 4}, ok=False,
                                           latency_s=0.0, error=str(e)))
            continue
        seeded_total += len(added)
        res.tool_calls.append(ToolCall(
            turn=0, name="hybrid_search", args={"query": q, "k": 4},
            ok=True, latency_s=0.0,
            result_preview=f"seeded {len(added)} source(s)",
        ))
    if seeded_total == 0:
        # Nothing matched even with three rewrites — no point looping.
        res.note = "No documents matched any sub-query."
        return res

    # 3) ReAct loop.
    parse_failures = 0
    user_history: list[str] = []        # we replay tool results as user turns
    for turn_n in range(1, max_turns + 1):
        # Build the user message for this turn.
        pool = _render_evidence_pool(box.evidence)
        nudge = _REFLECT_NUDGE if (turn_n % 2 == 0) else ""
        prior = "\n\n".join(user_history[-3:])  # last few obs kept in the prompt
        user_msg = (
            f"QUESTION: {question}\n\nSUB-QUERIES: {sub_queries}\n\n"
            f"{pool}\n\n{prior}\n\n{nudge}\n\n"
            "Issue ONE Action now."
        ).strip()

        try:
            raw = generator.complete(react_system, user_msg)
        except Exception as e:  # noqa: BLE001
            res.aborted = True
            res.reason = "generator_unavailable"
            res.note = (
                f"Generator failed mid-loop ({type(e).__name__}: {e}); "
                "returning partial evidence."
            )
            res.sources = list(box.evidence)
            _finalize_abort_citations(res)
            return res

        # Strip Qwen3.6 ``<think>…</think>`` reasoning before parsing the
        # ``Action:`` / ``Action-Input:`` block. Without this, the reasoning
        # model's chain-of-thought drowns the action regex and every turn
        # registers as parse_failure — the bug the attorney eval surfaced.
        # ``raw`` is preserved on the Turn for trace.
        parsed = _parse_action(_strip_reasoning(raw))
        if parsed is None:
            parse_failures += 1
            res.turns.append(Turn(n=len(res.turns) + 1, kind="act", raw_output=raw))
            user_history.append(
                "Observation: Your last output did not contain a valid `Action:` / "
                "`Action-Input:` block. Re-issue exactly two lines."
            )
            if parse_failures >= 2:
                res.aborted = True
                res.reason = "parse_failure"
                res.note = "Two consecutive malformed Actions; aborting."
                res.sources = list(box.evidence)
                _finalize_abort_citations(res)
                return res
            continue
        parse_failures = 0
        name, args = parsed
        res.turns.append(Turn(n=len(res.turns) + 1, kind="act",
                              raw_output=raw, action=name, action_input=args))

        # --- done ---
        if name == "done":
            answer_text = str(args.get("answer", "")).strip()
            res.final_answer = answer_text
            res.sources = list(box.evidence)
            res.citations = _extract_citations(answer_text, res.sources)
            # Shared refusal detector (rag.is_refusal): covers the canonical
            # sentence plus common paraphrases with the first-sentence /
            # zero-citation gate — the old single-string match missed
            # rephrasings like "I'm unable to find ...".
            refused = is_refusal(answer_text)
            res.grounded = bool(res.citations) and not refused
            if not res.grounded:
                res.note = (
                    "The model reported the sources do not answer this question."
                    if refused else
                    "Final answer is not grounded in the sources — verify manually."
                )
            return res

        # --- budget check before running the tool ---
        if len(res.tool_calls) >= tool_budget + len(sub_queries):  # +seed tools
            res.aborted = True
            res.reason = "budget"
            res.note = f"Tool budget ({tool_budget}) exhausted; no `done` issued."
            res.sources = list(box.evidence)
            _finalize_abort_citations(res)
            return res

        # --- dispatch the tool ---
        tc_start = now()
        try:
            obs = _dispatch_tool(box, name, args)
            latency = now() - tc_start
            if latency > _PER_TOOL_TIMEOUT_S:
                obs = (f"Observation ({name}): exceeded {_PER_TOOL_TIMEOUT_S:.0f}s "
                       "timeout (note: enforced post-hoc, not interrupted).")
                res.tool_calls.append(ToolCall(
                    turn=turn_n, name=name, args=args, ok=False,
                    latency_s=latency, error="timeout"))
            else:
                if isinstance(obs, list):  # search tools → list[Source]
                    res.tool_calls.append(ToolCall(
                        turn=turn_n, name=name, args=args, ok=True,
                        latency_s=latency,
                        result_preview=f"{len(obs)} new source(s)"))
                    obs = _render_observation(obs, name)
                else:                      # read_chunk → single Source
                    res.tool_calls.append(ToolCall(
                        turn=turn_n, name=name, args=args, ok=True,
                        latency_s=latency,
                        result_preview=f"[{obs.n}] {obs.title[:40]}"))
                    obs = _render_observation([obs], name)
        except Exception as e:  # noqa: BLE001
            latency = now() - tc_start
            res.tool_calls.append(ToolCall(
                turn=turn_n, name=name, args=args, ok=False,
                latency_s=latency, error=f"{type(e).__name__}: {e}"))
            obs = f"Observation ({name}): tool error — {type(e).__name__}: {e}"
        user_history.append(obs)

    # Hit max_turns without `done`.
    res.aborted = True
    res.reason = "budget"
    res.note = f"Reached max_turns={max_turns} without final answer."
    res.sources = list(box.evidence)
    _finalize_abort_citations(res)
    return res


def _dispatch_tool(box: _ToolBox, name: str, args: dict) -> Any:
    """Translate a model-issued (tool, args) into a method call on the box.

    Unknown tools and bad-args raise, which the loop converts into an
    Observation — the model gets a fair chance to retry rather than silent failure.
    """
    if name == "bm25_search":
        return box.bm25_search(str(args.get("query", "")), int(args.get("k", 5)))
    if name == "dense_search":
        return box.dense_search(str(args.get("query", "")), int(args.get("k", 5)))
    if name == "hybrid_search":
        return box.hybrid_search(str(args.get("query", "")), int(args.get("k", 5)))
    if name == "regex_search":
        return box.regex_search(str(args.get("pattern", "")), int(args.get("k", 5)))
    if name == "read_chunk":
        return box.read_chunk(str(args.get("doc_id", "")),
                              str(args.get("chunk_id", "")))
    raise KeyError(f"unknown tool {name!r} (expected one of {box.names()})")


def _finalize_abort_citations(res: DeepSearchResult) -> None:
    """FUNC-06 / FUNC-07: at every abort site that has gathered evidence but
    no model-driven ``done`` call, materialise a Citation list from the top
    sources so the UI can render an actionable, ordered cite block instead of
    a wall of raw evidence. Marks ``citations_from_abort=True`` and pins
    ``grounded=False`` — the model never claimed to ground these.
    """
    if not res.aborted:
        return
    if res.citations:
        # The done path already populated cites; do not override.
        return
    if not res.sources:
        return
    res.citations = _derive_citations_from_sources(res.sources, top_k=10)
    res.citations_from_abort = True
    res.grounded = False


def _derive_citations_from_sources(
    sources: list[Source], *, top_k: int = 10,
) -> list[Citation]:
    """FUNC-06 / FUNC-07: when the loop aborts with evidence in hand but no
    ``done`` call, the UI used to render raw sources without any inline cite
    list — so the page felt half-broken. We synthesise a Citation per top-N
    source (ranked by the score the retriever assigned) so the attorney still
    sees an ordered, citable evidence list. Marked separately
    (``citations_from_abort=True`` on the result) so the UI can caveat them.
    """
    if not sources:
        return []
    # Rank by score (already populated from the retrieval lane); ties keep the
    # original [n] order, which mirrors what the model would have seen.
    ordered = sorted(
        sources, key=lambda s: (-(s.score or 0.0), s.n),
    )[:max(1, top_k)]
    return [
        Citation(n=s.n, doc_id=s.doc_id, title=s.title,
                 citation=s.citation, page=s.page, section=s.section)
        for s in ordered
    ]


def _extract_citations(answer_text: str, sources: list[Source]) -> list[Citation]:
    """Pull inline citation markers from the final answer; map to gathered sources.

    Uses the shared ``rag.parse_citation_markers`` so comma-lists (``[1,2]``)
    and ranges (``[1-3]``) count exactly as they do in single-turn RAG — the
    legacy narrow ``\\[(\\d+)\\]`` regex here silently dropped those styles,
    making a properly-cited deep-search answer look ungrounded.
    """
    by_n = {s.n: s for s in sources}
    seen: list[int] = []
    max_index = max(by_n) if by_n else 1
    for n in parse_citation_markers(answer_text, max_index=max_index):
        if n in by_n and n not in seen:
            seen.append(n)
    out: list[Citation] = []
    for n in seen:
        s = by_n[n]
        out.append(Citation(n=n, doc_id=s.doc_id, title=s.title,
                            citation=s.citation, page=s.page, section=s.section))
    return out


__all__ = [
    "deep_search", "DeepSearchResult", "Turn", "ToolCall", "Citation",
    "_derive_citations_from_sources", "_finalize_abort_citations",
]
