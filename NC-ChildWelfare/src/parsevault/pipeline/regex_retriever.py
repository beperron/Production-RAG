"""Regex retrieval lane — the third leg of hybrid search.

Why this exists: BM25F + dense embeddings both under-rank queries that are
*exact* by nature — statute citations (``7B-1111``), federal code refs
(``42 U.S.C. 671``, ``45 CFR 1356``), Bates numbers (``DHHS_2026_00208256``),
and quoted phrases. These queries have a single correct answer (the doc that
literally contains the string); a lexical fusion that treats them as multi-term
soft-matches dilutes the signal. The regex lane gives them score 1.0 and ranks
by hit count — a deterministic, explainable shortcut.

Two execution paths, identical results:

* **ripgrep** (when available) — fast subprocess walk of ``outputs/*.md``.
* **Python ``re``** — fallback when ``rg`` isn't on PATH; same regex, same scores.

Hits are filename→doc_id keyed; the optional ``index`` arg back-resolves the
specific chunk that contains the match, so a RegexHit is fusable with SearchHit
through ``DocIndex.search_regex``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .layout import doc_id_for_md_path, iter_output_md, resolve_md_path

if TYPE_CHECKING:
    from .docindex import DocIndex


# --------------------------------------------------------------------------- #
# pattern catalogue — high-confidence query shapes
# --------------------------------------------------------------------------- #
# NC General Statute / Admin Code section: '7B-1111', '108A-86', '7B-1111.1'.
# Bare digit-letter combo is too loose; require a hyphen + digit suffix.
_NC_STATUTE = re.compile(r"\b\d{1,4}[A-Z]{0,2}-\d{1,4}(?:\.\d+)?\b")

# Federal U.S.C. citation: '42 USC 671', '42 U.S.C. 671', '42 U. S. C. § 671a'.
_USC = re.compile(
    r"\b\d{1,2}\s*U\.?\s*S\.?\s*C\.?\s*§?\s*\d{1,5}[a-z]?(?:\(\w+\))?\b",
    re.IGNORECASE,
)

# CFR: '45 CFR 1356', '45 C.F.R. § 1356.21'.
_CFR = re.compile(
    r"\b\d{1,2}\s*C\.?\s*F\.?\s*R\.?\s*§?\s*\d{1,5}(?:\.\d+)?\b",
    re.IGNORECASE,
)

# Bates / discovery numbers: 'DHHS_2026_00208256', 'ABC000123', 'PROD-12345'.
# Require an uppercase prefix (≥2 chars) + separator + ≥4 digits — typed enough
# to avoid catching ordinary acronyms like 'NC' or 'USC' alone.
_BATES = re.compile(r"\b[A-Z]{2,}[_\-]?\d{4,}(?:[_\-]\d+)*\b")

# Quoted phrase: '"exact phrase"' or '“…”'.
_QUOTED = re.compile(r'"([^"]{2,})"|“([^”]{2,})”')


@dataclass
class RegexHit:
    """One exact-pattern match, structurally close enough to ``SearchHit`` to fuse.

    ``score`` is 1.0 for pattern-y queries (matches are equally exact; rank by
    hit count); for loose queries, BM25-style saturated count in [0, 1].
    """

    doc_id: str
    chunk_id: str
    score: float
    snippet: str
    page_start: int = 0
    page_end: int = 0
    line_no: int = 0
    hit_count: int = 1
    pattern: str = ""  # the regex that fired (for debugging / dashboard chips)
    title: str = ""
    source_path: str = ""
    matched_text: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# query classification
# --------------------------------------------------------------------------- #
def classify_query(query: str) -> tuple[str, list[str]]:
    """Return ``(kind, patterns)``: kind is one of
    ``"statute" | "usc" | "cfr" | "bates" | "quoted" | "loose"``;
    ``patterns`` is the list of literal sub-strings to search (verbatim).

    For pattern-y queries we search the LITERAL match (most precise). For
    ``loose`` queries the caller builds a word-boundary alternation.
    """
    q = query.strip()
    if not q:
        return ("loose", [])

    # Quoted phrase wins outright — the user is asking for the literal string.
    if m := _QUOTED.search(q):
        phrase = m.group(1) or m.group(2)
        return ("quoted", [phrase])

    # Try each typed pattern in priority order — most specific first so a
    # citation embedded in a longer query still scores as pattern-y.
    for kind, rx in (("usc", _USC), ("cfr", _CFR),
                     ("statute", _NC_STATUTE), ("bates", _BATES)):
        hits = rx.findall(q)
        if hits:
            # Normalise whitespace inside USC/CFR ('42 U. S. C. 671' → '42 U.S.C. 671'
            # would over-canonicalise; instead, keep the user's form AND a compact
            # variant so both surface in the corpus search).
            literals = []
            for h in hits:
                literals.append(h)
                compact = re.sub(r"\s+", " ", h).strip()
                if compact != h:
                    literals.append(compact)
            return (kind, list(dict.fromkeys(literals)))

    return ("loose", [t for t in re.findall(r"\w+", q) if len(t) > 2])


def _build_regex(kind: str, literals: list[str]) -> re.Pattern[str]:
    """Compile a corpus-search regex for the classified query.

    Statute / USC / CFR citations are searched verbatim — but we expand
    whitespace and the optional period/`§` so corpus variants
    (``42 USC 671`` vs ``42 U.S.C. § 671``) still match. Bates / quoted
    are searched literally with word boundaries.
    """
    if kind in ("statute", "bates"):
        parts = [rf"\b{re.escape(lit)}\b" for lit in literals]
        return re.compile("|".join(parts))
    if kind == "quoted":
        # Anchor to the literal phrase; allow flexible inner whitespace.
        parts = [re.escape(lit).replace(r"\ ", r"\s+") for lit in literals]
        return re.compile("|".join(parts), re.IGNORECASE)
    if kind in ("usc", "cfr"):
        # '42 U.S.C. 671' → tolerate optional dots/spaces/`§`/`section`.
        flex = []
        for lit in literals:
            # Tokenise into (number)(USC|CFR letters)(number) groups, rebuild
            # with flexible separators so 'U.S.C.' matches 'USC' too.
            tokens = re.findall(r"\d+[a-z]?(?:\(\w+\))?|[A-Za-z]", lit)
            if not tokens:
                continue
            parts: list[str] = []
            for tok in tokens:
                if tok.isalpha():
                    parts.append(re.escape(tok) + r"\.?")
                else:
                    parts.append(r"§?\s*" + re.escape(tok))
            flex.append(r"\b" + r"\s*".join(parts) + r"\b")
        return re.compile("|".join(flex), re.IGNORECASE)
    # loose: word-boundary alternation across query tokens.
    parts = [rf"\b{re.escape(t)}\b" for t in literals]
    return re.compile("|".join(parts), re.IGNORECASE)


# --------------------------------------------------------------------------- #
# corpus search — ripgrep (preferred) and python fallback
# --------------------------------------------------------------------------- #
def _rg_path() -> str | None:
    return shutil.which("rg")


def _ripgrep_search(pattern: re.Pattern[str], outputs: Path) -> dict[str, list[tuple[int, str]]]:
    """Subprocess walk: ``rg`` over ``outputs/*.md``. Returns
    ``{doc_id: [(line_no, line_text), …]}`` (one entry per match)."""
    rg = _rg_path()
    if not rg:
        return {}
    cmd = [
        rg, "--no-heading", "--line-number", "--color=never",
        "-e", pattern.pattern, "--type=md", "-g", "*.md", str(outputs),
    ]
    # When the pattern was compiled with IGNORECASE we must tell rg too.
    if pattern.flags & re.IGNORECASE:
        cmd.insert(1, "--ignore-case")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}
    out: dict[str, list[tuple[int, str]]] = {}
    for raw in proc.stdout.splitlines():
        # rg format: '<path>:<lineno>:<text>'
        try:
            path_str, lineno_str, text = raw.split(":", 2)
        except ValueError:
            continue
        # Layout-aware doc_id: flat stem OR sha[:16] for sharded shards
        # (LAYOUT-2: sharded outputs use the full source-sha as filename,
        # but the index identifies docs by the 16-hex prefix).
        doc_id = doc_id_for_md_path(Path(path_str))
        try:
            lineno = int(lineno_str)
        except ValueError:
            continue
        out.setdefault(doc_id, []).append((lineno, text))
    return out


def _python_search(pattern: re.Pattern[str], outputs: Path) -> dict[str, list[tuple[int, str]]]:
    """Pure-Python equivalent of ``_ripgrep_search`` for environments lacking rg.

    Same return shape so downstream scoring is identical. Walks BOTH the
    flat ``outputs/*.md`` layout AND the sharded ``outputs/<2hex>/*.md``
    layout used by scanned two-column documents (LAYOUT-2)."""
    out: dict[str, list[tuple[int, str]]] = {}
    for md in iter_output_md(outputs):
        doc_id = doc_id_for_md_path(md)
        try:
            text = md.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                out.setdefault(doc_id, []).append((lineno, line))
    return out


# --------------------------------------------------------------------------- #
# page-anchor resolution
# --------------------------------------------------------------------------- #
_PAGE_ANCHOR = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")


def _page_for_line(md_path: Path, line_no: int) -> int:
    """Walk the file to find the last ``<!-- page: N -->`` anchor at or above
    ``line_no``. Returns 0 when no anchor is found (legacy / unanchored docs)."""
    try:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0
    page = 0
    for n, line in enumerate(text.splitlines(), start=1):
        if n > line_no:
            break
        if m := _PAGE_ANCHOR.search(line):
            try:
                page = int(m.group(1))
            except ValueError:
                continue
    return page


# --------------------------------------------------------------------------- #
# the retriever
# --------------------------------------------------------------------------- #
class RegexRetriever:
    """Exact-pattern lane over an indexed workspace's ``outputs/`` shards.

    Construction is cheap (no I/O). Each ``.search`` runs one rg/regex pass
    over the markdown shards, scores by classified-pattern + hit count, and
    optionally back-resolves chunk_ids when an ``index`` is supplied.
    """

    def __init__(self, workspace: str | Path, *, use_ripgrep: bool | None = None):
        self.workspace = Path(workspace)
        self.outputs = self.workspace / "outputs"
        # ``None`` → auto (use rg when found); True/False force the path so
        # tests can exercise both lanes deterministically.
        self._use_rg = (_rg_path() is not None) if use_ripgrep is None else use_ripgrep

    # ----- public API ------------------------------------------------------ #
    @property
    def has_ripgrep(self) -> bool:
        return _rg_path() is not None

    def search(
        self,
        query: str,
        k: int = 10,
        *,
        index: "DocIndex | None" = None,
    ) -> list[RegexHit]:
        """Top-k regex hits for ``query``. Returns [] when ``outputs/`` is absent
        (un-built workspace) or the query has no recoverable signal."""
        if not self.outputs.is_dir():
            return []
        kind, literals = classify_query(query)
        if not literals:
            return []
        pattern = _build_regex(kind, literals)
        matches = (_ripgrep_search(pattern, self.outputs) if self._use_rg
                   else _python_search(pattern, self.outputs))
        if not matches:
            return []

        # Score: high-confidence kinds (statute/usc/cfr/bates/quoted) anchor at
        # 1.0; rank tiebreak by hit count. Loose queries: BM25-style saturated
        # count so a doc with many matches outranks one with a single hit but
        # the curve flattens past ~10 matches.
        scored: list[RegexHit] = []
        for doc_id, lines in matches.items():
            hit_count = len(lines)
            if kind == "loose":
                # k1=1.5, saturate around 10 matches.
                base = hit_count / (hit_count + 1.5)
            else:
                base = 1.0
            # First line gives the snippet + line_no; the others contribute to
            # hit_count. Tiebreak on count by adding a tiny rank dust.
            line_no, line_text = lines[0]
            score = base + min(hit_count, 50) * 1e-4
            # Layout-aware md path resolution (LAYOUT-2): try flat first,
            # then sharded via the index's source_sha256 if available.
            sha = ""
            if index is not None:
                m = index.documents.get(doc_id)
                if m is not None:
                    sha = getattr(m, "source_sha256", "") or ""
            resolved = resolve_md_path(
                self.outputs, doc_id=doc_id, source_sha256=sha,
            )
            md_path = resolved or (self.outputs / f"{doc_id}.md")
            page = _page_for_line(md_path, line_no)
            chunk_id, page_start, page_end, title = self._resolve_chunk(
                index, doc_id, line_text, page,
            )
            matched_text = list(dict.fromkeys(
                m.group(0) for m in pattern.finditer("\n".join(t for _, t in lines[:5]))
            ))
            scored.append(
                RegexHit(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    score=round(score, 6),
                    snippet=_make_snippet(line_text, pattern),
                    page_start=page_start or page,
                    page_end=page_end or page,
                    line_no=line_no,
                    hit_count=hit_count,
                    pattern=kind,
                    title=title,
                    source_path=str(md_path),
                    matched_text=matched_text,
                )
            )

        scored.sort(key=lambda h: (-h.score, -h.hit_count, h.doc_id))
        return scored[:k]

    # ----- internals ------------------------------------------------------- #
    @staticmethod
    def _resolve_chunk(
        index: "DocIndex | None",
        doc_id: str,
        line_text: str,
        page_hint: int,
    ) -> tuple[str, int, int, str]:
        """Back-resolve the line to the index's chunk_id + page span + title.

        Two strategies, in order:
          1) page match — pick the chunk whose ``[page_start, page_end]`` covers
             ``page_hint`` (the most reliable signal: chunking is page-aware).
          2) substring match — pick the chunk whose ``text`` contains the line
             (the line is one of the chunk's source lines).

        Falls back to empty fields when no index is supplied or no chunk fits.
        """
        if index is None:
            return ("", 0, 0, "")
        meta = index.documents.get(doc_id)
        title = meta.title if meta is not None else ""
        # PERF-D2-1: prefer the O(per-doc) meta.chunk_ids fast path (same
        # CACHE-1 pattern as ``_chunks_for_doc``); fall back to the full
        # ``index.chunks`` scan only for legacy indexes where chunk_ids is
        # empty. Without this, every regex hit pays an O(all chunks) scan
        # (≈110K touches × 30 hits = 3.3M per /search-results at PoC scale).
        all_chunks = getattr(index, "chunks", {}) or {}
        cid_list = getattr(meta, "chunk_ids", None) if meta is not None else None
        if cid_list:
            candidates = [all_chunks[c] for c in cid_list if c in all_chunks]
        else:
            candidates = [c for c in all_chunks.values() if c.doc_id == doc_id]
        if not candidates:
            return ("", 0, 0, title)

        # Strategy 1: page coverage.
        if page_hint:
            for c in candidates:
                if c.page_start and c.page_start <= page_hint <= (c.page_end or c.page_start):
                    return (c.chunk_id, c.page_start, c.page_end, title)

        # Strategy 2: substring match on a normalized form (markdown can wrap
        # whitespace differently than the chunk store).
        needle = re.sub(r"\s+", " ", line_text).strip()
        if len(needle) >= 12:
            for c in candidates:
                hay = re.sub(r"\s+", " ", c.text)
                if needle in hay:
                    return (c.chunk_id, c.page_start, c.page_end, title)

        # Best-effort: first chunk of the doc — keeps the citation traceable
        # even if neither strategy fires.
        first = min(candidates, key=lambda c: c.ordinal)
        return (first.chunk_id, first.page_start, first.page_end, title)


def _make_snippet(line_text: str, pattern: re.Pattern[str], width: int = 240) -> str:
    """Center a width-bounded snippet on the first match in ``line_text``."""
    flat = re.sub(r"\s+", " ", line_text).strip()
    m = pattern.search(flat)
    if not m:
        return flat[:width]
    start = max(0, m.start() - width // 3)
    snip = flat[start : start + width]
    return ("…" if start > 0 else "") + snip + ("…" if start + width < len(flat) else "")


# Aliases used by tests; keep names short and exported.
__all__ = ["RegexHit", "RegexRetriever", "classify_query"]
