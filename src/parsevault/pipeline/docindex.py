"""Document metadata + retrieval index.

Reads source documents, converts them with the local extraction cascade, and
builds structured metadata and a searchable index so content can be *found* —
"which reading covers photosynthesis?", "the section on grade weights" — without
re-parsing the originals each time.

Two layers:

* **Metadata** (``DocMetadata``): per-document title, outline, counts (pages,
  words, tables, figures, links), reading time, language guess, key phrases,
  an extractive summary, content hash, and the extraction provenance
  (which lanes transcribed it).
* **Retrieval** (``DocIndex``): the document is chunked along its heading
  structure, and a pure-Python BM25 index over those chunks supports ranked
  search. Each hit carries its heading path and source document, so results are
  citable.

Design notes:
* No new dependencies and no network — consistent with the local-first pipeline.
  BM25 is lexical and CPU-only; it works the same on any machine.
* Retrieval is behind a small surface (``DocIndex.search``) so a vector/embedding
  backend can be added later (on the GPU box) without changing callers.
* Parsing is separated from extraction (``parse_document_metadata`` takes raw
  Markdown) so it is trivially testable without a real PDF.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

# The retrieval engine (analysis, BM25F, hybrid fusion) is the single source of
# truth for tokenization/stopwords; tokenize is re-exported for callers/tests.
from ._atomic import _atomic_write_text
from .extractors.base import PAGE_ANCHOR_RE
from .retrieval import (
    _STOPWORDS,
    _WORD_RE,
    Retriever,
    _cosine,
    char_ngrams,
    reciprocal_rank_fusion,
    tokenize,
)

INDEX_FILENAME = "docindex.json"
INDEX_VERSION = 1


class CloudConversionNotAcknowledged(RuntimeError):
    """Raised when LlamaParse (cloud) is selected without explicit public-data
    acknowledgement — the hardened egress gate (R2.5)."""


class IndexIntegrityError(RuntimeError):
    """Raised (under strict mode) when an index fails its integrity check (R5)."""


def _strict() -> bool:
    # M15: delegate to the canonical config.strict_mode() so this module and
    # config.py can never drift on accepted env-var values. Kept as a private
    # shim so existing imports inside this module still work.
    from ..config import strict_mode

    return strict_mode()

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_LINK_RE = re.compile(r"(?<!\!)\[[^\]]+\]\([^)]+\)")


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #
@dataclass
class Heading:
    level: int
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    ordinal: int  # position within the document
    heading_path: list[str]  # e.g. ["Course Schedule", "Grade Weights"]
    text: str
    char_len: int = 0
    token_estimate: int = 0
    # Page-level traceability: the physical page span this chunk covers in the
    # original file, and the extraction lane(s) that produced those pages. A
    # chunk crossing a page boundary records the full span. 0 = unknown (legacy).
    page_start: int = 0
    page_end: int = 0
    extraction_routes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DocMetadata:
    doc_id: str
    source_path: str
    title: str
    page_count: int
    word_count: int
    reading_minutes: float
    language: str
    outline: list[Heading] = field(default_factory=list)
    table_count: int = 0
    figure_count: int = 0
    link_count: int = 0
    key_phrases: list[str] = field(default_factory=list)
    summary: str = ""
    # content_sha256 is the EXTRACTION hash (over the converted Markdown) — it
    # changes whenever extraction changes. source_sha256 is the integrity hash
    # over the ORIGINAL file bytes — stable, and what doc_id derives from, so a
    # citation survives re-extraction (see _doc_id / R2).
    content_sha256: str = ""
    source_sha256: str = ""
    extractor: str = ""
    page_routes: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    # Absolute path of the converted Markdown body, recorded by the writer so
    # readers (e.g. event_extraction) never have to reverse-engineer the
    # layout convention. Empty for legacy indexes that predate this field;
    # readers should fall back to BOTH known layouts (F-3).
    md_path: str = ""
    # Document type (e.g. court_document, medical_record, case_note, …) with the
    # signals that justified it — explainable, for sensitive/auditable records.
    doc_type: str = ""
    doc_type_confidence: float = 0.0
    doc_type_signals: list[str] = field(default_factory=list)
    # Content-free layout signature (pipeline/doctype.layout_signature): the
    # document's structural *shape* (field/checkbox/table density, line stats).
    # Powers example-based find_similar ("documents that look like this form")
    # and carries no words, so it is safe to compare across cases.
    layout: dict = field(default_factory=dict)
    # Knowledge-base facets (see pipeline/faceting.py): the artifact kind, what
    # it is about, who it is for, plus agency form number and revision date.
    # These power topical/faceted recall over a reference library.
    category: str = ""       # form | manual | policy | letter | guidance | resource | other
    topic: str = ""          # intake | assessment | safety | foster_care | …
    audience: str = ""       # social_worker | supervisor | court | provider | family | mixed
    form_number: str = ""
    rev_date: str = ""
    # Provenance: where this document came from and when it was retrieved, so
    # every hit is traceable to an authoritative source (a court/agency cite).
    source_url: str = ""
    source_domain: str = ""
    section: str = ""          # collection section, e.g. "NC General Statutes"
    retrieved_at: str = ""     # ISO date the document was obtained
    # Per-page quality rollup (R3.5). quality_score = fraction of unflagged pages;
    # review_status ∈ unreviewed|no_flags|partially_verified|verified. Only the
    # human review path (``parsevault review accept|correct|reject``) may set
    # "verified"; "no_flags" is the machine no-flagged-pages state and the
    # honest default is "unreviewed". Per-page detail lives in the
    # docindex.pages.json sidecar.
    quality_score: float = 1.0
    flagged_pages: list[int] = field(default_factory=list)
    review_status: str = "unreviewed"
    # Source currency (R8). effective_date seeds from rev_date; currency_status ∈
    # current|superseded|unknown; superseded_by points at the newer revision.
    effective_date: str = ""
    superseded_by: str = ""
    currency_status: str = "unknown"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["outline"] = [asdict(h) for h in self.outline]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DocMetadata":
        d = dict(d)
        d["outline"] = [Heading(**h) for h in d.get("outline", [])]
        # Drop unknown keys so old indexes that pre-date newer optional fields
        # (or carry experimental ones) still load — and so adding a new optional
        # field doesn't have to come with a migration. New fields land via their
        # dataclass default.
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# --------------------------------------------------------------------------- #
# parsing / metadata construction
# --------------------------------------------------------------------------- #
def _doc_id(source_path: str, identity_sha: str) -> str:
    """Stable document id from filename + an *identity* hash.

    The identity hash SHOULD be the original file's ``source_sha256`` so that
    re-extracting the same file (different DPI, new converter) keeps the same id
    and prior citations stay valid (R2.2). Callers without source bytes (e.g.
    pure-Markdown tests) pass the extraction hash, preserving legacy behavior.
    """
    h = hashlib.sha1(f"{Path(source_path).name}:{identity_sha}".encode()).hexdigest()
    return h[:16]


# Generic front-matter headings that aren't a document's real name (R10.4 —
# single definition, used by parse + the build drivers).
_GENERIC_TITLES = {"table of contents", "contents", "purpose", "introduction",
                   "overview", "index", "background", "preface", "foreword",
                   "executive summary", "summary", "i. background", "i. introduction"}


def is_generic_title(title: str) -> bool:
    t = (title or "").strip()
    return t.lower().rstrip(".:") in _GENERIC_TITLES or len(t) < 4


def title_from_path(source_path: str) -> str:
    return Path(source_path).stem.replace("-", " ").replace("_", " ").strip().title()


def resolve_title(candidate: str, source_path: str) -> str:
    """The document title: a meaningful heading, else a filename-derived title."""
    return candidate if (candidate and not is_generic_title(candidate)) else title_from_path(source_path)


def _guess_language(tokens: list[str]) -> str:
    """Very light heuristic: high English-stopword density → 'en', else 'unknown'.

    Uses the raw (pre-stopword-filter) stream, so callers pass unfiltered tokens.
    """
    if not tokens:
        return "unknown"
    hits = sum(1 for t in tokens if t in _STOPWORDS)
    return "en" if hits / len(tokens) >= 0.12 else "unknown"


def _key_phrases(text: str, k: int = 12) -> list[str]:
    """Top unigrams + bigrams by frequency, stopword-filtered. Cheap but useful
    as tags; a global-IDF version is available once the doc is in an index."""
    toks = tokenize(text)
    uni = Counter(toks)
    raw_words = [m.group(0).lower() for m in _WORD_RE.finditer(text)]
    bi = Counter(
        f"{a} {b}"
        for a, b in zip(raw_words, raw_words[1:])
        if a not in _STOPWORDS and b not in _STOPWORDS and len(a) > 1 and len(b) > 1
    )
    scored = [(w, c) for w, c in uni.items() if c > 1]
    scored += [(w, c + 1) for w, c in bi.items() if c > 1]  # nudge bigrams up
    scored.sort(key=lambda x: (-x[1], x[0]))
    seen, out = set(), []
    for phrase, _c in scored:
        if phrase in seen:
            continue
        seen.add(phrase)
        out.append(phrase)
        if len(out) >= k:
            break
    return out


def _summary(markdown: str, limit: int = 320) -> str:
    """First substantial non-heading, non-table paragraph, trimmed."""
    for block in re.split(r"\n\s*\n", markdown):
        b = block.strip()
        if not b or b.startswith("#") or b.startswith("|"):
            continue
        b = re.sub(r"\s+", " ", b)
        return b[:limit].rsplit(" ", 1)[0] + ("…" if len(b) > limit else "")
    return ""


# Legal-structure segmentation (R7): statute/regulation section headers. Matches
# a leading "§ 7B-101" / "Sec. 1912" / "Section 5106a" or a bare NC-style
# "7B-101" / "10A-101" at line start — the legally-meaningful locators.
_LEGAL_SECTION_RE = re.compile(
    r"^\s{0,3}(?:(?:§+|Sec(?:tion)?\.?)\s*[\dA-Za-z][\dA-Za-z.\-]*"
    r"|\d+[A-Z]?-\d+[A-Za-z0-9.\-]*)\b",
)


def looks_like_legal(markdown: str, *, min_sections: int = 3) -> bool:
    """True if the text carries enough statutory/regulatory section headers to
    warrant legal-aware segmentation (cheap density check)."""
    clean = PAGE_ANCHOR_RE.sub("", markdown)
    n = sum(1 for line in clean.splitlines()
            if _LEGAL_SECTION_RE.match(line) and not _HEADING_RE.match(line))
    return n >= min_sections


def promote_legal_sections(markdown: str) -> str:
    """Promote statutory/regulatory section lines to Markdown headings (R7.1) so
    each numbered section becomes its own chunk with the section id in its
    heading path — the citation-meaningful boundary. Idempotent; leaves page
    anchors and existing headings untouched."""
    out: list[str] = []
    for line in markdown.splitlines():
        if (_LEGAL_SECTION_RE.match(line) and not _HEADING_RE.match(line)
                and not PAGE_ANCHOR_RE.match(line.strip())):
            out.append(f"## {line.strip()}")
        else:
            out.append(line)
    return "\n".join(out)


def chunk_markdown(
    markdown: str, doc_id: str, *, max_chars: int = 1500,
    page_routes: dict[int, str] | None = None, min_chars: int = 120,
) -> list[Chunk]:
    """Split Markdown into retrieval chunks along heading boundaries.

    Each heading starts a new section; an oversized section is further split on
    paragraph boundaries so no chunk is unwieldy for retrieval. Every chunk
    carries the heading path it lives under, so a hit is citable.

    Page anchors (``<!-- page: N -->``, emitted by extraction) are consumed here:
    they never enter chunk text, but they let every chunk record the physical
    page span it covers (``page_start``/``page_end``) and, via ``page_routes``
    (a ``{page_number: lane}`` map), which extraction lane(s) produced it.
    """
    page_routes = page_routes or {}
    # (heading_path, body) where body is a list of (page_number, line).
    sections: list[tuple[list[str], list[tuple[int, str]]]] = []
    path: list[tuple[int, str]] = []  # stack of (level, text)
    body: list[tuple[int, str]] = []
    current_page = 1

    def cur_path() -> list[str]:
        return [t for _l, t in path]

    def flush_section():
        if body and any(line.strip() for _p, line in body):
            sections.append((cur_path(), list(body)))
        body.clear()

    for line in markdown.splitlines():
        anchor = PAGE_ANCHOR_RE.match(line.strip())
        if anchor:
            current_page = int(anchor.group(1))
            continue
        m = _HEADING_RE.match(line)
        if m:
            flush_section()
            level = len(m.group(1))
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, m.group(2).strip()))
            # The heading text itself seeds the section body (so it is searchable).
            body.append((current_page, m.group(2).strip()))
        else:
            body.append((current_page, line))
    flush_section()

    chunks: list[Chunk] = []
    ordinal = 0
    for heading_path, body_pl in sections:
        pieces = _split_pl_to_size(body_pl, max_chars)
        # M13: when a section is split into multiple pieces, the heading line
        # was only seeded into piece 0's body — pieces 1, 2, ... lost the
        # heading text from their ``chunk.text``. BM25F still boosts via the
        # heading_path attribute, but a UI / RAG consumer reading raw chunk
        # text on the 2nd half of "§ 7B-1111" would not see the section id.
        # Prepend the leaf heading (in parens, marked as a continuation) to
        # every piece after the first so the citation context survives the
        # split.
        leaf = heading_path[-1] if heading_path else ""
        for i, piece in enumerate(pieces):
            text = "\n\n".join(
                "\n".join(line for _p, line in para) for para in piece
            ).strip()
            if not text:
                continue
            if i > 0 and leaf and not text.lstrip().startswith(leaf):
                text = f"({leaf}, continued)\n\n{text}"
            pages = [p for para in piece for p, _l in para]
            ps, pe = (min(pages), max(pages)) if pages else (0, 0)
            routes = sorted({page_routes.get(p, "") for p in range(ps, pe + 1)} - {""})
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:{ordinal}",
                    doc_id=doc_id,
                    ordinal=ordinal,
                    heading_path=heading_path,
                    text=text,
                    char_len=len(text),
                    token_estimate=max(1, len(text) // 4),
                    page_start=ps,
                    page_end=pe,
                    extraction_routes=routes,
                )
            )
            ordinal += 1
    return _coalesce_small_chunks(chunks, doc_id, min_chars=min_chars, max_chars=max_chars)


def _merge_chunks(a: Chunk, b: Chunk) -> Chunk:
    """Merge chunk ``b`` into its predecessor ``a`` (same document)."""
    text = f"{a.text}\n{b.text}".strip()
    pages = [p for p in (a.page_start, a.page_end, b.page_start, b.page_end) if p]
    return Chunk(
        chunk_id=a.chunk_id, doc_id=a.doc_id, ordinal=a.ordinal,
        heading_path=a.heading_path or b.heading_path,
        text=text, char_len=len(text), token_estimate=max(1, len(text) // 4),
        page_start=min(pages) if pages else 0,
        page_end=max(pages) if pages else 0,
        extraction_routes=sorted(set(a.extraction_routes) | set(b.extraction_routes)),
    )


def _coalesce_small_chunks(chunks: list[Chunk], doc_id: str, *,
                           min_chars: int, max_chars: int) -> list[Chunk]:
    """Merge sub-``min_chars`` chunks forward into the next (heading-only / TOC
    fragments are noise that dilute retrieval). Bounded so a merge never blows
    past ~1.6× ``max_chars``. Re-numbers ordinals/chunk_ids contiguously."""
    out: list[Chunk] = []
    pending: Chunk | None = None
    cap = int(max_chars * 1.6)
    for c in chunks:
        if pending is None:
            pending = c
        elif len(pending.text) < min_chars and len(pending.text) + len(c.text) + 1 <= cap:
            pending = _merge_chunks(pending, c)
        else:
            out.append(pending)
            pending = c
    if pending is not None:
        if len(pending.text) < min_chars and out:  # tail fragment → merge backward
            out[-1] = _merge_chunks(out[-1], pending)
        else:
            out.append(pending)
    for i, c in enumerate(out):  # contiguous ids after merges
        c.ordinal = i
        c.chunk_id = f"{doc_id}:{i}"
    return out


def _cap_paragraph(para: list[tuple[int, str]], max_chars: int) -> list[list[tuple[int, str]]]:
    """Split a paragraph longer than ``max_chars`` at line boundaries so a giant
    table or an unbroken wall of text can't become one oversized chunk. A single
    over-long line is hard-wrapped by character window."""
    if sum(len(line) for _p, line in para) + len(para) <= max_chars:
        return [para]
    out: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    cur_len = 0
    for page, line in para:
        if len(line) > max_chars:  # one line longer than the cap → hard-wrap it
            if cur:
                out.append(cur)
                cur, cur_len = [], 0
            for i in range(0, len(line), max_chars):
                out.append([(page, line[i:i + max_chars])])
            continue
        if cur and cur_len + len(line) + 1 > max_chars:
            out.append(cur)
            cur, cur_len = [], 0
        cur.append((page, line))
        cur_len += len(line) + 1
    if cur:
        out.append(cur)
    return out


def _paragraphs_pl(
    body_pl: list[tuple[int, str]], max_chars: int | None = None
) -> list[list[tuple[int, str]]]:
    """Group page-tagged lines into paragraphs on blank-line boundaries; when
    ``max_chars`` is given, oversized paragraphs are further split so no paragraph
    exceeds the cap (prevents 50KB single-table chunks)."""
    paras: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    for page, line in body_pl:
        if line.strip():
            cur.append((page, line))
        elif cur:
            paras.append(cur)
            cur = []
    if cur:
        paras.append(cur)
    if max_chars:
        capped: list[list[tuple[int, str]]] = []
        for para in paras:
            capped.extend(_cap_paragraph(para, max_chars))
        return capped
    return paras


def _split_pl_to_size(
    body_pl: list[tuple[int, str]], max_chars: int
) -> list[list[list[tuple[int, str]]]]:
    """Pack paragraphs into size-bounded pieces, preserving page tags.

    Returns a list of *pieces*; each piece is a list of paragraphs; each
    paragraph is a list of ``(page, line)``. Page tags survive so the caller can
    compute each chunk's page span.
    """
    paras = _paragraphs_pl(body_pl, max_chars)
    if not paras:
        return []
    pieces: list[list[list[tuple[int, str]]]] = []
    buf: list[list[tuple[int, str]]] = []
    buflen = 0
    for para in paras:
        plen = sum(len(line) for _p, line in para) + len(para)
        if buf and buflen + plen + 2 > max_chars:
            pieces.append(buf)
            buf, buflen = [para], plen
        else:
            buf.append(para)
            buflen += plen + 2
    if buf:
        pieces.append(buf)
    return pieces


def parse_document_metadata(
    markdown: str,
    *,
    source_path: str,
    page_count: int = 0,
    extractor: str = "",
    page_routes: list[str] | None = None,
    source_sha256: str = "",
    page_route_map: dict[int, str] | None = None,
    max_chunk_chars: int = 1500,
    classifier=None,
) -> tuple[DocMetadata, list[Chunk]]:
    """Build (DocMetadata, chunks) from already-extracted Markdown. Pure/testable.

    ``classifier`` is an optional DocTypeClassifier; when omitted, a default
    rule-based (CPU, local, explainable) classifier is used to set the doc type.

    ``markdown`` may carry page anchors (``<!-- page: N -->``); they are consumed
    for page-level chunk spans and stripped from all text statistics. ``doc_id``
    derives from ``source_sha256`` when given (stable across re-extraction),
    otherwise from the extraction hash (legacy/pure-Markdown callers).
    """
    content_sha = hashlib.sha256(markdown.encode("utf-8")).hexdigest()
    doc_id = _doc_id(source_path, source_sha256 or content_sha)
    # Statistics, headings, title, summary, facets all ignore the page anchors.
    clean = PAGE_ANCHOR_RE.sub("", markdown)

    headings = [
        Heading(level=len(m.group(1)), text=m.group(2).strip())
        for line in clean.splitlines()
        if (m := _HEADING_RE.match(line))
    ]
    # Prefer the first meaningful H1, skipping generic front-matter headings;
    # fall back to a filename-derived title (R10.4: shared resolve_title).
    h1 = next((h.text for h in headings if h.level == 1 and not is_generic_title(h.text)), "")
    title = resolve_title(h1, source_path)

    raw_words = [m.group(0).lower() for m in _WORD_RE.finditer(clean)]
    word_count = len(raw_words)
    table_count = sum(1 for line in clean.splitlines() if _TABLE_SEP_RE.match(line))
    figure_count = len(_IMAGE_RE.findall(clean))
    link_count = len(_LINK_RE.findall(clean))

    # Legal-structure-aware segmentation (R7): for statutes/regulations, promote
    # section headers so each numbered section is its own chunk with the section
    # id in its heading path. Detected from content, so it needs no facets.
    to_chunk = promote_legal_sections(markdown) if looks_like_legal(markdown) else markdown
    chunks = chunk_markdown(
        to_chunk, doc_id, max_chars=max_chunk_chars, page_routes=page_route_map,
    )

    if classifier is None:
        from .doctype import DocTypeClassifier

        classifier = DocTypeClassifier()
    dt = classifier.classify(clean)

    from .doctype import layout_signature

    layout = layout_signature(clean)

    from .faceting import compute_facets

    facets = compute_facets(clean, str(source_path))

    meta = DocMetadata(
        doc_id=doc_id,
        source_path=str(source_path),
        title=title,
        page_count=page_count,
        word_count=word_count,
        reading_minutes=round(word_count / 200.0, 1),
        language=_guess_language(raw_words),
        outline=headings,
        table_count=table_count,
        figure_count=figure_count,
        link_count=link_count,
        key_phrases=_key_phrases(clean),
        summary=_summary(clean),
        content_sha256=content_sha,
        source_sha256=source_sha256,
        extractor=extractor,
        page_routes=list(page_routes or []),
        chunk_ids=[c.chunk_id for c in chunks],
        doc_type=dt.label,
        doc_type_confidence=dt.confidence,
        doc_type_signals=dt.matched.get(dt.label, []),
        layout=layout,
        category=facets.category,
        topic=facets.topic,
        audience=facets.audience,
        form_number=facets.form_number,
        rev_date=facets.rev_date,
        effective_date=facets.rev_date,
        currency_status="current" if facets.rev_date else "unknown",
    )
    return meta, chunks


def _extract_with_provider(path: Path, config):
    """Extract ``path`` honoring ``config.parse_provider``, with local fallback.

    ``"llamaparse"`` (public data only) sends the document to LlamaParse and, if
    that fails for any reason other than exhausted credits, degrades to the local
    cascade so one document can't block a batch. Credit-exhaustion propagates so
    the caller can stop and alert.
    """
    from .extractors.router import _TEXT_SUFFIXES, LocalCascadeExtractor

    # Plain text is already ground truth — never send it to a cloud parser,
    # regardless of provider. The cascade routes it to the plain-text lane.
    if path.suffix.lower() in _TEXT_SUFFIXES:
        return LocalCascadeExtractor(config).extract(path)

    provider = getattr(config, "parse_provider", "local-cascade")
    if provider == "llamaparse":
        # Hardened gate (R2.5): cloud conversion sends the document off-machine,
        # so it is refused unless the caller has explicitly acknowledged the data
        # is public (CLI --public-data-acknowledged / API allow_cloud=True /
        # PARSE_PUBLIC_DATA_ACK=1). An implicit convention is not an audit control.
        if not getattr(config, "allow_cloud", False):
            raise CloudConversionNotAcknowledged(
                "PARSE_PROVIDER=llamaparse sends documents to the LlamaParse cloud "
                "API. Refusing without explicit public-data acknowledgement. Pass "
                "--public-data-acknowledged (CLI), allow_cloud=True (API), or set "
                "PARSE_PUBLIC_DATA_ACK=1 — and only for PUBLIC data."
            )
        from .extractors.llamaparse_extractor import (
            LlamaParseCreditsExhausted,
            LlamaParseExtractor,
        )

        lp = LlamaParseExtractor(
            config.llamaparse_api_key, base_url=config.llamaparse_base_url
        )
        try:
            return lp.extract(path)
        except LlamaParseCreditsExhausted:
            raise  # let the batch stop + alert; do not silently downgrade quality
        except Exception as e:  # noqa: BLE001 — any parse failure → local fallback
            import sys

            print(f"  ! LlamaParse failed for {path.name} ({e}); using local cascade",
                  file=sys.stderr)
            result = LocalCascadeExtractor(config).extract(path)
            result.degradations.append({
                "kind": "llamaparse_local_fallback",
                "detail": f"LlamaParse failed ({type(e).__name__}); used local cascade",
            })
            return result
    return LocalCascadeExtractor(config).extract(path)


def apply_provenance(meta: DocMetadata, prov: dict | None) -> DocMetadata:
    """Stamp a document's source/retrieval provenance onto its metadata."""
    if prov:
        meta.source_url = prov.get("source_url", meta.source_url)
        meta.source_domain = prov.get("source_domain", meta.source_domain)
        meta.section = prov.get("section", meta.section)
        meta.retrieved_at = prov.get("retrieved_at", meta.retrieved_at)
    return meta


def load_provenance_manifest(path: str | Path, *, retrieved_at: str = "") -> dict[str, dict]:
    """Load a SOURCE_INDEX-style manifest into a ``{filename: provenance}`` map.

    Accepts a JSON list of entries with ``filename`` (or ``path``) and any of
    ``source_url``/``source_domain``/``section``. ``retrieved_at`` is applied to
    every entry that doesn't carry its own.
    """
    import json

    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    skipped = 0
    for e in entries:
        name = e.get("filename") or Path(e.get("path", "")).name
        if not name:
            skipped += 1
            continue
        out[name] = {
            "source_url": e.get("source_url", ""),
            "source_domain": e.get("source_domain", ""),
            "section": e.get("section", ""),
            "retrieved_at": e.get("retrieved_at", retrieved_at),
        }
    if skipped:
        import sys
        print(
            f"WARNING: provenance manifest {path}: skipped {skipped} "
            f"entr{'y' if skipped == 1 else 'ies'} lacking filename/path — "
            "those docs will carry no source provenance",
            file=sys.stderr,
        )
    return out


# Defensive cap on per-event page/chunk detail arrays (build_events dashboard) —
# this corpus won't hit it, but no event payload should be allowed to grow
# unbounded with document size.
_EVENT_ARRAY_CAP = 500

_PREVIEW_CHARS = 220


def _preview(text: str, limit: int = _PREVIEW_CHARS) -> str:
    """Short, whitespace-collapsed snippet for the build dashboard — lets a
    viewer see the actual words a phase produced, not just counts about them."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rsplit(" ", 1)[0] + "…"


def build_document_metadata(
    source_path: str | Path, *, config=None, max_chunk_chars: int = 1500,
    outputs_dir: str | Path | None = None, provenance: dict | None = None,
    degradation_sink: list | None = None, archive_sources: bool = False,
    event_sink=None,
) -> tuple[DocMetadata, list[Chunk]]:
    """Convert a document and build its metadata + chunks.

    Uses the local cascade by default (fully local / no-egress). When
    ``config.parse_provider == "llamaparse"`` (PUBLIC data only), routes the
    document through the LlamaParse cloud API, falling back to the local cascade
    if a parse fails — so a single bad document never blocks the build.

    When ``outputs_dir`` is given, the converted Markdown is preserved there as
    ``{doc_id}.md`` (originals + Markdown + index together = a durable KB).
    ``provenance`` (source_url/source_domain/section/retrieved_at) is stamped onto
    the metadata so every hit is traceable to where and when it was obtained.
    """
    if config is None:
        from ..config import cascade_config_from_env

        config = cascade_config_from_env()
    path = Path(source_path)
    # Integrity hash over the ORIGINAL bytes (R2.1) — the stable identity that
    # doc_id and citations hang on, independent of how extraction renders it.
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""
    result = _extract_with_provider(path, config)
    if event_sink is not None:
        doc_id = _doc_id(str(path), source_sha256 or "")
        event_sink(
            "extract_done",
            doc_id=doc_id,
            path=str(path),
            extractor=result.extractor,
            page_count=result.page_count,
            pages=[
                {
                    "page_number": p.page_number,
                    "route": p.route,
                    "chars": len(p.markdown),
                    "elapsed_seconds": round(p.elapsed_seconds, 3),
                    "flagged": bool(p.quality.get("flagged")) if p.quality else False,
                    "flag_reasons": p.quality.get("flag_reasons", []) if p.quality else [],
                    "preview": _preview(p.markdown),
                }
                for p in result.pages[:_EVENT_ARRAY_CAP]
            ],
            degradations=result.degradations,
        )
    page_route_map = {p.page_number: p.route for p in result.pages}
    meta, chunks = parse_document_metadata(
        result.markdown,
        source_path=str(path),
        page_count=result.page_count,
        extractor=result.extractor,
        page_routes=result.page_routes,
        source_sha256=source_sha256,
        page_route_map=page_route_map,
        max_chunk_chars=max_chunk_chars,
    )
    if event_sink is not None:
        event_sink(
            "chunk_done",
            doc_id=meta.doc_id,
            chunk_count=len(chunks),
            chunks=[
                {
                    "chunk_id": c.chunk_id,
                    "heading_path": c.heading_path,
                    "char_len": c.char_len,
                    "token_estimate": c.token_estimate,
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "extraction_routes": c.extraction_routes,
                    "preview": _preview(c.text),
                }
                for c in chunks[:_EVENT_ARRAY_CAP]
            ],
        )
    apply_provenance(meta, provenance)
    # Surface extraction-time degradations (vlm→tesseract, llamaparse→local) into
    # the caller's ledger, stamped with this document's id + a timestamp (R4).
    if degradation_sink is not None and result.degradations:
        import datetime

        ts = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        for d in result.degradations:
            degradation_sink.append({"timestamp": ts, "doc_id": meta.doc_id, **d})
    # Aggregate per-page quality onto the document (R3.5).
    from .quality import PageQuality, document_quality

    pqs = [PageQuality(**p.quality) for p in result.pages if p.quality]
    if pqs:
        meta.quality_score, meta.flagged_pages, meta.review_status = document_quality(pqs)
    if outputs_dir is not None:
        out = Path(outputs_dir)
        out.mkdir(parents=True, exist_ok=True)
        md_path = out / f"{meta.doc_id}.md"
        _atomic_write_text(md_path, result.markdown)
        # Record the absolute path we wrote so downstream readers (e.g.
        # event_extraction) don't have to reverse-engineer the layout
        # convention (F-3).
        meta.md_path = str(md_path)
        _write_extraction_manifest(out / f"{meta.doc_id}.manifest.json", meta, result, config)
        if pqs:  # per-page quality + review status detail for the review queue (R3)
            _atomic_write_text(
                out / f"{meta.doc_id}.pages.json",
                json.dumps([q.to_dict() for q in pqs], indent=2, ensure_ascii=False),
            )
        if archive_sources and path.exists() and meta.source_sha256:
            # Archive the original alongside, keyed by content hash, so the cited
            # artifact can never drift from the indexed one (R2.4).
            import shutil

            sources_dir = out.parent / "sources"
            sources_dir.mkdir(parents=True, exist_ok=True)
            archived = sources_dir / f"{meta.source_sha256}{path.suffix.lower()}"
            if not archived.exists() and archived.resolve() != path.resolve():
                shutil.copy2(path, archived)
    return meta, chunks


def remap_index_ids(data: dict, vectors: dict | None, sha_for) -> dict[str, str]:
    """Rewrite doc_ids in a loaded index (+ vectors sidecar) to source-hash-based
    ids (R2.2). ``sha_for(doc_dict) -> source_sha256 | None`` supplies each
    original file's hash (None → leave that document unchanged). Mutates ``data``
    and ``vectors`` in place; returns the ``{old_doc_id: new_doc_id}`` map.
    """
    id_map: dict[str, str] = {}
    for d in data.get("documents", []):
        sha = sha_for(d)
        if not sha:
            continue
        new = _doc_id(d.get("source_path", d["doc_id"]), sha)
        d["source_sha256"] = sha
        if new != d["doc_id"]:
            id_map[d["doc_id"]] = new

    def remap_chunk_id(cid: str) -> str:
        doc, _, rest = cid.partition(":")
        return f"{id_map[doc]}:{rest}" if doc in id_map else cid

    for d in data.get("documents", []):
        if d["doc_id"] in id_map:
            d["chunk_ids"] = [remap_chunk_id(c) for c in d.get("chunk_ids", [])]
            d["doc_id"] = id_map[d["doc_id"]]
    for c in data.get("chunks", []):
        if c["doc_id"] in id_map:
            c["chunk_id"] = remap_chunk_id(c["chunk_id"])
            c["doc_id"] = id_map[c["doc_id"]]
    for e in data.get("degradations", []):
        if e.get("doc_id") in id_map:
            e["doc_id"] = id_map[e["doc_id"]]
    if vectors:
        for key in ("vectors", "groups"):
            if key in vectors:
                vectors[key] = {remap_chunk_id(cid): v for cid, v in vectors[key].items()}
        if "groups" in vectors:  # group VALUES are doc ids
            vectors["groups"] = {k: id_map.get(v, v) for k, v in vectors["groups"].items()}
    return id_map


def _write_extraction_manifest(path: Path, meta: DocMetadata, result, config) -> None:
    """Persist a reproducible record of how this document was converted (R2.3).

    Captures everything that influences the extraction hash — converter, model
    identifiers, DPI, OCR language, normalization steps, version, host, time —
    so any change to ``content_sha256`` is explainable and any conversion is
    reproducible.
    """
    import datetime
    import socket

    # PRD R1 traceability: record the FULL ordered per-page route list so
    # downstream consumers can answer "which lane was page N?" (MAN-1).
    # The legacy ``page_routes`` field stays as the deduped set for
    # backward compatibility with any reader that expected a small summary
    # ("did this doc touch the VLM at all?"); the new ``per_page_routes``
    # field carries one entry per physical page in order.
    manifest = {
        "doc_id": meta.doc_id,
        "source_path": meta.source_path,
        "source_sha256": meta.source_sha256,
        "content_sha256": meta.content_sha256,
        "parsevault_version": _parsevault_version(),
        "extractor": meta.extractor,
        "parse_provider": getattr(config, "parse_provider", "local-cascade"),
        "page_count": meta.page_count,
        # Backward-compat: deduped lane set.
        "page_routes": sorted(set(meta.page_routes)),
        # New: full ordered list (one per page) — R1 per-page accountability.
        "per_page_routes": list(meta.page_routes),
        "dpi": getattr(config, "dpi", None),
        "tesseract_lang": getattr(config, "tesseract_lang", None),
        "vlm_models": (
            {"quality": getattr(config, "quality_model", ""),
             "fast": getattr(config, "fast_model", "")}
            if getattr(config, "use_vlm", False) else None
        ),
        "normalization": ["normalize_unicode", "strip_running_headers"],
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
    }
    _atomic_write_text(path, json.dumps(manifest, indent=2, ensure_ascii=False))


def _parsevault_version() -> str:
    """Best-effort version tag for R2.3 reproducibility.

    Tries the installed package metadata first; falls back to the in-tree
    git HEAD short-SHA when running ``pip install -e .`` from a clone
    (H9 — without the fallback every manifest reads ``"unknown"``).
    """
    try:
        from importlib.metadata import version

        return version("parsevault")
    except Exception:  # noqa: BLE001 — version is informational
        pass
    # Editable install / in-tree run: read git HEAD if we're in a working tree.
    try:
        import subprocess

        # ``cwd`` rooted on this file so a different cwd doesn't pick the
        # wrong repo.
        here = Path(__file__).resolve().parent
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=here, capture_output=True, text=True, timeout=2, check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return f"git-{out.stdout.strip()}"
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


# --------------------------------------------------------------------------- #
# retrieval index (BM25, pure-python)
# --------------------------------------------------------------------------- #
@dataclass
class SearchHit:
    score: float
    chunk: Chunk
    title: str  # source document title, for citation
    heading_path: list[str]
    snippet: str
    # Provenance carried onto every hit so a result is independently citable.
    doc_id: str = ""
    source_url: str = ""
    source_domain: str = ""
    section: str = ""
    retrieved_at: str = ""
    # Page-level traceability mirrored from the chunk (R1.4).
    page_start: int = 0
    page_end: int = 0
    extraction_routes: list[str] = field(default_factory=list)
    # Caller-visible notes about how this result was produced — e.g. that the
    # reranker precision pass did not run (R4.4), so callers never silently get
    # a degraded ranking they believe was reranked.
    notes: list[str] = field(default_factory=list)
    # Which retrieval lane surfaced this hit (L-2 audit): "bm25" when only the
    # lexical lane ranked the chunk, "dense" when only the dense lane did,
    # "hybrid" when both did, "regex" for the exact-pattern lane, and "" when
    # the lane attribution is not available (e.g. legacy code paths). The
    # attorney trace surfaces this so the reader can tell *how* a passage
    # was found.
    lane: str = ""

    def page_span(self) -> str:
        """Human page reference: 'p. 14', 'pp. 14–15', or '' when unknown."""
        if not self.page_start:
            return ""
        if self.page_end and self.page_end != self.page_start:
            return f"pp. {self.page_start}–{self.page_end}"
        return f"p. {self.page_start}"

    def statutory_section(self) -> str:
        """The leading heading when it is a statutory/regulatory section id (R7.2)."""
        if self.heading_path and _LEGAL_SECTION_RE.match(self.heading_path[0]):
            return self.heading_path[0]
        return ""

    def citation(self) -> str:
        """A compact, source-traceable citation line for this hit.

        For statutes/regulations the section number is the legally-meaningful
        locator, so it leads the citation (R7.2)."""
        where = self.source_url or self.source_domain
        when = f", retrieved {self.retrieved_at}" if self.retrieved_at else ""
        statute = self.statutory_section()
        if statute:
            parts = [statute, self.title]
        else:
            parts = [self.title]
            if self.heading_path:
                parts.append(" › ".join(self.heading_path))
        if pp := self.page_span():
            parts.append(pp)
        if where:
            parts.append(f"{where}{when}")
        return " — ".join(parts)


@dataclass
class SimilarHit:
    """One document ranked by resemblance to an example (find_similar).

    The score fuses three explainable signals, each carried so an attorney can
    see *why* a record was surfaced: ``content_score`` (how much its salient
    vocabulary overlaps the example), ``layout_sim`` (how closely its structural
    shape matches — form vs narrative vs table), and ``type_match`` (same
    document type as the example).
    """

    doc_id: str
    title: str
    score: float
    content_score: float
    layout_sim: float
    type_match: bool
    doc_type: str
    snippet: str = ""
    source_path: str = ""
    why: list[str] = field(default_factory=list)


def _page_span(page_start: int = 0, page_end: int = 0) -> str:
    """'p. 14' / 'pp. 14–15' / '' — shared page-reference formatter."""
    if not page_start:
        return ""
    if page_end and page_end != page_start:
        return f"pp. {page_start}–{page_end}"
    return f"p. {page_start}"


def citation_text(
    meta: "DocMetadata", heading_path: list[str] | None = None,
    *, page_start: int = 0, page_end: int = 0,
) -> str:
    """A formatted, copy-paste citation + verification block for a brief.

    Deterministic facts only (source, page, date, integrity hashes) — suitable to
    paste into a legal document and independently verify. Pass the cited chunk's
    page span to anchor the citation to specific original pages.

    T2-1: when ``meta.currency_status`` is set, append the rendered currency
    + any staleness note onto the verify block so an attorney pasting the
    citation can see the supersession risk for the cited statute / form.
    Malpractice vector — superseded statutes should never paste as if
    they were authoritative.
    """
    where = " › ".join(heading_path) if heading_path else ""
    cite = [meta.title.strip() or meta.doc_id]
    if where:
        cite.append(where)
    if meta.section:
        cite.append(meta.section)
    head = ". ".join(p for p in cite if p)
    pp = _page_span(page_start, page_end)
    src = meta.source_url or meta.source_domain
    retrieved = f" Retrieved {meta.retrieved_at}" if meta.retrieved_at else ""
    # ``retrieved`` already reads " Retrieved <date>" (no terminal period) or
    # is empty. Only append the closing period when it's present — otherwise
    # the line ends "…ncdhhs.gov.." (double period) or "Title.." when there's
    # no source at all. This string is the attorney-facing "Copy citation".
    line = (f"{head}." + (f" {pp}." if pp else "")
            + (f" {src}." if src else "")
            + (f"{retrieved}." if retrieved else ""))
    src_hash = (f"source file SHA-256: {meta.source_sha256}; " if meta.source_sha256 else "")
    # T2-1: currency status — render only when populated and informative
    # (skip the "unknown" default that DocMetadata seeds at construction).
    cur_status = (getattr(meta, "currency_status", "") or "").strip()
    currency_bit = ""
    if cur_status and cur_status != "unknown":
        rendered = _render_currency(meta)
        sn = staleness_note(meta)
        if sn:
            currency_bit = f"currency: {rendered} — {sn}; "
        else:
            currency_bit = f"currency: {rendered}; "
    verify = (f"[Verification — {src_hash}extraction SHA-256: {meta.content_sha256}; "
              f"converted by {meta.extractor or 'native'}; "
              f"{currency_bit}"
              f"manifest {meta.doc_id}.manifest.json; "
              f"document id {meta.doc_id}]")
    return f"{line}\n{verify}"


def format_audit_trace(hit: "SearchHit", meta: "DocMetadata | None") -> str:
    """A full, human-readable audit chain for one retrieval — for legal use.

    Answers, for the matched passage: where it lives, where it came from and
    when, whether it is tamper-evident, and how it was produced.
    """
    section = " › ".join(hit.heading_path) if hit.heading_path else "(document body)"
    pp = hit.page_span() or "—"
    lanes = ", ".join(hit.extraction_routes) or "—"
    lines = [
        "  WHERE THIS LIVES (cite this):",
        f"    Document      : {hit.title}",
        f"    Section       : {section}",
        f"    Page(s)       : {pp}",
        f"    Passage ID    : {hit.chunk.chunk_id}",
    ]
    if meta:
        lines += [
            "  SOURCE (where it came from, when):",
            f"    Source URL    : {meta.source_url or '—'}",
            f"    Source        : {meta.source_domain or '—'}",
            f"    Collection    : {meta.section or '—'}",
            f"    Form number   : {meta.form_number or '—'}",
            f"    Retrieved on  : {meta.retrieved_at or '—'}",
            "  VERIFICATION (tamper-evident):",
            f"    Source hash   : SHA-256 {meta.source_sha256 or '—'} (original file)",
            f"    Extract hash  : SHA-256 {meta.content_sha256 or '—'} (converted text)",
            f"    Cited pages   : {pp} transcribed by {lanes}",
            f"    Converted by  : {meta.extractor or '—'} "
            f"({', '.join(sorted(set(meta.page_routes))) or 'native'})",
            f"    Quality       : score {meta.quality_score:.2f}, review "
            f"{meta.review_status}" + (f", flagged pages {meta.flagged_pages}"
                                        if meta.flagged_pages else ""),
            f"    Currency      : {_render_currency(meta)}"
            + (f" — {sn}" if (sn := staleness_note(meta)) else ""),
            f"    Manifest      : {meta.doc_id}.manifest.json",
            f"    Document ID   : {meta.doc_id}",
        ]
    return "\n".join(lines)


class DocIndex:
    """A persistable collection of documents + chunks with hybrid retrieval.

    Search is delegated to ``retrieval.Retriever`` (BM25F + char-trigram fusion,
    Porter stemming, optional query expansion). ``search`` is the stable surface:
    a dense/embedding retriever can be fused in (see ``embeddings``) without
    changing callers. Lexical retrieval remains dependency-free and CPU-only.
    """

    def __init__(self, *, expand: bool = False, retriever: Retriever | None = None,
                 embedder=None, dense_weight: float = 1.0, **kw):
        # ``kw`` accepts legacy/retriever knobs (k1, b, heading_boost, …).
        self.expand = expand
        self.dense_weight = dense_weight
        self.documents: dict[str, DocMetadata] = {}
        self.chunks: dict[str, Chunk] = {}
        # PERF-D2-2: monotonically bumped on every add()/remove(); used as a
        # cache key for ``_doc_texts`` and ``_term_doc_freq`` so /similar
        # doesn't pay the ~5–13 s per-call rebuild at PoC scale.
        self._chunks_version: int = 0
        self._doc_texts_cache: tuple[int, dict[str, str]] | None = None
        self._term_df_cache: tuple[int, Counter] | None = None
        # Degradation ledger (R4.1): one entry per tolerated quality compromise
        # (vlm→tesseract fallback, zero-vector substitution, reranker skip, …) so
        # nothing is silent. Persisted in the index and summarized in build reports.
        self.degradations: list[dict] = []
        if retriever is None:
            # Wire in domain synonym/acronym expansion by default (Tier 1 lexical
            # precision); a caller can pass its own retriever to opt out.
            from .synonyms import default_expander

            kw.setdefault("synonyms", default_expander())
            retriever = Retriever(**kw)
        self._retriever = retriever
        # Optional dense semantic lane (GPU server / sentence-transformers / CPU
        # hashing fallback). When absent, retrieval is the pure-CPU hybrid.
        self._dense = None
        if embedder is not None:
            from .embeddings import DenseStore

            self._dense = DenseStore(embedder)
            self._wire_dense_degradation()

    def _wire_dense_degradation(self) -> None:
        """Record a dense_zero_vector degradation whenever an embedding fails."""
        if self._dense is not None:
            self._dense.on_degrade = lambda cid, g: self.record_degradation(
                "dense_zero_vector", "embedding failed; chunk is lexical-only",
                doc_id=g, chunk_id=cid,
            )

    # -- mutation -------------------------------------------------------------
    def add(self, meta: DocMetadata, chunks: list[Chunk]) -> None:
        # Re-indexing the same content (same doc_id) replaces the prior copy.
        self.remove(meta.doc_id)
        # Authoritatively sync ``meta.chunk_ids`` with what was actually added
        # (P-1). A caller that re-chunked but forgot to refresh ``chunk_ids``
        # would otherwise persist stale ids.
        meta.chunk_ids = [c.chunk_id for c in chunks]
        self.documents[meta.doc_id] = meta
        for c in chunks:
            self.chunks[c.chunk_id] = c
            self._retriever.add(
                c.chunk_id,
                c.text,
                heading=" ".join(c.heading_path),
                group=c.doc_id,
            )
        # PERF-D2-2: invalidate the doc-texts / term-DF caches.
        self._chunks_version += 1
        if self._dense is not None:
            # Batch-embed this document's chunks in one round-trip.
            self._dense.add_many(
                [
                    (c.chunk_id, (" ".join(c.heading_path) + "\n" + c.text).strip(), c.doc_id)
                    for c in chunks
                ]
            )

    def remove(self, doc_id: str) -> None:
        self.documents.pop(doc_id, None)
        for cid in [cid for cid, c in self.chunks.items() if c.doc_id == doc_id]:
            self.chunks.pop(cid, None)
        self._retriever.remove_group(doc_id)
        if self._dense is not None:
            self._dense.remove_group(doc_id)
        # PERF-D2-2: invalidate the doc-texts / term-DF caches.
        self._chunks_version += 1

    def record_degradation(self, kind: str, detail: str = "", **ids) -> dict:
        """Append a degradation-ledger entry (R4.1). ``kind`` is one of
        vlm_fallback_tesseract / dense_zero_vector / embedder_unavailable /
        reranker_skipped / llamaparse_local_fallback / embedder_config_error.
        ``ids`` may carry ``doc_id`` and/or ``chunk_id``."""
        import datetime

        entry = {
            "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "kind": kind,
            "detail": detail,
            **{k: v for k, v in ids.items() if v},
        }
        self.degradations.append(entry)
        return entry

    # -- search ---------------------------------------------------------------
    def search(
        self, query: str, k: int = 5, *, doc_id: str | None = None, expand: bool | None = None,
        topic: str | None = None, category: str | None = None, audience: str | None = None,
        form_number: str | None = None, doc_type: str | None = None, group_by_doc: bool = False,
        reranker=None, rerank_pool: int | None = None, verified_only: bool = False,
        dedup: bool = False, dedup_threshold: float = 0.9,
    ) -> list[SearchHit]:
        """Hybrid lexical+dense search, optionally restricted to a facet and/or
        collapsed to one best hit per document.

        Facet filters (``topic``/``category``/``audience``/``form_number``/
        ``doc_type``) restrict results to documents with that metadata — e.g.
        ``topic="safety"`` to sweep every safe-sleep/abuse/neglect document.
        ``group_by_doc=True`` returns the single best section per document, so a
        topical query surfaces *all* relevant documents rather than many chunks
        of one.

        ``rerank_pool`` is a MINIMUM bound (P-3), not an exact pool size: the
        retrieval pool is widened to ``max(k*4, 20, k*8 if filtered, k*6 if
        reranking, rerank_pool)`` and the rerank stage takes the FIRST
        ``max(k*6, 40, rerank_pool)`` of that — so ``rerank_pool=20`` with
        ``k=10`` still reranks 40 candidates, not 20. To cap rerank cost,
        either lower ``k`` or accept the floor.
        """
        # Empty/whitespace-only query → no results, before ANY lane runs. The
        # lexical lane already returns [] for a blank query, but the dense lane
        # would happily embed "" and return arbitrary chunks at flat RRF scores.
        if not query or not query.strip():
            return []
        if not self.chunks:
            return []
        facet_filters = {
            k_: v for k_, v in (
                ("topic", topic), ("category", category), ("audience", audience),
                ("form_number", form_number), ("doc_type", doc_type),
            ) if v is not None
        }
        allowed: set[str] | None = None
        if facet_filters:
            allowed = {
                did for did, m in self.documents.items()
                if all(getattr(m, f, "") == v for f, v in facet_filters.items())
            }
            if not allowed:
                return []
        if verified_only:
            # Restrict to documents a HUMAN reviewer has verified (R3 / AC3.3).
            # "verified" is only set by ``parsevault review accept|correct|reject``;
            # the machine no-flags state is "no_flags" and does NOT qualify, so
            # the default for docs without the field is the honest "unreviewed".
            verified = {did for did, m in self.documents.items()
                        if getattr(m, "review_status", "") == "verified"}
            allowed = verified if allowed is None else (allowed & verified)
            if not allowed:
                return []
        # Widen the candidate pool when filtering/grouping so we still fill k.
        pool = max(k * 4, 20)
        if allowed is not None or group_by_doc:
            pool = max(pool, k * 8, 80)
        if reranker is not None:
            # Retrieve a wide pool of candidates for the cross-encoder to re-score.
            pool = max(pool, rerank_pool or max(k * 6, 40))
        lexical = self._retriever.search(
            query, k=pool, group=doc_id,
            expand=self.expand if expand is None else expand,
        )
        # L-2 audit: capture which lane(s) ranked each chunk so SearchHit.lane
        # can carry that provenance forward (Source.lane in rag.py mirrors it).
        lex_ids = {cid for cid, _ in lexical}
        dense_ids: set[str] = set()
        if self._dense is not None:
            # Fuse the dense semantic ranking with the lexical hybrid via RRF.
            dense = self._dense.search(query, k=pool, group=doc_id)
            dense_ids = {cid for cid, _ in dense}
            ranked = reciprocal_rank_fusion(
                [(lexical, 1.0), (dense, self.dense_weight)]
            )
        else:
            ranked = lexical
        if allowed is not None:
            ranked = [
                (cid, s) for cid, s in ranked
                if (c := self.chunks.get(cid)) is not None and c.doc_id in allowed
            ]
        if group_by_doc:
            seen: set[str] = set()
            collapsed: list[tuple[str, float]] = []
            for cid, s in ranked:
                c = self.chunks.get(cid)
                if c is not None and c.doc_id not in seen:
                    seen.add(c.doc_id)
                    collapsed.append((cid, s))
            ranked = collapsed
        # M-6: surface a query-time note when the dense lane is materially
        # degraded for this corpus (>5% of chunks have no vector). R4 wants
        # "no silent degradation" — the build_report already counts these,
        # but at query time the search code had no way to tell the caller
        # that the fused ranking ran over a partial dense lane.
        dense_degraded_note = ""
        if self._dense is not None and self._dense.missing and self.chunks:
            ratio = len(self._dense.missing) / max(1, len(self.chunks))
            if ratio > 0.05:
                dense_degraded_note = (
                    f"dense lane degraded: {len(self._dense.missing)} of "
                    f"{len(self.chunks)} chunks are lexical-only "
                    f"({ratio:.0%}); re-embed to restore semantic ranking"
                )
        rerank_note = ""
        if reranker is not None and ranked:
            # Cross-encoder precision pass over the retrieved pool, then take k.
            cand = ranked[: (rerank_pool or max(k * 6, 40))]
            texts = [self.chunks[cid].text for cid, _ in cand if cid in self.chunks]
            cand = [(cid, s) for cid, s in cand if cid in self.chunks]
            try:
                order = reranker.rerank(query, texts)
                ranked = [(cand[i][0], score) for i, score in order]
            except Exception as e:  # noqa: BLE001
                # No silent degradation (R4.4): record it and annotate the hits so
                # the caller knows the precision pass did not run.
                import logging

                logging.getLogger(__name__).warning("reranker skipped: %s", e)
                self.record_degradation("reranker_skipped", f"{type(e).__name__}: {e}")
                rerank_note = "reranker unavailable — results are hybrid-ranked, not reranked"
        dropped_by_dedup = 0
        if dedup:
            # Near-duplicate collapsing (Tier 1 precision): the corpus repeats
            # boilerplate across documents, so without this the top-k can be many
            # copies of one passage. Drop a candidate whose char-trigram profile
            # is near-identical to an already-kept hit, then fill k from the pool.
            #
            # TRADE-OFF (S-6): on STATUTORY text, two adjacent chunks of the
            # same section can be near-identical char-trigram-wise (recurring
            # heading + boilerplate) and dedup will drop the second, losing a
            # gold passage. ``rag.answer`` exposes ``dedup=`` as a kwarg so a
            # caller can disable it for statute-heavy questions; the default
            # stays on to preserve current ranking behavior.
            kept: list[tuple[str, float]] = []
            kept_vecs: list[Counter] = []
            for cid, s in ranked:
                c = self.chunks.get(cid)
                if c is None:
                    continue
                vec = char_ngrams(c.text)
                if any(_cosine(vec, kv) >= dedup_threshold for kv in kept_vecs):
                    dropped_by_dedup += 1
                    continue
                kept.append((cid, s))
                kept_vecs.append(vec)
                if len(kept) >= k:
                    break
            ranked = kept
            if dropped_by_dedup:
                # Record on the index degradation ledger so callers/auditors can
                # see when dedup is silently filtering candidates (S-6).
                self.record_degradation(
                    "dedup_dropped_chunks",
                    f"dropped {dropped_by_dedup} near-duplicate chunk(s) "
                    f"(threshold={dedup_threshold})",
                )
        else:
            ranked = ranked[:k]
        q_terms = tokenize(query)
        dedup_note = (
            f"dedup dropped {dropped_by_dedup} near-duplicate chunk(s)"
            if dropped_by_dedup else ""
        )
        hits: list[SearchHit] = []
        for cid, score in ranked:
            chunk = self.chunks.get(cid)
            if chunk is None:
                continue
            meta = self.documents.get(chunk.doc_id)
            notes = [rerank_note] if rerank_note else []
            if dedup_note:
                notes.append(dedup_note)
            if dense_degraded_note:
                notes.append(dense_degraded_note)  # M-6
            if meta and (sn := staleness_note(meta)):
                notes.append(sn)  # source-currency warning (R8.3)
            # L-2 audit: derive lane from which lane(s) surfaced this chunk.
            # "hybrid" = both lexical and dense ranked it; "bm25"/"dense" =
            # only one; "" when the index has no dense lane and we can't
            # distinguish (the field defaults to "" — lexical-only indexes
            # naturally have no dense contribution to attribute).
            if dense_ids:
                in_lex = cid in lex_ids
                in_dense = cid in dense_ids
                if in_lex and in_dense:
                    lane = "hybrid"
                elif in_dense:
                    lane = "dense"
                elif in_lex:
                    lane = "bm25"
                else:
                    lane = ""
            else:
                lane = "bm25" if cid in lex_ids else ""
            hits.append(
                SearchHit(
                    score=round(score, 6),
                    chunk=chunk,
                    title=meta.title if meta else chunk.doc_id,
                    heading_path=chunk.heading_path,
                    snippet=self._snippet(chunk.text, q_terms),
                    doc_id=chunk.doc_id,
                    source_url=meta.source_url if meta else "",
                    source_domain=meta.source_domain if meta else "",
                    section=meta.section if meta else "",
                    retrieved_at=meta.retrieved_at if meta else "",
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    extraction_routes=list(chunk.extraction_routes),
                    notes=notes,
                    lane=lane,
                )
            )
        return hits

    @staticmethod
    def _snippet(text: str, q_terms: list[str], width: int = 420) -> str:
        # Display-time cleanup only (stored chunks are untouched): converted
        # tables leak literal <br> tags + pipe-row chrome into snippets, and
        # OCR'd forms repeat the same sentence back-to-back.
        flat = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
        flat = flat.replace("|", " ")
        flat = re.sub(r"\s+", " ", flat).strip()
        sentences = re.split(r"(?<=[.!?]) ", flat)
        deduped = [s for i, s in enumerate(sentences)
                   if i == 0 or s != sentences[i - 1]]
        flat = " ".join(deduped)
        low = flat.lower()
        # Word-boundary match so 'act' centers on the word, not inside 'practice'
        # (R10.3). Fall back to first query-term occurrence, else the start.
        positions = []
        for t in q_terms:
            if m := re.search(rf"\b{re.escape(t)}\b", low):
                positions.append(m.start())
        pos = min(positions) if positions else 0
        start = max(0, pos - width // 3)
        snip = flat[start : start + width]
        return ("…" if start > 0 else "") + snip + ("…" if start + width < len(flat) else "")

    # -- exact-pattern lane (statute/USC/CFR/Bates/quoted) --------------------
    def search_regex(
        self, query: str, k: int = 10, *, workspace: str | Path | None = None,
    ):
        """Run the exact-pattern lane (RegexRetriever) over the on-disk
        ``outputs/*.md`` shards.

        Opt-in — ``search()`` semantics are unchanged. Returns ``list[RegexHit]``.
        ``workspace`` defaults to the path passed at the most recent ``load``;
        callers building an index in-memory should pass it explicitly.
        """
        from .regex_retriever import RegexRetriever

        ws = Path(workspace) if workspace else getattr(self, "_workspace", None)
        if ws is None:
            return []
        # Cache one RegexRetriever per workspace (reused across queries).
        cached = getattr(self, "_regex_retriever", None)
        if cached is None or getattr(cached, "workspace", None) != ws:
            cached = RegexRetriever(ws)
            self._regex_retriever = cached
        return cached.search(query, k=k, index=self)

    # -- example-based retrieval ("find documents that look like this") -------
    def _doc_texts(self) -> dict[str, str]:
        """Reconstruct each document's text from its chunks (ordinal order).

        PERF-D2-2: memoized on ``self._chunks_version``. At full-corpus scale
        (6,760 docs / 109,713 chunks) the uncached rebuild is several hundred
        MB of string allocation per call — /similar?doc=… was paying it on
        every click. ``add()`` / ``remove()`` bump the version to invalidate.
        """
        cache = self._doc_texts_cache
        if cache is not None and cache[0] == self._chunks_version:
            return cache[1]
        parts: dict[str, list[tuple[int, str]]] = {}
        for c in self.chunks.values():
            parts.setdefault(c.doc_id, []).append((c.ordinal, c.text))
        built = {
            did: "\n".join(t for _o, t in sorted(ps)) for did, ps in parts.items()
        }
        self._doc_texts_cache = (self._chunks_version, built)
        return built

    def _term_doc_freq(self, doc_texts: dict[str, str]) -> Counter:
        """PERF-D2-2: memoize the corpus DF when called with the canonical
        cached ``doc_texts`` (same identity → same version). Callers passing a
        DIFFERENT dict (e.g. a deliberately-subset corpus) bypass the cache
        and recompute, which is the safe behaviour.
        """
        cached_texts = self._doc_texts_cache
        if (
            cached_texts is not None
            and cached_texts[0] == self._chunks_version
            and cached_texts[1] is doc_texts
            and self._term_df_cache is not None
            and self._term_df_cache[0] == self._chunks_version
        ):
            return self._term_df_cache[1]
        df: Counter = Counter()
        for text in doc_texts.values():
            df.update(set(tokenize(text)))
        # Only stash the cache when this DF corresponds to the canonical corpus
        # texts (identity match) — that's the path /similar pays the cost on.
        if (
            cached_texts is not None
            and cached_texts[0] == self._chunks_version
            and cached_texts[1] is doc_texts
        ):
            self._term_df_cache = (self._chunks_version, df)
        return df

    def distinctive_terms(self, text: str, *, n: int = 24,
                          doc_texts: dict[str, str] | None = None) -> list[str]:
        """The example's most distinctive terms (TF·IDF over this index) — the
        salient vocabulary that makes a focused 'find me records like this' query
        instead of drowning in stopwords/boilerplate.

        Two filters keep the query discriminating: a term must actually occur in
        the corpus (``df >= 1``) — an example-only token (e.g. a party name unique
        to it) can retrieve nothing, so it only wastes a query slot — and it must
        not be ubiquitous (``df`` below ~60% of docs), since a term in nearly
        every document carries no signal about *which* ones resemble the example.
        """
        tf = Counter(t for t in tokenize(text) if len(t) > 2)
        if not tf:
            return []
        doc_texts = doc_texts if doc_texts is not None else self._doc_texts()
        ndoc = max(len(doc_texts), 1)
        df = self._term_doc_freq(doc_texts)
        ceil = max(2, int(0.6 * ndoc))
        scored = {
            t: c * math.log(ndoc / df[t])
            for t, c in tf.items()
            if 1 <= df.get(t, 0) <= ceil
        }
        return [t for t, _ in sorted(scored.items(), key=lambda kv: -kv[1])[:n]]

    def find_similar(
        self, example_text: str, *, k: int = 10, restrict_type: bool = False,
        same_type_boost: bool = True, layout_weight: float = 0.4, type_weight: float = 0.3,
        n_terms: int = 24, exclude_doc_ids: set[str] | None = None,
        layout_only: bool = False,
    ) -> list[SimilarHit]:
        """Rank documents in THIS index by resemblance to ``example_text``.

        The core "find all the documents that resemble the contents of this form"
        query. Three explainable signals are fused (per the case-workspace
        design): the example's distinctive **content** vocabulary (lexical, plus
        the dense lane when present), structural **layout** similarity (content-
        free shape), and document **type** match. Strictly scoped to this one
        index — isolation is the scope.

        **Content gates the pool.** "Resemble the *contents*" means a document
        with no content overlap is not a match — so layout and type only *re-rank
        within* the content matches, never surface a content-irrelevant document
        on shape alone. This matters for a single-case collection where every
        document shares the same type and near-identical layout: there, those two
        lanes are non-discriminating and must not drown out the content signal.
        (Pass ``layout_only=True`` to instead match on shape alone — e.g. finding
        every copy of a blank form template regardless of what was typed in.)

        ``restrict_type=True`` returns only same-type documents (the spec's
        same-subtype default, hardened). ``exclude_doc_ids`` drops the example
        itself if it already lives here.
        """
        from .doctype import classify_document, layout_signature, layout_similarity

        if not self.chunks:
            return []
        exclude = exclude_doc_ids or set()
        doc_texts = self._doc_texts()
        ex_type = classify_document(example_text).label
        ex_layout = layout_signature(example_text)
        ndocs = len(doc_texts)

        # 1) Content lane — distinctive terms drive the hybrid (BM25F + dense)
        #    search. A wide pool so recall isn't clipped on a large case.
        terms = self.distinctive_terms(example_text, n=n_terms, doc_texts=doc_texts)
        pool_n = min(ndocs, max(k * 10, 200))
        content_ranked: list[tuple[str, float]] = []
        # M4: cache the best snippet per doc from the content search so the per-
        # candidate snippet pass doesn't run a fresh search for each result.
        snippet_by_doc: dict[str, str] = {}
        if terms:
            hits = self.search(" ".join(terms), k=pool_n, group_by_doc=True)
            content_ranked = [(h.doc_id, h.score) for h in hits if h.doc_id not in exclude]
            for h in hits:
                if h.doc_id not in snippet_by_doc:
                    snippet_by_doc[h.doc_id] = h.snippet

        # 2) Candidate pool. Content gates it (unless layout_only). Optionally
        #    restrict to the example's type.
        if layout_only:
            candidates = [did for did in doc_texts if did not in exclude]
        else:
            candidates = [did for did, _s in content_ranked]
        if restrict_type:
            candidates = [did for did in candidates
                          if (m := self.documents.get(did)) and m.doc_type == ex_type]
        candidate_set = set(candidates)
        content_ranked = [(d, s) for d, s in content_ranked if d in candidate_set]
        content_pos = {did: i for i, (did, _s) in enumerate(content_ranked)}

        # 3) Layout lane — structural shape similarity, computed only over the pool.
        layout_sim = {}
        for did in candidates:
            meta = self.documents.get(did)
            sig = (meta.layout if meta and meta.layout else layout_signature(doc_texts[did]))
            layout_sim[did] = layout_similarity(ex_layout, sig)
        layout_ranked = sorted(layout_sim.items(), key=lambda kv: -kv[1])

        # 4) Fuse, all lanes restricted to the pool (RRF, scale-free). Content
        #    leads; layout and type re-rank within and break ties. In a
        #    homogeneous collection the uniform layout/type lanes leave content
        #    order essentially intact; in a mixed one they sharpen it.
        rankings = [
            ([(d, 0.0) for d, _ in content_ranked], 1.0 if not layout_only else 0.0),
            (layout_ranked, layout_weight if not layout_only else 1.0),
        ]
        if same_type_boost and not restrict_type:
            same = [(did, 1.0) for did in candidates
                    if (m := self.documents.get(did)) and m.doc_type == ex_type]
            rankings.append((same, type_weight))
        fused = reciprocal_rank_fusion(rankings)

        out: list[SimilarHit] = []
        for did, score in fused:
            meta = self.documents.get(did)
            if meta is None or did in exclude or did not in candidate_set:
                continue
            tmatch = meta.doc_type == ex_type
            lsim = layout_sim.get(did, 0.0)
            cscore = (1.0 - content_pos[did] / max(len(content_ranked), 1)) if did in content_pos else 0.0
            why = []
            if did in content_pos:
                why.append(f"content overlap (rank {content_pos[did] + 1})")
            if lsim >= 0.85:
                why.append(f"matching layout ({lsim:.2f})")
            if tmatch and ex_type != "other":
                why.append(f"same type ({ex_type})")
            # M4: reuse the cached snippet from the content search (single
            # hybrid pass) instead of running ``self.search`` per candidate.
            # Falls back to a fresh per-doc search only when this doc didn't
            # appear in the content-search pool (layout_only mode, or it
            # ranked beyond pool_n).
            snippet = snippet_by_doc.get(did)
            if snippet is None:
                best = self.search(
                    " ".join(terms) or example_text[:200], k=1, doc_id=did,
                )
                snippet = best[0].snippet if best else ""
            out.append(SimilarHit(
                doc_id=did, title=meta.title, score=round(score, 6),
                content_score=round(cscore, 4), layout_sim=round(lsim, 4),
                type_match=tmatch, doc_type=meta.doc_type,
                snippet=snippet,
                source_path=meta.source_path, why=why,
            ))
            if len(out) >= k:
                break
        return out

    # -- persistence ----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": INDEX_VERSION,
            "retrieval": {
                "k1": self._retriever.k1,
                "b": self._retriever.b,
                "heading_boost": self._retriever.heading_boost,
                "char_weight": self._retriever.char_weight,
                "expand": self.expand,
            },
            "documents": [m.to_dict() for m in self.documents.values()],
            "chunks": [c.to_dict() for c in self.chunks.values()],
            "degradations": self.degradations,
        }

    @staticmethod
    def _vectors_path(path: Path) -> Path:
        return path.with_suffix(path.suffix + ".vectors.json")

    @staticmethod
    def _integrity_path(path: Path) -> Path:
        return path.with_suffix(path.suffix + ".integrity")

    @staticmethod
    def _audit_path(path: Path) -> Path:
        return path.parent / "audit.log"

    def save(self, path: str | Path, *, operation: str = "build",
             affected: list[str] | None = None, audit: bool = True,
             vectors: str | bool = "auto") -> Path:
        """Persist the index (+ integrity sidecar + audit line).

        ``vectors`` controls the dense sidecar:
        * ``"auto"`` (default) — write it when the dense store has vectors;
          otherwise LEAVE any existing sidecar in place. A metadata-only save
          from a lexical-only session (``EMBED_MODEL=none``) must never wipe
          previously-computed vectors — deletion has to be explicit.
        * ``"keep"`` — never touch the sidecar, even when a dense store with
          vectors is present. Use for metadata-only updates (review status,
          re-score, provenance stamp).
        * ``"drop"`` (or ``vectors=False``) — explicitly delete the sidecar.
          The only path that removes ``.vectors.json``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        # Atomic: tmp + os.replace, so a SIGINT/OOM mid-write can never leave
        # a half-written index at ``path`` (F-5).
        _atomic_write_text(path, payload)
        index_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        # Persist dense vectors to a sidecar (model-tagged) so reloads don't
        # re-embed the whole corpus. Kept out of the main JSON to keep it small.
        vec_path = self._vectors_path(path)
        if vectors == "keep":
            pass  # metadata-only update — never touch the vectors sidecar
        elif vectors in (False, "drop"):
            # Explicit deletion request — the ONLY path that unlinks the sidecar.
            if vec_path.exists():
                vec_path.unlink()
        elif self._dense is not None and self._dense.vectors:
            _atomic_write_text(vec_path, json.dumps({
                "model": getattr(self._dense.embedder, "name", ""),
                "vectors": self._dense.vectors,
                "groups": self._dense.groups,
            }))
        # else: "auto" with an empty/absent dense store — leave any existing
        # sidecar in place. A lexical-only session saving metadata must not
        # silently destroy vectors and force a future full re-embed.
        # Integrity manifest (R5.1): tamper-evidence for the index file itself.
        self._write_integrity(path, index_sha)
        # Append-only audit chain (R5.2): one line per mutating save, each
        # carrying the prior index hash → a verifiable chain.
        if audit:
            self._append_audit(path, operation, index_sha, affected or [])
        return path

    def _write_integrity(self, path: Path, index_sha: str) -> None:
        import datetime

        _atomic_write_text(self._integrity_path(path), json.dumps({
            "index_sha256": index_sha,
            "version": INDEX_VERSION,
            "parsevault_version": _parsevault_version(),
            "saved_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        }, indent=2))

    def _append_audit(self, path: Path, operation: str, index_sha: str,
                      affected: list[str]) -> None:
        import datetime

        ap = self._audit_path(path)
        prev = ""
        prev_line_sha = "0" * 64
        if ap.exists():
            for line in reversed(ap.read_text(encoding="utf-8").splitlines()):
                if line.strip():
                    try:
                        parsed = json.loads(line)
                        prev = parsed.get("index_sha256", "")
                        # Per-line back-pointer (SEC-02). Legacy lines without a
                        # ``line_sha256`` field fall back to the all-zero genesis
                        # value, so the chain continues but won't retroactively
                        # validate the pre-chain prefix.
                        prev_line_sha = parsed.get("line_sha256", "0" * 64)
                    except json.JSONDecodeError:
                        prev = ""
                        prev_line_sha = "0" * 64
                    break
        entry = {
            "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "operation": operation,
            "doc_count": len(self.documents),
            "affected_doc_ids": affected,
            "prev_index_sha256": prev,
            "index_sha256": index_sha,
            # SEC-02 per-line hash chain. ``line_sha256`` covers the prior line's
            # ``line_sha256`` plus the canonical body of THIS line (excluding
            # ``line_sha256`` itself). Any mutation of any prior line invalidates
            # every subsequent ``line_sha256`` — detectable by
            # ``verify_audit_chain``.
            "prev_line_sha256": prev_line_sha,
        }
        body_canonical = json.dumps(entry, ensure_ascii=False, sort_keys=True)
        line_sha = hashlib.sha256(
            (prev_line_sha + body_canonical).encode("utf-8")
        ).hexdigest()
        entry["line_sha256"] = line_sha
        with ap.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    @classmethod
    def verify_integrity(cls, path: str | Path) -> bool:
        """True if the index file matches its integrity sidecar.

        When the sidecar is absent this returns True (no constraint to check)
        but emits a WARNING so the gap is visible. Callers that need tamper-
        evidence as a hard invariant should run under ``PARSEVAULT_STRICT=1``;
        ``load`` raises ``IndexIntegrityError`` in that mode when the sidecar
        is missing. See audit S-2.
        """
        path = Path(path)
        integ = cls._integrity_path(path)
        if not integ.exists():
            import logging
            logging.getLogger(__name__).warning(
                "integrity sidecar absent for %s — index cannot be tamper-verified",
                path,
            )
            return True
        expected = json.loads(integ.read_text(encoding="utf-8")).get("index_sha256", "")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        return (not expected) or expected == actual

    @classmethod
    def load(cls, path: str | Path, *, embedder=None) -> "DocIndex":
        """Load an index. With an ``embedder``, the dense lane is restored from
        the model-tagged vectors sidecar when present (instant) and only
        re-embedded if it is missing/stale (different model or chunk set)."""
        path = Path(path)
        # S-2: under strict mode, a missing integrity sidecar is a hard failure —
        # tamper-evidence can't be claimed without it (R5.1). Non-strict mode
        # keeps today's behavior (warn + load) so existing workspaces without
        # sidecars still open.
        integ_path = cls._integrity_path(Path(path))
        if not integ_path.exists() and _strict():
            raise IndexIntegrityError(
                f"integrity sidecar missing for {path} — tamper-evidence cannot be "
                f"verified (strict mode requires the .integrity sidecar).")
        # Integrity check (R5.1): a hand-edited index is detected here — warn, or
        # fail under strict mode. Verified against the sibling .integrity sidecar.
        if not cls.verify_integrity(path):
            import logging

            msg = (f"index integrity mismatch for {path} — the file does not match "
                   f"its .integrity manifest (it may have been edited).")
            if _strict():
                raise IndexIntegrityError(msg)
            logging.getLogger(__name__).warning(msg)
        data = json.loads(path.read_text(encoding="utf-8"))
        cfg = dict(data.get("retrieval", data.get("bm25", {})))
        expand = bool(cfg.pop("expand", False))
        # Build the lexical index first without embedding (no dense yet).
        idx = cls(expand=expand, embedder=None, **cfg)
        # Workspace = parent dir of docindex.json. Lets ``search_regex`` find
        # ``outputs/`` without the caller having to pass the path again.
        idx._workspace = path.parent
        idx.degradations = list(data.get("degradations", []))
        docs = {d["doc_id"]: DocMetadata.from_dict(d) for d in data.get("documents", [])}
        by_doc: dict[str, list[Chunk]] = {doc_id: [] for doc_id in docs}
        for cd in data.get("chunks", []):
            chunk = Chunk(**cd)
            by_doc.setdefault(chunk.doc_id, []).append(chunk)
        for doc_id, meta in docs.items():
            idx.add(meta, sorted(by_doc.get(doc_id, []), key=lambda c: c.ordinal))

        if embedder is not None:
            from .embeddings import DenseStore

            store = DenseStore(embedder)
            idx._dense = store
            idx._wire_dense_degradation()  # record zero-vectors during re-embed
            saved: dict = {}
            vec_path = cls._vectors_path(path)
            if vec_path.exists():
                # Corruption tolerance: a truncated/garbled/unreadable sidecar
                # must never brick index load — treat it as stale (re-embed
                # path) and warn. ValueError covers json.JSONDecodeError.
                try:
                    loaded = json.loads(vec_path.read_text(encoding="utf-8"))
                    if not isinstance(loaded, dict):
                        raise ValueError("sidecar root is not a JSON object")
                    saved = loaded
                except (OSError, ValueError, KeyError) as e:
                    import logging

                    logging.getLogger(__name__).warning(
                        "vectors sidecar %s unreadable (%s: %s) — treating as "
                        "stale; re-embedding", vec_path, type(e).__name__, e)
                    saved = {}
            chunk_ids = set(idx.chunks)
            saved_vectors = saved.get("vectors")
            saved_vectors = saved_vectors if isinstance(saved_vectors, dict) else {}
            # A sidecar without "groups" is legal (older/partial writes): the
            # group (doc_id) is recoverable from the chunk itself.
            saved_groups = saved.get("groups")
            saved_groups = saved_groups if isinstance(saved_groups, dict) else {}
            model_matches = bool(saved_vectors) and (
                saved.get("model") == getattr(embedder, "name", ""))
            # Load every model-matching vector that still corresponds to a live
            # chunk; sidecar entries for chunks no longer in the index are
            # dropped here (and thus dropped from disk on the persist below).
            usable = (chunk_ids & set(saved_vectors)) if model_matches else set()
            store.vectors = {k: saved_vectors[k] for k in usable}
            store.groups = {k: saved_groups.get(k) or idx.chunks[k].doc_id
                            for k in usable}
            missing = chunk_ids - usable
            if missing:
                # Delta embed: ONLY the chunks the sidecar doesn't cover. On a
                # model-tag mismatch ``usable`` is empty → full re-embed
                # (unchanged behavior); on a subset sidecar (index rebuilt
                # after the sidecar was written) this embeds just the new
                # chunks instead of the whole corpus.
                store.add_many([(c.chunk_id, c.text, c.doc_id)
                                for c in idx.chunks.values()
                                if c.chunk_id in missing])
                # Persist the refreshed sidecar so the embed cost is paid ONCE
                # — callers like the dashboard load-and-serve without ever
                # calling save(), which used to discard this work every launch.
                if store.vectors:
                    try:
                        _atomic_write_text(vec_path, json.dumps({
                            "model": getattr(embedder, "name", ""),
                            "vectors": store.vectors,
                            "groups": store.groups,
                        }))
                    except OSError as e:
                        import logging

                        logging.getLogger(__name__).warning(
                            "could not persist refreshed vectors sidecar %s: %s"
                            " — vectors will be recomputed on the next load",
                            vec_path, e)
            idx._dense = store
        return idx


# --------------------------------------------------------------------------- #
# SEC-02 — audit log per-line hash chain + verifier
# --------------------------------------------------------------------------- #
@dataclass
class AuditVerifyResult:
    """Outcome of walking an ``audit.log`` and recomputing each per-line hash.

    SEC-02: every line carries ``line_sha256 = sha256(prev_line_sha256 + body)``
    where ``body`` is the JSON-canonical form of the line with ``line_sha256``
    removed. ``verify_audit_chain`` recomputes this for every line; any
    mismatch (operation rewritten, affected_doc_ids edited, etc.) flips
    ``verified`` to False and pins ``first_bad_line`` (1-based).

    Lines written before SEC-02 lack the ``line_sha256`` field. Those are
    counted in ``legacy_lines`` and treated as PASS — we verify forward, we do
    not rewrite history. ``partial`` is True when any legacy line is in the
    log: an attacker who can forge a legacy-shaped line (no chain field)
    could smuggle content past the verifier because legacy lines do not
    advance ``prev_line_sha``. Callers that need full chain coverage should
    treat ``partial=True`` as "not a clean verification" rather than
    indistinguishable from the all-chained case (M-8 audit).
    """
    verified: bool
    total_lines: int
    first_bad_line: int | None = None
    bad_line_field: str | None = None
    legacy_lines: int = 0
    partial: bool = False
    workspace: str = ""
    audit_path: str = ""


def verify_audit_chain(workspace_path: str | Path) -> AuditVerifyResult:
    """Verify the SEC-02 per-line hash chain in ``<workspace_path>/audit.log``.

    Parameters
    ----------
    workspace_path :
        Directory holding ``audit.log`` (a KB workspace / case sandbox / any
        ``DocIndex`` parent dir). May also be the audit.log path itself.

    Returns
    -------
    AuditVerifyResult — see dataclass docstring.
    """
    wp = Path(workspace_path)
    audit_path = wp if wp.is_file() else wp / "audit.log"
    if not audit_path.exists():
        # No log = nothing to falsify. Treat as vacuously verified so an empty
        # fresh workspace doesn't trip the CLI gate.
        return AuditVerifyResult(
            verified=True, total_lines=0,
            workspace=str(wp), audit_path=str(audit_path),
        )

    lines = [ln for ln in audit_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    prev_line_sha = "0" * 64
    legacy_lines = 0
    for i, raw in enumerate(lines, start=1):
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            return AuditVerifyResult(
                verified=False, total_lines=len(lines),
                first_bad_line=i, bad_line_field="<malformed-json>",
                legacy_lines=legacy_lines, partial=legacy_lines > 0,
                workspace=str(wp), audit_path=str(audit_path),
            )

        if "line_sha256" not in entry:
            # Pre-SEC-02 line — pass-through. Don't advance ``prev_line_sha``
            # off the legacy line (we have nothing trustworthy to chain from);
            # leave the genesis value so the FIRST chained line still validates.
            legacy_lines += 1
            continue

        stored = entry["line_sha256"]
        body = {k: v for k, v in entry.items() if k != "line_sha256"}
        # The chain back-pointer must match what the previous chained line
        # advertised — catches re-ordering / line-deletion attacks even when
        # the body hash is recomputed by an attacker.
        declared_prev = body.get("prev_line_sha256", "0" * 64)
        if declared_prev != prev_line_sha:
            return AuditVerifyResult(
                verified=False, total_lines=len(lines),
                first_bad_line=i, bad_line_field="prev_line_sha256",
                legacy_lines=legacy_lines, partial=legacy_lines > 0,
                workspace=str(wp), audit_path=str(audit_path),
            )
        body_canonical = json.dumps(body, ensure_ascii=False, sort_keys=True)
        recomputed = hashlib.sha256(
            (prev_line_sha + body_canonical).encode("utf-8")
        ).hexdigest()
        if recomputed != stored:
            return AuditVerifyResult(
                verified=False, total_lines=len(lines),
                first_bad_line=i, bad_line_field="line_sha256",
                legacy_lines=legacy_lines, partial=legacy_lines > 0,
                workspace=str(wp), audit_path=str(audit_path),
            )
        prev_line_sha = stored

    return AuditVerifyResult(
        verified=True, total_lines=len(lines),
        legacy_lines=legacy_lines, partial=legacy_lines > 0,
        workspace=str(wp), audit_path=str(audit_path),
    )


# --------------------------------------------------------------------------- #
# evaluation — prove retrieval quality on a labeled set
# --------------------------------------------------------------------------- #
def _doc_matches(meta: "DocMetadata", label) -> tuple[bool, bool]:
    """Does ``label`` match ``meta``? Returns (matched, exact).

    Exact labels (R6.1) are dicts: ``{"doc_sha256": …}`` (source file hash),
    ``{"filename_exact": …}``, or ``{"doc_id": …}``. A bare string matches by
    doc_id (exact) or, failing that, a substring of the title/filename (matched
    but NOT exact — the eval flags queries that resolve only this way)."""
    if isinstance(label, dict):
        if "doc_sha256" in label:
            return (meta.source_sha256 == label["doc_sha256"], True)
        if "filename_exact" in label:
            return (Path(meta.source_path).name == label["filename_exact"], True)
        if "doc_id" in label:
            return (meta.doc_id == label["doc_id"], True)
        return (False, True)
    if label == meta.doc_id:
        return (True, True)
    low = str(label).lower()
    return (low in meta.title.lower() or low in Path(meta.source_path).name.lower(), False)


def _rebuild(index: "DocIndex", **retriever_kw) -> "DocIndex":
    """A new index over the same content with a different retriever config."""
    expand = retriever_kw.pop("expand", index.expand)
    new = DocIndex(expand=expand, **retriever_kw)
    by_doc: dict[str, list[Chunk]] = {}
    for c in index.chunks.values():
        by_doc.setdefault(c.doc_id, []).append(c)
    for doc_id, meta in index.documents.items():
        new.add(meta, sorted(by_doc.get(doc_id, []), key=lambda c: c.ordinal))
    return new


def evaluate(index: "DocIndex", labeled: list[dict], k: int = 5, *, reranker=None,
             per_query: bool = False) -> dict:
    """Compute mean Recall@k, MRR, and nDCG@k over labeled queries (R6).

    Each item is ``{"query": str, "relevant": [label, ...]}``; labels may be exact
    (``{"doc_sha256"|"filename_exact"|"doc_id": …}``) or bare strings (substring
    match — flagged). Queries whose relevant docs are absent from the index are
    reported (not silently skipped). With ``per_query=True`` the per-query ranks +
    top-k + decisive lane are returned for inspection.
    """
    import math as _m

    recall = mrr = ndcg = 0.0
    scored = 0
    reports: list[dict] = []
    substring_only: list[str] = []
    missing_relevant: list[str] = []
    has_dense = index._dense is not None

    for item in labeled:
        labels = item.get("relevant", [])
        rel_docs, exact_only = set(), True
        for d in index.documents:
            for lbl in labels:
                matched, exact = _doc_matches(index.documents[d], lbl)
                if matched:
                    rel_docs.add(d)
                    if not exact:
                        exact_only = False
        if not rel_docs:
            missing_relevant.append(item["query"])  # R6.5: report, don't hide
            if per_query:
                reports.append({"query": item["query"], "status": "no_relevant_in_index"})
            continue
        if not exact_only:
            substring_only.append(item["query"])
        scored += 1

        fused = [h.chunk.doc_id for h in
                 index.search(item["query"], k=k, group_by_doc=True, reranker=reranker)]
        topk, seen = [], set()
        for d in fused:
            if d not in seen:
                seen.add(d)
                topk.append(d)
        topk = topk[:k]

        retrieved_rel = [d for d in topk if d in rel_docs]
        recall += len(retrieved_rel) / len(rel_docs)
        mrr += next((1.0 / (i + 1) for i, d in enumerate(topk) if d in rel_docs), 0.0)
        dcg = sum(1.0 / _m.log2(i + 2) for i, d in enumerate(topk) if d in rel_docs)
        idcg = sum(1.0 / _m.log2(i + 2) for i in range(min(len(rel_docs), k)))
        ndcg += (dcg / idcg) if idcg else 0.0

        if per_query:
            ranks = {d: (topk.index(d) + 1 if d in topk else None) for d in rel_docs}
            reports.append({
                "query": item["query"],
                "relevant_ranks": ranks,
                "topk": topk,
                "found": bool(retrieved_rel),
                "decisive_lane": _decisive_lane(index, item["query"], topk, rel_docs, k)
                if has_dense else "lexical",
                "substring_only": not exact_only,
            })

    n = max(scored, 1)
    out = {
        "queries": scored,
        "recall@k": round(recall / n, 4),
        "mrr": round(mrr / n, 4),
        "ndcg@k": round(ndcg / n, 4),
        "k": k,
        "substring_only_queries": substring_only,
        "missing_relevant_queries": missing_relevant,
    }
    if per_query:
        out["per_query"] = reports
    return out


def _decisive_lane(index: "DocIndex", query: str, topk: list[str],
                   rel_docs: set[str], k: int) -> str:
    """Coarse attribution: did the dense lane move the best relevant doc into the
    top-k (vs lexical-only)? 'dense' if so, else 'lexical'."""
    best = next((d for d in topk if d in rel_docs), None)
    if best is None:
        return "none"
    lex = index._retriever.search(query, k=max(k * 4, 20), expand=index.expand)
    lex_docs, seen = [], set()
    for cid, _s in lex:
        c = index.chunks.get(cid)
        if c and c.doc_id not in seen:
            seen.add(c.doc_id)
            lex_docs.append(c.doc_id)
    lex_rank = lex_docs.index(best) + 1 if best in lex_docs[:k] else None
    fused_rank = topk.index(best) + 1
    return "dense" if (lex_rank is None or lex_rank > fused_rank) else "lexical"


def eval_regressed(current: dict, baseline: dict, *, tol: float = 0.02) -> list[str]:
    """Regression messages if recall@k or nDCG@k dropped > tol vs baseline (R6.4).
    Empty list = within tolerance (gate passes)."""
    msgs = []
    for metric in ("recall@k", "ndcg@k"):
        if metric in baseline and current.get(metric, 0.0) < baseline[metric] - tol:
            msgs.append(
                f"{metric} regressed: {current.get(metric)} < "
                f"{baseline[metric]} − {tol} (baseline {baseline[metric]})"
            )
    return msgs


def compare_configs(index: "DocIndex", labeled: list[dict], k: int = 5) -> dict[str, dict]:
    """Ablation: lexical-only vs hybrid vs hybrid+expansion, on the same content.

    This is how we *show* the hybrid stack beats plain BM25 on a given corpus
    rather than asserting it.
    """
    variants = {
        "bm25f (lexical)": _rebuild(index, char_weight=0.0, expand=False),
        "hybrid": _rebuild(index, char_weight=0.6, expand=False),
        "hybrid+expand": _rebuild(index, char_weight=0.6, expand=True),
    }
    return {name: evaluate(idx, labeled, k=k) for name, idx in variants.items()}


# --------------------------------------------------------------------------- #
# source currency (R8) — effective date, supersession, staleness
# --------------------------------------------------------------------------- #
def _date_key(s: str) -> tuple[int, int, int]:
    """A sortable (year, month, day) key from an ISO / MM-DD-YY / bare-year date."""
    s = (s or "").strip()
    if m := re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", s):
        return (int(m[1]), int(m[2]), int(m[3]))
    if m := re.match(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", s):
        y = int(m[3])
        return (y + 2000 if y < 100 else y, int(m[1]), int(m[2]))
    if m := re.search(r"(?:19|20)\d{2}", s):
        return (int(m.group()), 0, 0)
    return (0, 0, 0)


def mark_supersessions(index: "DocIndex") -> list[dict]:
    """Mark older revisions superseded when documents share a form number (R8.2).

    Returns one event per supersession (for the build report)."""
    import collections

    groups: dict[str, list[DocMetadata]] = collections.defaultdict(list)
    for m in index.documents.values():
        if m.form_number and m.effective_date:
            groups[m.form_number].append(m)
    events: list[dict] = []
    for form, metas in groups.items():
        if len(metas) < 2:
            continue
        ordered = sorted(metas, key=lambda m: _date_key(m.effective_date))
        newest = ordered[-1]
        for m in ordered[:-1]:
            if _date_key(m.effective_date) < _date_key(newest.effective_date):
                m.currency_status = "superseded"
                m.superseded_by = newest.doc_id
                events.append({
                    "form_number": form, "superseded_doc": m.doc_id,
                    "superseded_date": m.effective_date,
                    "superseded_by": newest.doc_id, "current_date": newest.effective_date,
                })
    return events


# Document types/sections that don't go stale on a calendar (statutes/case law).
_CURRENCY_EXEMPT = ("statute", "u.s. code", "u.s.c", "cfr", "regulation",
                    "general statutes", "supreme court", "case")


def _is_currency_exempt(meta: "DocMetadata") -> bool:
    """True when this doc is a statute/regulation/case law (no calendar staleness)."""
    haystack = f"{meta.section} {meta.doc_type} {meta.category}".lower()
    return any(k in haystack for k in _CURRENCY_EXEMPT)


def _render_currency(meta: "DocMetadata") -> str:
    """Human-readable currency rendering for the audit trace (M3).

    The raw ``currency_status`` (``current``/``unknown``/``superseded``) is
    correct internally but confusing in legal output: ``Currency : unknown``
    on a U.S.C. statute (which has no revision date by design) reads as "the
    system is uncertain" rather than "permanent text by category". Tighten
    the rendering so an attorney sees the right reason:

    * ``superseded`` — render the raw label (the staleness note carries detail).
    * exempt + no effective_date — "permanent (no revision date)".
    * has effective_date — append "effective <date>".
    * otherwise (unknown, non-exempt) — keep the raw label for honesty.
    """
    status = meta.currency_status or "unknown"
    if status == "superseded":
        return status
    if not meta.effective_date and _is_currency_exempt(meta):
        return "permanent (no revision date)"
    if meta.effective_date:
        return f"{status}, effective {meta.effective_date}"
    return status


def staleness_note(meta: "DocMetadata", *, horizon_months: int = 18, today=None) -> str:
    """A human staleness warning for a hit, or '' (R8.3).

    Superseded documents always warn. Policy/forms older than ``horizon_months``
    (by retrieval date) warn; statutes/regulations/case law are exempt."""
    if meta.currency_status == "superseded":
        by = f" (superseded by {meta.superseded_by})" if meta.superseded_by else ""
        return f"superseded revision{by}"
    haystack = f"{meta.section} {meta.doc_type} {meta.category}".lower()
    if any(k in haystack for k in _CURRENCY_EXEMPT):
        return ""
    if not meta.retrieved_at:
        return ""
    import datetime

    today = today or datetime.date.today()
    yr, mo, dy = _date_key(meta.retrieved_at)
    if not yr:
        return ""
    age_months = (today.year - yr) * 12 + (today.month - (mo or 1))
    if age_months >= horizon_months:
        return f"retrieved {meta.retrieved_at} (~{age_months} months ago) — verify currency"
    return ""


# --------------------------------------------------------------------------- #
# build report — the artifact a legal team reviews after each corpus update
# --------------------------------------------------------------------------- #
def _sample_queries(index: "DocIndex", n: int = 8) -> list[str]:
    """A few representative queries (document titles) for the latency probe."""
    out: list[str] = []
    for m in index.documents.values():
        t = (m.title or "").strip()
        if t and not is_generic_title(t):
            out.append(t)
        if len(out) >= n:
            break
    return out


def build_report(index: "DocIndex", *, eval_metrics: dict | None = None,
                 supersessions: list[dict] | None = None,
                 measure_latency: bool = True) -> dict:
    """Summarize a build for review (R4.5): coverage, lanes, every tolerated
    degradation, quality flags, supersessions, and the measured latency envelope
    (R9.1). This is what makes "no silent degradation" auditable."""
    import collections

    lanes: collections.Counter = collections.Counter()
    for c in index.chunks.values():
        for r in (c.extraction_routes or []):
            lanes[r] += 1
    deg_by_kind = collections.Counter(d.get("kind", "?") for d in index.degradations)
    dense_missing = sorted(index._dense.missing) if index._dense is not None else []
    flagged_docs = [m.doc_id for m in index.documents.values() if m.flagged_pages]
    superseded = [m.doc_id for m in index.documents.values()
                  if m.currency_status == "superseded"]
    report = {
        "documents": len(index.documents),
        "chunks": len(index.chunks),
        "pages_with_provenance": sum(1 for m in index.documents.values() if m.source_url),
        "chunks_by_lane": dict(lanes),
        "dense_missing_chunks": dense_missing,
        "flagged_documents": flagged_docs,
        "superseded_documents": superseded,
        "supersessions": supersessions or [],
        "degradation_counts": dict(deg_by_kind),
        "degradations": index.degradations,
        "eval": eval_metrics,
    }
    if measure_latency and index.chunks:  # record the operating envelope (R9.1)
        report["latency"] = latency_probe(index, _sample_queries(index))
    return report


def latency_probe(index: "DocIndex", queries: list[str], k: int = 5) -> dict:
    """Measure search latency over sample queries (R9.1) so the operating envelope
    is recorded, not assumed. Returns p50/p95/max ms + corpus size."""
    import time

    if not queries:
        return {"queries": 0, "chunks": len(index.chunks)}
    times: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        index.search(q, k=k)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()

    def pct(f: float) -> float:
        return times[min(len(times) - 1, int(f * len(times)))]

    return {
        "queries": len(times), "chunks": len(index.chunks), "k": k,
        "p50_ms": round(pct(0.5), 1), "p95_ms": round(pct(0.95), 1),
        "max_ms": round(times[-1], 1),
    }


def render_build_report(report: dict) -> str:
    """Human-readable rendering of :func:`build_report`."""
    lines = [
        "parsevault build report",
        "=" * 40,
        f"Documents          : {report['documents']}",
        f"Chunks             : {report['chunks']}",
        f"Docs w/ provenance : {report['pages_with_provenance']}",
        f"Chunks by lane     : {report['chunks_by_lane'] or '—'}",
    ]
    miss = report["dense_missing_chunks"]
    lines.append(f"Dense-missing      : {len(miss)} chunk(s) lexical-only"
                 + (f" (e.g. {miss[0]})" if miss else ""))
    lines.append(f"Flagged documents  : {len(report.get('flagged_documents', []))}")
    lines.append(f"Superseded docs    : {len(report.get('superseded_documents', []))}")
    deg = report["degradation_counts"]
    if deg:
        lines.append("Degradations       :")
        lines += [f"  - {k}: {v}" for k, v in deg.items()]
    else:
        lines.append("Degradations       : none — no silent quality loss")
    if lat := report.get("latency"):
        if lat.get("queries"):
            envelope = " ✓ within 300ms target" if lat["p95_ms"] <= 300 else " ⚠ over 300ms target"
            lines.append(f"Latency            : p50 {lat['p50_ms']}ms · p95 {lat['p95_ms']}ms "
                         f"({lat['queries']} queries, {lat['chunks']} chunks){envelope}")
    if report.get("eval"):
        lines.append(f"Eval               : {report['eval']}")
    return "\n".join(lines)


def write_build_report(index: "DocIndex", out_dir: str | Path, *,
                       eval_metrics: dict | None = None, stamp: str = "",
                       supersessions: list[dict] | None = None) -> Path:
    """Write ``build-report-{stamp}.json`` + ``.txt`` to ``out_dir``; return the
    JSON path. ``stamp`` should be an ISO-ish timestamp from the caller."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(index, eval_metrics=eval_metrics, supersessions=supersessions)
    suffix = f"-{stamp}" if stamp else ""
    json_path = out_dir / f"build-report{suffix}.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"build-report{suffix}.txt").write_text(render_build_report(report), encoding="utf-8")
    return json_path


# --------------------------------------------------------------------------- #
# folder ingest
# --------------------------------------------------------------------------- #
# Single source of truth for ingestable suffixes (R10.1) — imported by the
# dashboard so its uploads accept exactly what the pipeline ingests.
_DOC_SUFFIXES = {
    ".pdf", ".docx", ".txt", ".text", ".md", ".markdown",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp",
}


def _looks_like_outputs_dir(folder: Path) -> bool:
    """M14: heuristic for "this folder is a converted-Markdown OUTPUT, not a
    source folder". Returns True when a folder appears to be a build-output
    layout (sharded ``outputs/<sha[:2]>/<sha>.md`` or flat
    ``outputs/{doc_id}.md`` siblings), so ``build_index_from_folder`` can
    refuse to re-ingest its own outputs.

    Conservative — only the obvious cases trigger:
    * the folder is literally named ``outputs`` or sits next to a
      ``docindex.json`` (a workspace root's outputs/);
    * OR every regular file is ``.md``/``.markdown`` AND the basenames look
      like 16-hex doc_ids (the flat layout).
    """
    if folder.name == "outputs":
        return True
    if (folder.parent / "docindex.json").is_file() and folder.name == "outputs":
        return True
    try:
        files = [p for p in folder.iterdir() if p.is_file()]
    except OSError:
        return False
    if not files:
        return False
    md_only = all(p.suffix.lower() in (".md", ".markdown") for p in files)
    if not md_only:
        return False
    hex_ids = sum(
        1 for p in files
        if len(p.stem) == 16 and all(c in "0123456789abcdef" for c in p.stem)
    )
    return hex_ids >= max(2, len(files) // 2)


def build_index_from_folder(
    folder: str | Path,
    *,
    config=None,
    index: DocIndex | None = None,
    on_doc=None,
    embedder=None,
    outputs_dir: str | Path | None = None,
    provenance_map: dict[str, dict] | None = None,
    archive_sources: bool = False,
    allow_outputs_dir: bool = False,
) -> DocIndex:
    """Convert every document in ``folder`` and add it to a ``DocIndex``.

    ``on_doc(meta)`` is called after each document (for progress reporting).
    Pass an existing ``index`` to add to it incrementally. Pass an ``embedder``
    to also build the dense semantic lane (fused with the lexical hybrid at
    query time); omit it for lexical-only retrieval. Pass ``outputs_dir`` to
    preserve each document's converted Markdown as ``{doc_id}.md``.

    M14: ``_DOC_SUFFIXES`` includes ``.md``/``.markdown`` because plain
    Markdown is a first-class source; that means if a caller mistakenly
    points this function at a workspace's ``outputs/`` (which is full of
    converted ``{doc_id}.md`` shards), the build would re-ingest its own
    converted output — yielding doc_id mismatches and inflated counts.
    Refuse when the folder smells like a build output, unless
    ``allow_outputs_dir=True`` (an explicit override for the rare case where
    a user really does want to index a pile of plain-Markdown files).
    """
    idx = index or DocIndex(embedder=embedder)
    folder = Path(folder)
    if not allow_outputs_dir and _looks_like_outputs_dir(folder):
        raise ValueError(
            f"{folder} looks like a build-output directory (Markdown shards "
            "with 16-hex doc_ids, or literally named 'outputs'). Refusing to "
            "re-ingest converted output. Pass allow_outputs_dir=True to "
            "override (e.g. when indexing a hand-written Markdown collection)."
        )
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in _DOC_SUFFIXES:
            continue
        prov = (provenance_map or {}).get(path.name)
        meta, chunks = build_document_metadata(
            path, config=config, outputs_dir=outputs_dir, provenance=prov,
            degradation_sink=idx.degradations, archive_sources=archive_sources,
        )
        idx.add(meta, chunks)
        if on_doc:
            on_doc(meta)
    mark_supersessions(idx)  # flag older same-form revisions as superseded (R8.2)
    return idx
