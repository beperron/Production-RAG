# Build-time dashboard — design spec

**Status:** implemented, then revised. This doc is written to be handed
to an implementer as a self-contained spec — it names exact files, functions,
and line numbers as of the commit this was written against, plus the event
schema and UI layout. Re-verify line numbers before editing; the repo moves.

**2026-07-28 update:** the JSONL event log described below (§ Architecture,
§2, §4) has been replaced. `BuildEventLogger` now writes each event straight
into the hash-chained `build_log` table in the local Postgres store
(`postgres/init/001_schema.sql`) instead of `build_events.jsonl` — no
intermediate file, and the dashboard reads from Postgres instead of
scanning for JSONL files. The event *schema* (§1: `stage` + fields per
event) and the two-process decoupled architecture (build vs. dashboard,
now decoupled by Postgres rather than a file) are unchanged; the "No new
pip dependencies" non-goal in §Goal no longer holds — `psycopg[binary]` is
now a `build`-extra dependency. See `src/parsevault/pipeline/build_events.py`
and `scripts/build_dashboard_server.py` for the current implementation.

**2026-07-28 update (page-preview):** each page badge in the per-page strip
(§4 Panel layout) now supports hover-to-preview and click-to-open-source.
`PageResult.raster_path` (`extractors/base.py`) carries the archived-raster
PNG path for OCR/VLM-lane pages (set by `_archive_rasters` in
`extractors/router.py`); `build_kb.py` turns raster archival on by default
(`raster_archive_mode="all"`, into `<corpus>/rasters/`, sibling to
`outputs/`/`sources/`) instead of it being an opt-in dispute-review-only
setting. The `extract_done` event's per-page dict carries `raster_path`
(`null` for pages with none — native-text pages never get one, by design:
only pages that were already rasterized during extraction get a preview).
Two new dashboard routes serve these: `GET /api/raster?path=<png>` and
`GET /api/source?path=<pdf>` (inline, for a `#page=N` fragment to work),
both checked against a configurable allowlist (`BUILD_DASHBOARD_SERVE_ROOTS`,
default `knowledge-base:tests/Test-Docs`) since the dashboard otherwise has
no static-file serving at all. This makes hover/click-through
filesystem-local — it only works when the dashboard process can see the
same disk the build wrote to, which narrows the "point the dashboard at a
build on another machine over Postgres" architecture above, but that's
inherent to "open the actual file on disk" and matches the reviewer
workflow this was built for.

## Goal

`scripts/build_kb.py` / `scripts/build_legal_kb.py` already print one line per
document as they convert a source folder into a knowledge base. That's not
enough for the level of visibility wanted here: per-*page* extraction lane
and quality flags, per-*chunk* boundaries and sizes, and per-document dense-
embedding outcome, live, in a browser, while a build runs — a "pedantic demo"
of the pipeline, not just a progress bar.

Non-goals for this first version (keep scope contained):
- No new pip dependencies (project only depends on `requests`,
  `sentence-transformers`, `numpy` — see `pyproject.toml`). The dashboard
  server must use stdlib `http.server`, matching `scripts/law_search_server.py`.
- No WebSockets/SSE. Simple polling is plenty at this data volume (single-box
  build, a few thousand pages at most).
- No auth. Localhost-only, same posture as the existing search server.
- No changes to query-time code. (There's a related, separate gap — dense-
  lane degradation notes not reaching query results — written up in
  `docs/GAPS.md`. Don't conflate the two; this doc is build-time only.)

## Architecture

Two independent processes, decoupled by a JSONL event log on disk:

```
Terminal 1: python scripts/build_kb.py
              → writes structured events to
                knowledge-base/<collection>/build_events.jsonl (append, flush per line)
              → keeps its existing stdout prints unchanged

Terminal 2: python scripts/build_dashboard_server.py --events knowledge-base/nc-child-welfare/build_events.jsonl
              → serves http://127.0.0.1:8901
              → polls/re-reads the JSONL file (~1s) and renders it

Browser:    http://127.0.0.1:8901   (auto-refreshing panel)
```

Why decoupled rather than the build script serving its own dashboard: it lets
you point the dashboard at a build running on `orion` from a browser on
`vega` (or wherever), and it lets you re-open a *finished* build's event log
later to review it — same replay affordance the existing `build_report.json`
already gives you, just at finer granularity and incrementally.

## 1. Event schema

One JSON object per line, newline-delimited, written append-only. Every event
has `ts` (ISO 8601, local tz) and `stage`. Fields beyond that are stage-specific.

```jsonc
{"ts": "...", "stage": "build_start", "total_todo": 42, "already_done": 108, "provider": "local", "dense_enabled": true}

{"ts": "...", "stage": "doc_start", "index": 1, "total": 42, "path": "NC-Forms/DSS-5231.pdf"}

{"ts": "...", "stage": "extract_done", "doc_id": "...", "path": "...", "extractor": "tesseract",
 "page_count": 6,
 "pages": [
   {"page_number": 1, "route": "native", "chars": 1820, "flagged": false, "flag_reasons": []},
   {"page_number": 2, "route": "tesseract", "chars": 940, "flagged": true,
    "flag_reasons": ["low_ocr_confidence(52<65)"]}
 ],
 "degradations": [{"kind": "vlm_fallback_tesseract", "detail": "..."}]}

{"ts": "...", "stage": "chunk_done", "doc_id": "...",
 "chunk_count": 9,
 "chunks": [
   {"chunk_id": "...:0", "heading_path": ["Section 3", "Eligibility"], "char_len": 1210,
    "token_estimate": 302, "page_start": 1, "page_end": 2, "extraction_routes": ["native"]}
 ]}

{"ts": "...", "stage": "index_done", "doc_id": "...", "chunk_count": 9,
 "dense_embedded": 9, "dense_missing": 0}

{"ts": "...", "stage": "doc_error", "path": "...", "error": "..."}

{"ts": "...", "stage": "save_checkpoint", "docs_saved": 110, "elapsed_s": 340}

{"ts": "...", "stage": "build_done", "docs": 150, "chunks": 4820, "stamped": 148,
 "elapsed_s": 610, "report_path": "knowledge-base/nc-child-welfare/build_report_....json"}
```

`pages` and `chunks` arrays are per-document detail — that's the "pedantic"
part. Keep them capped defensively (e.g. don't emit more than a few hundred
page/chunk entries per event; this corpus won't hit that, but don't build in
an unbounded-payload footgun).

## 2. New module: `src/parsevault/pipeline/build_events.py`

A small, optional, side-effect-only event emitter — same shape as the
existing `degradation_sink` parameter pattern already used in
`build_document_metadata()` ([docindex.py:719](../src/parsevault/pipeline/docindex.py#L719)),
so it fits the codebase's existing conventions rather than inventing a new one.

```python
class BuildEventLogger:
    def __init__(self, path: str | Path):
        ...  # opens file in append mode, keeps handle open for the build's duration

    def event(self, stage: str, **fields) -> None:
        ...  # writes {"ts": <iso>, "stage": stage, **fields} + "\n", flush() immediately

    def close(self) -> None: ...
```

`flush()` after every write matters — the dashboard reads the file live, so a
buffered-but-unflushed line is invisible until the process exits.

## 3. Instrumentation points (minimal diffs)

**a) `build_document_metadata()`** — [docindex.py:716-793](../src/parsevault/pipeline/docindex.py#L716-L793)

Add an optional parameter, following the existing `degradation_sink` pattern:

```python
def build_document_metadata(
    source_path, *, config=None, max_chunk_chars=1500,
    outputs_dir=None, provenance=None, degradation_sink=None, archive_sources=False,
    event_sink=None,   # NEW: Callable[[str, dict], None] | None
) -> tuple[DocMetadata, list[Chunk]]:
```

Emit `extract_done` right after `result = _extract_with_provider(path, config)`
(around [docindex.py:741](../src/parsevault/pipeline/docindex.py#L741)) — the
per-page quality dicts are already sitting on `result.pages[i].quality`
(populated by `quality.py::score_pages`, see [quality.py:97-108](../src/parsevault/pipeline/quality.py#L97-L108)),
so this event is just a reshape of data that already exists, not new
computation.

Emit `chunk_done` right after `meta, chunks = parse_document_metadata(...)`
returns (around [docindex.py:743-752](../src/parsevault/pipeline/docindex.py#L743-L752)) —
`chunks` already carries `heading_path`, `char_len`, `token_estimate`,
`page_start`/`page_end`, `extraction_routes` per `Chunk` (see the dataclass
fields used in `chunk_markdown`, [docindex.py:381-393](../src/parsevault/pipeline/docindex.py#L381-L393)).
Again: no new computation, just emit what's already on the object.

Call `if event_sink: event_sink("extract_done", **payload)` /
`event_sink("chunk_done", **payload)` — mirror the existing
`if degradation_sink is not None: ...` guard style already in this function.

**b) `scripts/build_kb.py`** (and mirror in `build_legal_kb.py`)

- At the top, after the existing setup ([build_kb.py:60-65](../scripts/build_kb.py#L60-L65)):
  construct a `BuildEventLogger` pointed at `KB / "build_events.jsonl"`, emit
  `build_start`.
- Inside the loop ([build_kb.py:68-86](../scripts/build_kb.py#L68-L86)):
  - emit `doc_start` before the `try:`
  - pass `event_sink=logger.event` into `build_document_metadata(...)`
    ([build_kb.py:70-73](../scripts/build_kb.py#L70-L73))
  - **after** `idx.add(meta, chunks)` ([build_kb.py:74](../scripts/build_kb.py#L74)),
    emit `index_done`: `dense_missing` = `len(set(meta.chunk_ids) & idx._dense.missing)`
    if `idx._dense is not None` else `0`; `dense_embedded` = `chunk_count - dense_missing`.
    (`idx._dense.missing` is a `set[str]` of chunk ids with no usable vector —
    see [embeddings.py:319-344](../src/parsevault/pipeline/embeddings.py#L319-L344).)
  - on the `except Exception` branch ([build_kb.py:81-83](../scripts/build_kb.py#L81-L83)),
    also emit `doc_error`
  - on the periodic-save branch ([build_kb.py:84-86](../scripts/build_kb.py#L84-L86)),
    also emit `save_checkpoint`
- At the very end (after [build_kb.py:99](../scripts/build_kb.py#L99)), emit
  `build_done` with the same numbers already being printed to stdout, plus
  `report_path`.
- Keep every existing `print(...)` call exactly as-is. This is additive
  instrumentation, not a rewrite — if `BuildEventLogger` fails to open its
  file for any reason, the build must still run (log the error, don't raise;
  same "optional, degrades gracefully" posture as `degradation_sink`).

Apply the same instrumentation shape to `scripts/build_legal_kb.py` once the
pattern is proven in `build_kb.py` — don't do both at once; get one working
and reviewed first.

**2026-07-28 update (shared runner):** superseded. `build_legal_kb.py` had
drifted from this instrumentation for months (no `event_sink`, no raster
config — invisible to the dashboard, fixed on `bugfix/legal-kb-build-parity`).
The per-doc loop, event emission, checkpointing, and finalize sequence
described in this section now live in one place —
`src/parsevault/pipeline/build_runner.py`'s `run_corpus_build()` — which both
`build_kb.py` and `build_legal_kb.py` call. The event *contract* (§1) is
unchanged; it's just no longer possible for one script to silently miss it.
Only the genuinely corpus-specific pieces (provenance derivation, category/
topic vs. section labeling, suffix set) stay in the individual scripts.

## 4. New script: `scripts/build_dashboard_server.py`

Model directly on `scripts/law_search_server.py`'s structure: stdlib
`http.server.BaseHTTPRequestHandler` + `ThreadingHTTPServer`, no framework.
Reuse its `_CSS` block (the "paper / seal-red" theme) for visual consistency
rather than inventing new styling.

CLI: `python scripts/build_dashboard_server.py --events <path-to-jsonl> [--port 8901]`

Routes:
- `GET /` — the HTML shell + inline JS that polls `GET /api/events` on an
  interval (~1s) and re-renders.
- `GET /api/events` — reads the JSONL file fresh each call (fine at this
  scale; don't add incremental-offset complexity for a first version),
  returns `{"events": [...]}` as JSON.

### Panel layout

- **Header**: progress bar (`docs done / total` from the latest `doc_start`
  vs `build_start.total_todo`), elapsed time, current doc name, a big
  "degradation count" badge (red if > 0) sourced from counting `doc_error` +
  `extract_done[].degradations` + `index_done` entries with `dense_missing > 0`
  — this is the live version of the "no silent degradation" ledger the engine
  already keeps at build-report time ([docindex.py:2334-2361](../src/parsevault/pipeline/docindex.py#L2334-L2361)).
- **Document timeline** (newest first, or oldest-first with autoscroll — pick
  one, don't build both): one card per `doc_start`, populated in place as its
  `extract_done` / `chunk_done` / `index_done` / `doc_error` events arrive.
  Each card, expandable:
  - *Extraction*: extractor badge, page count, a per-page strip of colored
    badges by `route` (native/tesseract/vlm/llamaparse), flagged pages
    outlined and hoverable to show `flag_reasons`.
  - *Chunking*: chunk count, a compact table — heading path, char_len,
    page span, extraction_routes. This is the part that makes chunk
    boundaries actually inspectable instead of trusted blindly.
  - *Indexing*: `dense_embedded / chunk_count` — if `dense_missing > 0`,
    render it as a warning, not neutral text (this is the same signal the
    `docs/GAPS.md` gap describes as currently invisible at *query* time; here,
    at *build* time, we have a clean shot at surfacing it properly).
- **Footer**: appears once `build_done` arrives — final totals + a link/path
  to the full `build_report_*.json` for the deep-dive view.

## 5. Acceptance criteria

- Running `python scripts/build_kb.py` on a small test folder (a handful of
  PDFs) produces a `build_events.jsonl` alongside the existing `docindex.json`.
- Running `scripts/build_dashboard_server.py --events <that file>` while the
  build is still running shows documents appearing incrementally, each
  expandable to real per-page/per-chunk data — not just a spinner.
- Killing/restarting the dashboard server mid-build and reloading the page
  reconstructs the full state from the JSONL file (proves the design is
  actually stateless/file-driven, not relying on server memory that a crash
  would lose).
- `build_kb.py`'s existing stdout output and exit behavior are byte-for-byte
  unchanged if `build_events.jsonl` can't be written (e.g. read-only dir) —
  confirms the instrumentation is additive, not load-bearing.
- No new entries in `pyproject.toml` dependencies.
