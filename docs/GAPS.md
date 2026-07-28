# Known gaps

Tracked issues found by reading the code, not (yet) filed as tickets. Each
entry names the exact files/lines involved so a fix can be verified against
this description going forward.

## 1. Dense-lane degradation is computed but not surfaced to the caller

**What works:** the retrieval engine already has real "no silent degradation"
machinery for the dense (semantic embedding) lane.

- Per-query, `DocIndex.search()` checks whether the dense lane is materially
  degraded — more than 5% of chunks in the corpus have no embedding vector —
  and builds a note:
  [`src/parsevault/pipeline/docindex.py:1308-1316`](../src/parsevault/pipeline/docindex.py#L1308-L1316)
  ```python
  if self._dense is not None and self._dense.missing and self.chunks:
      ratio = len(self._dense.missing) / max(1, len(self.chunks))
      if ratio > 0.05:
          dense_degraded_note = (
              f"dense lane degraded: {len(self._dense.missing)} of "
              f"{len(self.chunks)} chunks are lexical-only "
              f"({ratio:.0%}); re-embed to restore semantic ranking"
          )
  ```
  This note is attached to every `SearchHit.notes` for that query
  ([docindex.py:1386-1387](../src/parsevault/pipeline/docindex.py#L1386-L1387)).
- Per-hit, provenance is tracked via `SearchHit.lane` (`"bm25"` / `"dense"` /
  `"hybrid"` / `""`) so you can tell after the fact which signal(s) surfaced a
  given result
  ([docindex.py:935-941](../src/parsevault/pipeline/docindex.py#L935-L941)).
- Build-time and audit-time, the full picture is durable: `record_degradation()`
  appends to a persistent ledger saved in the index sidecar
  ([docindex.py:1193-1206](../src/parsevault/pipeline/docindex.py#L1193-L1206)),
  and `build_report()` summarizes it (`dense_missing_chunks`,
  `degradation_counts`, etc. —
  [docindex.py:2338-2361](../src/parsevault/pipeline/docindex.py#L2338-L2361)).

**Where it breaks down:** `LawSearch._to_hit()`, the public-facing wrapper
both `scripts/query.py` and `scripts/law_search_server.py` consume, copies
`lane` onto the `Hit` it returns but never copies `notes`:
[`src/parsevault/lawsearch.py:208-226`](../src/parsevault/lawsearch.py#L208-L226).
The `dense_degraded_note` computed at query time is silently dropped before
it reaches either consumer.

- [`scripts/law_search_server.py:130-131`](../scripts/law_search_server.py#L130-L131)
  renders a `lane` badge per result (bm25/dense/hybrid) — a hit with no dense
  contribution is visible if you know to look for a missing "dense"/"hybrid"
  badge — but there is no explicit "dense lane degraded" banner anywhere on
  the page.
- [`scripts/query.py`](../scripts/query.py) prints only `section/title`,
  `citation`, `snippet` — no `lane`, no `notes`, at all.

**Also:** if the embedder is entirely absent — `EMBED_MODEL=none`, or it fails
to load and `strict_mode()` is off
([`src/parsevault/config.py:194-199`](../src/parsevault/config.py#L194-L199))
— `self._dense` is `None` from the start. In that case there is no "degraded"
note at all: the system runs lexical-only from the first query, and while it
logs an error server-side, nothing reaches the query response to tell the
user semantic ranking isn't running.

**Net effect:** the "no silent degradation" invariant this codebase clearly
cares about (see the `R4`/`M-6` markers in code comments) holds up to the
edge of `DocIndex`, then is dropped exactly at the boundary a live user (CLI
or web UI) actually observes.

**Possible fix (not yet done):** add `notes: list[str]` to `Hit` in
`lawsearch.py`, populate it in `_to_hit()` from `getattr(h, "notes", [])`, and
render it in both `query.py` (a printed line per result set) and
`law_search_server.py` (a banner, not just a badge — badges are easy to miss
and say nothing when the dense lane is *entirely* absent rather than merely
degraded).

---

*Add new entries above this line as they're found. Remove an entry once its
fix lands — this file is a gap tracker, not a changelog.*
