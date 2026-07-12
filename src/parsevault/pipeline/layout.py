"""Workspace layout helpers — flat vs sharded output resolution.

Two on-disk layouts coexist across the live indexes (F-3 origin):

* **Flat** — ``outputs/<doc_id>.md`` (16-hex doc_id). Used by the public KBs
  (nc-child-welfare, legal-authorities) and the standard build path
  (``DocIndex.build`` → ``build_document_metadata``).
* **Sharded** — ``outputs/<sha[:2]>/<sha>.md`` (full 64-hex source-sha
  filename). Used by the conversion pipeline for two-column scanned
  corpus to avoid 6,000+ files in one directory.

Several downstream consumers (rechunk driver, regex retriever, single-
workspace dashboard) historically only honored the flat layout, so the
sharded workspaces silently produced empty/zero results. This module gives
all of them one shared resolver so future consumers stay consistent.

Functions take an ``outputs`` directory (the ``outputs/`` root) plus
identifying info; layout is detected at read-time.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "resolve_md_path",
    "iter_output_md",
    "doc_id_for_md_path",
]


def resolve_md_path(
    outputs: Path,
    *,
    doc_id: str = "",
    source_sha256: str = "",
    md_path_hint: str = "",
) -> Path | None:
    """Return the on-disk ``.md`` path for a document, or ``None`` if absent.

    Resolution order (mirrors ``event_extraction._md_path_for_meta``):

      1. ``md_path_hint`` — explicit, recorded by the writer. Tried as-is
         and as outputs-relative.
      2. Sharded layout — ``outputs/<sha[:2]>/<source_sha256>.md`` when
         ``source_sha256`` is supplied.
      3. Flat layout — ``outputs/<doc_id>.md``.

    Returns ``None`` when nothing on disk is found so callers can branch.
    """
    if md_path_hint:
        candidate = Path(md_path_hint)
        if candidate.is_file():
            return candidate
        # Treat as relative to outputs or its parent.
        rel = outputs / candidate.name
        if rel.is_file():
            return rel
    if source_sha256:
        sharded = outputs / source_sha256[:2] / f"{source_sha256}.md"
        if sharded.is_file():
            return sharded
    if doc_id:
        flat = outputs / f"{doc_id}.md"
        if flat.is_file():
            return flat
    return None


def iter_output_md(outputs: Path):
    """Yield every ``.md`` file under ``outputs/`` regardless of layout.

    Walks BOTH ``outputs/*.md`` (flat) AND ``outputs/<2hex>/*.md`` (sharded)
    so consumers like ``RegexRetriever`` and ``rechunk`` cover both corpora
    with one loop. Yields ``Path`` objects in deterministic (sorted) order
    so test fixtures are stable.
    """
    if not outputs.is_dir():
        return
    # Flat files directly under outputs/.
    flat_seen = set()
    for p in sorted(outputs.glob("*.md")):
        if p.is_file():
            flat_seen.add(p.resolve())
            yield p
    # Sharded files under outputs/<2-hex>/*.md. Restrict to two-char dirs
    # that are valid hex shards so we don't accidentally walk other
    # workspace sub-trees (`outputs-vlm`, etc., are siblings — safe — but
    # being explicit costs nothing).
    for shard in sorted(outputs.iterdir()):
        if not shard.is_dir():
            continue
        if len(shard.name) != 2 or not all(c in "0123456789abcdef" for c in shard.name.lower()):
            continue
        for p in sorted(shard.glob("*.md")):
            if p.is_file() and p.resolve() not in flat_seen:
                yield p


def doc_id_for_md_path(path: Path) -> str:
    """Best-effort doc_id from an outputs/.md path.

    * Flat ``outputs/<doc_id>.md`` → ``<doc_id>`` (the stem).
    * Sharded ``outputs/<2hex>/<sha>.md`` → ``<sha>[:16]`` (the doc_id
      derivation used by ``parse_document_metadata`` is ``source_sha[:16]``;
      the sharded filename is the full source sha).

    For the sharded layout the parent directory's name is the two-char
    shard prefix; we mirror the index's identity by taking the first 16
    hex characters of the stem. Returns the bare stem when the layout
    can't be classified — the caller is responsible for index lookup.
    """
    stem = path.stem
    parent_name = path.parent.name
    if (
        len(parent_name) == 2
        and all(c in "0123456789abcdef" for c in parent_name.lower())
        and len(stem) == 64
        and stem.startswith(parent_name)
    ):
        return stem[:16]
    return stem
