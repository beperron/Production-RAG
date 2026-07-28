"""Shared build-loop plumbing for the per-corpus build scripts (build_kb.py,
build_legal_kb.py). Owns the parts that are identical across corpora --
event logging, checkpointing, the pre/post-loop provenance backfill, and the
finalize sequence -- so the dashboard's event contract only has one place
that can drift out of sync. Each script keeps its own corpus-specific logic
(provenance derivation, category/topic vs. section labeling, suffix set) and
passes it in as callbacks.
"""
from __future__ import annotations

import datetime
import time
import traceback
from pathlib import Path
from typing import Callable

from .build_events import BuildEventLogger
from .docindex import (
    DocIndex,
    DocMetadata,
    apply_provenance,
    build_document_metadata,
    is_generic_title,
    mark_supersessions,
    title_from_path,
    write_build_report,
)
from .extractors.llamaparse_extractor import LlamaParseCreditsExhausted

ProvenanceFn = Callable[[Path], "dict | None"]
LabelFn = Callable[[DocMetadata], str]


def _stamp_all(index: DocIndex, provenance_fn: ProvenanceFn) -> None:
    """Backfill provenance + fix generic titles on every document (covers resumes)."""
    for m in index.documents.values():
        if not m.source_url:
            apply_provenance(m, provenance_fn(Path(m.source_path)))
        if is_generic_title(m.title):  # shared title-fallback (R10.4)
            m.title = title_from_path(m.source_path)


def run_corpus_build(
    *,
    kb_name: str,
    src: Path,
    kb: Path,
    suffixes: set[str],
    cfg,
    emb,
    provenance_fn: ProvenanceFn,
    label_fn: LabelFn,
    checkpoint_every: int = 10,
) -> DocIndex:
    """Build (or resume) one corpus end to end, emitting the dashboard's
    build_log event contract (build_start/doc_start/index_done/doc_error/
    save_checkpoint/build_done) identically regardless of corpus.
    """
    out = kb / "outputs"
    idx_path = kb / "docindex.json"

    idx = DocIndex.load(idx_path, embedder=emb) if idx_path.is_file() else DocIndex(embedder=emb)
    _stamp_all(idx, provenance_fn)  # backfill provenance on already-built docs
    done = {m.source_path for m in idx.documents.values()}
    todo = [p for p in sorted(src.iterdir()) if p.suffix.lower() in suffixes and str(p) not in done]
    print(f"{kb_name}: {len(done)} done, {len(todo)} to do, provider={cfg.parse_provider}, "
          f"dense={'on' if emb else 'off'}", flush=True)

    build_id = f"{kb.name}-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    logger = BuildEventLogger(build_id)
    logger.event("build_start", total_todo=len(todo), already_done=len(done),
                 provider=cfg.parse_provider, dense_enabled=emb is not None)

    t0 = time.time()
    for i, path in enumerate(todo, 1):
        logger.event("doc_start", index=i, total=len(todo), path=str(path))
        try:
            meta, chunks = build_document_metadata(
                path, config=cfg, outputs_dir=out, provenance=provenance_fn(path),
                degradation_sink=idx.degradations, event_sink=logger.event,
            )
            idx.add(meta, chunks)
            dense_missing = (
                len(set(meta.chunk_ids) & idx._dense.missing) if idx._dense is not None else 0
            )
            logger.event("index_done", doc_id=meta.doc_id, chunk_count=len(chunks),
                         dense_embedded=len(chunks) - dense_missing, dense_missing=dense_missing)
            print(f"  [{i}/{len(todo)}] {path.name[:46]:48} {meta.page_count:>3}p "
                  f"[{','.join(sorted(set(meta.page_routes)))}] {label_fn(meta)}", flush=True)
        except LlamaParseCreditsExhausted as e:
            print(f"\n!!! LlamaParse credits exhausted at {path.name}: {e}", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(todo)}] {path.name}: ERROR {e}", flush=True)
            traceback.print_exc()
            logger.event("doc_error", path=str(path), error=str(e))
        if i % checkpoint_every == 0:
            idx.save(idx_path)
            print(f"  …saved ({len(idx.documents)} docs, {time.time()-t0:.0f}s)", flush=True)
            logger.event("save_checkpoint", docs_saved=len(idx.documents), elapsed_s=time.time() - t0)

    _stamp_all(idx, provenance_fn)
    supersessions = mark_supersessions(idx)
    idx.save(idx_path)
    report_path = write_build_report(
        idx, kb, stamp=datetime.datetime.now().strftime("%Y%m%dT%H%M%S"),
        supersessions=supersessions)
    stamped = sum(1 for m in idx.documents.values() if m.source_url)
    print(f"\nDONE: {len(idx.documents)} docs ({stamped} with provenance), "
          f"{len(idx.chunks)} chunks -> {idx_path} ({time.time()-t0:.0f}s)", flush=True)
    print(f"build report -> {report_path}", flush=True)
    logger.event("build_done", docs=len(idx.documents), chunks=len(idx.chunks), stamped=stamped,
                 elapsed_s=time.time() - t0, report_path=str(report_path))
    logger.close()
    return idx
