"""Backfill page-raster PNGs for an already-built corpus, for the build
dashboard's hover-preview feature (feature/jimmy-page-screenshot-preview).

build_kb.py is resumable — it skips any doc already in docindex.json — so a
plain re-run does nothing for a corpus that's already fully built; nothing
would ever get (re-)extracted, so no rasters would ever be archived. This
script instead re-runs extraction ONLY, via build_document_metadata() with
outputs_dir=None and the returned (meta, chunks) simply discarded — the
existing docindex.json, chunks, and dense embeddings are never touched.
The only two effects are:
  1. Page-raster PNGs saved to <corpus>/rasters/ (OCR/VLM-lane pages only),
     via the same CascadeConfig.raster_archive_* mechanism build_kb.py uses.
  2. A fresh extract_done build_log event per doc, under a new build_id, so
     the dashboard has cards to show the rasters on.

Config via env: KB_SRC (source dir, default knowledge-base/nc-child-welfare/
sources), KB_OUT (workspace, default knowledge-base/nc-child-welfare).

Resumable across interrupted runs: skips a doc if <corpus>/rasters/ already
has any {stem}-p*.png for it.
"""
import datetime
import os as _os
import sys
import time
import traceback
from pathlib import Path

if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
    print(__doc__)
    raise SystemExit(0)

from parsevault.config import cascade_config_from_env
from parsevault.pipeline.build_events import BuildEventLogger
from parsevault.pipeline.docindex import build_document_metadata

SRC = Path(_os.environ.get("KB_SRC", "knowledge-base/nc-child-welfare/sources"))
KB = Path(_os.environ.get("KB_OUT", "knowledge-base/nc-child-welfare"))
RASTERS = KB / "rasters"
SUFFIXES = {".pdf", ".docx"}

if not SRC.is_dir():
    raise SystemExit(f"source dir not found: {SRC} (set KB_SRC to the folder of PDFs/DOCX)")

cfg = cascade_config_from_env()
cfg.raster_archive_dir = _os.environ.get("OCR_RASTER_ARCHIVE_DIR") or str(RASTERS)
cfg.raster_archive_mode = _os.environ.get("OCR_RASTER_ARCHIVE_MODE") or "all"

all_docs = sorted(p for p in SRC.iterdir() if p.suffix.lower() in SUFFIXES)
already = {f.stem.rsplit("-p", 1)[0] for f in RASTERS.glob("*.png")} if RASTERS.is_dir() else set()
todo = [p for p in all_docs if p.stem not in already]
print(f"page-raster backfill: {len(all_docs)} docs, {len(already)} already have rasters, "
      f"{len(todo)} to do", flush=True)

build_id = f"{KB.name}-rasters-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
logger = BuildEventLogger(build_id)
logger.event("build_start", total_todo=len(todo), already_done=len(already), provider="local-cascade")

t0 = time.time()
for i, path in enumerate(todo, 1):
    logger.event("doc_start", index=i, total=len(todo), path=str(path))
    try:
        build_document_metadata(path, config=cfg, event_sink=logger.event)
        print(f"  [{i}/{len(todo)}] {path.name[:60]:62} ({time.time()-t0:.0f}s elapsed)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  [{i}/{len(todo)}] {path.name}: ERROR {e}", flush=True)
        traceback.print_exc()
        logger.event("doc_error", path=str(path), error=str(e))

logger.event("build_done", docs=len(todo), chunks=0, elapsed_s=time.time() - t0, report_path="")
logger.close()
print(f"\nDONE: {len(todo)} docs re-extracted in {time.time()-t0:.0f}s -> {RASTERS}", flush=True)
print(f"build_id -> {build_id} (pick it in the dashboard's build selector)", flush=True)
