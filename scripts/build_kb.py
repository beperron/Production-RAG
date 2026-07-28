"""Build the public NC child-welfare KB: LlamaParse + facets + dense + provenance."""
import datetime
import os as _os
import sys
from pathlib import Path

if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
    print(__doc__)
    print("Config via env: KB_SRC (source dir, default tests/Test-Docs/NC-Forms), "
          "KB_OUT (workspace, default knowledge-base/nc-child-welfare), "
          "PARSE_PROVIDER / LLAMAPARSE_API_KEY for the cloud converter. "
          "Resumable: skips docs already in the index.")
    raise SystemExit(0)

from parsevault.config import cascade_config_from_env, embedder_from_env
from parsevault.pipeline.build_runner import run_corpus_build
from parsevault.pipeline.docindex import load_provenance_manifest

SRC = Path(_os.environ.get("KB_SRC", "tests/Test-Docs/NC-Forms"))
KB = Path(_os.environ.get("KB_OUT", "knowledge-base/nc-child-welfare"))
MANIFEST = Path("knowledge-base/SOURCE_INDEX.json")
SUFFIXES = {".pdf", ".docx"}

cfg = cascade_config_from_env()
# Page-raster archival, on by default for this build: lets the dashboard show a
# hover-preview of the exact image an OCR/VLM lane saw, for every OCR/VLM page
# (not just flagged ones) — a reviewer's "does this look garbled?" check, not
# just the dispute-audit use case raster_archive_dir was originally built for.
# Still overridable via the existing env vars for that original purpose.
cfg.raster_archive_dir = _os.environ.get("OCR_RASTER_ARCHIVE_DIR") or str(KB / "rasters")
cfg.raster_archive_mode = _os.environ.get("OCR_RASTER_ARCHIVE_MODE") or "all"
emb = embedder_from_env()
pmap = load_provenance_manifest(MANIFEST, retrieved_at="") if MANIFEST.exists() else {}


def prov_for(path: Path) -> dict | None:
    """Per-document provenance: manifest entry + the file's download date (mtime)."""
    p = dict(pmap.get(path.name) or {})
    if p and not p.get("retrieved_at") and path.exists():
        p["retrieved_at"] = datetime.date.fromtimestamp(path.stat().st_mtime).isoformat()
    return p or None


if not SRC.is_dir():
    raise SystemExit(f"source dir not found: {SRC} (set KB_SRC to the folder of PDFs/DOCX)")

run_corpus_build(
    kb_name="KB build", src=SRC, kb=KB, suffixes=SUFFIXES, cfg=cfg, emb=emb,
    provenance_fn=prov_for, label_fn=lambda m: f"«{m.category}/{m.topic}»",
)
