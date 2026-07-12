"""Build the public NC child-welfare KB: LlamaParse + facets + dense + provenance."""
import datetime
import os as _os
import sys
import time
import traceback
from pathlib import Path

if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
    print(__doc__)
    print("Config via env: KB_SRC (source dir, default tests/Test-Docs/NC-Forms), "
          "KB_OUT (workspace, default knowledge-base/nc-child-welfare), "
          "PARSE_PROVIDER / LLAMAPARSE_API_KEY for the cloud converter. "
          "Resumable: skips docs already in the index.")
    raise SystemExit(0)

from parsevault.config import cascade_config_from_env, embedder_from_env
from parsevault.pipeline.docindex import (
    DocIndex,
    apply_provenance,
    build_document_metadata,
    load_provenance_manifest,
)
from parsevault.pipeline.extractors.llamaparse_extractor import LlamaParseCreditsExhausted

SRC = Path(_os.environ.get("KB_SRC", "tests/Test-Docs/NC-Forms"))
KB = Path(_os.environ.get("KB_OUT", "knowledge-base/nc-child-welfare"))
OUT = KB / "outputs"
IDX = KB / "docindex.json"
MANIFEST = Path("knowledge-base/SOURCE_INDEX.json")
SUFFIXES = {".pdf", ".docx"}

cfg = cascade_config_from_env()
emb = embedder_from_env()
pmap = load_provenance_manifest(MANIFEST, retrieved_at="") if MANIFEST.exists() else {}


def prov_for(path: Path) -> dict | None:
    """Per-document provenance: manifest entry + the file's download date (mtime)."""
    p = dict(pmap.get(path.name) or {})
    if p and not p.get("retrieved_at") and path.exists():
        p["retrieved_at"] = datetime.date.fromtimestamp(path.stat().st_mtime).isoformat()
    return p or None


def stamp_all(index: DocIndex) -> None:
    """Backfill provenance + fix generic titles on every document (covers resumes)."""
    from parsevault.pipeline.docindex import is_generic_title, title_from_path

    for m in index.documents.values():
        if not m.source_url:
            apply_provenance(m, prov_for(Path(m.source_path)))
        if is_generic_title(m.title):  # shared title-fallback (R10.4)
            m.title = title_from_path(m.source_path)


if not SRC.is_dir():
    raise SystemExit(f"source dir not found: {SRC} (set KB_SRC to the folder of PDFs/DOCX)")

idx = DocIndex.load(IDX, embedder=emb) if IDX.is_file() else DocIndex(embedder=emb)
stamp_all(idx)  # backfill provenance on already-built docs
done = {m.source_path for m in idx.documents.values()}
todo = [p for p in sorted(SRC.iterdir()) if p.suffix.lower() in SUFFIXES and str(p) not in done]
print(f"KB build: {len(done)} done, {len(todo)} to do, provider={cfg.parse_provider}, "
      f"dense={'on' if emb else 'off'}, provenance={'on' if pmap else 'off'}", flush=True)

t0 = time.time()
for i, path in enumerate(todo, 1):
    try:
        meta, chunks = build_document_metadata(
            path, config=cfg, outputs_dir=OUT, provenance=prov_for(path),
            degradation_sink=idx.degradations,
        )
        idx.add(meta, chunks)
        print(f"  [{i}/{len(todo)}] {path.name[:46]:48} {meta.page_count:>3}p "
              f"[{','.join(sorted(set(meta.page_routes)))}] «{meta.category}/{meta.topic}»",
              flush=True)
    except LlamaParseCreditsExhausted as e:
        print(f"\n!!! LlamaParse credits exhausted at {path.name}: {e}", flush=True)
        break
    except Exception as e:  # noqa: BLE001
        print(f"  [{i}/{len(todo)}] {path.name}: ERROR {e}", flush=True)
        traceback.print_exc()
    if i % 10 == 0:
        idx.save(IDX)
        print(f"  …saved ({len(idx.documents)} docs, {time.time()-t0:.0f}s)", flush=True)

stamp_all(idx)
from parsevault.pipeline.docindex import mark_supersessions, write_build_report

supersessions = mark_supersessions(idx)
idx.save(IDX)
report_path = write_build_report(
    idx, KB, stamp=datetime.datetime.now().strftime("%Y%m%dT%H%M%S"),
    supersessions=supersessions)
stamped = sum(1 for m in idx.documents.values() if m.source_url)
print(f"\nDONE: {len(idx.documents)} docs ({stamped} with provenance), "
      f"{len(idx.chunks)} chunks -> {IDX} ({time.time()-t0:.0f}s)", flush=True)
print(f"build report -> {report_path}", flush=True)
