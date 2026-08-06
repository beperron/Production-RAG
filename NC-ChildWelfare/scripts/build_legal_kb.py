"""Build the public legal-authorities KB: NC statutes/admin-code (PDF) + fetched
federal statutes/regulations/case law (.txt), with provenance.

Provenance sources, in order:
  * PDFs   → the SOURCE_INDEX.json manifest entry (by filename).
  * .txt   → the leading ``Source:``/``Wikisource:`` line embedded by the fetch.
  * else   → minimal (section guess + file mtime); logged so it's visible.

Extraction uses the local cascade by default (born-digital statutes have clean
text layers; .txt routes to the plain-text lane). PARSE_PROVIDER=llamaparse can
be set later (public data only) to re-convert scanned PDFs at higher fidelity.

Usage:
    python scripts/build_legal_kb.py            # extract + index
    python scripts/build_legal_kb.py --inspect  # extract to /tmp, report quality only
"""
import datetime
import re
import sys
import time
import traceback
from pathlib import Path
from urllib.parse import urlparse

from parsevault.config import cascade_config_from_env, embedder_from_env
from parsevault.pipeline.docindex import (
    DocIndex,
    build_document_metadata,
    load_provenance_manifest,
)

KB = Path("knowledge-base/legal-authorities")
SRC = KB / "sources"
OUT = KB / "outputs"
IDX = KB / "docindex.json"
MANIFEST = Path("knowledge-base/SOURCE_INDEX.json")
SUFFIXES = {".pdf", ".txt", ".docx"}

_SOURCE_RE = re.compile(r"^\s*source:\s*(\S+)", re.IGNORECASE)
_WIKISOURCE_RE = re.compile(r"^\s*wikisource:\s*(.+)$", re.IGNORECASE)


def _section_for(url: str) -> str:
    """Bucket a federal source URL into a human collection name."""
    if "/uscode/" in url:
        return "Federal Statutes (U.S. Code)"
    if "/cfr/" in url:
        return "Federal Regulations (CFR)"
    if "wikisource.org" in url or url.startswith("Wikisource"):
        return "U.S. Supreme Court Cases"
    return "Federal Legal Authorities"


def _txt_provenance(path: Path) -> dict:
    """Read a .txt file's embedded source line into a provenance dict."""
    retrieved = datetime.date.fromtimestamp(path.stat().st_mtime).isoformat()
    for line in path.read_text(errors="replace").splitlines()[:5]:
        if m := _SOURCE_RE.match(line):
            url = m.group(1)
            return {"source_url": url, "source_domain": urlparse(url).netloc,
                    "section": _section_for(url), "retrieved_at": retrieved}
        if m := _WIKISOURCE_RE.match(line):
            ref = m.group(1).strip()
            slug = ref.replace(" ", "_")
            return {"source_url": f"https://en.wikisource.org/wiki/{slug}",
                    "source_domain": "en.wikisource.org",
                    "section": "U.S. Supreme Court Cases", "retrieved_at": retrieved}
    # Case-law file with no embedded source line (e.g. "Troxel-v-Granville-2000"):
    # construct the canonical Wikisource opinion URL from the filename.
    if cm := re.match(r"(.+?-v-.+?)(?:-\d{4})?$", path.stem):
        slug = cm.group(1).replace("-v-", "_v._").replace("-", "_")
        return {"source_url": f"https://en.wikisource.org/wiki/{slug}",
                "source_domain": "en.wikisource.org",
                "section": "U.S. Supreme Court Cases", "retrieved_at": retrieved}
    return {"section": "U.S. Supreme Court Cases", "retrieved_at": retrieved}


def provenance_for(path: Path, pmap: dict) -> dict | None:
    """Per-document provenance: manifest entry, else embedded .txt source line."""
    p = dict(pmap.get(path.name) or {})
    if p:
        if not p.get("retrieved_at") and path.exists():
            p["retrieved_at"] = datetime.date.fromtimestamp(path.stat().st_mtime).isoformat()
        return p
    if path.suffix.lower() == ".txt":
        return _txt_provenance(path)
    # Unlisted PDF/DOCX: minimal provenance so it's still dated + flagged.
    print(f"  ! no manifest/source for {path.name}; minimal provenance", flush=True)
    return {"section": "NC Legal Authorities",
            "retrieved_at": datetime.date.fromtimestamp(path.stat().st_mtime).isoformat()}


def _fix_title(m) -> None:
    # Shared title-fallback (R10.4) — single definition lives in docindex.
    from parsevault.pipeline.docindex import is_generic_title, title_from_path

    if is_generic_title(m.title):
        m.title = title_from_path(m.source_path)


def _inspect(cfg, pmap) -> None:
    tmp = Path("/tmp/legal_inspect"); tmp.mkdir(exist_ok=True)
    files = [p for p in sorted(SRC.iterdir()) if p.suffix.lower() in SUFFIXES]
    print(f"INSPECT {len(files)} files -> {tmp}\n", flush=True)
    for p in files:
        try:
            meta, chunks = build_document_metadata(
                p, config=cfg, outputs_dir=tmp, provenance=provenance_for(p, pmap))
            text = "".join(c.text for c in chunks)
            alpha = sum(ch.isalpha() for ch in text) / max(1, len(text))
            flag = "  <-- LOW" if (len(text.split()) < 50 or alpha < 0.55) else ""
            print(f"{p.name[:52]:54} {meta.page_count:>3}p {len(chunks):>3}ch "
                  f"{len(text.split()):>6}w alpha={alpha*100:3.0f}% "
                  f"[{','.join(sorted(set(meta.page_routes)))}]{flag}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{p.name}: ERROR {e}", flush=True)


def main() -> None:
    cfg = cascade_config_from_env()
    pmap = load_provenance_manifest(MANIFEST, retrieved_at="") if MANIFEST.exists() else {}

    if "--inspect" in sys.argv:
        _inspect(cfg, pmap)
        return

    emb = embedder_from_env()
    idx = DocIndex.load(IDX, embedder=emb) if IDX.is_file() else DocIndex(embedder=emb)
    done = {m.source_path for m in idx.documents.values()}
    todo = [p for p in sorted(SRC.iterdir())
            if p.suffix.lower() in SUFFIXES and str(p) not in done]
    print(f"legal KB: {len(done)} done, {len(todo)} to do, provider={cfg.parse_provider}, "
          f"dense={'on' if emb else 'off'}", flush=True)

    t0 = time.time()
    for i, path in enumerate(todo, 1):
        try:
            meta, chunks = build_document_metadata(
                path, config=cfg, outputs_dir=OUT, provenance=provenance_for(path, pmap),
                degradation_sink=idx.degradations)
            _fix_title(meta)
            idx.add(meta, chunks)
            print(f"  [{i}/{len(todo)}] {path.name[:46]:48} {meta.page_count:>3}p "
                  f"[{','.join(sorted(set(meta.page_routes)))}] «{meta.section}»", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(todo)}] {path.name}: ERROR {e}", flush=True)
            traceback.print_exc()
        if i % 10 == 0:
            idx.save(IDX)

    from parsevault.pipeline.docindex import mark_supersessions, write_build_report

    supersessions = mark_supersessions(idx)
    idx.save(IDX)
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    report_path = write_build_report(idx, KB, stamp=stamp, supersessions=supersessions)
    stamped = sum(1 for m in idx.documents.values() if m.source_url)
    print(f"\nDONE: {len(idx.documents)} docs ({stamped} with source_url), "
          f"{len(idx.chunks)} chunks -> {IDX} ({time.time()-t0:.0f}s)", flush=True)
    print(f"build report -> {report_path}", flush=True)


if __name__ == "__main__":
    main()
