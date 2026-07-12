"""Generate a committable catalog (SUMMARY.md + MANIFEST.csv) from a KB index.

The index itself (docindex.json) is heavy and gitignored; this distills it into
two lightweight, human- and grep-friendly artifacts that *are* committed so the
KB's contents and provenance are visible in the repo without shipping the corpus.

Usage:
    python scripts/catalog_kb.py <kb-dir>          # e.g. knowledge-base/nc-child-welfare
"""
import collections
import csv
import json
import sys
from pathlib import Path


def load_docs(index_path: Path) -> list[dict]:
    d = json.loads(index_path.read_text())
    docs = d.get("documents", [])
    return list(docs.values()) if isinstance(docs, dict) else docs


def _filename(m: dict) -> str:
    return Path(m.get("source_path", "")).name


def write_manifest(docs: list[dict], out: Path) -> None:
    cols = [
        "doc_id", "filename", "title", "section", "category", "topic", "audience",
        "form_number", "rev_date", "page_count", "word_count", "extractor",
        "source_url", "source_domain", "retrieved_at", "content_sha256",
    ]
    rows = sorted(docs, key=lambda m: (m.get("section", ""), _filename(m).lower()))
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for m in rows:
            w.writerow({**{c: m.get(c, "") for c in cols}, "filename": _filename(m)})


def _table(counter: collections.Counter, label: str) -> list[str]:
    total = sum(counter.values()) or 1
    lines = [f"| {label} | Documents | Share |", "| --- | ---: | ---: |"]
    for k, n in counter.most_common():
        lines.append(f"| {k or '—'} | {n} | {n * 100 // total}% |")
    return lines


def write_summary(docs: list[dict], chunk_count: int, out: Path) -> None:
    n = len(docs)
    by_section = collections.Counter(m.get("section") for m in docs)
    by_category = collections.Counter(m.get("category") for m in docs)
    by_topic = collections.Counter(m.get("topic") for m in docs)
    by_extractor = collections.Counter(m.get("extractor") for m in docs)
    by_domain = collections.Counter(m.get("source_domain") for m in docs)
    with_prov = sum(1 for m in docs if m.get("source_url"))
    dates = sorted(d for d in (m.get("retrieved_at") for m in docs) if d)
    pages = sum(m.get("page_count", 0) for m in docs)
    words = sum(m.get("word_count", 0) for m in docs)

    L: list[str] = []
    L.append(f"# {out.parent.name} — knowledge base catalog")
    L.append("")
    L.append("> Generated from the index by `scripts/catalog_kb.py`. The index, "
             "sources, and converted outputs are gitignored; this catalog and "
             "`MANIFEST.csv` are the committed, lightweight view of the corpus.")
    L.append("")
    # Hand-written caveats survive regeneration by living in SCOPE_NOTE.md
    # next to the index (regen previously silently dropped them).
    scope_note = out.parent / "SCOPE_NOTE.md"
    if scope_note.is_file():
        L.append(scope_note.read_text().strip())
        L.append("")
    L.append("## At a glance")
    L.append("")
    L.append(f"- **Documents:** {n}")
    L.append(f"- **Chunks (retrievable passages):** {chunk_count}")
    L.append(f"- **Total pages / words:** {pages:,} / {words:,}")
    L.append(f"- **Provenance coverage:** {with_prov}/{n} "
             f"({with_prov * 100 // (n or 1)}%) carry a source URL + retrieval date")
    if dates:
        L.append(f"- **Retrieved between:** {dates[0]} → {dates[-1]}")
    L.append("")
    L.append("## By section")
    L.append("")
    L += _table(by_section, "Section")
    L.append("")
    L.append("## By category")
    L.append("")
    L += _table(by_category, "Category")
    L.append("")
    L.append("## By topic")
    L.append("")
    L += _table(by_topic, "Topic")
    L.append("")
    L.append("## Provenance & conversion")
    L.append("")
    L += _table(by_domain, "Source domain")
    L.append("")
    L += _table(by_extractor, "Converter route")
    L.append("")
    L.append("See `MANIFEST.csv` for the full per-document list with "
             "doc IDs, titles, source URLs, and SHA-256 content hashes.")
    L.append("")
    out.write_text("\n".join(L))


def main() -> None:
    # The workspace argument is REQUIRED: a hardcoded default meant a bare run
    # silently rewrote the committed nc-child-welfare catalog.
    if len(sys.argv) < 2:
        sys.exit(
            "usage: python scripts/catalog_kb.py <kb-dir>\n"
            "error: the workspace argument is required "
            "(no default — a bare run must never rewrite a committed catalog)"
        )
    if sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        raise SystemExit(0)
    kb = Path(sys.argv[1])
    index_path = kb / "docindex.json"
    if not index_path.is_file():
        sys.exit(f"no index at {index_path}")
    raw = json.loads(index_path.read_text())
    docs = list(raw["documents"].values()) if isinstance(raw["documents"], dict) else raw["documents"]
    chunk_count = len(raw.get("chunks", []))
    write_manifest(docs, kb / "MANIFEST.csv")
    write_summary(docs, chunk_count, kb / "SUMMARY.md")
    print(f"catalog: {len(docs)} docs, {chunk_count} chunks -> "
          f"{kb/'SUMMARY.md'} + {kb/'MANIFEST.csv'}")


if __name__ == "__main__":
    main()
