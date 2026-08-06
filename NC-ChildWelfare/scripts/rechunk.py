#!/usr/bin/env python3
"""Boundary-aware re-chunker for the public corpora.

Rebuilds docindex.json chunks from the committed converted markdown (no re-OCR):
  * sections on Markdown headings AND statutory section markers (§ 7B-XXXX.)
  * splits over-long text at enumerator boundaries ((a)/(1)) then sentence
    boundaries — never mid-word/mid-sentence
  * strips page-header/footer artifacts
  * preserves page spans from <!-- page: N --> anchors

Usage:
  python scripts/rechunk.py --inspect <doc_id>     # dry-run, print a doc's chunks
  python scripts/rechunk.py                          # rewrite both docindex.json files
"""
import argparse, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPS = {"legal-authorities": 1500, "nc-child-welfare": 1500}
MIN_CHARS = 200
PAGE_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")
HEAD_RE = re.compile(r"^(#{1,6})\s+(.*)$")
# statute section start: "§ 7B-1111." or "§ 108A-50.3." (optionally with title)
SEC_RE = re.compile(r"§\s*\d+[A-Z]?-\d+(?:\.\d+)?\.")
# page-footer / header junk lines
JUNK_RE = re.compile(r"^\s*(NC General Statutes.*\d+|Page \d+.*|\d+\s*)$", re.I)
# split before a statutory enumerator "(a) " "(1) " "(a1) " "(2a) "
ENUM_SPLIT = re.compile(r"(?<=[.\s])(?=\((?:[a-z]{1,2}\d?|\d{1,3}[a-z]?)\)\s)")
SENT_SPLIT = re.compile(r"(?<=[.;:])\s+(?=[A-Z(\"])")


def word_wrap(s, cap):
    out, cur = [], ""
    for w in s.split(" "):
        if cur and len(cur) + len(w) + 1 > cap:
            out.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        out.append(cur)
    return out


def atomize(text, cap):
    """Break text into atoms no larger than cap, at enumerator -> sentence ->
    word boundaries (last resort). Never cuts inside a word."""
    atoms = []
    for part in ENUM_SPLIT.split(text):
        part = part.strip()
        if not part:
            continue
        if len(part) <= cap:
            atoms.append(part); continue
        for sent in SENT_SPLIT.split(part):
            sent = sent.strip()
            if not sent:
                continue
            atoms.extend([sent] if len(sent) <= cap else word_wrap(sent, cap))
    return atoms


def pack(atoms_pl, cap):
    """Pack (page, atom) items into pieces <= cap; return [(text, p0, p1)]."""
    pieces, buf, pages = [], "", []
    for pg, a in atoms_pl:
        if buf and len(buf) + len(a) + 1 > cap:
            pieces.append((buf, min(pages), max(pages))); buf, pages = "", []
        buf = f"{buf} {a}".strip() if buf else a
        pages.append(pg)
    if buf:
        pieces.append((buf, min(pages), max(pages)))
    return pieces


def parse_md(md):
    """-> list of (page, line), page anchors consumed, junk lines dropped."""
    out, page = [], 1
    for line in md.splitlines():
        m = PAGE_RE.search(line.strip())
        if m:
            page = int(m.group(1)); continue
        if JUNK_RE.match(line):
            continue
        out.append((page, line))
    return out


def sectionize(lines):
    """Split (page,line) list into sections at headings or § markers.
    -> list of (heading, [(page,line)])."""
    sections, head, body = [], "", []
    def flush():
        if any(l.strip() for _p, l in body):
            sections.append((head, list(body)))
    for pg, line in lines:
        hm = HEAD_RE.match(line)
        sm = SEC_RE.match(line.strip())
        if hm:
            flush(); body.clear(); head = hm.group(2).strip(); body.append((pg, head))
        elif sm:
            flush(); body.clear()
            head = line.strip()[:90].split(". ")[0] + "."  # "§ 7B-1111." (+title cut)
            head = re.match(r"§\s*\d+[A-Z]?-\d+(?:\.\d+)?\.[^.]*\.?", line.strip())
            head = head.group(0).strip() if head else line.strip()[:80]
            body.append((pg, line))
        else:
            body.append((pg, line))
    flush()
    return sections


def chunk_doc(md, doc_id, cap):
    lines = parse_md(md)
    sections = sectionize(lines)
    chunks, ordn = [], 0
    for head, body in sections:
        # atoms tagged with their line's page
        atoms_pl = []
        for pg, line in body:
            line = line.strip()
            if not line:
                continue
            for a in atomize(line, cap):
                atoms_pl.append((pg, a))
        if not atoms_pl:
            continue
        leaf = head
        for i, (text, p0, p1) in enumerate(pack(atoms_pl, cap)):
            if i > 0 and leaf and not text.startswith(leaf):
                text = f"({leaf}, continued) {text}"
            chunks.append({"chunk_id": f"{doc_id}:{ordn}", "doc_id": doc_id,
                           "ordinal": ordn, "heading_path": [head] if head else [],
                           "text": text, "char_len": len(text),
                           "page_start": p0, "page_end": p1})
            ordn += 1
    # coalesce tiny chunks forward
    out = []
    for c in chunks:
        if out and out[-1]["char_len"] < MIN_CHARS and out[-1]["char_len"] + c["char_len"] < int(cap * 1.6):
            p = out[-1]
            p["text"] = f"{p['text']} {c['text']}".strip(); p["char_len"] = len(p["text"])
            p["page_end"] = c["page_end"]
        else:
            out.append(c)
    for i, c in enumerate(out):
        c["ordinal"] = i; c["chunk_id"] = f"{doc_id}:{i}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", metavar="DOC_ID", default=None)
    a = ap.parse_args()
    for coll, cap in CAPS.items():
        idx_path = ROOT / "knowledge-base" / coll / "docindex.json"
        idx = json.loads(idx_path.read_text())
        outdir = ROOT / "knowledge-base" / coll / "outputs"
        new_chunks, docs_done = [], 0
        for d in idx["documents"]:
            md_path = outdir / f"{d['doc_id']}.md"
            if not md_path.is_file():
                continue
            cs = chunk_doc(md_path.read_text(), d["doc_id"], cap)
            if a.inspect and d["doc_id"] == a.inspect:
                for c in cs:
                    if "7B-1111" in " ".join(c["heading_path"]) or a.inspect != "98afdb5710fb60ea":
                        end = c["text"][-46:].replace("\n", " ")
                        clean_end = bool(re.search(r'[.;:)\]"]\s*$', c["text"].strip()))
                        print(f"  ord={c['ordinal']} len={c['char_len']} pp{c['page_start']}-{c['page_end']} ends_clean={clean_end}  …{end!r}")
                return
            new_chunks.extend(cs); docs_done += 1
        if not a.inspect:
            idx["chunks"] = new_chunks
            idx_path.write_text(json.dumps(idx))
            print(f"{coll}: {docs_done} docs -> {len(new_chunks)} chunks (cap {cap})")


if __name__ == "__main__":
    main()
