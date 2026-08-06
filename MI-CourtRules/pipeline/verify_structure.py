#!/usr/bin/env python3
"""Reconcile the parsed body against the document's own printed Table of
Contents.

    verify_structure.py 0_source/michigan-court-rules.pdf 1_parsed/blocks.jsonl

Why this exists
---------------
In the bench-books work, 17 of 18 books were built, evaluated and published
with their section structure silently flattened. Every gate passed, because
every gate asked "is the corpus a faithful rendering of the source" and the
answer was yes. Nothing asked "did we find the sections the document says it
has", because nothing had an independent list of them.

This document ships that list: 18 pages of Table of Contents naming every
chapter, subchapter and rule, with page numbers. Parsing it separately gives a
ground truth that was never touched by the body parser, so a disagreement
between the two is real evidence rather than a restatement.

A structural defect has to survive BOTH parsers reading different pages of the
document in different formats to go unnoticed here.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

import fitz

RE_TOC_RULE = re.compile(
    r"^Rule\s+(?P<num>\d+\.\d+[A-Za-z]?)\.?\s+(?P<title>.*?)\s*\.{4,}\s*(?P<page>\d+)\s*$")
RE_TOC_SUBCH = re.compile(
    r"^Subchapter\s+(?P<num>[\d.]+)\s+(?P<title>.*?)\s*\.{4,}\s*(?P<page>\d+)\s*$")
RE_TOC_CHAP = re.compile(
    r"^Chapter\s+(?P<num>\d+[A-Za-z]?)\.\s+(?P<title>.*?)\s*\.{4,}\s*(?P<page>\d+)\s*$")


RE_OPEN = re.compile(
    r"^(?P<kind>Chapter|Subchapter|Rule)\s+(?P<num>[\d.]+[A-Za-z]?)\.?\s+(?P<title>.*)$")


def _close(buf, page):
    """Close an entry whose page number arrived on its own line."""
    m = RE_OPEN.match(buf)
    return {"kind": m.group("kind").lower(),
            "num": m.group("num").rstrip("."),
            "title": re.sub(r"\s*\.{2,}\s*$", "",
                            re.sub(r"\s+", " ", m.group("title"))).strip(),
            "printed_page": page}


def parse_toc(doc, toc_pages):
    """Read the printed contents. Entries wrap, so lines are joined until a
    dot-leader and page number close the entry."""
    entries = []
    buf = ""
    for pno in range(toc_pages):
        for raw in doc[pno].get_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("Table of Contents"):
                continue
            if re.fullmatch(r"[ivxlcdm]+|\d+", line):
                # A bare number closes an open entry whose title was too long
                # to leave room for dot leaders; otherwise it is a page folio.
                if buf and re.match(r"^(Chapter|Subchapter|Rule)\s", buf):
                    entries.append(_close(buf, int(line)))
                    buf = ""
                continue
            if line.startswith("Updated") or "Updated" in line and len(line) < 40:
                continue
            # A line that opens a new entry resets the buffer. Without this a
            # stray heading ("MICHIGAN COURT RULES OF 1985") prefixes the
            # buffer, ^Chapter never matches, and the whole page is swallowed.
            if re.match(r"^(Chapter|Subchapter|Rule)\s", line):
                buf = line
            elif buf:
                buf = (buf + " " + line).strip()
            else:
                continue
            for rx, kind in ((RE_TOC_RULE, "rule"),
                             (RE_TOC_SUBCH, "subchapter"),
                             (RE_TOC_CHAP, "chapter")):
                m = rx.match(buf)
                if m:
                    entries.append({
                        "kind": kind,
                        "num": m.group("num"),
                        "title": re.sub(r"\s+", " ", m.group("title")).strip(),
                        "printed_page": int(m.group("page")),
                    })
                    buf = ""
                    break
            else:
                # Third variant: the title nearly fills the column, so the
                # leaders are dropped and the page number follows a plain
                # space. Gated on length, because leaders are only omitted
                # when the title is long -- a short title ending in a number
                # must not be mistaken for a closed entry.
                m = re.match(r"^(Chapter|Subchapter|Rule)\s.*\s(\d{1,3})$", buf)
                if m and len(buf) >= 85:
                    entries.append(_close(buf[: buf.rfind(" ")], int(m.group(2))))
                    buf = ""
                elif len(buf) > 400:                          # runaway guard
                    buf = ""
    return entries


def load_blocks(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("blocks")
    ap.add_argument("--toc-pages", type=int, default=18)
    ap.add_argument("--page-offset", type=int, default=18,
                    help="pdf_page = printed_page + offset")
    ap.add_argument("-o", "--out", default="1_parsed/verify_report.json")
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    toc = parse_toc(doc, args.toc_pages)
    blocks = load_blocks(args.blocks)

    toc_rules = {e["num"]: e for e in toc if e["kind"] == "rule"}
    toc_subch = {e["num"]: e for e in toc if e["kind"] == "subchapter"}
    toc_chaps = {e["num"]: e for e in toc if e["kind"] == "chapter"}

    body_rules, body_subch, body_chaps = {}, {}, {}
    for b in blocks:
        if b["kind"] == "rule" and b["rule"]:
            body_rules.setdefault(b["rule"], b)
        elif b["kind"] == "subchapter" and b["subchapter"]:
            body_subch.setdefault(b["subchapter"], b)
        elif b["kind"] == "chapter" and b["chapter"]:
            body_chaps.setdefault(b["chapter"], b)

    findings = []

    def compare(name, expected, found):
        miss = sorted(set(expected) - set(found))
        extra = sorted(set(found) - set(expected))
        if miss:
            findings.append({"level": "ERROR", "code": f"{name}.missing",
                             "detail": f"in the contents but not parsed from the "
                                       f"body: {miss[:25]}", "count": len(miss)})
        if extra:
            findings.append({"level": "ERROR", "code": f"{name}.unexpected",
                             "detail": f"parsed from the body but absent from the "
                                       f"contents: {extra[:25]}", "count": len(extra)})
        return len(miss), len(extra)

    compare("chapters", toc_chaps, body_chaps)
    compare("subchapters", toc_subch, body_subch)
    compare("rules", toc_rules, body_rules)

    # page agreement -- catches a rule matched by number but found in the
    # wrong place, which a set comparison alone would miss
    off = []
    for num, e in toc_rules.items():
        b = body_rules.get(num)
        if not b:
            continue
        expected_pdf = e["printed_page"] + args.page_offset
        if abs(b["page"] - expected_pdf) > 1:
            off.append({"rule": num, "toc_printed": e["printed_page"],
                        "expected_pdf": expected_pdf, "found_pdf": b["page"],
                        "delta": b["page"] - expected_pdf})
    if off:
        findings.append({"level": "ERROR", "code": "rules.page_mismatch",
                         "detail": f"{len(off)} rules parsed more than one page "
                                   f"from where the contents place them",
                         "count": len(off), "examples": off[:10]})

    # title agreement
    title_diff = []
    for num, e in toc_rules.items():
        b = body_rules.get(num)
        if not b:
            continue
        body_title = re.sub(r"^Rule\s+\S+\s*", "", b["text"]).strip()
        norm = lambda s: re.sub(r"\s+", " ", s.replace("’", "'")).strip().lower()
        if norm(body_title) != norm(e["title"]):
            title_diff.append({"rule": num, "toc": e["title"], "body": body_title})
    if title_diff:
        findings.append({"level": "WARN", "code": "rules.title_mismatch",
                         "detail": f"{len(title_diff)} rule titles differ between "
                                   f"the contents and the body heading",
                         "count": len(title_diff), "examples": title_diff[:10]})

    # every body block must be attributable to a rule
    orphans = [b for b in blocks if b["kind"] == "body" and not b["rule"]]
    if orphans:
        findings.append({"level": "WARN", "code": "blocks.orphaned",
                         "detail": f"{len(orphans)} body blocks sit under no rule",
                         "count": len(orphans),
                         "examples": [{"id": o["id"], "page": o["page"],
                                       "text": o["text"][:90]} for o in orphans[:8]]})

    report = {
        "toc_entries": {"chapters": len(toc_chaps), "subchapters": len(toc_subch),
                        "rules": len(toc_rules)},
        "body_parsed": {"chapters": len(body_chaps), "subchapters": len(body_subch),
                        "rules": len(body_rules)},
        "errors": sum(1 for f in findings if f["level"] == "ERROR"),
        "warnings": sum(1 for f in findings if f["level"] == "WARN"),
        "findings": findings,
    }
    pathlib.Path(args.out).write_text(json.dumps(report, indent=1) + "\n")

    print(f"contents : {report['toc_entries']}")
    print(f"body     : {report['body_parsed']}")
    for f in findings:
        print(f"  {f['level']:<5} {f['code']}: {f['detail']}")
    print(f"\n{report['errors']} error(s), {report['warnings']} warning(s)")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
