#!/usr/bin/env python3
"""Parse the Michigan Court Rules PDF into a structured, citable corpus.

    parse_mcr.py 0_source/michigan-court-rules.pdf -o 1_parsed/

The document is born-digital FrameMaker output, so every word carries exact
geometry and font. Nothing here is OCR and nothing is guessed: the structure is
read off typographic conventions that were measured first, and every assumption
is asserted at run time rather than trusted.

The measured model
------------------
  footer            y > 730          chapter name / page / update date
  Chapter           Bold 18, centred
  Subchapter        Bold 14, x=72    WRAPS -- continuation is also Bold 14 x=72
  Rule              Bold 12, x=72
  body indent ladder, +21.6 per level:
      93.6  (A)      115.2  (1)      136.8  (a)      158.4  (i)     180.0  deeper
      108.0          a flush paragraph under a rule that has no (A) subrules
  A continuation line sits at its parent's indent + 21.6, which is also the
  next level's item indent -- so indent ALONE cannot tell a continuation from
  an item. The marker text and the vertical gap settle it.
  line spacing      14.0 within a paragraph, 20.0 between   -> break at >= 17

Why the citation path is the point
----------------------------------
Lawyers cite MCR 2.116(C)(10), not "page 312". Every emitted block carries the
full path it sits under, so a chunk can be cited, and a query naming a subrule
can be matched to it. That is the property this corpus exists to have.
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import re
import sys

import fitz

FOOTER_Y = 730.0
GAP_BREAK = 17.0          # 14.0 within a paragraph, 20.0 between
HEAD_WRAP_GAP = 18.5      # headings lead at 17.0; a wrap sits within this
SAME_LINE = 3.0           # fragments within this many points share a line
LADDER = 21.6             # points per hierarchy level

# indent -> depth. 108.0 is a flush paragraph under a rule with no subrules;
# it behaves as depth 1 without consuming a marker slot.
INDENT_DEPTH = {72.0: 0, 93.6: 1, 108.0: 1, 115.2: 2, 136.8: 3, 158.4: 4,
                180.0: 5, 201.6: 6}
INDENT_TOL = 1.5

# FrameMaker sets the marker in a fixed-width slot, and where the label is
# wide ("(10)", "(vii)") the following space can be absorbed entirely -- 195
# markers in this document have none. Requiring one silently demotes those
# items to body text and mis-cites everything beneath them. The vocabulary is
# tightened instead, so a parenthetical like "(see below)" cannot match.
RE_MARKER = re.compile(r"^\((?P<m>\d{1,3}|[A-Za-z]|[ivxlIVXL]{2,5})\)")
RE_RULE = re.compile(r"^Rule\s+(?P<num>\d+\.\d+[A-Za-z]?)\.?\s*(?P<title>.*)$")
RE_SUBCH = re.compile(r"^Subchapter\s+(?P<num>[\d.]+)\s*(?P<title>.*)$")
RE_CHAPTER = re.compile(r"^Chapter\s+(?P<num>\d+[A-Za-z]?)\.\s*(?P<title>.*)$")

RE_CHAPTER_META = re.compile(r"^Chapter Updated with MSC Order\(s\) Effective on (.+)$")

ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi",
         "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx"]


def successors(prev):
    """Every marker that could legitimately follow `prev`.

    A glyph alone cannot say whether (i) is the letter after (h) or roman one,
    so both readings are accepted and the SEQUENCE is what gets checked. That
    catches the defects worth catching -- a skipped or invented item -- without
    inventing violations wherever the alphabet meets roman numerals."""
    out = set()
    if prev.isdigit():
        return {str(int(prev) + 1)}
    if len(prev) == 1 and prev.isalpha():
        nxt = chr(ord(prev) + 1)
        out.add(nxt)
        if prev.lower() == "h":       # (h) -> (i), which may start romans
            out.add("ii" if prev.islower() else "II")
    low = prev.lower()
    if low in ROMAN:
        i = ROMAN.index(low)
        if i + 1 < len(ROMAN):
            r = ROMAN[i + 1]
            out.add(r if prev.islower() else r.upper())
    return out


FIRST = {"A", "1", "a", "i", "I", "(a)"}


def snap(x):
    """Snap a measured indent onto the ladder, or return None if it is off it."""
    for known in INDENT_DEPTH:
        if abs(x - known) <= INDENT_TOL:
            return known
    return None


# --------------------------------------------------------------------------
# 1. lines
# --------------------------------------------------------------------------

def extract_lines(doc):
    """Every non-footer line, in reading order, with geometry and style."""
    out = []
    for pno in range(doc.page_count):
        raw = []
        for block in doc[pno].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                if line["bbox"][1] > FOOTER_Y:
                    continue
                text = "".join(s["text"] for s in line["spans"])
                if not text.strip():
                    continue
                span = max(line["spans"], key=lambda s: len(s["text"]))
                raw.append({
                    "page": pno,
                    "y": round(line["bbox"][1], 1),
                    "x": round(line["bbox"][0], 1),
                    "text": text,
                    "size": round(span["size"], 1),
                    "bold": "Bold" in span["font"],
                    "italic": "Italic" in span["font"],
                })
        raw.sort(key=lambda r: (r["y"], r["x"]))

        # fragments printed at the same y are one visual line
        merged = []
        for r in raw:
            if merged and abs(r["y"] - merged[-1]["y"]) <= SAME_LINE:
                prev = merged[-1]
                joiner = "" if prev["text"].endswith(" ") else " "
                prev["text"] += joiner + r["text"]
            else:
                merged.append(r)
        out.extend(merged)
    return out


# --------------------------------------------------------------------------
# 2. blocks
# --------------------------------------------------------------------------

def classify(line):
    t = line["text"].strip()
    if line["bold"] and line["size"] >= 17:
        return "chapter"
    if line["bold"] and 13 <= line["size"] < 17:
        return "subchapter"
    if line["bold"] and line["size"] < 13:
        if RE_RULE.match(t):
            return "rule"
        if abs(line["x"] - 72.0) <= 1.5:
            return "rule_cont"    # a rule title that wrapped
        return "boldrun"          # bold lead-in inside body text
    return "body"


def join(prev_text, add):
    """Join a wrapped line. Legal text almost never soft-hyphenates, so a
    trailing hyphen is kept and the words are closed up."""
    prev_text = prev_text.rstrip()
    add = add.strip()
    if prev_text.endswith("-"):
        return prev_text + add
    return prev_text + " " + add


def build_blocks(lines):
    """Group lines into logical blocks, carrying indent depth and marker."""
    blocks, warnings = [], []
    for i, ln in enumerate(lines):
        kind = classify(ln)
        text = ln["text"].strip()
        prev = blocks[-1] if blocks else None
        snapped = snap(ln["x"])
        marker = None
        m = RE_MARKER.match(text)
        if m and snapped is not None and INDENT_DEPTH[snapped] >= 1:
            marker = m.group("m")

        same_page = prev is not None and prev["page_end"] == ln["page"]
        gap = ln["y"] - prev["y_end"] if same_page else None

        # --- a wrapped rule title rejoins its heading ------------------------
        if kind == "rule_cont":
            if (prev is not None and prev["kind"] == "rule" and same_page
                    and gap is not None and gap <= HEAD_WRAP_GAP):
                prev["text"] = join(prev["text"], text)
                prev["y_end"], prev["page_end"] = ln["y"], ln["page"]
                continue
            kind = "body"         # a bold x=72 line that continues no heading

        # --- headings -------------------------------------------------------
        if kind in ("chapter", "subchapter", "rule"):
            # A subchapter title that wraps continues in the SAME style at the
            # SAME x, 17.0pt below. Joining it is what stops phantom
            # subchapters called "Motions" or "Actions" from appearing -- the
            # exact defect that flattened 17 bench books.
            if (prev is not None and kind == "subchapter"
                    and prev["kind"] == "subchapter"
                    and not RE_SUBCH.match(text)
                    and same_page and gap is not None and gap <= HEAD_WRAP_GAP):
                prev["text"] = join(prev["text"], text)
                prev["y_end"], prev["page_end"] = ln["y"], ln["page"]
                continue
            if (prev is not None and kind == "rule"
                    and prev["kind"] == "rule"
                    and same_page and gap is not None and gap <= HEAD_WRAP_GAP
                    and not RE_RULE.match(text)):
                prev["text"] = join(prev["text"], text)
                prev["y_end"], prev["page_end"] = ln["y"], ln["page"]
                continue
            if kind == "subchapter" and not RE_SUBCH.match(text):
                # Not a section -- but it is the document's edition date, which
                # a legal corpus must carry. Keep it as front matter rather
                # than dropping it on the floor.
                blocks.append(_new(ln, "front_matter", text, snapped, None))
                continue
            blocks.append(_new(ln, kind, text, snapped, None))
            continue

        # --- body -----------------------------------------------------------
        if snapped is None:
            warnings.append(f"p{ln['page']} x={ln['x']} off the indent ladder: "
                            f"{text[:60]!r}")

        if RE_CHAPTER_META.match(text):
            blocks.append(_new(ln, "chapter_meta", text, snapped, None))
            continue
        if ln["page"] == 18:
            blocks.append(_new(ln, "front_matter", text, snapped, None))
            continue

        starts_new = (
            prev is None
            or prev["kind"] in ("chapter", "subchapter", "rule")
            or marker is not None                       # an explicit item
            or gap is None                              # page turn: see below
            or gap >= GAP_BREAK
        )
        # A page turn is not evidence either way -- a paragraph may run across
        # it. Continue unless the new line is itself an item or dedents.
        if gap is None and prev is not None and marker is None:
            prev_depth = INDENT_DEPTH.get(prev["indent"], 99)
            here = INDENT_DEPTH.get(snapped, 99)
            if here >= prev_depth:
                starts_new = False

        if starts_new:
            blocks.append(_new(ln, "body", text, snapped, marker))
        else:
            prev["text"] = join(prev["text"], text)
            prev["y_end"], prev["page_end"] = ln["y"], ln["page"]
    return blocks, warnings


def _new(ln, kind, text, snapped, marker):
    return {"kind": kind, "text": text, "page": ln["page"],
            "page_end": ln["page"], "y": ln["y"], "y_end": ln["y"],
            "x": ln["x"], "indent": snapped, "marker": marker,
            "depth": INDENT_DEPTH.get(snapped, None)}


# --------------------------------------------------------------------------
# 3. citation paths
# --------------------------------------------------------------------------

def b_subpath(stack):
    return "".join(f"({m})" for _, m in stack)


def assign_paths(blocks):
    """Walk the blocks, maintaining the Chapter/Subchapter/Rule/(A)(1)(a)(i)
    stack, and stamp every block with the citation it lives under."""
    chapter = subchapter = rule = None
    stack = []          # [(depth, marker)]
    problems = []

    for b in blocks:
        t = b["text"]
        if b["kind"] == "chapter":
            m = RE_CHAPTER.match(t)
            chapter = m.group("num") if m else t
            subchapter = rule = None
            stack = []
        elif b["kind"] == "subchapter":
            m = RE_SUBCH.match(t)
            subchapter = m.group("num") if m else t
            rule = None
            stack = []
        elif b["kind"] == "rule":
            m = RE_RULE.match(t)
            rule = m.group("num") if m else None
            if rule is None:
                problems.append(f"p{b['page']} unparsable rule heading: {t[:60]!r}")
            stack = []
        elif b["marker"] is not None:
            depth, mk = b["depth"], b["marker"]
            sibling = next((m for d, m in reversed(stack) if d == depth), None)
            if sibling is None:
                if mk not in FIRST and not mk.isdigit():
                    problems.append(
                        f"p{b['page']} MCR {rule}{b_subpath(stack)}: first item at "
                        f"depth {depth} is ({mk}), expected ({'/'.join(sorted(FIRST))})")
            elif mk not in successors(sibling):
                problems.append(
                    f"p{b['page']} MCR {rule}{b_subpath(stack)}: ({sibling}) is "
                    f"followed by ({mk}) -- sequence break, an item may be "
                    f"missing or misread")
            stack = [(d, m) for d, m in stack if d < depth]
            stack.append((depth, mk))

        b["chapter"], b["subchapter"], b["rule"] = chapter, subchapter, rule
        b["subpath"] = "".join(f"({mk})" for _, mk in stack)
        b["citation"] = (f"MCR {rule}{b['subpath']}" if rule else
                         (f"Subchapter {subchapter}" if subchapter else
                          (f"Chapter {chapter}" if chapter else None)))
    return problems


# --------------------------------------------------------------------------
# 4. output
# --------------------------------------------------------------------------

def to_markdown(blocks):
    out = []
    for b in blocks:
        t = b["text"]
        if b["kind"] == "chapter":
            out.append(f"\n# {t}\n")
        elif b["kind"] == "subchapter":
            out.append(f"\n## {t}\n")
        elif b["kind"] == "rule":
            out.append(f"\n### {t}\n")
        else:
            indent = "    " * max(0, (b["depth"] or 1) - 1)
            out.append(f"{indent}{t}\n")
    md = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", md).strip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("-o", "--outdir", default="1_parsed")
    ap.add_argument("--body-start", type=int, default=18,
                    help="0-based first body page; earlier pages are the TOC")
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(args.pdf)
    lines = [l for l in extract_lines(doc) if l["page"] >= args.body_start]
    blocks, warnings = build_blocks(lines)
    problems = assign_paths(blocks)

    body = [b for b in blocks if b["kind"] == "body"]
    rules = [b for b in blocks if b["kind"] == "rule"]

    with open(outdir / "blocks.jsonl", "w") as fh:
        for i, b in enumerate(blocks):
            fh.write(json.dumps({
                "id": f"mcr#B{i:05d}", "kind": b["kind"], "text": b["text"],
                "page": b["page"], "page_end": b["page_end"],
                "chapter": b["chapter"], "subchapter": b["subchapter"],
                "rule": b["rule"], "subpath": b["subpath"],
                "citation": b["citation"], "depth": b["depth"],
                "marker": b["marker"],
            }, ensure_ascii=False) + "\n")

    (outdir / "mcr.md").write_text(to_markdown(blocks))

    stats = {
        "pdf_pages": doc.page_count,
        "body_start_page": args.body_start,
        "lines": len(lines),
        "blocks": len(blocks),
        "chapters": sum(1 for b in blocks if b["kind"] == "chapter"),
        "subchapters": sum(1 for b in blocks if b["kind"] == "subchapter"),
        "rules": len(rules),
        "body_blocks": len(body),
        "blocks_with_rule": sum(1 for b in body if b["rule"]),
        "off_ladder_warnings": len(warnings),
        "path_problems": len(problems),
    }
    (outdir / "parse_stats.json").write_text(json.dumps(stats, indent=1) + "\n")

    print(json.dumps(stats, indent=1))
    for w in warnings[:15]:
        print("  WARN off-ladder:", w)
    for p in problems[:15]:
        print("  PROBLEM:", p)
    if len(warnings) > 15:
        print(f"  ... {len(warnings)-15} more off-ladder warnings")
    if len(problems) > 15:
        print(f"  ... {len(problems)-15} more path problems")
    return 0


if __name__ == "__main__":
    sys.exit(main())
