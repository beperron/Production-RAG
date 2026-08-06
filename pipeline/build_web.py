#!/usr/bin/env python3
"""Assemble web/ (the Vercel project root) from the pipeline sources.

Copies are one-way: web/ is a build product; edit pipeline/ and rerun.
The 40MB source PDF is replaced by its sha256 sidecar (provenance.Ledger
reads it when the PDF itself is absent).
"""
import hashlib
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

COPIES = [
    ("pipeline/mcr_search.py", "pipeline/mcr_search.py"),
    ("pipeline/provenance.py", "pipeline/provenance.py"),
    ("pipeline/querylog.py", "pipeline/querylog.py"),
    ("pipeline/serve.py", "pipeline/serve.py"),
    ("pipeline/cloud.py", "pipeline/cloud.py"),
    ("1_parsed/blocks.jsonl", "1_parsed/blocks.jsonl"),
    ("1_parsed/xrefs.jsonl", "1_parsed/xrefs.jsonl"),
    ("3_chunks/v_rule256.jsonl", "3_chunks/v_rule256.jsonl"),
]


def main():
    for src, dst in COPIES:
        d = WEB / dst
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / src, d)
        print(f"  {dst}")
    if (WEB / "static").exists():
        shutil.rmtree(WEB / "static")
    shutil.copytree(ROOT / "static", WEB / "static")
    print("  static/")
    pdf = ROOT / "0_source" / "michigan-court-rules.pdf"
    sidecar = WEB / "0_source" / "michigan-court-rules.pdf.sha256"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(hashlib.sha256(pdf.read_bytes()).hexdigest() + "\n")
    print("  0_source/ (sha sidecar only)")


if __name__ == "__main__":
    main()
