#!/usr/bin/env python3
"""Command-line search over the public NC law & policy knowledge bases.

    python scripts/query.py "grounds for termination of parental rights"
    python scripts/query.py "reasonable efforts to prevent removal" --answer

Public data only — the engine refuses any confidential workspace.
First run performs a one-time dense re-embed from the committed index.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from parsevault.lawsearch import LawSearch  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--collection", default=None, help="legal-authorities | nc-child-welfare")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--answer", action="store_true", help="also produce a grounded, cited answer")
    a = ap.parse_args()
    eng = LawSearch()
    if a.answer:
        coll, res = eng.answer(a.query, k=a.k, collection=a.collection)
        print(f"\n=== grounded answer ({coll}) ===\n{getattr(res,'answer','')}\n")
    print(f"=== top {a.k} results ===")
    for i, h in enumerate(eng.search(a.query, k=a.k, collection=a.collection), 1):
        print(f"\n[{i}] {h.section or h.title}")
        print(f"    {h.citation}")
        print(f"    {h.snippet[:220]}")

if __name__ == "__main__":
    main()
