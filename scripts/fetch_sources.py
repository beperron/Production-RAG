"""Re-download a KB's source PDFs from the URLs recorded in its MANIFEST.csv.

Source PDFs are gitignored (public, re-fetchable — see .gitignore) so a fresh
checkout has to pull them back down before `build_kb.py` can (re)run. This
reads MANIFEST.csv's `source_url` / `filename` columns and fetches anything
missing from the target `sources/` dir. Resumable: skips files already on
disk. Rate-limited to be polite to the origin server.

Usage:
    python scripts/fetch_sources.py <kb-dir>   # e.g. knowledge-base/nc-child-welfare
"""
import csv
import sys
import time
from pathlib import Path

import requests

DELAY_SECONDS = 0.5
TIMEOUT = 60


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        raise SystemExit(0)

    kb = Path(sys.argv[1])
    manifest = kb / "MANIFEST.csv"
    out_dir = kb / "sources"
    out_dir.mkdir(parents=True, exist_ok=True)

    with manifest.open() as f:
        rows = list(csv.DictReader(f))

    todo = [r for r in rows if not (out_dir / r["filename"]).exists()]
    print(f"fetch: {len(rows) - len(todo)} already on disk, {len(todo)} to fetch", flush=True)

    ok = fail = 0
    for i, row in enumerate(todo, 1):
        dest = out_dir / row["filename"]
        url = row["source_url"]
        try:
            r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "carolina-policy-search/1.0"})
            r.raise_for_status()
            dest.write_bytes(r.content)
            ok += 1
            print(f"[{i}/{len(todo)}] ok  {row['filename']} ({len(r.content)} bytes)", flush=True)
        except requests.RequestException as e:
            fail += 1
            print(f"[{i}/{len(todo)}] FAIL {row['filename']}: {e}", flush=True)
        time.sleep(DELAY_SECONDS)

    print(f"fetch done: {ok} ok, {fail} failed, {len(rows) - len(todo)} skipped (already present)")


if __name__ == "__main__":
    main()
