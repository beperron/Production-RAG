"""Optional build-time event emitter for the pedantic build dashboard.

Side-effect-only, same shape as the existing ``degradation_sink`` parameter
pattern used in ``build_document_metadata()`` (docindex.py). Writes one JSON
object per line to a JSONL file, flushed immediately so a dashboard polling
the file live sees events as they happen. Never raises — a build must keep
running even if the event log can't be written.
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path


class BuildEventLogger:
    def __init__(self, path: str | Path):
        self._fh = None
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(p, "a", encoding="utf-8")
        except OSError as e:
            print(f"WARNING: build event log disabled ({path}): {e}", file=sys.stderr)

    def event(self, stage: str, **fields) -> None:
        if self._fh is None:
            return
        ts = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        try:
            self._fh.write(json.dumps({"ts": ts, "stage": stage, **fields}, ensure_ascii=False) + "\n")
            self._fh.flush()
        except OSError as e:
            print(f"WARNING: build event write failed: {e}", file=sys.stderr)

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
