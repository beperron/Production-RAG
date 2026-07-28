"""Tests for scripts/fetch_sources.py.

Pure filesystem + CSV logic with requests.get monkeypatched out — no real
network calls. Imported as a top-level module (`scripts/` is on sys.path
via pyproject.toml's pythonpath option), matching the other scripts/ tests.
"""
from __future__ import annotations

import csv

import fetch_sources


def _write_manifest(kb, rows):
    with (kb / "MANIFEST.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "source_url"])
        w.writeheader()
        w.writerows(rows)


def test_skips_files_already_on_disk(tmp_path, monkeypatch):
    kb = tmp_path / "some-kb"
    kb.mkdir()
    sources = kb / "sources"
    sources.mkdir()
    (sources / "already-have.pdf").write_bytes(b"existing bytes")

    _write_manifest(kb, [
        {"filename": "already-have.pdf", "source_url": "https://example.com/already-have.pdf"},
        {"filename": "missing.pdf", "source_url": "https://example.com/missing.pdf"},
    ])

    calls = []

    class FakeResponse:
        content = b"fetched bytes"

        def raise_for_status(self):
            pass

    def fake_get(url, timeout=None, headers=None):
        calls.append(url)
        return FakeResponse()

    monkeypatch.setattr(fetch_sources.requests, "get", fake_get)
    monkeypatch.setattr(fetch_sources.time, "sleep", lambda _: None)
    monkeypatch.setattr("sys.argv", ["fetch_sources.py", str(kb)])

    fetch_sources.main()

    # Only the missing file triggered a network call — the existing one was
    # left untouched (both content and call count prove it wasn't re-fetched).
    assert calls == ["https://example.com/missing.pdf"]
    assert (sources / "already-have.pdf").read_bytes() == b"existing bytes"
    assert (sources / "missing.pdf").read_bytes() == b"fetched bytes"


def test_failed_fetch_does_not_write_a_file(tmp_path, monkeypatch):
    kb = tmp_path / "some-kb"
    kb.mkdir()
    (kb / "sources").mkdir()
    _write_manifest(kb, [
        {"filename": "broken.pdf", "source_url": "https://example.com/broken.pdf"},
    ])

    def fake_get(url, timeout=None, headers=None):
        raise fetch_sources.requests.RequestException("connection refused")

    monkeypatch.setattr(fetch_sources.requests, "get", fake_get)
    monkeypatch.setattr(fetch_sources.time, "sleep", lambda _: None)
    monkeypatch.setattr("sys.argv", ["fetch_sources.py", str(kb)])

    fetch_sources.main()

    assert not (kb / "sources" / "broken.pdf").exists()
