"""build_document_metadata's extract_done event carries per-page timing.

router.py now measures real per-page elapsed_seconds (see
tests/test_router_timing.py); this test pins the other half of the pipe —
that docindex.py's event_sink actually forwards it, since that's what lands
in the hash-chained build_log table in Postgres and is what the dashboard's
"Time by extraction route" chart reads.

Uses a real ``.txt`` document (the plain-text lane needs no Tesseract binary
or vLLM server) rather than mocking the extractor, so this exercises the
real router.py -> docindex.py wire, not just the event-shaping code.
"""
from __future__ import annotations

from parsevault.pipeline.docindex import build_document_metadata
from parsevault.pipeline.extractors.router import CascadeConfig


def _capture():
    events: list[tuple[str, dict]] = []

    def sink(stage, **fields):
        events.append((stage, fields))

    return events, sink


def test_extract_done_event_carries_page_elapsed_seconds(tmp_path):
    txt_path = tmp_path / "doc.txt"
    txt_path.write_text("Hello world, this is a short plain-text document.")

    events, sink = _capture()
    build_document_metadata(txt_path, config=CascadeConfig(use_vlm=False), event_sink=sink)

    extract_done = [fields for stage, fields in events if stage == "extract_done"]
    assert len(extract_done) == 1
    pages = extract_done[0]["pages"]
    assert len(pages) == 1
    assert pages[0]["route"] == "text"
    assert isinstance(pages[0]["elapsed_seconds"], float)
    assert pages[0]["elapsed_seconds"] >= 0.0
