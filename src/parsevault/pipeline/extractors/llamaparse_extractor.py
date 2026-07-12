"""LlamaParse lane — high-fidelity cloud conversion for PUBLIC data only.

LlamaParse (LlamaCloud) is the ParseBench leader for tables/structure. It is a
*cloud* service, so it MUST NOT be used on the private/no-egress path — it is
gated behind ``parse_provider="llamaparse"`` in config and only enabled for the
public knowledge base. The local cascade remains the default and the fallback if
a parse fails or credits are exhausted.

REST flow: POST ``/parsing/upload`` → poll ``/parsing/job/{id}`` → GET
``/parsing/job/{id}/result/markdown``.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from .base import BaseExtractor, ExtractionResult, PageResult, assemble_anchored

DEFAULT_LLAMAPARSE_URL = "https://api.cloud.llamaindex.ai/api/v1"


class LlamaParseCreditsExhausted(RuntimeError):
    """Raised on HTTP 402/429 so a batch build can stop or fall back and alert."""


class LlamaParseExtractor(BaseExtractor):
    name = "llamaparse"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_LLAMAPARSE_URL,
        timeout: float = 60.0,
        poll_interval: float = 2.0,
        max_poll_seconds: float = 600.0,
        result_type: str = "markdown",
    ):
        if not api_key:
            raise ValueError("LlamaParseExtractor requires an API key")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_poll_seconds = max_poll_seconds
        self.result_type = result_type

    @property
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "accept": "application/json"}

    def is_available(self) -> bool:
        return bool(self.api_key)

    def extract(self, file_path: Path, pages: list[int] | None = None) -> ExtractionResult:
        import requests

        start = time.time()
        file_path = Path(file_path)
        base = f"{self.base_url}/parsing"

        # 1. upload
        with open(file_path, "rb") as fh:
            r = requests.post(f"{base}/upload", headers=self._headers,
                              files={"file": fh}, timeout=self.timeout)
        self._raise_for_status(r)
        job_id = r.json()["id"]

        # 2. poll
        deadline = time.time() + self.max_poll_seconds
        status = "PENDING"
        while time.time() < deadline:
            jr = requests.get(f"{base}/job/{job_id}", headers=self._headers, timeout=self.timeout)
            self._raise_for_status(jr)
            status = jr.json().get("status", "")
            if status in ("SUCCESS", "COMPLETED", "PARTIAL_SUCCESS"):
                break
            if status in ("ERROR", "FAILED", "CANCELLED"):
                raise RuntimeError(f"LlamaParse job {job_id} {status}")
            time.sleep(self.poll_interval)
        else:
            raise RuntimeError(f"LlamaParse job {job_id} timed out (status {status})")

        # 3. fetch markdown
        mr = requests.get(f"{base}/job/{job_id}/result/{self.result_type}",
                          headers=self._headers, timeout=self.timeout)
        self._raise_for_status(mr)
        markdown = (mr.json().get(self.result_type) or "").strip()

        page_count = self._page_count(file_path, markdown)
        # LlamaParse separates pages with a horizontal rule; recover per-page
        # Markdown so page-level citation works (best-effort — falls back to a
        # single page when no separator is present).
        segments = [s.strip() for s in re.split(r"\n-{3,}\n", markdown) if s.strip()]
        page_objs = [
            PageResult(page_number=i, markdown=seg, route=self.name)
            for i, seg in enumerate(segments or [markdown], start=1)
        ]
        from ..quality import score_pages

        score_pages(page_objs)
        return ExtractionResult(
            pdf_name=file_path.stem,
            markdown=assemble_anchored(page_objs),
            page_count=page_count,
            elapsed_seconds=time.time() - start,
            extractor=self.name,
            page_routes=[p.route for p in page_objs],
            pages=page_objs,
        )

    @staticmethod
    def _raise_for_status(r) -> None:
        if r.status_code in (402, 429):
            raise LlamaParseCreditsExhausted(
                f"LlamaParse HTTP {r.status_code}: {r.text[:200]}"
            )
        if r.status_code >= 400:
            raise RuntimeError(f"LlamaParse HTTP {r.status_code}: {r.text[:200]}")

    @staticmethod
    def _page_count(file_path: Path, markdown: str) -> int:
        # Cheap, reliable page count for PDFs; otherwise infer from page breaks.
        if file_path.suffix.lower() == ".pdf":
            try:
                import fitz

                with fitz.open(file_path) as doc:
                    return len(doc)
            except Exception:
                pass
        return markdown.count("\n---\n") + 1 if markdown else 1
