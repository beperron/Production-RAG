"""Vision-LLM OCR lane — talks to a local vLLM server (OpenAI-compatible).

vLLM is the serving engine (highest open-source throughput: paged-attention KV
cache, continuous batching, tensor-parallel across the 2×A6000). We speak its
``/v1/chat/completions`` endpoint with an inline base64 image, so there is no
heavy client dependency — just ``requests`` (already required) — and the same
client works against any OpenAI-compatible server.

Nothing here leaves the machine: ``base_url`` defaults to localhost. This is the
quality lane for scanned/complex pages where layout fidelity matters.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import requests

from .base import BaseExtractor, ExtractionResult
from .models import REGISTRY, VLMModel
from .raster import DEFAULT_DPI, image_to_data_url, rasterize_page

DEFAULT_BASE_URL = "http://localhost:8000/v1"
_log = logging.getLogger(__name__)


class VLMUnavailable(RuntimeError):
    """Raised when the vLLM server is unreachable or unhealthy."""


class VLMExtractor(BaseExtractor):
    name = "vlm"

    def __init__(
        self,
        model: str | VLMModel,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "EMPTY",  # vLLM ignores it unless --api-key was set
        dpi: int = DEFAULT_DPI,
        timeout: float = 180.0,
        max_tokens: int = 8192,
        temperature: float = 0.0,
    ):
        self.spec: VLMModel | None = model if isinstance(model, VLMModel) else REGISTRY.get(model)
        # The API "model" must match the server's served-model-name. Our serve
        # script defaults that to the HF repo, so a registry key/spec resolves to
        # its repo; an unknown string is used verbatim (custom served name).
        if isinstance(model, VLMModel):
            self.model_id = model.hf_repo
        elif self.spec is not None:
            self.model_id = self.spec.hf_repo
        else:
            self.model_id = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.dpi = dpi
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature

    # -- health ---------------------------------------------------------------
    def is_available(self) -> bool:
        try:
            r = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5,
            )
            return r.status_code == 200
        except requests.RequestException:
            return False

    # -- per-image / per-page rendering --------------------------------------
    def render_image(self, image, *, prompt: str | None = None) -> str:
        text = self._chat(image, prompt or self._prompt())
        return _strip_code_fence(text)

    def render_page(self, page, *, prompt: str | None = None) -> str:
        return self.render_image(rasterize_page(page, self.dpi), prompt=prompt)

    # -- whole-document standalone interface ---------------------------------
    def extract(self, file_path: Path, pages: list[int] | None = None) -> ExtractionResult:
        start = time.time()
        suffix = file_path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
            from PIL import Image

            md = self.render_image(Image.open(file_path).convert("RGB"))
            return ExtractionResult(
                pdf_name=file_path.stem,
                markdown=md,
                page_count=1,
                elapsed_seconds=time.time() - start,
                extractor=f"{self.name}:{self.model_id}",
            )

        import fitz

        doc = fitz.open(file_path)
        page_count = len(doc)
        wanted = set(pages) if pages is not None else None
        parts: list[str] = []
        for page in doc:
            if wanted is not None and page.number not in wanted:
                continue
            parts.append(self.render_page(page))
        doc.close()
        return ExtractionResult(
            pdf_name=file_path.stem,
            markdown="\n\n".join(p for p in parts if p.strip()),
            page_count=page_count,
            elapsed_seconds=time.time() - start,
            extractor=f"{self.name}:{self.model_id}",
        )

    # -- internals ------------------------------------------------------------
    def _prompt(self) -> str:
        return self.spec.prompt if self.spec else VLMModel.__dataclass_fields__["prompt"].default

    def _chat(self, image, prompt: str) -> str:
        payload = {
            "model": self.model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise VLMUnavailable(f"vLLM request failed: {e}") from e
        if r.status_code != 200:
            raise VLMUnavailable(f"vLLM returned HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise VLMUnavailable(f"Unexpected vLLM response shape: {e}") from e
        # M5: distinguish a *normal-but-low* page from an empty model response
        # in the log stream. An empty string here makes the page silently empty
        # markdown, which then trips the cascade's OCR cross-check at 0.0
        # agreement and the quality scorer's "empty_page" flag — both correct,
        # but indistinguishable from a genuinely blank page. A WARNING with
        # the model id makes the cause grep-able. The caller still gets ""
        # (no behavior change) so downstream fallback paths still trigger.
        if not (content or "").strip():
            _log.warning(
                "vlm_empty_response model=%s — server returned empty content; "
                "page will be empty markdown (downstream OCR cross-check / "
                "quality flag will catch it)",
                self.model_id,
            )
            return ""
        return content


def _strip_code_fence(text: str) -> str:
    """Some models wrap the whole page in a ```markdown … ``` fence; unwrap it
    so the result is Markdown, not a Markdown code block.

    L2 audit note: when the fence is *single-line* (no newline after the opening
    backticks) ``first_nl == -1`` and the opening-fence strip is skipped — this
    is intentional. The trailing-fence strip below still runs, so a one-line
    ``` `` `text` `` `` payload still yields ``text``. The two branches are
    independent; neither relies on the other having matched.
    """
    s = text.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            # Multi-line fence: drop the opening ``` line (often "```markdown").
            s = s[first_nl + 1 :]
        # Always strip a trailing ``` if present — independent of the opening
        # strip above, so single-line fences degrade gracefully.
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()
