"""Local document-OCR VLM registry, keyed to GPU budget.

This is the *candidate set* the agent can serve locally via vLLM. It is
deliberately data-driven: ``benchmark.py`` (under ``scripts/ocr/``) runs these
on the user's own hardware and sample documents and records which actually win
on accuracy/latency — the registry only proposes, the benchmark disposes.

Why a registry rather than a single hard-coded model: document OCR quality is
extremely document-dependent (scanned vs born-digital, tables, math, multi-
column, language), and the best open model moves quickly. Encoding params,
VRAM cost, quantization, and vLLM serve args per model lets us (a) pick a sane
default for a given card and (b) regenerate serve commands without hand-editing.

VRAM math (rule of thumb): weights ≈ params_b × bytes/param (2 for bf16/fp16,
~1 for int8/awq-4bit is ~0.6). vLLM also needs KV-cache + activation headroom;
budget ~1.3–1.6× weights for a comfortable serve. ``min_vram_gb`` below already
bakes in that headroom for a single replica at the listed precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VLMModel:
    """A servable vision-LLM OCR candidate."""

    key: str
    hf_repo: str
    params_b: float
    tier: str  # "fast" (high-throughput escalation) or "quality" (best accuracy)
    precision: str  # default serve precision: "bf16" | "awq" | "fp8" | "int8"
    min_vram_gb: float  # comfortable single-replica budget at `precision`
    needs_tensor_parallel: int = 1  # >1 → shard across this many GPUs
    # Extra vLLM CLI args this model needs (beyond the common ones). Kept as a
    # list of already-split tokens so serve_model.sh can splice them verbatim.
    extra_vllm_args: list[str] = field(default_factory=list)
    # The instruction that turns the model into a faithful page transcriber.
    prompt: str = (
        "Convert this document page to clean, faithful GitHub-flavored Markdown. "
        "Preserve heading hierarchy, reading order, and lists. Render tables with "
        "pipe syntax. Render equations as LaTeX ($...$ inline, $$...$$ display). "
        "Transcribe text exactly; do not summarize, translate, or add commentary. "
        "For figures, emit a Markdown image with a concise descriptive alt text. "
        "Output only the Markdown."
    )
    notes: str = ""
    # Some OCR-specialist models ship their own task prompt / chat template and
    # ignore a generic instruction; flag them so the client can adapt.
    ocr_specialist: bool = False
    available: bool = True  # set False for "validate on your box first" entries


# Ordered roughly best-first within each tier for the 48GB+ class of card.
REGISTRY: dict[str, VLMModel] = {
    # ---- Quality tier (escalation target for complex/scanned pages) ----------
    "qwen2.5-vl-32b": VLMModel(
        key="qwen2.5-vl-32b",
        hf_repo="Qwen/Qwen2.5-VL-32B-Instruct",
        params_b=32.0,
        tier="quality",
        precision="bf16",
        min_vram_gb=80.0,  # ~64GB weights + KV; comfortable across 2×48GB (TP=2)
        needs_tensor_parallel=2,
        extra_vllm_args=["--limit-mm-per-prompt", "image=2"],
        notes="Top open general VLM for dense docs/tables/math. TP=2 on 2×A6000.",
    ),
    "qwen3-vl-32b": VLMModel(
        key="qwen3-vl-32b",
        hf_repo="Qwen/Qwen3-VL-32B-Instruct",
        params_b=32.0,
        tier="quality",
        precision="bf16",
        min_vram_gb=80.0,
        needs_tensor_parallel=2,
        notes="Successor line; if present on HF it generally edges out 2.5-VL. "
        "Validate availability on your box (benchmark skips it if the pull fails).",
        available=False,
    ),
    "qwen2.5-vl-7b": VLMModel(
        key="qwen2.5-vl-7b",
        hf_repo="Qwen/Qwen2.5-VL-7B-Instruct",
        params_b=7.0,
        tier="quality",
        precision="bf16",
        min_vram_gb=22.0,
        notes="Single-GPU quality fallback if the 32B is busy/OOM. Strong OCR.",
    ),
    # ---- Fast tier (high-throughput; clean-ish but non-trivial pages) --------
    "olmocr-7b": VLMModel(
        key="olmocr-7b",
        hf_repo="allenai/olmOCR-7B-0825",
        params_b=7.0,
        tier="fast",
        precision="bf16",
        min_vram_gb=22.0,
        ocr_specialist=True,
        notes="Qwen2.5-VL-7B fine-tuned purpose-built for full-page Markdown OCR. "
        "Excellent throughput/quality on born-digital + light scans.",
    ),
    "dots-ocr": VLMModel(
        key="dots-ocr",
        hf_repo="rednote-hilab/dots.ocr",
        params_b=3.0,
        tier="fast",
        precision="bf16",
        min_vram_gb=12.0,
        ocr_specialist=True,
        extra_vllm_args=["--trust-remote-code"],
        notes="Compact layout+text OCR, strong multilingual, very fast. Great "
        "default for the high-throughput lane.",
    ),
    "nanonets-ocr2-3b": VLMModel(
        key="nanonets-ocr2-3b",
        hf_repo="nanonets/Nanonets-OCR2-3B",
        params_b=3.0,
        tier="fast",
        precision="bf16",
        min_vram_gb=12.0,
        ocr_specialist=True,
        notes="Markdown OCR with tables, LaTeX, checkboxes, signatures. "
        "Good structured-doc alternative in the fast lane.",
    ),
    "paddleocr-vl": VLMModel(
        key="paddleocr-vl",
        hf_repo="PaddlePaddle/PaddleOCR-VL",
        params_b=0.9,
        tier="fast",
        precision="bf16",
        min_vram_gb=8.0,
        ocr_specialist=True,
        extra_vllm_args=["--trust-remote-code"],
        notes="Sub-1B ERNIE-based OCR-VL; punches far above its size. Lowest "
        "VRAM, highest pages/sec. Validate on your box.",
        available=False,
    ),
}


# Default picks per GPU class. The router uses `quality` for escalation and
# `fast` for the bulk lane; a third entry names a single-GPU safety net.
@dataclass(frozen=True)
class GPUPlan:
    label: str
    quality: str
    fast: str
    safety: str  # single-GPU model that always fits if TP serving is down
    rationale: str


def vllm_serve_args(
    key: str,
    *,
    port: int = 8000,
    gpu_memory_utilization: float = 0.92,
    max_model_len: int = 32768,
) -> list[str]:
    """Build the ``vllm serve`` argument list for a registry model.

    Single source of truth for serve_model.sh and the docs: tensor-parallel size,
    dtype, and any model-specific flags all come from the registry entry, so the
    shell script never hard-codes (and never drifts from) these values.
    """
    m = REGISTRY[key]
    args = [
        m.hf_repo,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(m.needs_tensor_parallel),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--dtype",
        "bfloat16" if m.precision == "bf16" else "auto",
    ]
    args += m.extra_vllm_args
    return args


def recommend_plan(total_vram_gb: float, num_gpus: int = 1) -> GPUPlan:
    """Pick quality/fast/safety models for the detected GPU budget.

    Conservative: requires the *quality* model to fit across all GPUs (TP) and
    the *fast*/*safety* models to fit on a single GPU, so a single card can keep
    serving if the multi-GPU server is unavailable.
    """
    per_gpu = total_vram_gb / max(num_gpus, 1)

    if total_vram_gb >= 72 and num_gpus >= 2:
        return GPUPlan(
            label="dual-48GB (e.g. 2×RTX A6000)",
            quality="qwen2.5-vl-32b",  # TP=2
            fast="olmocr-7b",
            safety="dots-ocr",
            rationale="32B quality model sharded TP=2 for the escalation lane; "
            "olmOCR-7B on a single card for the high-throughput lane; dots.ocr "
            "as the always-fits safety net.",
        )
    if per_gpu >= 40:
        return GPUPlan(
            label="single 40GB+ (e.g. A100-40/48GB)",
            quality="qwen2.5-vl-7b",
            fast="olmocr-7b",
            safety="dots-ocr",
            rationale="7B quality + olmOCR-7B fit comfortably on one large card.",
        )
    if per_gpu >= 22:
        return GPUPlan(
            label="single 24GB (e.g. RTX 4090/3090)",
            quality="qwen2.5-vl-7b",
            fast="olmocr-7b",
            safety="dots-ocr",
            rationale="7B-class at bf16 fits 24GB; olmOCR-7B shares the lane.",
        )
    if per_gpu >= 11:
        return GPUPlan(
            label="single 12GB",
            quality="dots-ocr",
            fast="dots-ocr",
            safety="paddleocr-vl",
            rationale="Compact OCR specialists only; no room for 7B at bf16.",
        )
    return GPUPlan(
        label="<12GB / CPU-only",
        quality="paddleocr-vl",
        fast="paddleocr-vl",
        safety="paddleocr-vl",
        rationale="Below the VLM comfort floor — rely on the Tesseract lane and "
        "the smallest OCR-VL only if a GPU is present.",
    )
