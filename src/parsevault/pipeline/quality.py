"""Per-page extraction quality scoring + human-review status (R3).

Every page gets a persisted, inspectable quality record so the legal team can
*see* conversion quality rather than trust it. Two independent signals bound the
two failure modes:

* **OCR mean confidence** (Tesseract pages) — flags low-confidence scans.
* **VLM↔OCR agreement** (VLM pages) — a token-overlap (Jaccard) cross-check
  against an independent Tesseract pass on the same raster, bounding the VLM's
  principal risk (hallucinated/dropped text) with a second measurement.

Plus a cheap **garbled** check (mojibake/structure) for any lane. Pages below
threshold are flagged into a review queue; a reviewer's accept/correct decision
is recorded as a verification status. This module is pure/deterministic and
model-free so it is trivially testable; the extractor supplies the raw signals.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .extractors.textnorm import looks_garbled

# Defaults (overridable by the caller / config).
DEFAULT_OCR_CONF_MIN = 65.0       # Tesseract mean conf below this → flag
DEFAULT_VLM_AGREEMENT_MIN = 0.55  # VLM↔OCR Jaccard below this → flag
_MIN_CHARS = 1                    # an empty page is itself a flag

_TOKEN_RE = re.compile(r"[0-9a-z]+")


def jaccard_tokens(a: str, b: str) -> float:
    """Jaccard overlap of lowercased alphanumeric tokens — the VLM↔OCR check."""
    ta = set(_TOKEN_RE.findall(a.lower()))
    tb = set(_TOKEN_RE.findall(b.lower()))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class PageQuality:
    page_number: int
    route: str = ""
    chars: int = 0
    garbled: bool = False
    ocr_mean_confidence: float | None = None   # Tesseract pages
    vlm_agreement: float | None = None         # VLM pages (Jaccard vs OCR)
    flagged: bool = False
    flag_reasons: list[str] = field(default_factory=list)
    # Human-review outcome (R3.3). status ∈ unreviewed|verified|corrected|rejected
    verification_status: str = "unreviewed"
    reviewer: str = ""
    reviewed_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def score_page(
    page_number: int,
    text: str,
    route: str,
    *,
    ocr_mean_confidence: float | None = None,
    vlm_agreement: float | None = None,
    ocr_conf_min: float = DEFAULT_OCR_CONF_MIN,
    vlm_agreement_min: float = DEFAULT_VLM_AGREEMENT_MIN,
) -> PageQuality:
    """Build a PageQuality record + flag decision from the available signals."""
    reasons: list[str] = []
    chars = len(text.strip())
    if chars < _MIN_CHARS:
        reasons.append("empty_page")
    garbled = bool(text.strip()) and looks_garbled(text)
    if garbled:
        reasons.append("garbled_text")
    if ocr_mean_confidence is not None and ocr_mean_confidence < ocr_conf_min:
        reasons.append(f"low_ocr_confidence({ocr_mean_confidence:.0f}<{ocr_conf_min:.0f})")
    if vlm_agreement is not None and vlm_agreement < vlm_agreement_min:
        reasons.append(f"low_vlm_agreement({vlm_agreement:.2f}<{vlm_agreement_min:.2f})")
    return PageQuality(
        page_number=page_number,
        route=route,
        chars=chars,
        garbled=garbled,
        ocr_mean_confidence=ocr_mean_confidence,
        vlm_agreement=vlm_agreement,
        flagged=bool(reasons),
        flag_reasons=reasons,
    )


def score_pages(pages, signals: dict[int, dict] | None = None):
    """Set ``.quality`` on each ``PageResult`` from optional per-page ``signals``
    ({page_number: {ocr_mean_confidence?, vlm_agreement?}}). Mutates + returns."""
    signals = signals or {}
    for p in pages:
        s = signals.get(p.page_number, {})
        p.quality = score_page(
            p.page_number, p.markdown, p.route,
            ocr_mean_confidence=s.get("ocr_mean_confidence"),
            vlm_agreement=s.get("vlm_agreement"),
        ).to_dict()
    return pages


def document_quality(pages: list[PageQuality]) -> tuple[float, list[int], str]:
    """Aggregate page quality → (quality_score, flagged_pages, review_status).

    ``quality_score`` is the fraction of pages that are not flagged (1.0 = all
    clean). ``review_status`` reflects how much of the flagged set a HUMAN has
    resolved: unreviewed | no_flags | partially_verified | verified.

    ``"verified"`` means a reviewer ran ``parsevault review
    accept|correct|reject`` and no flagged page is still unreviewed — it is a
    human sign-off, never a machine inference. The machine "nothing was
    flagged" state is ``"no_flags"`` (clean automated conversion, zero human
    eyes on it), so ``search --verified-only`` cannot be satisfied by an
    unaudited corpus.
    """
    if not pages:
        return 1.0, [], "no_flags"
    flagged = [p.page_number for p in pages if p.flagged]
    score = round(1.0 - len(flagged) / len(pages), 4)
    pending = [p for p in pages if p.flagged and p.verification_status == "unreviewed"]
    reviewed = [p for p in pages if p.verification_status != "unreviewed"]
    if not flagged:
        # No flagged pages: "verified" ONLY when a human actually reviewed at
        # least one page (e.g. a correction unflagged the last flagged page);
        # otherwise this is the machine no-flags state, not a human sign-off.
        status = "verified" if reviewed else "no_flags"
    elif not pending:
        status = "verified"
    elif len(pending) < len(flagged):
        status = "partially_verified"
    else:
        status = "unreviewed"
    return score, flagged, status
