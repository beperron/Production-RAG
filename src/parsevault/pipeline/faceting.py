"""Faceted classification for a reference knowledge base.

Beyond the record-type classifier (``doctype.py``, built for *filled case
records*), a public policy/forms library is best navigated by **facets**: what
kind of artifact it is (``category``), what it's *about* (``topic``), who it's
*for* (``audience``), plus the agency ``form_number`` and ``rev_date``. Topical
facets are what let a query like "safe sleep" surface every related form, manual
section, and policy at once (``safe sleep`` lives under ``topic=safety``).

Filename-first, then content — agency filenames (``dss-5239``, ``safe-sleep-…``)
are highly reliable. Ported from the project's ``classify_documents.py`` so the
knowledge base matches the curated classification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Order matters: the first matching category/topic wins (most specific first).
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "intake": ["intake", "report to central registry", "cps application", "screening"],
    "assessment": ["assessment", "evaluation", "finds", "investigation",
                   "structured decision making", "sdm"],
    "case_planning": ["case plan", "service plan", "treatment plan",
                      "permanency planning", "family services agreement"],
    "foster_care": ["foster care", "foster home", "licensing", "placement", "fhl"],
    "adoption": ["adoption", "adoptive", "decree of adoption", "post-adoption"],
    "kinship": ["kinship", "kinfam", "kingap", "relative", "unlicensed kinship"],
    "court": ["court", "decree", "petition", "verification of custody", "judicial",
              "summons", "consent to adoption"],
    "health": ["health history", "health summary", "medicaid", "medical",
               "perinatal", "substance use"],
    "safety": ["safe sleep", "firearm", "discipline", "abuse", "neglect",
               "trafficking", "suicide"],
    "administrative": ["administrative letter", "change notice", "fiscal",
                       "procurement", "personnel", "civil rights", "complaint"],
}

AUDIENCE_KEYWORDS: dict[str, list[str]] = {
    "social_worker": ["social worker", "caseworker", "case worker", "frontline"],
    "supervisor": ["supervisor", "program manager", "county director"],
    "court": ["court", "judge", "clerk", "petitioner", "judicial"],
    "provider": ["provider", "agency", "contract", "foster parent", "kinship provider"],
    "family": ["family", "parent", "caregiver", "child", "youth"],
}

FORM_NUMBER_RE = re.compile(r"\b(?:DSS|DHHS(?:-AS)?)[-_ ]?(\d{3,5}[A-Z]?)\b", re.IGNORECASE)
REV_DATE_RE = re.compile(
    r"\b(?:Rev\.?|Revised)\s*[:.]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)


@dataclass
class Facets:
    category: str = "other"   # form | manual | policy | letter | guidance | resource | other
    topic: str = "other"
    audience: str = "mixed"
    form_number: str = ""
    rev_date: str = ""


def _first_keyword_match(text_lower: str, keyword_map: dict[str, list[str]]) -> str:
    for cat, kws in keyword_map.items():
        if any(kw in text_lower for kw in kws):
            return cat
    return ""


def classify_topic(text: str, filename: str) -> str:
    fn = filename.lower().replace("-", " ").replace("_", " ")
    for topic, kws in TOPIC_KEYWORDS.items():
        if any(kw in fn for kw in kws):
            return topic
    return _first_keyword_match((text or "").lower(), TOPIC_KEYWORDS) or "other"


def classify_category(text: str, filename: str) -> str:
    """The artifact kind: form | manual | policy | guidance | resource | letter | other."""
    fn = filename.lower()
    if re.search(r"^dss[-_]?\d{3,5}[a-z]?[-_.]", fn) or re.search(r"^dhhs[-_]as[-_]?\d{4}", fn):
        return "form"
    if any(kw in fn for kw in ["manual", "guide", "resource", "practice", "tip",
                               "tool kit", "toolkit"]):
        return "manual" if "manual" in fn else "guidance"
    if ("administrative letter" in fn or "cws-al" in fn or "cws-cn" in fn
            or "change notice" in fn or re.search(r"_al[-_0-9]", fn) or re.search(r"_cn[-_0-9]", fn)):
        return "policy"
    head = (text or "").lower()[:1000]
    if "manual" in head:
        return "manual"
    if "administrative letter" in head or "change notice" in head:
        return "policy"
    if head.startswith("dear ") or "memorandum" in head:
        return "letter"
    return "other"


def classify_audience(text: str, filename: str) -> str:
    return _first_keyword_match((text or "").lower(), AUDIENCE_KEYWORDS) or "mixed"


def extract_form_number(text: str, filename: str) -> str:
    m = re.search(r"(DSS|DHHS(?:-AS)?)[-_ ]?(\d{3,5}[A-Z]?)", filename, re.IGNORECASE)
    if m:
        return f"{m.group(1).upper().replace(' ', '')}-{m.group(2).upper()}"
    m = FORM_NUMBER_RE.search((text or "")[:1500])
    if m:
        return f"DSS-{m.group(1).upper()}"
    return ""


def extract_rev_date(text: str, filename: str) -> str:
    m = re.search(r"(\d{1,2})[-./](\d{1,2})[-./](\d{2,4})", filename)
    if m:
        mo, dy, yr = m.groups()
        yr = ("20" + yr) if len(yr) == 2 else yr
        try:
            return f"{yr}-{int(mo):02d}-{int(dy):02d}"
        except ValueError:
            pass
    m = re.search(r"(\d{4})[-.](\d{2})[-.](\d{2})", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = REV_DATE_RE.search((text or "")[:2000])
    return m.group(1) if m else ""


def compute_facets(text: str, source_path: str) -> Facets:
    """Derive all facets for a document from its text and source filename."""
    from pathlib import Path

    filename = Path(source_path).name
    return Facets(
        category=classify_category(text, filename),
        topic=classify_topic(text, filename),
        audience=classify_audience(text, filename),
        form_number=extract_form_number(text, filename),
        rev_date=extract_rev_date(text, filename),
    )
