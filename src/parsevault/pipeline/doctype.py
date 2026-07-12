"""Document-type classification — explainable, local, child-welfare-aware.

Determines what a document *is* — court order, case note, medical record,
standardized form, assessment, correspondence, etc. — so downstream handling,
routing, and retrieval can be type-aware.

Context & guarantees (this is built for child-welfare / children's-rights work,
where records concern minors and are highly sensitive):

* **Local only.** Classification runs on this machine — deterministic CPU rules,
  optionally the local embedding model, optionally the local VLM. No document or
  derived text is ever sent off-box. This preserves the pipeline's no-egress
  guarantee for FERPA/PII-grade data.
* **Explainable / auditable.** The result names the exact signals that fired and
  the score for every candidate type — never an opaque label. In a setting where
  a classification may be reviewed or contested, the reasoning must be legible.
* **Conservative.** Low-confidence inputs return ``other`` rather than guessing;
  a human stays in the loop.

The CPU rule layer needs no model. An optional embedding layer (zero-shot vs.
type descriptions) and an optional VLM escalation (for genuinely ambiguous
pages) follow the same CPU-first / GPU-as-necessary design as the rest of the
system. Taxonomy is data-driven, so a jurisdiction/agency can adjust it.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Signal:
    """A weighted, named cue. ``pattern`` is matched case-insensitively."""

    name: str
    pattern: str
    weight: float = 1.0
    _re: re.Pattern | None = field(default=None, repr=False, compare=False)

    def compiled(self) -> re.Pattern:
        if self._re is None:
            self._re = re.compile(self.pattern, re.IGNORECASE | re.MULTILINE)
        return self._re


@dataclass
class TypeSpec:
    name: str
    description: str  # used for embedding zero-shot + human docs
    signals: list[Signal] = field(default_factory=list)
    # Optional structural scorer over computed features (tables, field density…).
    structural: Callable[[dict], float] | None = None


@dataclass
class DocTypeResult:
    label: str
    confidence: float  # 0..1 (top normalized score)
    margin: float  # gap to the runner-up (decisiveness)
    scores: dict[str, float]  # per-type normalized score
    matched: dict[str, list[str]]  # type -> signal names that fired
    method: str  # "rules" | "rules+embed" | "vlm"

    def explain(self) -> str:
        fired = ", ".join(self.matched.get(self.label, [])) or "(no strong signals)"
        return f"{self.label} (conf {self.confidence:.2f}, margin {self.margin:.2f}) — {fired}"


# --------------------------------------------------------------------------- #
# structural feature extraction (cheap, from text)
# --------------------------------------------------------------------------- #
_FIELD_LINE = re.compile(r"^\s*[A-Z][A-Za-z0-9 /()'.\-]{1,34}:\s*\S", re.MULTILINE)
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$", re.MULTILINE)
_CHECKBOX = re.compile(r"(\[[ xX]\]|☐|☑|□|■|\bYes\s*/\s*No\b)")


def document_features(text: str) -> dict:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    n = max(len(lines), 1)
    field_lines = len(_FIELD_LINE.findall(text))
    return {
        "len": len(text),
        "lines": n,
        "field_density": field_lines / n,  # "Label: value" density → forms
        "field_lines": field_lines,
        "tables": len(_TABLE_SEP.findall(text)),
        "checkboxes": len(_CHECKBOX.findall(text)),
    }


# --------------------------------------------------------------------------- #
# layout signature — content-free "looks like this" structure
# --------------------------------------------------------------------------- #
# The order is the vector order used by layout_distance/layout_similarity. Every
# component is normalized to roughly [0, 1] so a plain Euclidean distance is a
# fair shape comparison: a blank intake form and a filled one share a signature,
# while a narrative case note does not — *without* looking at a single word, so
# this is safe to compute on confidential records (no content crosses anywhere).
_LAYOUT_KEYS = (
    "field_density", "checkbox_density", "table_density",
    "avg_line_len", "short_line_ratio", "allcaps_ratio",
)


def layout_signature(text: str) -> dict:
    """Content-free structural fingerprint of a document's *shape*.

    Captures what distinguishes a form from a narrative from a table-heavy
    record — field/label density, checkbox and table-rule density, average and
    short-line ratios, all-caps heading ratio. Carries no words, so it can be
    compared across cases (or promoted through the de-id gate) without leaking
    content. Used by ``DocIndex.find_similar`` to rank "documents that look like
    this form".
    """
    feats = document_features(text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    n = max(len(lines), 1)
    avg_line_len = sum(len(ln) for ln in lines) / n
    short = sum(1 for ln in lines if len(ln) <= 40) / n
    allcaps = sum(1 for ln in lines if len(ln) > 3 and ln.upper() == ln and any(c.isalpha() for c in ln)) / n
    return {
        "field_density": round(min(feats["field_density"], 1.0), 4),
        "checkbox_density": round(min(feats["checkboxes"] / n, 1.0), 4),
        "table_density": round(min(feats["tables"] / n, 1.0), 4),
        # Normalize line length against ~120 cols; long prose lines saturate at 1.
        "avg_line_len": round(min(avg_line_len / 120.0, 1.0), 4),
        "short_line_ratio": round(short, 4),
        "allcaps_ratio": round(allcaps, 4),
    }


def layout_vector(sig: dict) -> list[float]:
    """The signature as a fixed-order vector (missing keys → 0.0)."""
    return [float(sig.get(k, 0.0)) for k in _LAYOUT_KEYS]


def layout_similarity(a: dict, b: dict) -> float:
    """Shape similarity in [0, 1] (1 = identical layout). Euclidean distance over
    the normalized signature, mapped through 1/(1+d)."""
    va, vb = layout_vector(a), layout_vector(b)
    dist = sum((x - y) ** 2 for x, y in zip(va, vb)) ** 0.5
    return 1.0 / (1.0 + dist)


# --------------------------------------------------------------------------- #
# default child-welfare taxonomy
# --------------------------------------------------------------------------- #
def _S(name: str, pattern: str, w: float = 1.0) -> Signal:
    return Signal(name, pattern, w)


DEFAULT_TAXONOMY: list[TypeSpec] = [
    TypeSpec(
        "court_document",
        "A legal/court filing or order in a child welfare case: petition, "
        "dependency or custody order, hearing notice, case caption.",
        [
            _S("caption", r"in the (matter|interest) of", 2.0),
            _S("court", r"\b(juvenile|family|district|superior) court\b", 1.5),
            _S("order", r"\b(it is (hereby )?ordered|the court (finds|orders))\b", 1.5),
            _S("petition", r"\bpetition(er)?\b", 1.0),
            _S("docket", r"\b(case|docket) no\.?\b", 1.0),
            _S("hearing", r"\bhearing\b", 0.5),
            _S("judge", r"\b(judge|magistrate|honorable)\b", 0.8),
            _S("vs", r"\bv(?:s)?\.\s", 0.8),
        ],
    ),
    TypeSpec(
        "expert_report",
        "A retained expert's / forensic report prepared for litigation: states the "
        "expert's qualifications and CV, the opinions offered and their bases, the "
        "materials reviewed, and compensation — distinct from the court's own filing.",
        [
            # Patterns use \s+ between words so they survive line wraps in the
            # extracted text, and accept plurals ("Expert Reports", "opinions").
            # Decisive front-matter title: a document headed "Expert Report of
            # <name>" is definitively one, even when it also names the case
            # ("...in the Matter of...", which otherwise reads as a court caption).
            _S("title", r"\bexpert\s+report\s+of\b", 2.5),
            _S("expert", r"\bexpert\s+(reports?|witness(es)?|opinions?|disclosures?|qualifications)\b", 2.2),
            _S("opinion", r"\b((it is |in )?my\s+(expert\s+)?opinions?|opinions?\s+(are\s+)?(offered|expressed|stated|rendered))\b", 1.4),
            _S("certainty", r"\breasonable\s+degree\s+of\b", 1.8),
            _S("materials", r"\bmaterials\s+(reviewed|relied\s+upon|considered)\b", 1.8),
            _S("cv", r"\bcurriculum\s+vitae\b", 1.2),
            _S("qualifications", r"\bqualifications\b", 0.7),
            _S("retained", r"\bretained\s+(on\s+behalf\s+of|by|to|as)\b", 1.2),
            _S("compensation", r"\b(rate\s+of\s+compensation|compensat\w*|hourly\s+rate|being\s+compensated)\b", 0.8),
            # Litigation context shared with court_document; weak on its own so it
            # only tips already-expert-leaning docs, not bare court filings.
            _S("litigation", r"\b(deposition|at\s+trial|in\s+this\s+(matter|litigation)|plaintiff|defendant)\b", 0.5),
        ],
    ),
    TypeSpec(
        "case_note",
        "A caseworker's progress/contact note documenting a visit, contact, or "
        "case activity, usually dated and narrative.",
        [
            _S("casenote", r"\b(case|progress|contact|running|case activity) note", 2.0),
            _S("caseworker", r"\b(case ?worker|social worker|cw|caseworker)\b", 1.0),
            _S("visit", r"\b(home visit|face[- ]to[- ]face|contact made|visited)\b", 1.0),
            _S("narrative_date", r"^\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", 0.6),
            _S("client", r"\b(client|family|the (child|youth|mother|father))\b", 0.4),
        ],
    ),
    TypeSpec(
        "cps_investigation",
        "A Child Protective Services investigation/disposition narrative: the "
        "agency's findings on an abuse/neglect allegation — the evidentiary "
        "standard applied, risk level, and the Central Registry / category "
        "disposition. (A specific kind of child-welfare case narrative.)",
        [
            # Content hallmarks of a CPS disposition — none of these is the
            # document's own label; together they are near-unique to this type.
            _S("preponderance", r"\bpreponderance\s+of\s+(the\s+)?evidence\b", 2.2),
            _S("registry", r"\bcentral\s+registry\b", 2.0),
            _S("disposition", r"\bdisposition(s|ed)?\b", 1.2),
            _S("category", r"\b(cat(egory)?\.?\s*(i{1,3}|iv|v|[1-5])\b|substantiat|unsubstantiat|unfounded)", 1.4),
            _S("allegation", r"\ballegations?\s+(of|that|was|were|is|are)\b", 1.0),
            _S("forensic", r"\bforensic(ally)?\s+interview", 1.2),
            _S("maltreatment", r"\b(child protective|cps|maltreatment|perpetrator|safety (plan|threat|factor))\b", 1.0),
            _S("risk_level", r"\b(risk|safety)\s+level\b", 0.7),
        ],
    ),
    TypeSpec(
        "medical_record",
        "A clinical/medical record: diagnosis, medications, provider notes, "
        "immunizations, or a health/physical exam.",
        [
            _S("diagnosis", r"\b(diagnosis|assessment/plan|chief complaint|icd-?10)\b", 1.8),
            _S("medication", r"\b(medication|prescri|dosage|mg\b|allerg(y|ies))\b", 1.2),
            _S("provider", r"\b(physician|provider|md\b|rn\b|clinic|pediatric)\b", 1.0),
            _S("vitals", r"\b(blood pressure|bp\b|heart rate|temperature|vitals|height|weight)\b", 1.0),
            _S("mrn", r"\b(mrn|medical record (no|number)|patient id)\b", 1.2),
            _S("immun", r"\b(immuniz|vaccin)\w*", 0.8),
        ],
    ),
    TypeSpec(
        "assessment",
        "A formal assessment or evaluation: psychological, risk, safety, family "
        "or developmental, with findings and recommendations.",
        [
            _S("assessment", r"\b(assessment|evaluation|screening)\b", 1.5),
            _S("risk_safety", r"\b(risk|safety) (assessment|factors|level)\b", 1.8),
            _S("findings", r"\b(findings|results|conclusions?)\b", 0.8),
            _S("recommend", r"\brecommendations?\b", 1.0),
            _S("scale", r"\b(score|scale|instrument|standardized)\b", 0.8),
            _S("clinician", r"\b(psychologist|clinician|evaluator|therapist)\b", 1.0),
        ],
    ),
    TypeSpec(
        "service_plan",
        "A case/service/permanency plan: goals, objectives, services, and "
        "responsible parties for a family.",
        [
            _S("plan", r"\b(service|case|permanency|safety|treatment) plan\b", 2.0),
            _S("goals", r"\b(goals?|objectives?|tasks?|action steps?)\b", 1.0),
            _S("permanency", r"\b(reunification|permanency|placement goal)\b", 1.2),
            _S("services", r"\b(services? (provided|referred)|service provider)\b", 0.8),
        ],
    ),
    TypeSpec(
        "consent_form",
        "A consent, authorization, or release form (e.g. release of information, "
        "consent to treat), typically signed.",
        [
            _S("consent", r"\b(consent|authoriz|release of information|roi)\b", 1.8),
            _S("hereby", r"\bi (hereby )?(authorize|consent|give permission)\b", 2.0),
            _S("signature", r"\b(signature|signed|/s/|date signed)\b", 1.0),
            _S("revoke", r"\b(revoke|withdraw (this )?consent)\b", 0.8),
        ],
    ),
    TypeSpec(
        "educational_record",
        "A school/education record: report card, IEP, attendance, enrollment, or "
        "school correspondence about a child.",
        [
            _S("school", r"\b(school|district|teacher|principal|classroom)\b", 1.0),
            _S("iep", r"\b(iep|individualized education|504 plan|special education)\b", 2.0),
            _S("report_card", r"\b(report card|grade level|gpa|transcript)\b", 1.5),
            _S("attendance", r"\b(attendance|absences|enrollment)\b", 1.0),
        ],
    ),
    TypeSpec(
        "correspondence",
        "A letter, email, or memo: salutation and sign-off, addressed to a person.",
        [
            _S("salutation", r"^\s*(dear|to whom it may concern)\b", 1.8),
            _S("signoff", r"\b(sincerely|regards|respectfully|yours truly)\b", 1.5),
            _S("memo", r"^\s*(memorandum|memo|re:|subject:)\b", 1.0),
            _S("email", r"^\s*(from:|to:|sent:|cc:)\b", 1.0),
        ],
    ),
    TypeSpec(
        "administrative_letter",
        "An agency policy issuance to staff — an administrative letter, change "
        "notice, or policy memo addressed to county directors/child-welfare "
        "staff, with subject/effective-date headers. Guidance, not a case record.",
        [
            _S("admin_letter", r"\b(administrative letter|dear colleague letter|policy memo(randum)?)\b", 2.4),
            _S("change_notice", r"\bchange notice( for manual)?\b", 2.2),
            _S("to_directors", r"\b(to:\s*)?(all\s+)?county directors of social services\b", 1.6),
            _S("memo_headers", r"^\s*(subject|effective date|attention|to|from|date)\s*:", 0.6),
            _S("issuer", r"\b(division of social services|family services manual|child welfare services section)\b", 0.8),
        ],
    ),
    TypeSpec(
        "policy_manual",
        "Policy/procedure reference material — a manual, handbook, appendix, "
        "funding guidance, or resource/desk guide — rather than a case record.",
        [
            _S("manual", r"\b(policy(\s+and\s+procedures?)?\s+manual|procedures?\s+manual|operations?\s+manual|implementation\s+manual|family\s+services\s+manual|funding\s+manual|reference\s+manual|user'?s?\s+guide|desk\s+guide|resource\s+guide|practice\s+guide)\b", 2.2),
            # A type keyword in a Markdown heading is a strong prior that the
            # document *is* that type (vs. merely discussing the topic in body
            # text) — this is what keeps reference manuals from being out-voted
            # by the case-work vocabulary they describe.
            _S("title_manual", r"(?m)^#{1,6}\s+.*\b(manual|appendix|funding|handbook)\b", 2.0),
            _S("appendix", r"\bappendix\s+[0-9ivx]", 1.6),
            _S("sdm", r"\bstructured\s+decision\s+making\b", 1.2),
            _S("reference_phrase", r"\b(this\s+manual|this\s+appendix|policy\s+and\s+procedures?)\b", 1.0),
            _S("toc", r"\btable\s+of\s+contents\b", 0.9),
            _S("structure_ref", r"\b(chapter\s+[0-9ivx]+|section\s+\d{3,4})\b", 0.7),
            _S("manual_word", r"\bmanual\b", 0.5),
        ],
    ),
    TypeSpec(
        "referral_intake",
        "A referral or intake document opening a case or service: reason for "
        "referral, intake details, reporter information.",
        [
            _S("referral", r"\b(referral|reason for referral|referred (by|for))\b", 1.8),
            _S("intake", r"\b(intake|screening (decision|in/out)|report (received|date))\b", 1.5),
            _S("reporter", r"\b(reporter|mandated reporter|allegation)\b", 1.0),
        ],
    ),
    TypeSpec(
        "identity_document",
        "An identity or vital record: birth certificate, ID card, social security.",
        [
            _S("birth_cert", r"\b(certificate of (live )?birth|birth certificate)\b", 2.0),
            _S("ssn", r"\b(social security (no|number|card)|ssn)\b", 1.2),
            _S("id_card", r"\b(identification card|driver'?s license|state id)\b", 1.0),
        ],
    ),
    TypeSpec(
        "standardized_form",
        "A structured fillable form: dense label:value fields, sections, "
        "checkboxes — type when no stronger content signal dominates.",
        [
            _S("form_word", r"\bform\b(\s+(no|number|#))?", 0.8),
            _S("section", r"^\s*(section|part)\s+[A-Z0-9]", 0.6),
        ],
        structural=lambda f: (
            1.6 * min(f["field_density"] * 4, 1.0)
            + (0.8 if f["checkboxes"] >= 3 else 0.0)
            + (0.5 if f["tables"] >= 1 else 0.0)
        ),
    ),
]

OTHER = "other"


# --------------------------------------------------------------------------- #
# classifier
# --------------------------------------------------------------------------- #
class DocTypeClassifier:
    def __init__(
        self,
        taxonomy: list[TypeSpec] | None = None,
        *,
        embedder=None,
        embed_weight: float = 0.6,
        min_confidence: float = 0.34,
        min_margin: float = 0.08,
    ):
        self.taxonomy = taxonomy or DEFAULT_TAXONOMY
        self.embedder = embedder
        self.embed_weight = embed_weight
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self._type_vecs: dict[str, list[float]] | None = None

    # -- rule scoring ---------------------------------------------------------
    def _rule_scores(self, text: str) -> tuple[dict[str, float], dict[str, list[str]]]:
        feats = document_features(text)
        scores: dict[str, float] = {}
        matched: dict[str, list[str]] = {}
        for spec in self.taxonomy:
            s = 0.0
            hits: list[str] = []
            for sig in spec.signals:
                if sig.compiled().search(text):
                    s += sig.weight
                    hits.append(sig.name)
            if spec.structural:
                bonus = spec.structural(feats)
                if bonus:
                    s += bonus
                    if bonus >= 0.5:
                        hits.append("structure")
            scores[spec.name] = s
            matched[spec.name] = hits
        return scores, matched

    # -- embedding (optional, zero-shot) -------------------------------------
    def _embed_type_vectors(self) -> dict[str, list[float]]:
        if self._type_vecs is None:
            vecs = self.embedder.embed(
                [f"{t.name.replace('_', ' ')}: {t.description}" for t in self.taxonomy],
                is_query=False,
            )
            self._type_vecs = {t.name: v for t, v in zip(self.taxonomy, vecs)}
        return self._type_vecs

    def _embed_scores(self, text: str) -> dict[str, float]:
        from .embeddings import cosine

        qv = self.embedder.embed_one(text[:2000], is_query=True)
        tv = self._embed_type_vectors()
        return {name: max(cosine(qv, v), 0.0) for name, v in tv.items()}

    # -- main -----------------------------------------------------------------
    def classify(self, text: str, *, vlm=None) -> DocTypeResult:
        if not text or not text.strip():
            return DocTypeResult(OTHER, 0.0, 0.0, {}, {}, "rules")

        rule, matched = self._rule_scores(text)
        method = "rules"
        combined = dict(rule)

        if self.embedder is not None:
            emb = self._embed_scores(text)
            # Blend normalized rule scores with embedding similarity.
            rnorm = _normalize(rule)
            enorm = _normalize(emb)
            combined = {
                name: (1 - self.embed_weight) * rnorm.get(name, 0.0)
                + self.embed_weight * enorm.get(name, 0.0)
                for name in rule
            }
            method = "rules+embed"

        probs = _normalize(combined)  # full distribution, retained for display
        label, _, _ = _top(probs)
        # Gate on dominance over the nearest rivals, not the full signal mass:
        # long, multi-topic documents trip many incidental low-weight signals
        # across unrelated types, which would otherwise dilute a clear winner's
        # confidence below threshold. Comparing against the top few competitors
        # keeps the decision length-robust while staying conservative for ties.
        confidence, margin = _topk_confidence(combined)
        ambiguous = confidence < self.min_confidence or margin < self.min_margin

        # Escalate genuinely ambiguous documents to the local VLM, if provided.
        # The VLM is the disambiguator, so its verdict is returned directly.
        if vlm is not None and ambiguous:
            v = self._vlm_classify(text, vlm)
            if v is not None:
                return DocTypeResult(v, max(round(confidence, 3), self.min_confidence),
                                     round(margin, 3), _round(probs), matched, "vlm")

        if ambiguous:
            # Keep the evidence, but do not assert a type a human should confirm.
            return DocTypeResult(OTHER, round(confidence, 3), round(margin, 3),
                                 _round(probs), matched, method)
        return DocTypeResult(label, round(confidence, 3), round(margin, 3),
                             _round(probs), matched, method)

    def _vlm_classify(self, text: str, vlm) -> str | None:
        labels = [t.name for t in self.taxonomy] + [OTHER]
        prompt = (
            "Classify this child-welfare document into exactly one type from this "
            "list (reply with only the label):\n" + ", ".join(labels) + "\n\n"
            "Document excerpt:\n" + text[:3000]
        )
        try:
            # VLMExtractor exposes a text chat via a 1x1 image-free path is not
            # available; callers pass an object with .classify_text(prompt). The
            # cascade's VLM client can be adapted; kept behind a tiny duck-type.
            reply = vlm.classify_text(prompt) if hasattr(vlm, "classify_text") else None
        except Exception:
            return None
        if not reply:
            return None
        reply = reply.strip().lower()
        for lbl in labels:
            if lbl in reply:
                return lbl
        return None


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    total = sum(v for v in scores.values() if v > 0)
    if total <= 0:
        return {k: 0.0 for k in scores}
    return {k: (v / total if v > 0 else 0.0) for k, v in scores.items()}


def _topk_confidence(scores: dict[str, float], k: int = 4) -> tuple[float, float]:
    """Confidence/margin of the top type relative to its k strongest rivals.

    Normalizing over only the top-k positive scores (rather than every type that
    fired) makes the decision robust to document length: a long document that
    incidentally trips many weak, unrelated signals no longer dilutes a clearly
    dominant type. For short, single-purpose documents the long tail is empty, so
    this reduces to the familiar near-1.0 confidence.
    """
    pos = sorted((v for v in scores.values() if v > 0), reverse=True)
    if not pos:
        return 0.0, 0.0
    denom = sum(pos[:k])
    top = pos[0]
    second = pos[1] if len(pos) > 1 else 0.0
    return top / denom, (top - second) / denom


def _top(probs: dict[str, float]) -> tuple[str, float, float]:
    if not probs:
        return OTHER, 0.0, 0.0
    ranked = sorted(probs.items(), key=lambda x: -x[1])
    label, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    return label, top, top - second


def _round(d: dict[str, float]) -> dict[str, float]:
    return {k: round(v, 3) for k, v in sorted(d.items(), key=lambda x: -x[1]) if v > 0}


def classify_document(text: str, *, embedder=None, taxonomy=None) -> DocTypeResult:
    """Convenience: classify a document's text with the default taxonomy."""
    return DocTypeClassifier(taxonomy, embedder=embedder).classify(text)
