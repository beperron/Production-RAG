# NC Child-Welfare Law & Policy Search — Quality Report

*Plain-language summary of measured retrieval quality and coverage.*

**Scope: public North Carolina statutes and agency policy only.** This tool
never accesses confidential case records; that separation is enforced in code
(`parsevault.lawsearch` refuses any confidential / private workspace).

---

## 1. What the tool does, in plain terms

It searches two public bodies of North Carolina child-welfare authority:

- **The law** — NC General Statutes, Chapter 7B (the Juvenile Code: abuse,
  neglect, dependency, and termination of parental rights), plus the NC
  Administrative Code.
- **The policy** — the NCDHHS child-welfare manuals, protocols, forms, and
  guidance that county agencies are expected to follow.

For a plain-English question it returns the governing statute section or policy
passage, each with a **source-traceable citation** (section number, page, and a
link to the official source). It can also produce a short **written answer** that
is **grounded** — every statement drawn from and cited to the retrieved sources,
and if the sources don't support an answer it says so rather than inventing one.

**Pipeline:** hybrid BM25F + GTE-base semantic retrieval per collection →
reciprocal-rank fusion across collections → Jina reranker blended with the fused
rank (α = 0.5). Grounded answers use a query rewrite (qwen3-8b) plus a cited
generator (deepseek-v4-flash), falling back to the local Qwen3.6 model. The cloud
stages are used **only here**, on public data, and degrade to the local engine
when no key is present.

## 2. How the quality was measured

We wrote test questions whose correct answer is known in advance (e.g. "what are
the grounds for terminating parental rights?" → `§ 7B-1111`), then measured how
reliably the tool surfaces that authority near the top of its results.

- **Top-3 accuracy (recall@3)** — how often the correct authority is in the first
  three results. The practical bar: a reader scanning three results finds it.
- **Top-1 accuracy (recall@1)** — how often the very first result is the exact
  governing authority.
- **MRR** — a 0–1 score, higher when the correct result sits nearer the top.

Test sets: **42 statute questions** and **28 policy questions**, each mapped to
the document(s) that correctly answer it. For policy, a question is credited when
*any* of the legitimately-correct documents on that topic is returned (several
policy documents often cover one subject); for statutes, only the exact governing
section counts. Gold sets and the eval harness live in
[`evals/law_search/`](../evals/law_search/) and are reproducible.

## 3. Results

| Body of authority        | Top-1 | Top-3 | Top-5 | MRR  |
|--------------------------|:-----:|:-----:|:-----:|:----:|
| Statutes (GS Chapter 7B) |  66%  | **90%** |  92%  | 0.79 |
| Policy (NCDHHS manuals)  |  89%  | **96%** | **100%** | 0.94 |

**In plain terms:** for the statutes, the governing section is in the top three
results **90%** of the time. For policy, the correct document is in the top three
**96%** of the time and in the top five **100%** of the time. The written-answer
feature was verified to produce correct, cited summaries — for example, it
correctly enumerated the statutory grounds for terminating parental rights and
cited the specific passages — and to **decline** when the retrieved material is
insufficient rather than guess.

### Honest limitations

- **Statute near-misses are between adjacent sections.** When the exact section
  isn't in the top three, the results above it are closely-related provisions —
  e.g. a question about the *answer* to a termination petition (`§ 7B-1108`)
  returns the immediately preceding *failure-to-answer* section (`§ 7B-1107`)
  first; a *modify/vacate* question returns the parallel delinquency-side
  "authority to modify or vacate" section first. These are genuine fine
  distinctions, not random errors, and they are the main reason statute top-1 is
  ~two-thirds rather than higher.
- **Curated statute corpus.** `legal-authorities` indexes a selected set of NC
  statutes (Chapter 7B core plus supporting sections), not the entire General
  Statutes. Questions outside that set will not resolve.
- **One policy near-miss** (adoption-assistance vendor agreement) returns the
  correct material at rank 4 — just outside the top three, inside the top five.
- **It is a research aid, not legal advice or authority.** Every result links to
  the official source, which remains controlling. A qualified reviewer should
  confirm currency and applicability.
- The benchmark is a starter set (70 questions); a larger, expert-reviewed set
  would sharpen these figures.

## 4. Domain areas relevant to an expert-witness review of quality of care

The corpus covers the North Carolina standards that define what adequate care and
casework are supposed to look like. The areas below are the ones an expert
evaluating **quality of care** would typically examine; each is paired with the
NC authority the tool retrieves. This is a summary of coverage, not a
recommendation.

| Quality-of-care issue | Why it bears on quality of care | NC authority the tool surfaces |
|---|---|---|
| **Safety & maltreatment assessment** | Whether risk was identified and acted on — the threshold question in most quality-of-care disputes. | GS § 7B-101 (definitions of abused/neglected/dependent), § 7B-302 (assessment), § 7B-303 (interference with assessment); DHHS *Assessing Safety and Risk*, *CPS Family & Investigative Assessments*, *Safety Planning*, *SDM/CPS Intake*. |
| **Duty to report & respond** | Whether mandated reports were made and screened appropriately. | GS § 7B-301 (duty to report); DHHS *CPS Intake Policy*. |
| **Reasonable efforts** | Whether the agency did enough to prevent removal and to reunify — a recurring quality-of-care and legal standard. | GS § 7B-507, § 7B-901; DHHS family-services / case-plan policy. |
| **Placement appropriateness** | Least-restrictive, relative-first, sibling, and stability considerations directly shape a child's care. | GS § 7B-503/505 (nonsecure custody placement), § 7B-903 (dispositional placement), § 7B-903.1 (duties while in DSS custody). |
| **Worker contact & oversight** | Frequency and quality of caseworker contact is a core measure of whether care was actually monitored. | DHHS *Required Contacts for In-Home Services*; *In-Home Services Policy*; *Review of Services*. |
| **Permanency & timeliness** | Delay in achieving permanency is itself a quality-of-care harm. | GS § 7B-906.1 (permanency planning hearings), § 7B-906.2 (permanent plans, concurrent planning), § 7B-908/909 (post-TPR review); DHHS *Permanency Planning Services Policy*. |
| **Family time / visitation** | Maintaining the parent–child and sibling relationship is a measured element of care. | GS § 7B-905.1 (visitation). |
| **Health & normalcy** | Medical/dental/behavioral-health access and age-appropriate "normalcy" activities for youth in care. | DHHS *Reasonable & Prudent Parent Standard*, *Pregnancy Services*, *Authentic Youth Engagement*, *National Youth in Transition Database*. |
| **Substance exposure / plan of safe care** | Response to prenatal or environmental substance exposure is a distinct quality-of-care obligation. | DHHS *Perinatal Substance Use*, *Substance Affected Infants & Plan of Safe Care*, *Drug Endangered Children*. |
| **Domestic violence & home safety** | Household risk factors that bear directly on a child's safety and care. | DHHS *Children's Domestic Violence Assessment Tool*, *Firearm Safety Guidance*, *Safe Sleep Practice Guide*, *Unsafe Discipline vs. Physical Abuse*. |
| **Serious injury & fatality review** | How the agency reviews the most serious care failures. | DHHS *State Child Fatality and Near-Fatality Review Policy*. |
| **Legal process & oversight** | Whether the hearings, findings, and reviews that protect the child occurred on time. | GS § 7B-405 (commencement), § 7B-500 (temporary custody), § 7B-506 (custody hearing), § 7B-801/802/805 (adjudication), § 7B-905 (dispositional order), § 7B-1000 (modification), § 7B-1001 (appeal), § 7B-601 (guardian ad litem). |
| **Termination & guardianship** | The standards governing the most permanent decisions about a child's care. | GS § 7B-1101/1103/1104/1108/1109/1110/1111/1112/1114; DHHS *KinGAP* guardianship-assistance policy. |

> **Boundary.** This tool and this report address only the **public standards** —
> what North Carolina law and policy require. They do not contain, and cannot
> speak to, any specific child's record. Applying these standards to a particular
> case is the province of the qualified reviewer.

---

*Prepared from an automated retrieval benchmark. Metrics reflect the starter test
sets described in §2 (42 statute + 28 policy questions) and will change as the
question sets are expanded and expert-reviewed. Reproduce with*
`python evals/law_search/unified_eval.py`.
