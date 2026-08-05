# Michigan Court Rules — parsing proof of concept

An isolated, high-accuracy parse of the **Michigan Court Rules** (874-page
born-digital FrameMaker PDF, updated July 31 2026) into a citable corpus.

Kept deliberately separate from the bench-book library: this is the reference
build, and its correctness standard is higher than the pipeline it may later
feed.

## Why this document is parseable exactly

It is born-digital — 873 of 874 pages carry a text layer and there are **zero
images**, so no OCR runs and no OCR error can enter. Every word carries exact
geometry and font, so the structure is *measured*, not guessed.

## The measured model

| element | style | x |
|---|---|---|
| Chapter | Bold 18, centred | ~190 |
| Subchapter | Bold 14 (wraps) | 72 |
| Rule | Bold 12 (wraps) | 72 |
| footer | size 10, y > 730 | stripped |

Indent ladder, +21.6pt per level:
`93.6 (A)` · `115.2 (1)` · `136.8 (a)` · `158.4 (i)` · `180.0` deeper.
`108.0` is a flush paragraph under a rule with no subrules.

Line spacing is bimodal — **14.0pt within a paragraph, 20.0pt between** — so
block boundaries are read off the document rather than inferred.

A continuation line sits at its parent's indent + 21.6, which is also the next
level's item indent. **Indent alone therefore cannot distinguish a continuation
from a new item**; the marker text and the vertical gap settle it.

## Two independent correctness proofs

**1. Structure — reconciled against the document's own table of contents.**
The 18-page printed contents names every chapter, subchapter and rule. It is
parsed by separate code reading different pages, then compared:

```
contents : chapters 9  subchapters 51  rules 625
body     : chapters 9  subchapters 51  rules 625
0 errors, 0 warnings
```

This exists because of a prior failure: 17 of 18 bench books were built,
evaluated and published with their section structure silently flattened. Every
gate passed, because every gate asked *"is the corpus a faithful rendering"* —
and it was. Nothing asked *"did we find the sections the document says it
has"*. A defect must now survive two parsers reading different pages in
different formats to go unnoticed.

**2. Completeness — word-multiset equality against the raw PDF.**
384,507 source words in, 384,507 out. Zero lost, zero invented.

## Defects this parse found and fixed

Each was invisible on inspection — the output read as plausible legal English:

| defect | scale |
|---|---|
| Wrapped subchapter titles became phantom subchapters (`Motions`, `Actions`) | 5 |
| Wrapped rule titles truncated, remainder leaked into body prose | 16 |
| **Markers with no trailing space absorbed as body text, mis-citing everything beneath** | **195** |
| TOC entries whose page number lands on its own line, or after a plain space | 5 |

The marker defect is the one that mattered most: `(10)Request for Copy...` has
no space after the label, so item `(10)` was never recognised and its children
inherited `(9)`'s citation path. In a corpus whose purpose is to answer *"what
does MCR 1.109(D)(10) require"*, that is a wrong answer delivered confidently.

## Citation paths are the point

Lawyers cite `MCR 2.116(C)(10)`, not "page 312". Every block carries the full
path it sits under, so a chunk can be cited and a query naming a subrule can
be matched to it.

## The retrieval system

Settled by a systematic sweep over 1,060 paired evaluation queries, every
comparison tested with McNemar on discordant pairs.

| | |
|---|---|
| chunks | rule-scoped, 256 tokens, plain text — 3,580 passages |
| embedder | Qwen3-Embedding-4B (2000d indexes losslessly, p = 0.80) |
| retrieval | **dense only**, citation router in front |
| answering | deepseek-v4-flash, cross-reference expansion |

**R@1 0.660 · R@10 0.947 · R@2048tok 0.944 · citation lookups 1.000**
**citation validity 1.000 · cites gold when gold retrieved 0.957**

### Eight priors carried from the CPS bake-off, eight corrected

| carried over | measured here |
|---|---|
| 512-token chunks, size a tie | 256 wins, p = 2e-05 |
| citation-path prefix, worth ~32 pts | **0.000**, p = 0.78 |
| parent stem | 0.000 |
| rule-level dedup | **−12 to −14 pts** |
| structure-aware chunking dominates | +0.022 overall, n.s. |
| hybrid retrieval | dense-only, **+15.6 pts** |
| reranking is the top lever | **−8.9 pts**, p = 1e-16 |
| HyDE closes the register gap | **−6.7 pts**, p = 1e-11 |

Every surviving optimisation was a **deletion**. The only two additions that
earned their place are the citation router (+11.5 pts) and cross-reference
expansion (answer quality only, directional).

The failures split on one line. Findings tied to **where the gold sits** did
not transfer — CPS scored at section level, this at provision level, and the
prefix, the stem and boundary-respect all discriminate at the rung that no
longer matters (342 rules split across chunks; not one pair shares a heading
path). Findings orthogonal to granularity transferred **exactly** — dense-only
over hybrid, and rerankers hurting strong retrieval. Both of those I overrode,
and both overrides cost points.

*The rule for the next manual: ask whether a finding depends on where your
gold sits. If it does, re-measure. If it does not, trust it.*

## Provenance

Every answer is falsifiable by the reader without trusting the system's
account of itself:

```
answer sentence -> citation      audited against 11,860 parsed citations
citation        -> chunk         deterministic; what the generator read
chunk           -> block ids     the parse's atoms
block           -> printed page  from PDF geometry
page            -> source sha256 the PDF the parse ran on
retrieval route  dense | citation-router | cross-reference, labelled
                 because they are different kinds of evidence
```

A passage that arrived by exact citation lookup is not the same evidence as
one that arrived by similarity, and the interface says which.

## Run

```bash
python3 pipeline/parse_mcr.py 0_source/michigan-court-rules.pdf -o 1_parsed
python3 pipeline/verify_structure.py 0_source/michigan-court-rules.pdf 1_parsed/blocks.jsonl
```

Requires PyMuPDF. The source PDF is not committed; fetch it from
`courts.michigan.gov`.
