# Testing record and lessons learned

Every number in this document was measured on this corpus this session, is
paired where a comparison is claimed, and can be recomputed from the files
named. Nothing is quoted from another project without saying so.

The one-line summary of the whole record: **thirteen mechanisms were tested;
ten of the author's own recommendations were wrong; every surviving
optimisation was a deletion or a measurement-forced choice.** The system is
good because the process assumed the author would be wrong, and made being
wrong cheap.

---

## 1. The parse (`pipeline/parse_mcr.py`, `1_parsed/`)

874-page born-digital FrameMaker PDF → 12,298 blocks → 11,860 citable
provisions across 625 rules.

**Verification — three independent proofs, all green:**

| proof | method | result |
|---|---|---|
| structure | reconcile against the document's own 18-page printed contents, parsed by separate code (`verify_structure.py`) | 625/625 rules, 51/51 subchapters, 9/9 chapters |
| completeness | word-multiset equality with the raw PDF | 384,507 = 384,507 |
| character fidelity | independent re-extraction, non-whitespace multiset | 1,886,132 = 1,886,132 |

**Adversarial audits:** two swarm rounds (48 + 21 agents), every finding
handed to a separate agent instructed to refute it. 43 findings survived and
were fixed. The worst: 195 markers with no trailing space silently absorbed
as body text (mis-citing everything beneath), 93 wrapped cross-references
torn off as phantom subrules (92 fabricated citations), 397 closing
paragraphs cited to a sibling instead of their parent.

**Lesson — presence is not attribution.** Both original gates (structure,
completeness) stayed green while 2.4% of blocks carried citations the
document does not support. A check that answers a real question can sit
exactly one level above where the defect lives. The fix each time was a
check at the level the claim is made: sub-rule attribution, sequence
validation, page-footer verification.

**Lesson — the instrument reports its own frame.** The off-by-one in printed
page numbers was caught only by opening the PDF and reading the footer
("Page 44") against the arithmetic (which said 45). When a number can be
checked against the physical artifact, check it there.

## 2. The evaluation set (`pipeline/gen_arm_*.py`, `2_eval/`)

1,092 queries: 513 Claude-written (validated by GLM), 454 GLM-written
(validated by Claude), 125 deterministic citation lookups. Gold is a
**citation string**, never a block id or position, so the set survives any
reparse — the failure that slid 394 of 1,722 labels in the bench-book work.

- Cross-validation was asymmetric: Claude rejected 44% of GLM's items, GLM
  20% of Claude's — undecidable from model evidence alone; the 120-item human
  review (`2_eval/REVIEW.html`) is weighted toward the disputed items.
- Regeneration fed the checker's specific complaint back to the original
  generator: items that had all failed passed at 57–66% on rewrite, and echo
  collapsed 0.19 → 0.09.
- Two harness bugs were charged to the questions before being found in the
  harness: validators shown a 40-character rule *title* as the whole
  provision (16 wrongful rejections), and 189 stale co-golds inherited by
  rewritten queries (each would have credited a wrong retrieval).
- 115 groups of byte-identical provisions (244 citations) are scored as
  mutual co-gold — no retriever can separate identical bytes.

**Lesson — the eval's difficulty is what made everything else measurable.**
Echo (query↔gold word overlap) was held to median ~0.18 by construction. An
easier set would have shown every dead mechanism below "working."

## 3. Retrieval (`pipeline/run_sweep.py`, `4_eval/results.jsonl`, `sweep_final.json`)

Embedder fixed by decision: Qwen3-Embedding-4B. 1,060 scoreable queries,
every comparison paired McNemar.

**Shipped: rule-scoped 256-token chunks · dense only · citation router.**
R@1 0.660 · R@10 0.947 · R@2048tok 0.944 · citation lookups 125/125.

**The metric inversion.** R@1 tracks citations-per-chunk almost exactly
(1.0→0.432 … 18.5→0.564): a chunk holding N citations satisfies
gold-containment for any of N provisions, so coarse chunks score better
without retrieving better. At a fixed 2,048-token reading budget the ranking
inverts end-to-end. R@1 and R@10 agreed with each other and were wrong the
same way — **two agreeing metrics are not corroboration when they share a
frame.**

**The graveyard** (all paired, all significant unless noted):

| mechanism | result | why it died here |
|---|---|---|
| BM25 hybrid (1:1) | −15.6 pts (p=8e-36) | destroys a strong dense signal; weight sweep climbs monotonically back to dense-only, exactly as the CPS bake-off said |
| bge-reranker-base | −8.9 pts (p=1e-16) | a 110M cross-encoder reordering a 4B embedder's output; a reranker helps when it is *stronger* than the retriever, not when there is merely room at the top |
| HyDE | −6.7 pts (p=1e-11) | the instruction-tuned embedder already bridges the query↔statute register gap; embedding a pseudo-document discards that asymmetric training |
| citation-path prefix | 0.000 (p=0.78) | resolves to the *rule*; gold is a *provision* one level below — 342 rules split across chunks, zero pairs share a path |
| parent-stem carry | 0.000 | shared by every sibling, distinguishes none |
| rule-level dedup | −12 to −14 pts | statute chunks from one rule are different *provisions*, not redundancy |
| 512/1024-token chunks | worse at budget (p=2e-05 / 6e-16) | fewer chances per reading budget |
| doc2query | not run, by mechanism | same gap HyDE targets, already closed by the embedder |
| score-threshold re-query | rejected pre-build | top-cosine barely separates hit from miss (0.684 vs 0.622); trigger wrong 6 times in 7 |

**The transfer rule** (the most reusable finding): of the priors carried from
the CPS bake-off, those tied to **where the gold sits** did not transfer
(prefix, stem, boundary respect, chunk size — CPS scored at section level,
this at provision level), while those orthogonal to gold granularity
transferred **exactly** (dense-only, rerankers harming strong retrieval).
Ask of any imported finding: does it depend on the granularity of the gold?
If yes, re-measure; if no, trust it.

**Structure dose-response replicated:** heading-aware vs structure-blind is
+0.051 where packing crosses a boundary (p=0.02) and −0.005 where it does not
(p=0.86) — the bench-book mechanism exactly, at one-fifth magnitude, because
enumerated statutory provisions survive being packed with neighbours in a way
policy prose does not.

## 4. Generation (`pipeline/eval_answers.py`, `4_eval/gen_*.jsonl`)

Five models, same 140 queries, same retrieval, judged by
mistral-large-3:675b (a family with no stake anywhere in the stack; the
judge was switched once when GLM became a contender and once on owner
instruction — a comparison judged by two different models is not one
comparison, so the mixed partial run was discarded).

**Shipped: glm-5.2** — cites-gold 0.880 (tied best), false refusal 0.102,
correct refusal 32/32, supported 0.948 at n=97, p50 2.3s.

- The refusal dial trades directly against accuracy: nemotron-super's 2.8%
  false-refusal came with the table's only significant cites-gold loss
  (0.778, p=0.001) and 4 answered negatives.
- deepseek-flash's 0.962 "supported" was survivorship — judged on 53 answers
  because it refused a quarter of the answerable set. **Denominators are part
  of the metric.**
- Zero fabricated citations from any model in the bake-off (~700 answers).

## 5. The answer-layer optimisation (`4_eval/opt_*.jsonl`)

Four paired arms on glm-5.2. **Shipped: token_budget=4096** (~k=16):
cites-gold 0.907, false refusal 0.074, supported 0.970, overreach 0.030,
both gates held.

**Killed: by-rule context assembly**, on three independent strikes — cites-gold
−0.093 (p=0.021), overreach 0.052→0.221, and **the campaign's only fabricated
citation** (MCR 6.010, a rule that does not exist, invented under the merged
format). The improvement swarm's failure population was real (900/1060
interleaved pools); its mechanism prediction was wrong (said overreach
15%→10%; measured 5%→22%). The combined arm confirmed the damage compounds
with a larger window (−0.204, p=1e-4) — prediction filed before the result.

**The prompt-v2 trap, twice avoided after being fallen into once.** A prompt
targeting overreach produced "supported 0.817→0.857, overreach ↓" — both
survivorship: it refused 56% of answerable questions and the judged set
shrank to the confident survivors. Standing rule since: **any intervention is
scored on the full denominator, refusals both directions, before its target
metric is read.**

## 6. Interface and human-factors (`pipeline/serve.py`, judge review)

48-agent review as a sceptical non-technical judge; 41 findings reproduced
against the live server. Fixed: decorative similarity scores (the RRF
constant, identical for every query — replaced with true cosine, drawer-only);
boilerplate assurance on every passage (now marked *cited / not cited in the
answer*, computed from the audit); statute (MCL) citations silently excluded
from "all N citations verified"; the judiciary's motto on an unattributed
prototype; page-number/PDF-sheet confusion (now both, linked into the PDF at
the exact sheet).

**Lesson — trust surfaces must not manufacture precision.** A number that
never varies (the RRF constant), a stamp applied to everything ("its text
answers the question"), and a verified-count that silently scopes itself are
all the same defect: the *costume* of verification. The judge persona caught
all three; the builder had caught none.

## 7. Process lessons, in order of how much they saved

1. **Paired arms with hard gates, or it didn't happen.** Eleven mechanism
   verdicts came from paired McNemar on identical queries; four reversed the
   author's stated position, two reversed the improvement swarm's verified
   estimate. Gates (correct-refusal 32/32, validity 1.000) are read before
   target metrics.
2. **File predictions before results.** The HyDE prediction, the both-arm
   prediction, and the swarm's estimates were all written down first; that is
   the only reason "the mechanism was confirmed" is distinguishable from
   rationalisation.
3. **Adversarial verification pays for itself.** Every swarm ran
   finder→refuter; refuters killed 6–30% of findings each round and twice
   improved on the finder's proposed fix.
4. **Harness bugs masquerade as subject failures.** The five worst numbers
   of the session (fabrications 3→0, false refusal 17.6%→8.3%, 52
   unanswerable rejections→45→29, zero-of-140 answered, page off-by-one) were
   all measurement error, found by reading raw records rather than
   summaries.
5. **A clean exit is the most dangerous failure.** The thread-race that
   answered 0/140 exited with code 0; the pgrep self-match deadlock waited
   forever silently; both looked like success or patience from outside.
6. **Every improvement that survived was a deletion.** The additions that
   earned their place: the citation router (+11.5), the wider reading window,
   graph expansion (answer-quality, directional), and the hardened
   normaliser. Everything decorative died on measurement.

## 8. File map (what to rerun, where numbers live)

| artifact | produces / holds |
|---|---|
| `pipeline/parse_mcr.py` → `1_parsed/blocks.jsonl` | the corpus |
| `pipeline/verify_structure.py` → `1_parsed/verify_report.json` | TOC gate |
| `pipeline/build_graph.py` → `1_parsed/xrefs.jsonl` | 943 classified edges |
| `pipeline/chunk_mcr.py` → `3_chunks/v_rule256.jsonl` | deployed chunks (variants: see `archive/`) |
| `pipeline/run_sweep.py` → `4_eval/results.jsonl`, `ranks.jsonl`, `sweep_final.json` | retrieval sweep, per-query ranks |
| `pipeline/eval_answers.py` → `4_eval/gen_*.jsonl`, `opt_*.jsonl` | answer-layer runs |
| `pipeline/compare_generators.py` | five-way paired table |
| `pipeline/significance.py` | McNemar + bootstrap used throughout |
| `2_eval/mcr_eval_v1.jsonl` + `REVIEW.html` | frozen eval set + pending human pass |
| `docs/AGENTIC-DESIGN.md`, `docs/IMPROVEMENT-ROADMAP.md` | what's next and what's rejected |
