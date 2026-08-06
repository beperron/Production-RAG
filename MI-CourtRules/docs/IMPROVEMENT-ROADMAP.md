# MCR RAG Improvement Roadmap

## Exchange rate: how retrieval points and answer points are compared

The product metric is **correct, gold-cited answers**. Answer-quality points are the base currency (1 pt = ~10.6 queries end-to-end). Retrieval R@1 points convert at the rate they move gold **across the answering-window boundary**, because cites-gold-when-in-context is 0.957–0.979 regardless of position inside the window: of the 304 rank-2-10 golds, 287 already reach the k=8 context, so a +22–27 pt R@1 gain converts to only ~+4.4 answer pts. R@1 retains independent value only for the ranked provision cards the user sees (card-ordering UX). All rankings below use answer-points-per-effort first, R@1/UX as tiebreaker. Latency/ops wins are ranked separately since they move zero eval metrics.

**Critical sequencing fact:** three verified opportunities (4096-token budget, k=16, margin-triggered expansion) are the *same lever* measured three ways, and three others (refusal-triggered second pass ×2, HyDE-union) are the *same lever*. The roadmap below deduplicates; do not stack their estimates.

---

## Group 1 — DO NOW (hours each, high confidence)

**1.1 Deepen the generation window: k=8 → 4096-token flat budget (~k=16).** *The single best gain/effort in the entire analysis.*
- Population: 73/1060 queries (6.9%) with gold outside the k=8 window; 41–50 recoverable at this depth; 14–15 of 111 disambiguation queries; 8 of glm-5.2's 11 false refusals; the exact "2.116(D)(4) unretrieved" overreach class.
- Gain: +3.7–4.5 pts cites_gold overall; disambiguation window-hit 0.838 → 0.964–0.973 (+~13 pts, weakest type → parity); false refusal 0.102 → ~0.06. Median context only 1,407 → 2,825 tokens. Saturates near 4096 — do not buy 6144.
- Graveyard-clean: the 512/1024 result was chunk *granularity* at fixed budget, not budget depth.
- **First step:** add a `--k`/token-budget flag to `pipeline/eval_answers.py` (k=8 is hard-coded at line 167), rerun the seed-20260805 140-query harness for deepseek and glm-5.2, McNemar on cites_gold flips, verify correct-refusal stays 32/32 and overreach doesn't rise.

**1.2 Assemble context BY RULE (merge same-rule chunks, rule order, title once).**
- Population: 900/1060 pools have 2+ same-rule chunks, 743 interleaved, 773 out of order; ≥4/14 judged overreach cases are sub-provision misattribution; 9 false refusals occurred with gold shown.
- Gain: plausible overreach 15%→10%, false refusal −2 to −4 pts, at **zero token cost** (deletes repeated headers). ~20 lines in `Engine.answer()`'s formatter.
- **First step:** implement the merged formatter preserving per-chunk citation labels (the citation_grounded checker keys on them), rerun the same harness *in the same batch as 1.1* — one eval run, 2×2 arms (k, formatter), paired McNemar.

**1.3 Cut generation timeout 180s → 15s with retrieval-only fallback.** Worst-case time-to-degraded 180s → 15s (12x); observed max generation is 6.9s, so 0/140 induced failures. **First step:** change `timeout=` at `mcr_search.py:262/298`, add the "generation unavailable — showing retrieved provisions" banner, replay the 140 queries.

**1.4 launchd KeepAlive + HF_HUB_OFFLINE=1.** Kills the ~30s cold start and the ~2.0s hub check (measured 5.3s → 3.3s load); removes the air-gap hang mode. **First step:** write the plist (RunAtLoad+KeepAlive), set env, reboot-test.

**1.5 Pre-render the four homepage chip answers at startup.** Chip clicks 1,774ms → ~5ms; invalidate on chunk-hash + GEN_MODEL. **First step:** memoize `answer()` for the four strings in `serve.py:41-46`. Lowest priority in this group — do not extend to a general answer cache (query stream is 99.7% unique).

---

## Group 2 — TEST NEXT (cheap A/B, run against the post-Group-1 baseline)

**2.1 Listwise rerank of top-20 by the generator (rerank-then-truncate).** *The big R@1 lever.*
- Population: 353 queries with gold at rank 2–20 (R@20 = 0.977). Gain: R@1 0.660 → 0.85–0.93 (+19–27 pts) — but per the exchange rate, only ~+4.4 answer pts, **largely overlapping 1.1's population** (the ~49 rank-9-20 golds). After 1.1 ships, its marginal answer gain is small; its main residual value is card-ordering and citation-first-position.
- Not the dead cross-encoder: bge-reranker was a 110M pointwise model; the generator's measured in-context discrimination is 0.979 (95/97, 10/10 disambiguation).
- **First step:** cached top-20 already exists in the .npy dot products — one ~1,060-call generator pass, McNemar vs baseline ranks. Decide on the answer-layer marginal *after* the 1.1 result, and gate the call on the small-margin trigger (median miss margin 0.031) to protect the 700 rank-1 golds.
- Fallbacks if its 2x-calls latency is rejected: **sibling-scoped micro-rerank** (+6.2 pts net R@1, fires on <20% of queries, 81-query population where everything above gold is a sibling) or **margin-triggered window expansion** (~+1.9 pts, zero extra calls). Do not build all three.

**2.2 Complete partially-retrieved rules at generation time.**
- Population: 38/1060 (52% of remaining top-8 misses) have the gold rule partially in the pool; 36/38 rescued by completion; 10/14 gold-unretrieved answers had the rule partially shown. Gain: +3.4 pts gold-shown, ~+3.2 pts cites_gold; also grounds sibling ungrounded citations. Overlaps 1.1 partially — measure the marginal on the post-k=16 residual. Rule-dedup graveyard doesn't apply (that removed chunks from the *ranking*; this adds them at *generation*).
- **First step:** three-arm harness run (baseline / complete-top3-rules at median 4.1k tok / complete-all at 7.8k), watch the 32 negatives.

**2.3 Pre-display verifier: mistral judge on the draft, note-conditioned regeneration.**
- Population: the entire residual unsupported mass — 5/97 glm overreach (4/5 have the correction already in context); 14/93 deepseek. The citation audit alone is measured insufficient (flags 0/5 glm cases). Regenerate-only, never refuse-on-verifier — avoids the prompt-v2 trap.
- **First step:** wire judge-then-regen behind a flag; acceptance evidence is the **hand-review of regen diffs** (~5–14 per run), not the judge's own counts (circularity).

**2.4 Progressive render: cards at ~90ms, stream the answer, append the audit.** ~1 day; time-to-first-content 1,774ms → ~100ms for 100% of queries; TTFT measured 0.67–1.32s. **First step:** SSE/chunked transfer from the existing ThreadingHTTPServer; move refusal styling to stream-end; verify streamed answers byte-identical on the 140 queries.

---

## Group 3 — AGENTIC PHASE (worth building, after Groups 1–2 reset the residual)

**3.1 Unified refusal-triggered second pass.** One mechanism, four verified variants merged: on refusal, (a) deterministically fetch any real-but-unretrieved rule the refusal *names* (6/11 refusals name gold's rule or a sibling — resolves deep ranks 37/49/63 that no k-bump reaches), (b) read ranks 9/17–20, (c) HyDE re-query with the "name the Michigan term of art" instruction, union everything, regenerate once.
- Population after k=16: the ~24–32 still-absent golds plus in-window refusals. Gain: +1.5–3 pts end-to-end (the honest range; trigger rates measured on n=11–14). HyDE graveyard doesn't apply — miss-gated and unioned, never replacing pass-1.
- **The one hard constraint:** the plain refusal trigger fires on all 32 correct refusals; the citation-guard variant fires on 1/32. Score correct-refusal every run; abort below 31/32. This is the mirror image of what killed prompt-v2.
- **First step:** measure the k=16 residual first (which failure IDs remain), then wire the loop behind a flag in `Engine.answer()`.

**3.2 Within-rule outline + generator-directed subrule fetch.** For the 28 right-rule-wrong-chunk misses (17 with <50-word fragment golds that can never win an embedding contest): append each retrieved rule's ~100–200-token heading outline, let the generator request subrule citations through the existing router. Ceiling +2.6 pts, plausible ~+1.6; additive-only fetches bound the risk. Build after 2.2, which eats part of the same population. **First step:** build outlines from `1_parsed/blocks.jsonl`'s heading tree; test on the 28 named qids.

**Also flagged (from the router audit, not yet a costed opportunity):** the citation router is 125/125 only on canonical input; lowercase/prefix-dropped/mashed citations either fail or — worse — silently route to the *wrong subrule at score 1.0*. Deterministic string-normalization fixes plus an MCL reverse index (349 statute cites) are cheap and touch nothing in the graveyard. Should be scoped as a Group-1-adjacent hardening task before real court-staff traffic arrives.

---

## Group 4 — EXPLICITLY REJECTED (do not re-litigate)

| Idea | Why it's dead |
|---|---|
| Multi-hop cross-reference retrieval | Measured incremental population **0/1060**: one-hop `Engine.expand` already ships and captures both rescuable queries; two hops rescue zero more. |
| Query decomposition | Zero-size failure population: every eval query is single-gold, and the 253 compound-looking queries retrieve *better* than average (R@1 0.680 vs 0.654). Untestable until multi-gold queries exist. |
| Rule-level score aggregation / rule-vote rerank | Swept 15 configs this session: best R@1 0.6698 but R@10 0.947→0.880 — rule-dedup's pathology in milder form. New graveyard entry. |
| IDF lexical tie-break on top-10/20 | Swept 10 configs: max +0.75 pts, inside noise (SE ±1.5). New graveyard entry. |
| Query-embedding cache | Saves 73ms of ~1,850ms on a 99.7%-unique stream. |
| Pre-computed xref expansion | Saves 0.1ms/query. |
| Naive provision-index RRF union, rule-prior fusion, two-stage rule-then-chunk, absolute-score confidence trigger, ALIASES A/B on the absent set (0/56 contain any alias token) | All measured null-to-negative this session. |
| Global HyDE, BM25/hybrid, bge-reranker, doc2query, citation-path prefix, parent stem, rule-dedup, 512/1024 chunks, prompt-v2 | Original graveyard stands; nothing found reverses any of them. The only sanctioned HyDE use is miss-gated + unioned inside 3.1. |

---

## The single highest-leverage item

**Deepen the generation window from k=8 to a 4096-token budget (item 1.1), shipped together with by-rule context assembly (1.2) in one eval run.**

Defense: it is the only item that is simultaneously *hours* of work (a config value plus a formatter), fully pre-verified by simulation against cached embeddings (gold-shown 0.931→0.977, reproduced independently three times across the disambiguation, answer-layer, and context-assembly analyses), and aimed at every top-severity measured failure at once — it converts ~+4 pts cites_gold at the measured 0.957 rate, takes disambiguation from the weakest query type (0.838 window-hit) to parity (0.964), roughly halves glm-5.2's false-refusal rate (0.102→~0.06), and directly eliminates the flagship overreach case ("the rule is silent on timing" while 2.116(D)(4) sat unretrieved at rank 9-16). The listwise rerank promises more R@1 points (+19–27), but under the stated exchange rate those convert to roughly the *same* ~4 answer points at 10x the engineering cost, double the LLM calls, and 1–2s added latency — and its true marginal value cannot even be estimated until the window fix resets the residual. Every other item on this roadmap should be measured against the post-1.1 baseline; that alone makes it the item to do first.