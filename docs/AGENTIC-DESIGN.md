# Agentic phase — design

The deterministic pipeline is finished and measured: R@1 0.660, R@10 0.947,
R@2048tok 0.944, citation validity 1.000. Every remaining failure class is one
a *fixed* pipeline cannot address, because fixing it requires reacting to what
this particular query and this particular draft answer turned out to need.
That reaction is the whole and only justification for an agentic phase.

**Design stance: a bounded state machine, not a free agent.** At most one
extra hop, a hard latency budget, every action recorded in provenance and
shown in the interface. A court tool must be explainable turn by turn;
"the system decided to search again, here is why, here is what it found" is
explainable. An open-ended loop is not.

---

## 1. The trigger question, measured first

The textbook pattern — re-query when retrieval confidence is low — **dies on
measurement** (935 non-router eval queries, deployed config):

| outcome | n | top-1 cosine, median |
|---|---|---|
| gold at rank 1 | 575 | 0.684 |
| gold at rank 2–10 | 304 | 0.643 |
| gold absent | 56 | 0.622 |

The distributions overlap heavily. A threshold trigger at 0.60 fires on 138
queries and catches 19 of the 56 misses — **14% precision, 34% recall**. An
alarm wrong six times in seven adds latency to healthy queries and would
train any observer to ignore it. Margin (top1−top2) separates ~3× better
(0.016 vs 0.045 medians) but is still not a usable gate alone.

**Rejected: score-threshold re-query.** The trigger for this corpus must be
*semantic*, and we already have one that costs nothing:

> The generator, holding the retrieved passages, says what is missing —
> "the passages do not state the deadline", "MCR 2.116(D) is referenced but
> not provided." It is not estimating uncertainty; it has *read* the context
> and can name the gap.

The measured populations that trigger reaches:

* **False refusals with gold unretrieved** — 10 of 19 refusals on answerable
  queries; refusing was locally correct, but a second targeted retrieval could
  convert several to answers.
* **Wrong-silence answers** — the `MCR 2.116(D)(4)` class: the answer implies
  the rules are silent while the governing provision sits unretrieved in the
  corpus. Part of the ~15% overreach population.
* **Named-but-missing references** — the answer or a passage cites a
  provision that was not retrieved; the audit already detects exactly this
  (`was_retrieved = false`), for free.

## 2. The one pattern worth building: targeted second retrieval

```
draft = generate(question, passages)                    # unchanged pipeline
gaps  = audit(draft) ∪ generator_gap_statement(draft)   # both already exist
if gaps and budget_remains:
    extra = retrieve(each gap)          # citation router first, dense second
    final = regenerate(question, passages + labelled extra)   # ONCE
    final.provenance += "second search: <gap>, requested because <reason>"
else:
    final = draft
```

Properties that make this fit a court tool:

* **The trigger is free.** The citation audit runs anyway; the gap statement
  is in the draft answer. No model call is spent deciding whether to act.
* **The retrieval is targeted, not a reformulation.** A gap names a citation
  ("2.116(D)") or a specific thing ("time limit for filing under (C)(10)").
  The citation router resolves the first kind exactly; dense handles the
  second with a far easier query than the user's original.
* **Bounded.** One regeneration, ever. If the second pass still has gaps, the
  answer ships with the gap named — which is the honest state.
* **Provenance-native.** The interface already labels passages by route;
  `second search` becomes a fourth route label with its recorded reason.
  Nothing about the trust story changes; the story gets one more honest line.

**Latency budget:** trigger fires on an estimated 10–15% of queries; cost is
one retrieval (~50ms) + one regeneration (~2–4s). p50 unchanged; p90 moves
only for the queries that were previously wrong or refused — the correct
place to spend seconds.

## 3. Patterns evaluated and held back

**Multi-hop chain following (depth ≥ 2).** The deterministic single-hop
expansion already supplies binding cross-references and measurably improves
the answer layer (+supported, −overreach, −false-refusal). Chains needing a
*second* hop exist (690 binding edges compose), but no eval query has yet
demonstrably failed for want of one. Build when a failure population exists,
not before.

**Query decomposition.** The eval contains almost no compound questions,
because practitioners mostly ask one thing at a time. If real usage logs show
compound questions, revisit; designing for them now optimises an imagined
user.

**Iterative retrieval loops (retrieve→reflect→retrieve…).** Unbounded loops
are unexplainable to this audience and the marginal population beyond
one targeted hop is, on current evidence, near zero. Rejected on principle
and on measurement.

**Confidence-threshold re-query.** Rejected above; the trigger is wrong six
times in seven.

## 4. Guardrails, non-negotiable

1. **The audit gates display, always.** A regenerated answer passes the same
   citation audit as a first draft; no agentic path bypasses it.
2. **≤ 1 extra hop, ≤ 1 regeneration.** Hard-coded, not configurable.
3. **Every action shown.** The second search appears in the interface with
   its reason, in judge-readable language: *"The first draft referenced
   MCR 2.116(D), which had not been retrieved; the system searched for it and
   revised the answer."*
4. **Deterministic fallback.** Any failure in the agentic path (timeout,
   empty retrieval, regeneration error) ships the original draft unchanged.
5. **No new authorities.** The second retrieval searches the same corpus with
   the same router; the agentic layer can never introduce text from outside
   the parsed rules.

## 5. Evaluation plan

Same discipline as everything else in this project:

* A/B on the 140-query answer eval, paired, mistral-judged: false refusal,
  wrong-silence rate, judged supported/overreach, latency p50/p90.
* Regression gate: citation validity must stay 1.000 and correct-refusal on
  the 32 negatives must not drop — the agentic layer must never manufacture
  an answer to a genuinely unanswerable question. This is the failure mode
  the design most invites, and it gets measured first.
* Ship only on a paired win. The prompt-v2 lesson stands: a mechanism that
  looks better on the metric it targets can be worse on the one that matters.
