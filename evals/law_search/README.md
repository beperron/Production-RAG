# Law & policy search benchmark

Reproducible retrieval benchmark for the **public** NC law/policy search surface
(`src/parsevault/lawsearch.py`). Public data only — never a confidential workspace.

- `law_gold.json` — 42 statute questions → the exact governing GS section(s).
- `policy_gold.json` — 28 quality-of-care policy questions → the document(s) that
  correctly answer each (an `accept` regex list; a hit counts if **any** matches,
  because several NCDHHS documents often cover one topic).
- `unified_eval.py` — runs both sets through the shipped pipeline (BM25F + GTE,
  RRF-fused, Jina-blend rerank) and reports R@1/R@3/R@5/MRR.

Run from the repo root (needs the public indexes + `~/.config/parsevault/lawsearch.env`
for the Jina key):

```bash
python evals/law_search/unified_eval.py
```

Latest result (RRF + Jina blend): statutes R@3 = 90% / R@5 = 92% / MRR 0.79;
policy R@3 = 96% / R@5 = 100% / MRR 0.94. Plain-language write-up:
[`docs/LAW_SEARCH_QUALITY.md`](../../docs/LAW_SEARCH_QUALITY.md).
