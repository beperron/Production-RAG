# archive/

Not part of the shipped system; kept on disk, out of git.

- `chunk_variants/` — the eight non-shipped chunking variants from the sweep.
  Regenerable deterministically: `pipeline/chunk_mcr.py --strategy ... --budget ...`
  (see `docs/TESTING-AND-LESSONS.md` §3 for which flags produce which variant).
- `superseded_evals/` — answer-eval runs from configurations later killed
  (prompt-v2, no-expand ablation). Their numbers are quoted in the docs;
  the records are kept for re-derivation only.
