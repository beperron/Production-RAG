# Court Rule Searcher — the Michigan Court Rules, with provenance

A retrieval-augmented search system over the **Michigan Court Rules**
(874 pages, 625 rules, 11,860 citable provisions, as amended through
July 31 2026), built for bench use: every answer cites provisions, every
citation is audited against the parsed corpus before it is shown, and every
passage traces to a printed page of the official PDF.

**Live:** <https://mi-court-rules.parallel42.ai> · a project of the Child &
Adolescent Data Lab, University of Michigan School of Social Work.

```
question ──► Qwen3-Embedding-4B (query) ──► pgvector similarity
         ──► citation router (exact MCR lookups never touch the index)
         ──► cross-reference graph expansion (overrides / excepts / conditions)
         ──► glm-5.2 composes a cited answer from ~4,096 tokens of passages
         ──► citation audit: every emitted citation checked against the corpus
             and against what was actually retrieved — then rendered
```

## The system, as shipped

Settled by paired tests end to end — every comparison McNemar on identical
queries, every intervention gated on correct-refusal and citation validity.
Full record: [`docs/TESTING-AND-LESSONS.md`](docs/TESTING-AND-LESSONS.md).

| layer | configuration | measured |
|---|---|---|
| corpus | 625 rules · 11,860 citable provisions | 3 independent proofs green |
| chunks | rule-scoped, 256 tokens, plain text | 3,580 passages |
| retrieval | Qwen3-Embedding-4B · dense only · citation router | R@1 0.660 · R@10 0.947 · R@2048tok 0.944 |
| graph | 943 classified cross-reference edges, bidirectional | overrides/excepts surfaced in answers |
| generation | glm-5.2 · 4096-token window · cross-ref expansion | cites-gold 0.907 · supported 0.970 |
| refusal | | false 0.074 · correct 32/32 |
| integrity | | citation validity 1.000 · fabrications 0 |

Evaluated on a 1,092-query set written by two generator models
cross-validating each other (each judged only the other's items), held to
median query↔gold word overlap ≈ 0.18 so lexical echo cannot inflate scores.

### What died on measurement (do not re-add)

BM25 hybrid (−15.6), bge reranker (−8.9), jina-reranker-v2 via API (−1.3,
p=0.50 — a strong reranker fights the 4B embedder to a draw, and a draw plus
latency is a loss), HyDE (−6.7), citation-path prefix (0.000), parent stem
(0.000), rule-dedup (−12), 512/1024-token chunks, by-rule context assembly
(overreach ×4 and the campaign's only fabricated citation), score-threshold
re-query (trigger wrong 6/7). Numbers and mechanisms in the testing record.

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
one that arrived by similarity, and the interface says which. Sources link
into the official PDF on courts.michigan.gov at the exact sheet.

## The parse (why the corpus is trustworthy)

The source is born-digital — 873 of 874 pages carry a text layer, zero
images — so structure is *measured* from geometry and font, never OCR'd or
guessed. Indent ladder +21.6 pt per level; line spacing bimodal (14.0 pt
within a paragraph, 20.0 pt between), so block boundaries are read off the
document.

**Three independent correctness proofs, all green:**

1. **Structure** — reconciled against the document's own 18-page printed
   table of contents, parsed by separate code reading different pages:
   9/9 chapters, 51/51 subchapters, 625/625 rules, 0 errors.
2. **Completeness** — word-multiset equality with the raw PDF:
   384,507 words in, 384,507 out.
3. **Character fidelity** — independent re-extraction, non-whitespace
   multiset: 1,886,132 = 1,886,132.

Adversarial audit swarms (48 + 21 agents, every finding handed to a refuter)
surfaced 43 real defects, each invisible on inspection — the worst: 195
markers with no trailing space silently absorbed as body text, mis-citing
everything beneath them. In a corpus whose purpose is to answer *"what does
MCR 1.109(D)(10) require"*, that is a wrong answer delivered confidently.

## Deployment

| piece | where |
|---|---|
| web + API | Vercel (`web/` — a serverless port of `pipeline/serve.py`; SSE streaming, feedback, query log) |
| vectors + telemetry | Supabase Postgres, schema `mcr` (pgvector HNSW, 2000-dim Matryoshka-truncated — measured lossless, p=0.804) |
| query embedding | OpenRouter `qwen/qwen3-embedding-4b` — **queries require the Qwen instruct template**; raw queries embed ~0.91 cosine from correct, templated 0.996+ |
| generation | glm-5.2 via Ollama Cloud |

Cloud-path acceptance gates: all 125 exact-citation lookups correct through
the deployed stack; 18/20 rank-1 agreement with the exact local scan on
identical vectors (`hnsw.ef_search = 400`); pgvector p50 39 ms.

The same Supabase project hosts a second, fully separate system (North
Carolina child-welfare policy search) in the `public` schema — no shared
tables or functions.

## Run locally

```bash
# parse + verify (requires PyMuPDF; fetch the PDF from courts.michigan.gov)
python3 pipeline/parse_mcr.py 0_source/michigan-court-rules.pdf -o 1_parsed
python3 pipeline/verify_structure.py 0_source/michigan-court-rules.pdf 1_parsed/blocks.jsonl

# web interface against the local engine
python3 pipeline/serve.py --port 8788    # -> http://127.0.0.1:8788

# deploy: assemble the serverless bundle, then ship it
python3 pipeline/build_web.py
cd web && vercel deploy --prod
```

Secrets live in `~/.config/parsevault/lawsearch.env` (never committed);
query logs in `5_logs/` are gitignored.

## Layout

```
pipeline/     parse, verify, chunk, sweep, eval, serve, cloud adapters
1_parsed/     blocks.jsonl (12,298 blocks), xrefs.jsonl (943 edges)
2_eval/       frozen 1,092-query eval set + human-review page
3_chunks/     deployed chunk variant (rule-scoped 256-token)
4_eval/       sweep results, per-query ranks, generation A/Bs
web/          Vercel deployment bundle (build product of build_web.py)
supabase/     mcr schema DDL
docs/         TESTING-AND-LESSONS.md · AGENTIC-DESIGN.md · IMPROVEMENT-ROADMAP.md
```
