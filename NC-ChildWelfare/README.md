# NC-ChildWelfare — North Carolina child-welfare law & policy search

Semantic search with grounded, cited answers over two **public** North
Carolina corpora:

- **`legal-authorities`** — NC General Statutes (Chapter 7B, the Juvenile
  Code) plus supporting NC Administrative Code. 28 documents, 7,449 passages.
- **`nc-child-welfare`** — NCDHHS child-welfare policy manuals, protocols,
  forms, and guidance (~4,150 pages). 502 documents, 10,851 passages.

> **Public data only.** The engine refuses any confidential / private
> workspace in code (`parsevault.lawsearch._assert_public`). It contains no
> case records and is not built to.

**Live:** <https://nc-policy.parallel42.ai>

## Deployed architecture

```
question ──► Jina v3 query embedding (1024-dim)
         ──► Supabase `hybrid_search` RPC: pgvector cosine + Postgres
             full-text, reciprocal-rank fused (pool 40)
         ──► full Jina rerank of the fused pool (measured better than
             blending: statutes R@3 80→90%, policy 92→96%)
         ──► grounded answer (deepseek-v4-flash via Ollama Cloud), drawn
             only from the retrieved passages, with inline citations;
             declines rather than guesses
```

- **Contextual Retrieval** (Anthropic): every chunk carries a one-sentence
  context blurb situating it in its document; the blurb feeds *both* the
  embedding and the full-text lane. The blurbs are committed
  (`knowledge-base/*/context.jsonl`) — they are expensive to regenerate and
  are the canonical copy alongside the database's `context` column.
- **Web app** — Next.js (`web/`), server-side only; deployed on Vercel
  (project `nc-policy`, root directory `NC-ChildWelfare/web`), auto-deploys
  on push to `main`.
- **Database** — Supabase Postgres, `public` schema (`documents`, `chunks`
  with vector(1024) + generated tsvector, `hybrid_search` function).
  Read-only row-level security: the app's anon key can select and call the
  RPC, never write; ingest uses the service role locally.
- The same Supabase project hosts the Michigan Court Rules system
  ([`../MI-CourtRules`](../MI-CourtRules)) in its own `mcr` schema — fully
  separate tables, functions, and policies.

### Rebuilding the database

`scripts/ingest_supabase.py` is idempotent and resumable: it loads
`docindex.json` + `context.jsonl` per collection, embeds context+content
with Jina v3 (`retrieval.passage`, 1024 dims), and upserts via the REST API,
skipping chunks already present. After an accidental database deletion in
Aug 2026, this script rebuilt all 530 documents / 18,300 chunks from the
committed corpus in one run.

```bash
env SUPABASE_URL=... SUPABASE_SERVICE_KEY=... JINA_API_KEY=... \
  python scripts/ingest_supabase.py            # add --limit 40 to smoke-test
```

Schema DDL: `supabase/migrations/` (0001 base, 0002 contextual — applying
0002 over 0001 requires dropping the 0001 `hybrid_search` signature first).

## Measured quality (starter benchmark, `evals/law_search/`)

| corpus | recall@1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|
| Statutes (GS 7B) | 66% | **90%** | 92% | 0.79 |
| Policy (NCDHHS)  | 89% | **96%** | **100%** | 0.94 |

Plain-language write-up: [`docs/LAW_SEARCH_QUALITY.md`](docs/LAW_SEARCH_QUALITY.md).
Attorney-facing explainer: [`docs/law_rag_brief.html`](docs/law_rag_brief.html).

## Local-first engine

The original local stack still works without any cloud dependency — BM25F +
GTE-base hybrid, fused across collections:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .

# First search triggers a one-time dense re-embed from the committed index
python scripts/query.py "grounds for termination of parental rights"
python scripts/query.py "reasonable efforts to prevent removal" --answer

# Local web UI:
python scripts/law_search_server.py --port 8787
```

Cloud enhancement (rerank, cloud generation) is env-gated via
`~/.config/parsevault/lawsearch.env` (chmod 600, never committed) and
degrades to the local engine when keys are absent.

## Layout

```
src/parsevault/          RAG/search engine (config, rag, lawsearch, agent_search)
  pipeline/              docindex, embeddings, reranker, retrieval, faceting
  pipeline/extractors/   conversion lanes (native / tesseract / docx / vlm / llamaparse)
scripts/                 query.py, law_search_server.py, build_kb*.py,
                         contextualize.py, ingest_supabase.py
web/                     Next.js app deployed to Vercel (nc-policy)
knowledge-base/          public corpora: outputs/ (markdown), docindex.json,
                         context.jsonl (committed), emb_*.json caches (ignored)
supabase/migrations/     database schema
evals/law_search/        reproducible retrieval benchmark
docs/                    quality report + attorney brief
```

Dense vector caches and source PDFs are **not** committed; the converted
markdown + `docindex.json` + `context.jsonl` are the shipped searchable form.

## License

Not yet chosen — decide a license before making this repository public.
