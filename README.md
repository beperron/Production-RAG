# carolina-policy-search

Local-first **semantic search over public North Carolina child-welfare law and
policy**, with grounded, cited answers. A retrieval-augmented generation (RAG)
engine over two public corpora:

- **`legal-authorities`** — NC General Statutes (Chapter 7B, the Juvenile Code)
  plus supporting NC Administrative Code.
- **`nc-child-welfare`** — NCDHHS child-welfare policy manuals, protocols, forms,
  and guidance (~4,150 pages).

> **Public data only.** The engine refuses any confidential / private workspace
> in code (`parsevault.lawsearch._assert_public`). It contains no case records and
> is not built to.

## What it does

- **Hybrid retrieval** — BM25F (keyword) + GTE-base (meaning), reciprocal-rank
  fused across collections, with an optional Jina reranker blended into the
  ranking. Every result carries a full source-traceable citation (source, section,
  page) and a SHA-256 content hash.
- **Grounded answers** — an optional written answer drawn *only* from the retrieved
  passages, with inline citations; it declines when the passages don't support an
  answer rather than guessing.

### Measured quality (starter benchmark, `evals/law_search/`)

| corpus | recall@1 | recall@3 | recall@5 | MRR |
|---|---|---|---|---|
| Statutes (GS 7B) | 66% | **90%** | 92% | 0.79 |
| Policy (NCDHHS)  | 89% | **96%** | **100%** | 0.94 |

Plain-language write-up: [`docs/LAW_SEARCH_QUALITY.md`](docs/LAW_SEARCH_QUALITY.md).
Attorney-facing explainer: [`docs/law_rag_brief.html`](docs/law_rag_brief.html).

## Quickstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .

# First search triggers a one-time dense re-embed from the committed index
# (a few minutes; vectors persist to a gitignored *.vectors.json sidecar).
python scripts/query.py "grounds for termination of parental rights"
python scripts/query.py "reasonable efforts to prevent removal" --answer

# Or the web UI:
python scripts/law_search_server.py --port 8787
#   → http://127.0.0.1:8787
```

### Optional cloud enhancement (public data only)

Query rewrite, Jina reranking, and cloud generation are **env-gated** and degrade
to the local engine when absent. Provide keys via a gitignored file:

```bash
mkdir -p ~/.config/parsevault
cat > ~/.config/parsevault/lawsearch.env <<EOF
JINA_API_KEY=...
OPENROUTER_API_KEY=...
EOF
chmod 600 ~/.config/parsevault/lawsearch.env
```

Local generation uses **Ollama** (Qwen3.6) if running; embeddings use GTE-base via
`sentence-transformers`.

## Layout

```
src/parsevault/          RAG/search engine (config, rag, lawsearch, agent_search)
  pipeline/              docindex, embeddings, reranker, retrieval, faceting
  pipeline/extractors/   conversion lanes (native / tesseract / docx / vlm / llamaparse)
scripts/                 query.py (CLI), law_search_server.py, build_kb*.py, catalog_kb.py
knowledge-base/          public corpora: outputs/ (markdown) + docindex.json + catalogs
evals/law_search/        reproducible retrieval benchmark
docs/                    quality report + attorney brief
```

Dense vectors and source PDFs are **not** committed (see `.gitignore`): vectors are
rebuilt locally on first run; the converted markdown + `docindex.json` are the
shipped searchable form.

## Rebuilding a knowledge base from source

```bash
pip install -e '.[build]'          # adds pymupdf, tesseract, etc.
python scripts/build_kb.py <folder-of-pdfs> --collection nc-child-welfare
python scripts/catalog_kb.py knowledge-base/nc-child-welfare
```

## License

Not yet chosen — decide a license before making this repository public.
