# Production-RAG

Production retrieval-augmented search systems over public law and policy,
organized by state and corpus. Each system is self-contained — its own
corpus, pipeline, evaluation set, web app, and database schema — and each
has a detailed README in its directory.

| system | corpus | live | code |
|---|---|---|---|
| **North Carolina — child welfare** | NC Juvenile Code (GS 7B) + Administrative Code · NCDHHS policy manuals (~4,150 pages) | [nc-policy.parallel42.ai](https://nc-policy.parallel42.ai) | [`NC-ChildWelfare/`](NC-ChildWelfare/) |
| **Michigan — court rules** | Michigan Court Rules (874 pages, 625 rules, 11,860 citable provisions) | [mi-court-rules.parallel42.ai](https://mi-court-rules.parallel42.ai) | [`MI-CourtRules/`](MI-CourtRules/) |

Both are projects of the Child & Adolescent Data Lab, University of
Michigan School of Social Work. **Public data only** — no case records,
no confidential material, and the engines refuse private workspaces in code.

## NC-ChildWelfare

Hybrid retrieval over statutes and policy: Jina v3 contextual embeddings
(every chunk carries a one-sentence situating blurb that feeds both the
vector and the full-text lane) fused with Postgres full-text search by
reciprocal rank, then fully reranked by Jina's cross-encoder. Grounded
answers (deepseek-v4-flash) draw only from retrieved passages and decline
rather than guess. Measured on a reproducible benchmark: statutes R@3 90%,
policy R@3 96%. → [`NC-ChildWelfare/README.md`](NC-ChildWelfare/README.md)

## MI-CourtRules

A provenance-first bench tool: a geometrically verified parse (three
independent correctness proofs — structure against the document's own
table of contents, word-multiset and character-multiset equality with the
raw PDF), dense-only retrieval with a 4B instruction-tuned embedder and an
exact-citation router, cross-reference graph expansion, and a citation
audit that checks every emitted citation against the corpus before it is
shown. Every mechanism was kept or killed by paired McNemar tests on a
1,092-query eval set; the README lists what died on measurement so it is
not re-added. Cites-gold 0.907 · citation validity 1.000 · fabrications 0.
→ [`MI-CourtRules/README.md`](MI-CourtRules/README.md)

## Shared infrastructure, deliberate separation

The two systems share one Supabase Postgres project but live in separate
schemas (`public` for NC, `mcr` for Michigan) — no shared tables, functions,
or policies. Web tiers deploy independently on Vercel: `nc-policy`
auto-deploys from this repo (root directory `NC-ChildWelfare/web`);
`michigan-court-rules` deploys by CLI from `MI-CourtRules/web`. Secrets are
never committed — keys live in a chmod-600 env file outside the repo, and
the apps receive them as platform environment variables.
