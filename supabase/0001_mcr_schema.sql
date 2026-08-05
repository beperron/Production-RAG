-- Michigan Court Rules -- retrieval corpus and provenance ledger.
--
-- A SEPARATE SCHEMA from carolina-policy-search's public tables, so the two
-- corpora share an instance without sharing a namespace, a migration history,
-- or a blast radius.
--
-- The design point: this database is the SYSTEM OF RECORD and the provenance
-- store, not the query path. Retrieval runs from memory -- 3,580 chunks at
-- 2560 dimensions is 36 MB and a numpy dot product answers in under a
-- millisecond, where a round trip to hosted pgvector costs 50-200 ms. Putting
-- retrieval here would make the system slower for no accuracy gain. What it
-- buys instead is a lineage a court can query: which passage supported which
-- sentence, which blocks that passage came from, which page of which PDF, and
-- which parse run produced it.

create schema if not exists mcr;
create extension if not exists vector;

-- ---------------------------------------------------------------- source ---
-- The exact bytes the parse ran on. A result is only reproducible if the
-- thing it was measured against still exists.
create table if not exists mcr.sources (
  source_id     text primary key,
  filename      text not null,
  sha256        text not null unique,
  page_count    int  not null,
  edition       text,
  fetched_at    timestamptz not null default now(),
  source_url    text
);

-- ----------------------------------------------------------- parse runs ---
-- Every corpus is stamped with the code that built it. The bench-book work
-- lost 394 of 1,722 gold labels to a corpus that changed underneath them;
-- this makes such a change detectable rather than silent.
create table if not exists mcr.parse_runs (
  run_id        text primary key,
  source_id     text not null references mcr.sources(source_id),
  parser_commit text,
  started_at    timestamptz not null default now(),
  n_blocks      int,
  n_rules       int,
  n_citations   int,
  -- the two proofs the parse has to pass
  toc_reconciled       boolean,   -- 625/625 rules vs the printed contents
  lossless_word_match  boolean,   -- 384,507 == 384,507
  notes         text
);

-- --------------------------------------------------------------- blocks ---
-- The parse's atoms. One provision, one page, one citation path.
create table if not exists mcr.blocks (
  block_id     text primary key,          -- mcr#B00786
  run_id       text not null references mcr.parse_runs(run_id) on delete cascade,
  kind         text not null,             -- rule | body | chapter | ...
  chapter      text,
  subchapter   text,
  rule         text,
  subpath      text,                      -- (C)(10)
  citation     text,                      -- MCR 2.116(C)(10)
  depth        int,
  pdf_page     int  not null,             -- 0-based sheet in the PDF
  printed_page int,                       -- what the page footer says
  text         text not null
);
create index if not exists blocks_citation_idx on mcr.blocks (citation);
create index if not exists blocks_rule_idx     on mcr.blocks (rule);
create index if not exists blocks_run_idx      on mcr.blocks (run_id);

-- --------------------------------------------------------------- chunks ---
-- What the retriever scores and the generator reads. Vectors are stored at
-- 2000 dimensions: Qwen3-Embedding-4B emits 2560, pgvector's HNSW index caps
-- at 2000, and Matryoshka truncation to 2000 measured statistically identical
-- to full (0.9509 vs 0.9491, p = 0.804). The constraint costs nothing.
create table if not exists mcr.chunks (
  chunk_id     text primary key,          -- mcr#C00239
  run_id       text not null references mcr.parse_runs(run_id) on delete cascade,
  rule         text,
  rule_title   text,
  chapter      text,
  subchapter   text,
  citation_first text,
  heading_path text,
  text         text not null,
  embed_text   text not null,
  n_tokens     int,
  sha256       text,
  embedding    vector(2000)
);
create index if not exists chunks_embedding_idx
  on mcr.chunks using hnsw (embedding vector_cosine_ops);
create index if not exists chunks_rule_idx on mcr.chunks (rule);

-- many-to-many: a chunk covers several provisions, a provision sits in one
create table if not exists mcr.chunk_blocks (
  chunk_id text not null references mcr.chunks(chunk_id) on delete cascade,
  block_id text not null references mcr.blocks(block_id) on delete cascade,
  primary key (chunk_id, block_id)
);

-- ------------------------------------------------- cross-reference graph ---
-- 1,629 edges, 690 of them load-bearing ("except as specified in", "subject
-- to"). An answer that quotes the source without the target is not merely
-- incomplete -- it can be wrong.
create table if not exists mcr.xrefs (
  from_citation text not null,
  to_citation   text not null,
  binding       boolean not null default false,
  context       text,
  primary key (from_citation, to_citation)
);
create index if not exists xrefs_to_idx on mcr.xrefs (to_citation);

-- ---------------------------------------------------------- evaluation ---
create table if not exists mcr.eval_queries (
  query_id     text primary key,
  query        text not null,
  query_type   text,
  arm          text,                      -- A claude | B glm-5.2 | C deterministic
  generator    text,
  gold         text[] not null,
  also_answered_by text[],
  status       text
);

create table if not exists mcr.eval_runs (
  eval_run_id  text primary key,
  run_id       text references mcr.parse_runs(run_id),
  config       jsonb not null,            -- chunker, embedder, mode, router...
  ran_at       timestamptz not null default now(),
  r_at_1       numeric,
  r_at_10      numeric,
  r_at_budget  numeric,
  mrr_at_10    numeric,
  n_queries    int
);

-- ------------------------------------------------------------ provenance ---
-- The audit trail a court would actually ask for: what was asked, what was
-- retrieved and by which route, what was answered, and whether every citation
-- in that answer resolved and had actually been shown to the model.
create table if not exists mcr.answers (
  answer_id    text primary key,
  asked_at     timestamptz not null default now(),
  question     text not null,
  answer       text,
  generator    text,
  run_id       text references mcr.parse_runs(run_id),
  refused      boolean,
  latency_ms   int
);

create table if not exists mcr.answer_passages (
  answer_id    text not null references mcr.answers(answer_id) on delete cascade,
  chunk_id     text not null references mcr.chunks(chunk_id),
  rank         int  not null,
  score        numeric,
  route        text not null,   -- dense | citation-router | cross-reference
  because_of   text,            -- for cross-reference: what conditions on it
  primary key (answer_id, chunk_id)
);

create table if not exists mcr.answer_citations (
  answer_id     text not null references mcr.answers(answer_id) on delete cascade,
  citation      text not null,
  exists_in_corpus boolean not null,
  was_retrieved boolean not null,
  primary key (answer_id, citation)
);

-- --------------------------------------------------------------- views ---
-- One row per sentence-supporting passage, from citation all the way back to
-- the page of the PDF. This is the query a clerk would run to check an answer.
create or replace view mcr.provenance as
select
  a.answer_id, a.question, a.answer, a.asked_at,
  ap.rank, ap.route, ap.score,
  c.chunk_id, c.citation_first, c.rule, c.rule_title,
  b.block_id, b.citation, b.printed_page, b.pdf_page,
  s.filename, s.sha256 as source_sha256, s.edition,
  pr.run_id, pr.parser_commit, pr.toc_reconciled, pr.lossless_word_match
from mcr.answers a
join mcr.answer_passages ap on ap.answer_id = a.answer_id
join mcr.chunks c          on c.chunk_id   = ap.chunk_id
join mcr.chunk_blocks cb   on cb.chunk_id  = c.chunk_id
join mcr.blocks b          on b.block_id   = cb.block_id
join mcr.parse_runs pr     on pr.run_id    = c.run_id
join mcr.sources s         on s.source_id  = pr.source_id;

comment on view mcr.provenance is
  'Full lineage of every passage behind every answer: answer -> passage -> '
  'chunk -> block -> printed page -> source PDF hash -> parse run.';
