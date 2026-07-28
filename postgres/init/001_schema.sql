-- carolina-policy-search — local Postgres schema.
-- Backs the LOCAL GTE-base (768-dim) retrieval stack + build-time
-- provenance log. Separate from supabase/migrations/ (cloud stack: Jina v3,
-- 1024-dim) — the two are not interchangeable, see memory
-- dual-storage-architecture.

create extension if not exists vector;

-- --------------------------------------------------------------------- --
-- documents / chunks — mirrors knowledge-base/*/docindex.json
-- --------------------------------------------------------------------- --
-- Stable identity + provenance fields get real columns (queried/joined
-- directly); everything else DocMetadata carries (outline, key_phrases,
-- summary, doc_type signals, layout, ...) goes in `metadata` so this table
-- doesn't need a migration every time that dataclass gains a field.
create table if not exists documents (
  doc_id          text primary key,
  source_path     text not null,
  title           text,
  page_count      int,
  word_count      int,
  category        text,       -- form | manual | policy | letter | guidance | resource | other
  topic           text,       -- intake | assessment | safety | foster_care | ...
  audience        text,       -- social_worker | supervisor | court | provider | family | mixed
  source_url      text,
  source_domain   text,
  section         text,       -- collection section, e.g. "NC General Statutes"
  retrieved_at    text,       -- ISO date the document was obtained
  effective_date  text,
  currency_status text default 'unknown',
  review_status   text default 'unreviewed',
  quality_score   real default 1.0,
  -- content_sha256: hash of the extracted Markdown (changes on re-extraction)
  -- source_sha256:  hash of the original PDF bytes (stable identity/integrity root)
  content_sha256  text,
  source_sha256   text,
  metadata        jsonb not null default '{}'::jsonb
);

create table if not exists chunks (
  chunk_id          text primary key,
  doc_id            text not null references documents (doc_id) on delete cascade,
  ordinal           int not null,
  heading_path      text[] not null default '{}',
  content           text not null,
  char_len          int,
  token_estimate    int,
  page_start        int,
  page_end          int,
  extraction_routes text[] not null default '{}',
  embedding         vector(768)
);

-- generated full-text column (keyword lane) — parity with the Supabase
-- schema's hybrid-search setup, for when a Postgres-side search path exists.
alter table chunks add column if not exists fts tsvector
  generated always as (to_tsvector('english', coalesce(content, ''))) stored;

create index if not exists chunks_embedding_idx on chunks using hnsw (embedding vector_cosine_ops);
create index if not exists chunks_fts_idx       on chunks using gin (fts);
create index if not exists chunks_doc_id_idx    on chunks (doc_id);

-- --------------------------------------------------------------------- --
-- build_log — hash-chained, append-only audit trail
-- --------------------------------------------------------------------- --
-- Persists what scripts/build_dashboard_server.py's build_events.jsonl
-- already records per build run, in a form that's tamper-evident: each
-- row's row_hash commits to the previous row's row_hash (Certificate-
-- Transparency-log style), so altering or deleting a past row breaks the
-- chain for every row after it. See memory provenance-no-blockchain for why
-- this was chosen over a blockchain (single-operator pipeline — nothing to
-- gain from distributed consensus).
--
-- prev_hash/row_hash are computed application-side at insert time (by the
-- loader that reads build_events.jsonl), not by Postgres — this schema only
-- stores the chain and enforces that it can never be edited after the fact.
create table if not exists build_log (
  id        bigserial primary key,
  build_id  text not null,
  seq       int not null,
  ts        timestamptz not null,
  stage     text not null,
  fields    jsonb not null default '{}'::jsonb,
  prev_hash text not null,
  row_hash  text not null,
  unique (build_id, seq)
);

create index if not exists build_log_build_id_idx on build_log (build_id);

create or replace function build_log_no_mutation() returns trigger as $$
begin
  raise exception 'build_log is append-only: % not permitted', tg_op;
end;
$$ language plpgsql;

drop trigger if exists build_log_no_update on build_log;
create trigger build_log_no_update
  before update on build_log
  for each row execute function build_log_no_mutation();

drop trigger if exists build_log_no_delete on build_log;
create trigger build_log_no_delete
  before delete on build_log
  for each row execute function build_log_no_mutation();

-- TRUNCATE doesn't fire row-level triggers (BEFORE/AFTER UPDATE|DELETE ...
-- FOR EACH ROW never runs for it), so the two triggers above don't actually
-- stop `truncate build_log` — it needs its own statement-level trigger.
drop trigger if exists build_log_no_truncate on build_log;
create trigger build_log_no_truncate
  before truncate on build_log
  for each statement execute function build_log_no_mutation();
