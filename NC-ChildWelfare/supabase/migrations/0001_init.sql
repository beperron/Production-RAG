-- carolina-policy-search — Supabase schema for cloud hybrid retrieval.
-- pgvector (Jina v3, 1024-dim) + Postgres full-text search, fused with RRF.
-- Public NC law & policy only.

create extension if not exists vector;

create table if not exists documents (
  doc_id      text primary key,
  collection  text not null,
  title       text,
  source_url  text,
  page_count  int
);

create table if not exists chunks (
  chunk_id     text primary key,
  doc_id       text,
  collection   text not null,
  ordinal      int,
  title        text,
  section      text,
  heading_path text,
  content      text not null,
  page_span    text,
  source_url   text,
  sha256       text,
  embedding    vector(1024)
);

-- generated full-text column (keyword lane)
alter table chunks add column if not exists fts tsvector
  generated always as (
    to_tsvector('english',
      coalesce(title,'') || ' ' || coalesce(section,'') || ' ' || content)
  ) stored;

create index if not exists chunks_embedding_idx on chunks using hnsw (embedding vector_cosine_ops);
create index if not exists chunks_fts_idx        on chunks using gin (fts);
create index if not exists chunks_collection_idx on chunks (collection);

-- Hybrid search: RRF-fuse dense (cosine) + lexical (ts_rank) rankings.
-- Called from the API via PostgREST:  rpc('hybrid_search', {...}).
create or replace function hybrid_search(
  query_text      text,
  query_embedding vector(1024),
  match_count     int  default 10,
  coll            text default null,
  pool            int  default 40,
  rrf_k           int  default 60
) returns table (
  chunk_id text, doc_id text, collection text, title text, section text,
  heading_path text, content text, page_span text, source_url text, score float
) language sql stable as $$
  with vec as (
    select c.chunk_id,
           row_number() over (order by c.embedding <=> query_embedding) as r
    from chunks c
    where coll is null or c.collection = coll
    order by c.embedding <=> query_embedding
    limit pool
  ),
  lex as (
    select c.chunk_id,
           row_number() over (
             order by ts_rank(c.fts, websearch_to_tsquery('english', query_text)) desc
           ) as r
    from chunks c
    where (coll is null or c.collection = coll)
      and c.fts @@ websearch_to_tsquery('english', query_text)
    limit pool
  ),
  fused as (
    select coalesce(v.chunk_id, l.chunk_id) as chunk_id,
           coalesce(1.0/(rrf_k + v.r), 0) + coalesce(1.0/(rrf_k + l.r), 0) as score
    from vec v full outer join lex l on v.chunk_id = l.chunk_id
  )
  select c.chunk_id, c.doc_id, c.collection, c.title, c.section, c.heading_path,
         c.content, c.page_span, c.source_url, f.score
  from fused f
  join chunks c on c.chunk_id = f.chunk_id
  order by f.score desc
  limit match_count;
$$;

-- Read-only public access to search (RLS): anon may SELECT + call the RPC,
-- but never write. Ingest uses the service role (bypasses RLS).
alter table documents enable row level security;
alter table chunks    enable row level security;
create policy "public read documents" on documents for select using (true);
create policy "public read chunks"    on chunks    for select using (true);
