-- Contextual Retrieval (Anthropic): per-chunk context blurb feeds BOTH the
-- embedding (done at ingest) and the BM25/full-text lane (Contextual BM25).

alter table chunks add column if not exists context text default '';

-- rebuild the generated full-text column to include the context blurb
alter table chunks drop column if exists fts;
alter table chunks add column fts tsvector generated always as (
  to_tsvector('english',
    coalesce(context,'') || ' ' || coalesce(title,'') || ' ' ||
    coalesce(section,'') || ' ' || content)
) stored;
create index if not exists chunks_fts_idx on chunks using gin (fts);

-- recreate hybrid_search so results also carry the context blurb
create or replace function hybrid_search(
  query_text text, query_embedding vector(1024),
  match_count int default 10, coll text default null,
  pool int default 40, rrf_k int default 60
) returns table (
  chunk_id text, doc_id text, collection text, title text, section text,
  heading_path text, content text, context text, page_span text,
  source_url text, score float
) language sql stable as $$
  with vec as (
    select c.chunk_id, row_number() over (order by c.embedding <=> query_embedding) as r
    from chunks c where coll is null or c.collection = coll
    order by c.embedding <=> query_embedding limit pool
  ),
  lex as (
    select c.chunk_id, row_number() over (
             order by ts_rank(c.fts, websearch_to_tsquery('english', query_text)) desc) as r
    from chunks c
    where (coll is null or c.collection = coll)
      and c.fts @@ websearch_to_tsquery('english', query_text) limit pool
  ),
  fused as (
    select coalesce(v.chunk_id, l.chunk_id) as chunk_id,
           coalesce(1.0/(rrf_k + v.r), 0) + coalesce(1.0/(rrf_k + l.r), 0) as score
    from vec v full outer join lex l on v.chunk_id = l.chunk_id
  )
  select c.chunk_id, c.doc_id, c.collection, c.title, c.section, c.heading_path,
         c.content, c.context, c.page_span, c.source_url, f.score
  from fused f join chunks c on c.chunk_id = f.chunk_id
  order by f.score desc limit match_count;
$$;
