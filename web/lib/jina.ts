// Jina embeddings (query) + reranker. Server-side only.
const EMB_URL = "https://api.jina.ai/v1/embeddings";
const RR_URL = "https://api.jina.ai/v1/rerank";
const RR_MODEL = "jina-reranker-v2-base-multilingual";

export async function embedQuery(q: string): Promise<number[] | null> {
  const key = process.env.JINA_API_KEY?.trim();
  if (!key) return null;
  const r = await fetch(EMB_URL, {
    method: "POST",
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
    body: JSON.stringify({ model: "jina-embeddings-v3", task: "retrieval.query", dimensions: 1024, input: [q] }),
  });
  if (!r.ok) return null;
  const d = await r.json();
  return d?.data?.[0]?.embedding ?? null;
}

// Returns per-document relevance scores aligned to `docs`, or null on failure.
export async function rerankScores(query: string, docs: string[]): Promise<number[] | null> {
  const key = process.env.JINA_API_KEY?.trim();
  if (!key || docs.length === 0) return null;
  const r = await fetch(RR_URL, {
    method: "POST",
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
    body: JSON.stringify({ model: RR_MODEL, query, documents: docs, top_n: docs.length }),
  });
  if (!r.ok) return null;
  const d = await r.json();
  const scores = new Array(docs.length).fill(0);
  for (const it of d?.results ?? []) scores[it.index] = it.relevance_score ?? 0;
  return scores;
}
