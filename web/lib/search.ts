// Server-side search orchestration: embed query -> Supabase hybrid_search RPC ->
// Jina rerank blended with the RRF rank (alpha=0.5, the measured-best config).
import { embedQuery, rerankScores } from "./jina";
import { clean } from "./clean";

export type Hit = {
  chunk_id: string; doc_id: string; collection: string; title: string;
  section: string; heading_path: string; content: string; page_span: string;
  source_url: string; score: number;
};

const BLEND = parseFloat(process.env.LAWSEARCH_RERANK_BLEND || "0.5");

async function rpcHybrid(query: string, embedding: number[] | null, coll: string | null, pool = 40): Promise<Hit[]> {
  const url = process.env.SUPABASE_URL!.replace(/\/$/, "");
  const key = process.env.SUPABASE_ANON_KEY!;
  const r = await fetch(`${url}/rest/v1/rpc/hybrid_search`, {
    method: "POST",
    headers: { apikey: key, Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
    body: JSON.stringify({
      query_text: query,
      query_embedding: embedding, // null -> lexical-only fallback still ranks via fts
      match_count: pool, coll, pool,
    }),
  });
  if (!r.ok) throw new Error(`hybrid_search ${r.status}: ${(await r.text()).slice(0, 200)}`);
  return (await r.json()) as Hit[];
}

export async function search(query: string, k = 10, collection: string | null = null): Promise<Hit[]> {
  const emb = await embedQuery(query);
  const fused = await rpcHybrid(query, emb, collection, 40);
  if (fused.length === 0) return [];
  const scores = await rerankScores(query, fused.map((h) => h.content?.slice(0, 1200) || ""));
  if (!scores) return fused.slice(0, k);
  // Full Jina rerank order. Measured on the Supabase/Jina-v3 stack: pure rerank
  // beats the old RRF-blend (statutes R@3 80->90%, policy 92->96%) because the
  // hybrid prior is weak here and blending it dragged Jina down.
  const ranked = fused
    .map((h, i) => ({ h, s: scores[i] }))
    .sort((a, b) => b.s - a.s);
  return ranked.slice(0, k).map((x) => ({ ...x.h, content: clean(x.h.content), section: clean(x.h.section), title: clean(x.h.title) }));
}
