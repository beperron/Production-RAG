// Grounded-answer context builder. Numbers sources in search order (so [n] in
// the answer maps to source card n) and neighbor-expands each with adjacent
// chunks so enumerated statute bodies / multi-paragraph policy are present.
import { Hit } from "./search";
import { clean } from "./clean";

export const REFUSAL = "The retrieved sources do not contain enough information to answer this.";
export const SYSTEM =
  "You answer questions about North Carolina child-welfare law and policy using ONLY the numbered sources provided. " +
  "Write in clear Markdown (short paragraphs and bullet lists). Cite every claim inline with [n] matching the source numbers. " +
  `If the sources do not support an answer, reply exactly: "${REFUSAL}" Do not use outside knowledge.`;

async function expand(h: Hit): Promise<string> {
  try {
    const url = process.env.SUPABASE_URL!.replace(/\/$/, "");
    const key = process.env.SUPABASE_ANON_KEY!;
    const ord = parseInt(h.chunk_id.split(":")[1] || "0", 10) || 0;
    const r = await fetch(
      `${url}/rest/v1/chunks?doc_id=eq.${h.doc_id}&ordinal=gte.${Math.max(0, ord - 1)}&ordinal=lte.${ord + 3}&order=ordinal&select=content`,
      { headers: { apikey: key, Authorization: `Bearer ${key}` } }
    );
    if (r.ok) {
      const rows = (await r.json()) as { content: string }[];
      if (rows.length) return clean(rows.map((x) => x.content).join("\n"));
    }
  } catch {}
  return clean(h.content || "");
}

// Build the numbered-source context string for the top N hits (search order).
export async function buildContext(hits: Hit[], n = 8): Promise<string> {
  const top = hits.slice(0, n);
  const blocks = await Promise.all(
    top.map(async (h, i) => `[${i + 1}] ${h.section || h.title} (${h.page_span})\n${(await expand(h)).slice(0, 2600)}`)
  );
  return blocks.join("\n\n");
}
