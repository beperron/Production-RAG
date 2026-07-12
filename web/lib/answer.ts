// Grounded, cited answer over retrieved passages via OpenRouter (deepseek-v4-flash).
// Expands each top hit with its neighboring chunks (same doc, adjacent ordinals)
// so enumerated statute bodies / multi-paragraph policy are present for grounding.
import { Hit } from "./search";

const URL = "https://openrouter.ai/api/v1/chat/completions";
const MODEL = process.env.OPENROUTER_GEN_MODEL || "deepseek/deepseek-v4-flash";
const REFUSAL = "The retrieved sources do not contain enough information to answer this.";

async function expand(h: Hit): Promise<string> {
  try {
    const url = process.env.SUPABASE_URL!.replace(/\/$/, "");
    const key = process.env.SUPABASE_ANON_KEY!;
    // chunk_id encodes the position as "<doc_id>:<ordinal>"
    const ord = parseInt(h.chunk_id.split(":")[1] || "0", 10) || 0;
    const lo = Math.max(0, ord - 1);
    const hi = ord + 3;
    const r = await fetch(
      `${url}/rest/v1/chunks?doc_id=eq.${h.doc_id}&ordinal=gte.${lo}&ordinal=lte.${hi}&order=ordinal&select=content`,
      { headers: { apikey: key, Authorization: `Bearer ${key}` } }
    );
    if (r.ok) {
      const rows = (await r.json()) as { content: string }[];
      if (rows.length) return rows.map((x) => x.content).join("\n");
    }
  } catch {}
  return h.content || "";
}

export async function answer(query: string, hits: Hit[]): Promise<{ text: string; grounded: boolean }> {
  const key = process.env.OPENROUTER_API_KEY?.trim();
  if (!key || hits.length === 0) return { text: REFUSAL, grounded: false };
  // expand the top 6 hits (one block per distinct document) with neighbor context
  const seen = new Set<string>();
  const top = hits.filter((h) => (seen.has(h.doc_id) ? false : (seen.add(h.doc_id), true))).slice(0, 6);
  const blocks = await Promise.all(top.map(async (h, i) =>
    `[${i + 1}] ${h.section || h.title} (${h.page_span})\n${(await expand(h)).slice(0, 2600)}`));
  const ctx = blocks.join("\n\n");
  const system =
    "You answer questions about North Carolina child-welfare law and policy using ONLY the numbered sources provided. " +
    "Cite every claim inline with [n]. If the sources do not support an answer, reply exactly: " +
    `"${REFUSAL}" Do not use outside knowledge.`;
  const user = `Question: ${query}\n\nSources:\n${ctx}`;
  const r = await fetch(URL, {
    method: "POST",
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
    body: JSON.stringify({ model: MODEL, temperature: 0, max_tokens: 1100,
      messages: [{ role: "system", content: system }, { role: "user", content: user }] }),
  });
  if (!r.ok) return { text: REFUSAL, grounded: false };
  const d = await r.json();
  const text = (d?.choices?.[0]?.message?.content || "").trim();
  const grounded = !!text && /\[\d+\]/.test(text) && !text.includes(REFUSAL);
  return { text: text || REFUSAL, grounded };
}
