// Grounded, cited answer over retrieved passages via OpenRouter (deepseek-v4-flash).
import { Hit } from "./search";

const URL = "https://openrouter.ai/api/v1/chat/completions";
const MODEL = process.env.OPENROUTER_GEN_MODEL || "deepseek/deepseek-v4-flash";
const REFUSAL = "The retrieved sources do not contain enough information to answer this.";

export async function answer(query: string, hits: Hit[]): Promise<{ text: string; grounded: boolean }> {
  const key = process.env.OPENROUTER_API_KEY?.trim();
  if (!key || hits.length === 0) return { text: REFUSAL, grounded: false };
  const ctx = hits
    .map((h, i) => `[${i + 1}] ${h.section || h.title} (${h.page_span})\n${(h.content || "").slice(0, 2000)}`)
    .join("\n\n");
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
