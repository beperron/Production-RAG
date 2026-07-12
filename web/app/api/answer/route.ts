import { NextRequest } from "next/server";
import { search } from "@/lib/search";
import { buildContext, SYSTEM, REFUSAL } from "@/lib/answer";

export const runtime = "nodejs";
export const maxDuration = 60;

const OR_URL = "https://openrouter.ai/api/v1/chat/completions";
const MODEL = process.env.OPENROUTER_GEN_MODEL || "deepseek/deepseek-v4-flash";

// Streams: first line is a JSON control message {"hits":[...]} (for the source
// cards), then the grounded answer streams token-by-token as plain text.
export async function GET(req: NextRequest) {
  const q = (req.nextUrl.searchParams.get("q") || "").trim();
  const collection = req.nextUrl.searchParams.get("collection") || null;
  const enc = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      try {
        const hits = q ? await search(q, 10, collection) : [];
        controller.enqueue(enc.encode(JSON.stringify({ hits }) + "\n"));
        const key = process.env.OPENROUTER_API_KEY?.trim();
        if (!q || hits.length === 0 || !key) {
          controller.enqueue(enc.encode(REFUSAL));
          controller.close();
          return;
        }
        const ctx = await buildContext(hits);
        const r = await fetch(OR_URL, {
          method: "POST",
          headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
          body: JSON.stringify({
            model: MODEL, temperature: 0, max_tokens: 1100, stream: true,
            messages: [{ role: "system", content: SYSTEM }, { role: "user", content: `Question: ${q}\n\nSources:\n${ctx}` }],
          }),
        });
        if (!r.ok || !r.body) {
          controller.enqueue(enc.encode(REFUSAL));
          controller.close();
          return;
        }
        const reader = r.body.getReader();
        const dec = new TextDecoder();
        let buf = "";
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          const lines = buf.split("\n");
          buf = lines.pop() || "";
          for (const line of lines) {
            const t = line.trim();
            if (!t.startsWith("data:")) continue;
            const d = t.slice(5).trim();
            if (d === "[DONE]") continue;
            try {
              const tok = JSON.parse(d)?.choices?.[0]?.delta?.content;
              if (tok) controller.enqueue(enc.encode(tok));
            } catch {}
          }
        }
        controller.close();
      } catch (e: any) {
        controller.enqueue(enc.encode("\n\n[error: " + String(e?.message || e).slice(0, 120) + "]"));
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no" },
  });
}
