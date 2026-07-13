import { NextRequest } from "next/server";
import { search } from "@/lib/search";
import { buildContext, SYSTEM, REFUSAL } from "@/lib/answer";

export const runtime = "nodejs";
export const maxDuration = 60;

// Grounded answers via Ollama Cloud (deepseek-v4-flash). Same model/prompt as
// before; consolidated onto Ollama and faster than the OpenRouter path.
const OLLAMA_URL = "https://ollama.com/api/chat";
const MODEL = process.env.OLLAMA_GEN_MODEL || "deepseek-v4-flash:cloud";

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
        const key = process.env.OLLAMA_KEY?.trim();
        if (!q || hits.length === 0 || !key) {
          controller.enqueue(enc.encode(REFUSAL));
          controller.close();
          return;
        }
        const ctx = await buildContext(hits);
        const r = await fetch(OLLAMA_URL, {
          method: "POST",
          headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json" },
          body: JSON.stringify({
            model: MODEL, stream: true, options: { temperature: 0, num_predict: 1100 },
            messages: [{ role: "system", content: SYSTEM }, { role: "user", content: `Question: ${q}\n\nSources:\n${ctx}` }],
          }),
        });
        if (!r.ok || !r.body) {
          controller.enqueue(enc.encode(REFUSAL));
          controller.close();
          return;
        }
        // Ollama streams newline-delimited JSON: {"message":{"content":"…"},"done":bool}
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
            if (!t) continue;
            try {
              const tok = JSON.parse(t)?.message?.content;
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
