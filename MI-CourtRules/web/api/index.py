"""Vercel entrypoint for the Michigan Court Rules bench book.

A single raw-ASGI app wrapping pipeline/serve.py's rendering unchanged.
Retrieval, prompts, the citation audit and the page HTML are the exact
functions the local server runs; only the engine underneath is the cloud
variant (OpenRouter embeddings + Supabase pgvector) and the query log
writes to mcr.* instead of SQLite. Blocking work runs in threads; the SSE
stream forwards generator tokens through a queue.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import queue
import sys
import threading
import time
import urllib.parse

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))

import serve                                            # noqa: E402
from cloud import CloudEngine, QueryLogPG               # noqa: E402
from provenance import Ledger                           # noqa: E402

_BOOT = threading.Lock()


def _init():
    with _BOOT:
        if serve.ENGINE is None:
            eng = CloudEngine(quiet=True)
            serve.LEDGER = Ledger()
            serve.QLOG = QueryLogPG(eng.db)
            serve.ENGINE = eng
    return serve.ENGINE


STATE_PDF = ("https://www.courts.michigan.gov/siteassets/rules-instructions-"
             "administrative-orders/michigan-court-rules/"
             "michigan-court-rules.pdf")


async def _respond(send, body, ctype, code=200, extra=()):
    headers = [(b"content-type", ctype.encode()),
               (b"cache-control", b"no-store" if b"html" in ctype.encode()
                else b"public, max-age=3600"), *extra]
    await send({"type": "http.response.start", "status": code,
                "headers": headers})
    await send({"type": "http.response.body", "body": body})


async def _read_body(receive):
    body = b""
    while True:
        m = await receive()
        body += m.get("body", b"")
        if not m.get("more_body"):
            return body


def _sse(event, data):
    return f"event: {event}\ndata: {data}\n\n".encode()


def _stream_worker(q_text, out):
    """Thread: retrieval happened already? No -- do everything blocking here,
    pushing SSE frames into the queue; None terminates."""
    try:
        eng = serve.ENGINE
        t0 = time.time()
        prep = eng.prepare(q_text)
        parts = []
        try:
            for tok in eng.stream_generate(prep["user"]):
                parts.append(tok)
                out.put(_sse("token", json.dumps(tok)))
        except Exception as exc:                         # noqa: BLE001
            parts.append(f"(generation unavailable: {exc})")
        ans = "".join(parts)
        r = {"question": q_text, "answer": ans, "hits": prep["hits"],
             "model": prep["model"]}
        dt = time.time() - t0
        answer_html, below, refused, head, v = serve.finish_html(q_text, r, dt)
        out.put(_sse("final", json.dumps({
            "answer_html": answer_html, "below": below,
            "refused": refused, "head": head,
            "announce": f"{head}. {len(prep['hits'])} supporting provisions, "
                        f"{v['n']} citations audited."})))
    except Exception as exc:                             # noqa: BLE001
        out.put(_sse("token", json.dumps(f"(search unavailable: {exc})")))
        out.put(_sse("final", json.dumps({
            "answer_html": "The service could not reach its index just now. "
                           "Please try again.",
            "below": "", "refused": True, "head": "Temporarily unavailable",
            "announce": "Temporarily unavailable"})))
    finally:
        out.put(None)


async def app(scope, receive, send):
    if scope["type"] != "http":
        return
    path = scope["path"]
    qs = urllib.parse.parse_qs(scope.get("query_string", b"").decode())
    if "x_path" in qs:
        # the catch-all rewrite hands us the original path here; the
        # original query params are merged alongside it
        path = "/" + qs.pop("x_path")[0].lstrip("/")
    elif path.startswith("/api/index"):
        path = "/"

    if path == "/health":
        eng = await asyncio.to_thread(_init)
        return await _respond(send, json.dumps({
            "ok": True, "chunks": len(eng.chunks),
            "blocks": len(serve.LEDGER.blocks),
            "model": serve.GEN_MODEL}).encode(), "application/json")

    if path == "/source.pdf":
        return await _respond(send, b"", "text/plain", 302,
                              [(b"location", STATE_PDF.encode())])

    if path == "/favicon.svg":
        f = ROOT / "static" / "favicon.svg"
        if f.exists():
            return await _respond(send, f.read_bytes(), "image/svg+xml")

    if path.startswith("/static/fonts/") and path.endswith(".woff2"):
        f = ROOT / "static" / "fonts" / pathlib.Path(path).name
        if f.exists():
            return await _respond(send, f.read_bytes(), "font/woff2")

    if path == "/api/feedback" and scope["method"] == "POST":
        await asyncio.to_thread(_init)
        try:
            d = json.loads((await _read_body(receive)) or b"{}")
            ok = await asyncio.to_thread(
                serve.QLOG.set_feedback,
                str(d.get("answer_id", ""))[:32], int(d.get("vote", 0)))
        except Exception:                                # noqa: BLE001
            ok = False
        return await _respond(send, json.dumps({"ok": bool(ok)}).encode(),
                              "application/json", 200 if ok else 400)

    if path == "/api/stream":
        await asyncio.to_thread(_init)
        q_text = (qs.get("q") or [""])[0]
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/event-stream"),
                                (b"cache-control", b"no-store")]})
        if not q_text.strip():
            await send({"type": "http.response.body",
                        "body": _sse("gone", "{}")})
            return
        out: queue.Queue = queue.Queue()
        threading.Thread(target=_stream_worker, args=(q_text, out),
                         daemon=True).start()
        while True:
            frame = await asyncio.to_thread(out.get)
            if frame is None:
                await send({"type": "http.response.body", "body": b""})
                return
            await send({"type": "http.response.body", "body": frame,
                        "more_body": True})

    if path in ("/architecture", "/about", "/how-it-works"):
        await asyncio.to_thread(_init)
        page = await asyncio.to_thread(serve.architecture_page)
        return await _respond(send, page.encode(), "text/html; charset=utf-8")

    if path in ("/", "/index.html"):
        await asyncio.to_thread(_init)
        q_text = (qs.get("q") or [""])[0]

        def build():
            if q_text.strip() and not qs.get("nojs"):
                # the shell page defers retrieval entirely to /api/stream --
                # on serverless the two requests may land on different
                # instances, so nothing is stashed in memory between them
                return serve.page(q_text, _shell_noprep(q_text), "")
            body, announce = serve.render(q_text)
            return serve.page(q_text, body, announce)

        html = await asyncio.to_thread(build)
        return await _respond(send, html.encode(), "text/html; charset=utf-8")

    await _respond(send, b"Not found", "text/plain; charset=utf-8", 404)


def _shell_noprep(q_text):
    """serve.render_shell without the ENGINE.prepare + stash: the stream
    endpoint re-derives retrieval from &q= (its qid-miss fallback path)."""
    import urllib.parse as up
    return f"""
<section class="answer" id="ansblock" aria-labelledby="ans">
  <h2 id="ans">Answer</h2>
  <div class="body" id="anstext"><span class="spinner" aria-hidden="true"></span>
  <em style="color:var(--muted)"> Composing a cited answer&hellip;</em></div>
</section>
<div id="below"></div>
<script>
(function(){{
  var box=document.getElementById('anstext'),
      below=document.getElementById('below'), started=false,
      live=document.getElementById('live'),
      es=new EventSource('/api/stream?q='+encodeURIComponent({q_text!r}));
  es.addEventListener('token',function(e){{
    if(!started){{box.textContent='';started=true;}}
    box.textContent+=JSON.parse(e.data);
  }});
  es.addEventListener('final',function(e){{
    var d=JSON.parse(e.data);
    box.innerHTML=d.answer_html;
    below.innerHTML=d.below;
    if(d.refused){{
      document.getElementById('ansblock').classList.add('refused');
      document.getElementById('ans').textContent=d.head;
    }}
    if(live) live.textContent=d.announce;
    es.close();
  }});
  es.addEventListener('gone',function(){{es.close();
    window.location='/?q='+encodeURIComponent({q_text!r})+'&nojs=1';}});
  es.onerror=function(){{es.close();}};
}})();
</script>
<noscript><meta http-equiv="refresh"
  content="0;url=/?q={up.quote(q_text)}&nojs=1"></noscript>"""
