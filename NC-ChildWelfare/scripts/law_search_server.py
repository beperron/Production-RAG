#!/usr/bin/env python3
"""Public NC law & policy semantic-search server — PUBLIC DATA ONLY.

A clean, localhost-only search surface over the public knowledge bases
(legal-authorities + nc-child-welfare), styled to match the Coursewright
"paper / seal-red" console theme. Hybrid BM25F + GTE semantic retrieval with
full source-traceable citations; an opt-in grounded-answer toggle (local
Qwen3.6 via Ollama).

It refuses any confidential workspace — see parsevault.lawsearch (the engine
that must never be pointed at a confidential workspace).

Run:  python scripts/law_search_server.py [--port 8900]
"""
from __future__ import annotations

import argparse
import html
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Optional cloud keys (Jina rerank / OpenRouter query rewrite) for this PUBLIC
# surface. Loaded from a gitignored operator-owned file so no secret is ever
# entered by the assistant or committed to the repo. You populate this file:
#   ~/.config/parsevault/lawsearch.env   with lines like  JINA_API_KEY=...
_SECRETS = Path.home() / ".config" / "parsevault" / "lawsearch.env"


def _load_secrets() -> None:
    if _SECRETS.is_file():
        for line in _SECRETS.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from parsevault.lawsearch import PUBLIC_COLLECTIONS, LawSearch  # noqa: E402

_ENGINE: LawSearch | None = None

_CSS = """
:root{--bg:#FBF9F4;--bg2:#F4F1E8;--card:#FFFFFF;--soft:#F7F4EC;--ink:#191A1C;
--muted:#6E6A60;--line:#E6E2D6;--line-soft:#EDE9DD;--seal:#C5302C;--seal-deep:#9C2420;
--ok:#3F7A57;--link:#2A5560;--display:'Besley',Georgia,serif;--mono:'JetBrains Mono',ui-monospace,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 'IBM Plex Sans',system-ui,sans-serif}
a{color:var(--link)}
.banner{background:#FCEEEE;border-bottom:1px solid var(--seal);color:var(--seal-deep);
  font:12px/1.4 var(--mono);padding:6px 20px;text-align:center;letter-spacing:.02em}
header{padding:26px 20px 8px;max-width:960px;margin:0 auto}
h1{font:600 26px/1.2 var(--display);margin:0 0 2px}
.sub{color:var(--muted);font-size:13px;margin:0 0 18px}
form.search{max-width:960px;margin:0 auto;padding:0 20px}
.searchrow{position:relative}
.search input[type=text]{width:100%;height:54px;padding:0 20px;background:#fff;border:1px solid var(--line);
  border-radius:10px;font-size:16px;color:var(--ink)}
.search input[type=text]:focus{outline:2px solid var(--seal);border-color:var(--seal)}
.controls{display:flex;gap:12px;align-items:center;margin:12px 0 4px;flex-wrap:wrap}
.controls select,.controls button{font:13px var(--mono);padding:8px 12px;border:1px solid var(--line);
  border-radius:8px;background:#fff;color:var(--ink);cursor:pointer}
.controls button{background:var(--seal);color:#fff;border-color:var(--seal-deep);font-weight:600}
.chk{font:12px var(--mono);color:var(--muted);display:flex;align-items:center;gap:6px}
main{max-width:960px;margin:14px auto 60px;padding:0 20px}
.meta{color:var(--muted);font:12px var(--mono);margin:8px 0 14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:0 0 12px}
.card .sec{font:600 15px/1.3 var(--display);margin:0 0 2px;color:var(--ink)}
.card .title{color:var(--muted);font-size:12.5px;margin:0 0 8px}
.card .snip{font-size:14px;color:#2b2c2e;margin:0 0 10px}
.card .snip mark{background:#F4E5E2;color:var(--seal-deep);padding:0 2px;border-radius:2px}
.badges{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.badge{font:11px var(--mono);border:1px solid var(--line);border-radius:999px;padding:2px 9px;color:var(--muted)}
.badge.coll{border-color:#C9BE9E;background:#FBF6E7;color:#7A5300}
.badge.lane{background:#EDF0F2;border-color:#DCE3E6;color:#44505A}
.badge.pg{background:#F0F1F2}
.cite{font:11.5px/1.4 var(--mono);color:var(--muted);margin-top:8px;word-break:break-word}
.cite a{color:var(--link)}
.answer{background:#F7FAF8;border:1px solid #CFE3D6;border-left:4px solid var(--ok);
  border-radius:10px;padding:16px 18px;margin:0 0 18px}
.answer h3{margin:0 0 8px;font:600 14px var(--mono);color:var(--ok)}
.answer .body{font-size:15px;line-height:1.6}
.answer .flag{color:var(--seal-deep);font:12px var(--mono)}
.empty{color:var(--muted);text-align:center;padding:40px 0;font-style:italic}
footer{max-width:960px;margin:0 auto;padding:20px;color:var(--muted);font:11px var(--mono);border-top:1px solid var(--line-soft)}
"""


def _page(query: str, collection: str, want_answer: bool, body: str) -> str:
    opts = ['<option value="">All public sources</option>']
    for name, label in PUBLIC_COLLECTIONS.items():
        sel = " selected" if collection == name else ""
        opts.append(f'<option value="{name}"{sel}>{html.escape(label)}</option>')
    ans_chk = " checked" if want_answer else ""
    q = html.escape(query)
    return f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>NC Law &amp; Policy Search</title><style>{_CSS}</style></head><body>
<div class=banner>PUBLIC North Carolina law &amp; policy only · no confidential case data · localhost</div>
<header><h1>NC Law &amp; Policy Search</h1>
<p class=sub>Semantic search over NC Administrative Code, General Statutes, and NCDHHS child-welfare policy — every result source-traceable.</p></header>
<form class=search method=get action="/search">
  <div class=searchrow><input type=text name=q value="{q}" placeholder="Search statutes, regulations, and policy — e.g. grounds for termination of parental rights" autofocus></div>
  <div class=controls>
    <select name=collection>{''.join(opts)}</select>
    <label class=chk><input type=checkbox name=answer value=1{ans_chk}> grounded answer (local model)</label>
    <button type=submit>Search</button>
  </div>
</form>
<main>{body}</main>
<footer>parsevault · public NC-law engine · {_pipeline_label()}</footer>
</body></html>"""


def _pipeline_label() -> str:
    st = _ENGINE.enhancement_status() if _ENGINE else {}
    rw = "Qwen-4B rewrite ✓" if st.get("query_rewrite") else "query-rewrite off"
    jr = "Jina rerank ✓" if st.get("jina_rerank") else "Jina rerank off"
    return f"{html.escape(rw)} → BM25F + GTE, RRF-fused → {html.escape(jr)} · source-traceable"


def _hit_card(h) -> str:
    coll = PUBLIC_COLLECTIONS.get(h.collection, h.collection).split("—")[0].strip()
    sec = html.escape(h.section or h.title or "(untitled)")
    title = html.escape(h.title) if (h.section and h.title and h.section != h.title) else ""
    snip = html.escape(h.snippet)
    badges = [f'<span class="badge coll">{html.escape(coll)}</span>']
    if h.lane:
        badges.append(f'<span class="badge lane">{html.escape(h.lane)}</span>')
    if h.page_span:
        badges.append(f'<span class="badge pg">{html.escape(h.page_span)}</span>')
    cite = html.escape(h.citation)
    if h.source_url:
        cite = cite.replace(html.escape(h.source_url),
                            f'<a href="{html.escape(h.source_url)}" target=_blank rel=noopener>{html.escape(h.source_url)}</a>')
    return f"""<div class=card>
  <div class=sec>{sec}</div>
  {f'<div class=title>{title}</div>' if title else ''}
  <div class=snip>{snip}</div>
  <div class=badges>{''.join(badges)}</div>
  <div class=cite>{cite}</div>
</div>"""


def _render_search(query: str, collection: str, want_answer: bool) -> str:
    assert _ENGINE is not None
    parts: list[str] = []
    if want_answer and query.strip():
        try:
            coll, res = _ENGINE.answer(query, k=10, collection=collection or None)
            ans = html.escape(getattr(res, "answer", "") or "")
            grounded = getattr(res, "grounded", False)
            flag = "" if grounded else '<div class=flag>⚠ not grounded — showing sources only</div>'
            if ans:
                parts.append(f'<div class=answer><h3>Grounded answer · {html.escape(PUBLIC_COLLECTIONS.get(coll, coll).split("—")[0].strip())}</h3>'
                             f'<div class=body>{ans}</div>{flag}</div>')
        except Exception as e:
            parts.append(f'<div class=answer><h3>Answer unavailable</h3>'
                         f'<div class=flag>{html.escape(str(e)[:160])}</div></div>')
    hits = _ENGINE.search(query, k=12, collection=collection or None) if query.strip() else []
    if query.strip():
        parts.append(f'<div class=meta>{len(hits)} result(s) · hybrid BM25F + semantic</div>')
    if hits:
        parts.extend(_hit_card(h) for h in hits)
    elif query.strip():
        parts.append('<div class=empty>No matching public statutes, regulations, or policy.</div>')
    else:
        parts.append('<div class=empty>Enter a query to search NC law &amp; policy.</div>')
    return "".join(parts)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, body: str, status: int = 200, ctype: str = "text/html; charset=utf-8"):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/health":
            return self._send("ok", ctype="text/plain")
        qs = parse_qs(u.query)
        query = (qs.get("q", [""])[0]).strip()
        collection = qs.get("collection", [""])[0]
        want_answer = qs.get("answer", [""])[0] in ("1", "true", "on")
        if u.path in ("/", "/search"):
            body = _render_search(query, collection, want_answer) if u.path == "/search" or query \
                else '<div class=empty>Enter a query to search NC law &amp; policy.</div>'
            return self._send(_page(query, collection, want_answer, body))
        self._send('<div class=empty>Not found.</div>', status=404)


def main():
    global _ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    _load_secrets()
    st = {"JINA_API_KEY": bool(os.environ.get("JINA_API_KEY")),
          "OPENROUTER_API_KEY": bool(os.environ.get("OPENROUTER_API_KEY"))}
    print(f"secrets loaded from {_SECRETS if _SECRETS.is_file() else '(none)'}: {st}", flush=True)
    print("loading public NC law/policy indexes (GTE-base)…", flush=True)
    _ENGINE = LawSearch()
    print(f"loaded: {_ENGINE.collections()}", flush=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"NC Law & Policy Search → http://{args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
