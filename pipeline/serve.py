#!/usr/bin/env python3
"""Michigan Court Rules — search with provenance.

    python3 pipeline/serve.py --port 8788   ->  http://127.0.0.1:8788

Server-rendered HTML from the stdlib: no build step, no framework, and no
external requests at run time -- Montserrat is self-hosted, so the interface
works on an air-gapped courtroom machine.

VISUAL DESIGN follows the Michigan Juvenile Data Dashboard's system: the same
palette tokens, Montserrat, 6px radii and shadow scale, so this reads as part
of the same family of court tools rather than a separate product.

ACCESSIBILITY -- WCAG 2.1 level AA, and the parts that matter here are not
decoration. A court system must be usable by staff with low vision, colour
vision deficiency, or a screen reader:

  * every palette colour used for text clears 4.5:1 on white, measured, not
    assumed (primary 4.95, success 4.85, error 7.11, caption 4.95)
  * RETRIEVAL ROUTE IS NEVER CONVEYED BY COLOUR ALONE. Each passage carries a
    text label and a symbol as well as a border colour, because the difference
    between "matched the citation you typed" and "ranked by similarity" is
    evidentiary and must survive greyscale
  * the citation audit is a real table with a caption and scoped headers, so a
    screen reader announces what each cell means
  * skip link, landmarks, one h1, visible focus rings at 3:1, 44px targets
  * results announced via aria-live so a screen-reader user knows the page
    changed and what came back
"""
from __future__ import annotations

import argparse
import html
import pathlib
import sys
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from mcr_search import Engine, GEN_MODEL, EMBEDDER      # noqa: E402
from provenance import Ledger                          # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
ENGINE: Engine | None = None
LEDGER: Ledger | None = None

EXAMPLES = [
    "How long does a defendant served in Michigan have to answer a complaint?",
    "MCR 2.116(C)(10)",
    "Can I serve process on someone confined in a psychiatric facility?",
    "When must a court appoint a guardian ad litem for a minor?",
    "When can a PPO action be dismissed?",
]

# label, short code, symbol, explanation. The symbol and the words carry the
# meaning; the colour only reinforces it.
ROUTE = {
    "citation-router": ("Exact citation", "router", "✦",
                        "Matched the citation you typed, not a similarity score."),
    "dense": ("Semantic match", "dense", "◆",
              "Ranked by meaning against the retrieval model."),
    "cross-reference": ("Supplied as a condition", "xref", "▲",
                        "An earlier passage makes this one a condition on itself."),
}

CSS = """
@font-face{font-family:Montserrat;font-style:normal;font-weight:400 700;
font-display:swap;src:url(/static/fonts/montserrat-1.woff2) format('woff2')}
:root{
--primary:#277C78;--primary-hover:#1d605d;--light-teal:#61C8AF;
--secondary:#142D3E;--secondary-2:#1f4257;
--success:#0D8252;--success-bg:#EAFBEB;
--warning:#E5A612;--warning-bg:#FFEFCA;--warning-text:#AE5400;
--error:#B30518;--error-bg:#FFE8E3;
--info:#0077A7;--info-bg:#E3F7FF;
--header-copy:#161618;--body-copy:#353535;--caption:#707070;
--border:#D0D0D0;--alt-bg:#F7F7F7;--smoke:#EAF1F4;--white:#fff;
--font:"Montserrat",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
--ls:0.3px;--radius:6px;--maxw:1180px;
--shadow:0 1px 3px rgba(20,45,62,.12),0 1px 2px rgba(20,45,62,.08);
--shadow-lg:0 4px 16px rgba(20,45,62,.14)}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--white);color:var(--body-copy);
font:16px/1.6 var(--font);letter-spacing:var(--ls)}
.skip{position:absolute;left:-9999px;top:0;background:var(--secondary);
color:#fff;padding:12px 18px;z-index:100;border-radius:0 0 var(--radius) 0}
.skip:focus{left:0}
:focus-visible{outline:3px solid var(--secondary);outline-offset:2px;border-radius:3px}
.bar{background:var(--secondary);color:#fff;padding:14px 20px}
.bar .in{max-width:var(--maxw);margin:0 auto;display:flex;gap:12px;align-items:center}
.bar img{width:34px;height:34px;flex:none}
.bar h1{font-size:18px;margin:0;font-weight:700;letter-spacing:.2px}
.bar p{margin:1px 0 0;font-size:12.5px;color:#c9d6de}
.sub{background:var(--smoke);border-bottom:1px solid var(--border);padding:14px 20px}
.sub .in,main,.foot .in{max-width:var(--maxw);margin:0 auto}
main{padding:24px 20px 64px}
form{display:flex;gap:10px;flex-wrap:wrap}
label.vh{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);
white-space:nowrap}
input[type=search]{flex:1;min-width:290px;min-height:46px;padding:11px 14px;
font:inherit;border:1px solid var(--border);border-radius:var(--radius);
background:#fff;color:var(--body-copy)}
button{min-height:46px;padding:11px 22px;font:inherit;font-weight:600;
border:1px solid transparent;border-radius:var(--radius);background:var(--primary);
color:#fff;cursor:pointer}
button:hover{background:var(--primary-hover)}
.meta{color:var(--caption);font-size:12.5px;margin-top:10px}
.meta code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px}
.ex{margin-top:12px}
.ex h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;
color:var(--caption);margin:0 0 7px;font-weight:700}
.ex a{display:inline-block;margin:0 6px 7px 0;padding:8px 13px;min-height:38px;
font-size:13.5px;color:var(--secondary-2);background:#fff;text-decoration:none;
border:1px solid var(--border);border-radius:20px}
.ex a:hover{border-color:var(--primary);color:var(--primary-hover)}
.answer{border:1px solid var(--border);border-left:4px solid var(--primary);
border-radius:var(--radius);padding:16px 18px;background:var(--success-bg);
box-shadow:var(--shadow);margin-bottom:14px;white-space:pre-wrap}
.answer.refused{border-left-color:var(--warning-text);background:var(--warning-bg)}
.answer h2{margin:0 0 9px;font-size:12px;font-weight:700;letter-spacing:.08em;
text-transform:uppercase;color:var(--secondary-2)}
table{width:100%;border-collapse:collapse;font-size:13.5px;background:#fff}
caption{text-align:left;font-weight:700;font-size:13px;color:var(--secondary);
padding:12px 14px 8px}
th,td{padding:8px 12px;border-bottom:1px solid var(--border);text-align:left}
th{font-size:11.5px;text-transform:uppercase;letter-spacing:.06em;
color:var(--caption);font-weight:700}
tbody tr:last-child td{border-bottom:none}
.audit{border:1px solid var(--border);border-radius:var(--radius);
overflow:hidden;margin-bottom:26px;box-shadow:var(--shadow)}
.ok{color:var(--success);font-weight:600}
.no{color:var(--error);font-weight:600}
.note{color:var(--warning-text);font-weight:600}
.hit{border:1px solid var(--border);border-left:4px solid var(--caption);
border-radius:var(--radius);padding:14px 16px;margin-bottom:12px;background:#fff;
box-shadow:var(--shadow)}
.hit.router{border-left-color:var(--success)}
.hit.dense{border-left-color:var(--primary)}
.hit.xref{border-left-color:var(--warning-text)}
.hit h3{margin:0;font-size:16.5px;color:var(--secondary);font-weight:700}
.tags{margin:6px 0 4px}
.tag{display:inline-block;font-size:11.5px;padding:2px 9px;margin:0 6px 5px 0;
border:1px solid var(--border);border-radius:20px;color:var(--caption);
background:var(--alt-bg)}
.tag.route{border-color:var(--primary);color:var(--primary-hover);font-weight:600}
.tag.route.router{border-color:var(--success);color:var(--success)}
.tag.route.xref{border-color:var(--warning-text);color:var(--warning-text)}
.why{color:var(--caption);font-size:12.5px;margin:2px 0 9px}
.body{font-size:14.5px;white-space:pre-wrap;max-height:260px;overflow:auto;
padding-top:9px;border-top:1px solid var(--border);color:var(--body-copy)}
.lineage{margin-top:9px;font:11.5px ui-monospace,SFMono-Regular,Menlo,monospace;
color:var(--caption)}
.count{font-size:13px;color:var(--caption);margin:20px 0 12px}
.beta{display:inline-block;margin-left:10px;padding:2px 9px;border-radius:20px;
background:var(--warning-bg);color:var(--secondary);font-size:11.5px;
font-weight:700;letter-spacing:.06em;text-transform:uppercase;vertical-align:middle}
.disclaimer{background:var(--warning-bg);border-bottom:1px solid var(--warning);
padding:14px 20px}
.disclaimer .in{max-width:var(--maxw);margin:0 auto;display:flex;gap:12px;
align-items:flex-start}
.disclaimer .mk{flex:none;font-size:19px;line-height:1.2;color:var(--warning-text)}
.disclaimer h2{margin:0 0 3px;font-size:13.5px;font-weight:700;color:var(--secondary)}
.disclaimer p{margin:0;font-size:13px;color:var(--body-copy);max-width:78ch}
.foot .disc{border:1px solid var(--warning);background:var(--warning-bg);
border-radius:var(--radius);padding:12px 14px;margin-bottom:12px;color:var(--body-copy)}
.foot{border-top:1px solid var(--border);background:var(--alt-bg);
padding:20px;color:var(--caption);font-size:12.5px}
.foot p{margin:0 0 7px}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
@media (prefers-contrast:more){.tag,.hit,.audit{border-color:var(--secondary)}}
"""


def page(q, body, announce=""):
    st = LEDGER.stats()
    ex = "".join(
        f'<a href="/?q={urllib.parse.quote(e)}">{html.escape(e)}</a>'
        for e in EXAMPLES)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#142D3E">
<meta name="description" content="Search the Michigan Court Rules with full
source provenance for every answer.">
<title>{html.escape(q) + ' — ' if q else ''}Michigan Court Rules search (Beta — not legal advice)</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<style>{CSS}</style></head>
<body>
<a class="skip" href="#results">Skip to results</a>
<header class="bar"><div class="in">
  <img src="/favicon.svg" alt="" width="34" height="34">
  <div><h1>Michigan Court Rules<span class="beta">Beta</span></h1>
  <p>Search with source provenance</p></div>
</div></header>

<aside class="disclaimer" aria-labelledby="disc-h">
  <div class="in">
    <span class="mk" aria-hidden="true">&#9888;</span>
    <div>
      <h2 id="disc-h">Beta software &mdash; not legal advice</h2>
      <p>This is a research prototype and is <strong>not legal advice</strong>.
      It does not create an attorney&ndash;client relationship and must not be
      relied on in any filing or decision. Answers are generated automatically
      and may be incomplete, out of date, or wrong. Always verify against the
      official Michigan Court Rules published by the Michigan Supreme Court at
      courts.michigan.gov before acting.</p>
    </div>
  </div>
</aside>

<div class="sub"><div class="in">
  <form method="get" action="/" role="search">
    <label class="vh" for="q">Search the Michigan Court Rules by question or citation</label>
    <input type="search" id="q" name="q" value="{html.escape(q)}"
      placeholder="Ask a question, or paste a citation such as MCR 2.116(C)(10)"
      autocomplete="off" {'autofocus' if not q else ''}>
    <button type="submit">Search</button>
  </form>
  <p class="meta">{st['rules']} rules · {st['citable_provisions']:,} citable
    provisions · {st['chunks']:,} passages · {html.escape(st['source']['edition'])}<br>
    Retrieval {html.escape(EMBEDDER)} · answers {html.escape(GEN_MODEL)} ·
    source <code>{html.escape(st['source']['file'])}</code>
    SHA-256 <code>{st['source']['sha256'][:16]}…</code></p>
  <nav class="ex" aria-label="Example searches">
    <h2 id="exh">Try an example</h2>{ex}</nav>
</div></div>

<main id="results" tabindex="-1">
  <p class="vh" role="status" aria-live="polite">{html.escape(announce)}</p>
  {body}
</main>

<footer class="foot"><div class="in">
  <p class="disc"><strong>Not legal advice.</strong> This beta tool is provided
  for research and evaluation. It is not a substitute for the official
  Michigan Court Rules or for advice from a licensed attorney, and no
  attorney&ndash;client relationship is created by its use. Verify every
  citation against the official text before relying on it.</p>
  <p>Every answer is drawn only from the passages listed beneath it and cites
  them inline. Where the court rules do not answer a question, the system says
  so rather than inferring an answer.</p>
  <p>Each passage records how it was found, which parsed blocks it came from,
  and the printed page it appears on, so any statement here can be checked
  against the rule itself. Source: Michigan Court Rules, courts.michigan.gov.</p>
</div></footer>
</body></html>"""


def audit_table(v):
    if not v["n"]:
        return ('<div class="audit"><table><caption>Citation audit</caption>'
                '<tbody><tr><td>This answer cites no rule, so there is nothing '
                'to audit.</td></tr></tbody></table></div>')
    rows = "".join(
        "<tr>"
        f"<th scope='row'>{html.escape(r['citation'])}</th>"
        f"<td class='{'ok' if r['exists'] else 'no'}'>"
        f"{'Yes' if r['exists'] else 'NO — not a real provision'}</td>"
        f"<td class='{'ok' if r['was_retrieved'] else 'note'}'>"
        f"{'Yes' if r['was_retrieved'] else 'No — repeated from another passage'}</td>"
        f"<td>{('p.' + str(r['printed_page'][0])) if r.get('printed_page') else '—'}</td>"
        "</tr>" for r in v["citations"])
    verdict = ("Every citation resolves to a real provision of the Michigan "
               "Court Rules." if v["all_exist"] else
               "A citation does not resolve to any provision.")
    return f"""<div class="audit"><table>
  <caption>Citation audit — {html.escape(verdict)}</caption>
  <thead><tr><th scope="col">Citation</th><th scope="col">Exists in corpus</th>
  <th scope="col">Retrieved for this answer</th><th scope="col">Page</th></tr></thead>
  <tbody>{rows}</tbody></table></div>"""


def hit_card(h, n):
    label, kind, sym, why = ROUTE.get(h["how"], (h["how"], "dense", "◆", ""))
    tr = LEDGER.trace(h["citation"]) if h.get("citation") else None
    pages = ", ".join(str(p) for p in (tr or {}).get("printed_pages", [])) or "—"
    blocks = " ".join((tr or {}).get("block_ids", [])[:6])
    because = (f'<span class="tag">condition on {html.escape(h["because_of"])}</span>'
               if h.get("because_of") else "")
    score = ("" if h["how"] == "citation-router"
             else f'<span class="tag">score {h["score"]:.4f}</span>')
    return f"""<article class="hit {kind}" aria-labelledby="h{n}">
  <h3 id="h{n}">{html.escape(h['citation'] or h['rule'])}</h3>
  <p class="tags">
    <span class="tag route {kind}"><span aria-hidden="true">{sym}</span>
      {html.escape(label)}</span>
    <span class="tag">result {n}</span>{score}
    <span class="tag">printed page {pages}</span>{because}</p>
  <p class="why">{html.escape(why)} — {html.escape(h['rule_title'])}</p>
  <div class="body">{html.escape(h['text'][:1800])}</div>
  <p class="lineage">chunk {html.escape(h['chunk_id'])} · blocks
    {html.escape(blocks)} · {h['n_tokens']} tokens</p>
</article>"""


def render(q):
    if not q.strip():
        return ("<p>Ask a question in plain language, or paste a citation. "
                "Citations are answered by exact lookup; everything else by "
                "semantic search across the parsed provisions. Every answer "
                "lists the passages behind it.</p>"), ""
    t0 = time.time()
    r = ENGINE.answer(q, k=6)
    dt = time.time() - t0
    hits, ans = r["hits"], r["answer"]
    v = LEDGER.verify_answer(ans, hits)
    refused = any(s in ans.lower() for s in
                  ("do not answer", "does not answer", "not answered",
                   "no provision", "cannot be answered", "do not state",
                   "do not specify", "passages provided do not"))
    head = ("Not answered by the court rules" if refused else "Answer")
    xr = sum(1 for h in hits if h["how"] == "cross-reference")
    body = "".join([
        f'<section class="answer{" refused" if refused else ""}" '
        f'aria-labelledby="ans"><h2 id="ans">{head}</h2>{html.escape(ans)}</section>',
        audit_table(v),
        f'<h2 class="count">{len(hits)} passages · {dt:.1f} seconds'
        + (f' · {xr} supplied as conditions' if xr else '') + '</h2>',
        *[hit_card(h, n + 1) for n, h in enumerate(hits)]])
    return body, (f"{head}. {len(hits)} supporting passages, "
                  f"{v['n']} citations audited.")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=3600"
                         if ctype != "text/html; charset=utf-8" else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        p = u.path
        if p == "/favicon.svg":
            f = STATIC / "favicon.svg"
            if f.exists():
                return self._send(f.read_bytes(), "image/svg+xml")
        if p.startswith("/static/fonts/") and p.endswith(".woff2"):
            f = STATIC / "fonts" / pathlib.Path(p).name
            if f.exists():
                return self._send(f.read_bytes(), "font/woff2")
        if p not in ("/", "/index.html"):
            return self._send(b"Not found", "text/plain; charset=utf-8", 404)
        q = (urllib.parse.parse_qs(u.query).get("q") or [""])[0]
        body, announce = render(q)
        self._send(page(q, body, announce).encode(),
                   "text/html; charset=utf-8")


def main():
    global ENGINE, LEDGER
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8788)
    args = ap.parse_args()
    print("building index...", flush=True)
    ENGINE = Engine(quiet=True)
    _ = ENGINE.vecs
    ENGINE.model.encode(["warm"], normalize_embeddings=True,
                        prompt_name="query", show_progress_bar=False)
    LEDGER = Ledger()
    print(f"ready · {len(ENGINE.chunks):,} passages · "
          f"http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
