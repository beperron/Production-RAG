#!/usr/bin/env python3
"""Michigan Court Rules — search with provenance.

    python3 pipeline/serve.py --port 8788   ->  http://127.0.0.1:8788

Design brief: a bench tool, not a dashboard. The reader is a judge or clerk
mid-task. What they need immediately: the answer, the rule it rests on, and
the printed page to check it against. Everything else -- similarity scores,
chunk ids, embedding models, source hashes -- exists for auditability and
lives one click away behind disclosure widgets, never deleted, never in the
first read.

The visual system is the Michigan court dashboard's (palette, Montserrat, 6px
radii), applied sparingly: one accent colour in the reading path, whitespace
doing the separating that boxes and badges did before. A reading measure of
~72ch, because answers are read, not scanned.

Accessibility is WCAG 2.1 AA as before: measured contrast, route never
conveyed by colour alone, real table semantics in the audit, skip link,
landmarks, aria-live announcements, 44px targets, honest <details> disclosure
widgets that keyboards and screen readers get for free.
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
    "How long does a defendant have to answer a complaint?",
    "MCR 2.116(C)(10)",
    "Serving process on someone in a psychiatric facility",
    "When must the court appoint a guardian ad litem?",
]

# Plain-English route descriptions. A judge should understand how a passage
# got here without knowing what an embedding is.
ROUTE = {
    "citation-router": ("Exact citation match", "✦",
                        "This is the provision whose citation you entered."),
    "dense": ("Found by meaning", "◆",
              "Retrieved because its text answers the question."),
    "cross-reference": ("Included as a related condition", "▲",
                        "A provision above is expressly subject to this one."),
}

CSS = """
@font-face{font-family:Montserrat;font-style:normal;font-weight:400 700;
font-display:swap;src:url(/static/fonts/montserrat-1.woff2) format('woff2')}
:root{
--primary:#277C78;--primary-hover:#1d605d;
--secondary:#142D3E;--secondary-2:#1f4257;
--success:#0D8252;--warning:#E5A612;--warning-bg:#FFEFCA;--warning-text:#AE5400;
--error:#B30518;
--ink:#161618;--body-copy:#353535;--caption:#707070;
--border:#E2E5E9;--rule:#D0D0D0;--alt-bg:#F7F7F7;--smoke:#EAF1F4;
--font:"Montserrat",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
--radius:6px;--maxw:820px;
--shadow:0 1px 3px rgba(20,45,62,.10),0 1px 2px rgba(20,45,62,.06)}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:#fff;color:var(--body-copy);font:16px/1.65 var(--font)}
.skip{position:absolute;left:-9999px;top:0;background:var(--secondary);
color:#fff;padding:12px 18px;z-index:100}
.skip:focus{left:0}
:focus-visible{outline:3px solid var(--secondary);outline-offset:2px;border-radius:3px}

header.bar{background:var(--secondary);color:#fff;padding:18px 24px}
.bar .in{max-width:var(--maxw);margin:0 auto;display:flex;gap:14px;align-items:center}
.bar img{width:36px;height:36px;flex:none}
.bar h1{font-size:19px;margin:0;font-weight:700;letter-spacing:.2px}
.bar .beta{margin-left:10px;padding:2px 10px;border-radius:20px;
background:rgba(255,255,255,.14);color:#fff;font-size:11px;font-weight:700;
letter-spacing:.08em;text-transform:uppercase;vertical-align:2px}
.bar p{margin:2px 0 0;font-size:13px;color:#b9c7d1}

.notice{background:var(--warning-bg);border-bottom:1px solid var(--warning);
padding:10px 24px;font-size:13.5px;color:var(--body-copy)}
.notice .in{max-width:var(--maxw);margin:0 auto}
.notice strong{color:var(--ink)}
.notice details{display:inline}
.notice summary{display:inline;cursor:pointer;color:var(--primary-hover);
font-weight:600;text-decoration:underline;text-underline-offset:2px}
.notice .more{margin:8px 0 2px;max-width:72ch}

.search{padding:30px 24px 6px}
.search .in{max-width:var(--maxw);margin:0 auto}
form{display:flex;gap:10px;flex-wrap:wrap}
label.vh,.vh{position:absolute;width:1px;height:1px;overflow:hidden;
clip:rect(0 0 0 0);white-space:nowrap}
input[type=search]{flex:1;min-width:280px;min-height:50px;padding:12px 16px;
font:inherit;font-size:16.5px;border:1.5px solid var(--rule);
border-radius:var(--radius);background:#fff;color:var(--ink)}
input[type=search]:focus{border-color:var(--primary)}
button.go{min-height:50px;padding:12px 26px;font:inherit;font-weight:600;
border:none;border-radius:var(--radius);background:var(--primary);color:#fff;
cursor:pointer}
button.go:hover{background:var(--primary-hover)}
.ex{margin:14px 0 0;font-size:13.5px;color:var(--caption)}
.ex a{color:var(--primary-hover);text-decoration:none;border-bottom:1px solid var(--border);
padding-bottom:1px;margin-right:16px;display:inline-block;margin-top:6px;min-height:24px}
.ex a:hover{border-color:var(--primary-hover)}

main{padding:26px 24px 70px}
main .in{max-width:var(--maxw);margin:0 auto}

.answer{margin:0 0 10px;padding:22px 26px;background:var(--smoke);
border-radius:var(--radius);border-left:4px solid var(--primary)}
.answer.refused{background:var(--warning-bg);border-left-color:var(--warning-text)}
.answer h2{margin:0 0 10px;font-size:12px;font-weight:700;letter-spacing:.09em;
text-transform:uppercase;color:var(--secondary-2)}
.answer .txt{font-size:17px;line-height:1.7;color:var(--ink);white-space:pre-wrap;
max-width:72ch}

.verify{margin:0 0 34px;font-size:13.5px;color:var(--caption);padding:10px 2px}
.verify .okmark{color:var(--success);font-weight:700}
.verify .warnmark{color:var(--warning-text);font-weight:700}
.verify details{margin-top:8px}
.verify summary{cursor:pointer;color:var(--primary-hover);font-weight:600;
min-height:24px;display:inline-block}
.verify summary:hover{text-decoration:underline}
table{width:100%;border-collapse:collapse;font-size:13.5px;background:#fff;
margin-top:10px;border:1px solid var(--border);border-radius:var(--radius)}
caption{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
th,td{padding:9px 12px;border-bottom:1px solid var(--border);text-align:left}
th{font-size:11px;text-transform:uppercase;letter-spacing:.06em;
color:var(--caption);font-weight:700}
tbody tr:last-child td{border-bottom:none}
.ok{color:var(--success);font-weight:600}
.no{color:var(--error);font-weight:600}
.mid{color:var(--warning-text);font-weight:600}

.sources h2{font-size:12px;font-weight:700;letter-spacing:.09em;
text-transform:uppercase;color:var(--caption);margin:0 0 14px}
.hit{padding:18px 0;border-top:1px solid var(--border)}
.hit:last-of-type{border-bottom:1px solid var(--border)}
.hit .top{display:flex;justify-content:space-between;gap:14px;align-items:baseline;
flex-wrap:wrap}
.hit h3{margin:0;font-size:16.5px;color:var(--secondary);font-weight:700}
.hit .pg{font-size:13px;color:var(--caption);white-space:nowrap}
.hit .how{margin:3px 0 10px;font-size:13px;color:var(--caption)}
.hit .how .sym{color:var(--primary)}
.hit .how.xref .sym{color:var(--warning-text)}
.hit .body{font-size:14.5px;line-height:1.65;max-width:72ch;white-space:pre-wrap;
max-height:14.5em;overflow:auto;color:var(--body-copy)}
.hit details.tech{margin-top:10px}
.hit details.tech summary{font-size:12.5px;color:var(--caption);cursor:pointer;
min-height:24px;display:inline-block}
.hit details.tech summary:hover{color:var(--primary-hover)}
.hit .techbody{margin-top:6px;font:12px ui-monospace,SFMono-Regular,Menlo,monospace;
color:var(--caption);background:var(--alt-bg);padding:10px 12px;
border-radius:var(--radius)}

footer{border-top:1px solid var(--border);background:var(--alt-bg);
padding:26px 24px 60px;color:var(--caption);font-size:13px}
footer .in{max-width:var(--maxw);margin:0 auto}
footer p{margin:0 0 8px;max-width:78ch}
footer details{margin-top:14px}
footer summary{cursor:pointer;font-weight:600;color:var(--secondary-2);
min-height:24px;display:inline-block}
footer summary:hover{color:var(--primary-hover)}
footer .techbody{margin-top:10px;font:12px ui-monospace,SFMono-Regular,Menlo,monospace;
background:#fff;border:1px solid var(--border);border-radius:var(--radius);
padding:12px 14px;overflow-x:auto}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
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
<meta name="description" content="Search the Michigan Court Rules. Every
answer cites the provisions behind it and the printed page to verify against.">
<title>{html.escape(q) + ' — ' if q else ''}Michigan Court Rules search (Beta)</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<style>{CSS}</style></head>
<body>
<a class="skip" href="#results">Skip to results</a>

<header class="bar"><div class="in">
  <img src="/favicon.svg" alt="" width="36" height="36">
  <div><h1>Michigan Court Rules<span class="beta">Beta</span></h1>
  <p>Every answer shows the provisions it rests on</p></div>
</div></header>

<div class="notice"><div class="in">
  <strong>Not legal advice.</strong> Answers are generated and may be wrong —
  verify against the official rules before relying on them.
  <details><summary>Full notice</summary>
    <p class="more">This is a research prototype. It does not create an
    attorney&ndash;client relationship and must not be relied on in any filing
    or decision. Answers are produced automatically from the text of the
    Michigan Court Rules and may be incomplete, out of date, or wrong. Always
    verify against the official Michigan Court Rules published by the Michigan
    Supreme Court at courts.michigan.gov before acting.</p>
  </details>
</div></div>

<div class="search"><div class="in">
  <form method="get" action="/" role="search">
    <label class="vh" for="q">Search the Michigan Court Rules by question or citation</label>
    <input type="search" id="q" name="q" value="{html.escape(q)}"
      placeholder="Ask a question, or enter a citation — MCR 2.116(C)(10)"
      autocomplete="off" {'autofocus' if not q else ''}>
    <button class="go" type="submit">Search</button>
  </form>
  <nav class="ex" aria-label="Example searches">Try:&nbsp; {ex}</nav>
</div></div>

<main id="results" tabindex="-1"><div class="in">
  <p class="vh" role="status" aria-live="polite">{html.escape(announce)}</p>
  {body}
</div></main>

<footer><div class="in">
  <p><strong>Not legal advice.</strong> This beta tool is for research and
  evaluation; it is not a substitute for the official Michigan Court Rules or
  for advice from a licensed attorney. Verify every citation against the
  official text before relying on it.</p>
  <p>{html.escape(st['source']['edition'])} &middot; {st['rules']} rules
  &middot; {st['citable_provisions']:,} provisions.</p>
  <details>
    <summary>Technical details</summary>
    <div class="techbody">
source        {html.escape(st['source']['file'])}<br>
sha-256       {st['source']['sha256']}<br>
provisions    {st['citable_provisions']:,} citable, from {st['blocks']:,} parsed blocks<br>
passages      {st['chunks']:,} (rule-scoped, 256-token budget)<br>
retrieval     {html.escape(EMBEDDER)}, dense, citation router<br>
answers       {html.escape(GEN_MODEL)}, restricted to retrieved passages<br>
verification  parse reconciled against the document's own contents
              (625/625 rules); word-count identity 384,507 = 384,507
    </div>
  </details>
</div></footer>
</body></html>"""


def verify_block(v):
    """One quiet line a judge can trust, expandable to the full audit."""
    if not v["n"]:
        return ('<div class="verify">This answer cites no rule.</div>')
    n_ok = sum(1 for r in v["citations"] if r["exists"])
    n_shown = sum(1 for r in v["citations"] if r["was_retrieved"])
    if v["all_exist"] and n_shown == v["n"]:
        line = (f'<span class="okmark" aria-hidden="true">✓</span> '
                f'All {v["n"]} citations verified against the rules and drawn '
                f'from the passages below.')
    elif v["all_exist"]:
        line = (f'<span class="okmark" aria-hidden="true">✓</span> '
                f'All {v["n"]} citations are real provisions; '
                f'<span class="mid">{v["n"] - n_shown} cited from a '
                f'cross-reference</span> rather than a retrieved passage.')
    else:
        line = (f'<span class="warnmark" aria-hidden="true">!</span> '
                f'{v["n"] - n_ok} citation(s) could not be verified against '
                f'the rules.')
    rows = "".join(
        "<tr>"
        f"<th scope='row'>{html.escape(r['citation'])}</th>"
        f"<td class='{'ok' if r['exists'] else 'no'}'>"
        f"{'Yes' if r['exists'] else 'No'}</td>"
        f"<td class='{'ok' if r['was_retrieved'] else 'mid'}'>"
        f"{'Yes' if r['was_retrieved'] else 'Via cross-reference'}</td>"
        f"<td>{('p.' + str(r['printed_page'][0])) if r.get('printed_page') else '—'}</td>"
        "</tr>" for r in v["citations"])
    return f"""<div class="verify">{line}
  <details><summary>Citation audit</summary>
  <table><caption>Audit of every citation in the answer</caption>
  <thead><tr><th scope="col">Citation</th><th scope="col">Real provision</th>
  <th scope="col">Retrieved</th><th scope="col">Page</th></tr></thead>
  <tbody>{rows}</tbody></table></details></div>"""


def hit_card(h, n):
    label, sym, why = ROUTE.get(h["how"], ("Found by meaning", "◆", ""))
    kind = "xref" if h["how"] == "cross-reference" else ""
    tr = LEDGER.trace(h["citation"]) if h.get("citation") else None
    pages = ", ".join(str(p) for p in (tr or {}).get("printed_pages", []))
    blocks = " ".join((tr or {}).get("block_ids", [])[:6])
    because = (f' — required by {html.escape(h["because_of"])}'
               if h.get("because_of") else "")
    score = ("" if h["how"] == "citation-router"
             else f"similarity {h['score']:.4f} · ")
    return f"""<article class="hit" aria-labelledby="h{n}">
  <div class="top">
    <h3 id="h{n}">{html.escape(h['citation'] or h['rule'])}</h3>
    <span class="pg">page {pages or '—'}</span>
  </div>
  <p class="how {kind}"><span class="sym" aria-hidden="true">{sym}</span>
    {html.escape(label)}{because} · {html.escape(h['rule_title'])}</p>
  <div class="body">{html.escape(h['text'][:1800])}</div>
  <details class="tech"><summary>Provenance</summary>
    <div class="techbody">{html.escape(why)}<br>
rank {n} · {score}chunk {html.escape(h['chunk_id'])}<br>
blocks {html.escape(blocks)}<br>
{h['n_tokens']} tokens · printed page {pages or '?'} of the source PDF</div>
  </details>
</article>"""


def render(q):
    if not q.strip():
        return ("<p style='max-width:64ch'>Ask a question in plain language, "
                "or enter a citation. Answers come only from the text of the "
                "Michigan Court Rules, and every answer lists the provisions "
                "it rests on.</p>"), ""
    t0 = time.time()
    r = ENGINE.answer(q, k=6)
    dt = time.time() - t0
    hits, ans = r["hits"], r["answer"]
    v = LEDGER.verify_answer(ans, hits)
    refused = any(s in ans.lower() for s in
                  ("do not answer", "does not answer", "not answered",
                   "no provision", "cannot be answered", "do not state",
                   "do not specify", "passages provided do not"))
    head = "The court rules do not answer this" if refused else "Answer"
    body = "".join([
        f'<section class="answer{" refused" if refused else ""}" '
        f'aria-labelledby="ans"><h2 id="ans">{head}</h2>'
        f'<div class="txt">{html.escape(ans)}</div></section>',
        verify_block(v),
        '<section class="sources" aria-labelledby="src">'
        f'<h2 id="src">Provisions behind this answer</h2>',
        *[hit_card(h, n + 1) for n, h in enumerate(hits)],
        '</section>'])
    return body, (f"{head}. {len(hits)} supporting provisions, "
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
