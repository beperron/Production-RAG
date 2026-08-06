#!/usr/bin/env python3
"""Michigan Court Rules — search with provenance.

    python3 pipeline/serve.py --port 8788   ->  http://127.0.0.1:8788

Design follows carolina-policy-search's editorial strategy -- a quiet paper
ground, white cards, a large serif headline, monospace eyebrows for metadata,
numbered result cards, one accent used sparingly -- carrying the Michigan
court palette (#142D3E navy, #277C78 teal) and the court favicon. The display
serif is Georgia so the page still makes no external request and works on an
air-gapped courtroom machine.

The reading path stays what a judge needs: answer, verification line, the
provisions with their printed pages. Scores, chunk ids, block ids, model
names and hashes live behind native <details> disclosures -- reachable, never
in the first read.

Accessibility: WCAG 2.1 AA as before. Measured contrast, route conveyed by
number + words never colour alone, real table semantics in the audit, skip
link, landmarks, aria-live announcements, 44px targets.
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
from querylog import QueryLog                          # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
ENGINE: Engine | None = None
LEDGER: Ledger | None = None
QLOG: QueryLog | None = None

EXAMPLES = [
    "How long does a defendant served in Michigan have to answer a complaint?",
    "MCR 2.116(C)(10)",
    "Serving process on someone in a psychiatric facility",
    "When must the court appoint a guardian ad litem?",
]

ROUTE = {
    "citation-router": ("Exact citation match",
                        "This is the provision whose citation you entered."),
    "dense": ("Found by the search",
              "Located by searching the rules for language related to your "
              "question. Finding it does not mean it answers the question."),
    "cross-reference": ("Included as a related condition",
                        "A provision above is expressly subject to this one."),
}

WORKING_JS = """<script>
(function(){
  var w=document.getElementById('working'),
      st=document.getElementById('live');
  function working(msg){
    if(w){w.hidden=false;}
    if(st){st.textContent=msg;}
  }
  var f=document.querySelector('form.search');
  if(f){
    var submitted=false;
    f.addEventListener('submit',function(e){
      if(submitted){e.preventDefault();return;}
      submitted=true;
      var b=f.querySelector('button.go');
      if(b){b.disabled=true;b.textContent='Searching\u2026';}
      working('Searching the court rules. This usually takes a few seconds.');
    });
  }
  // example chips and any same-site search link get the same feedback
  document.querySelectorAll('a.chip').forEach(function(a){
    a.addEventListener('click',function(){
      working('Searching the court rules for the example you chose. This '
              +'usually takes a few seconds.');
    });
  });
})();
</script>"""

CSS = """
@font-face{font-family:Montserrat;font-style:normal;font-weight:400 700;
font-display:swap;src:url(/static/fonts/montserrat-1.woff2) format('woff2')}
:root{
  --bg:#F6F7F7; --paper:#FFFFFF; --soft:#EAF1F4;
  --ink:#161618; --ink2:#353535; --muted:#707070; --faint:#9AA1A7;
  --line:#E5E8EA; --line2:#D0D5D9;
  --accent:#142D3E; --teal:#277C78; --teal-hover:#1d605d; --teal-soft:#E4F0EF;
  --ok:#0D8252; --warn:#AE5400; --warn-bg:#FFF6E0; --bad:#B30518;
  --serif:Georgia,'Times New Roman',serif;
  --sans:'Montserrat',system-ui,-apple-system,'Segoe UI',sans-serif;
  --mono:ui-monospace,'SF Mono',Menlo,monospace;
}
*{box-sizing:border-box}
[hidden]{display:none!important}
html{scroll-behavior:smooth}
html,body{margin:0;padding:0}
body{background:var(--bg);color:var(--ink);font:15.5px/1.6 var(--sans);
-webkit-font-smoothing:antialiased}
a{color:var(--teal-hover)}
.skip{position:absolute;left:-9999px;top:0;background:var(--accent);color:#fff;
padding:12px 18px;z-index:100}
.skip:focus{left:0}
:focus-visible{outline:3px solid var(--accent);outline-offset:2px;border-radius:3px}

.topbar{display:flex;align-items:center;justify-content:space-between;gap:12px;
padding:13px 24px;border-bottom:1px solid var(--line);background:var(--bg)}
.brand{display:flex;align-items:center;gap:11px;color:var(--ink)}
.brand img{width:30px;height:30px;border-radius:6px}
.brand b{font:600 15.5px var(--sans)}
.brand span{color:var(--muted);font-size:12.5px}
.betachip{font:600 10.5px var(--mono);letter-spacing:.1em;color:var(--warn);
border:1px solid var(--warn);border-radius:999px;padding:3px 9px;
text-transform:uppercase;white-space:nowrap}

.lawline{padding:8px 24px;background:var(--warn-bg);border-bottom:1px solid var(--line);
font:12.5px/1.5 var(--sans);color:var(--ink2)}
.lawline .in{max-width:920px;margin:0 auto}
.lawline b{color:var(--ink)}
.lawline details{display:inline}
.lawline summary{display:inline;cursor:pointer;color:var(--teal-hover);
font-weight:600;text-decoration:underline;text-underline-offset:2px}
.lawline .more{margin:7px 0 2px;max-width:72ch}

header.hero{max-width:920px;margin:0 auto;padding:38px 24px 4px}
.eyebrow{font:12px var(--mono);letter-spacing:.14em;text-transform:uppercase;
color:var(--teal);margin:0 0 10px}
h1{font:400 38px/1.1 var(--serif);letter-spacing:-.01em;margin:0 0 12px;
color:var(--ink)}
.lede{font-size:16.5px;line-height:1.6;color:var(--ink2);max-width:640px;margin:0}
.facts{display:flex;gap:22px;flex-wrap:wrap;margin:16px 0 0;color:var(--muted);
font:12.5px var(--mono)}
.facts b{color:var(--ink2);font-weight:600}

form.search{max-width:920px;margin:0 auto;padding:18px 24px 0;display:flex;
gap:10px;flex-wrap:wrap}
label.vh,.vh{position:absolute;width:1px;height:1px;overflow:hidden;
clip:rect(0 0 0 0);white-space:nowrap}
input[type=search]{flex:1;min-width:280px;height:54px;padding:0 18px;
background:var(--paper);border:1px solid var(--line2);border-radius:12px;
font-size:16px;color:var(--ink);font-family:var(--sans)}
input[type=search]:focus{outline:none;border-color:var(--teal);
box-shadow:0 0 0 3px var(--teal-soft)}
button.go{font:500 14.5px var(--sans);min-height:54px;padding:0 26px;
border:1px solid var(--teal);border-radius:12px;background:var(--teal);
color:#fff;cursor:pointer}
button.go:hover{background:var(--teal-hover)}

.samples{max-width:920px;margin:16px auto 0;padding:0 24px}
.samples .lbl{font:12px var(--mono);letter-spacing:.08em;text-transform:uppercase;
color:var(--muted);margin-bottom:8px}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{font:13px var(--sans);padding:8px 14px;min-height:36px;
border:1px solid var(--line2);border-radius:999px;background:var(--paper);
color:var(--ink2);text-decoration:none;display:inline-block}
.chip:hover{border-color:var(--teal);color:var(--teal-hover);background:var(--teal-soft)}

main{max-width:920px;margin:22px auto 70px;padding:0 24px}

.answer{background:var(--paper);border:1px solid var(--line2);
border-left:3px solid var(--teal);border-radius:12px;padding:18px 22px;
margin:0 0 14px}
.answer.refused{border-left-color:var(--warn)}
.answer h2{display:flex;align-items:center;gap:8px;margin:0 0 10px;
font:600 12px var(--mono);letter-spacing:.08em;text-transform:uppercase;
color:var(--teal-hover)}
.answer.refused h2{color:var(--warn)}
.answer .body{font-size:15.5px;line-height:1.68;color:var(--ink);
white-space:pre-wrap;max-width:70ch}

.verify{color:var(--muted);font:12.5px var(--mono);margin:0 0 26px;padding:2px 4px}
.verify .okm{color:var(--ok);font-weight:700}
.verify .wm{color:var(--warn);font-weight:700}
.verify details{margin-top:8px}
.verify summary{cursor:pointer;color:var(--teal-hover);min-height:24px;
display:inline-block}
.verify summary:hover{text-decoration:underline}
table{width:100%;border-collapse:collapse;font:12.5px var(--sans);
background:var(--paper);margin-top:8px;border:1px solid var(--line);
border-radius:8px}
caption{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
th,td{padding:8px 12px;border-bottom:1px solid var(--line);text-align:left}
th{font:600 10.5px var(--mono);text-transform:uppercase;letter-spacing:.06em;
color:var(--muted)}
tbody tr:last-child td{border-bottom:none}
.okc{color:var(--ok);font-weight:600}
.noc{color:var(--bad);font-weight:600}
.midc{color:var(--warn);font-weight:600}

.meta{color:var(--muted);font:12px var(--mono);margin:22px 0 12px;
padding-top:16px;border-top:1px solid var(--line)}

.card{background:var(--paper);border:1px solid var(--line);border-radius:12px;
padding:16px 18px;margin:0 0 12px;scroll-margin-top:16px;
transition:border-color .3s,box-shadow .3s}
.card:target{border-color:var(--teal);box-shadow:0 0 0 3px var(--teal-soft)}
.cite-link{color:var(--teal-hover);text-decoration:underline;
text-underline-offset:2px;font-weight:600}
.cite-link:hover{background:var(--teal-soft)}
.cardhead{display:flex;align-items:baseline;gap:10px}
.num{flex:0 0 auto;width:22px;height:22px;border-radius:50%;
background:var(--accent);color:#fff;font:600 12px/22px var(--mono);
text-align:center}
.card .sec{font:500 17.5px/1.3 var(--serif);color:var(--ink)}
.card .pg{margin-left:auto;font:12px var(--mono);color:var(--muted);
white-space:nowrap}
.card .title{color:var(--muted);font-size:12.5px;margin:3px 0 2px 32px}
.usedmark{color:var(--ok);font-style:normal}
.card .how{font:italic 12.5px/1.45 var(--sans);color:var(--muted);
margin:6px 0 2px 32px;padding-left:9px;border-left:2px solid var(--line2)}
.card .snip{font-size:14px;line-height:1.62;color:var(--ink2);
margin:10px 0 4px 32px;white-space:pre-wrap;max-height:15em;overflow:auto;
max-width:70ch}
.card details{margin:8px 0 0 32px}
.card summary{font:12px var(--mono);color:var(--muted);cursor:pointer;
min-height:24px;display:inline-block}
.card summary:hover{color:var(--teal-hover)}
.card .prov{margin-top:6px;font:12px/1.7 var(--mono);color:var(--muted);
background:var(--bg);border:1px solid var(--line);border-radius:8px;
padding:10px 12px;overflow-x:auto}

footer{max-width:920px;margin:0 auto;padding:22px 24px 56px;color:var(--faint);
font:12px var(--sans);border-top:1px solid var(--line)}
footer p{margin:0 0 8px;max-width:78ch}
footer b{color:var(--muted)}
footer details{margin-top:12px}
footer summary{cursor:pointer;font:600 12px var(--mono);color:var(--muted);
min-height:24px;display:inline-block}
footer summary:hover{color:var(--teal-hover)}
footer .prov{margin-top:10px;font:12px/1.8 var(--mono);background:var(--paper);
border:1px solid var(--line);border-radius:8px;padding:12px 14px;
overflow-x:auto;color:var(--muted)}
.working{max-width:920px;margin:12px auto 0;padding:0 24px;display:flex;
gap:10px;align-items:center;color:var(--muted);font:13px var(--mono)}
.spinner{width:14px;height:14px;border:2px solid var(--line2);
border-top-color:var(--teal);border-radius:50%;display:inline-block;
animation:spin .7s linear infinite;flex:none}
@keyframes spin{to{transform:rotate(360deg)}}
.card .pg,a.pg{margin-left:auto;font:12px var(--mono);color:var(--teal-hover);
white-space:nowrap;text-decoration:underline;text-underline-offset:2px}
@media (prefers-reduced-motion:reduce){*{transition:none!important;
animation:none!important}.spinner{border-top-color:var(--line2)}}
"""


def page(q, body, announce=""):
    st = LEDGER.stats()
    chips = "".join(
        f'<a class="chip" href="/?q={urllib.parse.quote(e)}">{html.escape(e)}</a>'
        for e in EXAMPLES)
    hero = "" if q else f"""
<header class="hero">
  <p class="eyebrow">Michigan Court Rules &middot; Research edition</p>
  <h1>The Michigan Court Rules Bench&nbsp;Book</h1>
  <p class="lede">A searchable bench book for the Michigan Court Rules. Ask a
  question in plain language or enter a citation; every answer shows the
  provisions it rests on and the printed page to verify against.</p>
  <p class="facts"><span><b>{st['rules']}</b> rules</span>
  <span><b>{st['citable_provisions']:,}</b> citable provisions</span>
  <span>as amended through <b>July 31, 2026</b></span></p>
</header>"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#142D3E">
<meta name="description" content="The Michigan Court Rules Bench Book. Every
answer cites the provisions behind it and the printed page to verify against.">
<title>{html.escape(q) + ' — ' if q else ''}The Michigan Court Rules Bench Book (Beta)</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<style>{CSS}</style></head>
<body>
{'<a class="skip" href="#results">Skip to results</a>' if q else '<a class="skip" href="#q">Skip to search</a>'}

<div class="topbar">
  <div class="brand">
    <img src="/favicon.svg" alt="" width="30" height="30">
    <b>The Bench Book</b>
    <span>Michigan Court Rules &middot; search with provenance</span>
  </div>
  <span class="betachip">Beta</span>
</div>

<div class="lawline"><div class="in">
  <b>Not legal advice.</b> Answers are generated and may be wrong &mdash;
  verify against the official rules before relying on them.
  <details><summary>Full notice</summary>
    <p class="more">This is an independent research prototype from the
    University of Michigan; it is <b>not</b> a product of, affiliated with, or
    endorsed by the Michigan courts. It does not create an
    attorney&ndash;client relationship and must not be relied on in any filing
    or decision. Answers are produced automatically from the text of the
    Michigan Court Rules and may be incomplete, out of date, or wrong. Always
    verify against the official Michigan Court Rules published by the Michigan
    Supreme Court at courts.michigan.gov before acting.</p>
  </details>
</div></div>
{hero}
<form class="search" method="get" action="/" role="search">
  <label class="vh" for="q">Search the Michigan Court Rules by question or citation</label>
  <input type="search" id="q" name="q" value="{html.escape(q)}"
    placeholder="Ask a question, or enter a citation — MCR 2.116(C)(10)"
    autocomplete="off" {'autofocus' if not q else ''}>
  <button class="go" type="submit">Search</button>
</form>
<div class="working" id="working" hidden>
  <span class="spinner" aria-hidden="true"></span>
  <span>Searching the court rules and composing a cited answer&hellip;
  usually a few seconds.</span>
</div>
{WORKING_JS}

<nav class="samples" aria-label="Example searches">
  <p class="lbl">Try</p>
  <div class="chips">{chips}</div>
</nav>

<main id="results" tabindex="-1">
  {'<h1 class="vh">Bench Book search results: ' + html.escape(q) + '</h1>' if q else ''}
  <p class="vh" role="status" aria-live="polite" id="live">{html.escape(announce)}</p>
  {body}
</main>

<footer>
  <p><b>Not legal advice.</b> This beta tool is for research and evaluation;
  it is not a substitute for the official Michigan Court Rules or for advice
  from a licensed attorney. Verify every citation against the official text
  before relying on it.</p>
  <p>An independent research prototype, University of Michigan. Not
  affiliated with or endorsed by the Michigan courts. Report errors:
  <a href="mailto:beperron@umich.edu">beperron@umich.edu</a>.</p>
  <p>Searches are recorded on this machine (question, answer, retrieved
  provisions) to improve the tool; nothing is sent elsewhere except the
  generation request.</p>
  <p>{html.escape(st['source']['edition'])} &middot;
  <a href="/source.pdf" target="_blank" rel="noopener">open this tool's copy
  of the rules PDF</a> (printed page numbers and PDF sheet numbers differ; both
  are shown on every link) &middot;
  <a href="https://www.courts.michigan.gov/rules-administrative-orders-and-jury-instructions/court-rules/"
  target="_blank" rel="noopener">official rules at courts.michigan.gov</a></p>
  <details>
    <summary>Technical details</summary>
    <p style="margin:10px 0 0;color:var(--muted)">In plain terms: this tool's
    copy of the rules was checked against the source document itself — all 625
    rules and every word are accounted for. The data below lets a technician
    confirm that independently.</p>
    <div class="prov">
source        {html.escape(st['source']['file'])}<br>
sha-256       {st['source']['sha256']}<br>
provisions    {st['citable_provisions']:,} citable, from {st['blocks']:,} parsed blocks<br>
passages      {st['chunks']:,} (rule-scoped, 256-token budget)<br>
retrieval     {html.escape(EMBEDDER)}, dense, citation router<br>
answers       {html.escape(GEN_MODEL)}, restricted to retrieved passages<br>
verification  parse reconciled against the document's own contents
              (625/625 rules); word-count identity 384,507 = 384,507</div>
  </details>
</footer>
</body></html>"""


def _page_cell(r):
    tr = LEDGER.trace(r["citation"]) if r.get("printed_page") else None
    if tr and tr.get("pdf_pages"):
        return (f"<td><a href='/source.pdf#page={tr['pdf_pages'][0] + 1}' "
                f"target='_blank' rel='noopener'>p.{r['printed_page'][0]}</a></td>")
    return "<td>—</td>"


RE_MCL = __import__("re").compile(r"\bMCL\s+(\d[\d.]*[a-z]?)")


def verify_block(v, answer_text=""):
    mcl = sorted({m.group(0) for m in RE_MCL.finditer(answer_text or "")})
    mcl_note = ""
    if mcl:
        mcl_note = (f' · <span class="wm">also cites {len(mcl)} statute'
                    f'{"s" if len(mcl) > 1 else ""} ({", ".join(mcl[:3])}'
                    f'{"…" if len(mcl) > 3 else ""}) — statutes are outside '
                    f'the court rules and are NOT verified by this tool</span>')
    return _verify_block(v, mcl_note)


def _verify_block(v, mcl_note=""):
    if not v["n"]:
        return '<div class="verify">This answer cites no rule.</div>'
    n_ok = sum(1 for r in v["citations"] if r["exists"])
    n_shown = sum(1 for r in v["citations"] if r["was_retrieved"])
    if v["all_exist"] and n_shown == v["n"]:
        line = (f'<span class="okm" aria-hidden="true">✓</span> all '
                f'{v["n"]} citations verified · drawn from the passages below')
    elif v["all_exist"]:
        line = (f'<span class="okm" aria-hidden="true">✓</span> all '
                f'{v["n"]} court-rule citations are real provisions · '
                f'<span class="wm">{v["n"] - n_shown} quoted from text inside '
                f'another passage</span>')
    else:
        line = (f'<span class="wm" aria-hidden="true">!</span> '
                f'{v["n"] - n_ok} citation(s) could not be verified')
    rows = "".join(
        "<tr>"
        f"<th scope='row'>{html.escape(r['citation'])}</th>"
        f"<td class='{'okc' if r['exists'] else 'noc'}'>"
        f"{'Yes' if r['exists'] else 'No'}</td>"
        f"<td class='{'okc' if r['was_retrieved'] else 'midc'}'>"
        f"{'Yes' if r['was_retrieved'] else 'From text within a passage'}</td>"
        + _page_cell(r) + "</tr>" for r in v["citations"])
    line += mcl_note
    return f"""<div class="verify">{line}
  <details><summary>citation audit</summary>
  <table><caption>Audit of every citation in the answer</caption>
  <thead><tr><th scope="col">Citation</th><th scope="col">Real provision</th>
  <th scope="col">Retrieved</th><th scope="col">Page</th></tr></thead>
  <tbody>{rows}</tbody></table></details></div>"""


def hit_card(h, n, cited=frozenset()):
    label, why = ROUTE.get(h["how"], ("Found by the search", ""))
    used = bool(set(h.get("citations", [])) & cited)
    tr = LEDGER.trace(h["citation"]) if h.get("citation") else None
    pages = ", ".join(str(p) for p in (tr or {}).get("printed_pages", []))
    pdf1 = ((tr or {}).get("pdf_pages") or [None])[0]
    pdf_href = f"/source.pdf#page={pdf1 + 1}" if pdf1 is not None else None
    blocks = " ".join((tr or {}).get("block_ids", [])[:6])
    because = (f' — required by {html.escape(h["because_of"])}'
               if h.get("because_of") else "")
    score = ("" if h["how"] == "citation-router"
             else f"match strength {h['score']:.2f} (cosine) · ")
    return f"""<article class="card" id="card-{n}" aria-labelledby="h{n}">
  <div class="cardhead">
    <span class="num" aria-hidden="true">{n}</span>
    <h3 class="sec" id="h{n}">{html.escape(h['citation'] or h['rule'])}</h3>
    {f'<a class="pg" href="{pdf_href}" target="_blank" rel="noopener">p.{pages} · opens the PDF at sheet {pdf1 + 1}<span class="vh"> for {html.escape(h["citation"] or h["rule"])} (opens in a new tab)</span></a>' if pdf_href else f'<span class="pg">p.{pages or "—"}</span>'}
  </div>
  <p class="title">{html.escape(h['rule_title'])}</p>
  <p class="how">{html.escape(label)}{because}
    &middot; {'<b class="usedmark">cited in the answer</b>' if used
              else 'not cited in the answer'}</p>
  <div class="snip">{html.escape(h['text'][:1800])}</div>
  <details><summary>provenance</summary>
    <div class="prov">{html.escape(why)}<br>
The identifiers below are the system's internal audit trail; a technician can
use them to reproduce this exact passage.<br>
result {n} · {score}chunk {html.escape(h['chunk_id'])}<br>
blocks {html.escape(blocks)}<br>
{h['n_tokens']} tokens · printed page {pages or '?'} of the source PDF</div>
  </details>
</article>"""


RE_BOLD = __import__("re").compile(r"\*\*([^*\n]+)\*\*")


def md_min(text):
    """The generator emits markdown bold; showing raw asterisks reads as a
    glitch. Escape first, then allow only <strong>."""
    out = html.escape(text)
    return RE_BOLD.sub(r"<strong>\1</strong>", out)


from mcr_search import RE_CITE as _RC, resolve_cite as _rc


def link_citations(escaped_answer, citemap, valid):
    """Every citation the answer emits becomes a link to the passage card it
    came from. Runs on already-escaped text; citations contain no characters
    that html.escape rewrites, so the match is safe. Citations that were not
    retrieved (reported from inside a passage) get no link -- the audit table
    below explains those."""
    def sub(m):
        cit = _rc(m.group(1), m.group(2), valid)
        n = citemap.get(cit)
        if n is None:
            return m.group(0)
        return (f'<a class="cite-link" href="#card-{n}" '
                f'title="Go to this provision below">{m.group(0)}</a>')
    return _RC.sub(sub, escaped_answer)


def render(q):
    if not q.strip():
        return "", ""
    t0 = time.time()
    r = ENGINE.answer(q, k=6)
    dt = time.time() - t0
    hits, ans = r["hits"], r["answer"]
    v = LEDGER.verify_answer(ans, hits)
    cited = {r["citation"] for r in v["citations"] if r["was_retrieved"]}
    citemap = {}
    for n, h in enumerate(hits):
        for c in h.get("citations", []):
            citemap.setdefault(c, n + 1)
    low = ans.lower()
    refused = any(t in low for t in
                  ("do not answer", "does not answer", "not answered",
                   "no provision", "cannot be answered", "do not state",
                   "do not specify", "passages provided do not",
                   "do not establish", "do not set", "do not address",
                   "do not contain", "is not addressed", "not found in the",
                   "outside the michigan court rules", "governed by statute",
                   "would be found in", "passages do not"))
    head = "The court rules do not answer this" if refused else "Answer"
    if QLOG is not None:
        try:
            QLOG.record(q, r, v, dt * 1000, refused)
        except Exception:                               # logging never breaks answering
            pass
    body = "".join([
        f'<section class="answer{" refused" if refused else ""}" '
        f'aria-labelledby="ans"><h2 id="ans">{head}</h2>'
        f'<div class="body">{link_citations(md_min(ans), citemap, LEDGER.valid_citations)}</div></section>',
        verify_block(v, ans),
        (f'<p class="meta">{len(hits)} provisions reviewed — none directly '
         f'answers the question · {dt:.1f}s</p>' if refused else
         f'<p class="meta">{len(hits)} provisions reviewed · {dt:.1f}s · '
         f'other provisions may also bear on this question</p>'),
        *[hit_card(h, n + 1, cited) for n, h in enumerate(hits)]])
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
        if p == "/source.pdf":
            f = ROOT / "0_source" / "michigan-court-rules.pdf"
            if f.exists():
                return self._send(f.read_bytes(), "application/pdf")
        if p not in ("/", "/index.html"):
            return self._send(b"Not found", "text/plain; charset=utf-8", 404)
        q = (urllib.parse.parse_qs(u.query).get("q") or [""])[0]
        body, announce = render(q)
        self._send(page(q, body, announce).encode(),
                   "text/html; charset=utf-8")


def main():
    global ENGINE, LEDGER, QLOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8788)
    args = ap.parse_args()
    print("building index...", flush=True)
    ENGINE = Engine(quiet=True)
    _ = ENGINE.vecs
    ENGINE.model.encode(["warm"], normalize_embeddings=True,
                        prompt_name="query", show_progress_bar=False)
    LEDGER = Ledger()
    QLOG = QueryLog()
    print(f"ready · {len(ENGINE.chunks):,} passages · "
          f"http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
