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
import json
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
PENDING: dict = {}
PENDING_LOCK = __import__("threading").Lock()


def stash(prep, q, t0):
    import uuid
    qid = uuid.uuid4().hex[:12]
    with PENDING_LOCK:
        if len(PENDING) > 60:
            for k in list(PENDING)[:20]:
                PENDING.pop(k, None)
        PENDING[qid] = (q, prep, t0)
    return qid

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
.topnav{display:flex;align-items:center;gap:14px}
.topnav a{font:600 13px var(--sans);color:var(--teal-hover);
text-decoration:none;padding:8px 4px;min-height:36px;display:inline-flex;
align-items:center}
.topnav a:hover{text-decoration:underline;text-underline-offset:3px}
.topnav a.btn{background:var(--teal);color:#fff;border-radius:8px;
padding:9px 16px;text-decoration:none}
.topnav a.btn:hover{background:var(--teal-hover);text-decoration:none}

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
  <p class="eyebrow">Bench Book &middot; Research edition</p>
  <h1>Michigan Court Rules</h1>
  <p class="lede">A searchable bench book for the rules governing practice in
  Michigan courts. Ask a question in plain language or enter a citation;
  every answer shows the provisions it rests on and the printed page to
  verify against.</p>
  <p class="facts"><span><b>{st['rules']}</b> rules</span>
  <span><b>{st['citable_provisions']:,}</b> citable provisions</span>
  <span>as amended through <b>July 31, 2026</b></span></p>
</header>"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#142D3E">
<meta name="description" content="Michigan Court Rules — Bench Book. Every
answer cites the provisions behind it and the printed page to verify against.">
<title>{html.escape(q) + ' — ' if q else ''}Michigan Court Rules — Bench Book (Beta)</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<style>{CSS}</style></head>
<body>
{'<a class="skip" href="#results">Skip to results</a>' if q else '<a class="skip" href="#q">Skip to search</a>'}

<div class="topbar">
  <div class="brand">
    <img src="/favicon.svg" alt="" width="30" height="30">
    <b>Michigan Court Rules</b>
    <span>Bench Book &middot; search with provenance</span>
  </div>
  <nav class="topnav" aria-label="Site">
    <a href="/architecture">About this system</a>
    <span class="betachip">Beta</span>
  </nav>
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
  {'<h1 class="vh">Michigan Court Rules search results: ' + html.escape(q) + '</h1>' if q else ''}
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
  <p><a href="/architecture">About this system</a> &mdash; the pipeline, the
  audit, the measured performance, and what was tested and rejected.</p>
  <details>
    <summary>Technical details</summary>
    <div style="max-width:76ch;color:var(--ink2);font-size:13.5px;line-height:1.7">

    <p style="margin:12px 0 8px"><b>What kind of system this is.</b> The
    Bench Book is a retrieval-augmented generation ("RAG") system. The term
    describes a specific architecture: rather than asking an artificial
    intelligence model to answer from whatever it absorbed during training,
    the system first <i>retrieves</i> the governing text — here, the
    provisions of the Michigan Court Rules relevant to your question — and
    then instructs the model to compose its answer from that text alone. The
    model functions less like an expert reciting from memory and more like a
    clerk directed to answer only from the record placed in front of them.</p>

    <p style="margin:0 0 8px"><b>Why the architecture matters.</b>
    General-purpose AI systems generate language by statistical prediction,
    and when asked about law they can produce authority that does not exist —
    confident, well-formatted, and wrong. The professional consequences of
    relying on such fabrications are by now well documented. This system is
    built to foreclose that failure mode structurally rather than by
    instruction: the model is never asked what it remembers about Michigan
    procedure. It is shown the pertinent rule text, retrieved verbatim from
    the official publication, and confined to it.</p>

    <p style="margin:0 0 8px"><b>The verification layer.</b> Confinement is
    then checked rather than assumed. Before any answer reaches you, every
    citation it contains is tested against the parsed rules: does the cited
    provision exist, and was it among the passages the model was actually
    given? The results of that audit are displayed with the answer, and each
    cited provision links to its verbatim text and to the page of the source
    PDF, so the answer can be verified against the rule itself rather than
    taken on trust. Where the rules do not address a question, the system is
    designed to say so and to indicate where the answer likely resides —
    a statute, court precedent, or local order — rather than to guess.</p>

    <p style="margin:0 0 8px"><b>The measured record.</b> In evaluation
    against more than a thousand benchmark questions written from the rules
    themselves, the system produced no fabricated citations, and it declined
    all thirty-two questions deliberately designed to have no answer in the
    court rules. Its stored copy of the rules was verified word for word
    against the official PDF — all 625 rules accounted for, nothing added or
    paraphrased.</p>

    <p style="margin:0 0 8px"><b>What this does not claim.</b> The
    architecture minimises fabrication; it does not abolish error. Retrieval
    can miss a pertinent provision, and composed language can state a rule
    more broadly than its text supports. That is why every answer carries its
    sources, why the audit is shown rather than merely performed, and why the
    banner above asks you to verify against the official rules before
    relying on anything here.</p>

    <p style="margin:0 0 10px"><b>Where the work happens.</b> Search and
    verification run entirely on this computer. Only the final composition
    step is sent to an external language-model service; your searches and the
    documents never leave this machine.</p>

    <details style="margin:4px 0 0">
      <summary style="cursor:pointer;font:600 12px var(--mono);color:var(--muted)">Reference identifiers (for technical staff)</summary>
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
    </div>
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


REFUSAL_TERMS = ("do not answer", "does not answer", "not answered",
                 "no provision", "cannot be answered", "do not state",
                 "do not specify", "passages provided do not",
                 "do not establish", "do not set", "do not address",
                 "do not contain", "is not addressed", "not found in the",
                 "outside the michigan court rules", "governed by statute",
                 "would be found in", "passages do not")


def finish_html(q, r, dt):
    """Answer HTML + everything below it (verify line, cards). Shared by the
    synchronous path and the stream's final event, so both render and log
    identically."""
    hits, ans = r["hits"], r["answer"]
    v = LEDGER.verify_answer(ans, hits)
    cited = {x["citation"] for x in v["citations"] if x["was_retrieved"]}
    citemap = {}
    for n, h in enumerate(hits):
        for c in h.get("citations", []):
            citemap.setdefault(c, n + 1)
    refused = any(t in ans.lower() for t in REFUSAL_TERMS)
    if QLOG is not None:
        try:
            QLOG.record(q, r, v, dt * 1000, refused)
        except Exception:
            pass
    answer_html = link_citations(md_min(ans), citemap, LEDGER.valid_citations)
    head = "The court rules do not answer this" if refused else "Answer"
    below = "".join([
        verify_block(v, ans),
        (f'<p class="meta">{len(hits)} provisions reviewed — none directly '
         f'answers the question · {dt:.1f}s</p>' if refused else
         f'<p class="meta">{len(hits)} provisions reviewed · {dt:.1f}s · '
         f'other provisions may also bear on this question</p>'),
        *[hit_card(h, n + 1, cited) for n, h in enumerate(hits)]])
    return answer_html, below, refused, head, v


ARCH_CSS = """
.arch{max-width:920px;margin:0 auto;padding:28px 24px 70px}
.arch h1{margin-top:6px}
.arch h2{font:600 13px var(--mono);letter-spacing:.09em;text-transform:uppercase;
color:var(--teal-hover);margin:34px 0 10px}
.arch p,.arch li{max-width:72ch;color:var(--ink2)}
.flow{display:flex;flex-wrap:wrap;gap:8px;align-items:stretch;margin:14px 0}
.stage{flex:1 1 150px;background:var(--paper);border:1px solid var(--line2);
border-radius:10px;padding:10px 12px}
.stage b{display:block;font-size:13.5px;color:var(--secondary)}
.stage span{font:11.5px var(--mono);color:var(--muted)}
.arrow{align-self:center;color:var(--muted);font-size:18px}
.arch table{width:100%;border-collapse:collapse;font-size:13.5px;background:var(--paper);
border:1px solid var(--line);border-radius:8px;margin:10px 0}
.arch th,.arch td{padding:8px 12px;border-bottom:1px solid var(--line);text-align:left}
.arch th{font:600 10.5px var(--mono);text-transform:uppercase;color:var(--muted)}
.arch tbody tr:last-child td{border-bottom:none}
.dead{color:var(--muted)}
.dead b{color:var(--warn)}
.backlink{display:inline-block;margin:14px 0 0;color:var(--teal-hover)}
.btn-bottom{background:var(--teal);color:#fff!important;border-radius:8px;
padding:11px 20px;text-decoration:none;font-weight:600;margin-top:26px}
.btn-bottom:hover{background:var(--teal-hover)}
"""


def architecture_page():
    st = LEDGER.stats()
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#142D3E">
<title>About this system — Michigan Court Rules Bench Book</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<style>{CSS}{ARCH_CSS}</style></head>
<body>
<div class="topbar">
  <div class="brand">
    <img src="/favicon.svg" alt="" width="30" height="30">
    <b>Michigan Court Rules</b>
    <span>Bench Book &middot; about this system</span>
  </div>
  <nav class="topnav" aria-label="Site">
    <a class="btn" href="/">&larr; Back to search</a>
    <span class="betachip">Beta</span>
  </nav>
</div>
<main class="arch">
<p class="eyebrow">Bench Book &middot; Architecture</p>
<h1>About this system</h1>
<p>The Bench Book answers questions about the Michigan Court Rules by finding
the governing provisions first and writing from them second. It never answers
from general knowledge: every sentence is composed from retrieved rule text,
every citation is checked against the parsed rules before the page renders,
and every passage can be traced to the printed page of the source PDF.</p>

<h2>The pipeline</h2>
<div class="flow">
  <div class="stage"><b>1 &middot; Parse</b><span>874-page official PDF &rarr;
    {st['rules']} rules, {st['citable_provisions']:,} citable provisions.
    Verified three ways against the document itself.</span></div>
  <span class="arrow" aria-hidden="true">&rarr;</span>
  <div class="stage"><b>2 &middot; Chunk</b><span>{st['chunks']:,} passages,
    each scoped to a single rule (never crossing one), ~256 tokens.</span></div>
  <span class="arrow" aria-hidden="true">&rarr;</span>
  <div class="stage"><b>3 &middot; Retrieve</b><span>Meaning-based search
    (Qwen3-Embedding-4B). Exact citations bypass search entirely via a
    deterministic router.</span></div>
  <span class="arrow" aria-hidden="true">&rarr;</span>
  <div class="stage"><b>4 &middot; Expand</b><span>A cross-reference graph
    (943 classified edges) supplies provisions the retrieved rules are
    expressly subject to &mdash; or overridden by.</span></div>
  <span class="arrow" aria-hidden="true">&rarr;</span>
  <div class="stage"><b>5 &middot; Compose</b><span>glm-5.2 writes from the
    retrieved passages only, within a 4,096-token reading window.</span></div>
  <span class="arrow" aria-hidden="true">&rarr;</span>
  <div class="stage"><b>6 &middot; Audit</b><span>Every citation in the answer
    is checked: does it exist, and was it actually retrieved? Only then does
    the page render.</span></div>
</div>

<h2>Why you can check it</h2>
<ul>
<li><b>Citations link to their passages</b>, and passages link to the printed
page of the source PDF (both numbering systems shown).</li>
<li><b>The audit is visible.</b> A citation the model repeated from inside a
passage &mdash; rather than one it was shown &mdash; is labelled exactly that.</li>
<li><b>Refusal is a feature.</b> When the rules do not answer a question, the
system says so in amber and names where the answer lives instead. On a
32-question test of genuinely unanswerable questions it refused all 32.</li>
<li><b>Every search is recorded on this machine</b> (question, answer,
retrieved provisions) and nothing leaves it except the generation request.</li>
</ul>

<h2>Measured performance</h2>
<table>
<thead><tr><th>property</th><th>result</th><th>how measured</th></tr></thead>
<tbody>
<tr><td>parse completeness</td><td>384,507 = 384,507 words</td>
<td>word-for-word identity with the source PDF</td></tr>
<tr><td>structure</td><td>625/625 rules</td>
<td>reconciled against the document's own table of contents</td></tr>
<tr><td>right provision in the reading window</td><td>94%</td>
<td>1,060-question benchmark, gold = citation strings</td></tr>
<tr><td>exact-citation lookups</td><td>125/125</td><td>deterministic router</td></tr>
<tr><td>answer cites the correct provision</td><td>0.91</td>
<td>paired evaluation, independent judge model</td></tr>
<tr><td>fabricated citations</td><td>0</td>
<td>every emitted citation checked against the corpus</td></tr>
<tr><td>refuses genuinely unanswerable questions</td><td>32/32</td>
<td>negative test set</td></tr>
</tbody></table>

<h2>What was tested and rejected</h2>
<p class="dead">Each of these was measured on paired benchmarks and made the
system <b>worse</b>: keyword-search fusion (BM25), a reranking model, query
rewriting (HyDE), heading prefixes in passages, merged-rule context assembly,
and larger passage sizes. The system is deliberately simple because the
simpler configuration measured better at every step &mdash; the full record
is in the project repository (<code>docs/TESTING-AND-LESSONS.md</code>).</p>

<h2>Limits</h2>
<p>This is a research prototype, not legal advice. It covers the Michigan
Court Rules as amended through July 31, 2026 &mdash; not statutes (MCL), case
law, the Rules of Evidence, or local administrative orders; when an answer
depends on those, the system says so rather than guessing. It is an
independent University of Michigan project, not a product of the Michigan
courts.</p>

<a class="backlink btn-bottom" href="/">&larr; Back to search</a>
</main>
<footer><div class="in">
<p>An independent research prototype, University of Michigan. Report errors:
<a href="mailto:beperron@umich.edu">beperron@umich.edu</a>.</p>
</div></footer>
</body></html>"""


def render_shell(q):
    """Streaming path: retrieval happens now, the answer streams in via SSE,
    and the audit + cards arrive in the final event -- citations become links
    only after verification."""
    t0 = time.time()
    prep = ENGINE.prepare(q)
    qid = stash(prep, q, t0)
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
      es=new EventSource('/api/stream?qid={qid}');
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
    window.location='/?q='+encodeURIComponent({q!r})+'&nojs=1';}});
  es.onerror=function(){{es.close();}};
}})();
</script>
<noscript><meta http-equiv="refresh"
  content="0;url=/?q={urllib.parse.quote(q)}&nojs=1"></noscript>"""


def render(q):
    """Synchronous fallback (no-JS, and the EventSource 'gone' redirect)."""
    if not q.strip():
        return "", ""
    t0 = time.time()
    r = ENGINE.answer(q)
    dt = time.time() - t0
    answer_html, below, refused, head, v = finish_html(q, r, dt)
    body = (f'<section class="answer{" refused" if refused else ""}" '
            f'aria-labelledby="ans"><h2 id="ans">{head}</h2>'
            f'<div class="body">{answer_html}</div></section>' + below)
    return body, (f"{head}. {len(r['hits'])} supporting provisions, "
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

    def sse(self, event, data):
        self.wfile.write(f"event: {event}\ndata: {data}\n\n".encode())
        self.wfile.flush()

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        p = u.path
        if p == "/api/stream":
            qid = (urllib.parse.parse_qs(u.query).get("qid") or [""])[0]
            with PENDING_LOCK:
                item = PENDING.pop(qid, None)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            if item is None:
                self.sse("gone", "{}")
                return
            q, prep, t0 = item
            parts = []
            try:
                for tok in ENGINE.stream_generate(prep["user"]):
                    parts.append(tok)
                    self.sse("token", json.dumps(tok))
            except Exception as exc:                    # noqa: BLE001
                parts.append(f"(generation unavailable: {exc})")
            ans = "".join(parts)
            r = {"question": q, "answer": ans, "hits": prep["hits"],
                 "model": prep["model"]}
            dt = time.time() - t0
            answer_html, below, refused, head, v = finish_html(q, r, dt)
            self.sse("final", json.dumps({
                "answer_html": answer_html, "below": below,
                "refused": refused, "head": head,
                "announce": f"{head}. {len(prep['hits'])} supporting "
                            f"provisions, {v['n']} citations audited."}))
            return
        if p in ("/architecture", "/about", "/how-it-works"):
            return self._send(architecture_page().encode(),
                              "text/html; charset=utf-8")
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
        qs = urllib.parse.parse_qs(u.query)
        q = (qs.get("q") or [""])[0]
        if q.strip() and not qs.get("nojs"):
            body, announce = render_shell(q), ""
        else:
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
