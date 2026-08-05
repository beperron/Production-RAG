#!/usr/bin/env python3
"""Michigan Court Rules — search with provenance.

    python3 pipeline/serve.py --port 8788   ->  http://127.0.0.1:8788

Server-rendered HTML from the stdlib: no build step, no framework, nothing to
install beyond the engine.

The design goal is that a reader can FALSIFY any answer without trusting the
system's account of itself. Every passage shows the route that surfaced it, the
block ids behind it, and the printed page it sits on, and every citation the
model emits is audited against the corpus before the page renders. A retrieval
that arrived by exact citation lookup is different evidence from one that
arrived by similarity, and the interface says which.
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

ENGINE: Engine | None = None
LEDGER: Ledger | None = None

EXAMPLES = [
    "How long does a defendant served in Michigan have to answer a complaint?",
    "MCR 2.116(C)(10)",
    "Can I serve process on someone confined in a psychiatric facility?",
    "When must a court appoint a guardian ad litem for a minor?",
    "When can a PPO action be dismissed?",
    "What is the statutory cap on noneconomic damages?",
]

ROUTE_LABEL = {
    "citation-router": ("exact citation", "router",
                        "matched the citation you typed, not a similarity score"),
    "dense": ("semantic match", "dense",
              "ranked by meaning against Qwen3-Embedding-4B"),
    "cross-reference": ("pulled in as a condition", "xref",
                        "an earlier passage makes this one a condition on itself"),
}

CSS = """
:root{color-scheme:light dark;--bg:#fff;--fg:#15171c;--mut:#666e7a;--line:#e3e6eb;
--card:#fafbfc;--accent:#1b5fa8;--ok:#0a7d55;--warn:#b26a00;--bad:#c0392b}
@media(prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#e7e9ec;--mut:#98a1ad;
--line:#262b33;--card:#161a20}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15.5px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{border-bottom:1px solid var(--line);padding:16px 20px 12px}
.wrap,main,footer{max-width:940px;margin:0 auto}
h1{font-size:18px;margin:0 0 3px}h1 span{color:var(--mut);font-weight:400}
.sub{color:var(--mut);font-size:12.5px}
form{display:flex;gap:8px;margin:14px 0 8px;flex-wrap:wrap}
input[type=text]{flex:1;min-width:280px;padding:11px 13px;font:inherit;
border:1px solid var(--line);border-radius:9px;background:var(--card);color:var(--fg)}
button{padding:11px 16px;font:inherit;border:1px solid transparent;border-radius:9px;
background:var(--accent);color:#fff;cursor:pointer}
.ex a{color:var(--accent);text-decoration:none;font-size:13px;border:1px solid var(--line);
border-radius:20px;padding:3px 10px;display:inline-block;margin:0 5px 6px 0}
main{padding:18px 20px 60px}
.answer{border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:9px;
padding:14px 16px;background:var(--card);margin-bottom:8px;white-space:pre-wrap}
.answer.refused{border-left-color:var(--warn)}
.answer h3{margin:0 0 8px;font-size:11.5px;color:var(--mut);font-weight:700;
letter-spacing:.06em;text-transform:uppercase}
.audit{border:1px solid var(--line);border-radius:9px;padding:10px 14px;margin-bottom:22px;
font-size:13px;background:var(--card)}
.audit table{width:100%;border-collapse:collapse}
.audit td{padding:3px 8px 3px 0;border-bottom:1px solid var(--line)}
.audit tr:last-child td{border-bottom:none}
.yes{color:var(--ok);font-weight:600}.no{color:var(--bad);font-weight:600}
.hit{border:1px solid var(--line);border-radius:9px;padding:12px 14px;margin-bottom:10px;
background:var(--card)}
.hit.router{border-left:3px solid var(--ok)}
.hit.xref{border-left:3px solid var(--warn)}
.hit .cite{font-weight:660;color:var(--accent);font-size:15.5px}
.hit .why{color:var(--mut);font-size:12px;margin:2px 0 7px}
.hit .body{font-size:14px;white-space:pre-wrap;max-height:250px;overflow:auto;
padding-top:6px;border-top:1px solid var(--line)}
.chip{display:inline-block;font-size:11px;border:1px solid var(--line);border-radius:20px;
padding:1px 8px;color:var(--mut);margin-left:6px}
.chip.ok{border-color:var(--ok);color:var(--ok)}
.chip.warn{border-color:var(--warn);color:var(--warn)}
.lineage{font:11.5px ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--mut);
margin-top:6px}
footer{padding:14px 20px 50px;color:var(--mut);font-size:12px;border-top:1px solid var(--line)}
code{font:12px ui-monospace,Menlo,monospace;color:var(--mut)}
"""


def page(q, body):
    st = LEDGER.stats()
    ex = "".join(f'<a href="/?q={urllib.parse.quote(e)}">{html.escape(e)}</a>'
                 for e in EXAMPLES)
    return f"""<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Michigan Court Rules — search</title><style>{CSS}</style>
<header><div class="wrap">
  <h1>Michigan Court Rules <span>— search with provenance</span></h1>
  <div class="sub">{st['rules']} rules · {st['citable_provisions']:,} citable
    provisions · {st['chunks']:,} passages · {st['source']['edition']}<br>
    retrieval {html.escape(EMBEDDER)} · answers {html.escape(GEN_MODEL)} ·
    source <code>{st['source']['file']}</code> sha256
    <code>{st['source']['sha256'][:16]}</code></div>
  <form method="get" action="/">
    <input type="text" name="q" value="{html.escape(q)}" autofocus
      placeholder="Ask a question, or paste a citation like MCR 2.116(C)(10)">
    <button type="submit">Search</button>
  </form>
  <div class="ex">{ex}</div>
</div></header>
<main>{body}</main>
<footer>Every answer is drawn only from the passages shown below it and cites
them inline; when the rules do not answer a question the system says so rather
than guessing. Each passage records how it was found, which parsed blocks it
came from, and the printed page it sits on, so any statement here can be
checked against the rule itself. Source: courts.michigan.gov.</footer>"""


def audit_table(v):
    if not v["n"]:
        return ('<div class="audit">The answer cites no rule. '
                'Nothing to audit.</div>')
    rows = "".join(
        f"<tr><td><b>{html.escape(r['citation'])}</b></td>"
        f"<td class='{'yes' if r['exists'] else 'no'}'>"
        f"{'exists in the corpus' if r['exists'] else 'NOT IN THE CORPUS'}</td>"
        f"<td class='{'yes' if r['was_retrieved'] else 'no'}'>"
        f"{'was retrieved' if r['was_retrieved'] else 'not retrieved — reported from a passage'}</td>"
        f"<td>{('p.' + str(r['printed_page'][0])) if r.get('printed_page') else ''}</td></tr>"
        for r in v["citations"])
    ok = ("Every citation resolves to a real provision."
          if v["all_exist"] else
          "<span class='no'>A citation does not resolve to any provision.</span>")
    return (f'<div class="audit"><b>Citation audit</b> — {ok}'
            f'<table>{rows}</table></div>')


def hit_card(h, n):
    label, kind, why = ROUTE_LABEL.get(h["how"], (h["how"], "", ""))
    tr = LEDGER.trace(h["citation"]) if h.get("citation") else None
    pages = ", ".join(str(p) for p in (tr or {}).get("printed_pages", []))
    blocks = " ".join((tr or {}).get("block_ids", [])[:6])
    because = (f"<span class='chip warn'>because {html.escape(h['because_of'])} "
               f"conditions on it</span>" if h.get("because_of") else "")
    return f"""<div class="hit {kind}">
  <span class="cite">{html.escape(h['citation'] or h['rule'])}</span>
  <span class="chip {'ok' if kind=='router' else 'warn' if kind=='xref' else ''}">{label}</span>
  <span class="chip">rank {n}</span>
  {"<span class='chip'>score %.4f</span>" % h['score'] if h['how']!='citation-router' else ""}
  <span class="chip">p.{pages or '?'}</span>{because}
  <div class="why">{html.escape(why)} — {html.escape(h['rule_title'])}</div>
  <div class="body">{html.escape(h['text'][:1800])}</div>
  <div class="lineage">chunk {h['chunk_id']} · blocks {html.escape(blocks)} ·
    {h['n_tokens']} tokens</div>
</div>"""


def render(q):
    if not q.strip():
        return ("<p style='color:var(--mut)'>Ask a question above, or try an "
                "example. Citations are answered by exact lookup; everything "
                "else by semantic search over the parsed provisions.</p>")
    t0 = time.time()
    r = ENGINE.answer(q, k=6)
    dt = time.time() - t0
    hits, ans = r["hits"], r["answer"]
    v = LEDGER.verify_answer(ans, hits)

    refused = any(s in ans.lower() for s in
                  ("do not answer", "does not answer", "not answered",
                   "no provision", "cannot be answered", "do not state",
                   "do not specify", "passages provided do not"))
    out = [f"""<div class="answer{' refused' if refused else ''}">
  <h3>{'Not answered by the court rules' if refused else 'Answer'}</h3>
  {html.escape(ans)}</div>""", audit_table(v),
           f"<div class='sub' style='margin:16px 0 10px'>"
           f"{len(hits)} passages · {dt:.1f}s · "
           f"{sum(1 for h in hits if h['how']=='cross-reference')} pulled in as "
           f"conditions</div>"]
    out += [hit_card(h, n + 1) for n, h in enumerate(hits)]
    return "".join(out)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path not in ("/", "/index.html"):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        q = (urllib.parse.parse_qs(u.query).get("q") or [""])[0]
        b = page(q, render(q)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


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
