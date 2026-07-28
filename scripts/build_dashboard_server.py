#!/usr/bin/env python3
"""Build-time pedantic dashboard — live view of a running scripts/build_kb.py.

Reads the JSONL event log a build writes (see
src/parsevault/pipeline/build_events.py) and serves a browser panel that
polls and re-renders it. Two independent processes, decoupled by the file on
disk: the dashboard can point at a build running on another machine, and can
reopen a finished build's log later for review.

Stdlib-only (http.server + ThreadingHTTPServer), matching
scripts/law_search_server.py's structure. No auth — localhost-only, same
posture as that server.

Run:  python scripts/build_dashboard_server.py --events knowledge-base/nc-child-welfare/build_events.jsonl [--port 8901]
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

_EVENTS_PATH: Path | None = None

# Palette lifted from scripts/law_search_server.py's "paper / seal-red" theme
# for visual consistency across the two tools.
_CSS = """
:root{--bg:#FBF9F4;--bg2:#F4F1E8;--card:#FFFFFF;--soft:#F7F4EC;--ink:#191A1C;
--muted:#6E6A60;--line:#E6E2D6;--line-soft:#EDE9DD;--seal:#C5302C;--seal-deep:#9C2420;
--ok:#3F7A57;--warn:#B8860B;--link:#2A5560;--display:'Besley',Georgia,serif;--mono:'JetBrains Mono',ui-monospace,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 'IBM Plex Sans',system-ui,sans-serif}
a{color:var(--link)}
.banner{background:#FCEEEE;border-bottom:1px solid var(--seal);color:var(--seal-deep);
  font:12px/1.4 var(--mono);padding:6px 20px;text-align:center;letter-spacing:.02em}
header{padding:22px 20px 14px;max-width:1040px;margin:0 auto}
h1{font:600 24px/1.2 var(--display);margin:0 0 10px}
.progress-wrap{margin:10px 0}
.progress-bar{height:10px;border-radius:6px;background:var(--soft);border:1px solid var(--line);overflow:hidden}
.progress-fill{height:100%;background:var(--seal);transition:width .3s ease}
.progress-label{font:12px var(--mono);color:var(--muted);margin-top:4px}
.stat-row{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 14px;min-width:140px}
.stat-label{font:11px var(--mono);color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.stat-val{font:600 16px var(--display);margin-top:2px}
.degrade-badge{background:var(--soft);color:var(--ok);font:600 15px var(--mono);border:1px solid var(--line)}
.degrade-badge.hot{background:#FCEEEE;color:var(--seal-deep);border-color:var(--seal)}
main{max-width:1040px;margin:14px auto 60px;padding:0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:0 0 10px;padding:0}
.card summary{cursor:pointer;list-style:none;padding:12px 16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.card summary::-webkit-details-marker{display:none}
.card .idx{font:12px var(--mono);color:var(--muted);min-width:44px}
.card .name{font:600 14px var(--display);flex:1;min-width:180px}
.card .status{font:11px var(--mono);border-radius:999px;padding:2px 9px;border:1px solid var(--line)}
.status.extracting{background:#FBF6E7;border-color:#C9BE9E;color:#7A5300}
.status.chunking{background:#EDF0F2;border-color:#DCE3E6;color:#44505A}
.status.indexing{background:#EDF0F2;border-color:#DCE3E6;color:#44505A}
.status.done{background:#F0F7F2;border-color:#CFE3D6;color:var(--ok)}
.status.error{background:#FCEEEE;border-color:var(--seal);color:var(--seal-deep)}
.card .body{padding:0 16px 14px;border-top:1px solid var(--line-soft)}
.sect{margin-top:12px}
.sect h4{font:600 12px var(--mono);text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:0 0 6px}
.sect .explain{font:italic 12.5px/1.5 var(--display);color:var(--muted);margin:0 0 8px;max-width:640px}
.pagestrip{display:flex;gap:3px;flex-wrap:wrap}
.pg{width:22px;height:22px;border-radius:4px;font:10px var(--mono);display:flex;align-items:center;
  justify-content:center;color:#fff;cursor:default}
.pg.native{background:#3F7A57}
.pg.tesseract{background:#B8860B}
.pg.vlm{background:#2A5560}
.pg.llamaparse{background:#7A4FA3}
.pg.text{background:#5B6B73}
.pg.flagged{outline:2px solid var(--seal-deep);outline-offset:1px}
table.chunktbl{width:100%;border-collapse:collapse;font:12.5px var(--mono)}
table.chunktbl th{text-align:left;color:var(--muted);font-weight:600;padding:4px 8px;border-bottom:1px solid var(--line)}
table.chunktbl td{padding:4px 8px;border-bottom:1px solid var(--line-soft);vertical-align:top}
table.chunktbl td.preview{font:12.5px/1.4 var(--display);color:var(--ink);white-space:normal;min-width:260px}
.pagequotes{display:flex;flex-direction:column;gap:6px;margin-top:8px}
.pagequote{background:var(--soft);border:1px solid var(--line-soft);border-left:3px solid var(--line);border-radius:0 6px 6px 0;
  padding:6px 10px;font:italic 13px/1.5 var(--display);color:var(--ink)}
.pagequote .pg-label{display:block;font:600 10.5px var(--mono);font-style:normal;color:var(--muted);text-transform:uppercase;letter-spacing:.03em;margin-bottom:2px}
.dense-ok{color:var(--ok);font:600 13px var(--mono)}
.dense-warn{color:var(--seal-deep);font:600 13px var(--mono);background:#FCEEEE;border-radius:6px;padding:2px 8px;display:inline-block}
.err{color:var(--seal-deep);font:13px var(--mono);white-space:pre-wrap}
.empty{color:var(--muted);text-align:center;padding:40px 0;font-style:italic}
footer{max-width:1040px;margin:0 auto;padding:20px;color:var(--muted);font:11px var(--mono);border-top:1px solid var(--line-soft)}
footer.complete{color:var(--ok)}
"""

_PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Build Dashboard</title><style>__CSS__</style></head><body>
<div class=banner>BUILD DASHBOARD &middot; reads __EVENTS_PATH__ &middot; localhost, read-only</div>
<header>
  <h1>Build Dashboard</h1>
  <div class=progress-wrap>
    <div class=progress-bar><div class=progress-fill id=progress-fill style="width:0%"></div></div>
    <div class=progress-label id=progress-label>waiting for events&hellip;</div>
  </div>
  <div class=stat-row>
    <div class=stat><div class=stat-label>Elapsed</div><div class=stat-val id=stat-elapsed>&ndash;</div></div>
    <div class=stat><div class=stat-label>Current doc</div><div class=stat-val id=stat-current>&ndash;</div></div>
    <div class="stat degrade-badge" id=stat-degrade>0 degradations</div>
  </div>
</header>
<main id=timeline><div class=empty>No events yet.</div></main>
<footer id=footer></footer>
<script>
const ROUTE_LABEL = {native:'N', tesseract:'T', vlm:'V', llamaparse:'L', text:'P'};
const ROUTE_NAME = {native:'native text layer', tesseract:'Tesseract OCR', vlm:'vision-language model', llamaparse:'LlamaParse (cloud)', text:'plain text'};

function esc(s){
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function buildCards(events){
  const cards = [];
  const byPath = new Map();
  const byDocId = new Map();
  let buildStart = null, buildDone = null, lastDocStart = null;
  let degradations = 0;

  for (const e of events){
    switch (e.stage){
      case 'build_start':
        buildStart = e;
        break;
      case 'doc_start': {
        lastDocStart = e;
        const card = {path: e.path, index: e.index, total: e.total, status: 'extracting'};
        cards.push(card);
        byPath.set(e.path, card);
        break;
      }
      case 'extract_done': {
        const card = byPath.get(e.path) || {path: e.path, status: 'extracting'};
        card.doc_id = e.doc_id;
        card.extractor = e.extractor;
        card.page_count = e.page_count;
        card.pages = e.pages || [];
        card.degradations = e.degradations || [];
        card.status = 'chunking';
        degradations += card.degradations.length;
        byDocId.set(e.doc_id, card);
        break;
      }
      case 'chunk_done': {
        const card = byDocId.get(e.doc_id);
        if (!card) break;
        card.chunk_count = e.chunk_count;
        card.chunks = e.chunks || [];
        card.status = 'indexing';
        break;
      }
      case 'index_done': {
        const card = byDocId.get(e.doc_id);
        if (!card) break;
        card.dense_embedded = e.dense_embedded;
        card.dense_missing = e.dense_missing;
        card.status = 'done';
        if (e.dense_missing > 0) degradations += 1;
        break;
      }
      case 'doc_error': {
        const card = byPath.get(e.path) || {path: e.path};
        if (!byPath.has(e.path)){ cards.push(card); byPath.set(e.path, card); }
        card.status = 'error';
        card.error = e.error;
        degradations += 1;
        break;
      }
      case 'build_done':
        buildDone = e;
        break;
      default:
        break;
    }
  }
  return {cards, buildStart, buildDone, lastDocStart, degradations};
}

function pageStrip(pages){
  return pages.map(p => {
    const route = p.route || '?';
    const flagged = p.flagged ? ' flagged' : '';
    const routeName = ROUTE_NAME[route] || route;
    const title = p.flagged
      ? esc((p.flag_reasons || []).join('; '))
      : `page ${p.page_number} — read via ${routeName} (${p.chars} chars)`;
    return `<div class="pg ${esc(route)}${flagged}" title="${title}">${ROUTE_LABEL[route] || '?'}</div>`;
  }).join('');
}

// A couple of actual page snippets, quoted verbatim, so the extraction step
// reads as "here are the words we read" rather than just badges and counts.
function pageQuotes(pages){
  const shown = pages.filter(p => p.preview).slice(0, 3);
  if (!shown.length) return '';
  const items = shown.map(p => `<div class=pagequote>
    <span class=pg-label>page ${p.page_number} &middot; ${esc(ROUTE_NAME[p.route] || p.route || '?')}</span>
    &ldquo;${esc(p.preview)}&rdquo;
  </div>`).join('');
  return `<div class=pagequotes>${items}</div>`;
}

function chunkTable(chunks){
  const rows = chunks.map(c => `<tr>
    <td>${esc((c.heading_path || []).join(' › '))}</td>
    <td class=preview>&ldquo;${esc(c.preview || '')}&rdquo;</td>
    <td>p${c.page_start}-${c.page_end}</td>
    <td>${esc((c.extraction_routes || []).join(','))}</td>
  </tr>`).join('');
  return `<table class=chunktbl><thead><tr><th>Heading</th><th>What's in this piece</th><th>Pages</th><th>Routes</th></tr></thead><tbody>${rows}</tbody></table>`;
}

function cardHtml(c){
  const name = esc(c.path.split('/').pop());
  const idx = (c.index != null && c.total != null) ? `${c.index}/${c.total}` : '';
  let body = '';
  if (c.page_count != null){
    body += `<div class=sect><h4>Extraction &middot; ${esc(c.extractor || '?')} &middot; ${c.page_count} pages</h4>
      <p class=explain>Reading the words off each page &mdash; picking a different method per page (plain text, a vision model, or OCR) depending on how the page is laid out.</p>
      <div class=pagestrip>${pageStrip(c.pages || [])}</div>
      ${pageQuotes(c.pages || [])}</div>`;
  }
  if (c.chunk_count != null){
    body += `<div class=sect><h4>Chunking &middot; ${c.chunk_count} chunks</h4>
      <p class=explain>Splitting the document into smaller, self-contained pieces small enough to search precisely, each still tagged with the section heading it came from.</p>
      ${chunkTable(c.chunks || [])}</div>`;
  }
  if (c.dense_embedded != null){
    const warn = c.dense_missing > 0;
    body += `<div class=sect><h4>Indexing</h4>
      <p class=explain>Converting each piece above into a form the search engine can match by meaning, not just by matching words &mdash; so a question phrased differently than the document can still find it.</p>
      <span class="${warn ? 'dense-warn' : 'dense-ok'}">
      ${c.dense_embedded}/${c.chunk_count} pieces indexed for meaning-based search${warn ? ` &mdash; ${c.dense_missing} missing` : ''}</span></div>`;
  }
  if (c.error){
    body += `<div class=sect><h4>Error</h4><div class=err>${esc(c.error)}</div></div>`;
  }
  return `<details class=card data-path="${esc(c.path)}">
    <summary><span class=idx>${idx}</span><span class=name>${name}</span><span class="status ${c.status}">${c.status}</span></summary>
    <div class=body>${body}</div>
  </details>`;
}

function fmtElapsed(seconds){
  const s = Math.max(0, Math.round(seconds));
  const m = Math.floor(s / 60), r = s % 60;
  return `${m}m ${r}s`;
}

async function tick(){
  let data;
  try {
    const resp = await fetch('/api/events');
    data = await resp.json();
  } catch (e) {
    return;
  }
  const {cards, buildStart, buildDone, lastDocStart, degradations} = buildCards(data.events || []);

  const total = buildStart ? buildStart.total_todo : 0;
  const done = lastDocStart ? lastDocStart.index : 0;
  const pct = total > 0 ? Math.min(100, Math.round(100 * done / total)) : 0;
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('progress-label').textContent = buildDone
    ? `${buildDone.docs} docs, ${buildDone.chunks} chunks — build complete`
    : (total ? `${done} / ${total} documents` : 'waiting for build_start…');

  let elapsedSec = 0;
  if (buildDone){
    elapsedSec = buildDone.elapsed_s;
  } else if (buildStart){
    elapsedSec = (Date.now() - new Date(buildStart.ts).getTime()) / 1000;
  }
  document.getElementById('stat-elapsed').textContent = fmtElapsed(elapsedSec);
  document.getElementById('stat-current').textContent = buildDone ? 'done' : (lastDocStart ? lastDocStart.path.split('/').pop() : '–');

  const degradeEl = document.getElementById('stat-degrade');
  degradeEl.textContent = `${degradations} degradation${degradations === 1 ? '' : 's'}`;
  degradeEl.classList.toggle('hot', degradations > 0);

  const timeline = document.getElementById('timeline');
  if (cards.length === 0){
    timeline.innerHTML = '<div class=empty>No events yet.</div>';
  } else {
    // Preserve which cards are expanded across re-renders.
    const openPaths = new Set(
      [...timeline.querySelectorAll('details[open]')].map(d => d.dataset.path)
    );
    timeline.innerHTML = cards.slice().reverse().map(cardHtml).join('');
    for (const d of timeline.querySelectorAll('details')){
      if (openPaths.has(d.dataset.path)) d.open = true;
    }
  }

  const footer = document.getElementById('footer');
  if (buildDone){
    footer.className = 'complete';
    footer.innerHTML = `build complete &middot; ${buildDone.docs} docs &middot; ${buildDone.chunks} chunks &middot; ` +
      `report: ${esc(buildDone.report_path)}`;
  }
}

tick();
setInterval(tick, 1000);
</script>
</body></html>"""


def _read_events(path: Path) -> list[dict]:
    events = []
    if not path.is_file():
        return events
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, body: bytes | str, status: int = 200, ctype: str = "text/html; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/health":
            return self._send("ok", ctype="text/plain")
        if u.path == "/api/events":
            events = _read_events(_EVENTS_PATH)
            return self._send(json.dumps({"events": events}), ctype="application/json")
        if u.path == "/":
            page = _PAGE.replace("__CSS__", _CSS).replace("__EVENTS_PATH__", str(_EVENTS_PATH))
            return self._send(page)
        self._send("not found", status=404, ctype="text/plain")


def main():
    global _EVENTS_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True, help="path to build_events.jsonl")
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    _EVENTS_PATH = Path(args.events)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"build dashboard -> http://{args.host}:{args.port}  (watching {_EVENTS_PATH})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
