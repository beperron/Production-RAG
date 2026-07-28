#!/usr/bin/env python3
"""Build-time pedantic dashboard — live view of a running scripts/build_kb.py.

Reads the build_log table a build writes straight into Postgres (see
src/parsevault/pipeline/build_events.py and postgres/init/001_schema.sql)
and serves a browser panel that polls and re-renders it. Two independent
processes, decoupled by Postgres rather than a file: the dashboard can point
at a build running on another machine on the same network (the local stack
binds 0.0.0.0:5433), and can reopen a finished build's log later for review.

Multiple builds share one build_log table, distinguished by build_id. The
dashboard lists every build_id it finds under /api/builds and a picker in
the UI switches which one /api/events streams — no restart needed when a
new build starts.

http.server + ThreadingHTTPServer, matching scripts/law_search_server.py's
structure, plus psycopg for the Postgres reads. No auth — localhost-only,
same posture as that server.

Run:  python scripts/build_dashboard_server.py [--dsn postgresql://...] [--port 8901]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import psycopg

from parsevault.pipeline.build_events import default_dsn

_DSN = default_dsn()

# Directories the dashboard is allowed to read raw files from (page rasters,
# source PDFs) for the hover-preview/click-through feature. Colon-separated;
# resolved once at import time. Configurable rather than hardcoded to
# knowledge-base/ because build_kb.py's own default KB_SRC is
# tests/Test-Docs/NC-Forms.
def _default_serve_roots() -> list[Path]:
    raw = os.environ.get("BUILD_DASHBOARD_SERVE_ROOTS", "knowledge-base:tests/Test-Docs")
    return [Path(p).resolve() for p in raw.split(":") if p]


_SERVE_ROOTS = _default_serve_roots()


def _safe_resolve(raw_path: str, *, suffix: str) -> Path | None:
    """Resolve a query-string path and confirm it's a real file, of the expected
    type, inside one of the configured serve roots. Returns None (-> 404) on any
    violation — this is the only thing standing between a localhost-only server
    and an arbitrary-file-read via ?path=../../etc/passwd."""
    try:
        p = Path(raw_path).resolve()
    except (OSError, ValueError):
        return None
    if p.suffix.lower() != suffix or not p.is_file():
        return None
    if not any(p == root or root in p.parents for root in _SERVE_ROOTS):
        return None
    return p

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
header{padding:22px 20px 14px;max-width:1320px;margin:0 auto}
h1{font:600 24px/1.2 var(--display);margin:0}
.header-top{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:0 0 10px}
.build-picker-wrap{display:flex;align-items:center;gap:8px;margin:0 0 12px}
.build-picker-wrap label{font:11px var(--mono);color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
#build-picker{font:13px var(--mono);background:var(--card);color:var(--ink);border:1px solid var(--line);
  border-radius:6px;padding:5px 8px;flex:1;max-width:520px}
.tz-toggle{display:flex;font:11px var(--mono);border:1px solid var(--line);border-radius:6px;overflow:hidden;flex:none}
.tz-btn{background:var(--card);color:var(--muted);border:none;padding:5px 10px;cursor:pointer}
.tz-btn.active{background:var(--seal);color:#fff}
.tz-btn:not(.active):hover{background:var(--soft)}
.run-pills{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 12px}
.run-pill{display:flex;align-items:center;gap:6px;font:11px var(--mono);color:var(--seal-deep);
  background:#FCEEEE;border:1px solid var(--seal);border-radius:999px;padding:3px 10px 3px 8px;
  cursor:pointer;max-width:280px}
.run-pill:hover{background:#FBDFDF}
.run-pill .dot{width:7px;height:7px;border-radius:50%;background:var(--seal);flex:none;
  animation:run-pill-throb 1.2s ease-in-out infinite}
.run-pill .run-id{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
@keyframes run-pill-throb{0%,100%{transform:scale(1);opacity:1}50%{transform:scale(1.6);opacity:.45}}
.build-path{font:11px var(--mono);color:var(--muted);margin:2px 0 0;word-break:break-all}
.progress-wrap{margin:10px 0}
.progress-bar{height:10px;border-radius:6px;background:var(--soft);border:1px solid var(--line);overflow:hidden}
.progress-fill{height:100%;background:var(--seal);transition:width .3s ease}
.progress-label{font:12px var(--mono);color:var(--muted);margin-top:4px}
.stat-row{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 14px;min-width:140px}
.stat-fill{flex:1 1 240px;min-width:0}
.stat-fill .stat-val{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.stat-label{font:11px var(--mono);color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.stat-val{font:600 16px var(--display);margin-top:2px}
.degrade-badge{background:var(--soft);color:var(--ok);font:600 15px var(--mono);border:1px solid var(--line)}
.degrade-badge.hot{background:#FCEEEE;color:var(--seal-deep);border-color:var(--seal)}
main{max-width:1320px;margin:14px auto 60px;padding:0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:0 0 10px;padding:0}
.card summary{cursor:pointer;list-style:none;padding:12px 16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.card summary::-webkit-details-marker{display:none}
.card .idx{font:12px var(--mono);color:var(--muted);min-width:44px}
.card .name{font:600 14px var(--display);flex:1;min-width:180px;overflow-wrap:anywhere}
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
  justify-content:center;color:#fff;cursor:pointer}
.pg.native{background:#3F7A57}
.pg.tesseract{background:#B8860B}
.pg.vlm{background:#2A5560}
.pg.llamaparse{background:#7A4FA3}
.pg.text{background:#5B6B73}
.pg.docx{background:#8B5E34}
.pg.flagged{outline:2px solid var(--seal-deep);outline-offset:1px}
.pg.has-preview{box-shadow:inset 0 0 0 1px rgba(255,255,255,.5)}
#page-preview-popup{display:none;position:absolute;z-index:50;background:var(--card);
  border:1px solid var(--line);border-radius:8px;padding:4px;box-shadow:0 4px 14px rgba(0,0,0,.18)}
#page-preview-popup img{display:block;max-width:320px;max-height:420px;border-radius:4px}
.timechart-wrap{margin-top:12px}
.timechart-wrap h2{font:600 12px var(--mono);text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin:0 0 8px}
.timechart{display:flex;flex-direction:column;gap:2px}
.timechart-row{display:flex;align-items:center;gap:8px}
.timechart-label{font:12px var(--mono);color:var(--ink);width:110px;flex:none;text-align:right}
.timechart-track{flex:1;background:var(--soft);border-radius:4px;height:18px;overflow:hidden}
.timechart-fill{height:100%;border-radius:0 4px 4px 0;transition:width .3s ease}
.timechart-val{font:12px var(--mono);color:var(--muted);width:64px;flex:none}
.timechart-empty{font:12px var(--mono);color:var(--muted);font-style:italic}
.chunktbl-wrap{overflow-x:auto}
table.chunktbl{width:100%;border-collapse:collapse;font:12.5px var(--mono)}
table.chunktbl th{text-align:left;color:var(--muted);font-weight:600;padding:4px 8px;border-bottom:1px solid var(--line)}
table.chunktbl td{padding:4px 8px;border-bottom:1px solid var(--line-soft);vertical-align:top;overflow-wrap:anywhere}
table.chunktbl td.heading{max-width:180px}
table.chunktbl td.size,table.chunktbl td.pages,table.chunktbl td.routes{color:var(--muted);white-space:nowrap}
table.chunktbl td.preview{font:12.5px/1.4 var(--display);color:var(--ink);white-space:normal;overflow-wrap:anywhere;min-width:260px}
.chunk-more-row td{text-align:center;padding:8px}
.chunk-more-btn{font:12px var(--mono);background:var(--soft);color:var(--link);border:1px solid var(--line);
  border-radius:6px;padding:5px 12px;cursor:pointer}
.chunk-more-btn:hover{background:var(--line-soft)}
.hash-chip{font:11px var(--mono);color:var(--muted);background:var(--soft);border:1px solid var(--line);
  border-radius:6px;padding:1px 7px;cursor:default;text-transform:none;letter-spacing:0}
.pagequotes{display:flex;flex-direction:column;gap:6px;margin-top:8px}
.pagequote{background:var(--soft);border:1px solid var(--line-soft);border-left:3px solid var(--line);border-radius:0 6px 6px 0;
  padding:6px 10px;font:italic 13px/1.5 var(--display);color:var(--ink)}
.pagequote .pg-label{display:block;font:600 10.5px var(--mono);font-style:normal;color:var(--muted);text-transform:uppercase;letter-spacing:.03em;margin-bottom:2px}
.dense-ok{color:var(--ok);font:600 13px var(--mono)}
.dense-warn{color:var(--seal-deep);font:600 13px var(--mono);background:#FCEEEE;border-radius:6px;padding:2px 8px;display:inline-block}
.err{color:var(--seal-deep);font:13px var(--mono);white-space:pre-wrap}
.empty{color:var(--muted);text-align:center;padding:40px 0;font-style:italic}
footer{max-width:1320px;margin:0 auto;padding:20px;color:var(--muted);font:11px var(--mono);border-top:1px solid var(--line-soft)}
footer.complete{color:var(--ok)}
"""

_PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Corpus Build Dashboard</title><style>__CSS__</style></head><body>
<div class=banner>CORPUS BUILD DASHBOARD &middot; localhost, read-only</div>
<header>
  <div class=header-top>
    <h1>Corpus Build Dashboard</h1>
    <div class=tz-toggle role=group aria-label="Timestamp timezone — applies to displayed clock times, not build IDs">
      <button type=button id=tz-local class=tz-btn>Local</button>
      <button type=button id=tz-utc class=tz-btn>UTC</button>
    </div>
  </div>
  <div class=run-pills id=run-pills></div>
  <div class=build-picker-wrap>
    <label for=build-picker>Build</label>
    <select id=build-picker><option value="">loading builds&hellip;</option></select>
  </div>
  <div class=build-path id=build-path></div>
  <div class=progress-wrap>
    <div class=progress-bar><div class=progress-fill id=progress-fill style="width:0%"></div></div>
    <div class=progress-label id=progress-label>waiting for events&hellip;</div>
  </div>
  <div class=stat-row>
    <div class=stat><div class=stat-label>Started</div><div class=stat-val id=stat-started>&ndash;</div></div>
    <div class=stat><div class=stat-label>Elapsed</div><div class=stat-val id=stat-elapsed>&ndash;</div></div>
    <div class="stat stat-fill"><div class=stat-label>Current doc</div><div class=stat-val id=stat-current title="">&ndash;</div></div>
    <div class="stat degrade-badge" id=stat-degrade>0 degradations</div>
  </div>
  <div class=timechart-wrap>
    <h2>Time by extraction route</h2>
    <div class=timechart id=timechart><div class=timechart-empty>no pages extracted yet</div></div>
  </div>
</header>
<main id=timeline><div class=empty>No events yet.</div></main>
<footer id=footer></footer>
<script>
const ROUTE_LABEL = {native:'N', tesseract:'T', vlm:'V', llamaparse:'L', text:'P', docx:'D'};
const ROUTE_NAME = {native:'native text layer', tesseract:'Tesseract OCR', vlm:'vision-language model', llamaparse:'LlamaParse (cloud)', text:'plain text', docx:'Word text layer'};
// Same hexes as the .pg.* CSS rules above — kept in sync by hand since the
// bar chart below is plain divs, not styled via those classes.
const ROUTE_COLOR = {native:'#3F7A57', tesseract:'#B8860B', vlm:'#2A5560', llamaparse:'#7A4FA3', text:'#5B6B73', docx:'#8B5E34'};
const ROUTE_ORDER = ['native', 'vlm', 'tesseract', 'llamaparse', 'text', 'docx'];

let CURRENT_BUILD = new URLSearchParams(location.search).get('build') || '';
let BUILDS = [];
let TZ_MODE = localStorage.getItem('buildDashboardTz') || 'local';  // 'local' | 'utc'

// Build/run IDs are opaque identifiers, not times — shown verbatim everywhere,
// never reformatted. The toggle only touches genuine wall-clock displays, e.g.
// event timestamps pulled from build_log's `ts` (real Postgres timestamptz).
function fmtTs(isoString){
  if (!isoString) return '–';
  const date = new Date(isoString);
  if (isNaN(date.getTime())) return isoString;
  const opts = {year: 'numeric', month: 'short', day: '2-digit',
                hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false};
  return TZ_MODE === 'utc'
    ? date.toLocaleString('en-US', {...opts, timeZone: 'UTC'}) + ' UTC'
    : date.toLocaleString('en-US', opts);
}

function setTzMode(mode){
  TZ_MODE = mode;
  localStorage.setItem('buildDashboardTz', mode);
  document.getElementById('tz-local').classList.toggle('active', mode === 'local');
  document.getElementById('tz-utc').classList.toggle('active', mode === 'utc');
  tick();  // re-render immediately with the new mode rather than waiting for the next poll
}

function esc(s){
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function buildCards(events){
  const cards = [];
  const byPath = new Map();
  const byDocId = new Map();
  let buildStart = null, buildDone = null, lastDocStart = null;
  let degradations = 0;
  const routeTime = {};  // route -> cumulative elapsed_seconds across all pages seen so far

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
        card.source_sha256 = e.source_sha256;
        card.extractor = e.extractor;
        card.page_count = e.page_count;
        card.pages = e.pages || [];
        card.degradations = e.degradations || [];
        card.extracted_at = e.ts;
        card.status = 'chunking';
        degradations += card.degradations.length;
        byDocId.set(e.doc_id, card);
        for (const p of card.pages){
          const route = p.route || '?';
          routeTime[route] = (routeTime[route] || 0) + (p.elapsed_seconds || 0);
        }
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
  return {cards, buildStart, buildDone, lastDocStart, degradations, routeTime};
}

function pageStrip(pages, docPath){
  return pages.map(p => {
    const route = p.route || '?';
    const flagged = p.flagged ? ' flagged' : '';
    const hasPreview = p.raster_path ? ' has-preview' : '';
    const routeName = ROUTE_NAME[route] || route;
    const timing = p.elapsed_seconds != null ? `, ${p.elapsed_seconds.toFixed(2)}s` : '';
    const title = p.flagged
      ? esc((p.flag_reasons || []).join('; '))
      : `page ${p.page_number} — read via ${routeName} (${p.chars} chars${timing})`;
    const rasterAttr = p.raster_path ? ` data-raster="${esc(p.raster_path)}"` : '';
    return `<div class="pg ${esc(route)}${flagged}${hasPreview}" title="${title}" data-page="${p.page_number}" data-doc="${esc(docPath)}"${rasterAttr}>${ROUTE_LABEL[route] || '?'}</div>`;
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

// Paths whose full chunk table has been expanded past the row cap. Kept
// outside cardHtml() because the whole timeline re-renders every tick — a
// plain DOM-open check (like <details open>) would forget the choice a
// second later.
const EXPANDED_CHUNK_TABLES = new Set();
const CHUNK_TABLE_ROW_CAP = 40;

function fmtChunkSize(c){
  const chars = c.char_len;
  if (chars == null) return '–';
  const tokens = c.token_estimate;
  return tokens != null ? `${chars} ch / ~${tokens} tok` : `${chars} ch`;
}

function chunkRowHtml(c){
  return `<tr>
    <td class=heading>${esc((c.heading_path || []).join(' › '))}</td>
    <td class=preview>&ldquo;${esc(c.preview || '')}&rdquo;</td>
    <td class=size>${fmtChunkSize(c)}</td>
    <td class=pages>p${c.page_start}-${c.page_end}</td>
    <td class=routes>${esc((c.extraction_routes || []).join(','))}</td>
  </tr>`;
}

function chunkTable(chunks, path){
  const expanded = EXPANDED_CHUNK_TABLES.has(path);
  const shown = expanded ? chunks : chunks.slice(0, CHUNK_TABLE_ROW_CAP);
  const hidden = chunks.length - shown.length;
  const rows = shown.map(chunkRowHtml).join('');
  let more = '';
  if (hidden > 0 || (expanded && chunks.length > CHUNK_TABLE_ROW_CAP)){
    const label = expanded ? 'Show fewer chunks' : `Show ${hidden} more chunks`;
    more = `<tr class=chunk-more-row><td colspan=5>
      <button type=button class=chunk-more-btn data-path="${esc(path)}">${label}</button>
    </td></tr>`;
  }
  return `<div class=chunktbl-wrap><table class=chunktbl><thead><tr>
    <th>Heading</th><th>What's in this piece</th><th>Size</th><th>Pages</th><th>Routes</th>
  </tr></thead><tbody>${rows}${more}</tbody></table></div>`;
}

// Cumulative wall-clock time spent per extraction lane, across every page
// seen so far in this build — local compute time is the honest cost proxy
// here (no cloud $ during a build: the cascade never leaves the machine).
function fmtRouteSeconds(seconds){
  return seconds < 60 ? `${seconds.toFixed(1)}s` : fmtElapsed(seconds);
}

function renderTimeChart(routeTime){
  const el = document.getElementById('timechart');
  const seen = new Set(ROUTE_ORDER);
  const routes = [...ROUTE_ORDER, ...Object.keys(routeTime).filter(r => !seen.has(r))];
  const entries = routes
    .filter(route => (routeTime[route] || 0) > 0)
    .map(route => ({route, seconds: routeTime[route]}));
  if (entries.length === 0){
    el.innerHTML = '<div class=timechart-empty>no pages extracted yet</div>';
    return;
  }
  const max = Math.max(...entries.map(e => e.seconds));
  el.innerHTML = entries.map(({route, seconds}) => {
    const pct = max > 0 ? Math.max(2, Math.round(100 * seconds / max)) : 0;
    const color = ROUTE_COLOR[route] || '#8A8577';
    const label = ROUTE_NAME[route] || route;
    return `<div class=timechart-row>
      <span class=timechart-label>${esc(label)}</span>
      <div class=timechart-track><div class=timechart-fill style="width:${pct}%;background:${color}"></div></div>
      <span class=timechart-val>${fmtRouteSeconds(seconds)}</span>
    </div>`;
  }).join('');
}

function hashChip(sha256){
  if (!sha256) return '';
  const short = sha256.slice(0, 12);
  return ` &middot; <span class=hash-chip title="sha256:${esc(sha256)}">sha256:${esc(short)}&hellip;</span>`;
}

function cardHtml(c){
  const name = esc(c.path.split('/').pop());
  const idx = (c.index != null && c.total != null) ? `${c.index}/${c.total}` : '';
  let body = '';
  if (c.page_count != null){
    body += `<div class=sect><h4>Extraction &middot; ${esc(c.extractor || '?')} &middot; ${c.page_count} pages &middot; ${esc(fmtTs(c.extracted_at))}${hashChip(c.source_sha256)}</h4>
      <p class=explain>Reading the words off each page &mdash; picking a different method per page (plain text, a vision model, or OCR) depending on how the page is laid out.</p>
      <div class=pagestrip>${pageStrip(c.pages || [], c.path)}</div>
      ${pageQuotes(c.pages || [])}</div>`;
  }
  if (c.chunk_count != null){
    body += `<div class=sect><h4>Chunking &middot; ${c.chunk_count} chunks</h4>
      <p class=explain>Splitting the document into smaller, self-contained pieces small enough to search precisely, each still tagged with the section heading it came from.</p>
      ${chunkTable(c.chunks || [], c.path)}</div>`;
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

function fmtBuildOption(b){
  const count = b.status === 'done'
    ? b.docs
    : (b.total_todo ? `${b.done}/${b.total_todo}` : b.docs_indexed);
  return `${b.label} — ${count} docs (${b.status})`;
}

function selectBuild(id){
  CURRENT_BUILD = id;
  const url = new URL(location);
  if (CURRENT_BUILD) url.searchParams.set('build', CURRENT_BUILD);
  else url.searchParams.delete('build');
  history.replaceState(null, '', url);
  tick();
}

function renderRunPills(builds){
  const wrap = document.getElementById('run-pills');
  const running = builds.filter(b => b.status === 'running');
  wrap.innerHTML = running.map(b =>
    `<div class=run-pill data-id="${esc(b.id)}" title="jump to ${esc(b.id)}">` +
    `<span class=dot></span><span class=run-id>${esc(b.id)}</span></div>`
  ).join('');
  for (const el of wrap.querySelectorAll('.run-pill')){
    el.addEventListener('click', () => {
      const picker = document.getElementById('build-picker');
      picker.value = el.dataset.id;
      selectBuild(el.dataset.id);
    });
  }
}

async function refreshBuilds(){
  let data;
  try {
    const resp = await fetch('/api/builds');
    data = await resp.json();
  } catch (e) {
    return;
  }
  BUILDS = data.builds || [];
  renderRunPills(BUILDS);
  const picker = document.getElementById('build-picker');
  if (BUILDS.length === 0){
    picker.innerHTML = '<option value="">no builds found</option>';
    CURRENT_BUILD = '';
    document.getElementById('build-path').textContent = '';
    return;
  }
  const stillValid = BUILDS.some(b => b.id === CURRENT_BUILD);
  if (!CURRENT_BUILD || !stillValid) CURRENT_BUILD = BUILDS[0].id;
  picker.innerHTML = BUILDS.map(b =>
    `<option value="${esc(b.id)}"${b.id === CURRENT_BUILD ? ' selected' : ''}>${esc(fmtBuildOption(b))}</option>`
  ).join('');
  const active = BUILDS.find(b => b.id === CURRENT_BUILD);
  document.getElementById('build-path').textContent = active ? active.path : '';
}

async function tick(){
  await refreshBuilds();
  if (!CURRENT_BUILD){
    document.getElementById('timeline').innerHTML = '<div class=empty>No builds found under the configured roots yet.</div>';
    return;
  }
  let data;
  try {
    const resp = await fetch('/api/events?build=' + encodeURIComponent(CURRENT_BUILD));
    data = await resp.json();
  } catch (e) {
    return;
  }
  const {cards, buildStart, buildDone, lastDocStart, degradations, routeTime} = buildCards(data.events || []);
  renderTimeChart(routeTime);

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
  document.getElementById('stat-started').textContent = buildStart ? fmtTs(buildStart.ts) : '–';
  const currentName = buildDone ? 'done' : (lastDocStart ? lastDocStart.path.split('/').pop() : '–');
  const statCurrent = document.getElementById('stat-current');
  statCurrent.textContent = currentName;
  statCurrent.title = currentName;

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
    for (const btn of timeline.querySelectorAll('.chunk-more-btn')){
      btn.addEventListener('click', () => {
        const path = btn.dataset.path;
        if (EXPANDED_CHUNK_TABLES.has(path)) EXPANDED_CHUNK_TABLES.delete(path);
        else EXPANDED_CHUNK_TABLES.add(path);
        tick();
      });
    }
  }

  const footer = document.getElementById('footer');
  if (buildDone){
    footer.className = 'complete';
    footer.innerHTML = `build complete &middot; ${buildDone.docs} docs &middot; ${buildDone.chunks} chunks &middot; ` +
      `finished ${esc(fmtTs(buildDone.ts))} &middot; report: ${esc(buildDone.report_path)}`;
  } else {
    footer.className = '';
    footer.innerHTML = '';
  }
}

document.getElementById('build-picker').addEventListener('change', (ev) => {
  selectBuild(ev.target.value);
});
document.getElementById('tz-local').addEventListener('click', () => setTzMode('local'));
document.getElementById('tz-utc').addEventListener('click', () => setTzMode('utc'));
setTzMode(TZ_MODE);

// Page-badge hover preview + click-through to the source file. Delegated on
// #timeline (a stable container) rather than bound per-.pg element, since the
// whole timeline is replaced via innerHTML every tick.
document.getElementById('timeline').addEventListener('click', (ev) => {
  const pg = ev.target.closest('.pg');
  if (!pg || !pg.dataset.doc) return;
  const url = '/api/source?path=' + encodeURIComponent(pg.dataset.doc) + '#page=' + pg.dataset.page;
  window.open(url, '_blank');
});

let previewPopup = null;
function ensurePreviewEl(){
  if (previewPopup) return previewPopup;
  const wrap = document.createElement('div');
  wrap.id = 'page-preview-popup';
  wrap.innerHTML = '<img id="page-preview-img">';
  document.body.appendChild(wrap);
  previewPopup = wrap;
  return wrap;
}
document.getElementById('timeline').addEventListener('mouseover', (ev) => {
  const pg = ev.target.closest('.pg.has-preview');
  if (!pg) return;
  const popup = ensurePreviewEl();
  const img = popup.querySelector('img');
  if (img.dataset.loadedFor !== pg.dataset.raster){
    img.src = '/api/raster?path=' + encodeURIComponent(pg.dataset.raster);
    img.dataset.loadedFor = pg.dataset.raster;
  }
  const rect = pg.getBoundingClientRect();
  popup.style.left = (rect.left + window.scrollX) + 'px';
  popup.style.top = (rect.bottom + window.scrollY + 6) + 'px';
  popup.style.display = 'block';
});
document.getElementById('timeline').addEventListener('mouseout', (ev) => {
  if (!ev.target.closest('.pg.has-preview')) return;
  const popup = document.getElementById('page-preview-popup');
  if (popup) popup.style.display = 'none';
});

tick();
setInterval(tick, 1000);
</script>
</body></html>"""


@contextmanager
def _connect():
    conn = psycopg.connect(_DSN)
    try:
        yield conn
    finally:
        conn.close()


def _read_events(conn, build_id: str) -> list[dict]:
    rows = conn.execute(
        "select ts, stage, fields from build_log where build_id = %s order by seq",
        (build_id,),
    ).fetchall()
    events = []
    for ts, stage, fields in rows:
        e = dict(fields or {})
        e["ts"] = ts.isoformat(timespec="seconds")
        e["stage"] = stage
        events.append(e)
    return events


def _discover_builds(conn) -> list[dict]:
    rows = conn.execute(
        "select build_id, max(ts) as last_seen "
        "from build_log group by build_id order by last_seen desc"
    ).fetchall()
    return [{"build_id": build_id, "last_seen": last_seen} for build_id, last_seen in rows]


# A killed/crashed build never writes build_done, so without this it would
# show as "running" (and throb in the pills) forever — there's no event that
# says "the process died". Once build_log stays quiet for this long with no
# terminal event, treat it as stalled rather than live. Generous on purpose:
# a single large scanned doc can spend minutes in Tesseract/VLM between
# events, and that's still a real running build, not a dead one.
_STALE_AFTER = datetime.timedelta(seconds=300)


def _summarize(events: list[dict], last_seen: datetime.datetime | None = None) -> dict:
    total_todo = 0
    done_idx = 0
    build_done = None
    build_aborted = None
    degradations = 0
    doc_errors = 0
    docs_indexed = 0
    for e in events:
        stage = e.get("stage")
        if stage == "build_start":
            total_todo = e.get("total_todo", 0)
        elif stage == "doc_start":
            done_idx = e.get("index", done_idx)
        elif stage == "extract_done":
            degradations += len(e.get("degradations") or [])
        elif stage == "index_done":
            docs_indexed += 1
            if (e.get("dense_missing") or 0) > 0:
                degradations += 1
        elif stage == "doc_error":
            doc_errors += 1
            degradations += 1
        elif stage == "build_done":
            build_done = e
        elif stage == "build_aborted":
            build_aborted = e
    if build_done:
        status = "done"
    elif build_aborted:
        status = "aborted"
    elif events:
        stale = last_seen is not None and (
            datetime.datetime.now(datetime.timezone.utc) - last_seen > _STALE_AFTER
        )
        status = "stalled" if stale else "running"
    else:
        status = "idle"
    return {
        "status": status,
        "total_todo": total_todo,
        "done": done_idx,
        "docs_indexed": docs_indexed,
        "doc_errors": doc_errors,
        "degradations": degradations,
        "elapsed_s": build_done.get("elapsed_s") if build_done else None,
        "chunks": build_done.get("chunks") if build_done else None,
        "docs": build_done.get("docs") if build_done else None,
    }


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
        if u.path == "/api/builds":
            try:
                with _connect() as conn:
                    builds = _discover_builds(conn)
                    out = []
                    for b in builds:
                        events = _read_events(conn, b["build_id"])
                        summary = _summarize(events, b["last_seen"])
                        out.append({
                            "id": b["build_id"],
                            "path": b["build_id"],
                            "label": b["build_id"],
                            "mtime": b["last_seen"].timestamp() if b["last_seen"] else 0,
                            **summary,
                        })
            except psycopg.Error as e:
                return self._send(json.dumps({"builds": [], "error": str(e)}), ctype="application/json")
            out.sort(key=lambda b: b["mtime"], reverse=True)
            return self._send(json.dumps({"builds": out}), ctype="application/json")
        if u.path == "/api/events":
            qs = parse_qs(u.query)
            build_id = (qs.get("build") or [""])[0]
            events = []
            if build_id:
                try:
                    with _connect() as conn:
                        events = _read_events(conn, build_id)
                except psycopg.Error:
                    events = []
            return self._send(json.dumps({"events": events}), ctype="application/json")
        if u.path == "/api/raster":
            qs = parse_qs(u.query)
            p = _safe_resolve((qs.get("path") or [""])[0], suffix=".png")
            if p is None:
                return self._send("not found or not allowed", status=404, ctype="text/plain")
            return self._send(p.read_bytes(), ctype="image/png")
        if u.path == "/api/source":
            qs = parse_qs(u.query)
            p = _safe_resolve((qs.get("path") or [""])[0], suffix=".pdf")
            if p is None:
                return self._send("not found or not allowed", status=404, ctype="text/plain")
            # No Content-Disposition: attachment — must render inline so the
            # browser's built-in PDF viewer honors a #page=N fragment.
            return self._send(p.read_bytes(), ctype="application/pdf")
        if u.path == "/":
            page = _PAGE.replace("__CSS__", _CSS)
            return self._send(page)
        self._send("not found", status=404, ctype="text/plain")


def main():
    global _DSN
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=None,
                     help="Postgres DSN for the build_log table "
                          "(default: $BUILD_LOG_DSN, or the local carolina-policy "
                          "stack on port 5433)")
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    if args.dsn:
        _DSN = args.dsn
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"build dashboard -> http://{args.host}:{args.port}  (postgres build_log)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
