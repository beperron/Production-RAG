#!/usr/bin/env python3
"""Assemble the three arms into one frozen evaluation set, and build the
review UI for the human pass.

    build_eval_set.py -o 2_eval/mcr_eval_v1.jsonl --review 2_eval/REVIEW.html

Gold is a citation string throughout, so the set survives a reparse. Every
record records which family wrote it and which family checked it, so the
leaderboard can later be re-run PER ARM -- if rank order holds across two
unrelated generators, the result is generator-independent, which is a claim a
single-generator benchmark cannot make.

The review sample is deliberately NOT random. A random sample estimates the
error rate; a disputed sample tells you which generator to trust, and that is
the open question this set cannot answer by itself.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import html
import json
import pathlib
import random
import sys

SOURCES = [
    ("2_eval/arm_a.merged.jsonl", "A"),
    ("2_eval/arm_a.reval.jsonl", "A"),
    ("2_eval/arm_b.validated.jsonl", "B"),
    ("2_eval/arm_b.reval.jsonl", "B"),
    ("2_eval/arm_c.jsonl", "C"),
]


def load_all(root):
    """Later files win: a regenerated+revalidated record supersedes its
    original, keyed by query_id."""
    best = {}
    for rel, _arm in SOURCES:
        p = root / rel
        if not p.exists():
            continue
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                best[r["query_id"]] = r
    return list(best.values())


def sample_for_review(recs, n, seed):
    """Strata ordered by how much a human decision is worth on each."""
    rng = random.Random(seed)
    used, out = set(), []

    def take(pool, k, why):
        pool = [r for r in pool if r["query_id"] not in used]
        rng.shuffle(pool)
        for r in pool[:k]:
            used.add(r["query_id"])
            out.append(dict(r, review_stratum=why))

    # 1. still failing after a rewrite -- keep or drop is a judgement call
    take([r for r in recs if r["status"] in ("rejected", "needs_revision")
          and r.get("regenerated")], int(n * 0.30), "failed-after-rewrite")
    # 2. a checker overrode the generator -- the disagreement itself
    take([r for r in recs if r["status"] in ("rejected", "needs_revision")],
         int(n * 0.20), "disputed")
    # 3. co-gold claims: is the second provision really co-valid?
    take([r for r in recs if r.get("also_answered_by")],
         int(n * 0.20), "co-gold-claimed")
    # 4. negatives: is it genuinely unanswerable from ANY rule?
    take([r for r in recs if r["query_type"] == "unanswerable"],
         int(n * 0.10), "unanswerable")
    # 5. an unbiased stratum so the error rate is still estimable
    take([r for r in recs if r["status"] == "accepted"],
         n - len(out), "accepted-random")
    return out


REVIEW = """<!doctype html>
<meta charset="utf-8"><title>MCR evaluation set — review</title>
<style>
:root{color-scheme:light dark;--bg:#fff;--fg:#111;--mut:#666;--line:#d8d8d8;
--ok:#0a7;--no:#c33;--fix:#c80;--card:#fafafa}
@media(prefers-color-scheme:dark){:root{--bg:#151515;--fg:#eee;--mut:#999;
--line:#333;--card:#1e1e1e}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);
padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:5}
#bar{flex:1;height:6px;background:var(--line);border-radius:3px;min-width:110px}
#bar>div{height:100%;background:var(--ok);border-radius:3px;width:0}
main{max-width:860px;margin:0 auto;padding:18px 16px 120px}
.q{font-size:19px;margin:12px 0 16px;font-weight:500}
.prov{border:1px solid var(--line);border-radius:8px;padding:12px 14px;
background:var(--card);white-space:pre-wrap;font-size:14px;max-height:340px;
overflow:auto}
.tag{display:inline-block;border:1px solid var(--line);border-radius:20px;
padding:1px 9px;font-size:12px;color:var(--mut);margin:0 6px 4px 0}
.tag.hot{border-color:var(--fix);color:var(--fix)}
.meta{color:var(--mut);font-size:13px;margin-bottom:8px}
.note{border-left:3px solid var(--fix);padding:8px 12px;margin:14px 0;
color:var(--mut);font-size:14px}
.prev{border-left:3px solid var(--line);padding:6px 12px;margin:10px 0;
color:var(--mut);font-size:13px}
footer{position:fixed;bottom:0;left:0;right:0;background:var(--bg);
border-top:1px solid var(--line);padding:10px 16px;display:flex;gap:8px;
justify-content:center;flex-wrap:wrap}
button{font:inherit;padding:8px 15px;border-radius:8px;border:1px solid var(--line);
background:var(--card);color:var(--fg);cursor:pointer}
button.ok{border-color:var(--ok);color:var(--ok)}
button.no{border-color:var(--no);color:var(--no)}
button.fix{border-color:var(--fix);color:var(--fix)}
kbd{font:12px ui-monospace,monospace;border:1px solid var(--line);
border-radius:4px;padding:0 4px;color:var(--mut)}
.done{text-align:center;padding:48px 0}
</style>
<header>
  <strong>MCR eval review</strong>
  <span id="pos" class="meta"></span>
  <div id="bar"><div></div></div>
  <span id="tally" class="meta"></span>
  <button id="dl">Download decisions</button>
</header>
<main id="app"></main>
<footer>
  <button class="ok" data-a="good">Question is sound <kbd>a</kbd></button>
  <button class="fix" data-a="badgold">Wrong gold <kbd>g</kbd></button>
  <button class="fix" data-a="ambiguous">Another rule answers too <kbd>c</kbd></button>
  <button class="no" data-a="drop">Drop it <kbd>r</kbd></button>
  <button data-a="skip">Skip <kbd>space</kbd></button>
  <button data-a="back">Back <kbd>&larr;</kbd></button>
</footer>
<script>
const ITEMS = __ITEMS__;
const HASH  = "__HASH__";
const KEY   = "mcr-eval-review-" + HASH.slice(0,12);
let dec = JSON.parse(localStorage.getItem(KEY) || "{}");
let i = 0;
const $ = s => document.querySelector(s);
const save = () => localStorage.setItem(KEY, JSON.stringify(dec));

function firstUndecided(){
  for (let k=0;k<ITEMS.length;k++) if(!dec[ITEMS[k].query_id]) return k;
  return ITEMS.length;
}
function render(){
  const done = Object.keys(dec).length;
  $("#bar>div").style.width = (100*done/ITEMS.length)+"%";
  const t={good:0,badgold:0,ambiguous:0,drop:0};
  Object.values(dec).forEach(d=>t[d.action]!==undefined&&t[d.action]++);
  $("#tally").textContent = `${t.good} sound / ${t.badgold} wrong gold / ${t.ambiguous} ambiguous / ${t.drop} dropped`;
  if(i>=ITEMS.length){
    $("#pos").textContent="";
    $("#app").innerHTML=`<div class="done"><h2>All ${ITEMS.length} reviewed.</h2>
      <p class="meta">Download the decisions file.</p></div>`;
    return;
  }
  const it=ITEMS[i], d=dec[it.query_id];
  $("#pos").textContent=`${i+1} / ${ITEMS.length}`;
  $("#app").innerHTML=`
    <div class="meta">
      <span class="tag hot">${it.review_stratum}</span>
      <span class="tag">${it.arm} · ${it.generator}</span>
      <span class="tag">${it.query_type}</span>
      <span class="tag">${it.status}</span>
      ${it.regenerated?'<span class="tag">rewritten</span>':''}
      ${it.echo!==undefined?`<span class="tag">echo ${it.echo}</span>`:''}
      ${d?`<span class="tag hot">recorded: ${d.action}</span>`:''}
    </div>
    <div class="q">${it.query}</div>
    <div class="meta"><b>Labelled to ${it.citation}</b>${
      it.also_answered_by&&it.also_answered_by.length
        ? ` &nbsp;·&nbsp; also credited: ${it.also_answered_by.join(", ")}` : ""}</div>
    <div class="prov">${it.provision||"(no text)"}</div>
    ${it.checker_note?`<div class="note"><b>${it.checker}:</b> ${it.checker_note}</div>`:""}
    ${it.query_previous?`<div class="prev"><b>before rewrite:</b> ${it.query_previous}</div>`:""}
    <div class="meta">Is this question answered by the provision above, and by
      that provision rather than some other rule?</div>`;
}
function record(action){
  dec[ITEMS[i].query_id]={action, at:new Date().toISOString()};
  save(); i++; render();
}
document.querySelectorAll("footer button").forEach(b=>b.onclick=()=>{
  const a=b.dataset.a;
  if(a==="skip"){i++;render();}
  else if(a==="back"){i=Math.max(0,i-1);render();}
  else record(a);
});
document.onkeydown=e=>{
  if(e.metaKey||e.ctrlKey)return;
  const k=e.key.toLowerCase();
  if(k==="a")record("good"); else if(k==="g")record("badgold");
  else if(k==="c")record("ambiguous"); else if(k==="r")record("drop");
  else if(k===" "){e.preventDefault();i++;render();}
  else if(k==="arrowleft"){i=Math.max(0,i-1);render();}
};
$("#dl").onclick=()=>{
  const blob=new Blob([JSON.stringify({set_hash:HASH,n:ITEMS.length,decisions:dec},null,1)],
    {type:"application/json"});
  const a=document.createElement("a");
  a.href=URL.createObjectURL(blob); a.download="review_decisions.json"; a.click();
};
i=firstUndecided(); render();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("-o", "--out", default="2_eval/mcr_eval_v1.jsonl")
    ap.add_argument("--review", default="2_eval/REVIEW.html")
    ap.add_argument("--review-n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260805)
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    recs = load_all(root)

    blocks = [json.loads(l) for l in open(root / "1_parsed/blocks.jsonl")]
    parts = collections.defaultdict(list)
    for b in blocks:
        if b.get("citation") and b["kind"] == "body":
            parts[b["citation"]].append(b["text"])
    text_of = {c: "\n\n".join(v) for c, v in parts.items()}

    ship = [r for r in recs if r["status"] in ("accepted",)
            or r["query_type"] == "citation_lookup"]
    for r in recs:
        r["provision"] = text_of.get(r["citation"], "")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in sorted(ship, key=lambda r: r["query_id"]):
            fh.write(json.dumps({k: v for k, v in r.items()
                                 if k != "provision"},
                                ensure_ascii=False, sort_keys=True) + "\n")

    h = hashlib.sha256()
    for r in sorted(ship, key=lambda r: r["query_id"]):
        h.update(json.dumps({k: r[k] for k in ("query_id", "query", "gold",
                                               "also_answered_by")},
                            sort_keys=True).encode())
    digest = h.hexdigest()

    # review sample -- flatten the checker's note for display
    sample = sample_for_review(recs, args.review_n, args.seed)
    for s in sample:
        v = s.get("validation") or {}
        s["checker"] = v.get("checker", "")
        s["checker_note"] = v.get("note", "")
    slim = [{k: s.get(k) for k in
             ("query_id", "query", "query_previous", "citation", "gold",
              "also_answered_by", "query_type", "arm", "generator", "status",
              "echo", "regenerated", "review_stratum", "checker",
              "checker_note", "provision")} for s in sample]
    for s in slim:
        for k in ("query", "query_previous", "provision", "checker_note"):
            if s.get(k):
                s[k] = html.escape(str(s[k]))
    page = REVIEW.replace("__ITEMS__", json.dumps(slim, ensure_ascii=False)) \
                 .replace("__HASH__", digest)
    pathlib.Path(args.review).write_text(page)

    st = collections.Counter(r["status"] for r in recs)
    print(f"assembled {len(recs)} records · shipping {len(ship)}")
    print(f"  status         : {dict(st)}")
    print(f"  by arm         : {dict(collections.Counter(r['arm'] for r in ship))}")
    print(f"  by type        : {dict(collections.Counter(r['query_type'] for r in ship))}")
    print(f"  rules covered  : {len({r['rule'] for r in ship})}")
    print(f"  with co-gold   : {sum(1 for r in ship if r.get('also_answered_by'))}")
    print(f"  set sha256     : {digest[:16]}")
    print(f"  wrote {out}")
    print(f"  wrote {args.review}  ({len(sample)} items)")
    print(f"  review strata  : {dict(collections.Counter(s['review_stratum'] for s in sample))}")


if __name__ == "__main__":
    sys.exit(main())
