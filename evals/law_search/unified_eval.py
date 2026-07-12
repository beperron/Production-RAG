"""Unified law+policy retrieval eval on the expanded gold sets.

Each gold entry has an `accept` list of regex patterns; a hit counts if ANY
pattern matches the hit's title+heading_path+section. Reports R@1/R@3/R@5/MRR
for RRF-only and RRF+Jina-blend (the shipped config), per corpus.
Counts only — never prints document text.
"""
import sys, os, json, re
from pathlib import Path
for line in (Path.home()/".config"/"parsevault"/"lawsearch.env").read_text().splitlines():
    line=line.strip()
    if line and not line.startswith("#") and "=" in line:
        k,v=line.split("=",1); os.environ[k.strip()]=v.strip()
sys.path.insert(0,'src')
from parsevault.pipeline.docindex import DocIndex
from parsevault.config import embedder_from_env
from parsevault.lawsearch import _jina_reranker

emb=embedder_from_env(); rr=_jina_reranker()
def htext(h): return f"{getattr(h,'title','')} {' '.join(getattr(h,'heading_path',[]) or [])} {(h.statutory_section() or getattr(h,'section','') or '')}"
def matches(h, pats):
    t=htext(h); return any(re.search(p,t,re.I) for p in pats)
def blend(q,h,a=0.5):
    docs=[(getattr(x,'chunk',None) and getattr(x.chunk,'text','')) or x.snippet for x in h]
    js=rr.scores(q,docs); jmax=max(js) or 1.0; jn=[s/jmax for s in js]
    comb=[(a*jn[i]+(1-a)*(1.0/(i+1)),h[i]) for i in range(len(h))]
    return [x for _,x in sorted(comb,key=lambda t:-t[0])]

def evalset(name, index_path, gold_file):
    idx=DocIndex.load(index_path, embedder=emb)
    gold=json.load(open(os.path.join(os.path.dirname(__file__),gold_file)))
    print(f"\n=== {name} ({len(gold)} gold queries) ===", flush=True)
    for label,use in [("RRF only",False),("RRF + Jina blend",True)]:
        ranks=[]; miss=[]
        for g in gold:
            pats=g.get("accept") or [g["expect"]]
            hits=idx.search(g["q"],k=40,dedup=False)
            if use: hits=blend(g["q"],hits)
            hits=hits[:10]
            r=next((i+1 for i,h in enumerate(hits) if matches(h,pats)),0)
            ranks.append(r)
            if not(1<=r<=3): miss.append(g["q"][:40])
        n=len(ranks); r1=sum(x==1 for x in ranks); r3=sum(1<=x<=3 for x in ranks); r5=sum(1<=x<=5 for x in ranks)
        mrr=sum((1/x if x else 0) for x in ranks)/n
        print(f"  {label:20s} R@1={100*r1//n:3d}%  R@3={100*r3//n:3d}%  R@5={100*r5//n:3d}%  MRR={mrr:.3f}", flush=True)
        if miss: print(f"    miss@3: {miss}", flush=True)

evalset("STATUTES (legal-authorities)","knowledge-base/legal-authorities/docindex.json","law_gold.json")
evalset("POLICY (nc-child-welfare)","knowledge-base/nc-child-welfare/docindex.json","policy_gold.json")
print("\n[unified-eval] COMPLETE", flush=True)
