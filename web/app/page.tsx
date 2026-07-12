"use client";
import { useState, useRef } from "react";
import ReactMarkdown from "react-markdown";

type Hit = {
  chunk_id: string; collection: string; title: string; section: string;
  content: string; page_span: string; source_url: string;
};
const COLL: Record<string, string> = {
  "legal-authorities": "Laws (GS 7B + NCAC)",
  "nc-child-welfare": "Policy (NCDHHS)",
};

export default function Home() {
  const [q, setQ] = useState("");
  const [collection, setCollection] = useState("");
  const [wantAnswer, setWantAnswer] = useState(false);
  const [phase, setPhase] = useState<"idle" | "searching" | "answering" | "done">("idle");
  const [hits, setHits] = useState<Hit[] | null>(null);
  const [answer, setAnswer] = useState("");
  const [err, setErr] = useState("");
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const mainRef = useRef<HTMLElement>(null);

  async function run(e: React.FormEvent) {
    e.preventDefault();
    if (!q.trim()) return;
    setErr(""); setAnswer(""); setHits(null); setExpanded({});
    const params = new URLSearchParams({ q, ...(collection ? { collection } : {}) });
    try {
      if (!wantAnswer) {
        setPhase("searching");
        const r = await fetch(`api/search?${params}`);
        const d = await r.json();
        if (d.error) setErr(d.error);
        setHits(d.hits || []);
        setPhase("done");
        return;
      }
      // streaming answer: first line = {hits}, then answer tokens
      setPhase("searching");
      const res = await fetch(`api/answer?${params}`);
      if (!res.body) throw new Error("no stream");
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let pre = "", header = false, ans = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = dec.decode(value, { stream: true });
        if (!header) {
          pre += chunk;
          const nl = pre.indexOf("\n");
          if (nl >= 0) {
            try { setHits(JSON.parse(pre.slice(0, nl)).hits || []); } catch {}
            ans = pre.slice(nl + 1);
            header = true;
            setPhase("answering");
            setAnswer(ans);
          }
        } else {
          ans += chunk;
          setAnswer(ans);
        }
      }
      setPhase("done");
    } catch (e: any) {
      setErr(String(e?.message || e));
      setPhase("done");
    }
  }

  // make [n] citations clickable -> scroll to source card n
  const linkified = answer.replace(/\[(\d+)\]/g, (_m, n) => `[\\[${n}\\]](#src-${n})`);

  return (
    <>
      <div className="banner">PUBLIC North Carolina law &amp; policy only · no confidential case data</div>
      <header>
        <h1>NC Law &amp; Policy Search</h1>
        <p className="sub">Semantic search over NC General Statutes (Ch. 7B), the NC Administrative Code, and NCDHHS child-welfare policy — every result source-traceable.</p>
      </header>
      <form className="search" onSubmit={run}>
        <div className="searchrow">
          <input value={q} onChange={(e) => setQ(e.target.value)} autoFocus
            placeholder="Search — e.g. grounds for termination of parental rights" />
        </div>
        <div className="controls">
          <select value={collection} onChange={(e) => setCollection(e.target.value)}>
            <option value="">All public sources</option>
            <option value="legal-authorities">Laws (GS 7B + NCAC)</option>
            <option value="nc-child-welfare">Policy (NCDHHS)</option>
          </select>
          <label className="chk">
            <input type="checkbox" checked={wantAnswer} onChange={(e) => setWantAnswer(e.target.checked)} />
            grounded answer
          </label>
          <button type="submit" disabled={phase === "searching" || phase === "answering"}>
            {phase === "searching" ? "Searching…" : phase === "answering" ? "Answering…" : "Search"}
          </button>
        </div>
      </form>
      <main ref={mainRef}>
        {err && <div className="answer"><div className="flag">⚠ {err}</div></div>}

        {(phase === "searching") && (
          <div className="status"><span className="spinner" /> Searching statutes &amp; policy…</div>
        )}

        {(phase === "answering" || (phase === "done" && wantAnswer && answer)) && (
          <div className="answer">
            <h3>Grounded answer {phase === "answering" && <span className="spinner sm" />}</h3>
            <div className="body md">
              <ReactMarkdown
                components={{ a: ({ href, children }) => <a href={href} className="cite-link">{children}</a> }}
              >{linkified}</ReactMarkdown>
            </div>
          </div>
        )}

        {hits && <div className="meta">{hits.length} source(s) · hybrid vector + keyword, Jina-reranked</div>}
        {hits && hits.length === 0 && phase === "done" && <div className="empty">No matching public statutes, regulations, or policy.</div>}

        {hits?.map((h, i) => {
          const open = expanded[h.chunk_id];
          const text = open ? h.content : (h.content || "").slice(0, 420);
          const long = (h.content || "").length > 420;
          return (
            <div className="card" id={`src-${i + 1}`} key={h.chunk_id}>
              <div className="cardhead">
                <span className="num">{i + 1}</span>
                <div className="sec">{h.section || h.title || "(untitled)"}</div>
              </div>
              {h.section && h.title && h.section !== h.title && <div className="title">{h.title}</div>}
              <div className="snip">{text}{!open && long ? "…" : ""}</div>
              {long && (
                <button className="morebtn" onClick={() => setExpanded((s) => ({ ...s, [h.chunk_id]: !open }))}>
                  {open ? "Show less ▴" : "Show full passage ▾"}
                </button>
              )}
              <div className="badges">
                <span className="badge coll">{COLL[h.collection] || h.collection}</span>
                {h.page_span && <span className="badge pg">{h.page_span}</span>}
              </div>
              {h.source_url && (
                <div className="cite"><a href={h.source_url} target="_blank" rel="noopener noreferrer">{h.source_url}</a></div>
              )}
            </div>
          );
        })}
        {!hits && phase === "idle" && <div className="empty">Enter a query to search NC law &amp; policy.</div>}
      </main>
      <footer>parsevault · public NC-law engine · Supabase pgvector + Jina rerank · deepseek grounded answers</footer>
    </>
  );
}
