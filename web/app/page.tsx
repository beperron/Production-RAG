"use client";
import { useState } from "react";
import ReactMarkdown from "react-markdown";

type Hit = {
  chunk_id: string; collection: string; title: string; section: string;
  content: string; page_span: string; source_url: string;
};
const COLL: Record<string, string> = {
  "legal-authorities": "Laws (GS 7B + NCAC)",
  "nc-child-welfare": "Policy (NCDHHS)",
};
const SAMPLES = [
  "Grounds for terminating parental rights",
  "What are reasonable efforts to prevent removal?",
  "How often must a caseworker visit a child in in-home services?",
  "Reasonable and prudent parent standard for foster youth",
  "Plan of safe care for a substance-exposed infant",
  "Timelines for permanency planning hearings",
];

export default function Home() {
  const [q, setQ] = useState("");
  const [collection, setCollection] = useState("");
  const [mode, setMode] = useState<"answer" | "sources">("answer"); // grounded is default
  const [phase, setPhase] = useState<"idle" | "searching" | "answering" | "done">("idle");
  const [hits, setHits] = useState<Hit[] | null>(null);
  const [answer, setAnswer] = useState("");
  const [err, setErr] = useState("");
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  async function run(query: string) {
    if (!query.trim()) return;
    setQ(query);
    setErr(""); setAnswer(""); setHits(null); setExpanded({});
    const params = new URLSearchParams({ q: query, ...(collection ? { collection } : {}) });
    try {
      if (mode === "sources") {
        setPhase("searching");
        const r = await fetch(`api/search?${params}`);
        const d = await r.json();
        if (d.error) setErr(d.error);
        setHits(d.hits || []);
        setPhase("done");
        return;
      }
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
            ans = pre.slice(nl + 1); header = true; setPhase("answering"); setAnswer(ans);
          }
        } else { ans += chunk; setAnswer(ans); }
      }
      setPhase("done");
    } catch (e: any) { setErr(String(e?.message || e)); setPhase("done"); }
  }

  const busy = phase === "searching" || phase === "answering";
  const linkified = answer.replace(/\[(\d+)\]/g, (_m, n) => `[\\[${n}\\]](#src-${n})`);

  return (
    <>
      <div className="topbar">
        <a className="brand" href="https://parallel42.ai" target="_blank" rel="noopener">
          <span className="mark">P42</span>
          <b>Parallel42</b><span>· NC Law &amp; Policy Search</span>
        </a>
        <a className="home" href="https://parallel42.ai" target="_blank" rel="noopener">parallel42.ai ↗</a>
      </div>

      <header className="hero">
        <p className="eyebrow">Public NC child-welfare law &amp; policy</p>
        <h1>Ask North Carolina&apos;s child-welfare rulebook.</h1>
        <p className="lede">
          A search tool over North Carolina&apos;s <b>public</b> child-welfare authority — the
          General Statutes (Chapter 7B, the Juvenile Code), the NC Administrative Code, and the
          NCDHHS policy manuals. It finds the governing statute or policy for a plain-English
          question and can write a short answer <b>grounded only in those sources, with citations</b>
          &nbsp;— or decline when the sources don&apos;t support one.
        </p>
        <div className="facts">
          <span><b>530</b> public documents</span>
          <span><b>19,449</b> indexed passages</span>
          <span><b>Every result</b> source-traceable</span>
          <span>No confidential data</span>
        </div>
      </header>

      <form className="search" onSubmit={(e) => { e.preventDefault(); run(q); }}>
        <div className="searchrow">
          <input value={q} onChange={(e) => setQ(e.target.value)} autoFocus
            placeholder="Ask about NC statutes, regulations, or child-welfare policy…" />
        </div>
        <div className="controls">
          <div className="seg" role="tablist">
            <button type="button" className={mode === "answer" ? "on" : ""} onClick={() => setMode("answer")}>Grounded answer</button>
            <button type="button" className={mode === "sources" ? "on" : ""} onClick={() => setMode("sources")}>Sources only</button>
          </div>
          <span className="info">?
            <span className="tip">
              <b>Grounded answer</b> — writes a short answer using only the retrieved passages, with
              inline [n] citations, and says so if the sources don&apos;t support an answer.<br /><br />
              <b>Sources only</b> — returns the matching statutes and policy passages with citations,
              and no generated text.
            </span>
          </span>
          <select value={collection} onChange={(e) => setCollection(e.target.value)}>
            <option value="">All public sources</option>
            <option value="legal-authorities">Laws (GS 7B + NCAC)</option>
            <option value="nc-child-welfare">Policy (NCDHHS)</option>
          </select>
          <button type="submit" className="go" disabled={busy}>
            {phase === "searching" ? "Searching…" : phase === "answering" ? "Answering…" : "Search"}
          </button>
        </div>
      </form>

      {phase === "idle" && (
        <div className="samples">
          <div className="lbl">Try an example</div>
          <div className="chips">
            {SAMPLES.map((s) => <button className="chip" key={s} onClick={() => run(s)}>{s}</button>)}
          </div>
        </div>
      )}

      <main>
        {err && <div className="answer"><div className="flag">⚠ {err}</div></div>}
        {phase === "searching" && <div className="status"><span className="spinner" /> Searching statutes &amp; policy…</div>}

        {(phase === "answering" || (phase === "done" && mode === "answer" && answer)) && (
          <div className="answer">
            <h3>Grounded answer {phase === "answering" && <span className="spinner sm" />}</h3>
            <div className="body md">
              <ReactMarkdown components={{ a: ({ href, children }) => <a href={href} className="cite-link">{children}</a> }}>
                {linkified}
              </ReactMarkdown>
            </div>
          </div>
        )}

        {hits && <div className="meta">{hits.length} source(s) · hybrid vector + keyword, Jina-reranked</div>}
        {hits && hits.length === 0 && phase === "done" && <div className="empty">No matching public statutes, regulations, or policy.</div>}

        {hits?.map((h, i) => {
          const open = expanded[h.chunk_id];
          const long = (h.content || "").length > 420;
          return (
            <div className="card" id={`src-${i + 1}`} key={h.chunk_id}>
              <div className="cardhead"><span className="num">{i + 1}</span>
                <div className="sec">{h.section || h.title || "(untitled)"}</div></div>
              {h.section && h.title && h.section !== h.title && <div className="title">{h.title}</div>}
              <div className="snip">{open ? h.content : (h.content || "").slice(0, 420)}{!open && long ? "…" : ""}</div>
              {long && <button className="morebtn" onClick={() => setExpanded((s) => ({ ...s, [h.chunk_id]: !open }))}>
                {open ? "Show less ▴" : "Show full passage ▾"}</button>}
              <div className="badges">
                <span className="badge coll">{COLL[h.collection] || h.collection}</span>
                {h.page_span && <span className="badge pg">{h.page_span}</span>}
              </div>
              {h.source_url && <div className="cite"><a href={h.source_url} target="_blank" rel="noopener noreferrer">{h.source_url}</a></div>}
            </div>
          );
        })}
      </main>

      <footer>
        Built by <a href="https://parallel42.ai" target="_blank" rel="noopener">Parallel42</a> · public NC law &amp; policy ·
        Supabase pgvector + Jina rerank · grounded answers cite their sources and decline when unsupported.
        Not legal advice.
      </footer>
    </>
  );
}
