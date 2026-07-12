"use client";
import { useState } from "react";

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
  const [loading, setLoading] = useState(false);
  const [hits, setHits] = useState<Hit[] | null>(null);
  const [answer, setAnswer] = useState<{ text: string; grounded: boolean } | null>(null);
  const [err, setErr] = useState("");

  async function run(e: React.FormEvent) {
    e.preventDefault();
    if (!q.trim()) return;
    setLoading(true); setErr(""); setAnswer(null);
    try {
      const params = new URLSearchParams({ q, ...(collection ? { collection } : {}) });
      const ep = wantAnswer ? "answer" : "search";
      const r = await fetch(`api/${ep}?${params}`);
      const d = await r.json();
      if (d.error) setErr(d.error);
      setHits(d.hits || []);
      if (wantAnswer) setAnswer({ text: d.answer || "", grounded: !!d.grounded });
    } catch (e: any) {
      setErr(String(e?.message || e));
    } finally {
      setLoading(false);
    }
  }

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
          <button type="submit" disabled={loading}>{loading ? "Searching…" : "Search"}</button>
        </div>
      </form>
      <main>
        {err && <div className="answer"><div className="flag">⚠ {err}</div></div>}
        {answer && (
          <div className="answer">
            <h3>Grounded answer</h3>
            <div className="body">{answer.text}</div>
            {!answer.grounded && <div className="flag">⚠ not grounded — showing sources below</div>}
          </div>
        )}
        {hits && <div className="meta">{hits.length} result(s) · hybrid vector + keyword, Jina-reranked</div>}
        {hits && hits.length === 0 && !loading && <div className="empty">No matching public statutes, regulations, or policy.</div>}
        {hits?.map((h) => (
          <div className="card" key={h.chunk_id}>
            <div className="sec">{h.section || h.title || "(untitled)"}</div>
            {h.section && h.title && h.section !== h.title && <div className="title">{h.title}</div>}
            <div className="snip">{(h.content || "").slice(0, 420)}</div>
            <div className="badges">
              <span className="badge coll">{COLL[h.collection] || h.collection}</span>
              {h.page_span && <span className="badge pg">{h.page_span}</span>}
            </div>
            {h.source_url && (
              <div className="cite"><a href={h.source_url} target="_blank" rel="noopener noreferrer">{h.source_url}</a></div>
            )}
          </div>
        ))}
        {!hits && <div className="empty">Enter a query to search NC law &amp; policy.</div>}
      </main>
      <footer>parsevault · public NC-law engine · Supabase pgvector + Jina rerank · deepseek grounded answers</footer>
    </>
  );
}
