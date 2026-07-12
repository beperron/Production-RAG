import { NextRequest, NextResponse } from "next/server";
import { search } from "@/lib/search";
import { answer } from "@/lib/answer";

export const runtime = "nodejs";
export const maxDuration = 60;

export async function GET(req: NextRequest) {
  const q = (req.nextUrl.searchParams.get("q") || "").trim();
  const collection = req.nextUrl.searchParams.get("collection") || null;
  if (!q) return NextResponse.json({ answer: "", grounded: false, hits: [] });
  try {
    const hits = await search(q, 10, collection);
    const a = await answer(q, hits);
    return NextResponse.json({ answer: a.text, grounded: a.grounded, hits });
  } catch (e: any) {
    return NextResponse.json({ error: String(e?.message || e).slice(0, 200), answer: "", grounded: false, hits: [] }, { status: 500 });
  }
}
