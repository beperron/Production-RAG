import { NextRequest, NextResponse } from "next/server";
import { search } from "@/lib/search";

export const runtime = "nodejs";
export const maxDuration = 30;

export async function GET(req: NextRequest) {
  const q = (req.nextUrl.searchParams.get("q") || "").trim();
  const collection = req.nextUrl.searchParams.get("collection") || null;
  const k = Math.min(20, parseInt(req.nextUrl.searchParams.get("k") || "10", 10) || 10);
  if (!q) return NextResponse.json({ hits: [] });
  try {
    const hits = await search(q, k, collection);
    return NextResponse.json({ hits });
  } catch (e: any) {
    return NextResponse.json({ error: String(e?.message || e).slice(0, 200), hits: [] }, { status: 500 });
  }
}
