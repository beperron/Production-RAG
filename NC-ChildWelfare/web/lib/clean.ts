// Deterministic cleanup for converted-document text (mirror of the Python
// clean() in scripts/ingest_supabase.py): <br>/tags, table pipes, stray
// emphasis underscores, orphaned bullets, runaway whitespace.
export function clean(s: string): string {
  return (s || "")
    .replace(/[\x00-\x08\x0b\x0c\x0e-\x1f]/g, "")
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/?[a-z][^>]*>/gi, "")
    .replace(/ /g, " ")
    .replace(/\|+/g, " ")
    .replace(/_+/g, " ")
    .replace(/[ \t]+/g, " ")
    .replace(/[ \t]+([.,;:)\]])/g, "$1")
    .replace(/([•·▪‣])[ \t]*\n+[ \t]*/g, "$1 ")
    .replace(/ *\n */g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}
