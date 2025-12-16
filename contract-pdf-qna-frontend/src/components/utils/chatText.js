/**
 * Strip noisy, system-appended transcript context from a displayed question.
 *
 * We only modify text when we detect a `transcribe:` appendix (case-insensitive),
 * which has been observed to look like:
 *   "...real question...\ntranscribe: Hello. state=..."
 */
export function stripTranscribeAppendix(text) {
  const raw = String(text || "");
  const idx = raw.search(/transcribe\s*:/i);
  if (idx === -1) return raw;

  // Prefer cutting on a line boundary if present.
  const before = raw.slice(0, idx);
  // If the appendix is mid-line, this still works fine.
  return before.replace(/\s+$/g, "").trim();
}


