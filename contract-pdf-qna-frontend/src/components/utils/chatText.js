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

const _norm = (v) => String(v ?? "").replace(/\s+/g, " ").trim();

const _isNotProvided = (v) => {
  const s = _norm(v).toLowerCase();
  return !s || s === "not provided" || s === "n/a" || s === "na" || s === "none";
};

const _prettifyKey = (k) => {
  const raw = _norm(k)
    .replace(/^\W+/, "")
    .replace(/\W+$/g, "")
    .replace(/[_-]+/g, " ");
  if (!raw) return "";
  // Keep common abbreviations readable.
  const lower = raw.toLowerCase();
  const map = {
    auth_number: "Auth #",
    authorized_total: "Authorized total",
    authorized_scope: "Authorized scope",
    outcome_signals: "Outcome signals",
    eligibility_signals: "Eligibility signals",
    requested_service: "Requested service",
    issue: "Issue",
    timing: "Timing",
    contractor_estimate_total_or_lumpsum: "Contractor estimate total",
  };
  if (map[lower]) return map[lower];
  // Title-case-ish for generic keys.
  return raw
    .split(" ")
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
};

/**
 * Parse the "extracted question" format that sometimes includes a leading
 * semicolon-separated key=value appendix, e.g.:
 *   "; issue=...; authorized_total=$90; ...] Please reconcile ... "
 *
 * Returns:
 *  - questionText: user-friendly question sentence
 *  - facts: [{ key, label, value }]
 */
export function parseExtractedQuestion(text) {
  const raw0 = stripTranscribeAppendix(text);
  const raw = String(raw0 ?? "").replace(/\r\n/g, "\n").trim();
  if (!raw) return { questionText: "", facts: [] };

  // Heuristic 1: if there's a trailing bracket before the real question, cut on the last `]`.
  let prefix = raw;
  let questionText = raw;
  const lastBracket = raw.lastIndexOf("]");
  if (lastBracket !== -1 && lastBracket < raw.length - 1) {
    const after = raw.slice(lastBracket + 1).trim();
    const before = raw.slice(0, lastBracket + 1).trim();
    // Only apply if the "after" looks like a real sentence.
    if (after && /[A-Z]/.test(after[0])) {
      prefix = before;
      questionText = after;
    }
  }

  // If the text starts with lots of `k=v; k=v; ...`, treat that as prefix too.
  // (Keep questionText as-is unless we can find a clean boundary.)
  const facts = [];
  try {
    const parts = prefix
      .split(";")
      .map((p) => p.trim())
      .filter(Boolean);
    for (const p of parts) {
      const eq = p.indexOf("=");
      if (eq === -1) continue;
      const key = _norm(p.slice(0, eq));
      let value = _norm(p.slice(eq + 1));

      // Clean up stray trailing bracket artifacts in values.
      value = value.replace(/\]$/g, "").trim();

      // Special-case auth_number that sometimes gets polluted with extra tokens.
      if (key.toLowerCase() === "auth_number") {
        const m = value.match(/^\d+/);
        if (m) value = m[0];
      }

      if (_isNotProvided(value)) continue;
      const label = _prettifyKey(key);
      if (!label) continue;
      facts.push({ key, label, value });
    }
  } catch {
    // ignore parsing errors; fall back to plain text
  }

  // Final cleanup: remove any leading punctuation.
  questionText = questionText.replace(/^[\s;:\-\]]+/, "").trim();
  return { questionText, facts };
}


