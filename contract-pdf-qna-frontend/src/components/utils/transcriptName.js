// Utilities for turning system-y transcript filenames into human-friendly labels.

const DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

function titleCaseWord(w) {
  if (!w) return "";
  return w.charAt(0).toUpperCase() + w.slice(1).toLowerCase();
}

function titleCasePhrase(s) {
  return String(s || "")
    .split(/\s+/)
    .filter(Boolean)
    .map(titleCaseWord)
    .join(" ");
}

/**
 * Given a transcript filename like:
 *   cs_call_99651_2024-03-05_SHIELDPLUS_claim-approval.txt
 *
 * Returns:
 *   { primary: "Call 99651 · 2024-03-05", secondary: "Claim approval", raw: "..." }
 */
export function formatTranscriptDisplayName(rawName) {
  const raw = String(rawName || "");
  const base = raw.replace(/\.[^/.]+$/, ""); // strip extension
  const tokens = base.split("_").filter(Boolean);

  // Attempt to parse common pattern
  const dateIdx = tokens.findIndex((t) => DATE_RE.test(t));
  const date = dateIdx >= 0 ? tokens[dateIdx] : "";

  // call id is often right after "call"
  let callId = "";
  const callIdx = tokens.findIndex((t) => t.toLowerCase() === "call");
  if (callIdx >= 0 && tokens[callIdx + 1] && /^\d+$/.test(tokens[callIdx + 1])) {
    callId = tokens[callIdx + 1];
  } else {
    // fallback: first numeric token
    const num = tokens.find((t) => /^\d+$/.test(t));
    callId = num || "";
  }

  // Derive a "reason"/topic from remaining tokens after date/plan-ish tokens.
  // We treat ALLCAPS tokens as plan-ish and skip them (plan already shown in pills).
  const rest = tokens.filter((t, idx) => {
    if (idx === dateIdx) return false;
    if (idx === callIdx) return false;
    if (idx === callIdx + 1 && callId) return false;
    if (/^[A-Z0-9]+$/.test(t) && t.length >= 4) return false;
    if (t.toLowerCase() === "cs") return false;
    if (t.toLowerCase() === "call") return false;
    return true;
  });

  const topicRaw = rest.join(" ").replace(/[-]+/g, " ").replace(/\s+/g, " ").trim();
  const secondary = topicRaw ? titleCasePhrase(topicRaw) : "";

  const primaryParts = [];
  if (callId) primaryParts.push(`Call ${callId}`);
  if (date) primaryParts.push(date);
  const primary = primaryParts.length ? primaryParts.join(" · ") : titleCasePhrase(base.replace(/[_-]+/g, " "));

  return { primary, secondary, raw };
}


