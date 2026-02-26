import React from "react";
import { StructuredCaseText } from "../structuredCaseText/structuredCaseText";
import "./itemizedFinalAnswer.scss";

const ITEM_START_RE = /^Item\s*#?\s*:?\s*(\d+)\b/i;
const ITEM_START_WITH_TITLE_RE = /^Item\s*#?\s*(\d+)\s*:\s*(.+)$/i;

const stripMdDecorators = (s) => {
  let t = String(s || "").trim();
  // Strip markdown heading / quote prefixes like "### " or "> "
  t = t.replace(/^#{1,6}\s*/g, "").replace(/^>\s*/g, "").trim();
  // Strip bold wrappers like "**Item: 1**" or "**Overall Next Steps:**"
  t = t.replace(/^\*\*/, "").replace(/\*\*$/, "").trim();
  return t;
};

const normalizeDecision = (s) => {
  const raw = String(s || "").trim().toUpperCase();
  if (!raw || raw === "—" || raw === "-" || raw === "N/A" || raw === "NA" || raw === "NONE") {
    return "NO_DECISION";
  }
  if (raw.includes("NEED") || raw.includes("INFO")) return "NEED_INFO";
  if (raw.includes("PARTIAL")) return "PARTIAL";
  if (raw.includes("APPROV") || raw.includes("ACCEPT")) return "APPROVED";
  if (raw.includes("REJECT") || raw.includes("DENY") || raw.includes("DENIED")) return "REJECTED";
  if (raw.includes("PENDING") || raw.includes("UNDECIDED") || raw.includes("UNDETERMINED")) {
    return "NO_DECISION";
  }
  return raw;
};

const parseItemSections = (text) => {
  const raw = String(text || "").replace(/\r\n/g, "\n");
  const lines = raw.split("\n");
  const starts = [];
  let isSingleItemFallback = false;

  // Heuristic: Some LLM outputs contain a single item without numbering, e.g.
  // "Item: Unknown" + bullet key/value lines. In that case, treat the whole text as one item.
  const looksLikeSingleItem = () => {
    const keyRe =
      /^(Item|Type|Related|Situation|Decision|Amount|Amounts|Why|Next steps?|What.?s covered|What.?s not covered|Limitations\s*\/\s*not covered)\s*:/i;
    let hits = 0;
    for (const ln of lines) {
      const probe = stripMdDecorators(ln).replace(/^[-•*]\s+/, "").trim();
      if (keyRe.test(probe)) hits += 1;
      if (hits >= 2) return true;
    }
    return false;
  };

  for (let i = 0; i < lines.length; i++) {
    const probe = stripMdDecorators(lines[i]);
    if (ITEM_START_RE.test(probe)) starts.push(i);
  }
  if (starts.length === 0) {
    if (looksLikeSingleItem()) {
      starts.push(0);
      isSingleItemFallback = true;
    } else {
      return { items: [], overall: "" };
    }
  }

  const sections = [];
  for (let i = 0; i < starts.length; i++) {
    const start = starts[i];
    const end = i + 1 < starts.length ? starts[i + 1] : lines.length;
    sections.push(lines.slice(start, end));
  }

  const items = sections.map((secLines) => {
    const item = {
      itemNo: isSingleItemFallback ? "1" : "",
      title: "",
      name: "",
      type: "",
      related: "",
      situation: "",
      decision: "",
      amount: "",
      covered: [],
      notCovered: [],
      amountsCustomer: [],
      amountsCompany: [],
      why: [],
      nextSteps: [],
      raw: secLines.join("\n").trim(),
    };

    let mode = "";
    for (const ln of secLines) {
      // Detect indentation level BEFORE stripping markdown (for nested bullets)
      const indentMatch = ln.match(/^(\s*)/);
      const currentIndent = indentMatch ? indentMatch[1].length : 0;
      const isNestedBullet = currentIndent > 2 && /^\s+[-•*]\s+/.test(ln);
      
      let t = stripMdDecorators(ln);
      if (!t) continue;

      const isBullet = /^[-•*]\s+/.test(t);
      const bulletText = isBullet ? t.replace(/^[-•*]\s+/, "").trim() : t;
      const base = bulletText;

      const mStartWithTitle = base.match(ITEM_START_WITH_TITLE_RE);
      if (mStartWithTitle) {
        item.itemNo = mStartWithTitle[1];
        item.title = (mStartWithTitle[2] || "").trim();
        mode = "";
        continue;
      }

      const mStart = base.match(ITEM_START_RE);
      if (mStart) {
        item.itemNo = mStart[1];
        mode = "";
        continue;
      }
      const kv = (label) => {
        // Try exact match first
        let re = new RegExp(`^${label}\\s*:\\s*(.+)$`, "i");
        let m = base.match(re);
        if (m) return m[1].trim();
        
        // Try with optional colon and whitespace variations
        re = new RegExp(`^${label}\\s*:?\\s*(.+)$`, "i");
        m = base.match(re);
        if (m) return m[1].trim();
        
        // Try case-insensitive partial match for dynamic fields
        if (base.toLowerCase().startsWith(label.toLowerCase())) {
          const afterLabel = base.slice(label.length).replace(/^[\s:]+/, "").trim();
          if (afterLabel) return afterLabel;
        }
        
        return "";
      };

      const itemName = kv("Item");
      if (itemName) {
        item.name = itemName.trim();
        mode = "";
        continue;
      }
      const type = kv("Type");
      if (type) {
        item.type = type.trim();
        mode = "";
        continue;
      }
      const related = kv("Related");
      if (related) {
        const trimmedRelated = related.trim();
        // Skip placeholder values like "None specified", "N/A", etc.
        if (trimmedRelated && !/^(none|n\/a|na|not specified|—|-)$/i.test(trimmedRelated)) {
          item.related = trimmedRelated;
        }
        mode = "";
        continue;
      }
      const situation = kv("Situation");
      if (situation) {
        item.situation = situation.trim();
        mode = "";
        continue;
      }
      const decision = kv("Decision");
      if (decision) {
        const trimmedDecision = decision.trim();
        // Only set decision if it's not empty and not a placeholder
        if (trimmedDecision && !/^[-—n\/a]+$/i.test(trimmedDecision)) {
          item.decision = trimmedDecision;
        }
        mode = "";
        continue;
      }
      // "Amount" (key fact) must not match "Amounts:" — use strict regex
      const amountMatch = base.match(/^Amount\s*:\s*(.*)$/i);
      if (amountMatch) {
        item.amount = (amountMatch[1] || "").trim();
        mode = "";
        continue;
      }

      // Inline "What's covered: None" / "Why: ..." / etc (common in backend markdown)
      const coveredInline = base.match(/^What.?s covered\s*:\s*(.*)$/i);
      if (coveredInline) {
        const v = String(coveredInline[1] || "").trim();
        if (v && !/^none$/i.test(v)) item.covered.push(v);
        mode = "";
        continue;
      }
      const notCoveredInline = base.match(
        /^What.?s not covered(?:\s*\/\s*limitations)?\s*:\s*(.*)$/i
      );
      if (notCoveredInline) {
        const v = String(notCoveredInline[1] || "").trim();
        if (v) {
          if (!/^none$/i.test(v)) item.notCovered.push(v);
          mode = "";
        } else {
          mode = "notCovered";
        }
        continue;
      }
      // Handle "Limitations / not covered:" pattern
      const limitationsInline = base.match(
        /^Limitations\s*\/\s*not covered\s*:\s*(.*)$/i
      );
      if (limitationsInline) {
        const v = String(limitationsInline[1] || "").trim();
        if (v) {
          // Allow "None specified" to be displayed
          if (!/^none$/i.test(v)) item.notCovered.push(v);
          mode = "";
        } else {
          mode = "notCovered";
        }
        continue;
      }
      const amountsInline = base.match(/^Amounts\s*:\s*(.*)$/i);
      if (amountsInline) {
        const v = String(amountsInline[1] || "").trim();
        if (!v || /^none$/i.test(v)) {
          // No inline value, expect nested bullets
          mode = "amounts";
        } else {
          // If backend sends a value inline, treat it as a customer line by default.
          item.amountsCustomer.push(v);
          mode = "amounts";
        }
        continue;
      }
      const whyInline = base.match(/^Why\s*:\s*(.*)$/i);
      if (whyInline) {
        const v = String(whyInline[1] || "").trim();
        if (v && !/^none$/i.test(v)) item.why.push(v);
        mode = "";
        continue;
      }
      const nextInline = base.match(/^Next steps?\s*:\s*(.*)$/i);
      if (nextInline) {
        const v = String(nextInline[1] || "").trim();
        if (v && !/^none$/i.test(v)) item.nextSteps.push(v);
        mode = "";
        continue;
      }

      // Allow "Overall Next Steps" or "Overall Next Step" (singular) to be treated as next steps content
      if (/^Overall Next Steps?\b/i.test(base.replace(/:$/, ""))) {
        mode = "overallNext";
        continue;
      }

      if (/^What.?s covered\b/i.test(base)) {
        mode = "covered";
        continue;
      }
      if (/^What.?s not covered\b/i.test(base)) {
        mode = "notCovered";
        continue;
      }
      if (/^Limitations\s*\/\s*not covered\b/i.test(base)) {
        mode = "notCovered";
        continue;
      }
      if (/^Amounts\b/i.test(base)) {
        mode = "amounts";
        continue;
      }
      if (/^Why\b/i.test(base)) {
        mode = "why";
        continue;
      }
      if (/^Next steps?\b/i.test(base)) {
        mode = "nextSteps";
        continue;
      }

      if (mode === "covered") {
        const trimmed = bulletText.trim();
        if (trimmed && !/^none$/i.test(trimmed)) item.covered.push(trimmed);
      } else if (mode === "notCovered") {
        // Handle nested bullets under "What's not covered"
        if (isNestedBullet) {
          const trimmed = bulletText.trim();
          if (trimmed && !/^none$/i.test(trimmed)) item.notCovered.push(trimmed);
        } else {
          // Regular bullet under notCovered section
          const trimmed = bulletText.trim();
          if (trimmed && !/^none$/i.test(trimmed)) item.notCovered.push(trimmed);
        }
      } else if (mode === "why") {
        const trimmed = bulletText.trim();
        if (trimmed && !/^none$/i.test(trimmed)) item.why.push(trimmed);
      } else if (mode === "nextSteps") {
        const trimmed = bulletText.trim();
        if (trimmed && !/^none$/i.test(trimmed)) item.nextSteps.push(trimmed);
      } else if (mode === "amounts") {
        // Handle nested bullets under "Amounts"
        if (isNestedBullet) {
          const trimmed = bulletText.trim();
          if (trimmed && !/^none$/i.test(trimmed)) {
            // Check if it's a customer or company line
            if (/customer|quoted|asked/i.test(trimmed)) {
              item.amountsCustomer.push(trimmed.replace(/^(customer\s*(quoted\/asked)?\s*:?\s*)/i, "").trim());
            } else if (/company|we can|can provide/i.test(trimmed)) {
              item.amountsCompany.push(trimmed.replace(/^(company\s*(can\s*provide)?\s*:?\s*)/i, "").trim());
            } else {
              // Default to customer if unclear
              item.amountsCustomer.push(trimmed);
            }
          }
        } else {
          // Regular bullet under amounts section
          const trimmed = bulletText.trim();
          if (trimmed && !/^none$/i.test(trimmed)) {
            if (/customer|quoted|asked/i.test(trimmed)) {
              item.amountsCustomer.push(trimmed.replace(/^(customer\s*(quoted\/asked)?\s*:?\s*)/i, "").trim());
            } else if (/company|we can|can provide/i.test(trimmed)) {
              item.amountsCompany.push(trimmed.replace(/^(company\s*(can\s*provide)?\s*:?\s*)/i, "").trim());
            } else {
              item.amountsCustomer.push(trimmed);
            }
          }
        }
      } else if (mode === "overallNext") {
        const trimmed = bulletText.trim();
        if (trimmed && !/^none$/i.test(trimmed)) item.nextSteps.push(trimmed);
      }
    }

    return item;
  });

  // Everything before the first item section becomes overall preface (optional).
  const overall = lines.slice(0, starts[0]).join("\n").trim();

  // Best-effort: split out an "Overall Next Steps" or "Overall Next Step" section (if present).
  let overallNextSteps = "";
  try {
    const tailIdx = lines.findIndex((l) =>
      /^Overall Next Steps?\b/i.test(stripMdDecorators(l).replace(/:$/, ""))
    );
    if (tailIdx >= 0) {
      overallNextSteps = lines.slice(tailIdx).join("\n").trim();
    }
  } catch (e) {
    overallNextSteps = "";
  }

  return { items, overall, overallNextSteps };
};

/** Exported for popup edit mode: parse draft text into { items, overall, overallNextSteps }. */
export const parseDraftSummary = (text) => parseItemSections(text || "");

/**
 * Serialize parsed structure back to draft summary text (same style the parser expects).
 * Round-trip: parseDraftSummary(serializeDraftSummary(parsed)) should yield equivalent structure.
 */
export const serializeDraftSummary = (parsed) => {
  if (!parsed || !Array.isArray(parsed.items)) return "";
  const lines = [];
  const str = (v) => (v == null || v === undefined ? "" : String(v).trim());
  const arrJoin = (arr) => (Array.isArray(arr) && arr.length ? arr.join("\n") : "");

  for (let i = 0; i < parsed.items.length; i++) {
    const it = parsed.items[i];
    const itemNo = it.itemNo || String(i + 1);
    const title = str(it.title || it.name) || `Item ${i + 1}`;
    lines.push(`Item #${itemNo}: ${title}`);
    if (str(it.type)) lines.push(`Type: ${it.type}`);
    if (str(it.related)) lines.push(`Related: ${it.related}`);
    if (str(it.situation)) lines.push(`Situation: ${it.situation}`);
    if (str(it.decision)) lines.push(`Decision: ${it.decision}`);
    if (str(it.amount)) lines.push(`Amount: ${it.amount}`);
    if (it.covered?.length) {
      lines.push("What's covered:");
      it.covered.forEach((c) => lines.push(`- ${str(c)}`));
    }
    if (it.notCovered?.length) {
      lines.push("Limitations / not covered:");
      it.notCovered.forEach((c) => lines.push(`- ${str(c)}`));
    }
    if (it.amountsCustomer?.length || it.amountsCompany?.length) {
      lines.push("Amounts:");
      (it.amountsCustomer || []).forEach((c) => lines.push(`- Customer: ${str(c)}`));
      (it.amountsCompany || []).forEach((c) => lines.push(`- Company: ${str(c)}`));
    }
    if (it.why?.length) {
      lines.push("Why:");
      it.why.forEach((w) => lines.push(`- ${str(w)}`));
    }
    if (it.nextSteps?.length) {
      lines.push("Next steps:");
      it.nextSteps.forEach((n) => lines.push(`- ${str(n)}`));
    }
    lines.push("");
  }
  if (str(parsed.overallNextSteps)) {
    lines.push("Overall Next Step");
    lines.push(str(parsed.overallNextSteps));
  }
  return lines.join("\n").trim();
};

/**
 * Build a list of only the summary fields that changed between previous and updated draft text.
 * Returns [{ fieldName, previousValue, updatedValue }] for display in the change log.
 * Only fields with different previous vs current values are included.
 */
export const buildSummaryFieldChanges = (previousText, updatedText) => {
  const prev = parseItemSections(previousText || "");
  const next = parseItemSections(updatedText || "");
  const changes = [];
  const prevItems = prev.items || [];
  const nextItems = next.items || [];

  const str = (v) => (v == null || v === undefined ? "" : String(v).trim());
  const arrStr = (arr) => (Array.isArray(arr) && arr.length ? arr.join("; ") : "");
  const emptyLabel = (s) => (s ? s : "Not specified");

  const maxItems = Math.max(prevItems.length, nextItems.length, 1);
  for (let i = 0; i < maxItems; i++) {
    const prefix = maxItems > 1 ? `Item ${i + 1} - ` : "";
    const prevItem = prevItems[i] || {};
    const nextItem = nextItems[i] || {};

    const fields = [
      { key: "Item", prev: prevItem.title || prevItem.name || "", next: nextItem.title || nextItem.name || "" },
      { key: "Type", prev: str(prevItem.type), next: str(nextItem.type) },
      { key: "Related", prev: str(prevItem.related), next: str(nextItem.related) },
      { key: "Situation", prev: str(prevItem.situation), next: str(nextItem.situation) },
      { key: "Decision", prev: str(prevItem.decision), next: str(nextItem.decision) },
      { key: "Amount", prev: str(prevItem.amount), next: str(nextItem.amount) },
      { key: "What's Covered", prev: arrStr(prevItem.covered), next: arrStr(nextItem.covered) },
      {
        key: "What's Not Covered / Limitations",
        prev: arrStr(prevItem.notCovered),
        next: arrStr(nextItem.notCovered),
      },
      {
        key: "Customer Quoted / Asked",
        prev: arrStr(prevItem.amountsCustomer),
        next: arrStr(nextItem.amountsCustomer),
      },
      {
        key: "Company Can Provide",
        prev: arrStr(prevItem.amountsCompany),
        next: arrStr(nextItem.amountsCompany),
      },
      { key: "Why", prev: arrStr(prevItem.why), next: arrStr(nextItem.why) },
      {
        key: "Next Steps (Item Level)",
        prev: arrStr(prevItem.nextSteps),
        next: arrStr(nextItem.nextSteps),
      },
    ];

    for (const f of fields) {
      const p = str(f.prev);
      const n = str(f.next);
      if (p !== n) {
        changes.push({
          fieldName: prefix + f.key,
          previousValue: emptyLabel(p) || "—",
          updatedValue: emptyLabel(n) || "—",
        });
      }
    }
  }

  // Overall Next Step (only if changed)
  const prevOverall = str(prev.overallNextSteps || "");
  const nextOverall = str(next.overallNextSteps || "");
  if (prevOverall !== nextOverall) {
    changes.push({
      fieldName: "Overall Next Step",
      previousValue: emptyLabel(prevOverall) || "—",
      updatedValue: emptyLabel(nextOverall) || "—",
    });
  }

  // If parsing produced no items, treat the whole text as one field (fallback).
  if (changes.length === 0 && prevItems.length === 0 && nextItems.length === 0 && str(previousText) !== str(updatedText)) {
    return [
      {
        fieldName: "Summary",
        previousValue: str(previousText) || "—",
        updatedValue: str(updatedText) || "—",
      },
    ];
  }
  return changes;
};

const DecisionBadge = ({ decision }) => {
  const norm = normalizeDecision(decision);
  // Same color logic as edit mode: approved=green, denied=red, no_decision/other=grey or existing style
  const cls = norm.toLowerCase().replace(/[^a-z0-9]+/g, "_");
  const toneClass =
    norm === "APPROVED"
      ? "ifa_badge_approved"
      : norm === "REJECTED"
        ? "ifa_badge_denied"
        : `ifa_badge_${cls}`;
  const displayText =
    norm === "NO_DECISION" ? "No Decision" : norm === "REJECTED" ? "Denied" : norm.replace(/_/g, " ");
  return <span className={`ifa_badge ${toneClass}`}>{displayText}</span>;
};

const cleanAmountLine = (s) => {
  const raw = String(s || "").trim();
  if (!raw) return "";
  // Remove redundant prefixes since we already render Customer/Company headings.
  return raw
    .replace(/^Customer\s*(quoted\/asked)?\s*:\s*/i, "")
    .replace(/^Company\s*(can\s*provide)?\s*:\s*/i, "")
    .trim();
};

export const ItemizedFinalAnswer = ({ text = "", title = "Final Answer", asCard = true }) => {
  const raw = String(text || "");
  if (!raw.trim()) return null;

  const parsed = parseItemSections(raw);
  const hasItems = Array.isArray(parsed.items) && parsed.items.length > 0;

  if (!hasItems) {
    return (
      <div className={`itemized_final_answer ${asCard ? "ifa_outer_card" : ""}`}>
        <div className="ifa_title">{title}</div>
        <StructuredCaseText text={raw} />
      </div>
    );
  }

  return (
    <div className={`itemized_final_answer ${asCard ? "ifa_outer_card" : ""}`}>
      <div className="ifa_title">{title}</div>
      {parsed.overall ? (
        <div className="ifa_overall">
          <StructuredCaseText text={parsed.overall} />
        </div>
      ) : null}

      <div className="ifa_cards">
        {parsed.items.map((it, idx) => {
          // Prefer the appliance name from "Item #1: Water Heater" header; fall back to detailed Item line.
          const applianceName = (it.title || "").trim() || (it.name || "").trim() || `Item ${idx + 1}`;
          const itemNo = it.itemNo || String(idx + 1);
          // Handle decision: empty string, null, undefined, or "—" all mean no decision
          const decision = (it.decision || "").trim() || "";
          const hasAmounts = Boolean(it.amountsCustomer?.length || it.amountsCompany?.length);

          // Key-fact "Amount" is independent of Amounts (Customer/Company); use item.amount when set, else fallback.
          const topAmount =
            (it.amount && String(it.amount).trim()) ||
            (it.amountsCompany?.[0] || "").replace(/^Company\s*(can\s*provide)?\s*:\s*/i, "").trim() ||
            (it.amountsCustomer?.[0] || "").replace(/^Customer\s*(quoted\/asked)?\s*:\s*/i, "").trim() ||
            (hasAmounts ? "See Amounts below" : "Not applicable");
          return (
            <details className="ifa_item" key={`${itemNo}-${idx}`} open={idx === 0}>
              <summary className="ifa_item_summary">
                <div className="ifa_item_summary_left">
                  <div className="ifa_item_label">{`ITEM ${itemNo}`}</div>
                  <div className="ifa_item_name">{applianceName}</div>
                </div>
                <div className="ifa_item_summary_right">
                  <div className="ifa_item_meta">
                    <span className="label">Decision</span>
                    <span className="value">
                      {decision ? (
                        <DecisionBadge decision={decision} />
                      ) : (
                        <span className="ifa_badge ifa_badge_no_decision">No Decision</span>
                      )}
                    </span>
                  </div>
                  <div className="ifa_item_meta">
                    <span className="label">Amount</span>
                    <span className="value">
                      <strong>{topAmount || "Not applicable"}</strong>
                    </span>
                  </div>
                  <span className="ifa_item_chevron" aria-hidden="true">
                    ▾
                  </span>
                </div>
              </summary>

              <div className="ifa_item_body">
                {/* Amounts near the top as requested */}
                {(it.amountsCustomer?.length || it.amountsCompany?.length) ? (
                  <div className="ifa_amounts ifa_amounts_top">
                    <div className="h">
                      <strong>Amounts</strong>
                    </div>
                    {it.amountsCustomer?.length ? (
                      <div className="sub">
                        <div className="k">
                          <strong>Customer</strong>
                        </div>
                        <ul>
                          {it.amountsCustomer
                            .map(cleanAmountLine)
                            .filter(Boolean)
                            .map((x, i) => (
                              <li key={i}>
                                <strong>{x}</strong>
                              </li>
                            ))}
                        </ul>
                      </div>
                    ) : null}
                    {it.amountsCompany?.length ? (
                      <div className="sub">
                        <div className="k">
                          <strong>Company</strong>
                        </div>
                        <ul>
                          {it.amountsCompany
                            .map(cleanAmountLine)
                            .filter(Boolean)
                            .map((x, i) => (
                              <li key={i}>
                                <strong>{x}</strong>
                              </li>
                            ))}
                        </ul>
                      </div>
                    ) : null}
                  </div>
                ) : null}

                <div className="ifa_meta">
                  {/* "Item" and "Type" are part of the requested bold set */}
                  {applianceName && applianceName !== `Item ${idx + 1}` ? (
                    <div className="row">
                      <div className="k">
                        <strong>Item</strong>
                      </div>
                      <div className="v">
                        <strong>{applianceName}</strong>
                      </div>
                    </div>
                  ) : null}
                  {it.type && it.type.trim() ? (
                    <div className="row">
                      <div className="k">
                        <strong>Type</strong>
                      </div>
                      <div className="v">
                        <strong>{it.type}</strong>
                      </div>
                    </div>
                  ) : null}
                  {it.related && it.related.trim() ? (
                    <div className="row">
                      <div className="k">
                        <strong>Related</strong>
                      </div>
                      <div className="v">{it.related}</div>
                    </div>
                  ) : null}
                </div>

                {/* Situation section with enhanced styling */}
                {it.situation && it.situation.trim() ? (
                  <div className="ifa_situation">
                    <div className="h">
                      <strong>Situation</strong>
                    </div>
                    <div className="content">{it.situation}</div>
                  </div>
                ) : null}

                {(it.covered?.length || it.notCovered?.length) ? (
                  <div className="ifa_split">
                    {it.covered?.length ? (
                      <div className="col">
                        <div className="h">
                          <strong>What's covered</strong>
                        </div>
                        <ul>
                          {it.covered.map((x, i) => (
                            <li key={i}>{x}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {it.notCovered?.length ? (
                      <div className="col ifa_limitations">
                        <div className="h">
                          <strong>Limitations / not covered</strong>
                        </div>
                        <ul>
                          {it.notCovered.map((x, i) => (
                            <li key={i}>{x}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                  </div>
                ) : null}

                {it.why?.length ? (
                  <div className="ifa_why">
                    <div className="h">
                      <strong>Why</strong>
                    </div>
                    <ul>
                      {it.why.map((x, i) => (
                        <li key={i}>{x}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}

                {it.nextSteps?.length ? (
                  <div className="ifa_next">
                    <div className="h">
                      <strong>Next steps</strong>
                    </div>
                    <ul>
                      {it.nextSteps.map((x, i) => (
                        <li key={i}>{x}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>
            </details>
          );
        })}
      </div>
    </div>
  );
};

const arrToText = (arr) => (Array.isArray(arr) ? arr.join("\n") : "");
// Preserve spaces while typing in edit mode (no trim/filter); trimming happens on save/serialize
const textToArr = (text) => (String(text || "").split("\n"));

/**
 * Editable form that mirrors ItemizedFinalAnswer layout. Same sections, but each value is an input/textarea.
 * Props: parsed = { items, overall, overallNextSteps }, onChange(parsed).
 */
export const ItemizedFinalAnswerEditable = ({ parsed = {}, onChange, asCard = true }) => {
  const items = Array.isArray(parsed.items) ? parsed.items : [];
  const updateItem = (idx, updates) => {
    const next = { ...parsed, items: [...(parsed.items || [])] };
    next.items[idx] = { ...(next.items[idx] || {}), ...updates };
    onChange(next);
  };

  if (items.length === 0) return null;

  return (
    <div className={`itemized_final_answer ${asCard ? "ifa_outer_card" : ""} ifa_editable`}>
      <div className="ifa_cards">
        {items.map((it, idx) => {
          const itemNo = it.itemNo || String(idx + 1);
          // Use raw values so spaces are preserved while typing (no trim on display)
          const itemName = String(it.title ?? it.name ?? "");
          const amountKeyFact = String(it.amount ?? "");

          return (
            <div className="ifa_card" key={`edit-${itemNo}-${idx}`}>
              <div className="ifa_item_header">
                <strong>{`ITEM ${itemNo}:`}</strong>
              </div>

              <div className="ifa_keyfacts">
                <div className="ifa_keyfacts_row">
                  <div className="k">
                    <strong>Decision</strong>
                  </div>
                  <div className="v">
                    {(() => {
                      const raw = (it.decision || "").trim();
                      const normalized =
                        /approv|accept/i.test(raw) ? "Approved" : /deny|reject|denied/i.test(raw) ? "Denied" : "";
                      const decisionClass = normalized === "Approved" ? "ifa_decision_approved" : normalized === "Denied" ? "ifa_decision_denied" : "ifa_decision_no_decision";
                      return (
                        <select
                          className={`ifa_select_decision ${decisionClass}`}
                          value={normalized}
                          onChange={(e) => updateItem(idx, { ...it, decision: e.target.value })}
                          aria-label="Decision"
                        >
                          <option value="">No decision</option>
                          <option value="Approved">Approved</option>
                          <option value="Denied">Denied</option>
                        </select>
                      );
                    })()}
                  </div>
                </div>
                <div className="ifa_keyfacts_row">
                  <div className="k">
                    <strong>Amount</strong>
                  </div>
                  <div className="v">
                    <input
                      type="text"
                      className="ifa_input"
                      value={amountKeyFact}
                      onChange={(e) => updateItem(idx, { ...it, amount: e.target.value })}
                      placeholder="Not specified"
                    />
                  </div>
                </div>
              </div>

              <div className="ifa_amounts ifa_amounts_top">
                <div className="h">
                  <strong>Amounts</strong>
                </div>
                <div className="sub">
                  <div className="k">
                    <strong>Customer</strong>
                  </div>
                  <textarea
                    className="ifa_textarea"
                    rows={2}
                    value={arrToText(it.amountsCustomer)}
                    onChange={(e) => updateItem(idx, { ...it, amountsCustomer: textToArr(e.target.value) })}
                    placeholder="Not specified"
                  />
                </div>
                <div className="sub">
                  <div className="k">
                    <strong>Company</strong>
                  </div>
                  <textarea
                    className="ifa_textarea"
                    rows={2}
                    value={arrToText(it.amountsCompany)}
                    onChange={(e) => updateItem(idx, { ...it, amountsCompany: textToArr(e.target.value) })}
                    placeholder="Not specified"
                  />
                </div>
              </div>

              <div className="ifa_meta">
                <div className="row">
                  <div className="k">
                    <strong>Item</strong>
                  </div>
                  <div className="v">
                    <input
                      type="text"
                      className="ifa_input"
                      value={itemName}
                      onChange={(e) => updateItem(idx, { ...it, title: e.target.value, name: e.target.value })}
                      placeholder="Item name"
                    />
                  </div>
                </div>
                <div className="row">
                  <div className="k">
                    <strong>Type</strong>
                  </div>
                  <div className="v">
                    <input
                      type="text"
                      className="ifa_input"
                      value={it.type || ""}
                      onChange={(e) => updateItem(idx, { ...it, type: e.target.value })}
                      placeholder="Type"
                    />
                  </div>
                </div>
                <div className="row">
                  <div className="k">
                    <strong>Related</strong>
                  </div>
                  <div className="v">
                    <input
                      type="text"
                      className="ifa_input"
                      value={it.related || ""}
                      onChange={(e) => updateItem(idx, { ...it, related: e.target.value })}
                      placeholder="Related"
                    />
                  </div>
                </div>
              </div>

              <div className="ifa_situation">
                <div className="h">
                  <strong>Situation</strong>
                </div>
                <textarea
                  className="ifa_textarea"
                  rows={3}
                  value={it.situation || ""}
                  onChange={(e) => updateItem(idx, { ...it, situation: e.target.value })}
                  placeholder="Situation"
                />
              </div>

              <div className="ifa_split">
                <div className="col">
                  <div className="h">
                    <strong>What's covered</strong>
                  </div>
                  <textarea
                    className="ifa_textarea"
                    rows={3}
                    value={arrToText(it.covered)}
                    onChange={(e) => updateItem(idx, { ...it, covered: textToArr(e.target.value) })}
                    placeholder="Not specified"
                  />
                </div>
                <div className="col ifa_limitations">
                  <div className="h">
                    <strong>Limitations / not covered</strong>
                  </div>
                  <textarea
                    className="ifa_textarea"
                    rows={3}
                    value={arrToText(it.notCovered)}
                    onChange={(e) => updateItem(idx, { ...it, notCovered: textToArr(e.target.value) })}
                    placeholder="Not specified"
                  />
                </div>
              </div>

              <div className="ifa_why">
                <div className="h">
                  <strong>Why</strong>
                </div>
                <textarea
                  className="ifa_textarea"
                  rows={2}
                  value={arrToText(it.why)}
                  onChange={(e) => updateItem(idx, { ...it, why: textToArr(e.target.value) })}
                  placeholder="Not specified"
                />
              </div>

              <div className="ifa_next">
                <div className="h">
                  <strong>Next steps</strong>
                </div>
                <textarea
                  className="ifa_textarea"
                  rows={2}
                  value={arrToText(it.nextSteps)}
                  onChange={(e) => updateItem(idx, { ...it, nextSteps: textToArr(e.target.value) })}
                  placeholder="Not specified"
                />
              </div>
            </div>
          );
        })}
      </div>

      <div className="ifa_overall_next_edit">
        <div className="h">
          <strong>Overall Next Step</strong>
        </div>
        <textarea
          className="ifa_textarea"
          rows={2}
          value={parsed.overallNextSteps || ""}
          onChange={(e) => onChange({ ...parsed, overallNextSteps: e.target.value })}
          placeholder="Overall next step"
        />
      </div>
    </div>
  );
};


