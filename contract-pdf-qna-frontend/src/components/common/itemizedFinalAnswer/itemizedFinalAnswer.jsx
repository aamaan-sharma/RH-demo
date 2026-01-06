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
  if (raw.includes("REJECT") || raw.includes("DENY")) return "REJECTED";
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
      /^(Item|Type|Related|Situation|Decision|Amounts|Why|Next steps?|What.?s covered|What.?s not covered|Limitations\s*\/\s*not covered)\s*:/i;
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

const DecisionBadge = ({ decision }) => {
  const norm = normalizeDecision(decision);
  const cls = norm.toLowerCase().replace(/[^a-z0-9]+/g, "_");
  const displayText = norm === "NO_DECISION" ? "No Decision" : norm.replace(/_/g, " ");
  return <span className={`ifa_badge ifa_badge_${cls}`}>{displayText}</span>;
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

          // Prefer the company-provided amount line if present; else customer; else not applicable.
          const topAmount =
            (it.amountsCompany?.[0] || "").replace(/^Company\s*(can\s*provide)?\s*:\s*/i, "").trim() ||
            (it.amountsCustomer?.[0] || "").replace(/^Customer\s*(quoted\/asked)?\s*:\s*/i, "").trim() ||
            (hasAmounts ? "See Amounts below" : "Not applicable");
          return (
            <div className="ifa_card" key={`${itemNo}-${idx}`}>
              <div className="ifa_item_header">
                <strong>{`ITEM ${itemNo}:`}</strong>
              </div>

              {/* Key facts (in order): Decision, Amount */}
              <div className="ifa_keyfacts">
                <div className="ifa_keyfacts_row">
                  <div className="k">
                    <strong>Decision</strong>
                  </div>
                  <div className="v">
                    {decision ? (
                      <DecisionBadge decision={decision} />
                    ) : (
                      <span className="ifa_badge ifa_badge_no_decision">No Decision</span>
                    )}
                  </div>
                </div>
                <div className="ifa_keyfacts_row">
                  <div className="k">
                    <strong>Amount</strong>
                  </div>
                  <div className="v">
                    <strong>{topAmount || "Not applicable"}</strong>
                  </div>
                </div>
              </div>

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
          );
        })}
      </div>
    </div>
  );
};


