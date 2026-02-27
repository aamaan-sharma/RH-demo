/**
 * Claims / Final Analyzed Answer parser.
 * Converts backend response text (markdown-style) into a structured shape
 * so rendering and reusability are consistent for all response types
 * (final answer, draft answer, single item, multiple coverage components).
 *
 * Usage: parseClaimsFinalAnswer(text) → { items, overall, overallNextSteps }
 * Each item: { itemNo, title, name, type, situation, decision, amount, covered,
 *              notCovered, amountsCustomer, amountsCompany, why, nextSteps,
 *              clauseReference, ... }
 */

const ITEM_START_RE = /^Item\s*#?\s*:?\s*(\d+)\b/i;
const ITEM_START_WITH_TITLE_RE = /^Item\s*#?\s*(\d+)\s*:\s*(.+)$/i;
const COVERAGE_COMPONENT_START_RE = /^Coverage\s+Component\s+(\d+)\b/i;
const COVERAGE_COMPONENT_WITH_TITLE_RE =
  /^Coverage\s+Component\s+(\d+)\s*:\s*(.+)$/i;

export const stripMdDecorators = (s) => {
  let t = String(s || "").trim();
  t = t
    .replace(/^#{1,6}\s*/g, "")
    .replace(/^>\s*/g, "")
    .trim();
  t = t.replace(/^\*\*/, "").replace(/\*\*$/, "").trim();
  return t;
};

/** Remove Plan and State lines from display (per product request). */
export const stripPlanAndStateFromText = (s) => {
  const raw = String(s || "").replace(/\r\n/g, "\n");
  const lines = raw.split("\n");
  const filtered = lines.filter(
    (line) =>
      !/^\s*Plan\s*:.*$/i.test(line.trim()) &&
      !/^\s*State\s*:.*$/i.test(line.trim()),
  );
  return filtered
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
};

/** Default shape for one claim/coverage item. Use for creating empty items or validating structure. */
export const DEFAULT_ITEM_SHAPE = () => ({
  itemNo: "",
  title: "",
  name: "",
  type: "",
  related: "",
  situation: "",
  decision: "",
  amount: "",
  answer: "",
  policyBasis: "",
  moneyReconciliation: [],
  covered: [],
  notCovered: [],
  amountsCustomer: [],
  amountsCompany: [],
  why: [],
  nextSteps: [],
  clauseReference: [],
  componentItems: [],
  raw: "",
});

/**
 * Parse Claims / Final Analyzed Answer text into a structured object.
 * Handles: multiple Coverage Components, single unnumbered item, plain text (no items).
 * @param {string} text - Raw response from backend
 * @returns {{ items: Array, overall: string, overallNextSteps: string }}
 */
export function parseClaimsFinalAnswer(text) {
  const raw = String(text || "").replace(/\r\n/g, "\n");
  const lines = raw.split("\n");
  const starts = [];
  let isSingleItemFallback = false;

  const keyRe =
    /^(Items?|Item|Type|Related|Situation|Decision|Decision posture|Amount|Amounts|Why|Next steps?|Answer|Policy basis|Money reconciliation|What.?s covered|What.?s not covered|Limitations\s*\/\s*not covered)\s*:/i;

  const looksLikeSingleItem = () => {
    let hits = 0;
    for (const ln of lines) {
      const probe = stripMdDecorators(ln)
        .replace(/^[-•*]\s+/, "")
        .trim();
      if (keyRe.test(probe)) hits += 1;
      if (hits >= 2) return true;
    }
    return false;
  };

  for (let i = 0; i < lines.length; i++) {
    const probe = stripMdDecorators(lines[i]);
    if (ITEM_START_RE.test(probe) || COVERAGE_COMPONENT_START_RE.test(probe)) {
      starts.push(i);
    }
  }
  if (starts.length === 0) {
    if (looksLikeSingleItem()) {
      starts.push(0);
      isSingleItemFallback = true;
    } else {
      return { items: [], overall: raw.trim(), overallNextSteps: "" };
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
      ...DEFAULT_ITEM_SHAPE(),
      raw: secLines.join("\n").trim(),
    };
    item.itemNo = isSingleItemFallback ? "1" : "";

    let mode = "";
    for (const ln of secLines) {
      const indentMatch = ln.match(/^(\s*)/);
      const currentIndent = indentMatch ? indentMatch[1].length : 0;
      const isNestedBullet = currentIndent > 2 && /^\s+[-•*]\s+/.test(ln);

      let t = stripMdDecorators(ln);
      if (!t) continue;

      const isBullet = /^[-•*]\s+/.test(t);
      const bulletText = isBullet ? t.replace(/^[-•*]\s+/, "").trim() : t;
      const base = bulletText.trim();

      const mStartWithTitle = base.match(ITEM_START_WITH_TITLE_RE);
      if (mStartWithTitle) {
        item.itemNo = mStartWithTitle[1];
        item.title = (mStartWithTitle[2] || "").trim();
        mode = "";
        continue;
      }

      const mCoverageWithTitle = base.match(COVERAGE_COMPONENT_WITH_TITLE_RE);
      if (mCoverageWithTitle) {
        item.itemNo = mCoverageWithTitle[1];
        item.title = (mCoverageWithTitle[2] || "").trim();
        mode = "";
        continue;
      }

      if (base.match(ITEM_START_RE)) {
        item.itemNo = base.match(ITEM_START_RE)[1];
        mode = "";
        continue;
      }
      if (base.match(COVERAGE_COMPONENT_START_RE)) {
        item.itemNo = base.match(COVERAGE_COMPONENT_START_RE)[1];
        mode = "";
        continue;
      }

      const kv = (label) => {
        let re = new RegExp(`^${label}\\s*:\\s*(.+)$`, "i");
        let m = base.match(re);
        if (m) return m[1].trim();
        re = new RegExp(`^${label}\\s*:?\\s*(.+)$`, "i");
        m = base.match(re);
        if (m) return m[1].trim();
        if (base.toLowerCase().startsWith(label.toLowerCase())) {
          const afterLabel = base
            .slice(label.length)
            .replace(/^[\s:]+/, "")
            .trim();
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
        if (
          trimmedRelated &&
          !/^(none|n\/a|na|not specified|—|-)$/i.test(trimmedRelated)
        ) {
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
      const answerVal = kv("Answer");
      if (answerVal) {
        item.answer = answerVal.trim();
        mode = "";
        continue;
      }
      const decision = kv("Decision");
      if (decision) {
        const trimmedDecision = decision.trim();
        if (trimmedDecision && !/^[-—n\/a]+$/i.test(trimmedDecision)) {
          item.decision = trimmedDecision;
        }
        mode = "";
        continue;
      }
      const decisionPosture = kv("Decision posture");
      if (decisionPosture) {
        const trimmed = decisionPosture.trim();
        if (trimmed && !/^[-—n\/a]+$/i.test(trimmed)) {
          item.decision = trimmed;
        }
        mode = "";
        continue;
      }
      const amountMatch = base.match(/^Amount\s*:\s*(.*)$/i);
      if (amountMatch) {
        item.amount = (amountMatch[1] || "").trim();
        mode = "";
        continue;
      }

      const coveredInline = base.match(/^What.?s covered\s*:\s*(.*)$/i);
      if (coveredInline) {
        const v = String(coveredInline[1] || "").trim();
        if (v && !/^none$/i.test(v)) item.covered.push(v);
        mode = "";
        continue;
      }
      const notCoveredInline = base.match(
        /^What.?s not covered(?:\s*\/\s*limitations)?\s*:\s*(.*)$/i,
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
      const limitationsInline = base.match(
        /^Limitations\s*\/\s*not covered\s*:\s*(.*)$/i,
      );
      if (limitationsInline) {
        const v = String(limitationsInline[1] || "").trim();
        if (v) {
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
          mode = "amounts";
        } else {
          item.amountsCustomer.push(v);
          mode = "amounts";
        }
        continue;
      }
      const whyInline = base.match(/^Why\s*:\s*(.*)$/i);
      if (whyInline) {
        const v = String(whyInline[1] || "").trim();
        if (v && !/^none$/i.test(v)) {
          item.why.push(v);
          mode = "";
        } else {
          mode = "why";
        }
        continue;
      }
      const policyBasisVal = base.match(/^Policy basis\s*:\s*(.*)$/i);
      if (policyBasisVal) {
        const v = String(policyBasisVal[1] || "").trim();
        if (v) item.policyBasis = v;
        mode = "";
        continue;
      }
      const moneyReconMatch = base.match(/^Money reconciliation\s*:?\s*(.*)$/i);
      if (moneyReconMatch) {
        const v = String(moneyReconMatch[1] || "").trim();
        if (v) item.moneyReconciliation.push(v);
        mode = "moneyReconciliation";
        continue;
      }
      const nextInline = base.match(/^Next steps?\s*:\s*(.*)$/i);
      if (nextInline) {
        const v = String(nextInline[1] || "").trim();
        if (v && !/^none$/i.test(v)) item.nextSteps.push(v);
        mode = "";
        continue;
      }
      const clauseRefInline = base.match(/^Clause Reference\s*:\s*(.*)$/i);
      if (clauseRefInline) {
        const v = String(clauseRefInline[1] || "").trim();
        if (v && !/^none$/i.test(v)) item.clauseReference.push(v);
        mode = "";
        continue;
      }

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
      if (/^Items\b/i.test(base)) {
        mode = "componentItems";
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
      if (/^Clause Reference\b/i.test(base)) {
        mode = "clauseReference";
        continue;
      }
      if (/^Policy basis\b/i.test(base)) {
        mode = "";
        continue;
      }
      if (/^Money reconciliation\b/i.test(base)) {
        mode = "moneyReconciliation";
        continue;
      }

      if (mode === "moneyReconciliation") {
        const trimmed = bulletText.trim();
        if (trimmed) item.moneyReconciliation.push(trimmed);
        continue;
      }
      if (mode === "covered") {
        const trimmed = bulletText.trim();
        if (trimmed && !/^none$/i.test(trimmed)) item.covered.push(trimmed);
      } else if (mode === "notCovered") {
        const trimmed = bulletText.trim();
        if (trimmed && !/^none$/i.test(trimmed)) item.notCovered.push(trimmed);
      } else if (mode === "why") {
        const trimmed = bulletText.trim();
        if (trimmed && !/^none$/i.test(trimmed)) item.why.push(trimmed);
      } else if (mode === "componentItems") {
        const trimmed = bulletText.trim();
        if (trimmed) {
          const colonIdx = trimmed.indexOf(":");
          if (colonIdx !== -1) {
            const name = trimmed.slice(0, colonIdx).trim();
            const details = trimmed.slice(colonIdx + 1).trim();
            item.componentItems.push({ name, details });
          } else {
            const parenMatch = trimmed.match(/^(.+?)\s*\(([^)]*)\)\s*$/);
            if (parenMatch) {
              item.componentItems.push({
                name: parenMatch[1].trim(),
                details: parenMatch[2].trim(),
              });
            } else {
              item.componentItems.push({ name: trimmed, details: "" });
            }
          }
        }
      } else if (mode === "nextSteps") {
        const trimmed = bulletText.trim();
        if (trimmed && !/^none$/i.test(trimmed)) item.nextSteps.push(trimmed);
      } else if (mode === "clauseReference") {
        const trimmed = bulletText.trim();
        if (trimmed && !/^none$/i.test(trimmed))
          item.clauseReference.push(trimmed);
      } else if (mode === "amounts") {
        const trimmed = bulletText.trim();
        if (trimmed && !/^none$/i.test(trimmed)) {
          if (/customer|quoted|asked/i.test(trimmed)) {
            item.amountsCustomer.push(
              trimmed
                .replace(/^(customer\s*(quoted\/asked)?\s*:?\s*)/i, "")
                .trim(),
            );
          } else if (/company|we can|can provide/i.test(trimmed)) {
            item.amountsCompany.push(
              trimmed
                .replace(/^(company\s*(can\s*provide)?\s*:?\s*)/i, "")
                .trim(),
            );
          } else {
            item.amountsCustomer.push(trimmed);
          }
        }
      } else if (mode === "overallNext") {
        const trimmed = bulletText.trim();
        if (trimmed && !/^none$/i.test(trimmed)) item.nextSteps.push(trimmed);
      }
    }

    return item;
  });

  const overall = lines.slice(0, starts[0]).join("\n").trim();
  let overallNextSteps = "";
  try {
    const tailIdx = lines.findIndex((l) =>
      /^Overall Next Steps?\b/i.test(stripMdDecorators(l).replace(/:$/, "")),
    );
    if (tailIdx >= 0) {
      overallNextSteps = lines.slice(tailIdx).join("\n").trim();
    }
  } catch (e) {
    overallNextSteps = "";
  }

  return { items, overall, overallNextSteps };
}
