import React, { useState } from "react";
import { StructuredCaseText } from "../structuredCaseText/structuredCaseText";
import {
  parseClaimsFinalAnswer,
  stripPlanAndStateFromText,
} from "./claimsFinalAnswerParser";

/** Re-export for consumers that want to parse once and pass parsed to ItemizedFinalAnswer or reuse the structure. */
export { parseClaimsFinalAnswer } from "./claimsFinalAnswerParser";
import "./itemizedFinalAnswer.scss";

/** Normalize to exactly 3 states: APPROVED, DENIED, PARTIAL */
const normalizeDecision = (s) => {
  const raw = String(s || "")
    .trim()
    .toUpperCase();
  if (!raw || raw === "—" || raw === "-" || raw === "N/A" || raw === "NA" || raw === "NONE") {
    return "PARTIAL";
  }
  if (raw.includes("APPROV") || raw.includes("ACCEPT")) return "APPROVED";
  if (raw.includes("REJECT") || raw.includes("DENY") || raw.includes("DENIED")) return "DENIED";
  if (raw.includes("PARTIAL")) return "PARTIAL";
  return "PARTIAL";
};

/** Exported for popup edit mode: parse draft text into { items, overall, overallNextSteps }. */
export const parseDraftSummary = (text) =>
  parseClaimsFinalAnswer(text || "");

/**
 * Serialize parsed structure back to draft summary text (same style the parser expects).
 * Round-trip: parseDraftSummary(serializeDraftSummary(parsed)) should yield equivalent structure.
 */
export const serializeDraftSummary = (parsed) => {
  if (!parsed || !Array.isArray(parsed.items)) return "";
  const lines = [];
  const str = (v) => (v == null || v === undefined ? "" : String(v).trim());
  const arrJoin = (arr) =>
    Array.isArray(arr) && arr.length ? arr.join("\n") : "";

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
      (it.amountsCustomer || []).forEach((c) =>
        lines.push(`- Customer: ${str(c)}`),
      );
      (it.amountsCompany || []).forEach((c) =>
        lines.push(`- Company: ${str(c)}`),
      );
    }
    if (it.componentItems?.length) {
      lines.push("Items:");
      it.componentItems.forEach((ci) =>
        lines.push(
          ci.details ? `  - ${str(ci.name)}: ${str(ci.details)}` : `  - ${str(ci.name)}`,
        ),
      );
    }
    if (it.why?.length) {
      lines.push("Why:");
      it.why.forEach((w) => lines.push(`- ${str(w)}`));
    }
    if (it.nextSteps?.length) {
      lines.push("Next steps:");
      it.nextSteps.forEach((n) => lines.push(`- ${str(n)}`));
    }
    if (it.clauseReference?.length) {
      lines.push("Clause Reference:");
      it.clauseReference.forEach((c) => lines.push(`- ${str(c)}`));
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
 * Normalize a value for equality: if two values normalize to the same string, they are
 * considered the same and must not appear in the changelog (do not show or store).
 */
function normalizeForChangelogCompare(value, fieldKey) {
  const s = String(value ?? "").trim().replace(/\s+/g, " ");
  if (!s) return "";
  const lower = s.toLowerCase();
  if (lower === "not specified" || lower === "n/a" || lower === "na" || lower === "—" || lower === "-") return "";
  if (fieldKey === "Decision") return s.toUpperCase();
  if (fieldKey === "Amount" || fieldKey === "Customer Quoted / Asked" || fieldKey === "Company Can Provide") {
    return s.replace(/\s+/g, "").replace(/,/g, "").replace(/^\$*/, "").trim();
  }
  return s;
}

function isSameValue(prev, next, fieldKey) {
  const p = normalizeForChangelogCompare(prev, fieldKey);
  const n = normalizeForChangelogCompare(next, fieldKey);
  return p === n || (!p && !n);
}

/**
 * Build changelog entries: only fields whose value actually changed (after normalization).
 * If previous and current normalize to the same value (e.g. DENIED vs Denied, $0 vs 0),
 * that field is not included and is not stored in the changelog.
 */
export const buildSummaryFieldChanges = (previousText, updatedText) => {
  const prev = parseClaimsFinalAnswer(previousText || "");
  const next = parseClaimsFinalAnswer(updatedText || "");
  const changes = [];
  const prevItems = prev.items || [];
  const nextItems = next.items || [];

  const str = (v) => (v == null || v === undefined ? "" : String(v).trim());
  const arrStr = (arr) =>
    Array.isArray(arr) && arr.length ? arr.join("; ") : "";
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
      { key: "What's Not Covered / Limitations", prev: arrStr(prevItem.notCovered), next: arrStr(nextItem.notCovered) },
      { key: "Customer Quoted / Asked", prev: arrStr(prevItem.amountsCustomer), next: arrStr(nextItem.amountsCustomer) },
      { key: "Company Can Provide", prev: arrStr(prevItem.amountsCompany), next: arrStr(nextItem.amountsCompany) },
      { key: "Why", prev: arrStr(prevItem.why), next: arrStr(nextItem.why) },
      { key: "Next Steps (Item Level)", prev: arrStr(prevItem.nextSteps), next: arrStr(nextItem.nextSteps) },
      { key: "Clause Reference", prev: arrStr(prevItem.clauseReference), next: arrStr(nextItem.clauseReference) },
    ];

    for (const f of fields) {
      const p = str(f.prev);
      const n = str(f.next);
      if (isSameValue(p, n, f.key)) continue;
      changes.push({
        fieldName: prefix + f.key,
        previousValue: emptyLabel(p) || "—",
        updatedValue: emptyLabel(n) || "—",
      });
    }
  }

  const prevOverall = str(prev.overallNextSteps || "");
  const nextOverall = str(next.overallNextSteps || "");
  if (!isSameValue(prevOverall, nextOverall, "Overall Next Step")) {
    changes.push({
      fieldName: "Overall Next Step",
      previousValue: emptyLabel(prevOverall) || "—",
      updatedValue: emptyLabel(nextOverall) || "—",
    });
  }

  if (
    changes.length === 0 &&
    prevItems.length === 0 &&
    nextItems.length === 0 &&
    !isSameValue(previousText, updatedText, "Summary")
  ) {
    return [
      { fieldName: "Summary", previousValue: str(previousText) || "—", updatedValue: str(updatedText) || "—" },
    ];
  }
  return changes;
};

export const DecisionBadge = ({ decision }) => {
  const norm = normalizeDecision(decision);
  const toneClass =
    norm === "APPROVED"
      ? "ifa_decision_approved"
      : norm === "DENIED"
        ? "ifa_decision_denied"
        : "ifa_decision_partial";
  const displayText = norm === "APPROVED" ? "APPROVED" : norm === "DENIED" ? "DENIED" : "PARTIAL";
  return (
    <span className={`ifa_decision_text ${toneClass}`}>{displayText}</span>
  );
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

/** If value looks like a dollar amount (e.g. $0, $123, 250) return it; otherwise return $0 for Final Analyzed Answer. */
const normalizeAmountForDisplay = (val) => {
  const s = String(val || "").trim();
  if (!s) return "$0";
  // Accept: $0, $123, $1,234.56, 0, 250, etc.
  if (/^\$?\s*[\d,]+(\.\d{0,2})?$/.test(s))
    return s.replace(/^\s+/, "").startsWith("$") ? s : `$${s}`;
  return "$0";
};

/** Parse display amount (e.g. "$0", "$252.38") to number for comparison. */
const parseAmountToNumber = (displayVal) => {
  const s = String(displayVal || "").replace(/[$,]/g, "").trim();
  const n = parseFloat(s);
  return Number.isFinite(n) ? n : 0;
};

/** Get amount parts for display: N/A when both $0, else Company/Customer values. Bold the party that pays (non-zero). */
const getAmountParts = (it) => {
  const custRaw = it.amountsCustomer?.length
    ? cleanAmountLine(it.amountsCustomer[0])
    : "";
  const compRaw = it.amountsCompany?.length
    ? cleanAmountLine(it.amountsCompany[0])
    : "";
  const single = (it.amount || "").trim();
  const isExplicitNa = /^(n\/a|na|not applicable|not stated|—|-)$/i.test(single);

  if (!custRaw && !compRaw && isExplicitNa) {
    return { companyVal: "N/A", customerVal: "N/A", bothZero: true };
  }

  const companyVal = compRaw ? normalizeAmountForDisplay(compRaw) : "$0";
  const customerVal = custRaw ? normalizeAmountForDisplay(custRaw) : "$0";
  const companyNum = parseAmountToNumber(companyVal);
  const customerNum = parseAmountToNumber(customerVal);
  const bothZero = companyNum === 0 && customerNum === 0;
  return { companyVal, customerVal, bothZero, companyNum, customerNum };
};

/** Renders amount: N/A when both $0; otherwise "Company - X, Customer - Y" with the paying party (non-zero) in bold. */
const AmountDisplay = ({ item }) => {
  const { companyVal, customerVal, bothZero, companyNum, customerNum } = getAmountParts(item);
  if (bothZero) {
    return <span className="ifa_amount_display">N/A</span>;
  }
  const boldCompany = companyNum !== 0;
  const boldCustomer = customerNum !== 0;
  return (
    <span className="ifa_amount_display">
      {boldCompany ? <strong>Company</strong> : "Company"} - {companyVal}, {boldCustomer ? <strong>Customer</strong> : "Customer"} - {customerVal}
    </span>
  );
};

/** Item name from claim for display (no "Claim 1" / "Coverage Component 1"). */
function getClaimDisplayName(claim) {
  const items = claim?.items;
  if (!Array.isArray(items) || items.length === 0) return "";
  return items.map((i) => (i?.name || "").trim()).filter(Boolean).join(", ") || "";
}

const CLAIM_ID_RE = /^q\d+$/i;

/** Derive a short label for the authorization scope from decision summary/reasons when the item name is missing or is a claim ID (e.g. q11). */
function getClaimDisplayLabel(claim) {
  if (!claim) return "";
  const fromItems = getClaimDisplayName(claim);
  if (fromItems && !CLAIM_ID_RE.test(fromItems.trim())) return fromItems;
  const summary = (claim.decisionSummary || "").trim();
  let s = summary || (Array.isArray(claim.reasons) && claim.reasons.length ? String(claim.reasons[0] || "").trim() : "");
  if (!s) return fromItems || "";
  s = s
    .replace(/^(?:The\s+)?authorized\s+/i, "")
    .replace(/^(?:The\s+)?authorization\s+(?:is\s+)?(?:for\s+)?/i, "")
    .replace(/\s+is\s+approved\.?$/i, "")
    .replace(/\s+has\s+been\s+approved\.?$/i, "")
    .replace(/\s+is\s+denied\.?$/i, "")
    .replace(/\s+was\s+denied\.?$/i, "")
    .replace(/\s+is\s+partially\s+approved\.?$/i, "")
    .replace(/\s+per\s+call\s+notes\.?$/i, "")
    .replace(/\s+/g, " ")
    .trim();
  if (s.length > 80) s = s.slice(0, 77).split(/\s+/).slice(0, -1).join(" ") + "...";
  return s || fromItems || "";
}

/** Build a display item for AmountDisplay from canonical claim.amounts (single source of truth). */
function getItemForClaimAmounts(claim, fallbackItem) {
  const amt = claim?.amounts;
  if (!amt || typeof amt !== "object") return fallbackItem;
  const company = amt.company_total != null || amt.authorized_by_company != null
    ? `$${Number(amt.company_total ?? amt.authorized_by_company ?? 0).toFixed(2)}`
    : "";
  const customer = amt.customer_out_of_pocket != null
    ? `$${Number(amt.customer_out_of_pocket).toFixed(2)}`
    : "";
  if (!company && !customer) return fallbackItem;
  return {
    ...(fallbackItem || {}),
    amountsCompany: company ? [company] : (fallbackItem?.amountsCompany || []),
    amountsCustomer: customer ? [customer] : (fallbackItem?.amountsCustomer || []),
    amount: [company, customer].filter(Boolean).length ? `Company ${company}, Customer ${customer}` : (fallbackItem?.amount || ""),
  };
}

export const ItemizedFinalAnswer = ({
  text = "",
  parsed: parsedProp,
  title = "Final Answer",
  asCard = true,
  hideSummaryDecisionAmount = false,
  compactLayout = false,
  claimForChat = null,
}) => {
  const raw = String(text || "").trim();
  const parsed =
    parsedProp && Array.isArray(parsedProp.items)
      ? parsedProp
      : raw
        ? parseClaimsFinalAnswer(text)
        : { items: [], overall: "", overallNextSteps: "" };
  if (!raw && !(parsed.items && parsed.items.length)) return null;

  const hasItems = Array.isArray(parsed.items) && parsed.items.length > 0;

  const wrapperClass = [
    "itemized_final_answer",
    asCard ? "ifa_outer_card" : "",
    compactLayout ? "ifa_compact" : "",
  ]
    .filter(Boolean)
    .join(" ");

  if (!hasItems) {
    const fallbackText = raw || (parsed.overall || "") || "";
    return (
      <div className={wrapperClass}>
        <div className="ifa_title">{title}</div>
        <StructuredCaseText text={stripPlanAndStateFromText(fallbackText)} />
      </div>
    );
  }

  const overallWithoutPlanState = stripPlanAndStateFromText(
    parsed.overall || "",
  );

  return (
    <div className={wrapperClass}>
      <div className="ifa_title">{title}</div>
      {overallWithoutPlanState ? (
        <div className="ifa_overall">
          <StructuredCaseText text={overallWithoutPlanState} />
        </div>
      ) : null}

      {/* Claim blocks: collapsible in Final Answer, always expanded in compact (per-question) */}
      <div
        className="ifa_tabs"
        role={compactLayout ? "list" : "tablist"}
        aria-label="Coverage Components"
      >
        {parsed.items.map((it, idx) => {
          // Prefer descriptive label (authorization scope) over claim ID (e.g. "Labor cost for two outlet repair" over "q11")
          const claimLabel = compactLayout && claimForChat && idx === 0 ? getClaimDisplayLabel(claimForChat) : "";
          const parsedTitleOrName = (it.title || "").trim() || (it.name || "").trim();
          const applianceName =
            claimLabel ||
            (parsedTitleOrName && !CLAIM_ID_RE.test(parsedTitleOrName) ? parsedTitleOrName : "") ||
            parsedTitleOrName ||
            "";
          const itemNo = it.itemNo || String(idx + 1);
          const useClaimData = Boolean(compactLayout && claimForChat && idx === 0);
          const decision = (useClaimData && (claimForChat?.decision != null && claimForChat.decision !== ""))
            ? claimForChat.decision
            : (it.decision || "");
          const amountItem = useClaimData ? getItemForClaimAmounts(claimForChat, it) : it;
          const isCollapsible = !compactLayout;

          const headerRow = (
            <div className="ifa_item_header_row">
              <div className="ifa_item_summary_left">
                {applianceName ? <div className="ifa_item_name">{applianceName}</div> : null}
              </div>
              {!hideSummaryDecisionAmount ? (
                <div className="ifa_item_summary_right">
                  <div className="ifa_item_meta">
                    <span className="label">Decision</span>
                    <span className="value">
                      <DecisionBadge decision={decision} />
                    </span>
                  </div>
                  <div className="ifa_item_meta ifa_item_meta_amount">
                    <span className="label">Amount</span>
                    <span className="value">
                      <AmountDisplay item={amountItem} />
                    </span>
                  </div>
                  {isCollapsible ? (
                    <span className="ifa_item_chevron" aria-hidden="true">
                      ▾
                    </span>
                  ) : null}
                </div>
              ) : null}
            </div>
          );

          const bodyContent = (
            <div className="ifa_item_body">
              <div className="ifa_two_col">
                {Array.isArray(it.componentItems) && it.componentItems.length > 0 && !useClaimData ? (
                  <div className="ifa_row ifa_row_block">
                    <div className="k">
                      <strong>Items</strong>
                    </div>
                    <div className="v">
                      <ul className="ifa_component_items">
                        {it.componentItems.map((ci, ciIdx) => (
                          <li key={ciIdx}>
                            <strong>{ci.name || "—"}</strong>
                            {ci.details ? `: ${ci.details}` : ""}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                ) : null}
                {compactLayout ? (
                  (() => {
                    const notProvidedRe = /^(not\s+provided|not\s+applicable|not\s+available|n\/a|na|not\s+stated|—|-|none)$/i;
                    const isNotProvidedOrNa = (line) => {
                      const s = String(line || "").trim();
                      const afterColon = s.replace(/^[^:]+:\s*/, "").trim();
                      return !s || notProvidedRe.test(s) || notProvidedRe.test(afterColon);
                    };
                    const filtered = (it.moneyReconciliation || []).filter(
                      (line) => !isNotProvidedOrNa(line),
                    );
                    if (filtered.length === 0)
                      return (
                        <div className="ifa_row">
                          <div className="k">
                            <strong>Amount</strong>
                          </div>
                          <div className="v">
                            <AmountDisplay item={it} />
                          </div>
                        </div>
                      );
                    return (
                      <div className="ifa_row ifa_row_block ifa_row_money_recon">
                        <div className="k">
                          <strong>Money reconciliation</strong>
                        </div>
                        <div className="v">
                          <div className="ifa_money_recon">
                            {filtered.map((line, i) => {
                              let trimmed = String(line || "")
                                .trim()
                                .replace(/^[-•*]\s+/, "");
                              const isSectionHeader = /:\s*$/.test(trimmed);
                              const isKeyValue = /^[^:]+:\s*.+/.test(trimmed) && !isSectionHeader;
                              let label = "";
                              let value = "";
                              if (isKeyValue) {
                                const idx = trimmed.indexOf(":");
                                if (idx !== -1) {
                                  label = trimmed.slice(0, idx + 1);
                                  value = trimmed.slice(idx + 1).trim();
                                  if (/cannot|cannot\s+determine|can\s+not\s+determine|undetermined/i.test(value))
                                    value = "$0";
                                }
                              } else {
                                if (/cannot|cannot\s+determine|can\s+not\s+determine|undetermined/i.test(trimmed))
                                  trimmed = trimmed.replace(
                                    /\b(cannot|cannot\s+determine|can\s+not\s+determine|undetermined)\b/gi,
                                    "$0",
                                  );
                              }
                              return (
                                <div
                                  key={i}
                                  className={
                                    isSectionHeader
                                      ? "ifa_money_recon_header"
                                      : isKeyValue
                                        ? "ifa_money_recon_kv"
                                        : "ifa_money_recon_line"
                                  }
                                >
                                  {isKeyValue ? (
                                    <>
                                      <span className="ifa_money_recon_label">{label}</span>
                                      <span className="ifa_money_recon_value">{value}</span>
                                    </>
                                  ) : (
                                    trimmed
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      </div>
                    );
                  })()
                ) : null}
                {useClaimData ? (
                  <>
                    {claimForChat.decisionSummary ? (
                      <div className="ifa_row">
                        <div className="k">
                          <strong>Summary</strong>
                        </div>
                        <div className="v">{claimForChat.decisionSummary}</div>
                      </div>
                    ) : null}
                    {claimForChat.amounts?.customer_out_of_pocket > 0 ? (
                      <div className="ifa_row">
                        <div className="k">
                          <strong>Customer responsibility</strong>
                        </div>
                        <div className="v">
                          ${Number(claimForChat.amounts.customer_out_of_pocket).toFixed(2)}
                        </div>
                      </div>
                    ) : null}
                    {claimForChat.authorization_code && (claimForChat.amounts?.company_total > 0 || claimForChat.amounts?.authorized_by_company > 0) ? (
                      <div className="ifa_row">
                        <div className="k">
                          <strong>Authorized</strong>
                        </div>
                        <div className="v">
                          ${Number(claimForChat.amounts?.company_total ?? claimForChat.amounts?.authorized_by_company ?? 0).toFixed(2)}
                          {claimForChat.authorization_code ? ` (Auth code: ${claimForChat.authorization_code})` : ""}
                        </div>
                      </div>
                    ) : null}
                    {Array.isArray(claimForChat.reasons) && claimForChat.reasons.length > 0 ? (
                      <div className="ifa_row">
                        <div className="k">
                          <strong>Why</strong>
                        </div>
                        <div className="v">
                          {claimForChat.reasons.map((r, i) => (
                            <div key={i} className="ifa_why_item">
                              {r}
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : null}
                    <div className="ifa_row">
                      <div className="k">
                        <strong>Next steps</strong>
                      </div>
                      <div className="v">
                        {Array.isArray(claimForChat.nextSteps) && claimForChat.nextSteps.length > 0
                          ? claimForChat.nextSteps.join(" ")
                          : "N/A"}
                      </div>
                    </div>
                    {Array.isArray(claimForChat.policyBasis) && claimForChat.policyBasis.length > 0 ? (
                      <div className="ifa_row">
                        <div className="k">
                          <strong>Policy basis</strong>
                        </div>
                        <div className="v">{claimForChat.policyBasis.join(" ")}</div>
                      </div>
                    ) : null}
                  </>
                ) : (
                  <>
                    <div className="ifa_row">
                      <div className="k">
                        <strong>Why</strong>
                      </div>
                      <div className="v">
                        {it.why?.length
                          ? it.why.map((w, i) => (
                              <div key={i} className="ifa_why_item">
                                {w}
                              </div>
                            ))
                          : "N/A"}
                      </div>
                    </div>
                    <div className="ifa_row">
                      <div className="k">
                        <strong>Next steps</strong>
                      </div>
                      <div className="v">
                        {it.nextSteps?.length ? it.nextSteps.join(" ") : "N/A"}
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>
          );

          if (isCollapsible) {
            return (
              <details
                className="ifa_item"
                key={`${itemNo}-${idx}`}
                open={idx === 0}
              >
                <summary className="ifa_item_summary">{headerRow}</summary>
                {bodyContent}
              </details>
            );
          }

          return (
            <div
              className="ifa_item ifa_item_expanded"
              key={`${itemNo}-${idx}`}
            >
              {headerRow}
              {bodyContent}
            </div>
          );
        })}
      </div>
    </div>
  );
};

const arrToText = (arr) => (Array.isArray(arr) ? arr.join("\n") : "");
// Preserve spaces while typing in edit mode (no trim/filter); trimming happens on save/serialize
const textToArr = (text) => String(text || "").split("\n");

/**
 * Editable form that mirrors ItemizedFinalAnswer layout. Same sections, but each value is an input/textarea.
 * Props: parsed = { items, overall, overallNextSteps }, onChange(parsed).
 */
export const ItemizedFinalAnswerEditable = ({
  parsed = {},
  onChange,
  asCard = true,
}) => {
  const items = Array.isArray(parsed.items) ? parsed.items : [];
  const updateItem = (idx, updates) => {
    const next = { ...parsed, items: [...(parsed.items || [])] };
    next.items[idx] = { ...(next.items[idx] || {}), ...updates };
    onChange(next);
  };

  if (items.length === 0) return null;

  return (
    <div
      className={`itemized_final_answer ${asCard ? "ifa_outer_card" : ""} ifa_editable`}
    >
      <div className="ifa_cards">
        {items.map((it, idx) => {
          const itemNo = it.itemNo || String(idx + 1);
          const itemName = String(it.title ?? it.name ?? "");

          return (
            <div className="ifa_card" key={`edit-${itemNo}-${idx}`}>
              <div className="ifa_item_header">
                <strong>{itemName ? `${itemName}` : `Item ${itemNo}`}</strong>
              </div>

              {/* Only: Claim Component, Amount (Customer and Company), Decision, Why, Next steps */}
              <div className="ifa_meta ifa_meta_item_name_first">
                <div className="row">
                  <div className="k">
                    <strong>Claim Component</strong>
                  </div>
                  <div className="v">
                    <input
                      type="text"
                      className="ifa_input"
                      value={itemName}
                      onChange={(e) =>
                        updateItem(idx, {
                          ...it,
                          title: e.target.value,
                          name: e.target.value,
                        })
                      }
                      placeholder="e.g. Evaporator Coil"
                    />
                  </div>
                </div>
              </div>

              <div className="ifa_keyfacts">
                <div className="ifa_keyfacts_row">
                  <div className="k">
                    <strong>Decision</strong>
                  </div>
                  <div className="v">
                    {(() => {
                      const raw = (it.decision || "").trim();
                      const normalized = /approv|accept/i.test(raw)
                        ? "Approved"
                        : /deny|reject|denied/i.test(raw)
                          ? "Denied"
                          : "Partial";
                      const decisionClass =
                        normalized === "Approved"
                          ? "ifa_decision_approved"
                          : normalized === "Denied"
                            ? "ifa_decision_denied"
                            : "ifa_decision_partial";
                      return (
                        <select
                          className={`ifa_select_decision ${decisionClass}`}
                          value={normalized}
                          onChange={(e) =>
                            updateItem(idx, { ...it, decision: e.target.value })
                          }
                          aria-label="Decision"
                        >
                          <option value="Approved">Approved</option>
                          <option value="Denied">Denied</option>
                          <option value="Partial">Partial</option>
                        </select>
                      );
                    })()}
                  </div>
                </div>
              </div>

              <div className="ifa_amounts ifa_amounts_top">
                <div className="h">
                  <strong>Amount (Customer and Company)</strong>
                </div>
                <div className="sub">
                  <div className="k">
                    <strong>Customer</strong>
                  </div>
                  <textarea
                    className="ifa_textarea"
                    rows={2}
                    value={arrToText(it.amountsCustomer)}
                    onChange={(e) =>
                      updateItem(idx, {
                        ...it,
                        amountsCustomer: textToArr(e.target.value),
                      })
                    }
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
                    onChange={(e) =>
                      updateItem(idx, {
                        ...it,
                        amountsCompany: textToArr(e.target.value),
                      })
                    }
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
                  onChange={(e) =>
                    updateItem(idx, { ...it, why: textToArr(e.target.value) })
                  }
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
                  onChange={(e) =>
                    updateItem(idx, {
                      ...it,
                      nextSteps: textToArr(e.target.value),
                    })
                  }
                  placeholder="Not specified"
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
