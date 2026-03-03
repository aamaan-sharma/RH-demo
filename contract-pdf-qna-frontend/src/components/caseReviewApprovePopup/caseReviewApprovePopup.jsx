import React, { useEffect, useState } from "react";
import "./caseReviewApprovePopup.scss";
import { ItemizedDecision } from "../common/itemizedDecision/itemizedDecision";
import {
  ItemizedFinalAnswer,
  ItemizedFinalAnswerEditable,
  buildSummaryFieldChanges,
  parseDraftSummary,
  serializeDraftSummary,
} from "../common/itemizedFinalAnswer/itemizedFinalAnswer";

/** Format ISO date string as DD/MM/YY XX:YY AM/PM (12hr local). */
const formatChangeLogDate = (isoString) => {
  if (!isoString || typeof isoString !== "string") return "—";
  const d = new Date(isoString);
  if (Number.isNaN(d.getTime())) return "—";
  const day = String(d.getDate()).padStart(2, "0");
  const month = String(d.getMonth() + 1).padStart(2, "0");
  const year = String(d.getFullYear()).slice(-2);
  let hours = d.getHours();
  const minutes = d.getMinutes();
  const ampm = hours >= 12 ? "PM" : "AM";
  hours = hours % 12 || 12;
  const time = `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")} ${ampm}`;
  return `${day}/${month}/${year} ${time}`;
};

/** Derive item name from change log entry changes, e.g. "Refrigerator (not cooling properly)". */
const getItemNameFromChanges = (changes) => {
  if (!Array.isArray(changes) || changes.length === 0) return "Summary";
  const itemChange = changes.find(
    (c) =>
      c?.fieldName === "Item" ||
      (typeof c?.fieldName === "string" && /^Item\s+\d+\s*-\s*Item$/i.test(c.fieldName.trim()))
  );
  const title = itemChange
    ? (itemChange.updatedValue ?? itemChange.previousValue ?? "").trim() || "Item"
    : "Summary";
  const situationChange = changes.find(
    (c) =>
      typeof c?.fieldName === "string" && /^Item\s+\d+\s*-\s*Situation$/i.test(c.fieldName.trim())
  );
  const situation = situationChange
    ? (situationChange.updatedValue ?? situationChange.previousValue ?? "").trim()
    : "";
  if (situation) return `${title} (${situation})`;
  return title || "Item(s) updated";
};

/** Strip "**Overall Next Steps:**" and leading "- " from Overall Next Step value so it renders as plain text. */
const cleanOverallNextStepDisplay = (value) => {
  if (value == null || value === "") return value;
  const s = String(value).trim();
  const lines = s.split("\n");
  const result = [];
  const labelRe = /^\s*(\*\*Overall Next Steps?\*\*:?\s*|Overall Next Step\s*:?\s*)$/i;
  for (const line of lines) {
    const trimmed = line.trim();
    if (labelRe.test(trimmed)) continue;
    const withoutBullet = trimmed.replace(/^\s*[-•*]\s+/, "");
    result.push(withoutBullet);
  }
  const out = result.join("\n").trim();
  return out || s;
};

/** Group changes by item for separate tables. Returns [{ heading, displayName, changes, stripPrefix }]. */
const getChangesByItem = (changes) => {
  if (!Array.isArray(changes) || changes.length === 0) return [];
  const byKey = {};
  const seen = new Set();
  const otherChanges = [];
  for (const c of changes) {
    const name = (c?.fieldName || "").trim();
    const itemMatch = name.match(/^(Item\s+\d+)\s*-\s*(.+)$/i);
    if (itemMatch) {
      const heading = itemMatch[1];
      if (!byKey[heading]) {
        byKey[heading] = { heading, changes: [], stripPrefix: `${itemMatch[1]} - ` };
        seen.add(heading);
      }
      byKey[heading].changes.push(c);
    } else if (name === "Overall Next Step") {
      const heading = "Overall Next Step";
      if (!byKey[heading]) {
        byKey[heading] = { heading, changes: [], stripPrefix: "" };
        seen.add(heading);
      }
      byKey[heading].changes.push(c);
    } else {
      otherChanges.push(c);
    }
  }
  const itemOrder = [...seen].sort((a, b) => {
    if (a === "Overall Next Step") return 1;
    if (b === "Overall Next Step") return -1;
    const numA = parseInt(a.replace(/\D/g, ""), 10) || 0;
    const numB = parseInt(b.replace(/\D/g, ""), 10) || 0;
    return numA - numB;
  });
  const groups = itemOrder.map((key) => byKey[key]);
  if (otherChanges.length > 0) {
    groups.push({ heading: "Summary", displayName: "", changes: otherChanges, stripPrefix: "" });
  }
  groups.forEach((g) => {
    if (g.displayName !== undefined) return;
    if (g.heading === "Overall Next Step") {
      const c = g.changes[0];
      const prev = (c?.previousValue ?? "").trim().toLowerCase();
      const curr = (c?.updatedValue ?? "").trim().toLowerCase();
      const isNotSpecified =
        (prev === "not specified" || prev === "" || prev === "—") &&
        (curr === "not specified" || curr === "" || curr === "—");
      g.displayName = isNotSpecified ? "" : "Overall Next Step";
    } else if (/^Item\s+\d+$/i.test(g.heading)) {
      const itemChange = g.changes.find((c) =>
        (c?.fieldName || "").trim().endsWith(" - Item"),
      );
      const name = (itemChange?.updatedValue || itemChange?.previousValue || "").trim();
      g.displayName = name || g.heading;
    } else {
      g.displayName = "";
    }
  });
  return groups;
};

const decisionToneClass = (decision) => {
  const raw = String(decision || "");
  const slug = raw
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");

  const positive = ["approve", "approved", "accept", "accepted", "yes", "covered"];
  const negative = ["deny", "denied", "reject", "rejected", "no", "not_covered"];
  const review = [
    "cannot_determine",
    "cant_determine",
    "unknown",
    "indeterminate",
    "needs_review",
    "review",
    "partial",
    "maybe",
  ];

  const tone = positive.includes(slug)
    ? "positive"
    : negative.includes(slug)
      ? "negative"
      : review.includes(slug)
        ? "review"
        : "neutral";

  return `decision_tone_${tone}`;
};

const CaseReviewApprovePopup = ({
  isOpen,
  onClose,
  onApprove,
  onReject,
  caseId,
  transcriptName,
  caseName,
  metadata = {},
  decision,
  aiFinalDraft = "",
  authorizedAnswer = "",
  setAuthorizedAnswer,
  isApproving = false,
  isRejecting = false,
  isClosed = false,
  userName = "",
  caseDisposition = "",
  summaryEditLog = [],
  onSaveDraftSummary,
}) => {
  if (!isOpen) return null;

  const [comments, setComments] = useState("");
  const [isEditMode, setIsEditMode] = useState(false);
  const [editableParsed, setEditableParsed] = useState(null);
  const [editableDraftText, setEditableDraftText] = useState("");
  const [isSavingDraft, setIsSavingDraft] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setEditableParsed(null);
      setEditableDraftText("");
      setIsEditMode(false);
    }
  }, [isOpen, aiFinalDraft]);

  // Keep comments blank every time the modal is opened.
  useEffect(() => {
    if (isOpen) setComments("");
  }, [isOpen]);

  const canApprove =
    !isEditMode &&
    !isClosed &&
    !isApproving &&
    !isRejecting &&
    typeof onApprove === "function" &&
    Boolean((authorizedAnswer || aiFinalDraft || "").trim());

  const canReject =
    !isEditMode &&
    !isClosed &&
    !isApproving &&
    !isRejecting &&
    typeof onReject === "function";

  const handleEdit = () => {
    const parsed = parseDraftSummary(aiFinalDraft || "");
    if (Array.isArray(parsed.items) && parsed.items.length > 0) {
      setEditableParsed(parsed);
      setEditableDraftText("");
    } else {
      setEditableParsed(null);
      setEditableDraftText(aiFinalDraft || "");
    }
    setIsEditMode(true);
  };

  const handleSaveDraft = async () => {
    if (typeof onSaveDraftSummary !== "function") return;
    const previous = (aiFinalDraft || "").trim();
    const updated =
      editableParsed && editableParsed.items?.length > 0
        ? serializeDraftSummary(editableParsed).trim()
        : (editableDraftText || "").trim();
    if (previous === updated) {
      setIsEditMode(false);
      setEditableParsed(null);
      return;
    }
    const changes = buildSummaryFieldChanges(previous, updated);
    setIsSavingDraft(true);
    try {
      await onSaveDraftSummary(previous, updated, changes);
      setIsEditMode(false);
      setEditableParsed(null);
    } finally {
      setIsSavingDraft(false);
    }
  };

  const handleCancelEdit = () => {
    setEditableParsed(null);
    setEditableDraftText("");
    setIsEditMode(false);
  };

  const hasEditableItems = editableParsed && Array.isArray(editableParsed.items) && editableParsed.items.length > 0;
  const displaySummary = aiFinalDraft || "";

  return (
    <div
      className="case_review_backdrop"
      role="dialog"
      aria-modal="true"
      onClick={onClose}
    >
      <div className="case_review_modal" onClick={(e) => e.stopPropagation()}>
        <div className="header">
          <div className="title">Review & Proceed</div>
          <button type="button" className="close" onClick={onClose}>
            ✕
          </button>
        </div>

        <div className="body">
          <div className="section">
            <div className="section_title">Case</div>
            <div className="meta_grid">
              <div className="meta_item">
                <div className="k">Case ID</div>
                <div className="v">{caseName || transcriptName || caseId || "—"}</div>
              </div>
              <div className="meta_item">
                <div className="k">State</div>
                <div className="v">{metadata?.state || "—"}</div>
              </div>
              <div className="meta_item">
                <div className="k">Contract</div>
                <div className="v">{metadata?.contractType || "—"}</div>
              </div>
              <div className="meta_item">
                <div className="k">Plan</div>
                <div className="v">{metadata?.plan || "—"}</div>
              </div>
              <div className="meta_item">
                <div className="k">Status</div>
                <div className="v">{isClosed ? "Closed" : "Open"}</div>
              </div>
              {caseDisposition ? (
                <div className="meta_item">
                  <div className="k">Disposition</div>
                  <div className="v">{String(caseDisposition).toUpperCase()}</div>
                </div>
              ) : null}
            </div>
          </div>

          <div className="section">
            <div className="section_title_row">
              <span className="section_title">Final analyzed answer</span>
              {!isEditMode && aiFinalDraft && typeof onSaveDraftSummary === "function" && !isClosed ? (
                <button type="button" className="edit_summary_btn" onClick={handleEdit}>
                  Edit
                </button>
              ) : null}
            </div>
            <div className="hint">
              This is the structured summary you will proceed with and forward.
            </div>
            {displaySummary || isEditMode ? (
              <div className="ai_draft">
                <div className="ai_label">AI draft summary</div>
                {isEditMode ? (
                  <>
                    {hasEditableItems ? (
                      <div className="ai_text">
                        <ItemizedFinalAnswerEditable
                          parsed={editableParsed}
                          onChange={setEditableParsed}
                          asCard={true}
                        />
                      </div>
                    ) : (
                      <div className="ai_text ai_draft_edit_wrapper">
                        <textarea
                          className="ai_draft_textarea"
                          rows={14}
                          value={editableDraftText}
                          onChange={(e) => setEditableDraftText(e.target.value)}
                          placeholder="Edit the AI draft summary…"
                          aria-label="AI draft summary"
                        />
                      </div>
                    )}
                    <div className="save_draft_row">
                      <button
                        type="button"
                        className="cancel_draft_btn"
                        onClick={handleCancelEdit}
                        disabled={isSavingDraft}
                      >
                        Cancel
                      </button>
                      <button
                        type="button"
                        className="save_draft_btn"
                        onClick={handleSaveDraft}
                        disabled={isSavingDraft}
                      >
                        {isSavingDraft ? "Saving…" : "Save"}
                      </button>
                    </div>
                  </>
                ) : (
                  <div className="ai_text">
                    <ItemizedFinalAnswer text={displaySummary} title="" asCard={true} />
                  </div>
                )}
              </div>
            ) : null}
            {summaryEditLog && summaryEditLog.length > 0 ? (
              <div className="change_log_section">
                <div className="change_log_title">Change Log</div>
                <div className="change_log_list">
                  {[...summaryEditLog].reverse().map((entry, idx) => {
                    const changes = entry.changes || [];
                    if (changes.length === 0) return null;
                    return (
                      <div className="change_log_entry" key={idx}>
                        <div className="change_log_item_header">
                          <div className="change_log_header_row">
                            <span className="change_log_date">{formatChangeLogDate(entry.editedAt)}</span>
                            <span className="change_log_user">{entry.editedBy || "—"}</span>
                          </div>
                        </div>
                        {getChangesByItem(changes).map((group, gIdx) => (
                          <div className="change_log_table_group" key={gIdx}>
                            {group.displayName ? (
                              <div className="change_log_table_heading">{group.displayName}</div>
                            ) : null}
                            <table className="change_log_table">
                              <thead>
                                <tr>
                                  <th className="change_log_th change_log_th_field">Field</th>
                                  <th className="change_log_th change_log_th_prev">Previous</th>
                                  <th className="change_log_th change_log_th_curr">Current</th>
                                </tr>
                              </thead>
                              <tbody>
                                {group.changes.map((c, i) => {
                                  const displayField = group.stripPrefix
                                    ? (c.fieldName || "").replace(group.stripPrefix, "")
                                    : (c.fieldName || "—");
                                  const isOverallNextStep =
                                    (c.fieldName || "").trim() === "Overall Next Step";
                                  const prevDisplay = isOverallNextStep
                                    ? cleanOverallNextStepDisplay(c.previousValue)
                                    : c.previousValue;
                                  const currDisplay = isOverallNextStep
                                    ? cleanOverallNextStepDisplay(c.updatedValue)
                                    : c.updatedValue;
                                  return (
                                    <tr key={i}>
                                      <td className="change_log_td change_log_td_field">
                                        {displayField || "—"}
                                      </td>
                                      <td className="change_log_td change_log_td_prev">
                                        {prevDisplay ?? "—"}
                                      </td>
                                      <td className="change_log_td change_log_td_curr">
                                        {currDisplay ?? "—"}
                                      </td>
                                    </tr>
                                  );
                                })}
                              </tbody>
                            </table>
                          </div>
                        ))}
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : null}
            <div className="comments_header">
              <div className="label">Comments</div>
            </div>
            <textarea
              className="authorized_textarea"
              rows={8}
              value={comments}
              onChange={(e) => setComments(e.target.value)}
              placeholder="Add comments (optional)…"
              disabled={isClosed}
            />
          </div>
        </div>

        <div className="footer">
          <button type="button" className="secondary" onClick={onClose}>
            Exit
          </button>
          <button
            type="button"
            className="danger"
            onClick={() => onReject?.(comments)}
            disabled={!canReject}
            title={isClosed ? "Case is already closed." : isEditMode ? "Save or exit edit mode first." : ""}
          >
            {isRejecting ? "Rejecting…" : "Reject & Proceed"}
          </button>
          <button
            type="button"
            className="primary"
            onClick={() => onApprove?.(comments)}
            disabled={!canApprove}
            title={isClosed ? "Case is already closed." : isEditMode ? "Save or exit edit mode first." : ""}
          >
            {isApproving ? "Approving…" : "Approve & Proceed"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default CaseReviewApprovePopup;


