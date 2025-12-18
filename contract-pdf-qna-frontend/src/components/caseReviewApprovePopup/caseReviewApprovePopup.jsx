import React, { useEffect, useState } from "react";
import "./caseReviewApprovePopup.scss";
import { ItemizedDecision } from "../common/itemizedDecision/itemizedDecision";
import { ItemizedFinalAnswer } from "../common/itemizedFinalAnswer/itemizedFinalAnswer";

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
  caseId,
  transcriptName,
  caseName,
  metadata = {},
  decision,
  aiFinalDraft = "",
  authorizedAnswer = "",
  setAuthorizedAnswer,
  isApproving = false,
  isClosed = false,
  userName = "",
}) => {
  if (!isOpen) return null;

  const [comments, setComments] = useState("");

  // Keep comments blank every time the modal is opened.
  useEffect(() => {
    if (isOpen) setComments("");
  }, [isOpen]);

  const canProceed =
    !isClosed &&
    !isApproving &&
    typeof onApprove === "function" &&
    Boolean((authorizedAnswer || aiFinalDraft || "").trim());

  return (
    <div className="case_review_backdrop" role="dialog" aria-modal="true">
      <div className="case_review_modal">
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
            </div>
          </div>

          <div className="section">
            <div className="section_title">Final authorized answer</div>
            <div className="hint">
              This is the structured summary you will proceed with and forward.
            </div>
            {aiFinalDraft ? (
              <div className="ai_draft">
                <div className="ai_label">AI draft summary</div>
                <div className="ai_text">
                  <ItemizedFinalAnswer text={aiFinalDraft} title="" asCard={true} />
                </div>
              </div>
            ) : null}
            <div className="comments_header">
              <div className="label">Comments</div>
              {userName ? <div className="author">{userName}</div> : null}
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
            className="primary"
            onClick={onApprove}
            disabled={!canProceed}
            title={isClosed ? "Case is already closed." : ""}
          >
            {isApproving ? "Proceeding…" : "Proceed & Close Case"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default CaseReviewApprovePopup;


