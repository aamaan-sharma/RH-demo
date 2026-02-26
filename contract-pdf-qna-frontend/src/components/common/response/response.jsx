import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import responseIcon from "../../../assets/response.svg";
import responseBlueIcon from "../../../assets/response_blue.svg";
import documentsIcon from "../../../assets/documents.svg";
import thumbsDownIcon from "../../../assets/thumbs_down.svg";
import thumbsUpIcon from "../../../assets/thumbs_up.svg";
import copyIcon from "../../../assets/copy.svg";
import { ItemizedFinalAnswer } from "../itemizedFinalAnswer/itemizedFinalAnswer";
import TryAgainButton from "../tryAgainButton/tryAgainButton";
import Popup from "../popup/popup";
import { API_BASE_URL } from "../../../config";
import { getIdToken } from "../../../utils/authStorage";
import "./response.scss";

const renderInlineBold = (text) => {
  const s = String(text ?? "");
  if (!s.includes("**")) return s;

  // Split on **...** pairs and render <strong> for bold segments.
  const parts = s.split("**");
  const out = [];
  for (let i = 0; i < parts.length; i++) {
    const chunk = parts[i];
    if (!chunk) continue;
    if (i % 2 === 1) {
      out.push(
        <strong key={`b-${i}`} className="inline_bold">
          {chunk}
        </strong>,
      );
    } else {
      out.push(<React.Fragment key={`t-${i}`}>{chunk}</React.Fragment>);
    }
  }
  return out;
};

const renderResponseContent = (response) => {
  const raw = String(response ?? "");
  if (!raw) return null;

  const lines = raw.replace(/\r\n/g, "\n").split("\n");

  const blocks = [];
  let paraLines = [];
  let listItems = [];

  const flushPara = () => {
    if (paraLines.length === 0) return;
    const text = paraLines.join("\n").trimEnd();
    if (text) {
      blocks.push(
        <div key={`p-${blocks.length}`} className="resp_paragraph">
          {text.split("\n").map((ln, idx) => (
            <React.Fragment key={idx}>
              {idx > 0 ? <br /> : null}
              {renderInlineBold(ln)}
            </React.Fragment>
          ))}
        </div>,
      );
    }
    paraLines = [];
  };

  const flushList = () => {
    if (listItems.length === 0) return;
    blocks.push(
      <ul key={`ul-${blocks.length}`} className="resp_bullets">
        {listItems.map((li, idx) => (
          <li key={idx}>{renderInlineBold(li)}</li>
        ))}
      </ul>,
    );
    listItems = [];
  };

  for (const line of lines) {
    const trimmed = line.trim();
    const isBullet = /^-\s+/.test(trimmed);

    if (isBullet) {
      // Switch from paragraph to list mode.
      flushPara();
      listItems.push(trimmed.replace(/^-\s+/, ""));
      continue;
    }

    if (trimmed === "") {
      // Blank line breaks blocks.
      flushPara();
      flushList();
      continue;
    }

    // Normal paragraph line.
    flushList();
    paraLines.push(line);
  }

  flushPara();
  flushList();

  // If nothing special, fall back to raw
  if (blocks.length === 0) return raw;
  return blocks;
};

const _asStr = (v) => (v === null || v === undefined ? "" : String(v));

const _prettyTitle = (s) => {
  const raw = _asStr(s).trim();
  if (!raw) return "";
  return raw.replace(/[_]+/g, " ").replace(/\s+/g, " ").trim();
};

const pickClauseData = (chunk) => {
  // Supports:
  // - string chunk (legacy)
  // - {content, metadata} (new detailed chunk)
  // - odd shapes: {pageContent}, {text}, etc.
  if (chunk && typeof chunk === "object") {
    const content = _asStr(
      chunk.content ||
        chunk.page_content ||
        chunk.pageContent ||
        chunk.text ||
        "",
    ).trim();
    const metadata =
      chunk.metadata && typeof chunk.metadata === "object"
        ? chunk.metadata
        : chunk.meta && typeof chunk.meta === "object"
          ? chunk.meta
          : null;
    return { content, metadata };
  }
  return { content: _asStr(chunk).trim(), metadata: null };
};

const shortenUnit = (u) => {
  const s = _asStr(u).trim();
  if (!s) return "";
  if (s.length <= 64) return s;
  return `${s.slice(0, 40)}…${s.slice(-18)}`;
};

const parseUnitId = (unitIdRaw) => {
  // Expected pattern (example):
  //   Frontdoor Supplemental Coverage Documents.pdf::p232::3.E
  const raw = _asStr(unitIdRaw).trim();
  if (!raw) return { file: "", page: "", clause: "" };
  const parts = raw.split("::");
  const file = _asStr(parts[0] || "").trim();
  const pageToken = _asStr(parts[1] || "").trim();
  const clause = _asStr(parts[2] || "").trim();
  let page = "";
  const m = pageToken.match(/^p(\d+)$/i);
  if (m) page = m[1];
  return { file, page, clause };
};

const formatChunkReference = (metadata) => {
  if (!metadata || typeof metadata !== "object") return "";

  const src = metadata.source;

  // source can be either:
  // - string (older /history normalization)
  // - object (Milvus `source` JSON: {title,page_no,bbox,unit_id,clause_no,subclause_no,...})
  let title = "";
  let page = "";
  let clause = "";
  let unit = "";

  if (typeof src === "string") {
    title = _prettyTitle(src);
  } else if (src && typeof src === "object") {
    title = _prettyTitle(src.title);
    const p = src.page_no ?? src.page ?? src.pageNumber ?? src.page_no;
    if (p !== null && p !== undefined && _asStr(p).trim())
      page = `p.${_asStr(p).trim()}`;
    const c = _asStr(src.clause_no || src.clause || "").trim();
    const sc = _asStr(src.subclause_no || src.subclause || "").trim();
    if (c && sc)
      clause =
        sc.startsWith(".") || c.endsWith(".")
          ? `clause ${c}${sc}`
          : `clause ${c}.${sc}`;
    else if (c) clause = `clause ${c}`;
    unit = _asStr(src.unit_id || "").trim();
  }

  // Extra common metadata fallbacks
  if (!title)
    title = _prettyTitle(
      metadata.title || metadata.document || metadata.file || "",
    );

  const parts = [title, page, clause].filter(Boolean);
  const ref = parts.length ? parts.join(" · ") : "";
  // We display file/page/clause chips separately in the UI; keep this concise.
  return ref;
};

const Response = ({
  response,
  chatId,
  conversationId,
  chats,
  setChats,
  relevantChunks = [],
  variant = "default",
  headerLabel,
  hideHeader = false,
  tone = "default", // default | blue
  isError = false,
  onRetry = null,
  showReferenceIcon = true,
  showActions,
}) => {
  const navigate = useNavigate();
  const popupRef = useRef(null);

  // State for action icons
  const [showFeedbackPopup, setShowFeedbackPopup] = useState(false);
  const [feedbackResponse, setFeedbackResponse] = useState("");
  const [copiedToClipboard, setCopiedToClipboard] = useState(false);
  const [sendingReaction, setSendingReaction] = useState(false);
  const [feedbackMode, setFeedbackMode] = useState("down"); // "down" | "up"

  const existingReaction = (() => {
    try {
      const cid = String(chatId || "");
      if (!cid) return "";
      const found = (chats || []).find(
        (c) => String(c?.chat_id || c?.chatId || "") === cid,
      );
      return String(found?.reaction || "");
    } catch {
      return "";
    }
  })();
  const reactionLocked =
    existingReaction === "up" || existingReaction === "down";

  const isLoading = response === "Loading Response";
  const isErrorState =
    isError || (response && response.includes("Please try again"));
  const isBlue = tone === "blue";
  const headerIcon = isBlue ? responseBlueIcon : responseIcon;

  // Close feedback popup when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (popupRef.current && !popupRef.current.contains(event.target)) {
        setShowFeedbackPopup(false);
      }
    };
    if (showFeedbackPopup) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [showFeedbackPopup]);

  // Handle reference icon click - navigate to referred clauses page
  const handleReferenceClick = () => {
    if (conversationId && chatId) {
      window.open(
        `/conversation/${conversationId}/chat/${chatId}/referred-clauses`,
        "_blank",
        "noopener,noreferrer",
      );
    }
  };

  // Handle feedback icon click - toggle popup
  const handleFeedbackClick = () => {
    if (reactionLocked) return;
    setFeedbackMode("down");
    setShowFeedbackPopup((prev) => !prev);
  };

  const handleThumbsUpClick = () => {
    if (reactionLocked) return;
    setFeedbackMode("up");
    setShowFeedbackPopup(true);
  };

  const _setReactionLocal = (reaction) => {
    if (!setChats) return;
    const cid = String(chatId || "");
    if (!cid) return;
    setChats((prev) =>
      (prev || []).map((c) =>
        String(c?.chat_id || c?.chatId || "") === cid ? { ...c, reaction } : c,
      ),
    );
  };

  const sendReaction = async (reaction, detailText = "") => {
    const cid = String(conversationId || "");
    const chid = String(chatId || "");
    if (!cid || !chid) return;
    // Once a reaction exists (from /history or local submit), don't allow changes.
    if (reactionLocked) return;
    const token = getIdToken();
    if (!token) return;
    if (sendingReaction) return;

    setSendingReaction(true);
    try {
      await axios.post(
        `${API_BASE_URL}/feedback?conversation-id=${encodeURIComponent(cid)}&chat-id=${encodeURIComponent(chid)}`,
        {
          reaction,
          response: detailText || "",
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        },
      );
      _setReactionLocal(reaction);
    } catch (e) {
      console.error("Failed to send feedback reaction:", e);
    } finally {
      setSendingReaction(false);
    }
  };

  // Handle share icon click - copy response to clipboard
  const handleShareClick = async () => {
    try {
      await navigator.clipboard.writeText(response || "");
      setCopiedToClipboard(true);
      setTimeout(() => setCopiedToClipboard(false), 2000);
    } catch (err) {
      console.error("Failed to copy to clipboard:", err);
    }
  };

  // Submit feedback handler
  const submitFeedback = () => {
    if (!feedbackResponse.trim()) return;
    if (reactionLocked) return;
    // Feedback (up/down) with optional freeform text
    sendReaction(feedbackMode, feedbackResponse.trim());
    // Reset and close popup
    setFeedbackResponse("");
    setShowFeedbackPopup(false);
  };

  return (
    <div
      className={`response_wrapper ${variant === "finalAnswer" ? "final_answer" : ""} ${variant === "draftAnswer" ? "draft_answer" : ""} ${
        isBlue ? "tone_blue" : ""
      } ${hideHeader ? "response_no_header" : ""}`}
    >
      {!hideHeader ? (
        <div className="response_section">
          <img src={headerIcon} alt="response icon" />
          <div className="text">
            {isLoading ? (
              <div className="loading_header" aria-live="polite">
                <span className="label">Generating response</span>
                <span className="typing_dots" aria-hidden="true">
                  <span />
                  <span />
                  <span />
                </span>
              </div>
            ) : (
              headerLabel ||
              (variant === "finalAnswer"
                ? "Final Answer (AI)"
                : variant === "draftAnswer"
                  ? "AI Draft Answer"
                  : "Generated by AI")
            )}
          </div>
          {!isLoading && <div className="line"></div>}
        </div>
      ) : isLoading ? (
        <div className="response_section response_loading_only" aria-live="polite">
          <span className="label">Generating response</span>
          <span className="typing_dots" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
        </div>
      ) : null}

      {isLoading ? (
        <div className="response_loading_body" aria-hidden="true">
          <div className="skeleton_line w90" />
          <div className="skeleton_line w82" />
          <div className="skeleton_line w65" />
        </div>
      ) : (
        <>
          <div
            className={`response_text ${isErrorState ? "error_response" : ""}`}
          >
            {variant === "finalAnswer" ? (
              <ItemizedFinalAnswer text={response} title="" asCard={true} />
            ) : variant === "draftAnswer" ? (
              <ItemizedFinalAnswer
                text={response}
                title=""
                asCard={false}
                compactLayout={true}
                hideSummaryDecisionAmount={true}
              />
            ) : (
              renderResponseContent(response)
            )}
          </div>
          {isErrorState && onRetry && (
            <div className="error_actions">
              <TryAgainButton onRetry={onRetry} />
            </div>
          )}

          {Array.isArray(relevantChunks) && relevantChunks.length > 0 ? (
            <div className="chunks_wrapper">
              <div className="chunks_title">Referenced contract clauses</div>
              <div className="chunks_list">
                {relevantChunks.map((chunk, index) => {
                  const { content, metadata } = pickClauseData(chunk);
                  const ref = formatChunkReference(metadata);
                  const src =
                    metadata && typeof metadata === "object"
                      ? metadata.source
                      : null;
                  const pageNo =
                    src && typeof src === "object"
                      ? (src.page_no ?? src.page ?? src.pageNumber ?? null)
                      : null;
                  const clauseNo =
                    src && typeof src === "object"
                      ? _asStr(src.clause_no || src.clause || "").trim()
                      : "";
                  const subClauseNo =
                    src && typeof src === "object"
                      ? _asStr(src.subclause_no || src.subclause || "").trim()
                      : "";
                  const unitIdRaw =
                    src && typeof src === "object"
                      ? _asStr(src.unit_id || "").trim()
                      : "";
                  const unitParsed = parseUnitId(unitIdRaw);
                  const fileFromUnit = unitParsed.file;
                  const pageFromUnit = unitParsed.page;
                  const clauseFromUnit = unitParsed.clause;
                  const title =
                    src && typeof src === "object"
                      ? _prettyTitle(src.title || "")
                      : _prettyTitle(typeof src === "string" ? src : "");
                  const pageDisplay =
                    _asStr(pageNo).trim() || _asStr(pageFromUnit).trim();
                  const clauseDisplay = (() => {
                    if (clauseNo) {
                      if (!subClauseNo) return clauseNo;
                      return `${clauseNo}${subClauseNo.startsWith(".") ? "" : "."}${subClauseNo}`;
                    }
                    return _asStr(clauseFromUnit).trim();
                  })();

                  return (
                    <details className="chunk_item" key={index}>
                      <summary className="chunk_summary">
                        <span className="chunk_summary_text">{`Clause ${index + 1}`}</span>
                        <button
                          type="button"
                          className="chunk_close"
                          aria-label={`Close Clause ${index + 1}`}
                          title="Close"
                          onClick={(e) => {
                            // Don't toggle the <details> via the <summary> click.
                            e.preventDefault();
                            e.stopPropagation();
                            const detailsEl =
                              e.currentTarget?.closest?.("details");
                            if (detailsEl) detailsEl.open = false;
                          }}
                        >
                          ×
                        </button>
                      </summary>
                      <div className="chunk_body">
                        {ref ? <div className="chunk_ref">{ref}</div> : null}

                        <div
                          className="chunk_meta_row"
                          aria-label="Clause metadata"
                        >
                          {title ? (
                            <div className="meta_chip" title={title}>
                              <span className="k">Title</span>
                              <span className="v">{title}</span>
                            </div>
                          ) : null}
                          {fileFromUnit ? (
                            <div className="meta_chip" title={fileFromUnit}>
                              <span className="k">File</span>
                              <span className="v">{fileFromUnit}</span>
                            </div>
                          ) : null}
                          {pageDisplay ? (
                            <div className="meta_chip">
                              <span className="k">Page</span>
                              <span className="v">{pageDisplay}</span>
                            </div>
                          ) : null}
                          {clauseDisplay ? (
                            <div className="meta_chip">
                              <span className="k">Clause</span>
                              <span className="v">{clauseDisplay}</span>
                            </div>
                          ) : null}
                          {/* Keep full unit_id in tooltip for debugging (not shown in UI). */}
                          {unitIdRaw && !fileFromUnit ? (
                            <div className="meta_chip" title={unitIdRaw}>
                              <span className="k">Unit</span>
                              <span className="v">
                                {shortenUnit(unitIdRaw)}
                              </span>
                            </div>
                          ) : null}
                        </div>

                        <div
                          className="chunk_text"
                          aria-label={`Clause ${index + 1} text`}
                        >
                          {content || "(No clause text found)"}
                        </div>
                      </div>
                    </details>
                  );
                })}
              </div>
            </div>
          ) : null}

          {/* Action Icons - Reference, Feedback, Share */}
          {(typeof showActions === "boolean"
            ? showActions
            : variant !== "finalAnswer") && (
            <div className="icon_wrapper">
              {/* Reference Icon - View referred clauses */}
              {showReferenceIcon ? (
                <div
                  className="icon_container"
                  onClick={handleReferenceClick}
                  title="View referred clauses"
                >
                  <img src={documentsIcon} alt="Reference clauses" />
                </div>
              ) : null}

              {/* Thumbs Up Icon - Helpful */}
              <div
                className={`icon_container ${
                  existingReaction === "up" ||
                  (showFeedbackPopup && feedbackMode === "up")
                    ? "active selected"
                    : ""
                } ${reactionLocked ? "locked" : ""}`}
                onClick={() => {
                  if (sendingReaction || reactionLocked) return;
                  handleThumbsUpClick();
                }}
                title={
                  existingReaction === "up"
                    ? "Feedback submitted (helpful)"
                    : reactionLocked
                      ? "Feedback already submitted"
                      : "Mark helpful (add reason)"
                }
              >
                <img src={thumbsUpIcon} alt="Helpful" />
              </div>

              {/* Feedback Icon - Report unhelpful response */}
              <div
                className={`icon_container ${
                  showFeedbackPopup || existingReaction === "down"
                    ? "active selected"
                    : ""
                } ${reactionLocked ? "locked" : ""}`}
                onClick={() => {
                  if (sendingReaction || reactionLocked) return;
                  handleFeedbackClick();
                }}
                title={
                  existingReaction === "down"
                    ? "Feedback submitted (not helpful)"
                    : reactionLocked
                      ? "Feedback already submitted"
                      : "Report feedback"
                }
              >
                <img src={thumbsDownIcon} alt="Feedback" />
              </div>

              {/* Share Icon - Copy to clipboard */}
              <div
                className={`icon_container ${copiedToClipboard ? "active" : ""}`}
                onClick={handleShareClick}
                title={copiedToClipboard ? "Copied!" : "Copy response"}
              >
                <img src={copyIcon} alt="Copy" />
              </div>
            </div>
          )}

          {/* Feedback Popup */}
          {showFeedbackPopup && (
            <Popup
              popupRef={popupRef}
              closePopup={() => setShowFeedbackPopup(false)}
              feedbackResponse={feedbackResponse}
              setFeedbackResponse={setFeedbackResponse}
              submitFeedback={submitFeedback}
              title={
                feedbackMode === "up"
                  ? "Why was it helpful?"
                  : "Why was it not helpful?"
              }
              chipOptions={
                feedbackMode === "up"
                  ? ["Accurate", "Clear", "Relevant", "Other"]
                  : ["Doesn’t address my problem", "Inadequate", "Other"]
              }
              placeholder={
                feedbackMode === "up"
                  ? "Optional: share what was helpful…"
                  : "Please share your feedback here.."
              }
            />
          )}
        </>
      )}
    </div>
  );
};

export default Response;
