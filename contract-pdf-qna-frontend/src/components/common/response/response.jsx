import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import responseIcon from "../../../assets/response.svg";
import responseBlueIcon from "../../../assets/response_blue.svg";
import documentsIcon from "../../../assets/documents.svg";
import thumbsDownIcon from "../../../assets/thumbs_down.svg";
import shareIcon from "../../../assets/share.svg";
import { ItemizedFinalAnswer } from "../itemizedFinalAnswer/itemizedFinalAnswer";
import TryAgainButton from "../tryAgainButton/tryAgainButton";
import Popup from "../popup/popup";
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
        </strong>
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
        </div>
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
      </ul>
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

const Response = ({
  response,
  chatId,
  conversationId,
  chats,
  setChats,
  relevantChunks = [],
  variant = "default",
  headerLabel,
  tone = "default", // default | blue
  isError = false,
  onRetry = null,
}) => {
  const navigate = useNavigate();
  const popupRef = useRef(null);
  
  // State for action icons
  const [showFeedbackPopup, setShowFeedbackPopup] = useState(false);
  const [feedbackResponse, setFeedbackResponse] = useState("");
  const [copiedToClipboard, setCopiedToClipboard] = useState(false);

  const isLoading = response === "Loading Response";
  const isErrorState = isError || (response && response.includes("Please try again"));
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
        "noopener,noreferrer"
      );
    }
  };

  // Handle feedback icon click - toggle popup
  const handleFeedbackClick = () => {
    setShowFeedbackPopup((prev) => !prev);
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
    
    // TODO: Send feedback to backend API
    console.log("Feedback submitted:", {
      chatId,
      conversationId,
      feedback: feedbackResponse,
    });
    
    // Reset and close popup
    setFeedbackResponse("");
    setShowFeedbackPopup(false);
  };

  return (
    <div
      className={`response_wrapper ${variant === "finalAnswer" ? "final_answer" : ""} ${
        isBlue ? "tone_blue" : ""
      }`}
    >
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
            headerLabel || (variant === "finalAnswer" ? "Final Answer (AI)" : "Generated by AI")
          )}
        </div>
        {!isLoading && <div className="line"></div>}
      </div>

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
              <div className="chunks_title">Contract clauses (referred info)</div>
              <div className="chunks_list">
                {relevantChunks.map((chunk, index) => {
                  const score =
                    chunk && typeof chunk === "object" ? chunk.score : undefined;
                  const titleSuffix =
                    score !== undefined && score !== null
                      ? ` (score: ${score})`
                      : "";
                  const metadata =
                    chunk && typeof chunk === "object"
                      ? chunk.metadata || chunk.meta || undefined
                      : undefined;
                  const source =
                    metadata && typeof metadata === "object"
                      ? metadata.source || metadata.file || metadata.document || metadata.doc || ""
                      : "";
                  const additional =
                    metadata && typeof metadata === "object"
                      ? metadata.section ||
                        metadata.heading ||
                        metadata.title ||
                        metadata.page ||
                        metadata.clause ||
                        ""
                      : "";
                  const refParts = [source, additional].filter(Boolean);
                  const refSuffix = refParts.length ? ` — ${refParts.join(" · ")}` : "";
                  const content =
                    chunk && typeof chunk === "object"
                      ? chunk.content || JSON.stringify(chunk, null, 2)
                      : String(chunk);
                  return (
                    <details className="chunk_item" key={index}>
                      <summary className="chunk_summary">
                        <span className="chunk_summary_text">
                        {`Clause ${index + 1}${titleSuffix}${refSuffix}`}
                        </span>
                        <button
                          type="button"
                          className="chunk_close"
                          aria-label={`Close Clause ${index + 1}`}
                          title="Close"
                          onClick={(e) => {
                            // Don't toggle the <details> via the <summary> click.
                            e.preventDefault();
                            e.stopPropagation();
                            const detailsEl = e.currentTarget?.closest?.("details");
                            if (detailsEl) detailsEl.open = false;
                          }}
                        >
                          ×
                        </button>
                      </summary>
                      <pre className="chunk_content">{content}</pre>
                    </details>
                  );
                })}
              </div>
            </div>
          ) : null}

          {/* Action Icons - Reference, Feedback, Share */}
          {variant !== "finalAnswer" && (
            <div className="icon_wrapper">
              {/* Reference Icon - View referred clauses */}
              <div 
                className="icon_container" 
                onClick={handleReferenceClick}
                title="View referred clauses"
              >
                <img src={documentsIcon} alt="Reference clauses" />
              </div>

              {/* Feedback Icon - Report unhelpful response */}
              <div 
                className={`icon_container ${showFeedbackPopup ? "active" : ""}`}
                onClick={handleFeedbackClick}
                title="Report feedback"
              >
                <img src={thumbsDownIcon} alt="Feedback" />
              </div>

              {/* Share Icon - Copy to clipboard */}
              <div 
                className={`icon_container ${copiedToClipboard ? "active" : ""}`}
                onClick={handleShareClick}
                title={copiedToClipboard ? "Copied!" : "Copy response"}
              >
                <img src={shareIcon} alt="Share" />
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
            />
          )}
        </>
      )}
    </div>
  );
};

export default Response;
