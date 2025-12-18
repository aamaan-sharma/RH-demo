import React, { useEffect, useRef } from "react";
import "./callsTranscriptModal.scss";
import { formatTranscriptDisplayName } from "../utils/transcriptName";
import TryAgainButton from "../common/tryAgainButton/tryAgainButton";

const CallsTranscriptModal = ({
  isOpen,
  onClose,
  transcripts,
  searchTerm,
  onSearchTermChange,
  statusFilter,
  onStatusFilterChange,
  onSelectTranscript,
  onToggleStatus,
  isLoading,
  isLoadingMore,
  hasMore,
  onLoadMore,
  error,
  onRetry,
}) => {
  if (!isOpen) return null;

  const bodyRef = useRef(null);
  const loadMoreLockRef = useRef(false);

  useEffect(() => {
    // Release lock after the "load more" request finishes so scrolling can trigger again.
    if (!isLoadingMore) {
      loadMoreLockRef.current = false;
    }
  }, [isLoadingMore]);

  useEffect(() => {
    // If the modal body isn't scrollable yet (first page fits), auto-load more pages until it is.
    // This makes the "scroll to load next 10" UX reachable without requiring a huge first page.
    if (!hasMore || isLoading || isLoadingMore || !onLoadMore) return;
    const el = bodyRef.current;
    if (!el) return;

    const notScrollableYet = el.scrollHeight <= el.clientHeight + 20;
    if (notScrollableYet && !loadMoreLockRef.current) {
      loadMoreLockRef.current = true;
      onLoadMore();
    }
  }, [transcripts?.length, hasMore, isLoading, isLoadingMore, onLoadMore]);

  const handleScroll = (e) => {
    if (!hasMore || isLoading || isLoadingMore) return;
    const el = e.currentTarget;
    const nearBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 40;
    if (nearBottom && onLoadMore && !loadMoreLockRef.current) {
      loadMoreLockRef.current = true;
      onLoadMore();
    }
  };

  return (
    <div className="calls_modal_backdrop">
      <div className="calls_modal">
        <div className="calls_modal_header">
          <div className="title">Select a Case</div>
          <button type="button" className="close_button" onClick={onClose}>
            ✕
          </button>
        </div>
        <div className="calls_modal_controls">
          <input
            type="text"
            className="search_input"
            placeholder="Search cases"
            value={searchTerm}
            onChange={(e) => onSearchTermChange(e.target.value)}
          />
        </div>
        <div className="calls_modal_body" ref={bodyRef} onScroll={handleScroll}>
          {isLoading ? (
            <div className="loading">
              <div className="spinner" aria-hidden="true" />
              <div className="loading_text">Loading cases...</div>
            </div>
          ) : error ? (
            <div className="error_state">
              <div className="error_text">Failed to load cases. Please try again.</div>
              <TryAgainButton onRetry={onRetry} />
            </div>
          ) : transcripts.length === 0 ? (
            <div className="empty_state">No cases found.</div>
          ) : (
            <>
              <div className="transcript_grid">
                {transcripts.map((item) => {
                  const display = formatTranscriptDisplayName(item?.name);
                  return (
                    <button
                      key={item.id}
                      type="button"
                      className="transcript_card"
                      onClick={() => onSelectTranscript(item)}
                    >
                      <div className="name" title={display.raw || item.name}>
                        <div className="primary">{display.primary || item.name}</div>
                        {display.secondary ? (
                          <div className="secondary">{display.secondary}</div>
                        ) : null}
                      </div>
                      <div className="meta">
                        <span>{item.stateName}</span>
                        <span>{item.contractType}</span>
                        <span>{item.planName}</span>
                      </div>
                      {item.status === "inactive" ? (
                        <div
                          className="status archived"
                          title="Click to toggle status"
                          onClick={(e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            onToggleStatus && onToggleStatus(item);
                          }}
                        >
                          Archived
                        </div>
                      ) : null}
                    </button>
                  );
                })}
              </div>
              {isLoadingMore ? (
                <div className="loading_more">
                  <div className="spinner" aria-hidden="true" />
                  <div className="loading_text">Loading more…</div>
                </div>
              ) : null}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default CallsTranscriptModal;

