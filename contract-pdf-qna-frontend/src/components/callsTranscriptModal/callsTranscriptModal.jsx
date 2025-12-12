import React from "react";
import "./callsTranscriptModal.scss";

const CallsTranscriptModal = ({
  isOpen,
  onClose,
  transcripts,
  searchTerm,
  onSearchTermChange,
  statusFilter,
  onStatusFilterChange,
  onSelectTranscript,
  isLoading,
}) => {
  if (!isOpen) return null;

  return (
    <div className="calls_modal_backdrop">
      <div className="calls_modal">
        <div className="calls_modal_header">
          <div className="title">Add Transcript</div>
          <button type="button" className="close_button" onClick={onClose}>
            ✕
          </button>
        </div>
        <div className="calls_modal_controls">
          <input
            type="text"
            className="search_input"
            placeholder="Search transcripts by name"
            value={searchTerm}
            onChange={(e) => onSearchTermChange(e.target.value)}
          />
        </div>
        <div className="calls_modal_body">
          {isLoading ? (
            <div className="loading">Loading transcripts...</div>
          ) : transcripts.length === 0 ? (
            <div className="empty_state">No transcripts found.</div>
          ) : (
            <div className="transcript_grid">
              {transcripts.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className="transcript_card"
                  onClick={() => onSelectTranscript(item)}
                >
                  <div className="name">{item.name}</div>
                  <div className="meta">
                    <span>{item.stateName}</span>
                    <span>{item.contractType}</span>
                    <span>{item.planName}</span>
                  </div>
                  <div className={`status ${item.status}`}>{item.status}</div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CallsTranscriptModal;

