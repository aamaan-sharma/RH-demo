import React from "react";
import "./callsExistingConversationPopup.scss";

const CallsExistingConversationPopup = ({
  isOpen,
  onClose,
  onStartNew,
  onGoExisting,
}) => {
  if (!isOpen) return null;

  return (
    <div className="calls_existing_backdrop">
      <div className="calls_existing_modal">
        <div className="header">
          <div className="title">
            A conversation already exists for this transcript.
          </div>
          <button type="button" className="close_button" onClick={onClose}>
            ✕
          </button>
        </div>
        <div className="body">
          <p>
            Would you like to continue with the existing chat or start a new
            one?
          </p>
        </div>
        <div className="actions">
          <button
            type="button"
            className="secondary"
            onClick={onGoExisting}
          >
            Go to existing chat
          </button>
          <button type="button" className="primary" onClick={onStartNew}>
            Start new chat
          </button>
          <button type="button" className="ghost" onClick={onClose}>
            Exit
          </button>
        </div>
      </div>
    </div>
  );
};

export default CallsExistingConversationPopup;


