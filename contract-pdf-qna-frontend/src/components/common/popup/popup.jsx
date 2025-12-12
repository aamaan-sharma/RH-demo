import React, { useState, useRef } from "react";
import cancelIcon from "../../../assets/cancel.svg";
import "./popup.scss";

const Popup = ({
  popupRef,
  closePopup,
  feedbackResponse,
  setFeedbackResponse,
  submitFeedback,
}) => {
  const [selectedChip, setSelectedChip] = useState(""); // Track the selected chip
  const textareaRef = useRef(null); // Ref to focus the textarea

  const chipOptions = ["Doesn’t address my problem", "Inadequate", "Other"];

  const handleChipClick = (chip) => {
    setSelectedChip(chip);
    if (chip === "Other") {
      setFeedbackResponse("");
      if (textareaRef.current) {
        textareaRef.current.focus();
      }
    } else {
      setFeedbackResponse(chip);
    }
  };

  const isSubmitEnabled = feedbackResponse.trim() !== ""; // Check if the textarea has text

  const resetPopup = () => {
    setSelectedChip("");
    setFeedbackResponse(""); // Clear the feedback when closing the popup
  };

  const handleTextareaChange = (e) => {
    const text = e.target.value;
    setFeedbackResponse(text);

    // Reset the selected chip if the textarea is cleared
    if (text.trim() === "") {
      setSelectedChip("");
    }
  };

  return (
    <div className="popup_wrapper" ref={popupRef}>
      <div className="header">
        <div className="title">Why was it not helpful?</div>
        <img
          src={cancelIcon}
          alt="cancel icon"
          onClick={() => {
            resetPopup(); // Reset state when closing the popup
            closePopup();
          }}
        />
      </div>
      <div className="chip_section">
        {chipOptions.map((chip, index) => (
          <div
            key={index}
            className={`chips ${selectedChip === chip ? "selected" : ""}`}
            onClick={() => handleChipClick(chip)}
          >
            {chip}
          </div>
        ))}
      </div>
      <div className="textarea_section">
        <textarea
          ref={textareaRef} // Attach ref to the textarea
          className="form_control"
          placeholder={"Please share your feedback here.."}
          rows="3"
          value={feedbackResponse}
          onKeyDown={(e) => {
            if (e.key === "Enter" && isSubmitEnabled) {
              submitFeedback();
              resetPopup();
              closePopup();
            }
          }}
          onChange={handleTextareaChange} // Change handler for textarea
        />
      </div>
      <div
        className={`submit_button ${isSubmitEnabled ? "enabled" : ""}`}
        onClick={() => {
          if (isSubmitEnabled) {
            submitFeedback();
            resetPopup();
            closePopup();
          }
        }}
      >
        Submit
      </div>
    </div>
  );
};

export default Popup;
