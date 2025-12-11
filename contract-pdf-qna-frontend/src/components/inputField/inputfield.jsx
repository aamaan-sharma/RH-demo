import React, { useEffect, useRef } from "react";
import MicrophoneIcon from "../../assets/microphone.svg";
import SendIcon from "../../assets/send.svg";
import "./InputField.scss";
import recordIcon from "../../assets/record.svg";

const InputField = ({
  listening,
  transcript,
  description,
  textareaRef,
  setDescription,
  handleInputEnter,
  onMicrophoneClick,
  handleEnter,
}) => {
  const cursorPositionRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.focus();
      if (cursorPositionRef.current !== null) {
        textareaRef.current.setSelectionRange(
          cursorPositionRef.current,
          cursorPositionRef.current
        );
      }
    }
  }, [description, textareaRef]);

  const handleTextChange = (event) => {
    const newDescription = event.target.value;
    setDescription(newDescription);

    const cursorPosition = event.target.selectionStart;
    cursorPositionRef.current = cursorPosition;
  };
  return (
    <div className="input-container">
      <div className="text-input-container">
        <textarea
          ref={textareaRef}
          name="description"
          value={listening ? transcript : description}
          onChange={(event) => {
            handleTextChange(event);
          }}
          rows={1}
          placeholder={
            listening ? "Listening..." : "Looking for something specific?"
          }
          onKeyDown={handleEnter}
        />
      </div>
      <div className={`microphone-button ${listening ? "activate" : ""}`}>
        {description.length && !listening ? (
          <img
            src={SendIcon}
            className="record-ellipse"
            alt="Ellipse Icon"
            onClick={handleInputEnter}
          />
        ) : (
          <img
            src={listening ? recordIcon : MicrophoneIcon}
            className="record-ellipse"
            alt="Ellipse Icon"
            onClick={onMicrophoneClick}
          />
        )}
      </div>
    </div>
  );
};

export default InputField;
