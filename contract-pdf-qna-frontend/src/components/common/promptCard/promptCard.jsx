import React from "react";
import "./promptCard.scss";

const PromptCard = ({ title, description, setInput, input, icon }) => {
  const handleClick = () => {
    if (input === description) {
      setInput("");
    } else {
      setInput(description);
    }
  };
  return (
    <div
      className={`card_container ${input == description ? "selected" : ""}`}
      onClick={handleClick}
    >
      <div className="topic">{title}</div>
      <div className="prompt_info">{description}</div>
      <div className="icon_section">
        <img src={icon} alt="frame icon" />
      </div>
    </div>
  );
};

export default PromptCard;
