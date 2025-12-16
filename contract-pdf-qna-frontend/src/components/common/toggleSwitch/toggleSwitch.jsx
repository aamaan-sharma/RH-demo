import React from "react";
import "./toggleSwitch.scss";

const ToggleSwitch = ({ setGptModel, selectedModel }) => {
  const toggleSwitch = (newModel) => {
    setGptModel(newModel);
  };

  return (
    <div className="toggle-switch">
      <div
        className={`toggle-slider position-${
          (selectedModel || "Search").toLowerCase()
        }`}
      />
      <div
        className={`toggle-option ${
          selectedModel === "Search" ? "selected" : ""
        }`}
        onClick={() => toggleSwitch("Search")}
      >
        Search
      </div>
      <div
        className={`toggle-option ${
          selectedModel === "Infer" ? "selected" : ""
        }`}
        onClick={() => toggleSwitch("Infer")}
      >
        Infer
      </div>
      <div
        className={`toggle-option ${
          selectedModel === "Calls" ? "selected" : ""
        }`}
        onClick={() => toggleSwitch("Calls")}
      >
        Claims
      </div>
    </div>
  );
};

export default ToggleSwitch;
