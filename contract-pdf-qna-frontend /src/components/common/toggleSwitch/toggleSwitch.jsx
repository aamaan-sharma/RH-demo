import React, { useState, useEffect } from "react";
import "./toggleSwitch.scss";

const ToggleSwitch = ({ setGptModel, selectedModel, userEmail, isfixed }) => {
  const allowedUsers = [
    "sagar.borana@mindstix.com",
    "hitanshu.machhi@mindstix.com",
    "ambikeshwar.singh@mindstix.com",
    "aditi.goyal@mindstix.com",
    "kartik.dabre@mindstix.com",
    "badal.oza@mindstix.com",
    "roshan@mindstix.com",
    "roshan.kulkarni@mindstix.com",
    "vaibhav.thakur@mindstix.com",
    "aamod.kale@mindstix.com",
    "saloni.luktuke@mindstix.com",
    "shrihari.eknathe@mindstix.com",
  ];
  const [isSearch, setIsSearch] = useState(selectedModel === "Search");

  useEffect(() => {
    setIsSearch(selectedModel === "Search");
  }, [selectedModel]);

  const toggleSwitch = (newModel) => {
    if (isfixed || !allowedUsers.includes(userEmail)) {
      return;
    }

    setIsSearch(newModel === "Search"); 
    setGptModel(newModel);
  };

  return (
    <div className="toggle-switch">
      <div
        className={`toggle-option ${isSearch ? "selected" : ""}`}
        onClick={() => toggleSwitch("Search")}
      >
        Search
      </div>
      <div
        className={`toggle-option ${!isSearch ? "selected" : ""} infer`}
        onClick={() => toggleSwitch("Infer")} 
      >
        Infer
      </div>
      <div className={`toggle-slider ${isSearch ? "left" : "right"}`} />
    </div>
  );
};

export default ToggleSwitch;
