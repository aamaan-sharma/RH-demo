import React from "react";
import "./tryAgainButton.scss";

const TryAgainButton = ({ onRetry, className = "" }) => {
  return (
    <button
      type="button"
      className={`try_again_button ${className}`}
      onClick={onRetry}
    >
      Try Again
    </button>
  );
};

export default TryAgainButton;

