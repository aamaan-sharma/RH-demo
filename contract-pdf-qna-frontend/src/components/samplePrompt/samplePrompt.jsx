import React, { useState, useEffect } from "react";
import "./SamplePrompt.scss";
import shuffleIcon from "../../assets/shuffle.svg";
import { searchPrompts, inferPrompts } from "../../constant.js";
import PromptCard from "../common/promptCard/promptCard.jsx";

const shuffleArray = (array) => {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
};

const getRandomPrompts = (prompts, count) => {
  const shuffled = shuffleArray(prompts);
  return shuffled.slice(0, Math.min(count, prompts.length));
};

const SamplePrompt = ({ gptModel, input, setInput }) => {
  const [displayedPrompts, setDisplayedPrompts] = useState([]);

  const getAllPrompts = () => {
    if (gptModel === "Search") {
      return searchPrompts.flatMap((category) =>
        category.questions.map((question) => ({
          ...question,
          category_title: category.category_title,
          icon: category.icon,
        }))
      );
    } else if (gptModel === "Infer") {
      return inferPrompts.flatMap((category) =>
        category.questions.map((question) => ({
          ...question,
          category_title: category.category_title,
          icon: category.icon,
        }))
      );
    }
    return [];
  };

  const updatePrompts = () => {
    const allPrompts = getAllPrompts();
    const newPrompts = getRandomPrompts(allPrompts, 3);
    setDisplayedPrompts(newPrompts);
  };

  useEffect(() => {
    updatePrompts();
  }, [gptModel]);

  return (
    <div className="prompt_wrapper">
      <div className="title">What would you like to focus on today?</div>
      <div className="subtitle">
        Your AI-powered copilot is ready to assist you!
      </div>
      <div className="queries_part">
        <div className="query">Most Frequently Asked Customer Queries:</div>
        <div
          className="refresh_icon"
          onClick={updatePrompts}
          title="Refresh Prompts"
        >
          <img className="icon" src={shuffleIcon} alt="Refresh" />
        </div>
      </div>
      <div className="card_list">
        {displayedPrompts.length > 0 ? (
          displayedPrompts.map((item, index) => (
            <PromptCard
              key={`${item.text}-${index}`}
              title={item.category_title}
              description={item.text}
              icon={item.icon}
              setInput={setInput}
              input={input}
            />
          ))
        ) : (
          <p>No prompts available.</p>
        )}
      </div>
    </div>
  );
};

export default SamplePrompt;
