import React, { useEffect, useState } from "react";
import "./referredClauses.scss";
import { useLocation } from "react-router-dom";
import axios from "axios";
import { API_BASE_URL } from "../../config";

const ReferredClauses = () => {
  const location = useLocation();
  const conversationId = location.pathname.split("/")[2];
  const chatId = location.pathname.split("/")[4];
  const [state, setState] = useState("");
  const [contract, setContract] = useState("");
  const [plan, setPlan] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [clauses, setClauses] = useState("");
  const [gptModel, setGptModel] = useState("");
  const [wordCount, setWordCount] = useState();

  // Retrieve the token directly from session storage
  const bearerToken = sessionStorage.getItem("idToken");

  function removeMetadata(inputString) {
    const startIndex = inputString.indexOf("{'source':");
    if (startIndex === -1) {
      return inputString;
    }

    const endIndex = inputString.indexOf("}), Document(page_content=");
    if (endIndex === -1) {
      return inputString;
    }

    return (
      inputString.substring(0, startIndex) +
      inputString.substring(endIndex + 27)
    );
  }

  useEffect(() => {
    const apiUrl = `${API_BASE_URL}/referred-clauses?conversation-id=${conversationId}&chat-id=${chatId}`;

    axios
      .get(apiUrl, {
        headers: {
          Authorization: `Bearer ${bearerToken}`,
        },
      })
      .then((response) => {
        if (
          response.data.message === "Token is invalid" ||
          response.data.message === "Token has expired" ||
          response.data.message === "Token is missing"
        ) {
          return;
        }
        setClauses(response.data.referredClauses);
        setState(response.data.selectedState);
        setContract(response.data.contractType);
        setPlan(response.data.selectedPlan);
        setQuestion(response.data.question);
        setAnswer(response.data.answer);
        setGptModel(response.data.gpt_model);
        setWordCount(response.data.word_count);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  }, [chatId, conversationId, bearerToken]);

  if (chatId === "undefined") return null;

  let splitedText = clauses.split("[Document(page_content=");
  let relevantDoc = "";

  for (let i = 1; i < splitedText.length; i++) {
    const content = splitedText[i].split("metadata=");

    for (let j = 0; j < content.length - 1; j++) {
      let con = removeMetadata(content[j]);
      relevantDoc += con;
      relevantDoc += `\n\n---\n\n`;
    }
  }

  relevantDoc = relevantDoc.replace(/\\n/g, "\n").replace(/\\t/g, "\t");
  const lines = relevantDoc.split("\n");

  return (
    <div className="referred-clauses">
      <div className="info-row">
        <div className="info-column">
          <div className="info-title">Selected State</div>
          <div className="info-value">{state}</div>
        </div>
        <div className="info-column">
          <div className="info-title">Selected Contract</div>
          <div className="info-value">{contract}</div>
        </div>
        <div className="info-column">
          <div className="info-title">Selected Plan</div>
          <div className="info-value">{plan}</div>
        </div>
        <div className="info-column">
          <div className="info-title">Word Count</div>
          <div className="info-value">{wordCount}</div>
        </div>
      </div>
      <hr className="horizontal-line" />
      <div className="question-row">
        <div className="question-section">
          <div className="content-label">Question:</div>
          <div className="content-text">{question}</div>
        </div>
        <div className="gpt-model-chip">Type: {gptModel}</div>
      </div>
      <hr className="horizontal-line" />
      <div className="response-row">
        <div className="response-section">
          <div className="content-label">Response:</div>
          <div className="content-text">{answer}</div>
        </div>
      </div>
      <hr className="horizontal-line" />
      <div className="clauses-section">
        <div className="clauses-title">Referred Clauses</div>
        {lines.map((line, index) =>
          line === "---" ? (
            <hr className="horizontal-line" key={index} />
          ) : (
            <div className="clause-text" key={index}>
              {line.replace(/\t/g, "\u2003\u2003")}
            </div>
          )
        )}
      </div>
    </div>
  );
};

export default ReferredClauses;
