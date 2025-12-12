import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import CancelIcon from "../../../assets/cancel.svg";
import CheckIcon from "../../../assets/check.svg";
import PenIcon from "../../../assets/pen.svg";
import TrashIcon from "../../../assets/trash.svg";
import { API_BASE_URL } from "../../../config";
import "./HistoryButton.scss";

const HistoryButton = ({
  setError,
  name,
  conversationId,
  isActive = false,
  setIsActive,
  getSidebarHistory,
  bearerToken,
}) => {
  const navigate = useNavigate();
  const [conversationName, setConversationName] = useState(name);
  const [isEditActive, setIsEditActive] = useState(false);

  const setChatUrl = () => {
    let path = `/conversation/${conversationId}`;
    navigate(path);
    setError("");
    setIsActive(conversationId);
  };

  const deleteConversation = (e) => {
    e.stopPropagation(); // Prevent triggering the setChatUrl
    const apiUrl = `${API_BASE_URL}/delete?conversation-id=${conversationId}`;
    axios
      .delete(apiUrl)
      .then(() => {
        getSidebarHistory(bearerToken);
        if (isActive === conversationId) {
          navigate("/#");
          setIsActive(null);
        }
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  };

  const editConversationName = (e) => {
    e.stopPropagation(); // Prevent triggering the setChatUrl
    const apiUrl = `${API_BASE_URL}/edit-conversation-name?conversation-id=${conversationId}`;
    const requestBody = {
      newName: conversationName,
    };
    axios
      .patch(apiUrl, requestBody)
      .then(() => {
        getSidebarHistory(bearerToken);
      })
      .catch((error) => {
        console.error("Error:", error);
      });
    setIsEditActive(false);
  };

  useEffect(() => {
    setConversationName(name);
    if (isActive) {
      setIsEditActive(false);
    }
  }, [isActive, name]);

  return (
    <div
      className={`history_wrapper ${
        isActive === conversationId ? "active" : ""
      }`}
      onClick={setChatUrl}
    >
      {isEditActive ? (
        <input
          type="text"
          autoFocus
          className="input_text"
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              editConversationName(e);
            }
          }}
          value={conversationName}
          onChange={(e) => {
            setConversationName(e.target.value);
          }}
        />
      ) : (
        <button className="input_text">{conversationName}</button>
      )}
      <div className="icons_container">
        {isEditActive ? (
          <>
            <img
              src={CheckIcon}
              alt="Check Icon"
              onClick={(e) => editConversationName(e)}
              className="icons"
            />
            <img
              src={CancelIcon}
              alt="Cancel Icon"
              onClick={(e) => {
                e.stopPropagation();
                setConversationName(name);
                setIsEditActive(false);
              }}
              className="icons"
            />
          </>
        ) : (
          <>
            <img
              src={PenIcon}
              alt="Pen Icon"
              onClick={(e) => {
                e.stopPropagation();
                setIsEditActive(true);
              }}
              className="icons"
            />
            <img
              src={TrashIcon}
              alt="Trash Icon"
              onClick={(e) => deleteConversation(e)}
              className="icons"
            />
          </>
        )}
      </div>
    </div>
  );
};

export default HistoryButton;
