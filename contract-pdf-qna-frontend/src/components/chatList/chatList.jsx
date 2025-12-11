import React, { useEffect, useRef, useState } from "react";
import Question from "../common/question/question";
import Response from "../common/response/response";
import "./chatList.scss";

const ChatList = ({
  chats,
  setChats,
  conversationId,
  qaResult,
  isLoadingQA,
  selectedTranscript,
}) => {
  const lastChatRef = useRef(null);
  const [expandedChunks, setExpandedChunks] = useState({});

  useEffect(() => {
    if (lastChatRef.current) {
      lastChatRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [chats, qaResult, isLoadingQA]);

  if (isLoadingQA) {
    return (
      <div className="chatList_wrapper">
        <div className="loader_card">
          <div className="loader_title">
            Generating your answer<span className="ellipsis">...</span>
          </div>
          <div className="loader_subtitle">
            We’re analyzing the transcript and extracting answers.
          </div>
        </div>
      </div>
    );
  }

  if (qaResult) {
    return (
      <div className="chatList_wrapper">
        <div className="transcript_header">
          <div className="transcript_name">
            {selectedTranscript?.name ?? "Transcript"}
          </div>
          <div className="transcript_meta">
            {selectedTranscript?.updatedAt
              ? new Date(selectedTranscript.updatedAt).toLocaleString()
              : ""}
          </div>
        </div>

        <div className="qa_list">
          {qaResult.questions?.map((q, idx) => (
            <div key={idx} className="qa_card">
              <div className="qa_question">
                <span className="doc_icon">📄</span>
                <span>{q.question}</span>
              </div>
              <div className="qa_chunks">
                {q.chunks?.map((chunk, cidx) => {
                  const key = `${idx}-${cidx}`;
                  const expanded = expandedChunks[key];
                  const preview =
                    chunk.split(" ").slice(0, 16).join(" ") +
                    (chunk.split(" ").length > 16 ? "..." : "");
                  return (
                    <div
                      key={cidx}
                      className={`qa_chunk ${expanded ? "expanded" : ""}`}
                      onClick={() =>
                        setExpandedChunks((prev) => ({
                          ...prev,
                          [key]: !expanded,
                        }))
                      }
                    >
                      {expanded ? chunk : preview}
                      <div className="qa_chunk_toggle">
                        {expanded ? "Show less" : "Show more"}
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="qa_answer">{q.answer}</div>
            </div>
          ))}
        </div>

        {qaResult.finalAnswer && (
          <div className="summary_card">
            <div className="summary_title">Final Summary</div>
            <div className="summary_body">{qaResult.finalAnswer}</div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="chatList_wrapper">
      {chats?.map((chat, index) => (
        <div key={index} ref={index === chats.length - 1 ? lastChatRef : null}>
          {chat.entered_query && <Question text={chat.entered_query} />}
          {chat.response && (
            <Response
              response={chat.response}
              chatId={chat.chat_id}
              conversationId={conversationId}
              chats={chats}
              setChats={setChats}
            />
          )}
        </div>
      ))}
    </div>
  );
};

export default ChatList;
