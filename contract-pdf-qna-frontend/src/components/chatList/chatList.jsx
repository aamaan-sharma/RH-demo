import React, { useEffect, useRef } from "react";
import Question from "../common/question/question";
import Response from "../common/response/response";
import { stripTranscribeAppendix } from "../utils/chatText";

const isTranscriptExtractedChat = (chat) => {
  const id = chat?.questionId || chat?.chat_id;
  return typeof id === "string" && /^q\d+$/i.test(id);
};

const isFinalAnswerChat = (chat) => {
  const id = chat?.questionId || chat?.chat_id;
  return id === "final_answer" || chat?.entered_query === "Final Answer for transcript";
};

const ChatList = ({ chats, setChats, conversationId, isCallsMode = false, serverError = null, onRetryChat = null }) => {
  const lastChatRef = useRef(null);

  useEffect(() => {
    if (lastChatRef.current) {
      lastChatRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [chats]);

  if (isCallsMode) {
    const extracted = [];
    const followUps = [];
    let finalAnswer = null;

    (chats || []).forEach((chat) => {
      if (isFinalAnswerChat(chat)) {
        finalAnswer = chat;
        return;
      }
      if (chat?.source === "transcript_extracted" || isTranscriptExtractedChat(chat)) {
        extracted.push(chat);
        return;
      }
      followUps.push(chat);
    });

    return (
      <div className="chatList_wrapper">
        {extracted.length > 0 ? (
          <div className="calls_case_questions">
            <div className="section_title">Extracted questions</div>
            <div className="questions_list">
              {extracted.map((chat, idx) => {
                const qText = stripTranscribeAppendix(chat?.entered_query || "");
                return (
                  <details className="question_item" key={chat?.chat_id || chat?.questionId || idx}>
                    <summary className="question_summary">
                      <span className="q_left">
                        <span className="q_index">{`Q${idx + 1}`}</span>
                        <span className="q_text">{qText}</span>
                      </span>
                      <span className="q_dropdown_icon" aria-hidden="true">
                        ▸
                      </span>
                    </summary>
                    {chat?.response ? (
                      <Response
                        response={chat.response}
                        chatId={chat.chat_id}
                        conversationId={conversationId}
                        chats={chats}
                        setChats={setChats}
                        relevantChunks={chat.relevantChunks || chat.relevant_chunks || []}
                        headerLabel="AI Draft Answer"
                        tone="blue"
                        isError={chat.isError}
                        onRetry={chat.isError && onRetryChat ? onRetryChat : null}
                      />
                    ) : null}
                  </details>
                );
              })}
            </div>
          </div>
        ) : null}

        {followUps.length > 0 ? (
          <div className="calls_case_chat">
            <div className="section_title">Case chat</div>
            {followUps.map((chat, index) => (
              <div
                key={index}
                ref={index === followUps.length - 1 ? lastChatRef : null}
                className="chat_item"
              >
                {chat?.entered_query ? (
                  <Question text={chat.entered_query} label="You" meta="Case follow-up" />
                ) : null}
                {chat?.response ? (
                  <Response
                    response={chat.response}
                    chatId={chat.chat_id}
                    conversationId={conversationId}
                    chats={chats}
                    setChats={setChats}
                    relevantChunks={chat.relevantChunks || chat.relevant_chunks || []}
                    headerLabel="Assistant (Case Context)"
                    isError={chat.isError}
                    onRetry={chat.isError && onRetryChat ? onRetryChat : null}
                  />
                ) : null}
              </div>
            ))}
          </div>
        ) : null}

        {finalAnswer?.response ? (
          <div className="calls_final_answer">
            <div className="section_title">Final authorized answer (draft)</div>
            <Response
              response={finalAnswer.response}
              chatId={finalAnswer.chat_id}
              conversationId={conversationId}
              chats={chats}
              setChats={setChats}
              relevantChunks={finalAnswer.relevantChunks || finalAnswer.relevant_chunks || []}
              variant="finalAnswer"
              headerLabel="Final Authorized Answer (Draft)"
              tone="blue"
            />
          </div>
        ) : null}
      </div>
    );
  }

  return (
    <div className="chatList_wrapper">
      {chats?.map((chat, index) => (
        <div key={index} ref={index === chats.length - 1 ? lastChatRef : null}>
          {chat.entered_query && !isFinalAnswerChat(chat) && (
            <Question
              text={chat.entered_query}
              label={
                isCallsMode && (chat.source === "transcript_extracted" || isTranscriptExtractedChat(chat))
                  ? "Transcript"
                  : "You"
              }
              meta={
                isCallsMode && (chat.source === "transcript_extracted" || isTranscriptExtractedChat(chat))
                  ? "Extracted question"
                  : null
              }
            />
          )}
          {chat.response && (
            <Response
              response={chat.response}
              chatId={chat.chat_id}
              conversationId={conversationId}
              chats={chats}
              setChats={setChats}
              relevantChunks={chat.relevantChunks || chat.relevant_chunks || []}
              variant={isCallsMode && isFinalAnswerChat(chat) ? "finalAnswer" : "default"}
              isError={chat.isError}
              onRetry={chat.isError && onRetryChat ? onRetryChat : null}
            />
          )}
        </div>
      ))}
    </div>
  );
};

export default ChatList;
