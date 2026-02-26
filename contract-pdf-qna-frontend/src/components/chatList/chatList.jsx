import React, { useEffect, useRef } from "react";
import Question from "../common/question/question";
import Response from "../common/response/response";
import { parseExtractedQuestion, stripTranscribeAppendix } from "../utils/chatText";

const isTranscriptExtractedChat = (chat) => {
  const id = chat?.questionId || chat?.chat_id;
  return typeof id === "string" && /^q\d+$/i.test(id);
};

const isFinalAnswerChat = (chat) => {
  const id = chat?.questionId || chat?.chat_id;
  return id === "final_answer" || chat?.entered_query === "Final Answer for transcript";
};

// Use first non-empty chunk array so placeholder or detail chunks show when the other is empty
const getRelevantChunks = (chat) => {
  const detail = chat?.relevantChunksDetail || chat?.relevant_chunks_detail || [];
  const textOnly = chat?.relevantChunks || chat?.relevant_chunks || [];
  return Array.isArray(detail) && detail.length > 0 ? detail : textOnly;
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
                const parsed = parseExtractedQuestion(chat?.entered_query || "");
                const qText = parsed?.questionText || stripTranscribeAppendix(chat?.entered_query || "");
                const facts = Array.isArray(parsed?.facts) ? parsed.facts : [];
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
                    {facts.length > 0 ? (
                      <div className="question_facts" aria-label="Extracted case facts">
                        {facts.map((f, i) => (
                          <div className="fact_chip" key={`${f.key || "k"}-${i}`} title={`${f.label}: ${f.value}`}>
                            <span className="k">{f.label}</span>
                            <span className="v">{f.value}</span>
                          </div>
                        ))}
                      </div>
                    ) : null}
                    {chat?.response ? (
                      <Response
                        response={chat.response}
                        chatId={chat.chat_id}
                        conversationId={conversationId}
                        chats={chats}
                        setChats={setChats}
                        showReferenceIcon={false}
                        relevantChunks={getRelevantChunks(chat)}
                        headerLabel="AI Draft Answer"
                        tone="blue"
                        isError={chat.isError}
                        onRetry={chat.isError && onRetryChat ? onRetryChat : null}
                        showActions={false}
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
            <div className="section_title">Chat</div>
            {followUps.map((chat, index) => (
              <div
                key={index}
                ref={index === followUps.length - 1 ? lastChatRef : null}
                className="chat_item"
              >
                {chat?.entered_query ? (
                  <Question text={chat.entered_query} label="You" />
                ) : null}
                {chat?.response ? (
                  <Response
                    response={chat.response}
                    chatId={chat.chat_id}
                    conversationId={conversationId}
                    chats={chats}
                    setChats={setChats}
                    showReferenceIcon={false}
                    relevantChunks={getRelevantChunks(chat)}
                    headerLabel="Assistant"
                    isError={chat.isError}
                    onRetry={chat.isError && onRetryChat ? onRetryChat : null}
                    showActions={false}
                  />
                ) : null}
              </div>
            ))}
          </div>
        ) : null}

        {finalAnswer?.response ? (
          <div className="calls_final_answer">
            <div className="section_title">Final analyzed answer</div>
            <Response
              response={finalAnswer.response}
              chatId={finalAnswer.chat_id}
              conversationId={conversationId}
              chats={chats}
              setChats={setChats}
              showReferenceIcon={false}
              relevantChunks={getRelevantChunks(finalAnswer)}
              variant="finalAnswer"
              headerLabel="Final Analyzed Answer"
              tone="blue"
              showActions={true}
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
              showReferenceIcon={true}
              relevantChunks={getRelevantChunks(chat)}
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
