import React, { useEffect, useRef } from "react";
import Question from "../common/question/question";
import Response from "../common/response/response";

const isTranscriptExtractedChat = (chat) => {
  const id = chat?.questionId || chat?.chat_id;
  return typeof id === "string" && /^q\d+$/i.test(id);
};

const isFinalAnswerChat = (chat) => {
  const id = chat?.questionId || chat?.chat_id;
  return id === "final_answer" || chat?.entered_query === "Final Answer for transcript";
};

const ChatList = ({ chats, setChats, conversationId, isCallsMode = false }) => {
  const lastChatRef = useRef(null);

  useEffect(() => {
    if (lastChatRef.current) {
      lastChatRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [chats]);

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
            />
          )}
        </div>
      ))}
    </div>
  );
};

export default ChatList;
