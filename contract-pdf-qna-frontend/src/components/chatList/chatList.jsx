import React, { useEffect, useRef } from "react";
import Question from "../common/question/question";
import Response from "../common/response/response";

const ChatList = ({ chats, setChats, conversationId }) => {
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
