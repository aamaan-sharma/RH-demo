import React, { useEffect, useRef } from "react";
import Question from "../common/question/question";
import Response from "../common/response/response";
import {
  DecisionBadge,
  parseDraftSummary,
} from "../common/itemizedFinalAnswer/itemizedFinalAnswer";
import {
  parseExtractedQuestion,
  stripTranscribeAppendix,
  stripEvidenceAndAfter,
} from "../utils/chatText";

/** Get decision and amount for display from itemized response (same parsing as answer body). */
function getDecisionAndAmountFromResponse(responseText) {
  const raw = String(responseText ?? "")
    .replace(/\r\n/g, "\n")
    .trim();
  const parsed = parseDraftSummary(raw);
  const first = parsed?.items?.[0];
  if (first) {
    const decision = (first.decision || "").trim();
    let amount = (first.amount || "").trim();
    if (
      !amount &&
      (first.amountsCompany?.length || first.amountsCustomer?.length)
    ) {
      const clean = (s) =>
        String(s || "")
          .trim()
          .replace(/^(?:Company|Customer)\s*:\s*/i, "")
          .trim() || "$0";
      const company = clean(first.amountsCompany?.[0]);
      const customer = clean(first.amountsCustomer?.[0]);
      amount = `Company ${company}, Customer ${customer}`;
    }
    if (!amount) amount = "N/A";
    return { decision, amount };
  }
  // Fallback: match "Decision: X" and "Amount(s): X" anywhere in text (e.g. non-itemized or single-line)
  let decision = "";
  let amount = "";
  const lines = raw.split("\n");
  for (const line of lines) {
    const t = line.trim();
    const dm = t.match(/Decision\s*:\s*(.+)/i);
    if (dm) decision = dm[1].trim();
    const am = t.match(/^Amount\s*:\s*(.+)/i);
    if (am) amount = am[1].trim();
    if (!amount && /^Amounts\s*:/.test(t)) {
      const cust = raw.match(/Customer\s*:\s*([^\n]+)/i);
      const comp = raw.match(/Company\s*:\s*([^\n]+)/i);
      const c1 = cust ? cust[1].trim() : "$0";
      const c2 = comp ? comp[1].trim() : "$0";
      amount = `Company ${c2}, Customer ${c1}`;
    }
  }
  // Also try to find Customer/Company amounts anywhere in raw (e.g. bullet lines)
  if (!amount && (/Customer\s*:/i.test(raw) || /Company\s*:/i.test(raw))) {
    const cust = raw.match(/Customer\s*:\s*([^\n]+)/i);
    const comp = raw.match(/Company\s*:\s*([^\n]+)/i);
    const c1 = cust ? cust[1].trim() : "$0";
    const c2 = comp ? comp[1].trim() : "$0";
    amount = `Company ${c2}, Customer ${c1}`;
  }
  if (!amount) amount = "N/A";
  return { decision, amount };
}

const isTranscriptExtractedChat = (chat) => {
  const id = chat?.questionId || chat?.chat_id;
  return typeof id === "string" && /^q\d+$/i.test(id);
};

const isFinalAnswerChat = (chat) => {
  const id = chat?.questionId || chat?.chat_id;
  return (
    id === "final_answer" ||
    chat?.entered_query === "Final Answer for transcript"
  );
};

// Use first non-empty chunk array so placeholder or detail chunks show when the other is empty
const getRelevantChunks = (chat) => {
  const detail =
    chat?.relevantChunksDetail || chat?.relevant_chunks_detail || [];
  const textOnly = chat?.relevantChunks || chat?.relevant_chunks || [];
  return Array.isArray(detail) && detail.length > 0 ? detail : textOnly;
};

const ChatList = ({
  chats,
  setChats,
  conversationId,
  isCallsMode = false,
  claimDecision = null,
  serverError = null,
  onRetryChat = null,
}) => {
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
      if (
        chat?.source === "transcript_extracted" ||
        isTranscriptExtractedChat(chat)
      ) {
        extracted.push(chat);
        return;
      }
      followUps.push(chat);
    });

    return (
      <div className="chatList_wrapper">
        {finalAnswer?.response ? (
          <div className="calls_final_answer">
            <Response
              response={finalAnswer.response}
              chatId={finalAnswer.chat_id}
              conversationId={conversationId}
              chats={chats}
              setChats={setChats}
              showReferenceIcon={false}
              relevantChunks={getRelevantChunks(finalAnswer)}
              variant="finalAnswer"
              headerLabel="Final Summary"
              tone="blue"
              showActions={true}
            />
          </div>
        ) : null}

        {extracted.length > 0 ? (
          <div className="calls_case_questions">
            <div className="section_title">Extracted questions</div>
            <div className="questions_list">
              {extracted.map((chat, idx) => {
                const parsed = parseExtractedQuestion(
                  chat?.entered_query || "",
                );
                const qText = stripEvidenceAndAfter(
                  parsed?.questionText ||
                    stripTranscribeAppendix(chat?.entered_query || ""),
                );
                const facts = Array.isArray(parsed?.facts) ? parsed.facts : [];
                const chatId = chat?.chat_id || chat?.questionId;
                const claim = claimDecision?.claims?.find(
                  (c) =>
                    (c?.claimId || "").toString() === (chatId || "").toString(),
                );
                const decisionFromClaim = claim?.decision;
                const { decision: decisionFromResponse } =
                  getDecisionAndAmountFromResponse(chat?.response || "");
                const decision =
                  decisionFromClaim != null && decisionFromClaim !== ""
                    ? decisionFromClaim
                    : decisionFromResponse;
                return (
                  <details
                    className="question_item"
                    key={chat?.chat_id || chat?.questionId || idx}
                  >
                    <summary className="question_summary">
                      <div className="question_item_header">
                        <div className="question_item_row">
                          <span
                            className="q_index"
                            aria-hidden="true"
                          >{`Q${idx + 1}`}</span>
                          <span className="q_text">{qText}</span>
                          {chat?.response ? (
                            <span className="question_decision_holder">
                              {decision ? (
                                <DecisionBadge decision={decision} />
                              ) : (
                                <span className="calls_badge calls_badge_no_decision">
                                  No Decision
                                </span>
                              )}
                            </span>
                          ) : null}
                        </div>
                        <span className="q_dropdown_icon" aria-hidden="true">
                          ▸
                        </span>
                      </div>
                    </summary>
                    <div className="question_item_body">
                      {facts.length > 0 ? (
                        <div
                          className="question_facts"
                          aria-label="Extracted case facts"
                        >
                          {facts.map((f, i) => (
                            <div
                              className="fact_chip"
                              key={`${f.key || "k"}-${i}`}
                              title={`${f.label}: ${f.value}`}
                            >
                              <span className="k">{f.label}</span>
                              <span className="v">{f.value}</span>
                            </div>
                          ))}
                        </div>
                      ) : null}
                      {chat?.response ? (
                        <div className="question_answer_block">
                          <Response
                            response={chat.response}
                            chatId={chat.chat_id}
                            conversationId={conversationId}
                            chats={chats}
                            setChats={setChats}
                            showReferenceIcon={false}
                            relevantChunks={getRelevantChunks(chat)}
                            variant="draftAnswer"
                            hideHeader={true}
                            tone="blue"
                            isError={chat.isError}
                            onRetry={
                              chat.isError && onRetryChat ? onRetryChat : null
                            }
                            showActions={false}
                            claimForChat={claim}
                          />
                        </div>
                      ) : null}
                    </div>
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
                isCallsMode &&
                (chat.source === "transcript_extracted" ||
                  isTranscriptExtractedChat(chat))
                  ? "Transcript"
                  : "You"
              }
              meta={
                isCallsMode &&
                (chat.source === "transcript_extracted" ||
                  isTranscriptExtractedChat(chat))
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
              variant={
                isCallsMode && isFinalAnswerChat(chat)
                  ? "finalAnswer"
                  : "default"
              }
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
