import "regenerator-runtime";
import axios from "axios";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import SpeechRecognition, {
  useSpeechRecognition,
} from "react-speech-recognition";
import "regenerator-runtime";
import FilterSection from "../filterSection/filterSection";
import Header from "../header/header";
import InputField from "../inputField/inputfield";
import SamplePrompt from "../samplePrompt/samplePrompt";
import SideBar from "../sideBar/sideBar";
import { setHeaders } from "../utils/apiUtils";
import { API_BASE_URL, TRANSCRIPTS_API_BASE_URL } from "../../config";
import "./home.scss";
import ChatList from "../chatList/chatList";
import CallsTranscriptModal from "../callsTranscriptModal/callsTranscriptModal";

const TRANSCRIPTS_PAGE_SIZE = 9;

const Home = ({ bearerToken, setBearerToken }) => {
  const location = useLocation();
  let navigate = useNavigate();
  const conversationId = location.pathname.split("/")[2]
    ? location.pathname.split("/")[2]
    : "";
  const [chats, setChats] = useState([]);
  const [userEmail, setUserEmail] = useState("");
  const [gptModel, setGptModelState] = useState("Search"); // "Search" | "Infer" | "Calls"
  const [isCallsMode, setIsCallsMode] = useState(false);
  const chatRef = useRef();

  const [selectedContract, setSelectedContract] = useState("");
  const [selectedPlan, setSelectedPlan] = useState("");
  const [selectedState, setSelectedState] = useState("");
  const [refreshToken, setRefreshToken] = useState("");
  const [error, setError] = useState("");
  const [input, setInput] = useState("");
  const [userImage, setUserImage] = useState("");
  const [isScrollable, setIsScrollable] = useState(false);
  const [isTranscriptModalOpen, setIsTranscriptModalOpen] = useState(false);
  const [transcripts, setTranscripts] = useState([]);
  const [transcriptSearch, setTranscriptSearch] = useState("");
  // Transcript list status filter (modal)
  const [transcriptStatusFilter, setTranscriptStatusFilter] = useState("active");
  const [isLoadingTranscripts, setIsLoadingTranscripts] = useState(false);
  const [isLoadingMoreTranscripts, setIsLoadingMoreTranscripts] = useState(false);
  const [transcriptsOffset, setTranscriptsOffset] = useState(0);
  const [transcriptsHasMore, setTranscriptsHasMore] = useState(true);
  const [finalSummary, setFinalSummary] = useState("");
  const [conversationStatus, setConversationStatus] = useState("active");
  const [callsTranscriptName, setCallsTranscriptName] = useState("");
  const [callsGenerationStage, setCallsGenerationStage] = useState("idle"); // idle | generating | done
  const [sidebarRefreshTick, setSidebarRefreshTick] = useState(0);
  const [isCheckingExistingTranscriptConversation, setIsCheckingExistingTranscriptConversation] =
    useState(false);
  const [existingTranscriptConversations, setExistingTranscriptConversations] = useState([]);
  const [isTranscriptChoiceOpen, setIsTranscriptChoiceOpen] = useState(false);
  const [pendingTranscript, setPendingTranscript] = useState(null);
  const [selectedExistingConversationId, setSelectedExistingConversationId] = useState("");

  const hasFinalAnswerChat = chats?.some(
    (c) => c?.entered_query === "Final Answer for transcript"
  );

  axios.interceptors.request.use(setHeaders, (error) => {
    Promise.reject(error);
  });

  const handleSetGptModel = (model) => {
    // Keep Calls mode in sync with selected model
    if (model === "Calls") {
      setIsCallsMode(true);
    } else {
      setIsCallsMode(false);
    }
    setGptModelState(model);
  };

  const fetchTranscripts = useCallback(
    (searchTerm = "", status = transcriptStatusFilter, offset = 0, append = false) => {
      const limit = TRANSCRIPTS_PAGE_SIZE;
      if (offset === 0) {
        setIsLoadingTranscripts(true);
      } else {
        setIsLoadingMoreTranscripts(true);
      }

      // Map UI filters to backend query params for the transcripts service
      const params = {
        // Use the new transcripts backend search param (supports alias `q` as well)
        search: searchTerm || undefined,
        limit,
        offset,
        // Status filter (active|inactive)
        status: status || undefined,
      };

      axios
        .get(`${TRANSCRIPTS_API_BASE_URL}/transcripts`, { params })
        .then((response) => {
          const apiTranscripts = response?.data?.transcripts || [];
          const hasMore = Boolean(response?.data?.hasMore);

          // Adapt backend transcript shape to what the UI expects
          const mappedTranscripts = apiTranscripts.map((t) => ({
            // Use fileName as a stable identifier
            id: t.fileName,
            name: t.fileName,
            // Map metadata fields with safe fallbacks
            stateName: t.state || "N/A",
            contractType: t.contractType || "N/A",
            planName: t.planType || "N/A",
            // Backend exposes status (stored in Mongo); default active
            status: (t.status || "active").toLowerCase(),
            // Keep raw fields in case they are needed later
            filePath: t.filePath,
            uploadDate: t.uploadDate,
          }));

          setTranscripts((prev) => (append ? [...prev, ...mappedTranscripts] : mappedTranscripts));
          setTranscriptsHasMore(hasMore);
          setTranscriptsOffset(offset + limit);
        })
        .catch((error) => {
          console.error("Error fetching transcripts:", error);
        })
        .finally(() => {
          setIsLoadingTranscripts(false);
          setIsLoadingMoreTranscripts(false);
        });
    },
    [transcriptStatusFilter]
  );

  const handleOpenTranscriptModal = () => {
    if (!sessionStorage.getItem("idToken")) {
      setError("login");
      return;
    }
    // Opening the transcript modal implies we are in Calls mode
    setIsCallsMode(true);
    setGptModelState("Calls");
    setIsTranscriptModalOpen(true);
    setTranscriptsOffset(0);
    setTranscriptsHasMore(true);
    fetchTranscripts("", transcriptStatusFilter, 0, false);
  };

  const handleTranscriptSearchChange = (value) => {
    setTranscriptSearch(value);
    setTranscriptsOffset(0);
    setTranscriptsHasMore(true);
    fetchTranscripts(value, transcriptStatusFilter, 0, false);
  };

  const handleTranscriptStatusChange = (status) => {
    setTranscriptStatusFilter(status);
    setTranscriptsOffset(0);
    setTranscriptsHasMore(true);
    fetchTranscripts(transcriptSearch, status, 0, false);
  };

  const startNewCallsConversation = (transcript, opts = {}) => {
    if (!transcript) return;

    // Prefer metadata coming from the transcripts service; abort if missing
    const contractType =
      transcript.contractType && transcript.contractType !== "N/A"
        ? transcript.contractType
        : null;
    const planName =
      transcript.planName && transcript.planName !== "N/A"
        ? transcript.planName
        : null;
    const stateName =
      transcript.stateName && transcript.stateName !== "N/A"
        ? transcript.stateName
        : null;

    if (!contractType || !planName || !stateName) {
      console.error(
        "Cannot process transcript – missing contractType/plan/state metadata",
        transcript
      );
      return;
    }

    // Update dropdowns immediately (so UI reflects what we're generating against)
    setSelectedState(stateName);
    setSelectedContract(contractType);
    setSelectedPlan(planName);

    const requestBody = {
      transcriptFileName: transcript.id,
      contractType,
      selectedPlan: planName,
      selectedState: stateName,
      // Use underlying QA behaviour; UI will still label this as "Calls"
      gptModel: gptModel === "Infer" ? "Infer" : "Search",
      extractQuestions: true,
      newConversation: Boolean(opts?.newConversation),
      conversationName: transcript.name || transcript.id,
    };

    // Show a transcript header + generation stage while processing
    setCallsTranscriptName(transcript.name || transcript.id);
    setCallsGenerationStage("generating");
    setChats([]);
    setFinalSummary("");
    setIsTranscriptModalOpen(false);
    // Trigger sidebar refresh shortly after the backend receives the request (it now creates a processing stub early).
    setTimeout(() => setSidebarRefreshTick((t) => t + 1), 600);

    axios
      .post(`${TRANSCRIPTS_API_BASE_URL}/transcripts/process`, requestBody)
      .then((response) => {
        const conversationIdFromApi = response?.data?.conversationId || "";
        const questions = response?.data?.questions || [];
        const apiFinalSummary = response?.data?.finalSummary || "";
        const extractionWarning = response?.data?.warning;
        setFinalSummary(apiFinalSummary);
        setConversationStatus((response?.data?.status || "active").toLowerCase());
        setCallsGenerationStage("done");
        setCallsTranscriptName(
          response?.data?.transcriptMetadata?.fileName ||
            response?.data?.transcriptId ||
            transcript.name ||
            transcript.id
        );

        // Map transcript questions/answers into existing chat shape.
        // Keep chunks in JSON so the UI can render them as structured "Data Chunks".
        const mappedChats =
          questions.length > 0
            ? questions.map((q) => {
                const chunks = q.relevantChunks || [];
                const isFinal = q.questionId === "final_answer";
                return {
                  entered_query: q.question,
                  response: q.answer,
                  chat_id: q.questionId,
                  // keep extra metadata in case we want it later
                  questionId: q.questionId,
                  questionType: q.questionType,
                  userIntent: q.userIntent,
                  relevant_chunks: chunks,
                  underlying_model: requestBody.gptModel,
                  source: isFinal ? "final_answer" : "transcript_extracted",
                };
              })
            : [
                {
                  entered_query: "",
                  response:
                    extractionWarning ||
                    "No questions were extracted for this transcript.",
                  source: "transcript_extracted",
                },
              ];

        setChats(mappedChats);
        setIsCallsMode(true);
        setGptModelState("Calls");
        setSidebarRefreshTick((t) => t + 1);

        // Navigate to the persisted conversation so the input switches from "Add Transcript" to ChatInput
        if (conversationIdFromApi) {
          navigate(`/conversation/${conversationIdFromApi}`);
        }
      })
      .catch((error) => {
        setCallsGenerationStage("idle");
        setChats([
          {
            entered_query: "",
            response: "An error occurred while processing your request.",
          },
        ]);
        console.error(
          "Error processing transcript with /transcripts/process:",
          error
        );
      });
  };

  const handleSelectTranscript = (transcript) => {
    if (!transcript) return;
    // Close the picker and check Mongo for existing conversations (blocking)
    setIsTranscriptModalOpen(false);
    setPendingTranscript(transcript);
    setIsCheckingExistingTranscriptConversation(true);

    axios
      .get(`${TRANSCRIPTS_API_BASE_URL}/transcripts/conversations`, {
        params: { transcriptFileName: transcript.id },
      })
      .then((resp) => {
        const convs = resp?.data?.conversations || [];
        if (Array.isArray(convs) && convs.length > 0) {
          setExistingTranscriptConversations(convs);
          setSelectedExistingConversationId(convs?.[0]?.conversationId || "");
          setIsTranscriptChoiceOpen(true);
        } else {
          // No existing -> normal flow
          startNewCallsConversation(transcript, { newConversation: false });
        }
      })
      .catch((err) => {
        console.error("Error checking transcript conversations:", err);
        // Fail open: proceed with normal flow
        startNewCallsConversation(transcript, { newConversation: false });
      })
      .finally(() => {
        setIsCheckingExistingTranscriptConversation(false);
      });
  };

  useEffect(() => {
    if (conversationId !== "") {
      const apiUrl = `${API_BASE_URL}/history?conversation-id=${conversationId}`;
      axios
        .get(apiUrl)
        .then((response) => {
          if (
            response.data.message === "Token is invalid" ||
            response.data.message === "Token has expired" ||
            response.data.message === "Token is missing"
          ) {
            return;
          }
          setChats(response.data.chats);
          setSelectedState(response.data.selectedState);
          setSelectedContract(response.data.contractType);
          setSelectedPlan(response.data.selectedPlan);
          setFinalSummary(response.data.finalSummary || "");
          setConversationStatus((response.data.status || "active").toLowerCase());

          const transcriptNameFromApi =
            response?.data?.transcriptMetadata?.fileName ||
            response?.data?.transcriptId ||
            "";
          if (transcriptNameFromApi) {
            setCallsTranscriptName(transcriptNameFromApi);
            setCallsGenerationStage("done");
          } else {
            setCallsTranscriptName("");
            setCallsGenerationStage("idle");
          }
      const modelFromHistory = response.data.gptModel || "Search";
      setGptModelState(modelFromHistory);
      setIsCallsMode(modelFromHistory === "Calls");
          setInput("");
        })
        .catch((error) => {
          console.error("Error:", error);
        });
    } else {
      setChats([]);
      setSelectedState("State");
      setSelectedContract("Contract Type");
      setSelectedPlan("Plan");
      // Keep "New Chat" in the current mode (Search/Infer/Calls)
      setIsCallsMode(gptModel === "Calls");
      setFinalSummary("");
      setConversationStatus("active");
      setCallsTranscriptName("");
      setCallsGenerationStage("idle");
    }
  }, [conversationId]);

  useEffect(() => {
    const chatContainer = chatRef.current;
    if (chatContainer) {
      const isContentOverflowing =
        chatContainer.scrollHeight > chatContainer.clientHeight;
      setIsScrollable(isContentOverflowing);
    }
  }, [chats]);

  const handleInputSubmit = () => {
    if (!sessionStorage.getItem("idToken")) {
      setError("login");
      return;
    }

    if (input === "") return;

    if (
      chats.length > 0 &&
      chats[chats.length - 1].response === "Loading Response"
    ) {
      return;
    }

    if (
      selectedState === "State" &&
      selectedContract === "Contract Type" &&
      selectedPlan === "Plan"
    ) {
      setError("state contract plan");
      return;
    }
    if (selectedState === "State" && selectedContract === "Contract Type") {
      setError("state contract");
      return;
    }
    if (selectedState === "State" && selectedPlan === "Plan") {
      setError("state plan");
      return;
    }
    if (selectedContract === "Contract Type" && selectedPlan === "Plan") {
      setError("contract plan");
      return;
    }
    if (selectedState === "State") {
      setError("state");
      return;
    }
    if (selectedContract === "Contract Type") {
      setError("contract");
      return;
    }
    if (selectedPlan === "Plan") {
      setError("plan");
      return;
    }

    setError("");
    setError("");

    const isCallsConversationActive = isCallsMode && conversationId !== "";

    if (
      chats.length > 0 &&
      chats[chats.length - 1].response ===
        "An error occurred while processing your request."
    ) {
      setChats((prevChats) => [
        ...prevChats.slice(0, -1),
        { entered_query: input, response: "Loading Response", source: "user" },
      ]);
    } else {
      setChats((prevChats) => [
        ...prevChats,
        { entered_query: input, response: "Loading Response", source: "user" },
      ]);
    }

    if (!isCallsMode && conversationId === "") {
      setChats([{ entered_query: input, response: "Loading Response", source: "user" }]);
      let path = `/c/`;
      navigate(path);
    }

    let requestBody = {
      enteredQuery: input,
      contractType: selectedContract,
      selectedPlan: selectedPlan,
      selectedState: selectedState,
    };

    if (isCallsMode) {
      if (!isCallsConversationActive) {
        // Should not reach here because input is hidden before a Calls conversation is active
        setChats((prevChats) => prevChats.slice(0, -1));
        setInput("");
        return;
      }
      const callsUnderlyingModel = chats?.[0]?.underlying_model || "Search";
      const apiUrl = `${API_BASE_URL}/start?conversation-id=${conversationId}`;
      axios
        .post(apiUrl, { ...requestBody, gptModel: callsUnderlyingModel })
        .then((response) => {
          if (
            response.data.message === "Token is invalid" ||
            response.data.message === "Token has expired" ||
            response.data.message === "Token is missing"
          ) {
            setError("login");
            setChats((prevChats) => [
              ...prevChats.slice(0, -1),
              {
                entered_query: input,
                response: "An error occurred while processing your request.",
              },
            ]);
          } else {
            setChats((prevChats) => [
              ...prevChats.slice(0, -1),
              {
                entered_query: input,
                response: response.data.aiResponse,
                chat_id: response.data.chatId,
                underlying_model: callsUnderlyingModel,
                source: "user",
              },
            ]);
          }
        })
        .catch((error) => {
          setChats((prevChats) => [
            ...prevChats.slice(0, -1),
            {
              entered_query: input,
              response: "An error occurred while processing your request.",
            },
          ]);
          console.error("Error:", error);
        });
    } else {
      requestBody = {
        ...requestBody,
        gptModel: gptModel,
      };
      const apiUrl = `${API_BASE_URL}/start?conversation-id=${conversationId}`;
      axios
        .post(apiUrl, requestBody)
        .then((response) => {
          if (
            response.data.message === "Token is invalid" ||
            response.data.message === "Token has expired" ||
            response.data.message === "Token is missing"
          ) {
            setError("login");
            setChats((prevChats) => [
              ...prevChats.slice(0, -1),
              {
                entered_query: input,
                response: "An error occurred while processing your request.",
              },
            ]);
          } else {
            setChats((prevChats) => [
              ...prevChats.slice(0, -1),
              {
                entered_query: input,
                response: response.data.aiResponse,
                chat_id: response.data.chatId,
                source: "user",
              },
            ]);
            let path = `/conversation/${response.data.conversationId}`;

            navigate(path);
          }
        })
        .catch((error) => {
          setChats((prevChats) => [
            ...prevChats.slice(0, -1),
            {
              entered_query: input,
              response: "An error occurred while processing your request.",
            },
          ]);
          console.error("Error:", error);
        });
    }
    setInput("");
  };

  const textareaRef = useRef(null);
  const { listening, transcript, finalTranscript, resetTranscript } =
    useSpeechRecognition();

  const startRecording = () => {
    if (SpeechRecognition.browserSupportsSpeechRecognition()) {
      SpeechRecognition.startListening({ continuous: true, language: "en-GB" });
    }
  };
  const stopRecording = () => {
    SpeechRecognition.stopListening();
    setInput(finalTranscript);
    resetTranscript();
  };

  const onMicrophoneClick = () => {
    if (listening) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  useEffect(() => {
    const adjustHeight = () => {
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto";
        const maxHeight = 60;
        textareaRef.current.style.height =
          Math.min(textareaRef.current.scrollHeight, maxHeight) + "px";
      }
    };
    adjustHeight();
    if (textareaRef.current) {
      textareaRef.current.addEventListener("input", adjustHeight);
    }
    const inputRef = textareaRef.current;

    return () => {
      if (inputRef) {
        inputRef.removeEventListener("input", adjustHeight);
      }
    };
  }, [input]);

  const handleEnter = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleInputSubmit();
    }
  };

  if (!SpeechRecognition.browserSupportsSpeechRecognition()) {
    return null;
  }

  return (
    <div className="home_container">
      <div className="sidebar_container">
        <SideBar
          error={error}
          setError={setError}
          userEmail={userEmail}
          setUserEmail={setUserEmail}
          bearerToken={bearerToken}
          setBearerToken={setBearerToken}
          refreshToken={refreshToken}
          setRefreshToken={setRefreshToken}
          setGptModel={handleSetGptModel}
          selectedModel={gptModel}
          sidebarRefreshTick={sidebarRefreshTick}
          setSelectedContract={setSelectedContract}
          setSelectedPlan={setSelectedPlan}
          setSelectedState={setSelectedState}
          setUserImage={setUserImage}
        />
      </div>
      <div className="main_container">
        <Header userIconImage={userImage} />
        <div className="chat_section_wrapper">
          <div className="chat_section">
            <FilterSection
              error={error}
              setError={setError}
              selectedContract={selectedContract}
              setSelectedContract={setSelectedContract}
              selectedPlan={selectedPlan}
              setSelectedPlan={setSelectedPlan}
              selectedState={selectedState}
              setSelectedState={setSelectedState}
              setGptModel={handleSetGptModel}
              selectedModel={gptModel}
              userEmail={userEmail}
              isCallsMode={isCallsMode}
              transcriptStatusFilter={transcriptStatusFilter}
              onTranscriptStatusChange={handleTranscriptStatusChange}
              conversationStatus={conversationStatus}
              isConversationActive={isCallsMode && conversationId !== ""}
              onConversationStatusChange={(status) => {
                if (!conversationId) return;
                axios
                  .patch(
                    `${API_BASE_URL}/conversation/status?conversation-id=${conversationId}`,
                    { status }
                  )
                  .then(() => {
                    setConversationStatus(status);
                  })
                  .catch((err) => {
                    console.error("Error updating conversation status:", err);
                  });
              }}
            />

            {chats.length === 0 && !isCallsMode ? (
              <SamplePrompt
                gptModel={gptModel}
                input={input}
                setInput={setInput}
              />
            ) : chats.length > 0 || (isCallsMode && callsTranscriptName) ? (
              <div
                className={`chat_container  ${isScrollable ? "setHeight" : ""}`}
                ref={chatRef}
              >
                <>
                  {isCallsMode && callsTranscriptName ? (
                    <div className="calls_transcript_header">
                      <div className="title">
                        You uploaded: <span className="file">{callsTranscriptName}</span>
                      </div>
                      {callsGenerationStage === "generating" ? (
                        <div className="subtle_status">Generating…</div>
                      ) : callsGenerationStage === "done" ? (
                        <>
                          <div className="subtle_status">Generated response</div>
                          <div className="subtle_hint">Here are your details</div>
                        </>
                      ) : null}
                    </div>
                  ) : null}
                  <ChatList
                    chats={chats}
                    setChats={setChats}
                    conversationId={conversationId}
                    isCallsMode={isCallsMode}
                  />
                  {isCallsMode && finalSummary && !hasFinalAnswerChat ? (
                    <div className="calls_summary">
                      <div className="title">Final Summary</div>
                      <div className="text">{finalSummary}</div>
                    </div>
                  ) : null}
                </>
              </div>
            ) : null}
          </div>
          <div className="inpufield_wrapper">
            {isCallsMode && conversationId === "" ? (
              <button
                type="button"
                className="add_transcript_button"
                onClick={handleOpenTranscriptModal}
              >
                Add Transcript
              </button>
            ) : (
              <InputField
                listening={listening}
                transcript={transcript}
                handleInputEnter={() => {
                  handleInputSubmit();
                }}
                handleEnter={handleEnter}
                description={input}
                setDescription={setInput}
                textareaRef={textareaRef}
                onMicrophoneClick={onMicrophoneClick}
              />
            )}
          </div>
        </div>
        <CallsTranscriptModal
          isOpen={isTranscriptModalOpen}
          onClose={() => setIsTranscriptModalOpen(false)}
          transcripts={transcripts}
          searchTerm={transcriptSearch}
          onSearchTermChange={handleTranscriptSearchChange}
          statusFilter={transcriptStatusFilter}
          onStatusFilterChange={handleTranscriptStatusChange}
          onSelectTranscript={handleSelectTranscript}
          onToggleStatus={(t) => {
            const nextStatus = t.status === "active" ? "inactive" : "active";
            axios
              .patch(`${TRANSCRIPTS_API_BASE_URL}/transcripts/status`, {
                transcriptFileName: t.id,
                status: nextStatus,
              })
              .then(() => {
                setTranscripts((prev) =>
                  prev.map((x) =>
                    x.id === t.id ? { ...x, status: nextStatus } : x
                  )
                );
              })
              .catch((err) => {
                console.error("Error updating transcript status:", err);
              });
          }}
          isLoading={isLoadingTranscripts}
          isLoadingMore={isLoadingMoreTranscripts}
          hasMore={transcriptsHasMore}
          onLoadMore={() => {
            if (isLoadingTranscripts || isLoadingMoreTranscripts || !transcriptsHasMore) return;
            fetchTranscripts(transcriptSearch, transcriptStatusFilter, transcriptsOffset, true);
          }}
        />

        {isCheckingExistingTranscriptConversation ? (
          <div className="blocking_overlay" role="dialog" aria-modal="true">
            <div className="blocking_card">
              <div className="spinner" aria-hidden="true" />
              <div className="text">Checking existing conversations…</div>
            </div>
          </div>
        ) : null}

        {isTranscriptChoiceOpen && pendingTranscript ? (
          <div className="blocking_overlay" role="dialog" aria-modal="true">
            <div className="choice_card">
              <div className="title">Conversation already exists</div>
              <div className="subtitle">
                We found an existing conversation for{" "}
                <span className="file">{pendingTranscript.name || pendingTranscript.id}</span>.
              </div>
              <div className="existing_list">
                {(existingTranscriptConversations || []).map((c) => {
                  const id = c?.conversationId;
                  const name = c?.conversationName || id;
                  const ts = c?.updatedAt || c?.createdAt;
                  let tsLabel = "";
                  try {
                    tsLabel = ts ? new Date(ts).toLocaleString() : "";
                  } catch (e) {
                    tsLabel = "";
                  }
                  return (
                    <label className="existing_item" key={id}>
                      <input
                        type="radio"
                        name="existing_conversation"
                        checked={selectedExistingConversationId === id}
                        onChange={() => setSelectedExistingConversationId(id)}
                      />
                      <div className="existing_text">
                        <div className="existing_name">{name}</div>
                        {tsLabel ? <div className="existing_meta">Last updated: {tsLabel}</div> : null}
                      </div>
                    </label>
                  );
                })}
              </div>
              <div className="actions">
                <button
                  type="button"
                  className="primary"
                  onClick={() => {
                    setIsTranscriptChoiceOpen(false);
                    setExistingTranscriptConversations([]);
                    const convId = selectedExistingConversationId;
                    setSelectedExistingConversationId("");
                    setPendingTranscript(null);
                    if (convId) {
                      setIsCallsMode(true);
                      setGptModelState("Calls");
                      navigate(`/conversation/${convId}`);
                    }
                  }}
                >
                  Open selected
                </button>
                <button
                  type="button"
                  className="secondary"
                  onClick={() => {
                    const t = pendingTranscript;
                    setIsTranscriptChoiceOpen(false);
                    setExistingTranscriptConversations([]);
                    setSelectedExistingConversationId("");
                    setPendingTranscript(null);
                    startNewCallsConversation(t, { newConversation: true });
                  }}
                >
                  Start new conversation
                </button>
                <button
                  type="button"
                  className="ghost"
                  onClick={() => {
                    setIsTranscriptChoiceOpen(false);
                    setExistingTranscriptConversations([]);
                    setSelectedExistingConversationId("");
                    setPendingTranscript(null);
                  }}
                >
                  Exit
                </button>
              </div>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
};

export default Home;
