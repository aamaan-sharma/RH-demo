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
import { API_BASE_URL } from "../../config";
import "./home.scss";
import ChatList from "../chatList/chatList";
import CallsTranscriptModal from "../callsTranscriptModal/callsTranscriptModal";
import CallsExistingConversationPopup from "../callsExistingConversationPopup/callsExistingConversationPopup";

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
  const [transcriptStatusFilter, setTranscriptStatusFilter] = useState("active");
  const [isLoadingTranscripts, setIsLoadingTranscripts] = useState(false);
  const [selectedTranscript, setSelectedTranscript] = useState(null);
  const [existingCallsConversations, setExistingCallsConversations] = useState(
    []
  );
  const [isExistingConvPopupOpen, setIsExistingConvPopupOpen] = useState(false);

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
    (searchTerm = "", status = transcriptStatusFilter) => {
      setIsLoadingTranscripts(true);
      const params = {};
      if (searchTerm) {
        params.q = searchTerm;
      }
      if (status && status !== "all") {
        params.status = status;
      }
      axios
        .get(`${API_BASE_URL}/calls/transcripts`, { params })
        .then((response) => {
          setTranscripts(response.data.items || []);
        })
        .catch((error) => {
          console.error("Error fetching transcripts:", error);
        })
        .finally(() => {
          setIsLoadingTranscripts(false);
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
    fetchTranscripts("", transcriptStatusFilter);
  };

  const handleTranscriptSearchChange = (value) => {
    setTranscriptSearch(value);
    fetchTranscripts(value, transcriptStatusFilter);
  };

  const handleTranscriptStatusChange = (status) => {
    setTranscriptStatusFilter(status);
    fetchTranscripts(transcriptSearch, status);
  };

  const startNewCallsConversation = (transcript) => {
    if (!transcript) return;
    axios
      .post(`${API_BASE_URL}/calls/conversations`, {
        transcriptId: transcript.id,
      })
      .then((response) => {
        const { conversation } = response.data;
        setSelectedState(conversation.stateName);
        setSelectedContract(conversation.contractType);
        setSelectedPlan(conversation.planName);
        setChats([]);
        setIsCallsMode(true);
        setGptModelState("Calls");
        setIsTranscriptModalOpen(false);
        setExistingCallsConversations([]);
        setSelectedTranscript(transcript);
        const path = `/conversation/${conversation.id}`;
        navigate(path);
      })
      .catch((error) => {
        console.error("Error starting Calls conversation:", error);
      });
  };

  const handleSelectTranscript = (transcript) => {
    setSelectedTranscript(transcript);
    axios
      .get(
        `${API_BASE_URL}/calls/conversations/by-transcript/${transcript.id}`
      )
      .then((response) => {
        if (response.data.exists && response.data.conversations.length > 0) {
          setExistingCallsConversations(response.data.conversations);
          setIsExistingConvPopupOpen(true);
        } else {
          startNewCallsConversation(transcript);
        }
      })
      .catch((error) => {
        console.error(
          "Error checking existing Calls conversations for transcript:",
          error
        );
        startNewCallsConversation(transcript);
      });
  };

  const handleGoToExistingCallsConversation = () => {
    if (!existingCallsConversations.length) return;
    const conversation = existingCallsConversations[0];
    setIsExistingConvPopupOpen(false);
    setIsTranscriptModalOpen(false);
    setIsCallsMode(true);
    setGptModelState("Calls");
    setSelectedTranscript(null);
    setExistingCallsConversations([]);
    const path = `/conversation/${conversation.id}`;
    navigate(path);
  };

  const handleStartNewCallsConversationFromPopup = () => {
    setIsExistingConvPopupOpen(false);
    if (selectedTranscript) {
      startNewCallsConversation(selectedTranscript);
    }
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
      setGptModelState("Search");
      setIsCallsMode(false);
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
        { entered_query: input, response: "Loading Response" },
      ]);
    } else {
      setChats((prevChats) => [
        ...prevChats,
        { entered_query: input, response: "Loading Response" },
      ]);
    }

    if (!isCallsMode && conversationId === "") {
      setChats([{ entered_query: input, response: "Loading Response" }]);
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
      const apiUrl = `${API_BASE_URL}/calls/start?conversation-id=${conversationId}`;
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
          setGptModel={setGptModelState}
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
            />

            {chats.length === 0 && !isCallsMode ? (
              <SamplePrompt
                gptModel={gptModel}
                input={input}
                setInput={setInput}
              />
            ) : chats.length > 0 ? (
              <div
                className={`chat_container  ${isScrollable ? "setHeight" : ""}`}
                ref={chatRef}
              >
                <ChatList
                  chats={chats}
                  setChats={setChats}
                  conversationId={conversationId}
                />
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
          isLoading={isLoadingTranscripts}
        />
        <CallsExistingConversationPopup
          isOpen={isExistingConvPopupOpen}
          onClose={() => {
            setIsExistingConvPopupOpen(false);
            setExistingCallsConversations([]);
          }}
          onStartNew={handleStartNewCallsConversationFromPopup}
          onGoExisting={handleGoToExistingCallsConversation}
        />
      </div>
    </div>
  );
};

export default Home;
