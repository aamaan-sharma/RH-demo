import "regenerator-runtime";
import axios from "axios";
import React, { useEffect, useRef, useState } from "react";
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

const Home = ({ bearerToken, setBearerToken }) => {
  const location = useLocation();
  let navigate = useNavigate();
  const conversationId = location.pathname.split("/")[2]
    ? location.pathname.split("/")[2]
    : "";
  const [chats, setChats] = useState([]);
  const [userEmail, setUserEmail] = useState("");
  const [gptModel, setGptModel] = useState("Search");
  const chatRef = useRef();

  const [selectedContract, setSelectedContract] = useState("");
  const [selectedPlan, setSelectedPlan] = useState("");
  const [selectedState, setSelectedState] = useState("");
  const [refreshToken, setRefreshToken] = useState("");
  const [error, setError] = useState("");
  const [input, setInput] = useState("");
  const [userImage, setUserImage] = useState("");
  const [isScrollable, setIsScrollable] = useState(false);

  axios.interceptors.request.use(setHeaders, (error) => {
    Promise.reject(error);
  });

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
          setGptModel(response.data.gptModel);
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
      setGptModel("Search");
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

    if (conversationId === "") {
      setChats([{ entered_query: input, response: "Loading Response" }]);
      let path = `/c/`;
      navigate(path);
    }

    let requestBody = {
      enteredQuery: input,
      contractType: selectedContract,
      selectedPlan: selectedPlan,
      gptModel: gptModel,
      selectedState: selectedState,
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
          setGptModel={setGptModel}
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
              setGptModel={setGptModel}
              selectedModel={gptModel}
              userEmail={userEmail}
            />

            {chats.length === 0 ? (
              <SamplePrompt
                gptModel={gptModel}
                input={input}
                setInput={setInput}
              />
            ) : (
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
            )}
          </div>
          <div className="inpufield_wrapper">
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
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
