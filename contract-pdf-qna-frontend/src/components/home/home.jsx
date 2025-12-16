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
import CaseReviewApprovePopup from "../caseReviewApprovePopup/caseReviewApprovePopup";
import { formatTranscriptDisplayName } from "../utils/transcriptName";

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
  const [authorizedFinalAnswer, setAuthorizedFinalAnswer] = useState("");
  const [authorizedApprovedAt, setAuthorizedApprovedAt] = useState(null);
  const [conversationStatus, setConversationStatus] = useState("active");
  const [callsTranscriptName, setCallsTranscriptName] = useState("");
  const [callsGenerationStage, setCallsGenerationStage] = useState("idle"); // idle | generating | done
  const [callsProgressText, setCallsProgressText] = useState("");
  const [callsActiveStep, setCallsActiveStep] = useState("extract"); // extract | answer | final
  const [callsGeneratedAt, setCallsGeneratedAt] = useState(null); // ISO string
  const [callsTotalQuestions, setCallsTotalQuestions] = useState(0);
  const [callsAnsweredCount, setCallsAnsweredCount] = useState(0);
  const [callsClaimDecision, setCallsClaimDecision] = useState(null);
  const [sidebarRefreshTick, setSidebarRefreshTick] = useState(0);
  const [isCheckingExistingTranscriptConversation, setIsCheckingExistingTranscriptConversation] =
    useState(false);
  const [existingTranscriptConversations, setExistingTranscriptConversations] = useState([]);
  const [loggedInUserName, setLoggedInUserName] = useState("");
  const [isReviewApproveOpen, setIsReviewApproveOpen] = useState(false);
  const [isApprovingCase, setIsApprovingCase] = useState(false);
  const [justApproved, setJustApproved] = useState(false);
  const [recentlyClosedConversationId, setRecentlyClosedConversationId] = useState("");

  useEffect(() => {
    // Pull display name from the Google login payload stored by SideBar.
    try {
      const raw = sessionStorage.getItem("payloadObject");
      if (!raw) return;
      const obj = JSON.parse(raw);
      const name = obj?.name || "";
      if (name) setLoggedInUserName(name);
    } catch (e) {
      // ignore
    }
  }, []);

  const hasFinalAnswerChat = chats?.some(
    (c) => c?.entered_query === "Final Answer for transcript"
  );

  axios.interceptors.request.use(setHeaders, (error) => {
    Promise.reject(error);
  });

  const handleSetGptModel = (model) => {
    // If leaving Claims (Calls) while a case is open, reset to "new chat" UI for the selected mode.
    // This ensures switching to Search/Infer doesn't keep showing the Claims case conversation.
    const isLeavingCalls = isCallsMode && model !== "Calls";
    if (isLeavingCalls) {
      setIsTranscriptModalOpen(false);
      setIsCheckingExistingTranscriptConversation(false);
      setCallsGenerationStage("idle");
      setCallsProgressText("");
      setCallsTranscriptName("");
      setCallsClaimDecision(null);
      setFinalSummary("");
      setAuthorizedFinalAnswer("");
      setAuthorizedApprovedAt(null);
      setConversationStatus("active");
      setChats([]);
      setInput("");
      // Exit the conversation route so `conversationId` becomes empty.
      navigate("/#");
    }

    // Keep Calls/Claims mode in sync with selected model
    setIsCallsMode(model === "Calls");
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
    // Prevent opening a new transcript while we are still streaming answers / summary
    if (callsGenerationStage === "generating") {
      return;
    }
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
    setCallsActiveStep("extract");
    setCallsGeneratedAt(null);
    setCallsProgressText("Starting transcript processing…");
    setCallsTotalQuestions(0);
    setCallsAnsweredCount(0);
    setCallsClaimDecision(null);
    setChats([]);
    setFinalSummary("");
    setIsTranscriptModalOpen(false);
    // Trigger sidebar refresh shortly after the backend receives the request (it now creates a processing stub early).
    setTimeout(() => setSidebarRefreshTick((t) => t + 1), 600);

    const runNonStreamingFallback = () => {
      setCallsActiveStep("answer");
      setCallsProgressText("Generating answers…");
      axios
        .post(`${TRANSCRIPTS_API_BASE_URL}/transcripts/process`, requestBody)
        .then((response) => {
          const conversationIdFromApi = response?.data?.conversationId || "";
          const questions = response?.data?.questions || [];
          const apiFinalSummary = response?.data?.finalSummary || "";
          const extractionWarning = response?.data?.warning;
          setFinalSummary(apiFinalSummary);
          setCallsClaimDecision(response?.data?.claimDecision || null);
          setAuthorizedFinalAnswer(apiFinalSummary);
        setAuthorizedApprovedAt(null);
        setConversationStatus((response?.data?.status || "active").toLowerCase());
          setCallsGenerationStage("done");
          setCallsActiveStep("final");
          setCallsProgressText("");
          setCallsGeneratedAt(new Date().toISOString());
          setCallsTranscriptName(
            response?.data?.transcriptMetadata?.fileName ||
              response?.data?.transcriptId ||
              transcript.name ||
              transcript.id
          );

          const mappedChats =
            questions.length > 0
              ? questions.map((q) => {
                  const chunks = q.relevantChunks || [];
                  const isFinal = q.questionId === "final_answer";
                  return {
                    entered_query: q.question,
                    response: q.answer,
                    chat_id: q.questionId,
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

          if (conversationIdFromApi) {
            navigate(`/conversation/${conversationIdFromApi}`);
          }
        })
        .catch((error) => {
          setCallsGenerationStage("idle");
          setCallsProgressText("");
          setChats([
            {
              entered_query: "",
              response: "An error occurred while processing your request.",
            },
          ]);
          console.error("Error processing transcript with /transcripts/process:", error);
        });
    };

    const runStreaming = async () => {
      const token = sessionStorage.getItem("idToken");
      if (!token) {
        runNonStreamingFallback();
        return;
      }

      const streamUrl = `${TRANSCRIPTS_API_BASE_URL}/transcripts/process/stream`;
      let conversationIdFromStream = "";

      try {
        const resp = await fetch(streamUrl, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: "Bearer " + token,
          },
          body: JSON.stringify(requestBody),
        });

        if (!resp.ok) {
          throw new Error(`Streaming request failed: ${resp.status}`);
        }
        if (!resp.body) {
          throw new Error("Streaming response body is not available");
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        const appendChat = (q) => {
          const chunks = q.relevantChunks || [];
          const isFinal = q.questionId === "final_answer";
          setChats((prev) => [
            ...(prev || []),
            {
              entered_query: q.question || "",
              response: q.answer || "",
              chat_id: q.questionId,
              questionId: q.questionId,
              questionType: q.questionType,
              userIntent: q.userIntent,
              relevant_chunks: chunks,
              underlying_model: requestBody.gptModel,
              source: isFinal ? "final_answer" : "transcript_extracted",
            },
          ]);
        };

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          buffer = buffer.replace(/\r\n/g, "\n");

          const parts = buffer.split("\n\n");
          buffer = parts.pop() || "";

          for (const part of parts) {
            const lines = part.split("\n").filter(Boolean);
            let eventType = "message";
            let dataStr = "";
            for (const line of lines) {
              if (line.startsWith("event:")) {
                eventType = line.slice(6).trim();
              } else if (line.startsWith("data:")) {
                dataStr += line.slice(5).trim();
              }
            }
            if (!dataStr) continue;

            let payload = null;
            try {
              payload = JSON.parse(dataStr);
            } catch (e) {
              payload = { raw: dataStr };
            }

            if (eventType === "status") {
              const stage = payload?.stage;
              if (stage === "started") {
                setCallsActiveStep("extract");
                setCallsProgressText("Starting transcript processing…");
              }
              if (stage === "conversation_created") {
                conversationIdFromStream = payload?.conversationId || "";
                setConversationStatus((payload?.status || "active").toLowerCase());
                setCallsActiveStep("extract");
                setCallsProgressText("Preparing workspace…");
              }
              if (stage === "cached") {
                // Cached path: answers will stream quickly; still show activity.
                const convId = payload?.conversationId || "";
                if (convId) conversationIdFromStream = convId;
                setConversationStatus((payload?.status || "active").toLowerCase());
                setCallsActiveStep("answer");
                setCallsProgressText("Loading cached results…");
              }
              if (stage === "transcript_loading") {
                setCallsActiveStep("extract");
                setCallsProgressText("Loading transcript from GCS…");
              }
              if (stage === "transcript_loaded") {
                const fn = payload?.transcriptMetadata?.fileName;
                if (fn) setCallsTranscriptName(fn);
                setCallsActiveStep("extract");
                setCallsProgressText("Transcript loaded. Analyzing…");
              }
              if (stage === "extracting_questions") {
                setCallsActiveStep("extract");
                setCallsProgressText("Extracting relevant customer questions…");
              }
              if (stage === "questions_ready") {
                const total = Number(payload?.totalQuestions || 0);
                setCallsTotalQuestions(total);
                setCallsAnsweredCount(0);
                setCallsActiveStep("answer");
                if (payload?.warning) {
                  setCallsProgressText(`${payload.warning} Generating answer…`);
                } else {
                  setCallsProgressText(
                    total > 0 ? `Found ${total} question(s). Generating answers…` : "Generating answers…"
                  );
                }
              }
              if (stage === "initializing_retriever") {
                setCallsActiveStep("answer");
                setCallsProgressText("Loading knowledge base (Milvus)…");
              }
              if (stage === "answering") {
                setCallsActiveStep("answer");
                setCallsProgressText("Generating answers…");
              }
              if (stage === "answering_question") {
                const idx = Number(payload?.index || 0);
                const total = Number(callsTotalQuestions || payload?.totalQuestions || 0);
                const label = total > 0 ? `Generating answer ${idx} of ${total}…` : `Generating answer ${idx}…`;
                setCallsActiveStep("answer");
                setCallsProgressText(label);
              }
            } else if (eventType === "answer") {
              appendChat(payload || {});
              setCallsAnsweredCount((prev) => {
                const next = (prev || 0) + 1;
                const total = callsTotalQuestions || 0;
                if (total > 0) {
                  if (next < total) {
                    setCallsActiveStep("answer");
                    setCallsProgressText(`Received answer ${next} of ${total}. Generating next…`);
                  } else {
                    setCallsActiveStep("final");
                    setCallsProgressText("Generating final summary…");
                  }
                } else {
                  setCallsActiveStep("final");
                  setCallsProgressText("Generating final summary…");
                }
                return next;
              });
            } else if (eventType === "final") {
              setFinalSummary(payload?.finalSummary || "");
              setCallsActiveStep("final");
              setCallsProgressText("Final summary ready. Finishing…");
            } else if (eventType === "claimDecision") {
              setCallsClaimDecision(payload || null);
            } else if (eventType === "done") {
              setCallsGenerationStage("done");
              setCallsActiveStep("final");
              setCallsProgressText("");
              setCallsGeneratedAt(new Date().toISOString());
              setSidebarRefreshTick((t) => t + 1);
              if (conversationIdFromStream) {
                navigate(`/conversation/${conversationIdFromStream}`);
              }
            } else if (eventType === "error") {
              const msg = payload?.error || payload?.message || "Streaming error";
              throw new Error(msg);
            }
          }
        }
      } catch (err) {
        console.error("Streaming transcript processing failed, falling back:", err);
        runNonStreamingFallback();
      }
    };

    runStreaming();
  };

  const handleSelectTranscript = (transcript) => {
    if (!transcript) return;
    // Close the picker and check Mongo for existing conversations (blocking)
    setIsTranscriptModalOpen(false);
    setIsCheckingExistingTranscriptConversation(true);

    axios
      .get(`${TRANSCRIPTS_API_BASE_URL}/transcripts/conversations`, {
        params: { transcriptFileName: transcript.id },
      })
      .then((resp) => {
        const convs = resp?.data?.conversations || [];
        if (Array.isArray(convs) && convs.length > 0) {
          // Requirement: always open the existing conversation if found.
          const convId = convs?.[0]?.conversationId || "";
          if (convId) {
            setIsCallsMode(true);
            setGptModelState("Calls");
            navigate(`/conversation/${convId}`);
            return;
          }
        }
        // No existing -> normal flow (create / reuse backend behavior)
        startNewCallsConversation(transcript, { newConversation: false });
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
          setCallsClaimDecision(response?.data?.claimDecision || null);
          setAuthorizedFinalAnswer(
            response.data.authorizedFinalAnswer || response.data.finalSummary || ""
          );
          setAuthorizedApprovedAt(response.data.authorizedApprovedAt || null);
          setConversationStatus((response.data.status || "active").toLowerCase());
          setCallsGeneratedAt(response.data.updatedAt || response.data.createdAt || null);

          const transcriptNameFromApi =
            response?.data?.transcriptMetadata?.fileName ||
            response?.data?.transcriptId ||
            "";
          if (transcriptNameFromApi) {
            setCallsTranscriptName(transcriptNameFromApi);
            setCallsGenerationStage("done");
            // If backend doesn't supply timestamps, fall back to "now" so the UI can still show something.
            if (!response.data.updatedAt && !response.data.createdAt) {
              setCallsGeneratedAt(new Date().toISOString());
            }
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
      setAuthorizedFinalAnswer("");
      setAuthorizedApprovedAt(null);
      setConversationStatus("active");
      setCallsTranscriptName("");
      setCallsGenerationStage("idle");
      setCallsClaimDecision(null);
      setCallsGeneratedAt(null);
    }
  }, [conversationId]);

  const extractedQuestionsForReview = (chats || [])
    .filter((c) => {
      const id = c?.questionId || c?.chat_id;
      if (id === "final_answer") return false;
      if (c?.source === "transcript_extracted") return true;
      return typeof id === "string" && /^q\d+$/i.test(id);
    })
    .map((c) => {
      const evidence = c?.relevantChunks || c?.relevant_chunks || [];
      return {
        id: c?.chat_id || c?.questionId,
        question: c?.entered_query || "",
        answer: c?.response || "",
        evidenceCount: Array.isArray(evidence) ? evidence.length : 0,
      };
    });

  const handleApproveCase = () => {
    if (!conversationId) return;
    if (!authorizedFinalAnswer?.trim()) return;
    setIsApprovingCase(true);
    axios
      .patch(`${API_BASE_URL}/conversation/authorize?conversation-id=${conversationId}`, {
        authorizedFinalAnswer: authorizedFinalAnswer,
        status: "inactive",
      })
      .then((resp) => {
        setConversationStatus("inactive");
        setAuthorizedApprovedAt(resp?.data?.authorizedApprovedAt || new Date().toISOString());
        setIsReviewApproveOpen(false);
        setJustApproved(true);
        // Signal sidebar to immediately move this case into "Closed" (optimistic UX).
        setRecentlyClosedConversationId(conversationId);
        setSidebarRefreshTick((t) => t + 1);
        setTimeout(() => setJustApproved(false), 2500);
        setTimeout(() => setRecentlyClosedConversationId(""), 3500);
      })
      .catch((err) => {
        console.error("Error approving case:", err);
      })
      .finally(() => {
        setIsApprovingCase(false);
      });
  };

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

    // Do not allow chatting on a closed case.
    if (isCallsMode && conversationId && conversationStatus === "inactive") {
      return;
    }

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
          recentlyClosedConversationId={recentlyClosedConversationId}
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
              isCallsGenerating={isCallsMode && callsGenerationStage === "generating"}
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
            ) : isCallsMode &&
              conversationId === "" &&
              chats.length === 0 &&
              !callsTranscriptName ? (
              <div className="calls_intro">
                <div className="calls_intro_card">
                  <div className="calls_intro_badge">Claims</div>
                  <div className="calls_intro_title">
                    Upload a transcript to generate coverage-focused answers
                  </div>
                  <div className="calls_intro_subtitle">
                    Select a claim transcript to automatically extract key coverage / repair questions,
                    generate answers, and attach the supporting evidence used for each response.
                  </div>

                  <div className="calls_intro_steps">
                    <div className="calls_intro_step">
                      <div className="label">1. Add transcript</div>
                      <div className="text">
                        Choose a transcript from the searchable, paginated list.
                      </div>
                    </div>
                    <div className="calls_intro_step">
                      <div className="label">2. Process & extract</div>
                      <div className="text">
                        Relevant customer questions are extracted into clear, reviewable prompts.
                      </div>
                    </div>
                    <div className="calls_intro_step">
                      <div className="label">3. Answers + referenced chunks</div>
                      <div className="text">
                        Answers are generated with the most relevant policy chunks so validation is fast.
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : chats.length > 0 || (isCallsMode && callsTranscriptName) ? (
              <div
                className={`chat_container  ${isScrollable ? "setHeight" : ""}`}
                ref={chatRef}
              >
                <>
                  {isCallsMode && callsTranscriptName ? (
                    <div className="calls_transcript_header">
                      <div className="header_row">
                        <div className="title">
                          Case transcript:{" "}
                          {(() => {
                            const display = formatTranscriptDisplayName(callsTranscriptName);
                            return (
                              <span className="file" title={display.raw || callsTranscriptName}>
                                {display.primary || callsTranscriptName}
                              </span>
                            );
                          })()}
                        </div>
                        {conversationId ? (
                          <div className="header_actions">
                            <div
                              className={`status_badge ${
                                conversationStatus === "inactive" ? "closed" : "open"
                              }`}
                              title={
                                conversationStatus === "inactive"
                                  ? "Closed"
                                  : "Open"
                              }
                            >
                              {conversationStatus === "inactive" ? "Closed" : "Open"}
                            </div>
                            <button
                              type="button"
                              className="review_approve_button"
                              onClick={() => setIsReviewApproveOpen(true)}
                              disabled={conversationStatus === "inactive"}
                              title={
                                conversationStatus === "inactive"
                                  ? "Case is closed."
                                  : "Review the final output and proceed."
                              }
                            >
                              Review &amp; Proceed
                            </button>
                          </div>
                        ) : null}
                      </div>
                      {callsGenerationStage === "generating" ? (
                        <div className="calls_stepper" aria-label="Processing steps">
                          <div className={`step ${callsActiveStep === "extract" ? "active" : ""} ${callsActiveStep !== "extract" ? "done" : ""}`}>
                            {callsActiveStep === "extract" ? (
                              <span className="mini_spinner" aria-hidden="true" />
                            ) : (
                              <span className="mini_check" aria-hidden="true">✓</span>
                            )}
                            Extract questions
                          </div>
                          <div className={`step ${callsActiveStep === "answer" ? "active" : ""} ${callsActiveStep === "final" ? "done" : ""}`}>
                            {callsActiveStep === "answer" ? (
                              <span className="mini_spinner" aria-hidden="true" />
                            ) : callsActiveStep === "final" ? (
                              <span className="mini_check" aria-hidden="true">✓</span>
                            ) : null}
                            Generate answers
                          </div>
                          <div className={`step ${callsActiveStep === "final" ? "active" : ""}`}>
                            {callsActiveStep === "final" ? <span className="mini_spinner" aria-hidden="true" /> : null}
                            Build final draft
                          </div>
                        </div>
                      ) : (callsGenerationStage === "done" || conversationStatus === "inactive") ? (
                        <div className="subtle_hint">
                          {(() => {
                            const ts = authorizedApprovedAt || callsGeneratedAt;
                            if (!ts) return null;
                            const label =
                              conversationStatus === "inactive" ? "Closed" : "Generated";
                            return (
                              <span className="approved_at">
                                {label}:{" "}
                                {new Date(ts).toLocaleString(undefined, {
                                  year: "numeric",
                                  month: "short",
                                  day: "2-digit",
                                  hour: "2-digit",
                                  minute: "2-digit",
                                })}
                              </span>
                            );
                          })()}
                        </div>
                      ) : null}
                    </div>
                  ) : null}
                  {isCallsMode && callsClaimDecision ? (
                    <div
                      className={`calls_decision ${(() => {
                        const raw = String(callsClaimDecision?.decision || "");
                        const slug = raw
                          .toLowerCase()
                          .replace(/[^a-z0-9]+/g, "_")
                          .replace(/^_+|_+$/g, "");

                        const positive = ["approve", "approved", "accept", "accepted", "yes", "covered"];
                        const negative = ["deny", "denied", "reject", "rejected", "no", "not_covered"];
                        const review = [
                          "cannot_determine",
                          "cant_determine",
                          "unknown",
                          "indeterminate",
                          "needs_review",
                          "review",
                          "partial",
                          "maybe",
                        ];

                        const tone = positive.includes(slug)
                          ? "positive"
                          : negative.includes(slug)
                            ? "negative"
                            : review.includes(slug)
                              ? "review"
                              : "neutral";

                        return `calls_decision_${slug || "unknown"} calls_decision_tone_${tone}`;
                      })()}`}
                    >
                      <div className="headline">
                        <span className="label">Decision:</span>{" "}
                        <span className="value">{callsClaimDecision.decision || "—"}</span>
                      </div>
                      {callsClaimDecision.shortAnswer ? (
                        <div className="short">{callsClaimDecision.shortAnswer}</div>
                      ) : null}
                      {Array.isArray(callsClaimDecision.reasons) && callsClaimDecision.reasons.length > 0 ? (
                        <ul className="reasons">
                          {callsClaimDecision.reasons.slice(0, 4).map((r, idx) => (
                            <li key={idx}>{r}</li>
                          ))}
                        </ul>
                      ) : null}
                      {Array.isArray(callsClaimDecision.citedChunks) && callsClaimDecision.citedChunks.length > 0 ? (
                        <div className="cite">
                          Based on {callsClaimDecision.citedChunks.length} policy chunk(s).
                        </div>
                      ) : null}
                    </div>
                  ) : null}
                  <ChatList
                    chats={chats}
                    setChats={setChats}
                    conversationId={conversationId}
                    isCallsMode={isCallsMode}
                  />
                  {isCallsMode && callsGenerationStage === "generating" ? (
                    <div className="calls_progress" aria-live="polite">
                      <span className="mini_spinner" aria-hidden="true" />
                      <div className="text">{callsProgressText || "Generating…"}</div>
                    </div>
                  ) : null}
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
          <div
            className={`inpufield_wrapper ${
              isCallsMode && conversationId && conversationStatus === "inactive" ? "disabled" : ""
            }`}
          >
            {isCallsMode && conversationId === "" ? (
              <button
                type="button"
                className="add_transcript_button"
                onClick={handleOpenTranscriptModal}
                disabled={callsGenerationStage === "generating"}
              >
                Add Transcript
              </button>
            ) : isCallsMode && conversationId && conversationStatus === "inactive" ? (
              <div className="chat_disabled_banner" role="status" aria-live="polite">
                Chat disabled — this case is closed.
              </div>
            ) : (
              <>
                {isCallsMode && conversationId ? (
                  null
                ) : null}
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
                  disabled={isCallsMode && conversationId && conversationStatus === "inactive"}
                  placeholder={
                    isCallsMode && conversationId && conversationStatus === "inactive"
                      ? "Case is closed. Chat is disabled."
                      : undefined
                  }
                />
              </>
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

        <CaseReviewApprovePopup
          isOpen={isReviewApproveOpen}
          onClose={() => setIsReviewApproveOpen(false)}
          onApprove={handleApproveCase}
          caseId={conversationId}
          transcriptName={callsTranscriptName}
          // Case ID in the popup should match exactly what comes from GCS (raw filename).
          caseName={callsTranscriptName || conversationId}
          metadata={{
            state: selectedState,
            contractType: selectedContract,
            plan: selectedPlan,
          }}
          decision={callsClaimDecision}
          aiFinalDraft={finalSummary}
          authorizedAnswer={authorizedFinalAnswer}
          setAuthorizedAnswer={setAuthorizedFinalAnswer}
          isApproving={isApprovingCase}
          isClosed={conversationStatus === "inactive"}
          userName={loggedInUserName}
        />

        {justApproved ? (
          <div className="case_thankyou_toast" role="status" aria-live="polite">
            Thank you, case forwarded.
          </div>
        ) : null}
      </div>
    </div>
  );
};

export default Home;
