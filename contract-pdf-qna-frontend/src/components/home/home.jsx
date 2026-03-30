import "regenerator-runtime";
import axios from "axios";
import { useCallback, useEffect, useRef, useState } from "react";
import PropTypes from "prop-types";
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
import { getIdToken, getPayloadObjectRaw } from "../../utils/authStorage";
import "./home.scss";
import ChatList from "../chatList/chatList";
import CallsTranscriptModal from "../callsTranscriptModal/callsTranscriptModal";
import CaseReviewApprovePopup from "../caseReviewApprovePopup/caseReviewApprovePopup";
import { formatTranscriptDisplayName } from "../utils/transcriptName";
import { ItemizedFinalAnswer } from "../common/itemizedFinalAnswer/itemizedFinalAnswer";
import TryAgainButton from "../common/tryAgainButton/tryAgainButton";

const TRANSCRIPTS_PAGE_SIZE = 16;
const HISTORY_POLL_INITIAL_DELAY_MS = 5000;
const HISTORY_POLL_INTERVAL_MS = 5000;

const Home = ({ bearerToken, setBearerToken }) => {
  const location = useLocation();
  let navigate = useNavigate();
  const conversationId = location.pathname.split("/")[2]
    ? location.pathname.split("/")[2]
    : "";
  // Prevent stale async completions (chat requests / transcript processing) from overwriting the UI
  // after the user navigates to a different conversation or mode.
  const viewKeyRef = useRef(0);
  const callsStreamAbortRef = useRef(null);
  const callsProcessRunIdRef = useRef(0);
  const chatAbortRef = useRef(null);
  const chatRequestIdRef = useRef(0);
  const transcriptViewerAbortRef = useRef(null);
  const transcriptViewerRequestIdRef = useRef(0);
  const checkExistingAbortRef = useRef(null);
  const checkExistingRequestIdRef = useRef(0);

  // Scoped loader overlay: blur ONLY the right panel below the mode bar.
  const mainPanelRef = useRef(null);
  const modeBarRef = useRef(null);
  const [rightPanelOverlayStyle, setRightPanelOverlayStyle] = useState(null);

  // Latest-request-wins guards (avoid older responses overwriting UI)
  const historyAbortRef = useRef(null);
  const historyRequestIdRef = useRef(0);
  const transcriptsAbortRef = useRef(null);
  const transcriptsRequestIdRef = useRef(0);
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
  const [isTranscriptViewerOpen, setIsTranscriptViewerOpen] = useState(false);
  const [isTranscriptViewerLoading, setIsTranscriptViewerLoading] =
    useState(false);
  const [transcriptViewerContent, setTranscriptViewerContent] = useState("");
  const [transcriptViewerError, setTranscriptViewerError] = useState(null);
  const [transcripts, setTranscripts] = useState([]);
  const [transcriptSearch, setTranscriptSearch] = useState("");
  // Transcript list status filter (modal)
  const [transcriptStatusFilter, setTranscriptStatusFilter] =
    useState("active");
  const [isLoadingTranscripts, setIsLoadingTranscripts] = useState(false);
  const [isLoadingMoreTranscripts, setIsLoadingMoreTranscripts] =
    useState(false);
  const [transcriptsOffset, setTranscriptsOffset] = useState(0);
  const [transcriptsHasMore, setTranscriptsHasMore] = useState(true);
  const [finalSummary, setFinalSummary] = useState("");
  const [authorizedFinalAnswer, setAuthorizedFinalAnswer] = useState("");
  const [authorizedApprovedAt, setAuthorizedApprovedAt] = useState(null);
  const [caseDisposition, setCaseDisposition] = useState("");
  const [caseClosedAt, setCaseClosedAt] = useState(null);
  const [conversationStatus, setConversationStatus] = useState("active");
  const [callsTranscriptName, setCallsTranscriptName] = useState("");
  const [callsGenerationStage, setCallsGenerationStage] = useState("idle"); // idle | generating | done
  const [callsProgressText, setCallsProgressText] = useState("");
  const [callsActiveStep, setCallsActiveStep] = useState("extract"); // extract | answer | final
  const [callsGeneratedAt, setCallsGeneratedAt] = useState(null); // ISO string
  const [callsTotalQuestions, setCallsTotalQuestions] = useState(0);
  const [, setCallsAnsweredCount] = useState(0);
  const [callsClaimDecision, setCallsClaimDecision] = useState(null);
  const [isCallsProcessing, setIsCallsProcessing] = useState(false);
  const [sidebarRefreshTick, setSidebarRefreshTick] = useState(0);
  const [
    isCheckingExistingTranscriptConversation,
    setIsCheckingExistingTranscriptConversation,
  ] = useState(false);
  const [isLoadingConversation, setIsLoadingConversation] = useState(false);
  const [loggedInUserName, setLoggedInUserName] = useState("");
  const [isReviewApproveOpen, setIsReviewApproveOpen] = useState(false);
  const [summaryEditLog, setSummaryEditLog] = useState([]);
  const [isApprovingCase, setIsApprovingCase] = useState(false);
  const [isRejectingCase, setIsRejectingCase] = useState(false);
  const [justApproved, setJustApproved] = useState(false);
  const [justRejected, setJustRejected] = useState(false);
  const [recentlyClosedConversationId, setRecentlyClosedConversationId] =
    useState("");
  const transcriptSearchDebounceRef = useRef(null);
  // Error state for 500 status codes
  const [serverError, setServerError] = useState(null); // { type: 'chat' | 'transcript' | 'sidebar' | 'conversation', retryFn: function }
  const lastFailedRequestRef = useRef(null); // Store last failed request details for retry
  // Avoid UI flicker: when we navigate to /conversation/:id right after creating/answering,
  // we already have the chats in memory, so skip the immediate /history refetch + chat reset once.
  const skipNextHistoryFetchRef = useRef(null);

  useEffect(() => {
    // Any navigation or mode switch invalidates in-flight async updates.
    viewKeyRef.current += 1;
    // Abort any in-progress transcript stream so it can't update the newly opened conversation.
    try {
      if (callsStreamAbortRef.current) callsStreamAbortRef.current.abort();
    } catch {
      // ignore
    } finally {
      callsStreamAbortRef.current = null;
    }
    // Abort any in-flight chat request (Search/Infer/Calls followup)
    try {
      if (chatAbortRef.current) chatAbortRef.current.abort();
    } catch {
      // ignore
    } finally {
      chatAbortRef.current = null;
    }
    // Abort transcript viewer load
    try {
      if (transcriptViewerAbortRef.current)
        transcriptViewerAbortRef.current.abort();
    } catch {
      // ignore
    } finally {
      transcriptViewerAbortRef.current = null;
    }
    // Abort "check existing conversations" request
    try {
      if (checkExistingAbortRef.current) checkExistingAbortRef.current.abort();
    } catch {
      // ignore
    } finally {
      checkExistingAbortRef.current = null;
    }
  }, [conversationId, gptModel, isCallsMode]);

  useEffect(() => {
    if (!isLoadingConversation) {
      setRightPanelOverlayStyle(null);
      return;
    }

    const update = () => {
      const mainEl = mainPanelRef.current;
      const barEl = modeBarRef.current;
      if (!mainEl || !barEl) return;
      const mainRect = mainEl.getBoundingClientRect();
      const barRect = barEl.getBoundingClientRect();
      setRightPanelOverlayStyle({
        top: Math.max(0, Math.round(barRect.bottom)),
        left: Math.max(0, Math.round(mainRect.left)),
      });
    };

    // Measure after paint so layout is stable.
    const raf = window.requestAnimationFrame(update);
    window.addEventListener("resize", update);
    // Capture scrolls in nested containers too.
    window.addEventListener("scroll", update, true);
    return () => {
      window.cancelAnimationFrame(raf);
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
    };
  }, [isLoadingConversation]);

  useEffect(() => {
    // Pull display name from the Google login payload stored by SideBar.
    try {
      const raw = getPayloadObjectRaw();
      if (!raw) return;
      const obj = JSON.parse(raw);
      const name = obj?.name || "";
      if (name) setLoggedInUserName(name);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    // If a protected route redirects here, reflect it in the existing error/highlight UX.
    const q = new URLSearchParams(location.search);
    const err = q.get("error");
    if (err === "login") setError("login");
  }, [location.search]);

  const hasFinalAnswerChat = chats?.some(
    (c) => c?.entered_query === "Final Answer for transcript",
  );
  const canReviewProceed =
    Boolean(conversationId) &&
    (hasFinalAnswerChat || Boolean((finalSummary || "").trim())) &&
    callsGenerationStage !== "generating" &&
    !isCallsProcessing;

  axios.interceptors.request.use(setHeaders, (error) => {
    Promise.reject(error);
  });

  const handleSetGptModel = (model) => {
    // If clicking on the same mode that's already active, do nothing
    if (gptModel === model) {
      return;
    }

    // Reset all state when switching modes to start a new chat/case
    setIsTranscriptModalOpen(false);
    setIsTranscriptViewerOpen(false);
    setIsTranscriptViewerLoading(false);
    setTranscriptViewerContent("");
    setTranscriptViewerError(null);
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
    // This ensures switching modes opens a new chat/case.
    navigate("/#");

    // Keep Calls/Claims mode in sync with selected model
    setIsCallsMode(model === "Calls");
    setGptModelState(model);
  };

  const fetchTranscripts = useCallback(
    (
      searchTerm = "",
      status = transcriptStatusFilter,
      offset = 0,
      append = false,
    ) => {
      // Latest-wins: cancel any previous transcripts list request
      try {
        if (transcriptsAbortRef.current) transcriptsAbortRef.current.abort();
      } catch {
        // ignore
      }
      const requestId = (transcriptsRequestIdRef.current += 1);
      const abortController = new AbortController();
      transcriptsAbortRef.current = abortController;

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
        .get(`${TRANSCRIPTS_API_BASE_URL}/transcripts`, {
          params,
          signal: abortController.signal,
        })
        .then((response) => {
          if (transcriptsRequestIdRef.current !== requestId) return;
          setServerError(null);
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

          setTranscripts((prev) =>
            append ? [...prev, ...mappedTranscripts] : mappedTranscripts,
          );
          setTranscriptsHasMore(hasMore);
          setTranscriptsOffset(offset + limit);
        })
        .catch((error) => {
          if (error?.name === "CanceledError" || error?.code === "ERR_CANCELED") {
            return;
          }
          if (transcriptsRequestIdRef.current !== requestId) return;
          console.error("Error fetching transcripts:", error);
          const status = error?.response?.status;
          if (status === 500) {
            setServerError({
              type: "transcript",
              retryFn: () =>
                fetchTranscripts(searchTerm, status, offset, append),
            });
            lastFailedRequestRef.current = {
              searchTerm,
              status,
              offset,
              append,
            };
          }
        })
        .finally(() => {
          if (transcriptsRequestIdRef.current !== requestId) return;
          setIsLoadingTranscripts(false);
          setIsLoadingMoreTranscripts(false);
        });
    },
    [transcriptStatusFilter],
  );

  const handleOpenTranscriptModal = () => {
    // Prevent opening a new transcript while we are still streaming answers / summary
    if (callsGenerationStage === "generating") {
      return;
    }
    if (!getIdToken()) {
      setError("login");
      return;
    }
    // Reset transcript state to null when modal opens
    setTranscripts([]);
    setTranscriptSearch("");
    setTranscriptsOffset(0);
    setTranscriptsHasMore(true);
    setServerError(null);
    // Opening the transcript modal implies we are in Calls mode
    setIsCallsMode(true);
    setGptModelState("Calls");
    setIsTranscriptModalOpen(true);
    fetchTranscripts("", transcriptStatusFilter, 0, false);
  };

  const handleToggleTranscriptViewer = async () => {
    if (isTranscriptViewerOpen) {
      try {
        if (transcriptViewerAbortRef.current)
          transcriptViewerAbortRef.current.abort();
      } catch {
        // ignore
      } finally {
        transcriptViewerAbortRef.current = null;
      }
      setIsTranscriptViewerOpen(false);
      return;
    }
    if (!callsTranscriptName) return;

    const transcriptUrl = `${TRANSCRIPTS_API_BASE_URL}/transcripts/${encodeURIComponent(
      callsTranscriptName,
    )}`;
    console.log(transcriptUrl);
    // Cancel any previous in-flight transcript fetch; latest click wins.
    try {
      if (transcriptViewerAbortRef.current)
        transcriptViewerAbortRef.current.abort();
    } catch {
      // ignore
    }
    const requestId = (transcriptViewerRequestIdRef.current += 1);
    const abortController = new AbortController();
    transcriptViewerAbortRef.current = abortController;
    setIsTranscriptViewerOpen(true);
    setIsTranscriptViewerLoading(true);
    setTranscriptViewerContent("");
    setTranscriptViewerError(null);

    try {
      const resp = await axios.get(transcriptUrl, {
        signal: abortController.signal,
      });
      if (transcriptViewerRequestIdRef.current !== requestId) return;
      const payload = resp?.data || {};
      const content =
        payload?.textContent ||
        payload?.content ||
        payload?.parsedData?.content ||
        payload?.parsedData?.transcript ||
        payload?.parsedData?.text ||
        "";
      setTranscriptViewerContent(content);
    } catch (err) {
      // Ignore cancels (user clicked again / navigated).
      if (err?.name === "CanceledError" || err?.code === "ERR_CANCELED") return;
      console.error("Error fetching transcript from GCS:", err);
      setTranscriptViewerError("Unable to load transcript. Please try again.");
    } finally {
      if (transcriptViewerRequestIdRef.current === requestId) {
        setIsTranscriptViewerLoading(false);
      }
    }
  };

  const handleTranscriptSearchChange = (value) => {
    setTranscriptSearch(value);
    setTranscriptsOffset(0);
    setTranscriptsHasMore(true);

    // Clear existing debounce timeout
    if (transcriptSearchDebounceRef.current) {
      clearTimeout(transcriptSearchDebounceRef.current);
    }

    // Set new debounce timeout (300ms delay)
    transcriptSearchDebounceRef.current = setTimeout(() => {
      fetchTranscripts(value, transcriptStatusFilter, 0, false);
    }, 300);
  };

  const handleTranscriptStatusChange = (status) => {
    setTranscriptStatusFilter(status);
    setTranscriptsOffset(0);
    setTranscriptsHasMore(true);
    fetchTranscripts(transcriptSearch, status, 0, false);
  };

  const startNewCallsConversation = (transcript, opts = {}) => {
    if (!transcript) return;
    const viewKeyAtStart = viewKeyRef.current;
    const runId = (callsProcessRunIdRef.current += 1);

    // If a previous transcript processing request is still running, cancel it.
    try {
      if (callsStreamAbortRef.current) callsStreamAbortRef.current.abort();
    } catch {
      // ignore
    } finally {
      callsStreamAbortRef.current = null;
    }

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
        transcript,
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
    setCallsProgressText("Creating case…");
    setCallsTotalQuestions(0);
    setCallsAnsweredCount(0);
    setCallsClaimDecision(null);
    setChats([]);
    setFinalSummary("");
    setIsTranscriptModalOpen(false);

    let stubConversationId = "";
    const processAbortController = new AbortController();
    callsStreamAbortRef.current = processAbortController;
    const createStub = async () => {
      try {
        const resp = await axios.post(
          `${TRANSCRIPTS_API_BASE_URL}/transcripts/conversation/stub`,
          {
            transcriptFileName: transcript.id,
            contractType,
            selectedPlan: planName,
            selectedState: stateName,
            gptModel: requestBody.gptModel,
            newConversation: Boolean(opts?.newConversation),
            conversationName: transcript.name || transcript.id,
          },
          { signal: processAbortController.signal },
        );
        stubConversationId = resp?.data?.conversationId || "";
        if (stubConversationId) {
          requestBody.conversationId = stubConversationId;
          // Sidebar should show the in-progress case immediately (yellow dot).
          setSidebarRefreshTick((t) => t + 1);
        }
      } catch (e) {
        if (e?.name === "CanceledError" || e?.code === "ERR_CANCELED") return;
        // Fail open: the stream endpoint will create the stub if this call fails.
        stubConversationId = "";
      }
    };

    const runNonStreamingFallback = () => {
      if (viewKeyRef.current !== viewKeyAtStart) return;
      if (callsProcessRunIdRef.current !== runId) return;
      setCallsActiveStep("answer");
      setCallsProgressText("Generating answers…");
      axios
        .post(`${TRANSCRIPTS_API_BASE_URL}/transcripts/process`, requestBody, {
          signal: processAbortController.signal,
        })
        .then((response) => {
          if (viewKeyRef.current !== viewKeyAtStart) return;
          if (callsProcessRunIdRef.current !== runId) return;
          const conversationIdFromApi = response?.data?.conversationId || "";
          const questions = response?.data?.questions || [];
          const apiFinalSummary = response?.data?.finalSummary || "";
          const extractionWarning = response?.data?.warning;
          setFinalSummary(apiFinalSummary);
          setCallsClaimDecision(response?.data?.claimDecision || null);
          setAuthorizedFinalAnswer(apiFinalSummary);
          setAuthorizedApprovedAt(null);
          setConversationStatus(
            (response?.data?.status || "active").toLowerCase(),
          );
          setCallsGenerationStage("done");
          setCallsActiveStep("final");
          setCallsProgressText("");
          setCallsGeneratedAt(new Date().toISOString());
          setCallsTranscriptName(
            response?.data?.transcriptMetadata?.fileName ||
              response?.data?.transcriptId ||
              transcript.name ||
              transcript.id,
          );

          const mappedChats =
            questions.length > 0
              ? questions.map((q) => {
                  const chunks = q.relevantChunks || [];
                  const chunksDetail = q.relevantChunksDetail || q.relevant_chunks_detail || [];
                  const isFinal = q.questionId === "final_answer";
                  return {
                    entered_query: q.question,
                    response: q.answer,
                    chat_id: q.questionId,
                    questionId: q.questionId,
                    questionType: q.questionType,
                    userIntent: q.userIntent,
                    relevant_chunks: chunks,
                    relevant_chunks_detail: chunksDetail,
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
          if (viewKeyRef.current !== viewKeyAtStart) return;
          if (callsProcessRunIdRef.current !== runId) return;
          if (error?.name === "CanceledError" || error?.code === "ERR_CANCELED")
            return;
          const status = error?.response?.status;
          setCallsGenerationStage("idle");
          setCallsProgressText("");
          const errorMessage =
            status === 500
              ? "An error occurred while processing your request. Please try again."
              : "An error occurred while processing your request.";

          setChats([
            {
              entered_query: "",
              response: errorMessage,
              isError: status === 500,
            },
          ]);

          if (status === 500) {
            setServerError({
              type: "transcript",
              retryFn: () => {
                // Retry transcript processing
                const transcript = {
                  id: requestBody.transcriptFileName,
                  name: requestBody.transcriptFileName,
                };
                startNewCallsConversation(transcript, {
                  newConversation: false,
                });
              },
            });
          }
          console.error(
            "Error processing transcript with /transcripts/process:",
            error,
          );
        });
    };

    const runStreaming = async () => {
      const token = getIdToken();
      if (!token) {
        runNonStreamingFallback();
        return;
      }

      // Ensure Mongo stub exists BEFORE heavy processing begins so sidebar updates first.
      await createStub();
      if (viewKeyRef.current !== viewKeyAtStart) return;
      if (callsProcessRunIdRef.current !== runId) return;
      setCallsProgressText("Starting transcript processing…");

      const streamUrl = `${TRANSCRIPTS_API_BASE_URL}/transcripts/process/stream`;
      let conversationIdFromStream = "";
      const abortController = processAbortController;

      try {
        const resp = await fetch(streamUrl, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: "Bearer " + token,
          },
          body: JSON.stringify(requestBody),
          signal: abortController.signal,
        });

        if (!resp.ok) {
          const status = resp.status;
          if (status === 500) {
            setServerError({
              type: "transcript",
              retryFn: () => {
                const transcript = {
                  id: requestBody.transcriptFileName,
                  name: requestBody.transcriptFileName,
                };
                startNewCallsConversation(transcript, {
                  newConversation: false,
                });
              },
            });
          }
          throw new Error(`Streaming request failed: ${status}`);
        }
        if (!resp.body) {
          throw new Error("Streaming response body is not available");
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        const appendChat = (q) => {
          if (viewKeyRef.current !== viewKeyAtStart) return;
          if (callsProcessRunIdRef.current !== runId) return;
          const chunks = q.relevantChunks || [];
          const chunksDetail = q.relevantChunksDetail || q.relevant_chunks_detail || [];
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
              relevant_chunks_detail: chunksDetail,
              underlying_model: requestBody.gptModel,
              source: isFinal ? "final_answer" : "transcript_extracted",
            },
          ]);
        };

        while (true) {
          if (viewKeyRef.current !== viewKeyAtStart) {
            try {
              abortController.abort();
            } catch {
              // ignore
            }
            break;
          }
          if (callsProcessRunIdRef.current !== runId) {
            try {
              abortController.abort();
            } catch {
              // ignore
            }
            break;
          }
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
            } catch {
              payload = { raw: dataStr };
            }

            if (eventType === "status") {
              if (viewKeyRef.current !== viewKeyAtStart) continue;
              if (callsProcessRunIdRef.current !== runId) continue;
              const stage = payload?.stage;
              if (stage === "started") {
                setCallsActiveStep("extract");
                setCallsProgressText("Starting transcript processing…");
                // If backend included a conversationId (new processing path), refresh sidebar immediately
                // so the in-progress case shows up with the correct (yellow) status dot.
                if (payload?.conversationId) {
                  setSidebarRefreshTick((t) => t + 1);
                }
              }
              if (stage === "conversation_created") {
                conversationIdFromStream = payload?.conversationId || "";
                setConversationStatus(
                  (payload?.status || "active").toLowerCase(),
                );
                setCallsActiveStep("extract");
                setCallsProgressText("Preparing workspace…");
                // Now that the backend has created/updated the Mongo stub, refresh sidebar so it appears immediately.
                setSidebarRefreshTick((t) => t + 1);
              }
              if (stage === "cached") {
                // Cached path: answers will stream quickly; still show activity.
                const convId = payload?.conversationId || "";
                if (convId) conversationIdFromStream = convId;
                setConversationStatus(
                  (payload?.status || "active").toLowerCase(),
                );
                setCallsActiveStep("answer");
                setCallsProgressText("Loading cached results…");
                // Ensure cached/in-progress conversations are visible in sidebar.
                setSidebarRefreshTick((t) => t + 1);
              }
              if (stage === "transcript_loading") {
                setCallsActiveStep("extract");
                setCallsProgressText("Loading transcript…");
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
                    total > 0
                      ? `Found ${total} question(s). Generating answers…`
                      : "Generating answers…",
                  );
                }
              }
              if (stage === "initializing_retriever") {
                setCallsActiveStep("answer");
                setCallsProgressText("Loading…");
              }
              if (stage === "answering") {
                setCallsActiveStep("answer");
                setCallsProgressText("Generating answers…");
              }
              if (stage === "answering_question") {
                const idx = Number(payload?.index || 0);
                const total = Number(
                  callsTotalQuestions || payload?.totalQuestions || 0,
                );
                const label =
                  total > 0
                    ? `Generating answer ${idx} of ${total}…`
                    : `Generating answer ${idx}…`;
                setCallsActiveStep("answer");
                setCallsProgressText(label);
              }
            } else if (eventType === "answer") {
              if (viewKeyRef.current !== viewKeyAtStart) continue;
              if (callsProcessRunIdRef.current !== runId) continue;
              appendChat(payload || {});
              setCallsAnsweredCount((prev) => {
                const next = (prev || 0) + 1;
                const total = callsTotalQuestions || 0;
                if (total > 0) {
                  if (next < total) {
                    setCallsActiveStep("answer");
                    setCallsProgressText(
                      `Received answer ${next} of ${total}. Generating next…`,
                    );
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
              if (viewKeyRef.current !== viewKeyAtStart) continue;
              if (callsProcessRunIdRef.current !== runId) continue;
              setFinalSummary(payload?.finalSummary || "");
              setCallsActiveStep("final");
              setCallsProgressText("Final summary ready. Finishing…");
            } else if (eventType === "claimDecision") {
              if (viewKeyRef.current !== viewKeyAtStart) continue;
              if (callsProcessRunIdRef.current !== runId) continue;
              setCallsClaimDecision(payload || null);
            } else if (eventType === "done") {
              if (viewKeyRef.current !== viewKeyAtStart) continue;
              if (callsProcessRunIdRef.current !== runId) continue;
              setCallsGenerationStage("done");
              setCallsActiveStep("final");
              setCallsProgressText("");
              setCallsGeneratedAt(new Date().toISOString());
              setSidebarRefreshTick((t) => t + 1);
              if (conversationIdFromStream) {
                navigate(`/conversation/${conversationIdFromStream}`);
              }
            } else if (eventType === "error") {
              const msg =
                payload?.error || payload?.message || "Streaming error";
              throw new Error(msg);
            }
          }
        }
      } catch (err) {
        if (viewKeyRef.current !== viewKeyAtStart) return;
        if (callsProcessRunIdRef.current !== runId) return;
        if (err?.name === "AbortError") return;
        console.error(
          "Streaming transcript processing failed, falling back:",
          err,
        );
        const status =
          err?.response?.status || (err?.message?.includes("500") ? 500 : null);
        if (status === 500 && !serverError) {
          setServerError({
            type: "transcript",
            retryFn: () => {
              const transcript = {
                id: requestBody.transcriptFileName,
                name: requestBody.transcriptFileName,
              };
              startNewCallsConversation(transcript, { newConversation: false });
            },
          });
        }
        runNonStreamingFallback();
      } finally {
        if (callsStreamAbortRef.current === abortController) {
          callsStreamAbortRef.current = null;
        }
      }
    };

    runStreaming();
  };

  const handleSelectTranscript = (transcript) => {
    if (!transcript) return;
    // Cancel any in-flight "existing conversations" check; latest selection wins.
    try {
      if (checkExistingAbortRef.current) checkExistingAbortRef.current.abort();
    } catch {
      // ignore
    }
    const requestId = (checkExistingRequestIdRef.current += 1);
    const abortController = new AbortController();
    checkExistingAbortRef.current = abortController;
    // Close the picker and check Mongo for existing conversations (blocking)
    setIsTranscriptModalOpen(false);
    setIsCheckingExistingTranscriptConversation(true);

    axios
      .get(`${TRANSCRIPTS_API_BASE_URL}/transcripts/conversations`, {
        params: { transcriptFileName: transcript.id },
        signal: abortController.signal,
      })
      .then((resp) => {
        if (checkExistingRequestIdRef.current !== requestId) return;
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
        if (err?.name === "CanceledError" || err?.code === "ERR_CANCELED")
          return;
        console.error("Error checking transcript conversations:", err);
        // Fail open: proceed with normal flow
        startNewCallsConversation(transcript, { newConversation: false });
      })
      .finally(() => {
        if (checkExistingRequestIdRef.current === requestId) {
          setIsCheckingExistingTranscriptConversation(false);
        }
      });
  };

  const runHistoryFetch = useCallback(
    (cid, opts = {}) => {
      const id = String(cid || "");
      if (!id) return;
      const clearChats = opts?.clearChats !== false;

      // Latest-wins: cancel previous /history request
      try {
        if (historyAbortRef.current) historyAbortRef.current.abort();
      } catch {
        // ignore
      }
      const requestId = (historyRequestIdRef.current += 1);
      const abortController = new AbortController();
      historyAbortRef.current = abortController;

      setIsLoadingConversation(true);
      if (clearChats) setChats([]);

      const apiUrl = `${API_BASE_URL}/history?conversation-id=${id}`;
      axios
        .get(apiUrl, { signal: abortController.signal })
        .then((response) => {
          if (historyRequestIdRef.current !== requestId) return;
          setServerError(null);
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
          setSummaryEditLog(response.data.summaryEditLog || []);
          setCallsClaimDecision(response?.data?.claimDecision || null);
          const isProcessing = Boolean(response?.data?.processing);
          const hasFinal = Boolean((response?.data?.finalSummary || "").trim());
          const shouldBlockForProcessing =
            isProcessing ||
            (Boolean(
              response?.data?.transcriptId || response?.data?.transcriptMetadata,
            ) &&
              !hasFinal);
          setIsCallsProcessing(shouldBlockForProcessing);
          if (shouldBlockForProcessing) {
            setCallsProgressText("Analyzing transcript…");
            setCallsActiveStep("final");
          }
          setAuthorizedFinalAnswer(
            response.data.authorizedFinalAnswer || response.data.finalSummary || "",
          );
          setAuthorizedApprovedAt(response.data.authorizedApprovedAt || null);
          setConversationStatus((response.data.status || "active").toLowerCase());
          setCallsGeneratedAt(
            response.data.updatedAt || response.data.createdAt || null,
          );

          const transcriptNameFromApi =
            response?.data?.transcriptMetadata?.fileName ||
            response?.data?.transcriptId ||
            "";
          if (transcriptNameFromApi) {
            setCallsTranscriptName(transcriptNameFromApi);
            setCallsGenerationStage(shouldBlockForProcessing ? "generating" : "done");
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
          if (error?.name === "CanceledError" || error?.code === "ERR_CANCELED") {
            return;
          }
          if (historyRequestIdRef.current !== requestId) return;
          console.error("Error:", error);
          const status = error?.response?.status;
          if (status === 500) {
            setServerError({
              type: "conversation",
              retryFn: () => runHistoryFetch(id, { clearChats: false }),
            });
          }
        })
        .finally(() => {
          setIsLoadingConversation(false);
        });
    },
    [],
  );

  useEffect(() => {
    if (conversationId !== "") {
      runHistoryFetch(conversationId, { clearChats: true });
    } else {
      try {
        if (historyAbortRef.current) historyAbortRef.current.abort();
      } catch {
        // ignore
      } finally {
        historyAbortRef.current = null;
      }
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
      setIsCallsProcessing(false);
      setIsLoadingConversation(false);
    }
  }, [conversationId, gptModel, runHistoryFetch]);

  // If a transcript conversation is still processing (cloud analysis), keep showing the loader and
  // poll /history until processing is false and finalSummary is available.
  useEffect(() => {
    if (!isCallsMode) return;
    if (!conversationId) return;
    if (!callsTranscriptName) return;
    if (!isCallsProcessing) return;
    if (isLoadingConversation) return;

    const viewKeyAtStart = viewKeyRef.current;
    let cancelled = false;

    const poll = async () => {
      if (cancelled) return;
      if (viewKeyRef.current !== viewKeyAtStart) return;
      try {
        const resp = await axios.get(
          `${API_BASE_URL}/history?conversation-id=${conversationId}`,
        );
        if (cancelled) return;
        if (viewKeyRef.current !== viewKeyAtStart) return;

        const isProcessing = Boolean(resp?.data?.processing);
        const hasFinal = Boolean((resp?.data?.finalSummary || "").trim());
        const shouldBlockForProcessing =
          isProcessing ||
          (Boolean(
            resp?.data?.transcriptId || resp?.data?.transcriptMetadata,
          ) &&
            !hasFinal);

        setIsCallsProcessing(shouldBlockForProcessing);
        // Always keep chats up-to-date while processing (ChatGPT-style incremental updates).
        setChats(resp?.data?.chats || []);
        setFinalSummary(resp?.data?.finalSummary || "");
        setSummaryEditLog(resp?.data?.summaryEditLog || []);
        setCallsClaimDecision(resp?.data?.claimDecision || null);
        setAuthorizedFinalAnswer(
          resp?.data?.authorizedFinalAnswer || resp?.data?.finalSummary || "",
        );
        setAuthorizedApprovedAt(resp?.data?.authorizedApprovedAt || null);
        setConversationStatus((resp?.data?.status || "active").toLowerCase());
        setCaseDisposition(resp?.data?.caseDisposition || "");
        setCaseClosedAt(resp?.data?.closedAt || null);

        if (!shouldBlockForProcessing) {
          setCallsGenerationStage("done");
          setCallsProgressText("");
          // Ensure sidebar dot flips from yellow->green as soon as processing completes.
          setSidebarRefreshTick((t) => t + 1);
          return;
        }
        setCallsGenerationStage("generating");
        setCallsProgressText("Analyzing transcript…");
      } catch {
        // Ignore transient errors while polling.
      }
      setTimeout(poll, HISTORY_POLL_INTERVAL_MS);
    };

    const t = setTimeout(poll, HISTORY_POLL_INITIAL_DELAY_MS);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [
    isCallsMode,
    conversationId,
    callsTranscriptName,
    isCallsProcessing,
    isLoadingConversation,
  ]);

  const handleApproveCase = (reviewComments = "") => {
    if (!conversationId) return;
    if (!authorizedFinalAnswer?.trim()) return;
    setIsApprovingCase(true);
    axios
      .patch(
        `${API_BASE_URL}/conversation/authorize?conversation-id=${conversationId}`,
        {
          authorizedFinalAnswer: authorizedFinalAnswer,
          status: "inactive",
          reviewComments: reviewComments || "",
        },
      )
      .then((resp) => {
        setConversationStatus("inactive");
        setCaseDisposition(resp?.data?.caseDisposition || "approved");
        setCaseClosedAt(resp?.data?.closedAt || new Date().toISOString());
        setAuthorizedApprovedAt(
          resp?.data?.authorizedApprovedAt || new Date().toISOString(),
        );
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

  const handleSaveDraftSummary = (previousSummary, updatedSummary, changes = []) => {
    if (!conversationId) return Promise.reject(new Error("No conversation"));
    const body = {
      finalSummary: updatedSummary,
      previousSummary: previousSummary,
    };
    if (Array.isArray(changes) && changes.length > 0) {
      body.changes = changes.map((c) => ({
        fieldName: c.fieldName,
        previousValue: c.previousValue,
        updatedValue: c.updatedValue,
      }));
    }
    const newSummary = updatedSummary;
    return axios
      .patch(`${API_BASE_URL}/conversation/draft-summary?conversation-id=${conversationId}`, body)
      .then((resp) => {
        const savedSummary = resp?.data?.finalSummary ?? newSummary;
        setFinalSummary(savedSummary);
        setSummaryEditLog(resp?.data?.summaryEditLog ?? []);
        setChats((prev) =>
          (prev || []).map((c) =>
            c?.questionId === "final_answer" || c?.entered_query === "Final Answer for transcript"
              ? { ...c, response: savedSummary }
              : c
          )
        );
      });
  };

  const handleRejectCase = (reviewComments = "") => {
    if (!conversationId) return;
    setIsRejectingCase(true);
    axios
      .patch(
        `${API_BASE_URL}/conversation/close?conversation-id=${conversationId}`,
        {
          disposition: "rejected",
          reviewComments: reviewComments || "",
        },
      )
      .then((resp) => {
        setConversationStatus("inactive");
        setCaseDisposition(resp?.data?.caseDisposition || "rejected");
        setCaseClosedAt(resp?.data?.closedAt || new Date().toISOString());
        setIsReviewApproveOpen(false);
        setJustRejected(true);
        // Signal sidebar to immediately move this case into "Closed" (optimistic UX).
        setRecentlyClosedConversationId(conversationId);
        setSidebarRefreshTick((t) => t + 1);
        setTimeout(() => setJustRejected(false), 2500);
        setTimeout(() => setRecentlyClosedConversationId(""), 3500);
      })
      .catch((err) => {
        console.error("Error rejecting case:", err);
      })
      .finally(() => {
        setIsRejectingCase(false);
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

  // Cleanup debounce timeout on unmount
  useEffect(() => {
    return () => {
      if (transcriptSearchDebounceRef.current) {
        clearTimeout(transcriptSearchDebounceRef.current);
      }
    };
  }, []);

  const handleInputSubmit = () => {
    const viewKeyAtSubmit = viewKeyRef.current;
    // Cancel any previous in-flight chat request; latest submit wins.
    try {
      if (chatAbortRef.current) chatAbortRef.current.abort();
    } catch {
      // ignore
    }
    const requestId = (chatRequestIdRef.current += 1);
    const abortController = new AbortController();
    chatAbortRef.current = abortController;
    if (!getIdToken()) {
      setError("login");
      return;
    }

    if (input === "") return;

    // Do not allow chatting on a closed Claims/Calls case.
    if (isCallsMode && conversationId && conversationStatus === "inactive") {
      return;
    }

    if (
      chats.length > 0 &&
      chats[chats.length - 1].response === "Loading Response"
    ) {
      return;
    }

    // Search/Infer require contract filters; Claims follow-ups should work from case context alone.
    if (!isCallsMode) {
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
      // Keep the current UI in-place while generating; do NOT navigate to /c/ (it causes a reset/flicker).
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
      const apiUrl = `${API_BASE_URL}/claims/followup?conversation-id=${conversationId}`;
      axios
        .post(
          apiUrl,
          {
            enteredQuery: input,
            // Ensure backend always has plan metadata for Milvus retrieval (and can persist it if missing).
            contractType: selectedContract,
            selectedPlan: selectedPlan,
            selectedState: selectedState,
          },
          { signal: abortController.signal },
        )
        .then((response) => {
          if (viewKeyRef.current !== viewKeyAtSubmit) return;
          if (chatRequestIdRef.current !== requestId) return;
          setServerError(null);
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
                gpt_model: "Calls",
                relevantChunks: response.data.relevantChunks,
                relevantChunksDetail: response.data.relevantChunksDetail,
              },
            ]);
          }
        })
        .catch((error) => {
          if (viewKeyRef.current !== viewKeyAtSubmit) return;
          if (chatRequestIdRef.current !== requestId) return;
          if (error?.name === "CanceledError" || error?.code === "ERR_CANCELED")
            return;
          const status = error?.response?.status;
          const errorMessage =
            status === 500
              ? "An error occurred while processing your request. Please try again."
              : "An error occurred while processing your request.";

          setChats((prevChats) => [
            ...prevChats.slice(0, -1),
            {
              entered_query: input,
              response: errorMessage,
              isError: status === 500,
            },
          ]);

          if (status === 500) {
            setServerError({
              type: "chat",
              retryFn: () => {
                const lastInput = input;
                setInput(lastInput);
                setTimeout(() => handleInputSubmit(), 100);
              },
            });
            lastFailedRequestRef.current = {
              type: "calls-chat",
              requestBody: { enteredQuery: input },
              apiUrl,
              input,
            };
          }
          console.error("Error:", error);
        });
    } else {
      requestBody = {
        ...requestBody,
        gptModel: gptModel,
      };
      const apiUrl = `${API_BASE_URL}/start?conversation-id=${conversationId}`;
      axios
        .post(apiUrl, requestBody, { signal: abortController.signal })
        .then((response) => {
          if (viewKeyRef.current !== viewKeyAtSubmit) return;
          if (chatRequestIdRef.current !== requestId) return;
          setServerError(null);
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
            const nextId = response?.data?.conversationId;
            if (nextId) {
              skipNextHistoryFetchRef.current = String(nextId);
              navigate(`/conversation/${nextId}`);
            }
          }
        })
        .catch((error) => {
          if (viewKeyRef.current !== viewKeyAtSubmit) return;
          if (chatRequestIdRef.current !== requestId) return;
          if (error?.name === "CanceledError" || error?.code === "ERR_CANCELED")
            return;
          const status = error?.response?.status;
          const errorMessage =
            status === 500
              ? "An error occurred while processing your request. Please try again."
              : "An error occurred while processing your request.";

          setChats((prevChats) => [
            ...prevChats.slice(0, -1),
            {
              entered_query: input,
              response: errorMessage,
              isError: status === 500,
            },
          ]);

          if (status === 500) {
            setServerError({
              type: "chat",
              retryFn: () => {
                const lastInput = input;
                setInput(lastInput);
                setTimeout(() => handleInputSubmit(), 100);
              },
            });
            lastFailedRequestRef.current = {
              type: "search-infer-chat",
              requestBody,
              apiUrl,
              input,
            };
          }
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
      <div className="main_container" ref={mainPanelRef}>
        <Header userIconImage={userImage} />
        <div className="chat_section_wrapper">
          <div className="chat_section">
            <div ref={modeBarRef}>
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
                isCallsGenerating={
                  isCallsMode && callsGenerationStage === "generating"
                }
                isClaimsHomepage={
                  isCallsMode && !conversationId
                }
                transcriptStatusFilter={transcriptStatusFilter}
                onTranscriptStatusChange={handleTranscriptStatusChange}
                conversationStatus={conversationStatus}
                isConversationActive={isCallsMode && conversationId !== ""}
                onConversationStatusChange={(status) => {
                  if (!conversationId) return;
                  axios
                    .patch(
                      `${API_BASE_URL}/conversation/status?conversation-id=${conversationId}`,
                      { status },
                    )
                    .then(() => {
                      setConversationStatus(status);
                    })
                    .catch((err) => {
                      console.error("Error updating conversation status:", err);
                    });
                }}
              />
            </div>

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
              <div className="prompt_wrapper calls_prompt_wrapper">
                <div className="title">How can I help you today?</div>
                <div className="subtitle">
                  Your AI-powered copilot is ready to assist you!
                </div>
                <div className="queries_part">
                  <div className="query">
                    Choose a case, generate coverage guidance, and prepare an
                    itemized authorization draft.
                  </div>
                </div>
                <div className="card_list">
                  <div className="card_container calls_landing_card">
                    <div className="topic">1. Choose a case</div>
                    <div className="prompt_info">
                      Choose a case from the list to analyze the customer
                      conversations.
                    </div>
                  </div>
                  <div className="card_container calls_landing_card">
                    <div className="topic">2. Claim Coverage Information</div>
                    <div className="prompt_info">
                      Summarizes coverage outcomes per item and attaches
                      referred contract clauses for quick validation.
                    </div>
                  </div>
                  <div className="card_container calls_landing_card">
                    <div className="topic">3. Authorization Information</div>
                    <div className="prompt_info">
                      Review the itemized final draft + decision, add comments,
                      then proceed &amp; close the case.
                    </div>
                  </div>
                </div>
              </div>
            ) : chats.length > 0 ||
              isLoadingConversation ||
              (isCallsMode && callsTranscriptName) ? (
              <div
                className={`chat_container  ${isScrollable ? "setHeight" : ""}`}
                ref={chatRef}
              >
                <>
                  {serverError?.type === "conversation" ? (
                    <div className="conversation_error">
                      <div className="error_text">
                        Failed to load conversation. Please try again.
                      </div>
                      <TryAgainButton
                        onRetry={() => {
                          if (serverError?.retryFn) {
                            setServerError(null);
                            serverError.retryFn();
                          }
                        }}
                      />
                    </div>
                  ) : null}
                  {isCallsMode && callsTranscriptName ? (
                    <div className="calls_transcript_header">
                      <div className="header_row">
                        <div className="title_block">
                          <span className="title_overline">Case transcript</span>
                          {(() => {
                            const display =
                              formatTranscriptDisplayName(callsTranscriptName);
                            return (
                              <h2 className="title_primary" title={display.raw || callsTranscriptName}>
                                {display.primary || callsTranscriptName}
                              </h2>
                            );
                          })()}
                        </div>
                        <div className="header_actions">
                          {caseDisposition ? (
                            <div
                              className={`disposition_badge ${
                                String(caseDisposition).toLowerCase() ===
                                "approved"
                                  ? "approved"
                                  : String(caseDisposition).toLowerCase() ===
                                      "rejected"
                                    ? "rejected"
                                    : "neutral"
                              }`}
                              title={`Disposition: ${caseDisposition}`}
                            >
                              {String(caseDisposition).toUpperCase()}
                            </div>
                          ) : null}
                          <button
                            type="button"
                            className="review_approve_button show_transcript_button"
                            onClick={handleToggleTranscriptViewer}
                          >
                            Show Transcript
                          </button>
                          {canReviewProceed ? (
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
                          ) : null}
                        </div>
                      </div>
                      <div className="case_meta_row" role="region" aria-label="Case details">
                        <div className="meta_item">
                          <span className="meta_label">State</span>
                          <span className="meta_value">{selectedState}</span>
                        </div>
                        <div className="meta_divider" aria-hidden="true" />
                        <div className="meta_item">
                          <span className="meta_label">Contract</span>
                          <span className="meta_value">{selectedContract}</span>
                        </div>
                        <div className="meta_divider" aria-hidden="true" />
                        <div className="meta_item">
                          <span className="meta_label">Plan</span>
                          <span className="meta_value">{selectedPlan}</span>
                        </div>
                        <div className="meta_divider" aria-hidden="true" />
                        <div
                          className={`meta_item meta_status ${conversationStatus === "inactive" ? "closed" : "open"}`}
                        >
                          <span className="meta_label">Status</span>
                          <span className="meta_value">
                            {callsGenerationStage === "generating"
                              ? "Processing…"
                              : conversationStatus === "inactive"
                                ? "Closed"
                                : "Open"}
                          </span>
                        </div>
                      </div>
                      {isCheckingExistingTranscriptConversation ||
                      isLoadingConversation ? (
                        <div
                          className="conversation_loading"
                          role="status"
                          aria-live="polite"
                        >
                          <span className="mini_spinner" aria-hidden="true" />
                          <div className="text">
                            {isCheckingExistingTranscriptConversation
                              ? "Checking existing conversations…"
                              : "Loading conversation…"}
                          </div>
                        </div>
                      ) : null}
                      {callsGenerationStage === "generating" ? (
                        <div
                          className="calls_stepper"
                          aria-label="Processing steps"
                        >
                          <div
                            className={`step ${callsActiveStep === "extract" ? "active" : ""} ${callsActiveStep !== "extract" ? "done" : ""}`}
                          >
                            {callsActiveStep === "extract" ? (
                              <span
                                className="mini_spinner"
                                aria-hidden="true"
                              />
                            ) : (
                              <span className="mini_check" aria-hidden="true">
                                ✓
                              </span>
                            )}
                            Extract questions
                          </div>
                          <div
                            className={`step ${callsActiveStep === "answer" ? "active" : ""} ${callsActiveStep === "final" ? "done" : ""}`}
                          >
                            {callsActiveStep === "answer" ? (
                              <span
                                className="mini_spinner"
                                aria-hidden="true"
                              />
                            ) : callsActiveStep === "final" ? (
                              <span className="mini_check" aria-hidden="true">
                                ✓
                              </span>
                            ) : null}
                            Generate answers
                          </div>
                          <div
                            className={`step ${callsActiveStep === "final" ? "active" : ""}`}
                          >
                            {callsActiveStep === "final" ? (
                              <span
                                className="mini_spinner"
                                aria-hidden="true"
                              />
                            ) : null}
                            Build final draft
                          </div>
                        </div>
                      ) : callsGenerationStage === "done" ||
                        conversationStatus === "inactive" ? (
                        <div className="generated_date_block">
                          {(() => {
                            const ts =
                              caseClosedAt ||
                              authorizedApprovedAt ||
                              callsGeneratedAt;
                            if (!ts) return null;
                            const label =
                              conversationStatus === "inactive"
                                ? "Closed on"
                                : "Generated on";
                            const d = new Date(ts);
                            const datePart = d.toLocaleDateString(undefined, {
                              weekday: "short",
                              month: "long",
                              day: "numeric",
                              year: "numeric",
                            });
                            const timePart = d.toLocaleTimeString(undefined, {
                              hour: "numeric",
                              minute: "2-digit",
                              hour12: true,
                            });
                            return (
                              <>
                                <span className="generated_date_label">
                                  {label}
                                </span>
                                <span className="generated_date_value">
                                  {datePart}
                                  <span className="generated_date_sep"> · </span>
                                  {timePart}
                                </span>
                              </>
                            );
                          })()}
                        </div>
                      ) : null}
                    </div>
                  ) : null}
                  <ChatList
                    chats={chats}
                    setChats={setChats}
                    conversationId={conversationId}
                    isCallsMode={isCallsMode}
                    claimDecision={callsClaimDecision}
                    serverError={serverError}
                    onRetryChat={() => {
                      if (
                        serverError?.type === "chat" &&
                        serverError?.retryFn
                      ) {
                        setServerError(null);
                        serverError.retryFn();
                      }
                    }}
                  />
                  {isCallsMode && callsGenerationStage === "generating" ? (
                    <div className="calls_progress" aria-live="polite">
                      <span className="mini_spinner" aria-hidden="true" />
                      <div className="text">
                        {callsProgressText || "Generating…"}
                      </div>
                    </div>
                  ) : null}
                  {isCallsMode && finalSummary && !hasFinalAnswerChat ? (
                    <div className="calls_summary">
                      <div className="title">Final Summary</div>
                      <ItemizedFinalAnswer
                        text={finalSummary}
                        title=""
                        asCard={false}
                      />
                    </div>
                  ) : null}
                </>
              </div>
            ) : null}
          </div>
          {isLoadingConversation && rightPanelOverlayStyle ? (
            <div
              className="right_panel_overlay"
              style={{
                top: rightPanelOverlayStyle.top,
                left: rightPanelOverlayStyle.left,
              }}
              role="status"
              aria-live="polite"
              aria-label="Loading conversation"
            >
              <div className="right_panel_overlay_card">
                <div className="spinner" aria-hidden="true" />
                <div className="text">Loading conversation…</div>
              </div>
            </div>
          ) : null}
          <div
            className={`inpufield_wrapper ${isCallsMode ? "inpufield_wrapper--claims" : ""} ${
              isCallsMode && conversationId && conversationStatus === "inactive"
                ? "disabled"
                : ""
            }`}
          >
            {isCallsMode && conversationId === "" ? (
              callsGenerationStage === "generating" ? (
                <InputField
                  listening={false}
                  transcript={""}
                  handleInputEnter={() => {}}
                  handleEnter={() => {}}
                  description={""}
                  setDescription={() => {}}
                  textareaRef={textareaRef}
                  onMicrophoneClick={() => {}}
                  disabled={true}
                  placeholder={""}
                />
              ) : (
                <button
                  type="button"
                  className="add_transcript_button"
                  onClick={handleOpenTranscriptModal}
                >
                  Select Case
                </button>
              )
            ) : isCallsMode &&
              conversationId &&
              conversationStatus === "inactive" ? (
              <div
                className="chat_disabled_banner"
                role="status"
                aria-live="polite"
              >
                Chat disabled — this case is closed.
              </div>
            ) : (
              <>
                {isCallsMode && conversationId ? null : null}
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
                  disabled={
                    isCallsMode &&
                    conversationId &&
                    conversationStatus === "inactive"
                  }
                  placeholder={
                    isCallsMode &&
                    conversationId &&
                    conversationStatus === "inactive"
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
          error={serverError?.type === "transcript" ? serverError : null}
          onRetry={() => {
            if (serverError?.type === "transcript" && serverError?.retryFn) {
              setServerError(null);
              serverError.retryFn();
            }
          }}
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
                    x.id === t.id ? { ...x, status: nextStatus } : x,
                  ),
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
            if (
              isLoadingTranscripts ||
              isLoadingMoreTranscripts ||
              !transcriptsHasMore
            )
              return;
            fetchTranscripts(
              transcriptSearch,
              transcriptStatusFilter,
              transcriptsOffset,
              true,
            );
          }}
        />

        <CaseReviewApprovePopup
          isOpen={isReviewApproveOpen}
          onClose={() => setIsReviewApproveOpen(false)}
          onApprove={handleApproveCase}
          onReject={handleRejectCase}
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
          isRejecting={isRejectingCase}
          isClosed={conversationStatus === "inactive"}
          userName={loggedInUserName}
          caseDisposition={caseDisposition}
          summaryEditLog={summaryEditLog}
          onSaveDraftSummary={handleSaveDraftSummary}
        />

        {isTranscriptViewerOpen ? (
          <div className="calls_modal_backdrop" role="dialog" aria-modal="true">
            <div className="calls_modal transcript_viewer_modal">
              <div className="calls_modal_header">
                <div className="title">Transcript</div>
                <button
                  type="button"
                  className="close_button"
                  onClick={() => setIsTranscriptViewerOpen(false)}
                  aria-label="Close transcript"
                >
                  ×
                </button>
              </div>
              <div className="calls_modal_body transcript_viewer_body">
                {isTranscriptViewerLoading ? (
                  <div className="loading">
                    <span className="spinner" aria-hidden="true" />
                    <span className="loading_text">Loading transcript…</span>
                  </div>
                ) : transcriptViewerError ? (
                  <div className="error_state">
                    <div className="error_text">{transcriptViewerError}</div>
                  </div>
                ) : (
                  <pre className="transcript_viewer_content">
                    {transcriptViewerContent || "No transcript content found."}
                  </pre>
                )}
              </div>
            </div>
          </div>
        ) : null}

        {justApproved ? (
          <div className="case_thankyou_toast" role="status" aria-live="polite">
            Thank you, case forwarded.
          </div>
        ) : null}
        {justRejected ? (
          <div className="case_thankyou_toast" role="status" aria-live="polite">
            Case rejected &amp; closed.
          </div>
        ) : null}
      </div>
    </div>
  );
};

export default Home;

Home.propTypes = {
  bearerToken: PropTypes.any,
  setBearerToken: PropTypes.func,
};
