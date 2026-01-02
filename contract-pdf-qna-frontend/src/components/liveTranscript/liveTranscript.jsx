import { useEffect, useMemo, useRef, useState } from "react";
import { io } from "socket.io-client";
import "./liveTranscript.scss";
import { BACKEND_BASE, CCP_URL, REGION } from "../../config";

const LiveTranscript = () => {
  const ccpContainerRef = useRef(null);
  const ccpInitializedRef = useRef(false);
  const [ccpInitError, setCcpInitError] = useState(null);
  const [ccpInitRequested, setCcpInitRequested] = useState(false);
  const [ccpInitInProgress, setCcpInitInProgress] = useState(false);
  
  // Refs for auto-scroll functionality
  const transcriptScrollerRef = useRef(null);
  const suggestionsScrollerRef = useRef(null);
  const shouldAutoScrollRef = useRef(true);

  const socket = useMemo(() => {
    return io(BACKEND_BASE, {
      transports: ["websocket", "polling"],
      reconnection: true,
      reconnectionAttempts: Infinity,
      reconnectionDelay: 500,
    });
  }, []);

  const [agentName, setAgentName] = useState("-");
  const [contactId, setContactId] = useState(null);
  const [callState, setCallState] = useState("IDLE");

  const [transcripts, setTranscripts] = useState([]);
  // Reflect Amazon Connect login/CCP readiness (NOT backend socket connection).
  const [isConnected, setIsConnected] = useState(false);

  // Live Copilot UI state (populated by backend `suggestion_update`)
  const [copilotUser, setCopilotUser] = useState(null);
  const [copilotCards, setCopilotCards] = useState([]);
  const [copilotStatus, setCopilotStatus] = useState(null);

  const getTurnRole = (speaker) => {
    const x = String(speaker ?? "")
      .trim()
      .toLowerCase();
    if (!x) return "unknown";
    if (
      x === "csr" ||
      x.includes("csr") ||
      x.includes("agent") ||
      x.includes("rep") ||
      x.includes("representative") ||
      x.includes("support")
    ) {
      return "agent";
    }
    if (
      x === "customer" ||
      x.includes("customer") ||
      x.includes("caller") ||
      x.includes("homeowner") ||
      x.includes("policyholder") ||
      x.includes("member")
    ) {
      return "customer";
    }
    return "unknown";
  };

  const seenRef = useRef(new Set());

  const formatOffsetToTime = (ms) => {
    const n = Number(ms);
    if (!Number.isFinite(n) || n < 0) return "—";
    const totalSeconds = Math.floor(n / 1000);
    const mm = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
    const ss = String(totalSeconds % 60).padStart(2, "0");
    return `${mm}:${ss}`;
  };

  // Format timestamp for suggestions (e.g., "2:34 PM")
  const formatSuggestionTime = (timestamp) => {
    if (!timestamp) return "";
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  };

  const isCallConnected = callState === "CONNECTED";

  const getEffectiveCcpUrl = () => {
    if (!CCP_URL) return CCP_URL;
    try {
      const u = new URL(CCP_URL);
      const p = u.pathname.replace(/\/+$/, "");
      if (p === "" || p === "/") u.pathname = "/ccp-v2/";
      else if (p === "/connect/ccp-v2") u.pathname = "/ccp-v2/";
      else if (p === "/connect/ccp") u.pathname = "/ccp/";
      return u.toString();
    } catch {
      return CCP_URL;
    }
  };

  const initCcp = () => {
    if (!ccpContainerRef.current || !window.connect) {
      setCcpInitError(
        "Amazon Connect Streams not loaded (window.connect missing)."
      );
      setCcpInitInProgress(false);
      return;
    }

    const container = ccpContainerRef.current;
    const effectiveCcpUrl = getEffectiveCcpUrl();

    if (!effectiveCcpUrl) {
      setCcpInitError("CCP URL is missing. Set VITE_CCP_URL.");
      setCcpInitInProgress(false);
      return;
    }

    if (ccpInitializedRef.current) return;
    if (container.querySelector("iframe")) return;
    ccpInitializedRef.current = true;
    setCcpInitError(null);
    setCcpInitInProgress(true);

    try {
      window.connect.core.initCCP(container, {
        ccpUrl: effectiveCcpUrl,
        loginPopup: true,
        loginPopupAutoClose: true,
        ccpAckTimeout: 15000,
        region: REGION,
        softphone: {
          allowFramedSoftphone: true,
        },
      });

      window.connect.agent((agent) => {
        setAgentName(agent.getName ? agent.getName() : "Agent");
        setIsConnected(true);
        setCcpInitInProgress(false);
        // Best-effort: keep status in sync if Streams exposes state changes.
        try {
          const update = () => {
            // `getState()` exists on agent in Streams; if unavailable, keep "connected" once agent resolves.
            const s = agent.getState ? agent.getState() : null;
            if (s) setIsConnected(true);
          };
          update();
          if (agent.onStateChange) agent.onStateChange(update);
        } catch {
          // ignore
        }
      });

      window.connect.contact((contact) => {
        const id = contact.getContactId ? contact.getContactId() : null;
        if (id) {
          setContactId(id);
          setCallState("CONTACT_CREATED");
          socket.emit("join_session", { sessionId: id });
          // reset transcript view for new call
          seenRef.current.clear();
          seenSuggestionsRef.current.clear(); // Clear suggestion dedup set for new call
          setTranscripts([]);
          setCopilotUser(null);
          setCopilotCards([]);
          setCopilotStatus(null);
        }

        if (contact.onConnecting) {
          contact.onConnecting(() => setCallState("CONNECTING"));
        }
        if (contact.onConnected) {
          contact.onConnected(() => {
            setCallState("CONNECTED");
            // Enable copilot ONLY when call is connected (Analyze Live tab requirement)
            try {
              const sid = contact.getContactId ? contact.getContactId() : id;
              if (sid) socket.emit("copilot_enable", { sessionId: sid });
            } catch {
              // ignore
            }
          });
        }
        if (contact.onEnded) {
          contact.onEnded(() => {
            setCallState("ENDED");
            // Disable copilot when call ends
            try {
              const sid = contact.getContactId ? contact.getContactId() : id;
              if (sid) socket.emit("copilot_disable", { sessionId: sid });
            } catch {
              // ignore
            }
          });
        }
      });

      // Best-effort: if Streams reports auth issues, mark disconnected.
      try {
        if (window.connect?.core?.onAuthFail) {
          window.connect.core.onAuthFail(() => setIsConnected(false));
        }
      } catch {
        // ignore
      }
    } catch (e) {
      ccpInitializedRef.current = false;
      setCcpInitError(e?.message || String(e));
      setIsConnected(false);
      setCcpInitInProgress(false);
    }
  };

  const handleCcpLoginClick = () => {
    setCcpInitRequested(true);
    setCcpInitError(null);
    // Streams opens a named popup; pre-open it on user gesture to avoid popup blockers.
    try {
      const loginPopupName =
        window.connect?.MasterTopics?.LOGIN_POPUP || "connect::loginPopup";
      window.open("about:blank", loginPopupName, "width=420,height=720");
    } catch {
      // ignore
    }
    // Now initialize CCP (still requires user gesture for the popup).
    initCcp();
  };

  useEffect(() => {
    const handler = (msg) => {
      // Only show current call's transcript (sessionId == ContactId)
      if (!contactId) return;
      if (msg?.sessionId !== contactId) return;

      const key = [
        msg?.sessionId,
        msg?.speaker,
        msg?.beginOffsetMillis,
        msg?.endOffsetMillis,
        msg?.text,
        msg?.isPartial,
      ]
        .map((x) => String(x ?? ""))
        .join("|");

      if (seenRef.current.has(key)) return;
      seenRef.current.add(key);

      setTranscripts((prev) => [
        ...prev,
        {
          speaker: msg?.speaker || "UNKNOWN",
          text: msg?.text || "",
          begin: msg?.beginOffsetMillis ?? null,
          end: msg?.endOffsetMillis ?? null,
          isPartial: !!msg?.isPartial,
        },
      ]);
    };

    socket.on("transcript_update", handler);
    return () => {
      socket.off("transcript_update", handler);
    };
  }, [socket, contactId]);

  // Track seen suggestion keys to avoid duplicates
  const seenSuggestionsRef = useRef(new Set());

  useEffect(() => {
    const handler = (msg) => {
      // Only show current call's copilot suggestions
      if (!contactId) return;
      if (msg?.sessionId !== contactId) return;

      // Accept flexible payload shapes, prefer `customer` + `cards`
      const customer = msg?.customer || msg?.user || null;
      const newCards = Array.isArray(msg?.cards)
        ? msg.cards
        : Array.isArray(msg?.suggestions)
        ? msg.suggestions
        : [];

      setCopilotUser(customer);
      
      // ACCUMULATE suggestions - keep history of suggestions during the call
      // Add timestamp and unique ID to each new card for tracking
      setCopilotCards((prevCards) => {
        const updatedCards = [...prevCards];
        
        newCards.forEach((card) => {
          // Create a unique key for deduplication
          const cardKey = [
            card?.title || "",
            card?.csrScript || card?.text || "",
            card?.evidence || "",
          ].join("|");
          
          // Only add if not already seen
          if (!seenSuggestionsRef.current.has(cardKey)) {
            seenSuggestionsRef.current.add(cardKey);
            updatedCards.push({
              ...card,
              timestamp: Date.now(),
              id: `suggestion-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            });
          }
        });
        
        // Keep most recent 10 suggestions to avoid clutter
        return updatedCards.slice(-10);
      });
    };

    const statusHandler = (msg) => {
      if (!contactId) return;
      if (msg?.sessionId && msg?.sessionId !== contactId) return;
      setCopilotStatus(msg);
    };

    socket.on("suggestion_update", handler);
    socket.on("copilot_status", statusHandler);
    return () => {
      socket.off("suggestion_update", handler);
      socket.off("copilot_status", statusHandler);
    };
  }, [socket, contactId]);

  useEffect(() => {
    const container = ccpContainerRef.current;
    return () => {
      ccpInitializedRef.current = false;
      if (container) container.innerHTML = "";
      setIsConnected(false);
      try {
        socket?.disconnect?.();
      } catch {
        // ignore
      }
    };
  }, [socket]);

  // Auto-scroll transcripts to bottom when new messages arrive
  useEffect(() => {
    if (transcriptScrollerRef.current && shouldAutoScrollRef.current) {
      const scroller = transcriptScrollerRef.current;
      // Smooth scroll to bottom
      scroller.scrollTo({
        top: scroller.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [transcripts]);

  // Auto-scroll suggestions panel when new suggestions arrive
  useEffect(() => {
    if (suggestionsScrollerRef.current && copilotCards.length > 0) {
      const scroller = suggestionsScrollerRef.current;
      // Scroll to top since newest suggestions are displayed first (reversed order)
      scroller.scrollTo({
        top: 0,
        behavior: "smooth",
      });
    }
  }, [copilotCards]);

  // Handle manual scroll detection for transcript scroller
  const handleTranscriptScroll = () => {
    if (!transcriptScrollerRef.current) return;
    const scroller = transcriptScrollerRef.current;
    const { scrollTop, scrollHeight, clientHeight } = scroller;
    // If user is within 100px of bottom, enable auto-scroll; otherwise disable
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 100;
    shouldAutoScrollRef.current = isNearBottom;
  };

  return (
    <div className="live_transcript_layout">
      {/* LEFT: Amazon Connect CCP */}
      <aside className="lt_left_ccp">
        <div className="live_transcript_header lt_left_ccp_header">
          <div className="title">Amazon Connect CCP</div>
          <div className={`lt_conn_pill ${isConnected ? "lt_conn_pill--connected" : "lt_conn_pill--disconnected"}`}>
            {isConnected ? "Connected" : "Disconnected"}
          </div>
        </div>

        {!isConnected ? (
          <div className="live_transcript_card lt_ccp_login">
            <div className="label">AMAZON CONNECT</div>
            <div className="lt_ccp_login_body">
              <div className="lt_ccp_login_text">
                {ccpInitRequested
                  ? ccpInitInProgress
                    ? "Waiting for Amazon Connect login…"
                    : "Complete login in the popup, then return here."
                  : "Click to login, then the CCP panel will appear here."}
              </div>
              <button
                type="button"
                className="back_button lt_ccp_login_button"
                onClick={handleCcpLoginClick}
                disabled={ccpInitInProgress}
              >
                {ccpInitInProgress ? "Logging in…" : "Login to Amazon Connect"}
              </button>
              {ccpInitError ? (
                <div className="lt_ccp_error">{ccpInitError}</div>
              ) : null}
            </div>
          </div>
        ) : null}

        <div className="live_transcript_card lt_ccp_meta">
          <div className="lt_kv">
            <span className="k">Agent:</span>
            <span className="v">{agentName}</span>
          </div>
          <div className="lt_kv">
            <span className="k">Call State:</span>
            <span className="v">{callState}</span>
          </div>
          {contactId ? (
            <div className="lt_kv">
              <span className="k">ContactId / SessionId:</span>
              <span className="v mono" title={contactId}>
                {contactId}
              </span>
            </div>
          ) : null}
        </div>

        {/* IMPORTANT: keep the container mounted so initCCP() has a DOM node to attach to.
            Otherwise clicking "Login" would only open about:blank and initCCP() would no-op. */}
        <div className="live_transcript_card lt_ccp_frame">
          <div ref={ccpContainerRef} id="ccp-container" />
        </div>
      </aside>

      {/* CENTER: Transcript */}
      <main className="lt_center">
        <div className="live_transcript_center_body">
          <div className="live_transcript_card lt_transcript_card">
            {/* Streaming indicator - shows when call is active */}
            {isCallConnected && (
              <div className="lt_transcript_status_bar">
                <div className="lt_streaming_badge">
                  <span className="lt_streaming_dot"></span>
                  Live
                </div>
              </div>
            )}
            <div 
              className="lt_transcript_scroller"
              ref={transcriptScrollerRef}
              onScroll={handleTranscriptScroll}
            >
              {transcripts.length === 0 ? (
                !isConnected ? (
                  // Not logged in - Show login steps
                  <div className="lt_transcript_empty_state">
                    <div className="lt_transcript_empty_icon">
                      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M22 16.92V19.92C22.0011 20.1985 21.9441 20.4742 21.8325 20.7293C21.7209 20.9845 21.5573 21.2136 21.3521 21.4019C21.1468 21.5901 20.9046 21.7335 20.6407 21.8227C20.3769 21.9119 20.0974 21.9451 19.82 21.92C16.7428 21.5856 13.787 20.5341 11.19 18.85C8.77382 17.3147 6.72533 15.2662 5.18999 12.85C3.49997 10.2412 2.44824 7.27099 2.11999 4.18C2.095 3.90347 2.12787 3.62476 2.21649 3.36162C2.30512 3.09849 2.44756 2.85669 2.63476 2.65162C2.82196 2.44655 3.0498 2.28271 3.30379 2.17052C3.55777 2.05833 3.83233 2.00026 4.10999 2H7.10999C7.5953 1.99522 8.06579 2.16708 8.43376 2.48353C8.80173 2.79999 9.04207 3.23945 9.10999 3.72C9.23662 4.68007 9.47144 5.62273 9.80999 6.53C9.94454 6.88792 9.97366 7.27691 9.8939 7.65088C9.81415 8.02485 9.62886 8.36811 9.35999 8.64L8.08999 9.91C9.51355 12.4135 11.5765 14.4765 14.08 15.9L15.35 14.63C15.6219 14.3611 15.9651 14.1758 16.3391 14.0961C16.7131 14.0163 17.1021 14.0454 17.46 14.18C18.3673 14.5185 19.3099 14.7534 20.27 14.88C20.7558 14.9485 21.1996 15.1907 21.5177 15.5627C21.8359 15.9347 22.0057 16.4108 22 16.92Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                      </svg>
                    </div>
                    <div className="lt_transcript_empty_title">Waiting for Connection</div>
                    <div className="lt_transcript_empty_subtitle">
                      Please login to Amazon Connect to start receiving calls
                    </div>
                    <div className="lt_transcript_empty_steps">
                      <div className="lt_step_item">
                        <span className="lt_step_number">1</span>
                        <span className="lt_step_text">Login to Amazon Connect CCP</span>
                      </div>
                      <div className="lt_step_item">
                        <span className="lt_step_number">2</span>
                        <span className="lt_step_text">Set your status to Available</span>
                      </div>
                      <div className="lt_step_item">
                        <span className="lt_step_number">3</span>
                        <span className="lt_step_text">Accept incoming call</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  // Logged in but no active call - Ready to listen
                  <div className="lt_transcript_ready_state">
                    <div className="lt_ready_icon_container">
                      <div className="lt_ready_icon">
                        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                          <path d="M12 1C12 1 12 1 12 1C5.92487 1 1 5.92487 1 12C1 18.0751 5.92487 23 12 23C18.0751 23 23 18.0751 23 12C23 5.92487 18.0751 1 12 1Z" stroke="currentColor" strokeWidth="2"/>
                          <path d="M8 14C8 14 9.5 16 12 16C14.5 16 16 14 16 14" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
                          <circle cx="9" cy="10" r="1" fill="currentColor"/>
                          <circle cx="15" cy="10" r="1" fill="currentColor"/>
                        </svg>
                      </div>
                      <div className="lt_ready_pulse"></div>
                    </div>
                    <div className="lt_ready_title">Ready to Listen</div>
                    <div className="lt_ready_subtitle">
                      Waiting for incoming calls. The live transcript will appear here automatically once a call is connected.
                    </div>
                    <div className="lt_ready_status">
                      <span className="lt_ready_dot"></span>
                      <span>Agent Online</span>
                    </div>
                  </div>
                )
              ) : (
                <div className="lt_chat">
                  {transcripts.map((t, idx) => {
                    const role = getTurnRole(t.speaker);
                    const rowClass =
                      role === "agent"
                        ? "lt_row lt_row--right"
                        : "lt_row lt_row--left";
                    const bubbleClass =
                      role === "agent"
                        ? "lt_bubble lt_bubble--agent"
                        : role === "customer"
                        ? "lt_bubble lt_bubble--customer"
                        : "lt_bubble lt_bubble--unknown";

                    return (
                      <div key={idx} className={rowClass}>
                        <div className={bubbleClass}>
                          <div className="lt_meta">
                            <span className="lt_speaker">
                              {String(t?.speaker || "UNKNOWN").toUpperCase()}
                            </span>
                            <span className="lt_time">
                              {formatOffsetToTime(t?.begin)}
                            </span>
                          </div>
                          <div className="lt_text">{t?.text || ""}</div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              {isCallConnected ? (
                <div className="lt_row lt_row--left" aria-label="Typing">
                  <div className="lt_bubble lt_bubble--unknown lt_typing_bubble">
                    <div className="lt_typing_dots" aria-hidden="true">
                      <span />
                      <span />
                      <span />
                    </div>
                  </div>
                </div>
              ) : null}
            </div>
          </div>
        </div>
      </main>

      {/* RIGHT: AI Suggestions Panel */}
      <aside className="lt_right lt_ai_panel">
        {/* USER DETAILS - Commented out per request
        <div className="live_transcript_card lt_right_section">
          <div className="label">USER DETAILS</div>
          {copilotUser ? (
            <div className="lt_placeholder">
              <div>Verified: {copilotUser?.verified ? "Yes" : "No"}</div>
              <div>
                Name: {copilotUser?.name || copilotUser?.fullName || "—"}
              </div>
              <div>Phone: {copilotUser?.phone || "—"}</div>
              <div>Contract Type: {copilotUser?.contractType || "—"}</div>
              <div>Plan: {copilotUser?.plan || "—"}</div>
              <div>
                State: {copilotUser?.state || copilotUser?.selectedState || "—"}
              </div>
            </div>
          ) : (
            <div className="lt_placeholder">
              <div>Verified: —</div>
              <div>Name: —</div>
              <div>Phone: —</div>
              <div>Contract Type: —</div>
              <div>Plan: —</div>
              <div>State: —</div>
            </div>
          )}
          {copilotStatus ? (
            <div
              className="lt_placeholder"
              style={{ marginTop: 10, opacity: 0.8 }}
            >
              <div>
                Copilot: {copilotStatus?.enabled ? "Enabled" : "Disabled"}
              </div>
            </div>
          ) : null}
        </div>
        */}

        <div className="lt_ai_suggestions_container">
          <div className="lt_ai_header">
            <div className="lt_ai_icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" fill="currentColor"/>
              </svg>
            </div>
            <div className="lt_ai_title">AI Assistant</div>
            {copilotCards && copilotCards.length > 0 && (
              <div className="lt_ai_count">{copilotCards.length}</div>
            )}
            <div className="lt_ai_badge">Live</div>
          </div>

          <div className="lt_ai_body" ref={suggestionsScrollerRef}>
            {copilotCards && copilotCards.length > 0 ? (
              <div className="lt_suggestion_list">
                {/* Show most recent suggestions first */}
                {[...copilotCards].reverse().map((c, idx) => {
                  const priority = c?.priority || "medium";
                  const isLatest = idx === 0; // First item is latest since array is reversed
                  const priorityClass = `lt_suggestion_card lt_priority_${priority}${isLatest ? " lt_suggestion_latest" : ""}`;
                  const cardKey = c?.id || `card-${idx}`;
                  return (
                    <div key={cardKey} className={priorityClass}>
                      <div className="lt_suggestion_header">
                        <span className="lt_suggestion_icon">
                          {priority === "high" ? "🔴" : priority === "low" ? "🟢" : "🟡"}
                        </span>
                        <span className="lt_suggestion_title">
                          {c?.title || c?.heading || "Suggestion"}
                        </span>
                        {isLatest && (
                          <span className="lt_latest_badge">
                            <span className="lt_latest_dot"></span>
                            Latest
                          </span>
                        )}
                        {c?.timestamp && (
                          <span className="lt_suggestion_time">
                            {formatSuggestionTime(c.timestamp)}
                          </span>
                        )}
                      </div>
                      {/* Show user intent that triggered this suggestion */}
                      {c?.userIntent && (
                        <div className="lt_user_intent">
                          <span className="lt_intent_label">User Intent:</span>
                          <span className="lt_intent_text">{c.userIntent}</span>
                        </div>
                      )}
                      {c?.csrScript ? (
                        <div className="lt_suggestion_script">
                          <div className="lt_script_label">Say this:</div>
                          <div className="lt_script_text">"{c.csrScript}"</div>
                        </div>
                      ) : c?.text ? (
                        <div className="lt_suggestion_script">
                          <div className="lt_script_text">{c.text}</div>
                        </div>
                      ) : null}
                      {c?.evidence ? (
                        <div className="lt_suggestion_evidence">
                          <span className="lt_evidence_label">Triggered by:</span>
                          <span className="lt_evidence_text">"{c.evidence}"</span>
                        </div>
                      ) : null}
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="lt_ai_empty_state">
                <div className="lt_empty_icon">
                  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2L15.09 8.26L22 9.27L17 14.14L18.18 21.02L12 17.77L5.82 21.02L7 14.14L2 9.27L8.91 8.26L12 2Z" fill="currentColor"/>
                  </svg>
                </div>
                <div className="lt_empty_title">Ready to Assist</div>
                <div className="lt_empty_subtitle">
                  AI suggestions will appear here during an active call
                </div>
              </div>
            )}
          </div>
        </div>
      </aside>
    </div>
  );
};

export default LiveTranscript;
