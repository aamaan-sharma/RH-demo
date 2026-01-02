import { useEffect, useMemo, useRef, useState } from "react";
import { io } from "socket.io-client";
import "./liveTranscript.scss";
import { BACKEND_BASE, CCP_URL, REGION } from "../../config";
import { getIdToken } from "../../utils/authStorage";

const LiveTranscript = () => {
  const ccpContainerRef = useRef(null);
  const ccpInitializedRef = useRef(false);
  const [ccpInitError, setCcpInitError] = useState(null);
  const [ccpInitRequested, setCcpInitRequested] = useState(false);
  const [ccpInitInProgress, setCcpInitInProgress] = useState(false);

  const socket = useMemo(() => {
    return io(BACKEND_BASE, {
      transports: ["websocket", "polling"],
      reconnection: true,
      auth: {
        token: getIdToken(),   // 🔐 JWT goes here
      },
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
        const name = agent.getName ? agent.getName() : "Agent";

        setAgentName(name);
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
      
      // ACCUMULATE suggestions instead of replacing
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

  return (
    <div className="live_transcript_layout">
      {/* LEFT: Amazon Connect CCP */}
      <aside className="lt_left_ccp">
        <div className="live_transcript_header lt_left_ccp_header">
          <div className="title">Amazon Connect CCP</div>
          <div className="lt_conn_pill">
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
        <div className="live_transcript_header lt_center_header">
          <div className="title_row">
            <div className="title">Call Transcript</div>
            {isCallConnected ? (
              <div className="lt_streaming_badge">
                Streaming
                <span className="lt_ellipsis" aria-hidden="true">
                  …
                </span>
              </div>
            ) : null}
          </div>
        </div>

        <div className="live_transcript_center_body">
          <div className="live_transcript_card lt_transcript_card">
            <div className="lt_transcript_scroller">
              {transcripts.length === 0 ? (
                <div className="lt_empty_state">No transcript yet.</div>
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

          <div className="lt_ai_body">
            {copilotCards && copilotCards.length > 0 ? (
              <div className="lt_suggestion_list">
                {/* Show most recent suggestions first */}
                {[...copilotCards].reverse().map((c, idx) => {
                  const priority = c?.priority || "medium";
                  const priorityClass = `lt_suggestion_card lt_priority_${priority}`;
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
                        {c?.timestamp && (
                          <span className="lt_suggestion_time">
                            {formatSuggestionTime(c.timestamp)}
                          </span>
                        )}
                      </div>
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
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M21 15C21 15.5304 20.7893 16.0391 20.4142 16.4142C20.0391 16.7893 19.5304 17 19 17H7L3 21V5C3 4.46957 3.21071 3.96086 3.58579 3.58579C3.96086 3.21071 4.46957 3 5 3H19C19.5304 3 20.0391 3.21071 20.4142 3.58579C20.7893 3.96086 21 4.46957 21 5V15Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                </div>
                <div className="lt_empty_title">Ready to assist</div>
                <div className="lt_empty_tips">
                  <div className="lt_tip_item">
                    <span className="lt_tip_bullet">💡</span>
                    <span>Verify customer identity with phone number</span>
                  </div>
                  <div className="lt_tip_item">
                    <span className="lt_tip_bullet">🔍</span>
                    <span>Confirm appliance and symptom details</span>
                  </div>
                  <div className="lt_tip_item">
                    <span className="lt_tip_bullet">📋</span>
                    <span>Summarize next steps and timelines</span>
                  </div>
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
