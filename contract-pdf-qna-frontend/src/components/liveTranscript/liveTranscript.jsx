import React, { useEffect, useMemo, useRef, useState } from "react";
import { io } from "socket.io-client";
import "./liveTranscript.scss";
import { BACKEND_BASE, CCP_URL, REGION } from "../../config";


const LiveTranscript = () => {
  const ccpContainerRef = useRef(null);

  const socket = useMemo(() => {
    // Use websocket transport first; fallback allowed by socket.io
    return io(BACKEND_BASE, {
      transports: ["websocket", "polling"],
      reconnection: true,
      reconnectionAttempts: Infinity,
      reconnectionDelay: 500,
    });
  }, [BACKEND_BASE]);

  const [agentName, setAgentName] = useState("-");
  const [contactId, setContactId] = useState(null);
  const [callState, setCallState] = useState("IDLE");
  const [transcripts, setTranscripts] = useState([]);

  const getTurnRole = (speaker) => {
    const x = String(speaker ?? "").trim().toLowerCase();
    // Backend normalizes to "Customer" | "CSR" | "Unknown" but keep fallbacks just in case
    if (!x) return "unknown";
    if (x === "csr" || x.includes("csr") || x.includes("agent") || x.includes("rep") || x.includes("representative") || x.includes("support")) {
      return "agent";
    }
    if (x === "customer" || x.includes("customer") || x.includes("caller") || x.includes("homeowner") || x.includes("policyholder") || x.includes("member")) {
      return "customer";
    }
    return "unknown";
  };

  // De-dupe because your backend currently emits both:
  // 1) socketio.emit("transcript_update", data)  (global)
  // 2) socketio.emit("transcript_update", data, room=sessionId) (room)
  const seenRef = useRef(new Set());

  useEffect(() => {
    // ---- 1) init CCP once page loads ----
    if (!ccpContainerRef.current || !window.connect) return;

    window.connect.core.initCCP(ccpContainerRef.current, {
      ccpUrl: CCP_URL, 
      loginPopup: true,
      loginPopupAutoClose: true,
      region: REGION, 
      softphone: {
        allowFramedSoftphone: true, 
      },
    });

    // ---- 2) agent info ----
    window.connect.agent((agent) => {
      setAgentName(agent.getName ? agent.getName() : "Agent");

      agent.onStateChange(() => {
        // optional: reflect agent state if you want
      });
    });

    // ---- 3) contact lifecycle ----
    // When a new contact (call) arrives/starts, Streams gives a Contact object.
    // We'll use contactId as your sessionId (matches Lambda: sessionId = ContactId).
    window.connect.contact((contact) => {
      const id = contact.getContactId ? contact.getContactId() : null;
      if (id) {
        setContactId(id);
        setCallState("CONTACT_CREATED");

        // Join backend room (so you can isolate transcripts per call)
        socket.emit("join_session", { sessionId: id });

        // reset transcript view for new call
        seenRef.current.clear();
        setTranscripts([]);
      }

      // Optional state hooks
      if (contact.onConnecting) {
        contact.onConnecting(() => setCallState("CONNECTING"));
      }
      if (contact.onConnected) {
        contact.onConnected(() => setCallState("CONNECTED"));
      }
      if (contact.onEnded) {
        contact.onEnded(() => {
          setCallState("ENDED");
          // keep transcript visible after end; you can also clear contactId if you want
          // setContactId(null);
        });
      }
    });

    // Cleanup not strictly necessary for a POC
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [CCP_URL, REGION, socket]);

  useEffect(() => {
    // ---- 4) receive transcripts from backend ----
    const handler = (msg) => {
      // Only show current call's transcript (sessionId == ContactId)
      if (contactId && msg?.sessionId !== contactId) return;

      // de-dupe (backend emits global + room)
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

  return (
    <div style={{ display: "flex", height: "100vh", fontFamily: "Arial, sans-serif" }}>
      {/* LEFT: CCP */}
      <div style={{ width: 420, borderRight: "1px solid #ddd", padding: 12 }}>
        <h3 style={{ margin: "8px 0" }}>Amazon Connect CCP</h3>
        <div style={{ fontSize: 13, marginBottom: 8 }}>
          <div><b>Agent:</b> {agentName}</div>
          <div><b>Call State:</b> {callState}</div>
          <div><b>ContactId / SessionId:</b> {contactId || "-"}</div>
          <div style={{ marginTop: 8, fontSize: 12, color: "#555" }}>
            If you see a blank iframe: allowlist this web app domain in Connect and use correct CCP URL.
          </div>
        </div>

        <div
          ref={ccpContainerRef}
          id="ccp-container"
          style={{
            width: "100%",
            height: "80vh",
            border: "1px solid #ccc",
            borderRadius: 6,
            overflow: "hidden",
          }}
        />
      </div>

      {/* RIGHT: Transcript */}
      <div style={{ flex: 1, padding: 16 }}>
        <h3 style={{ margin: "8px 0" }}>Live Transcript</h3>
        <div style={{ fontSize: 13, color: "#666", marginBottom: 12 }}>
          This shows text only for the current <b>ContactId</b>. Lambda already sends <code>sessionId = ContactId</code>.
        </div>

        <div
          className="lt_chat"
          style={{
            height: "85vh",
            overflowY: "auto",
            border: "1px solid #ddd",
            borderRadius: 8,
            padding: 12,
            background: "#fafafa",
          }}
        >
          {transcripts.length === 0 ? (
            <div style={{ color: "#777" }}>
              No transcripts yet. Start a call in CCP and wait for Transcribe → Kinesis → Lambda → Backend → UI.
            </div>
          ) : (
            transcripts.map((t, idx) => (
              (() => {
                const role = getTurnRole(t.speaker);
                const rowClass =
                  role === "agent"
                    ? "lt_row lt_row--right"
                    : role === "customer"
                      ? "lt_row lt_row--left"
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
                        <span className="lt_speaker">{t.speaker}</span>
                        {t.begin != null && t.end != null ? (
                          <span className="lt_time">{`(${t.begin}–${t.end}ms)`}</span>
                        ) : null}
                        {t.isPartial ? <span className="lt_partial">partial</span> : null}
                      </div>
                      <div className="lt_text">{t.text}</div>
                    </div>
                  </div>
                );
              })()
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default LiveTranscript;

