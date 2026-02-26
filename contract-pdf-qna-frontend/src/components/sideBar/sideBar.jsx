import { googleLogout, useGoogleLogin } from "@react-oauth/google";
import axios from "axios";
import { useCallback, useEffect, useRef, useState } from "react";
import PropTypes from "prop-types";
import { useLocation, useNavigate } from "react-router-dom";
import plusIcon from "../../assets/plus.svg";
import "./sideBar.scss";
import HistoryButton from "./historyButton/historyButton.jsx";
import settingIcon from "../../assets/setting.svg";
import analyzeLiveIcon from "../../assets/analyze_live.svg";
import bulbIcon from "../../assets/bulb.svg";
import loginIcon from "../../assets/login.svg";
import {
  API_BASE_URL,
  GOOGLE_CLIENT_ID,
  GOOGLE_CLIENT_SECRET,
} from "../../config";
import TryAgainButton from "../common/tryAgainButton/tryAgainButton";
import {
  clearAuthTokens,
  getIdToken,
  getPayloadObjectRaw,
  getRefreshToken,
  setAuthTokens,
} from "../../utils/authStorage";

const tokenUrl = "https://oauth2.googleapis.com/token";

const SideBar = (props) => {
  const [sidebarHistory, setSidebarHistory] = useState([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userName, setUserName] = useState("");
  const [isActive, setIsActive] = useState(null);
  const [sidebarError, setSidebarError] = useState(null);
  const location = useLocation();
  const cleanedAuthUrlRef = useRef(false);
  const lastModeRef = useRef(null);
  const claimsPollTimerRef = useRef(null);
  const sidebarAbortRef = useRef(null);
  const sidebarRequestIdRef = useRef(0);

  let navigate = useNavigate();

  const refreshIdTokenAsync = useCallback(async () => {
    const refreshToken = getRefreshToken();
    if (!refreshToken) throw new Error("Missing refresh token");

    const params = new URLSearchParams();
    params.set("client_id", GOOGLE_CLIENT_ID);
    params.set("client_secret", GOOGLE_CLIENT_SECRET);
    params.set("grant_type", "refresh_token");
    params.set("refresh_token", refreshToken);

    const uninterceptedAxiosInstance = axios.create();
    const resp = await uninterceptedAxiosInstance.post(tokenUrl, params, {
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
    });
    const idToken = resp?.data?.id_token;
    if (!idToken) throw new Error("No id_token in refresh response");

    props.setBearerToken(idToken);
    const parts = idToken.split(".");
    const decodedPayload = atob(parts[1]);
    const payloadObject = JSON.parse(decodedPayload);
    setAuthTokens({ idToken, payloadObject });
    return idToken;
  }, [props]);

  const setChatUrl = () => {
    props.setError("");
    let path = `/#`;
    // Keep New Chat in the same mode the user is currently in.
    props.setGptModel(props.selectedModel || "Search");
    navigate(path);
  };

  const getSidebarHistory = (token, mode = "Search", opts = {}) => {
    const showLoading = opts?.showLoading !== false;
    const apiUrl = `${API_BASE_URL}/sidebar`;
    const config = {
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      params: {
        mode: mode || "Search",
      },
    };

    // Latest-wins: cancel any previous sidebar request
    try {
      if (sidebarAbortRef.current) sidebarAbortRef.current.abort();
    } catch {
      // ignore
    }
    const requestId = (sidebarRequestIdRef.current += 1);
    const abortController = new AbortController();
    sidebarAbortRef.current = abortController;
    config.signal = abortController.signal;

    if (showLoading) setIsLoadingHistory(true);
    axios
      .get(apiUrl, config)
      .then((response) => {
        if (sidebarRequestIdRef.current !== requestId) return;
        setSidebarError(null);
        // Backend returns an array; keep this resilient.
        const data = response?.data;
        setSidebarHistory(Array.isArray(data) ? data : []);
      })
      .catch((error) => {
        if (error?.name === "CanceledError" || error?.code === "ERR_CANCELED") {
          return;
        }
        if (sidebarRequestIdRef.current !== requestId) return;
        // Handle errors
        console.error("Error:", error);
        const status = error?.response?.status;
        if ((status === 401 || status === 403) && getRefreshToken()) {
          // Token expired/invalid: refresh once and retry.
          refreshIdTokenAsync()
            .then((newToken) => {
              getSidebarHistory(newToken, mode, { showLoading: false });
            })
            .catch(() => {
              logout();
            });
          return;
        }
        if (status === 500) {
          setSidebarError({
            retryFn: () => getSidebarHistory(token, mode),
          });
        } else {
          setSidebarHistory([]);
        }
      })
      .finally(() => {
        if (sidebarRequestIdRef.current !== requestId) return;
        setIsLoadingHistory(false);
      });
  };

  const login = useGoogleLogin({
    onSuccess: () => {
      setIsLoggedIn(true);
    },
    flow: "auth-code",
    ux_mode: "redirect",
    redirect_uri: window.location.origin,
    access_type: "online",
    client_id: GOOGLE_CLIENT_ID,
    client_secret: GOOGLE_CLIENT_SECRET,
    scope:
      "openid https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile",
  });

  const logout = useCallback(() => {
    props.setSelectedContract("Contract Type");
    props.setSelectedPlan("Plan");
    props.setSelectedState("State");
    setIsLoggedIn(false);
    props.setError("");
    props.setUserImage("");
    setSidebarHistory([]);
    setUserName("");
    clearAuthTokens();

    googleLogout();
    let path = `/#`;
    navigate(path);
  }, [navigate, props]);

  useEffect(() => {
    const queryParams = new URLSearchParams(location.search);
    let urlCode = queryParams.get("code");

    if (urlCode && !props.bearerToken) {
      const params = {
        code: urlCode,
        client_id: GOOGLE_CLIENT_ID,
        client_secret: GOOGLE_CLIENT_SECRET,
        redirect_uri: window.location.origin,
        grant_type: "authorization_code",
      };

      axios
        .post(tokenUrl, null, {
          params: params,
        })
        .then((response) => {
          const idToken = response.data.id_token;
          // const accessToken = response.data.access_token;
          const refreshToken = response.data.refresh_token;

          props.setBearerToken(idToken);
          props.setRefreshToken(refreshToken);
          const parts = idToken.split(".");
          const decodedPayload = atob(parts[1]);
          const payloadObject = JSON.parse(decodedPayload);

          setUserName(payloadObject.name);

          props.setUserImage(payloadObject.picture);
          props.setUserEmail(payloadObject.email);
          setIsLoggedIn(true);

          getSidebarHistory(idToken, props.selectedModel || "Search");
          setAuthTokens({ idToken, refreshToken, payloadObject });

          // IMPORTANT: remove OAuth query params from browser history so back button doesn't land on auth URLs.
          // Keep the current pathname + hash but clear the search string.
          if (!cleanedAuthUrlRef.current && location.search) {
            cleanedAuthUrlRef.current = true;
            navigate(
              { pathname: location.pathname, search: "", hash: location.hash },
              { replace: true },
            );
          }
        })
        .catch((error) => {
          console.error("Error exchanging code for tokens:", error);
        });
    }
    const payloadObjectRaw = getPayloadObjectRaw();
    var payloadObject = payloadObjectRaw ? JSON.parse(payloadObjectRaw) : null;
    if (payloadObject && !userName) {
      setUserName(payloadObject.name);
      props.setUserImage(payloadObject.picture);
      props.setUserEmail(payloadObject.email);
      setIsLoggedIn(true);
      getSidebarHistory(getIdToken(), props.selectedModel || "Search");
      props.setBearerToken(getIdToken());
      props.setRefreshToken(getRefreshToken());

      // If we got here with OAuth params already in the URL (e.g. refresh/redirect),
      // clean them so back button doesn't revisit them.
      if (!cleanedAuthUrlRef.current && location.search) {
        cleanedAuthUrlRef.current = true;
        navigate(
          { pathname: location.pathname, search: "", hash: location.hash },
          { replace: true },
        );
      }
    }
  }, [userName, location.search, location.pathname, location.hash, navigate]);

  useEffect(() => {
    setIsActive(location.pathname.split("/")[2]);
  }, [isLoggedIn, location.pathname]);

  useEffect(() => {
    if (!isLoggedIn) return;
    const mode = props.selectedModel || "Search";
    const modeChanged = lastModeRef.current !== mode;
    lastModeRef.current = mode;
    // Only show the sidebar loader on first load or when switching modes.
    // For background refreshes (e.g. transcript processing updates), keep the list visible.
    const showLoading =
      modeChanged ||
      !(Array.isArray(sidebarHistory) && sidebarHistory.length > 0);
    getSidebarHistory(getIdToken(), mode, { showLoading });
  }, [isLoggedIn, props.selectedModel, props.sidebarRefreshTick]);

  // Claims-only: while any case is processing (yellow dot), keep refreshing sidebar in the background
  // so it flips to green as soon as analysis completes, even if the user is viewing other chats.
  useEffect(() => {
    const mode = props.selectedModel || "Search";
    const isClaims = mode === "Calls";
    const hasProcessing =
      isClaims &&
      Array.isArray(sidebarHistory) &&
      sidebarHistory.some((c) => Boolean(c?.processing));

    // Clear any existing poller
    if (claimsPollTimerRef.current) {
      clearInterval(claimsPollTimerRef.current);
      claimsPollTimerRef.current = null;
    }

    if (!isLoggedIn || !isClaims || !hasProcessing) return;

    claimsPollTimerRef.current = setInterval(() => {
      const token = getIdToken();
      if (!token) return;
      // Silent refresh; keep list visible.
      getSidebarHistory(token, "Calls", { showLoading: false });
    }, 5000);

    return () => {
      if (claimsPollTimerRef.current) {
        clearInterval(claimsPollTimerRef.current);
        claimsPollTimerRef.current = null;
      }
    };
  }, [isLoggedIn, props.selectedModel, sidebarHistory]);

  // Optimistically move a case into "Closed" immediately after approval.
  useEffect(() => {
    const closedId = props.recentlyClosedConversationId;
    if (!closedId) return;
    setSidebarHistory((prev) => {
      const arr = Array.isArray(prev) ? prev : [];
      return arr.map((c) =>
        String(c?.conversationId || "") === String(closedId)
          ? {
              ...c,
              status: "inactive",
              updatedAt: new Date().toISOString(),
            }
          : c,
      );
    });
  }, [props.recentlyClosedConversationId]);

  // Refresh Id token
  const refreshIdToken = () => {
    const refreshToken = getRefreshToken();
    if (!refreshToken) return;

    const params = new URLSearchParams();
    params.set("client_id", GOOGLE_CLIENT_ID);
    params.set("client_secret", GOOGLE_CLIENT_SECRET);
    params.set("grant_type", "refresh_token");
    // Google expects the param name "refresh_token" (not "refreshToken")
    params.set("refresh_token", refreshToken);

    const uninterceptedAxiosInstance = axios.create();
    uninterceptedAxiosInstance
      .post("https://oauth2.googleapis.com/token", params, {
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
      })
      .then((response) => {
        const idToken = response.data.id_token;
        props.setBearerToken(idToken);
        const parts = idToken.split(".");
        const decodedPayload = atob(parts[1]);
        const payloadObject = JSON.parse(decodedPayload);
        setAuthTokens({ idToken, payloadObject });
      })
      .catch((error) => {
        console.error("Error exchanging code for tokens:", error);
        logout();
      });
  };

  const handleTimeout = useCallback(() => {
    const lastActiveTime = sessionStorage.getItem("lastActiveTime");
    const currentTime = Math.floor(Date.now() / 1000);
    const elapsedTime = lastActiveTime ? currentTime - lastActiveTime : 0;
    if (elapsedTime < 50 * 60 - 15) {
      // User was recently active, refresh token
      refreshIdToken();
      // Set another timeout for the next refresh
      let nextTimeoutId = setTimeout(
        handleTimeout,
        (50 * 60 - elapsedTime) * 1000,
      );
      sessionStorage.setItem("timeoutId", nextTimeoutId);
    } else {
      // User was inactive so logout
      logout();
    }
  }, [logout]);

  // Track activity so we can refresh tokens for active users and logout idle users.
  useEffect(() => {
    const markActive = () => {
      try {
        sessionStorage.setItem(
          "lastActiveTime",
          String(Math.floor(Date.now() / 1000)),
        );
      } catch {
        // ignore
      }
    };

    // Mark active immediately on mount, then on common interactions.
    markActive();
    const events = [
      "mousemove",
      "mousedown",
      "keydown",
      "touchstart",
      "scroll",
    ];
    events.forEach((evt) =>
      window.addEventListener(evt, markActive, { passive: true }),
    );
    return () => {
      events.forEach((evt) => window.removeEventListener(evt, markActive));
    };
  }, []);

  useEffect(() => {
    if (isLoggedIn && !sessionStorage.getItem("timeoutId")) {
      // Set the initial timeout
      let id = setTimeout(handleTimeout, 50 * 60 * 1000);
      sessionStorage.setItem("timeoutId", id);
    }
  }, [handleTimeout, isLoggedIn, logout]);

  // IMPORTANT: Do NOT auto-logout on mount; this breaks login persistence across refresh/reopen.

  return (
    <div className="sidebar_wrapper">
      <div className="promo_section"></div>
      <div className="new_chat_button" onClick={() => setChatUrl()}>
        <img src={plusIcon} alt="plus icon" />
        <div className="button_name">
          {(props.selectedModel || "Search") === "Calls"
            ? "New Case"
            : "New Chat"}
        </div>
      </div>

      <div className="dashed_line"></div>

      <div className="title">
        {(props.selectedModel || "Search") === "Calls" ? "Claims" : "Recent"}
      </div>

      <div className="scrollable_section">
        <div className="history_section">
          {isLoadingHistory ? (
            <div className="history_loading">
              <div className="spinner" aria-hidden="true" />
              <div className="text">Loading history…</div>
            </div>
          ) : sidebarError ? (
            <div className="history_error">
              <div className="error_text">
                Failed to load history. Please try again.
              </div>
              <TryAgainButton
                onRetry={() => {
                  setSidebarError(null);
                  if (sidebarError?.retryFn) {
                    sidebarError.retryFn();
                  }
                }}
              />
            </div>
          ) : (props.selectedModel || "Search") === "Calls" ? (
            (() => {
              const sortedHistory = (sidebarHistory || [])
                .slice()
                .sort((a, b) => {
                  const at = Date.parse(a?.updatedAt || "") || 0;
                  const bt = Date.parse(b?.updatedAt || "") || 0;
                  return bt - at;
                });
              const openCases = sortedHistory.filter(
                (c) => (c?.status || "active").toLowerCase() !== "inactive",
              );
              const closedCases = sortedHistory.filter(
                (c) => (c?.status || "active").toLowerCase() === "inactive",
              );
              const shouldOpenClosed = Boolean(
                props.recentlyClosedConversationId,
              );

              const renderCaseRow = (chat, index) => (
                <HistoryButton
                  key={chat?.conversationId || index}
                  setError={props.setError}
                  name={chat.conversationName}
                  conversationId={chat.conversationId}
                  conversationMode={chat.conversationMode}
                  status={chat.status}
                  processing={Boolean(chat.processing)}
                  setGptModel={props.setGptModel}
                  isActive={isActive}
                  setIsActive={setIsActive}
                  bearerToken={props.bearerToken}
                  getSidebarHistory={(token) =>
                    getSidebarHistory(token, props.selectedModel || "Search")
                  }
                />
              );

              return (
                <div className="cases_wrapper">
                  <details className="case_group" open>
                    <summary className="case_group_summary">
                      Open <span className="count">{openCases.length}</span>
                    </summary>
                    <div className="case_group_list">
                      {openCases.length > 0 ? (
                        openCases.map(renderCaseRow)
                      ) : (
                        <div className="empty_state">No open cases.</div>
                      )}
                    </div>
                  </details>
                  <details
                    className="case_group"
                    open={shouldOpenClosed ? true : undefined}
                  >
                    <summary className="case_group_summary">
                      Closed <span className="count">{closedCases.length}</span>
                    </summary>
                    <div className="case_group_list">
                      {closedCases.length > 0 ? (
                        closedCases.map(renderCaseRow)
                      ) : (
                        <div className="empty_state">No closed cases.</div>
                      )}
                    </div>
                  </details>
                </div>
              );
            })()
          ) : sidebarHistory && sidebarHistory.length > 0 ? (
            sidebarHistory.map((chat, index) => (
              <HistoryButton
                key={index}
                setError={props.setError}
                name={chat.conversationName}
                conversationId={chat.conversationId}
                conversationMode={chat.conversationMode}
                setGptModel={props.setGptModel}
                isActive={isActive}
                setIsActive={setIsActive}
                bearerToken={props.bearerToken}
                getSidebarHistory={(token) =>
                  getSidebarHistory(token, props.selectedModel || "Search")
                }
              />
            ))
          ) : (
            <div className="empty_state">No recent chats.</div>
          )}
        </div>
        <div className="gredient"></div>
      </div>

      <div className="options_container">
        <div
          className="setting_section"
          onClick={() => {
            const token = getIdToken();
            if (!isLoggedIn || !token) {
              props.setError("login");
              navigate("/?error=login", { replace: true });
              return;
            }
            window.open(
              `${window.location.origin}/live-transcript`,
              "_blank",
              "noopener,noreferrer",
            );
          }}
        >
          <img src={analyzeLiveIcon} alt="Setting Icon" />
          <div className="setting_text">Analyze Live</div>
        </div>
        <div
          className="setting_section"
          onClick={() =>
            window.open(`http://34.28.68.164:3000/dashboards/f/ddizmsq6ca2o0e/`)
          }
        >
          <img src={bulbIcon} alt="Setting Icon" />
          <div className="setting_text">Insights</div>
        </div>
        {isLoggedIn ? (
          <div className="setting_section" onClick={() => logout()}>
            <img src={loginIcon} alt="Setting Icon" />
            <div className="setting_text">Logout</div>
          </div>
        ) : (
          <div
            className={`setting_section ${
              props.error === "login" ? "highlight" : ""
            }`}
            onClick={() => login()}
          >
            <img src={loginIcon} alt="Setting Icon" />
            <div className="setting_text">Login</div>
          </div>
        )}
        <div className="setting_section">
          <img src={settingIcon} alt="Setting Icon" />
          <div className="setting_text">Settings</div>
        </div>
      </div>
    </div>
  );
};

export default SideBar;

SideBar.propTypes = {
  bearerToken: PropTypes.any,
  setBearerToken: PropTypes.func,
  refreshToken: PropTypes.any,
  setRefreshToken: PropTypes.func,
  error: PropTypes.any,
  setError: PropTypes.func,
  userEmail: PropTypes.any,
  setUserEmail: PropTypes.func,
  setGptModel: PropTypes.func,
  selectedModel: PropTypes.any,
  sidebarRefreshTick: PropTypes.any,
  recentlyClosedConversationId: PropTypes.any,
  setSelectedContract: PropTypes.func,
  setSelectedPlan: PropTypes.func,
  setSelectedState: PropTypes.func,
  setUserImage: PropTypes.func,
};
