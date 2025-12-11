import { googleLogout, useGoogleLogin } from "@react-oauth/google";
import axios from "axios";
import React, { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { useLocation } from "react-router-dom";
import plusIcon from "../../assets/plus.svg";
import "./sideBar.scss";
import HistoryButton from "./historyButton/historyButton.jsx";
import settingIcon from "../../assets/setting.svg";
import analyzeLiveIcon from "../../assets/analyze_live.svg";
import bulbIcon from "../../assets/bulb.svg";
import loginIcon from "../../assets/login.svg";
import { API_BASE_URL } from "../../config";

const tokenUrl = "https://oauth2.googleapis.com/token";

const SideBar = (props) => {
  const [sidebarHistory, setSidebarHistory] = useState([]);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userName, setUserName] = useState("");
  const [isActive, setIsActive] = useState(null);
  const location = useLocation();

  let navigate = useNavigate();

  const setChatUrl = () => {
    props.setError("");
    let path = `/#`;
    props.setGptModel("Search");
    navigate(path);
  };

  const getSidebarHistory = (token) => {
    const apiUrl = `${API_BASE_URL}/sidebar`;
    const config = {
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
    };
    axios
      .get(apiUrl, config)
      .then((response) => {
        if (
          response.data.message !== "No data found" &&
          response.data.message !== "Token is missing" &&
          response.data.message !== "Token is invalid" &&
          response.data.message !== "Token has expired"
        ) {
          setSidebarHistory(response.data);
        } else if (response.data.message === "No data found") {
          setSidebarHistory([]);
        }
      })
      .catch((error) => {
        // Handle errors
        console.error("Error:", error);
      });
  };

  const login = useGoogleLogin({
    onSuccess: (codeResponse) => {
      setIsLoggedIn(true);
    },
    flow: "auth-code",
    ux_mode: "redirect",
    redirect_uri: window.location.origin,
    access_type: "online",
    client_id:
      " ",
    client_secret: " ",
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
    sessionStorage.removeItem("payloadObject");
    sessionStorage.removeItem("idToken");
    sessionStorage.removeItem("refreshToken");
    sessionStorage.removeItem("timeoutId");

    googleLogout();
    let path = `/#`;
    navigate(path);
  }, [navigate, props]);

  useEffect(() => {
    const queryParams = new URLSearchParams(window.location.search);
    let urlCode = queryParams.get("code");

    if (urlCode && !props.bearerToken) {
      const params = {
        code: urlCode,
        client_id:
          " ",
        client_secret: " ",
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
          sessionStorage.setItem("idToken", idToken);
          sessionStorage.setItem("refreshToken", refreshToken);
          const parts = idToken.split(".");
          const decodedPayload = atob(parts[1]);
          const payloadObject = JSON.parse(decodedPayload);

          setUserName(payloadObject.name);

          props.setUserImage(payloadObject.picture);
          props.setUserEmail(payloadObject.email);
          setIsLoggedIn(true);

          getSidebarHistory(idToken);
          sessionStorage.setItem(
            "payloadObject",
            JSON.stringify(payloadObject)
          );
        })
        .catch((error) => {
          console.error("Error exchanging code for tokens:", error);
        });
    }
    var payloadObject = JSON.parse(
      JSON.parse(JSON.stringify(sessionStorage.getItem("payloadObject")))
    );
    if (payloadObject && !userName) {
      setUserName(payloadObject.name);
      props.setUserImage(payloadObject.picture);
      props.setUserEmail(payloadObject.email);
      setIsLoggedIn(true);
      getSidebarHistory(sessionStorage.getItem("idToken"));
      props.setBearerToken(sessionStorage.getItem("idToken"));
      props.setRefreshToken(sessionStorage.getItem("refreshToken"));
    }
  }, [userName]);

  useEffect(() => {
    if (isLoggedIn) getSidebarHistory(sessionStorage.getItem("idToken"));
    setIsActive(location.pathname.split("/")[2]);
  }, [isLoggedIn, location.pathname]);

  // Refresh Id token
  const refreshIdToken = () => {
    const params = {
      client_id:
        " ",
      client_secret: " ",
      grant_type: "refresh_token",
      refreshToken: sessionStorage.getItem("refreshToken"),
    };

    const uninterceptedAxiosInstance = axios.create();
    uninterceptedAxiosInstance
      .post("https://oauth2.googleapis.com/token", params)
      .then((response) => {
        const idToken = response.data.id_token;
        props.setBearerToken(idToken);
        sessionStorage.setItem("idToken", idToken);
        const parts = idToken.split(".");
        const decodedPayload = atob(parts[1]);
        const payloadObject = JSON.parse(decodedPayload);
        sessionStorage.setItem("payloadObject", JSON.stringify(payloadObject));
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
        (50 * 60 - elapsedTime) * 1000
      );
      sessionStorage.setItem("timeoutId", nextTimeoutId);
    } else {
      // User was inactive so logout
      logout();
    }
  }, [logout]);

  useEffect(() => {
    if (isLoggedIn && !sessionStorage.getItem("timeoutId")) {
      // Set the initial timeout
      let id = setTimeout(handleTimeout, 50 * 60 * 1000);
      sessionStorage.setItem("timeoutId", id);
    }
  }, [handleTimeout, isLoggedIn, logout]);

  useEffect(() => {
    sessionStorage.clear();
    logout();
  }, []);

  return (
    <div className="sidebar_wrapper">
      <div className="promo_section">Powered by Enzyme</div>
      <div className="new_chat_button" onClick={() => setChatUrl()}>
        <img src={plusIcon} alt="plus icon" />
        <div className="button_name">New Chat</div>
      </div>

      <div className="dashed_line"></div>

      <div className="title">Recent</div>

      <div className="scrollable_section">
        <div className="history_section">
          {sidebarHistory.map((chat, index) => (
            <HistoryButton
              key={index}
              setError={props.setError}
              name={chat.conversationName}
              conversationId={chat.conversationId}
              isActive={isActive}
              setIsActive={setIsActive}
              bearerToken={props.bearerToken}
              getSidebarHistory={getSidebarHistory}
            />
          ))}
        </div>
        <div className="gredient"></div>
      </div>

      <div className="options_container">
        <div className="setting_section">
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
