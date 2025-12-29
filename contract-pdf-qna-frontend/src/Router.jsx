import React, { useState } from "react";
import { Navigate, Route, BrowserRouter as Router, Routes } from "react-router-dom";
import Home from "./components/home/home";
import Insights from "./components/insights/insights";
import ReferredClauses from "./components/referredClauses/referredClauses";
import LiveTranscript from "./components/liveTranscript/liveTranscript";
import { getIdToken } from "./utils/authStorage";

const RequireLogin = ({ children }) => {
  const token = getIdToken();
  if (!token) return <Navigate to="/?error=login" replace />;
  return children;
};

const AppRoutes = () => {
  const [bearerToken, setBearerToken] = useState("");
  return (
    <Router>
      <Routes>
        <Route
          path="/"
          element={
            <Home bearerToken={bearerToken} setBearerToken={setBearerToken} />
          }
        />
        <Route
          path="/conversation/:conversationId"
          element={
            <Home bearerToken={bearerToken} setBearerToken={setBearerToken} />
          }
        />
        <Route
          path="/c/"
          exact
          element={
            <Home bearerToken={bearerToken} setBearerToken={setBearerToken} />
          }
        />
        <Route
          path="/conversation/:conversationId/chat/:chatId/referred-clauses"
          element={<ReferredClauses />}
        />
        <Route path="/insights" element={<Insights />} />
        <Route
          path="/live-transcript"
          element={
            <RequireLogin>
              <LiveTranscript />
            </RequireLogin>
          }
        />
      </Routes>
    </Router>
  );
};

export default AppRoutes;
