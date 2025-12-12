import React, { useState } from "react";
import { Route, BrowserRouter as Router, Routes } from "react-router-dom";
import Home from "./components/home/home";
import Insights from "./components/insights/insights";
import ReferredClauses from "./components/referredClauses/referredClauses";

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
      </Routes>
    </Router>
  );
};

export default AppRoutes;
