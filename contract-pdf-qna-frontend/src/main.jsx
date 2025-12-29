import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./index.css";
import axios from "axios";
import { installMockApi, isMockApiEnabled } from "./mock/mockApi";
import "amazon-connect-streams";

if (isMockApiEnabled()) {
  installMockApi(axios);
}

createRoot(document.getElementById("root")).render(
  // NOTE: Amazon Connect Streams (connect.core.initCCP) is not compatible with React 18
  // StrictMode's dev-only double-mount behavior; it logs provider/iframe init errors.
  <App />
);
