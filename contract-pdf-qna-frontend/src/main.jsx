import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import axios from "axios";
import { installMockApi, isMockApiEnabled } from "./mock/mockApi";
import "amazon-connect-streams";

if (isMockApiEnabled()) {
  installMockApi(axios);
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
