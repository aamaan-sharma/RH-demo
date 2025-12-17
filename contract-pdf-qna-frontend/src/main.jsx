import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import axios from "axios";
import { installMockApi, isMockApiEnabled } from "./mock/mockApi";

if (isMockApiEnabled()) {
  installMockApi(axios);
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
