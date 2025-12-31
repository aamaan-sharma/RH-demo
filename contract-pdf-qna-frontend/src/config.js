// API Configuration
// Main backend (existing Langchain/ADK services)
export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8001";

// Transcripts backend (GCP transcript listing & processing service)
export const TRANSCRIPTS_API_BASE_URL =
  import.meta.env.VITE_TRANSCRIPTS_API_BASE_URL || "http://localhost:8001";

// Live Transcript (Amazon Connect + websocket backend) Configuration
// NOTE (Vite): env vars must be prefixed with VITE_ to be exposed to the browser.
export const CCP_URL =
  import.meta.env.VITE_CCP_URL;

export const REGION = import.meta.env.VITE_REGION;

export const BACKEND_BASE =
  import.meta.env.VITE_BACKEND_BASE || "http://localhost:8001";

// Google OAuth Configuration
export const GOOGLE_CLIENT_ID =
  import.meta.env.VITE_GOOGLE_CLIENT_ID ;
export const GOOGLE_CLIENT_SECRET =
  import.meta.env.VITE_GOOGLE_CLIENT_SECRET;

