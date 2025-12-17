/**
 * Minimal mock API hook for local development.
 *
 * This file is intentionally lightweight: mocks are OFF by default and will not
 * affect production builds unless explicitly enabled.
 *
 * Enable by setting one of:
 * - VITE_ENABLE_MOCK_API=true  (recommended)
 * - localStorage.setItem('enableMockApi', 'true')
 */

export function isMockApiEnabled() {
  try {
    const envFlag = String(import.meta?.env?.VITE_ENABLE_MOCK_API || "").toLowerCase();
    if (envFlag === "true" || envFlag === "1" || envFlag === "yes") return true;
  } catch (e) {
    // ignore
  }

  try {
    const lsFlag = String(window?.localStorage?.getItem("enableMockApi") || "").toLowerCase();
    if (lsFlag === "true" || lsFlag === "1" || lsFlag === "yes") return true;
  } catch (e) {
    // ignore
  }

  return false;
}

/**
 * Install mock handlers onto an axios instance.
 * Currently a no-op stub so the app can boot without mocks being present.
 * Extend this with request interceptors if you need offline/demo data.
 */
export function installMockApi(axiosInstance) {
  if (!axiosInstance || typeof axiosInstance.interceptors?.request?.use !== "function") return;

  // Example placeholder: you can add interceptors here later.
  // We only log once to avoid console noise.
  if (!window.__mockApiInstalled) {
    window.__mockApiInstalled = true;
    // eslint-disable-next-line no-console
    console.info("[mockApi] enabled (no-op). Add handlers in src/mock/mockApi.js if needed.");
  }
}


