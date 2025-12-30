export function getIdToken() {
  return localStorage.getItem("idToken") || sessionStorage.getItem("idToken");
}

export function getRefreshToken() {
  return (
    localStorage.getItem("refreshToken") ||
    sessionStorage.getItem("refreshToken")
  );
}

export function getPayloadObjectRaw() {
  return (
    localStorage.getItem("payloadObject") ||
    sessionStorage.getItem("payloadObject")
  );
}

export function setAuthTokens({ idToken, refreshToken, payloadObject }) {
  if (idToken) {
    localStorage.setItem("idToken", idToken);
    sessionStorage.setItem("idToken", idToken);
  }
  if (refreshToken) {
    localStorage.setItem("refreshToken", refreshToken);
    sessionStorage.setItem("refreshToken", refreshToken);
  }
  if (payloadObject) {
    const raw =
      typeof payloadObject === "string"
        ? payloadObject
        : JSON.stringify(payloadObject);
    localStorage.setItem("payloadObject", raw);
    sessionStorage.setItem("payloadObject", raw);
  }
}

export function clearAuthTokens() {
  ["idToken", "refreshToken", "payloadObject", "timeoutId"].forEach((k) => {
    try {
      localStorage.removeItem(k);
    } catch {
      // ignore
    }
    try {
      sessionStorage.removeItem(k);
    } catch {
      // ignore
    }
  });
}
