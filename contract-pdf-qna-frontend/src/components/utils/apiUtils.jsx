import { getIdToken } from "../../utils/authStorage";

export const setHeaders = (config) => {
  const token = getIdToken();
  if (token) {
    config.headers["Authorization"] = "Bearer " + token;
    config.headers["Content-Type"] = "application/json";
  }
  sessionStorage.setItem("lastActiveTime", Math.floor(Date.now() / 1000));
  return config;
};
