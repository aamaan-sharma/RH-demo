#!/usr/bin/env sh
set -eu

# Prefer a mounted custom CA bundle if present, otherwise fall back to certifi.
# (This keeps the image portable across macOS/Linux/Windows hosts.)
if [ -f "/run/secrets/cacert.pem" ]; then
  export SSL_CERT_FILE="${SSL_CERT_FILE:-/run/secrets/cacert.pem}"
  export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-/run/secrets/cacert.pem}"
  export AIOHTTP_CA_BUNDLE="${AIOHTTP_CA_BUNDLE:-/run/secrets/cacert.pem}"
else
  CERT_PATH="$(python -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
  if [ -n "${CERT_PATH:-}" ]; then
    export SSL_CERT_FILE="${SSL_CERT_FILE:-$CERT_PATH}"
    export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$CERT_PATH}"
    export AIOHTTP_CA_BUNDLE="${AIOHTTP_CA_BUNDLE:-$CERT_PATH}"
  fi
fi

# BigQuery creds: prefer mounted file if present, otherwise accept env JSON/base64.
if [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] && [ -f "/run/secrets/bigquery.json" ]; then
  export GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/bigquery.json
fi

# BigQuery creds via env (do NOT bake bigquery.json into image)
# Provide either:
# - BIGQUERY_SERVICE_ACCOUNT_JSON (raw JSON string), OR
# - BIGQUERY_SERVICE_ACCOUNT_JSON_BASE64 (base64-encoded JSON)
if [ -n "${BIGQUERY_SERVICE_ACCOUNT_JSON_BASE64:-}" ]; then
  echo "$BIGQUERY_SERVICE_ACCOUNT_JSON_BASE64" | base64 -d > /tmp/bigquery.json
  export GOOGLE_APPLICATION_CREDENTIALS=/tmp/bigquery.json
elif [ -n "${BIGQUERY_SERVICE_ACCOUNT_JSON:-}" ]; then
  printf '%s' "$BIGQUERY_SERVICE_ACCOUNT_JSON" > /tmp/bigquery.json
  export GOOGLE_APPLICATION_CREDENTIALS=/tmp/bigquery.json
fi

exec python app.py

