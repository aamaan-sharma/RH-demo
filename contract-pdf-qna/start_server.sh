#!/bin/bash

# Start Flask backend server with proper SSL certificate configuration

cd "$(dirname "$0")"

# Prefer the local venv if present (this repo vendors `venv/`).
if [ -f ".venv1/bin/activate" ]; then
  # shellcheck disable=SC1091
  source ".venv1/bin/activate"
fi

# Choose python from the (possibly activated) environment.
PYTHON_BIN="${PYTHON_BIN:-python}"

# Set SSL certificate environment variables
# CRITICAL: These must be set BEFORE Python imports fsspec/gcsfs
CERT_PATH="$($PYTHON_BIN -m certifi 2>/dev/null)"
if [ -n "$CERT_PATH" ]; then
  export SSL_CERT_FILE="$CERT_PATH"
  export REQUESTS_CA_BUNDLE="$CERT_PATH"
  export AIOHTTP_CA_BUNDLE="$CERT_PATH"
fi

echo "=========================================="
echo "Starting Flask backend server..."
echo "=========================================="
echo "SSL Certificates configured:"
echo "  SSL_CERT_FILE: $SSL_CERT_FILE"
echo "  REQUESTS_CA_BUNDLE: $REQUESTS_CA_BUNDLE"
echo "  AIOHTTP_CA_BUNDLE: $AIOHTTP_CA_BUNDLE"
echo ""

# Kill any existing server processes
echo "Stopping any existing server processes..."
pkill -f "python.*app.py" 2>/dev/null
sleep 2

# Start the server
echo "Starting server..."
$PYTHON_BIN app.py

