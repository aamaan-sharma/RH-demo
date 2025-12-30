## Docker (cross-platform: macOS / Linux / Windows)

This folder can be containerized with Docker so it runs consistently across OSes.

### What is NOT baked into the image
- `.env` files
- `bigquery.json`
- `cacert.pem`

Provide configuration via environment variables instead.

### Build + run (Docker)

```bash
docker build --platform linux/amd64 -t contract-pdf-qna:latest .

docker run --rm --platform linux/amd64 -p 8001:8001 \
  -e OPENAI_API_KEY="..." \
  -e MONGO_URI="..." \
  -e MILVUS_HOST="..." \
  -e JWT_AUDIENCE="..." \
  -e JWKS_URL="..." \
  -e BIGQUERY_SERVICE_ACCOUNT_JSON_BASE64="..." \
  contract-pdf-qna:latest
```

If you’re not on Apple Silicon (or you don’t need `linux/amd64`), you can use the standard commands:

```bash
docker build -t contract-pdf-qna:latest .

docker run --rm -p 8001:8001 \
  -e OPENAI_API_KEY="..." \
  -e MONGO_URI="..." \
  -e MILVUS_HOST="..." \
  -e JWT_AUDIENCE="..." \
  -e JWKS_URL="..." \
  -e BIGQUERY_SERVICE_ACCOUNT_JSON_BASE64="..." \
  contract-pdf-qna:latest
```

The backend listens on `0.0.0.0:8001` (see `app.py`).

### BigQuery credentials (env-based)

Set one of:
- `BIGQUERY_SERVICE_ACCOUNT_JSON_BASE64`: base64-encoded service account JSON
- `BIGQUERY_SERVICE_ACCOUNT_JSON`: raw JSON string

The container entrypoint will write this to `/tmp/bigquery.json` and set:
- `GOOGLE_APPLICATION_CREDENTIALS=/tmp/bigquery.json`

### Docker Compose

```bash
docker compose up --build
```

Edit `docker-compose.yml` to pass the env vars you need (or export them in your shell).

