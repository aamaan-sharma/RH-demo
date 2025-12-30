FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Minimal OS deps (CA certs for outbound HTTPS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY requirements.txt /app/requirements.txt

# Skip optional/unavailable deps during image build (e.g. "closest")
RUN grep -vE '^closest(\s|$)' /app/requirements.txt > /tmp/requirements.docker.txt \
  && pip install --no-cache-dir -r /tmp/requirements.docker.txt

# Copy app code (do NOT bake in .env / bigquery.json / cacert.pem)
COPY . /app

EXPOSE 8001

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]

