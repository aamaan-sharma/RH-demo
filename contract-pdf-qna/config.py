import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")
MILVUS_HOST = os.getenv("MILVUS_HOST")
MODEL_INTENT = os.getenv("COPILOT_MODEL_INTENT", "gpt-3.5-turbo")
MODEL_SUGGEST = os.getenv("COPILOT_MODEL_SUGGEST", "gpt-4o")
VERBOSE_DEBUG = os.getenv("VERBOSE_DEBUG", "").lower() in ("1", "true", "yes", "y")
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE")
JWKS_URL = os.getenv("JWKS_URL")
GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "ahs-demo-transcripts")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "generative-ai-390411")
MOTORHEAD_API_KEY = os.getenv("MOTORHEAD_API_KEY")
MOTORHEAD_CLIENT_ID = os.getenv("MOTORHEAD_CLIENT_ID")