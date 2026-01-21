import os
from dataclasses import dataclass


@dataclass
class Settings:
    openai_api_key: str | None = None
    mongo_uri: str | None = None
    milvus_host: str | None = None
    jwt_audience: str | None = None
    jwks_url: str | None = None
    gcp_bucket_name: str | None = None
    gcp_project_id: str | None = None
    motorhead_api_key: str | None = None
    motorhead_client_id: str | None = None
    model_intent: str = "gpt-3.5-turbo"
    model_suggest: str = "gpt-4o"
    verbose_debug: bool = False
    flask_secret: str | None = None
    async_mode: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            mongo_uri=os.getenv("MONGO_URI"),
            milvus_host=os.getenv("MILVUS_HOST"),
            jwt_audience=os.getenv("JWT_AUDIENCE"),
            jwks_url=os.getenv("JWKS_URL"),
            gcp_bucket_name=os.getenv("GCP_BUCKET_NAME", "ahs-demo-transcripts"),
            gcp_project_id=os.getenv("GCP_PROJECT_ID", "generative-ai-390411"),
            motorhead_api_key=os.getenv("MOTORHEAD_API_KEY"),
            motorhead_client_id=os.getenv("MOTORHEAD_CLIENT_ID"),
            model_intent=os.getenv("COPILOT_MODEL_INTENT", "gpt-3.5-turbo"),
            model_suggest=os.getenv("COPILOT_MODEL_SUGGEST", "gpt-4o"),
            verbose_debug=str(os.getenv("VERBOSE_DEBUG", "")).lower() in ("1", "true", "yes", "y"),
            flask_secret=os.getenv("FLASK_SECRET_KEY", "dev-secret"),
            async_mode=os.getenv("SOCKETIO_ASYNC_MODE"),
        )


__all__ = ["Settings"]
