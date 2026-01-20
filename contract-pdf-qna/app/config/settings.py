"""Application configuration settings.

Centralizes all environment variable loading and configuration management.
"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def _flag_enabled(var_name: str, default: str = "0") -> bool:
    """Check if an environment variable flag is enabled."""
    raw = (os.getenv(var_name, default) or "").strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


def _optional_positive_int_env(var_name: str) -> Optional[int]:
    """Return a positive int from env var, otherwise None (unset/invalid/<=0)."""
    raw = (os.getenv(var_name) or "").strip()
    if not raw:
        return None
    try:
        val = int(raw)
        return val if val > 0 else None
    except Exception:
        return None


def _copilot_session_ttl_seconds() -> int:
    """Get copilot session TTL in seconds from environment."""
    try:
        raw = (os.getenv("COPILOT_SESSION_TTL_SECONDS") or "").strip()
        ttl = int(raw) if raw else 1800
        return ttl if ttl > 0 else 1800
    except Exception:
        return 1800


class Settings:
    """Application settings loaded from environment variables."""
    
    # Flask Configuration
    FLASK_SECRET_KEY: str = os.getenv("FLASK_SECRET_KEY", "dev-secret")
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "0").lower() in ("1", "true", "yes", "y")
    PORT: int = int(os.getenv("PORT", "5000"))
    
    # SocketIO Configuration
    SOCKETIO_ASYNC_MODE: str = os.getenv("SOCKETIO_ASYNC_MODE", "threading")
    
    # Authentication
    JWT_AUDIENCE: Optional[str] = os.getenv("JWT_AUDIENCE")
    JWKS_URL: Optional[str] = os.getenv("JWKS_URL")
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    MODEL_INTENT: str = os.getenv("COPILOT_MODEL_INTENT", "gpt-3.5-turbo")
    MODEL_SUGGEST: str = os.getenv("COPILOT_MODEL_SUGGEST", "gpt-4o")
    
    # MongoDB Configuration
    MONGO_URI: Optional[str] = os.getenv("MONGO_URI")
    MONGO_DB_NAME: Optional[str] = os.getenv("MONGO_DB_NAME")
    
    # Milvus Configuration
    MILVUS_HOST: Optional[str] = os.getenv("MILVUS_HOST")
    MILVUS_RETRIEVER_K: int = _optional_positive_int_env("MILVUS_RETRIEVER_K") or 25
    MILVUS_FALLBACK_K: int = _optional_positive_int_env("MILVUS_FALLBACK_K") or MILVUS_RETRIEVER_K
    MILVUS_MAX_RETURN_CHUNKS: Optional[int] = _optional_positive_int_env("MILVUS_MAX_RETURN_CHUNKS")
    
    # GCP Configuration
    GCP_BUCKET_NAME: str = os.getenv("GCP_BUCKET_NAME", "ahs-demo-transcripts")
    GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "generative-ai-390411")
    GCP_SERVICE_ACCOUNT_PATH: Optional[str] = os.getenv("GCP_SERVICE_ACCOUNT_PATH")
    
    # Motorhead Memory Configuration
    MOTORHEAD_API_KEY: Optional[str] = os.getenv("MOTORHEAD_API_KEY")
    MOTORHEAD_CLIENT_ID: Optional[str] = os.getenv("MOTORHEAD_CLIENT_ID")
    
    # Live Copilot Configuration
    ENABLE_LIVE_COPILOT: bool = _flag_enabled("ENABLE_LIVE_COPILOT", "0")
    COPILOT_SESSION_TTL_SECONDS: int = _copilot_session_ttl_seconds()
    VERBOSE_DEBUG: bool = os.getenv("VERBOSE_DEBUG", "").lower() in ("1", "true", "yes", "y")
    
    # Internal Processing
    INTERNAL_PROCESS_SECRET: Optional[str] = os.getenv("INTERNAL_PROCESS_SECRET")
    
    # OpenTelemetry Configuration
    OTEL_SERVICE_NAME: str = os.getenv("OTEL_SERVICE_NAME", "CSR Copilot")
    OTEL_EXPORTER_OTLP_ENDPOINT: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4318")
    OTEL_EXPORTER_OTLP_PROTOCOL: str = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    OTEL_TRACE_INCLUDE_PAYLOADS: bool = _flag_enabled("OTEL_TRACE_INCLUDE_PAYLOADS", "0")
    OTEL_TRACE_PAYLOAD_PREVIEW_CHARS: int = _optional_positive_int_env("OTEL_TRACE_PAYLOAD_PREVIEW_CHARS") or 0
    OTEL_TRACE_LLM_CALL_SPANS: bool = _flag_enabled("OTEL_TRACE_LLM_CALL_SPANS", "1")
    OTEL_TRACE_TOOL_CALL_SPANS: bool = _flag_enabled("OTEL_TRACE_TOOL_CALL_SPANS", "1")
    
    @classmethod
    def validate(cls) -> None:
        """Validate required configuration settings."""
        required_settings = [
            ("OPENAI_API_KEY", cls.OPENAI_API_KEY),
            ("MONGO_URI", cls.MONGO_URI),
            ("MILVUS_HOST", cls.MILVUS_HOST),
        ]
        
        missing = [name for name, value in required_settings if not value]
        if missing:
            raise ValueError(f"Missing required configuration: {', '.join(missing)}")


# Create a singleton instance
settings = Settings()
