import os
import re

# ============================================================================
# Helper functions for environment variable parsing
# ============================================================================

def _optional_positive_int_env(var_name: str):
    """Return a positive int from env var, otherwise None (unset/invalid/<=0)."""
    raw = (os.getenv(var_name) or "").strip()
    if not raw:
        return None
    try:
        val = int(raw)
        return val if val > 0 else None
    except Exception:
        return None


def _env_int(name: str, default: int) -> int:
    """Return a positive int from env var, otherwise return default."""
    try:
        raw = (os.getenv(name) or "").strip()
        if not raw:
            return default
        v = int(raw)
        return v if v > 0 else default
    except Exception:
        return default


# ============================================================================
# Milvus configuration constants
# ============================================================================

# Milvus retrieval sizing:
# - MILVUS_RETRIEVER_K controls the vector search top-k used by LangChain retrievers.
# - MILVUS_MAX_RETURN_CHUNKS controls how many chunks we return to the API (None = no cap).
MILVUS_RETRIEVER_K = _optional_positive_int_env("MILVUS_RETRIEVER_K") or 10
MILVUS_FALLBACK_K = _optional_positive_int_env("MILVUS_FALLBACK_K") or MILVUS_RETRIEVER_K
MILVUS_MAX_RETURN_CHUNKS = _optional_positive_int_env("MILVUS_MAX_RETURN_CHUNKS")


# ============================================================================
# State aliases mapping (shared between app.py and live_copilot.py)
# ============================================================================

CLEAR_STATE_ALIASES = {
    # Abbreviation -> collection prefix used in Milvus
    "AZ": "Arizona",
    "CA": "California",
    "GA": "Georgia",
    "MD": "Maryland",
    "MN": "Minnesota",
    "NV": "Nevada",
    "TX": "Texas",
    "UT": "Utah",
    "WI": "Wisconsin",
}


# ============================================================================
# Placeholder values for chunks
# ============================================================================

_PLACEHOLDER_CHUNK_VALUES = {
    "[]",
    "",
    "(No supporting excerpts found)",
}


# ============================================================================
# GCP configuration
# ============================================================================

GCP_SERVICE_ACCOUNT_PATH = os.getenv("GCP_SERVICE_ACCOUNT_PATH", None)  # Optional: path to service account JSON


# ============================================================================
# Transcript metadata cache version
# ============================================================================

TRANSCRIPT_METADATA_CACHE_VERSION = "v2"


# ============================================================================
# Live Copilot constants
# ============================================================================

# Hardcoded: emit suggestions at most once per second (no env needed)
COPILOT_COOLDOWN_SECONDS = 1
COPILOT_MAX_VERIFICATION_ASKS = _env_int("COPILOT_MAX_VERIFICATION_ASKS", 2)


# ============================================================================
# Phone number regex pattern
# ============================================================================

_PHONE_RE = re.compile(r"(?:(?:\+?1\s*)?)\(?\s*(\d{3})\s*\)?[\s.-]?(\d{3})[\s.-]?(\d{4})")

