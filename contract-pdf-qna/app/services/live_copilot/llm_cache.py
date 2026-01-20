"""LLM instance caching for Live Copilot."""
import threading
from typing import Optional
from langchain_openai import ChatOpenAI
from app.config.settings import settings


# Global cache for LLM instances - reused across all transcript events
_llm_intent_cache: Optional[ChatOpenAI] = None
_llm_suggest_cache: Optional[ChatOpenAI] = None
_llm_diagnostics_cache: Optional[ChatOpenAI] = None
_llm_cache_lock = threading.Lock()  # Thread-safe initialization


def get_intent_llm() -> ChatOpenAI:
    """
    Get or create cached intent detection LLM instance (thread-safe).
    Reuses the same instance across all transcript events for efficiency.
    """
    global _llm_intent_cache
    if _llm_intent_cache is None:
        with _llm_cache_lock:  # Prevent race condition during initialization
            if _llm_intent_cache is None:  # Double-check pattern
                _llm_intent_cache = ChatOpenAI(
                    temperature=0.0,
                    model=settings.MODEL_INTENT,
                    max_tokens=200,  # Limit response length for speed
                    timeout=10.0,  # Fail fast on slow API calls
                    openai_api_key=settings.OPENAI_API_KEY,
                )
    return _llm_intent_cache


def get_suggest_llm() -> ChatOpenAI:
    """
    Get or create cached suggestion generation LLM instance (thread-safe).
    Reuses the same instance across all transcript events for efficiency.
    """
    global _llm_suggest_cache
    if _llm_suggest_cache is None:
        with _llm_cache_lock:  # Prevent race condition during initialization
            if _llm_suggest_cache is None:  # Double-check pattern
                _llm_suggest_cache = ChatOpenAI(
                    temperature=0.0,
                    model=settings.MODEL_SUGGEST,
                    max_tokens=500,  # Limit response length
                    timeout=15.0,  # Allow more time for suggestion generation
                    openai_api_key=settings.OPENAI_API_KEY,
                )
    return _llm_suggest_cache


def get_diagnostics_llm() -> ChatOpenAI:
    """
    Get or create cached diagnostics LLM instance (thread-safe).
    Reuses the same instance across all transcript events for efficiency.
    """
    global _llm_diagnostics_cache
    if _llm_diagnostics_cache is None:
        with _llm_cache_lock:  # Prevent race condition during initialization
            if _llm_diagnostics_cache is None:  # Double-check pattern
                _llm_diagnostics_cache = ChatOpenAI(
                    temperature=0.2,
                    model=settings.MODEL_SUGGEST,
                    max_tokens=300,  # Limit response length
                    timeout=10.0,  # Fail fast on slow API calls
                    openai_api_key=settings.OPENAI_API_KEY,
                )
    return _llm_diagnostics_cache
