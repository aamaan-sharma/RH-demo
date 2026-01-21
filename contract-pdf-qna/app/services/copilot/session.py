import os
import threading
from time import time
from typing import Dict, Optional

from ..copilot_service import LIVE_COPILOT_AVAILABLE
from ...extensions import tracer

_copilot_enabled_sessions: Dict[str, float] = {}
_copilot_session_context: Dict[str, dict] = {}
_copilot_sessions_lock = threading.Lock()
_session_trace_ctx: Dict[str, object] = {}
_session_trace_lock = threading.Lock()


def flag_enabled(var_name: str, default: str = "0") -> bool:
    raw = (os.getenv(var_name, default) or "").strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


def copilot_session_ttl_seconds() -> int:
    try:
        raw = (os.getenv("COPILOT_SESSION_TTL_SECONDS") or "").strip()
        ttl = int(raw) if raw else 1800
        return ttl if ttl > 0 else 1800
    except Exception:
        return 1800


def get_parent_trace_context(session_id: str):
    """
    Ensure a single trace per sessionId by creating one root span.
    """
    if not session_id or tracer is None:
        return None
    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.trace import NonRecordingSpan
    except Exception:
        return None

    with _session_trace_lock:
        existing = _session_trace_ctx.get(session_id)
        if existing is None:
            with tracer.start_as_current_span("csr_copilot.session") as root:
                root.set_attribute("live.session_id", session_id)
            try:
                existing = root.get_span_context()
            except Exception:
                existing = None
            if existing is not None:
                _session_trace_ctx[session_id] = existing

        if existing is None:
            return None

        try:
            parent_span = NonRecordingSpan(existing)
            return otel_trace.set_span_in_context(parent_span)
        except Exception:
            return None


__all__ = [
    "LIVE_COPILOT_AVAILABLE",
    "_copilot_enabled_sessions",
    "_copilot_session_context",
    "_copilot_sessions_lock",
    "flag_enabled",
    "copilot_session_ttl_seconds",
    "get_parent_trace_context",
]
