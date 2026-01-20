"""OpenTelemetry tracing utilities."""
import threading
from typing import Optional, Any
from app.config.settings import settings

# Session-level trace context (1 trace per live sessionId)
_session_trace_ctx: dict = {}  # sessionId -> opentelemetry.trace.SpanContext
_session_trace_lock = threading.Lock()


def get_or_create_session_trace_context(session_id: str) -> Optional[Any]:
    """Ensure a single trace per sessionId by creating ONE root span.
    
    Args:
        session_id: Session ID
        
    Returns:
        Span context or None
    """
    if not session_id:
        return None
    
    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.trace import NonRecordingSpan
        from monitoring_module import tracer
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
