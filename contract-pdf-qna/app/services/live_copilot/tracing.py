"""Tracing utilities for Live Copilot."""
import os
import json
from typing import Any
from contextvars import ContextVar
from monitoring_module import tracer

_live_session_id_var: ContextVar[str] = ContextVar("live_session_id", default="")


def trace_include_payloads() -> bool:
    """Check if payload tracing is enabled."""
    raw = (os.getenv("OTEL_TRACE_INCLUDE_PAYLOADS", "0") or "").strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


def payload_preview_chars() -> int:
    """Get max characters for payload preview."""
    try:
        raw = (os.getenv("OTEL_TRACE_PAYLOAD_PREVIEW_CHARS", "0") or "").strip()
        n = int(raw) if raw else 0
        if n <= 0:
            return 0
        # Hard cap to reduce accidental PII leakage / huge spans.
        return min(n, 2000)
    except Exception:
        return 0


def preview(obj: Any) -> str:
    """
    Produce a bounded, single-line-ish preview string for tracing attributes.
    This should only be used when trace_include_payloads() is true.
    """
    try:
        if obj is None:
            s = ""
        elif isinstance(obj, str):
            s = obj
        else:
            try:
                s = json.dumps(obj, sort_keys=True, default=str)
            except Exception:
                s = str(obj)
        s = (s or "").replace("\r", " ").replace("\n", " ").strip()
        n = payload_preview_chars()
        if n <= 0:
            return ""
        if len(s) <= n:
            return s
        return s[:n] + "…"
    except Exception:
        return ""


def live_session_id() -> str:
    """Get current live session ID from context."""
    try:
        from .utils import s
        return s(_live_session_id_var.get())
    except Exception:
        return ""


def set_session_attr(span) -> None:
    """Set session ID attribute on span."""
    try:
        sid = live_session_id()
        if sid:
            span.set_attribute("live.session_id", sid)
    except Exception:
        pass


def span_common(span, agent_name: str, agent_role: str, from_agent: str) -> None:
    """
    Apply consistent metadata across spans for correlation + agent attribution.
    Do not change span names/hierarchy; this is additive metadata only.
    """
    try:
        set_session_attr(span)
        if agent_name:
            span.set_attribute("agent.name", agent_name)
        if agent_role:
            span.set_attribute("agent.role", agent_role)
        if from_agent:
            span.set_attribute("agent.from", from_agent)
        span.set_attribute("agent.type", "simulated")
        span.set_attribute("agent.orchestration", "sequential")
    except Exception:
        pass


def get_tracer():
    """Get tracer instance."""
    return tracer
