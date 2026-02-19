import os
import re
import json
import hashlib
from time import time

from typing import Any, Dict, List, Optional

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)) or default)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except Exception:
        return default



def _log(level: str, icon: str, message: str, **kwargs):
    """Structured logging helper for Live Copilot."""
    extra = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    prefix = f"[LIVE_COPILOT] {icon}"
    if extra:
        print(f"{prefix} {message} | {extra}")
    else:
        print(f"{prefix} {message}")



def _now_epoch() -> int:
    return int(time())


def _s(s: Any) -> str:
    return str(s or "").strip()

def _norm_text(s: str) -> str: return re.sub(r"\s+", " ", _s(s).lower()).strip()

def _fingerprint(obj: Any) -> str:
    try:
        raw = json.dumps(obj, sort_keys=True, default=str)
    except Exception:
        raw = str(obj)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]



def _trace_include_payloads() -> bool:
    raw = (os.getenv("OTEL_TRACE_INCLUDE_PAYLOADS", "0") or "").strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


def _payload_preview_chars() -> int:
    # Bounded preview sizing; must be safe and opt-in.
    try:
        raw = (os.getenv("OTEL_TRACE_PAYLOAD_PREVIEW_CHARS", "0") or "").strip()
        n = int(raw) if raw else 0
        if n <= 0:
            return 0
        # Hard cap to reduce accidental PII leakage / huge spans.
        return min(n, 2000)
    except Exception:
        return 0


def _preview(obj: Any) -> str:
    """
    Produce a bounded, single-line-ish preview string for tracing attributes.
    This should only be used when _trace_include_payloads() is true.
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
        n = _payload_preview_chars()
        if n <= 0:
            return ""
        if len(s) <= n:
            return s
        return s[:n] + "…"
    except Exception:
        return ""
