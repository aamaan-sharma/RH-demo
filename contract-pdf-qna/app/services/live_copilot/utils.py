"""Utility functions for Live Copilot."""
import os
import re
import json
import hashlib
from typing import Any


def env_int(name: str, default: int) -> int:
    """Get integer from environment variable."""
    try:
        raw = (os.getenv(name) or "").strip()
        if not raw:
            return default
        v = int(raw)
        return v if v > 0 else default
    except Exception:
        return default


def s(value: Any) -> str:
    """Convert value to string and strip whitespace."""
    return str(value or "").strip()


def norm_text(text: str) -> str:
    """Normalize text: lowercase and collapse whitespace."""
    return re.sub(r"\s+", " ", s(text).lower()).strip()


def fingerprint(obj: Any) -> str:
    """Generate fingerprint hash for object."""
    try:
        raw = json.dumps(obj, sort_keys=True, default=str)
    except Exception:
        raw = str(obj)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def now_epoch() -> int:
    """Get current epoch timestamp."""
    from time import time
    return int(time())


def log(level: str, icon: str, message: str, **kwargs):
    """Structured logging helper for Live Copilot."""
    extra = " | ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    prefix = f"[LIVE_COPILOT] {icon}"
    if extra:
        print(f"{prefix} {message} | {extra}")
    else:
        print(f"{prefix} {message}")
