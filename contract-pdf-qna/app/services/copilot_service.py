from typing import Any, Dict, Optional

try:
    from live_copilot import handle_transcript_event  # type: ignore

    LIVE_COPILOT_AVAILABLE = True
except Exception:
    handle_transcript_event = None
    LIVE_COPILOT_AVAILABLE = False


def handle_transcript_event_safe(payload: Dict[str, Any], parent_context: Optional[Any] = None):
    """
    Thin safety wrapper around live_copilot.handle_transcript_event so routes
    can call it without worrying about missing dependencies.
    """
    if not LIVE_COPILOT_AVAILABLE or handle_transcript_event is None:
        return None
    try:
        return handle_transcript_event(payload, parent_context=parent_context)
    except Exception:
        # Fail-soft to keep webhook responsive.
        return None


__all__ = ["handle_transcript_event_safe", "LIVE_COPILOT_AVAILABLE"]
