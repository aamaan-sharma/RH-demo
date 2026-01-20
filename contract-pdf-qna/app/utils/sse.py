"""Server-Sent Events utilities."""
import json
from typing import Dict, Any


def format_sse(event: str, data: Dict[str, Any]) -> str:
    """Format a Server-Sent Event (SSE) message.
    
    Args:
        event: Event type name
        data: Event data dictionary
        
    Returns:
        Formatted SSE string
    """
    try:
        payload = json.dumps(data, ensure_ascii=False)
    except Exception:
        payload = json.dumps({"error": "Failed to encode SSE payload"})
    return f"event: {event}\ndata: {payload}\n\n"
