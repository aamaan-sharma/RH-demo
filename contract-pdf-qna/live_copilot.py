"""
Live Copilot - Backward compatibility wrapper.

This file is kept for backward compatibility. All functionality has been migrated to:
app/services/live_copilot/

New code should import directly from:
    from app.services.live_copilot import handle_transcript_event
"""
from app.services.live_copilot import handle_transcript_event

__all__ = ['handle_transcript_event']
