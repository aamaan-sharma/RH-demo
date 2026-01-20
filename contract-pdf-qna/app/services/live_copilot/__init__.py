"""
Live Copilot Orchestrator (PoC)

Invoked by webhook route (in a SocketIO background task) ONLY when:
- ENABLE_LIVE_COPILOT=1, AND
- session is enabled by Analyze Live UI (copilot_enable), AND
- transcript event arrives for that session.

This module must be safe and fail-soft: callers will swallow exceptions so /webhook remains unchanged.

Uses modular services:
- TranscriptProcessorService for INFER processing
- app.config.settings for configuration
- No dependency on app.py or config.py

Return payload shape (consumed by LiveTranscript UI):
{
  "sessionId": "...",
  "intent": "...",
  "confidence": 0.0,
  "customer": { "verified": bool, "name": "...", "plan": "...", "contractType": "...", "state": "...", "phone": "..." },
  "cards": [ { "title": "...", "csrScript": "...", "evidence": "...", "priority": "high|medium|low" } ],
  "createdAt": "epoch_seconds"
}
"""
from .orchestrator import handle_transcript_event

__all__ = ['handle_transcript_event']
