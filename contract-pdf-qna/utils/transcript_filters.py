"""
Shared transcript gating utilities used by both the webhook route and Live Copilot.
Helps reduce latency/cost by skipping trivial or irrelevant transcript events.
"""

import re
from typing import Any, Dict

def s(x: Any) -> str:
    return str(x or "").strip()

def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", s(text).lower()).strip()

def is_trivial_utterance(text: str) -> bool:
    """
    Detect trivial utterances that don't need AI processing.
    Saves cost and improves speed by skipping LLM calls for simple responses.
    """
    if not text:
        return True
    
    normalized = norm_text(text)
    if not normalized:
        return True
    
    # Very short responses (1-2 words) that are likely trivial
    words = normalized.split()
    if len(words) <= 2:
        trivial_responses = {
            "hi", "hello", "hey", "hi there", "hello there",
            "yes", "yeah", "yep", "yup", "sure", "okay", "ok", "alright", "fine",
            "no", "nope", "nah",
            "thanks", "thank you", "thank", "thx",
            "bye", "goodbye", "see you", "later",
            "uh", "um", "hmm", "huh",
            "i'm fine", "i am fine", "doing good", "doing well",
            "got it", "understood", "i see",
            "please", "please help",
            "how are you", "how are you doing",
        }
        if normalized in trivial_responses:
            return True
    
    # Very short text (less than 10 chars after normalization)
    if len(normalized) < 10:
        return True
    
    return False


def should_start_copilot(payload: Dict[str, Any]) -> bool:
    # Must be final transcript
    if bool(payload.get("isPartial", True)):
        return False

    speaker = s(payload.get("speaker")).lower()
    if speaker != "customer":
        return False

    text = s(payload.get("text"))
    if not text:
        return False

    if is_trivial_utterance(text):
        return False

    return True