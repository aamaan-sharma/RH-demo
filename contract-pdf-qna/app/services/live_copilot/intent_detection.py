"""Intent detection for Live Copilot."""
import json
import re
from typing import Dict, Any
from langchain_core.output_parsers import StrOutputParser
from token_module import CallbackHandler
from utils.prompts import _intent_prompt
from .llm_cache import get_intent_llm
from .tracing import trace_include_payloads, preview, set_session_attr
from .utils import s


def call_intent_llm(*, transcript: str, handler: CallbackHandler, span) -> Dict[str, Any]:
    """Call LLM for intent detection."""
    llm = get_intent_llm()  # Use cached instance
    chain = _intent_prompt | llm | StrOutputParser()
    raw = (chain.invoke({"transcript": transcript}, config={"callbacks": [handler]}) or "").strip()
    if trace_include_payloads():
        span.set_attribute("llm.prompt.preview", preview(transcript))
        span.set_attribute("llm.response.preview", preview(raw))
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{[\s\S]*\}$", raw)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {
        "intent": "OTHER",
        "confidence": 0.2,
        "entities": {
            "phone": "",
            "appliance": "",
            "symptom": "",
            "money_amount": "",
            "timeline": "",
            "claimId": "",
            "question": "",
        },
        "requiresVerification": False,
        "evidenceQuote": "",
    }
