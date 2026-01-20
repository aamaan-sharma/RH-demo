"""Suggestion generation for Live Copilot."""
import json
import re
from typing import List, Dict, Any
from langchain_core.output_parsers import StrOutputParser
from token_module import CallbackHandler
from utils.prompts import _suggest_prompt, _diagnostics_prompt
from .llm_cache import get_suggest_llm, get_diagnostics_llm
from .tracing import trace_include_payloads, preview
from .utils import log
from app.config.settings import settings


def diagnostics_steps(*, transcript: str, handler: CallbackHandler, span) -> Dict[str, Any]:
    """Generate generic troubleshooting guidance without coverage promises."""
    llm = get_diagnostics_llm()  # Use cached instance
    chain = _diagnostics_prompt | llm | StrOutputParser()
    raw = (chain.invoke({"transcript": transcript}, config={"callbacks": [handler]}) or "").strip()
    if trace_include_payloads():
        span.set_attribute("llm.prompt.preview", preview(transcript))
        span.set_attribute("llm.response.preview", preview(raw))
    try:
        return json.loads(raw)
    except Exception:
        return {"steps": [], "questions": []}


def call_suggest_llm_traced(
    *,
    intent: str,
    customer_verified: bool,
    customer_context: Dict[str, Any],
    tool_result: Dict[str, Any],
    transcript: str,
    evidence: str,
    handler: CallbackHandler,
    span,
) -> List[Dict[str, Any]]:
    """Generate CSR suggestion cards using LLM."""
    if handler is None or span is None:
        raise RuntimeError("_call_suggest_llm_traced requires handler and span")

    if settings.VERBOSE_DEBUG:
        log("debug", "🔍", f"Generating suggestions | intent={intent} | verified={customer_verified}")

    # Use temperature=0.0 for deterministic, consistent outputs
    llm = get_suggest_llm()  # Use cached instance
    chain = _suggest_prompt | llm | StrOutputParser()
    prompt_payload = {
        "intent": intent,
        "customer_verified": bool(customer_verified),
        "customer_context": json.dumps(customer_context or {}, default=str),
        "tool_result": json.dumps(tool_result or {}, default=str),
        "transcript": transcript,
    }
    raw = (chain.invoke(prompt_payload, config={"callbacks": [handler]}) or "").strip()

    if trace_include_payloads():
        try:
            span.set_attribute("llm.prompt.preview", preview(prompt_payload))
            span.set_attribute("llm.response.preview", preview(raw))
        except Exception:
            pass

    if settings.VERBOSE_DEBUG:
        log("debug", "📄", f"Suggestion LLM response: {raw[:150]}...")

    # Clean markdown if present
    cleaned = raw
    if "```json" in cleaned:
        cleaned = re.sub(r"```json\n?", "", cleaned)
    if "```" in cleaned:
        cleaned = re.sub(r"```\n?", "", cleaned)
    cleaned = cleaned.strip()

    try:
        obj = json.loads(cleaned)
        cards = obj.get("cards") if isinstance(obj, dict) else None
        if isinstance(cards, list) and cards:
            log("info", "💡", f"Generated {len(cards)} suggestion cards", intent=intent)
            # Ensure evidence populated
            for c in cards:
                if isinstance(c, dict) and not c.get("evidence") and evidence:
                    c["evidence"] = evidence
            return cards
    except Exception as e:
        log("warn", "⚠️", f"Suggestion parse error: {e}")
        pass
    return [
        {
            "title": "Next step",
            "csrScript": "I can help. Could you tell me a bit more about what happened and what you're trying to get resolved today?",
            "evidence": evidence or "",
            "priority": "medium",
        }
    ]
