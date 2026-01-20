"""Question extraction for Live Copilot."""
import json
import re
from typing import List
from time import time
from langchain_core.output_parsers import StrOutputParser
from token_module import CallbackHandler
from utils.prompts import _question_extract_prompt
from .llm_cache import get_suggest_llm
from .tracing import trace_include_payloads, preview
from .utils import s, norm_text


def should_extract_questions(text: str) -> bool:
    """Check if questions should be extracted from text."""
    t = norm_text(text)
    if not t:
        return False
    if "?" in text:
        return True
    # Heuristics: coverage/policy intent
    cues = ["covered", "cover", "limit", "deductible", "fee", "cost", "refund", "cancel", "renew", "service request"]
    return any(c in t for c in cues)


def extract_questions_llm(*, transcript: str, handler: CallbackHandler, span) -> List[str]:
    """Extract questions from transcript using LLM."""
    llm = get_suggest_llm()  # Use cached instance
    chain = _question_extract_prompt | llm | StrOutputParser()
    raw = (chain.invoke({"transcript": transcript}, config={"callbacks": [handler]}) or "").strip()
    if trace_include_payloads():
        span.set_attribute("llm.prompt.preview", preview(transcript))
        span.set_attribute("llm.response.preview", preview(raw))
    
    # Clean markdown code blocks if present
    cleaned = raw
    if "```json" in cleaned:
        cleaned = re.sub(r"```json\n?", "", cleaned)
    if "```" in cleaned:
        cleaned = re.sub(r"```\n?", "", cleaned)
    cleaned = cleaned.strip()
    
    # Also try to find JSON object in the response
    if not cleaned.startswith("{"):
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            cleaned = match.group(0)
    
    try:
        obj = json.loads(cleaned)
        qs = obj.get("questions") if isinstance(obj, dict) else []
        if not isinstance(qs, list):
            return []
        out: List[str] = []
        for q in qs:
            q = s(q)
            if q:
                out.append(q)
        return out[:3]
    except Exception as e:
        return []


def queue_questions(st, questions: List[str]) -> bool:
    """Queue questions for processing. Returns True if queue changed."""
    changed = False
    for q in questions:
        qn = s(q)
        k = norm_text(qn)
        if not k:
            continue
        if k in st.answered:
            continue
        if any(item.get("k") == k for item in st.pending_questions):
            continue
        st.pending_questions.append({"k": k, "q": qn, "ts": time()})
        changed = True
    # cap
    if len(st.pending_questions) > 12:
        st.pending_questions = st.pending_questions[-12:]
    return changed
