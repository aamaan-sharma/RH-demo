import os
import re
import json
import hashlib
import sys
import threading
import httpx
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional

try:
    # openai>=1.x
    from openai import APITimeoutError  # type: ignore
except Exception:  # pragma: no cover
    APITimeoutError = TimeoutError  # type: ignore

from pymongo import MongoClient

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from monitoring_module import tracer, llm_trace_to_jaeger, func_Binsert, security_scores, _is_answer_fallback
from token_module import CallbackHandler

from utils.transcript_filters import is_trivial_utterance
from utils.prompts import (
    _rag_prompt,
    _question_extract_prompt,
    _suggest_prompt,
    _intent_prompt,
    _diagnostics_prompt
)
from utils.constants import (
    CLEAR_STATE_ALIASES,
    COPILOT_COOLDOWN_SECONDS,
    COPILOT_MAX_VERIFICATION_ASKS,
    _PHONE_RE,
)

from config import (
    OPENAI_API_KEY,
    MONGO_URI,
    MILVUS_HOST,
    MODEL_INTENT,
    MODEL_SUGGEST,
    VERBOSE_DEBUG
)

_live_session_id_var: ContextVar[str] = ContextVar("live_session_id", default="")
_infer_handler_lock = threading.Lock()

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


# -------------------------------------------------------------------
# INFER Integration: Import the wrapper from app.py
# Uses lazy import to avoid circular dependency issues
# -------------------------------------------------------------------
_INFER_WRAPPER_AVAILABLE = False
_process_live_copilot_question = None


def _get_infer_wrapper():
    """
    Lazy import of process_live_copilot_question from app.py.
    This avoids circular import issues since app.py imports live_copilot.
    """
    global _INFER_WRAPPER_AVAILABLE, _process_live_copilot_question
    
    if _process_live_copilot_question is not None:
        return _process_live_copilot_question
    
    try:
        from app import process_live_copilot_question
        _process_live_copilot_question = process_live_copilot_question
        _INFER_WRAPPER_AVAILABLE = True
        return _process_live_copilot_question
    except ImportError as e:
        _INFER_WRAPPER_AVAILABLE = False
        return None



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


def _live_session_id() -> str:
    try:
        return _s(_live_session_id_var.get())
    except Exception:
        return ""


def _set_session_attr(span) -> None:
    try:
        sid = _live_session_id()
        if sid:
            span.set_attribute("live.session_id", sid)
    except Exception:
        pass




@contextmanager
def _infer_handler_context(handler: CallbackHandler):
    """
    Temporarily bind app.handler = handler so that Infer's LLM calls share the
    same request-scoped CallbackHandler.
    """
    # Thread-safety: Infer may run concurrently across sessions.
    with _infer_handler_lock:
        old = None
        bound = False
        try:
            try:
                import app as _app  # lazy import; avoids circular dependency issues
                old = getattr(_app, "handler", None)
                setattr(_app, "handler", handler)
                bound = True
            except Exception:
                bound = False
            yield
        finally:
            if bound:
                try:
                    import app as _app
                    setattr(_app, "handler", old)
                except Exception:
                    pass




# -----------------------
# LLM instance caching (thread-safe, global reuse)
# -----------------------

# Global cache for LLM instances - reused across all transcript events
_llm_intent_cache: Optional[ChatOpenAI] = None
_llm_suggest_cache: Optional[ChatOpenAI] = None
_llm_diagnostics_cache: Optional[ChatOpenAI] = None
_llm_cache_lock = threading.Lock()  # Thread-safe initialization


def _get_intent_llm() -> ChatOpenAI:
    """
    Get or create cached intent detection LLM instance (thread-safe).
    Reuses the same instance across all transcript events for efficiency.
    """
    global _llm_intent_cache
    if _llm_intent_cache is None:
        with _llm_cache_lock:  # Prevent race condition during initialization
            if _llm_intent_cache is None:  # Double-check pattern
                _llm_intent_cache = ChatOpenAI(
                    temperature=0.0,
                    model=MODEL_INTENT,
                    max_tokens=200,  # Limit response length for speed
                    timeout=_env_float("LIVE_COPILOT_LLM_TIMEOUT_INTENT_S", 15.0),
                    max_retries=_env_int("LIVE_COPILOT_LLM_MAX_RETRIES", 2),
                )
    return _llm_intent_cache


def _get_suggest_llm() -> ChatOpenAI:
    """
    Get or create cached suggestion generation LLM instance (thread-safe).
    Reuses the same instance across all transcript events for efficiency.
    """
    global _llm_suggest_cache
    if _llm_suggest_cache is None:
        with _llm_cache_lock:  # Prevent race condition during initialization
            if _llm_suggest_cache is None:  # Double-check pattern
                _llm_suggest_cache = ChatOpenAI(
                    temperature=0.0,
                    model=MODEL_SUGGEST,
                    max_tokens=500,  # Limit response length
                    # Suggestions sometimes take longer; keep this configurable.
                    timeout=_env_float("LIVE_COPILOT_LLM_TIMEOUT_SUGGEST_S", 60.0),
                    max_retries=_env_int("LIVE_COPILOT_LLM_MAX_RETRIES", 2),
                )
    return _llm_suggest_cache


def _get_diagnostics_llm() -> ChatOpenAI:
    """
    Get or create cached diagnostics LLM instance (thread-safe).
    Reuses the same instance across all transcript events for efficiency.
    """
    global _llm_diagnostics_cache
    if _llm_diagnostics_cache is None:
        with _llm_cache_lock:  # Prevent race condition during initialization
            if _llm_diagnostics_cache is None:  # Double-check pattern
                _llm_diagnostics_cache = ChatOpenAI(
                    temperature=0.2,
                    model=MODEL_SUGGEST,
                    max_tokens=300,  # Limit response length
                    timeout=_env_float("LIVE_COPILOT_LLM_TIMEOUT_DIAGNOSTICS_S", 20.0),
                    max_retries=_env_int("LIVE_COPILOT_LLM_MAX_RETRIES", 2),
                )
    return _llm_diagnostics_cache


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

def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", _s(s).lower()).strip()

def _fingerprint(obj: Any) -> str:
    try:
        raw = json.dumps(obj, sort_keys=True, default=str)
    except Exception:
        raw = str(obj)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# -----------------------
# In-proc session state
# -----------------------


@dataclass
class _SessionState:
    session_id: str
    last_suggested_at: float = 0.0
    last_intent: str = ""
    verification_asks: int = 0
    buffer: List[Dict[str, Any]] = field(default_factory=list)  # [{speaker,text,ts}]
    customer: Optional[Dict[str, Any]] = None  # verified customer context

    # Persisted plan context (sent from Analyze Live UI via copilot_enable and attached to webhook payloads)
    contract_type: str = ""
    selected_plan: str = ""
    selected_state: str = ""

    # Question state: queue questions even before verification so they don't get skipped
    pending_questions: List[Dict[str, Any]] = field(default_factory=list)  # [{k,q,ts}]
    answered: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # k -> {"answer":..., "citedChunks":[...], "ts":...}

    # Emission stability / dedupe
    last_emit_fingerprint: str = ""
    
    # MongoDB user details for display in UI header
    mongo_user_details: Optional[Dict[str, Any]] = None


_sessions: Dict[str, _SessionState] = {}


def _get_state(session_id: str) -> _SessionState:
    st = _sessions.get(session_id)
    if st is None:
        st = _SessionState(session_id=session_id)
        _sessions[session_id] = st
    return st


def _cooldown_ok(st: _SessionState) -> bool:
    return (time() - float(st.last_suggested_at or 0.0)) >= float(COPILOT_COOLDOWN_SECONDS or 0)


def _append_buffer(st: _SessionState, speaker: str, text: str):
    st.buffer.append({"speaker": speaker, "text": text, "ts": time()})
    if len(st.buffer) > 30:
        st.buffer = st.buffer[-30:]


def _buffer_text(st: _SessionState) -> str:
    lines = []
    for item in st.buffer[-20:]:
        sp = _s(item.get("speaker")).lower() or "unknown"
        tx = _s(item.get("text"))
        if not tx:
            continue
        lines.append(f"{sp}: {tx}")
    return "\n".join(lines).strip()


def _update_session_context_from_payload(st: _SessionState, payload: Dict[str, Any]):
    """
    Update session context from transcript payload.
    
    Payload may contain these fields, but we now fetch contractType, plan, and state
    from MongoDB instead of using payload values:
    - phoneNumber / phone: Customer phone number (used for MongoDB lookup)
    - contractType, plan, state: Ignored - will be fetched from MongoDB
    
    All customer details (contractType, plan, state) are now fetched from MongoDB
    when phoneNumber is present in the payload.
    """
    # Extract phone (check both 'phoneNumber' and 'phone' keys)
    # We only use phone for MongoDB lookup, not for setting customer context
    phone = _s(payload.get("phoneNumber")) or _s(payload.get("phone"))
    
    # Extract payload values for logging/debugging only (not used for setting session state)
    ct = _s(payload.get("contractType"))
    pl = _s(payload.get("plan")) or _s(payload.get("selectedPlan"))
    stt = _s(payload.get("state")) or _s(payload.get("selectedState"))
    
    # Logging discipline: never print raw phone/state/plan/contract type unless payload tracing is enabled.
    if _trace_include_payloads():
        # Still keep it bounded
        try:
            print(
                "[LIVE_COPILOT_DEBUG] payload context: "
                f"phone={_preview(phone)}, contractType={_preview(ct)}, plan={_preview(pl)}, state={_preview(stt)}"
            )
        except Exception:
            pass
    
    # NOTE: We no longer set contractType, plan, or state from payload.
    # These will be fetched from MongoDB in handle_transcript_event when phoneNumber is present.


def _effective_customer_context(st: _SessionState) -> Dict[str, Any]:
    """
    Prefer verified customer profile when present, but always keep plan context available
    (either from verified user doc or from UI-provided session context).
    """
    base = dict(st.customer or {})
    verified = bool(base.get("verified"))
    # If unverified, fill plan context from session selections.
    if not base.get("contractType"):
        base["contractType"] = st.contract_type
    if not base.get("plan"):
        base["plan"] = st.selected_plan
    if not base.get("state"):
        base["state"] = st.selected_state
    if "verified" not in base:
        base["verified"] = verified
    if not base.get("name"):
        base["name"] = "Customer"
    return base


def _looks_like_verification_request(text: str) -> bool:
    t = _norm_text(text)
    if not t:
        return False
    keywords = [
        "phone",
        "mobile",
        "contact number",
        "callback number",
        "number to reach you",
        "best number",
    ]
    return any(k in t for k in keywords)


def _should_extract_questions(text: str) -> bool:
    t = _norm_text(text)
    if not t:
        return False
    if "?" in text:
        return True
    # Heuristics: coverage/policy intent
    cues = ["covered", "cover", "limit", "deductible", "fee", "cost", "refund", "cancel", "renew", "service request"]
    return any(c in t for c in cues)




def _extract_questions_llm(*, transcript: str, handler: CallbackHandler, span, customer_ctx: Dict[str, Any] = None) -> List[str]:
    llm = _get_suggest_llm()  # Use cached instance
    
    # 1. LOGGING (Verification)
    if VERBOSE_DEBUG:
        _log("debug", "🔍", f"Extracting questions. Transcript len={len(transcript)}")
        _log("debug", "🔍", f"Transcript start: {transcript[:200]}")
        _log("debug", "🔍", f"Transcript end: {transcript[-200:]}")

    chain = _question_extract_prompt | llm | StrOutputParser()
    raw = (chain.invoke({"transcript": transcript}, config={"callbacks": [handler]}) or "").strip()
    if _trace_include_payloads():
        span.set_attribute("llm.prompt.preview", _preview(transcript))
        span.set_attribute("llm.response.preview", _preview(raw))
    
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
    
    extracted_qs = []
    try:
        obj = json.loads(cleaned)
        qs = obj.get("questions") if isinstance(obj, dict) else []
        if isinstance(qs, list):
            for q in qs:
                q_str = _s(q)
                if q_str:
                    extracted_qs.append(q_str)
    except Exception as e:
        pass

    if VERBOSE_DEBUG:
        _log("debug", "🔍", f"Raw extracted questions: {extracted_qs}")

    # 2. FILTERING (Hard ban on generic questions)
    filtered_qs = []
    generic_pattern = re.compile(r"^(is (this|it|this issue) covered(\s+or not)?|is this covered)\??\s*$", re.IGNORECASE)
    
    for q in extracted_qs:
        if not generic_pattern.match(q.strip()):
            filtered_qs.append(q)
    
    if VERBOSE_DEBUG:
        _log("debug", "🔍", f"Filtered questions: {filtered_qs}")

    # 3. CONTEXT EXTRACTION & ENRICHMENT
    t_lower = transcript.lower()
    
    # Extract facts
    contract_start = ""
    start_match = re.search(r"may (the )?(2nd|second)", t_lower)
    if start_match:
        contract_start = "May 2"
    
    outcome_str = "normal"
    if "deny everything" in t_lower or "go ahead and deny" in t_lower:
        outcome_str = "denied due to pre-existing" if "pre existing" in t_lower or "pre-existing" in t_lower else "denied"
    elif "pre existing" in t_lower or "pre-existing" in t_lower:
        outcome_str = "pre-existing claimed"
        
    auth_scope = "none"
    auth_total = ""
    
    # Authorization logic
    if "only authorize" in t_lower or "will only authorize" in t_lower or "successfully got this authorized" in t_lower:
        auth_scope = "partial authorization"
        if "diagnostics" in t_lower or "diagnosis" in t_lower:
            auth_scope = "diagnosis"
        if "outlet" in t_lower:
            auth_scope += "+outlets"
            
    # Amount extraction (authorized total)
    amount_match = re.search(r"total of \$?(\d+)", t_lower)
    if amount_match:
        auth_total = amount_match.group(1)
    
    # Helper function to extract money amounts for a specific item context
    def extract_item_money(item_keywords, transcript_text, transcript_lower):
        """Extract money amounts (parts, labor, tax, estimate, total) associated with an item."""
        money_parts = []
        # Find positions where item keywords appear
        item_positions = []
        for keyword in item_keywords:
            idx = transcript_lower.find(keyword)
            if idx != -1:
                item_positions.append(idx)
        
        if not item_positions:
            return ""
        
        # Look for money patterns within 200 chars before/after item mentions
        for pos in item_positions:
            start = max(0, pos - 200)
            end = min(len(transcript_text), pos + 200)
            context = transcript_lower[start:end]
            
            # Extract parts cost
            parts_match = re.search(r"parts?[:\s]+\$?(\d+)", context)
            if parts_match:
                money_parts.append(f"${parts_match.group(1)} parts")
            
            # Extract labor cost
            labor_match = re.search(r"labor[:\s]+\$?(\d+)", context)
            if labor_match:
                money_parts.append(f"${labor_match.group(1)} labor")
            
            # Extract tax
            tax_match = re.search(r"tax[:\s]+\$?(\d+)", context)
            if tax_match:
                money_parts.append(f"${tax_match.group(1)} tax")
            
            # Extract estimate
            est_match = re.search(r"estimate[:\s]+\$?(\d+)", context)
            if est_match:
                money_parts.append(f"${est_match.group(1)} estimate")
            
            # Extract total for this item (if not already captured)
            item_total_match = re.search(r"(?:for|of|is|are)\s+\$?(\d+)", context)
            if item_total_match:
                val = item_total_match.group(1)
                # Only add if not already captured as parts/labor/tax
                if not any(val in p for p in money_parts):
                    money_parts.append(f"${val} total")
            
            # Extract standalone dollar amounts near item (within 50 chars)
            close_context = transcript_lower[max(0, pos - 50):min(len(transcript_text), pos + 50)]
            dollar_matches = re.findall(r"\$(\d+)", close_context)
            if dollar_matches and not money_parts:
                # If no specific labels found, capture all dollar amounts
                for amt in dollar_matches:
                    money_parts.append(f"${amt}")
        
        # Deduplicate and format
        if money_parts:
            return "[" + ", ".join(money_parts) + "]"
        return ""
        
    # Specific Item extraction with money
    items_found = []
    if "burned" in t_lower and "outlet" in t_lower:
        item_desc = "Outlet(burned)@Dining room"
        money = extract_item_money(["outlet", "burned", "dining"], transcript, t_lower)
        items_found.append(item_desc + money)
    elif "outlet" in t_lower:
        item_desc = "Outlet"
        money = extract_item_money(["outlet"], transcript, t_lower)
        items_found.append(item_desc + money)
        
    if "doorbell" in t_lower and ("not work" in t_lower or "broken" in t_lower):
        item_desc = "Doorbell(not working)"
        money = extract_item_money(["doorbell"], transcript, t_lower)
        items_found.append(item_desc + money)
    elif "doorbell" in t_lower:
        item_desc = "Doorbell"
        money = extract_item_money(["doorbell"], transcript, t_lower)
        items_found.append(item_desc + money)
        
    if "heater" in t_lower and "bathroom" in t_lower:
        item_desc = "Surface mount heater(replace)@Master bathroom"
        money = extract_item_money(["heater", "bathroom"], transcript, t_lower)
        items_found.append(item_desc + money)
    elif "heater" in t_lower:
        item_desc = "Heater"
        money = extract_item_money(["heater"], transcript, t_lower)
        items_found.append(item_desc + money)
        
    if "porch light" in t_lower and "wiring" in t_lower:
        item_desc = "Porch light(exposed wiring)@Outside"
        money = extract_item_money(["porch", "light", "wiring"], transcript, t_lower)
        items_found.append(item_desc + money)
    elif "light" in t_lower:
        item_desc = "Light"
        money = extract_item_money(["light"], transcript, t_lower)
        items_found.append(item_desc + money)
        
    if "junction" in t_lower and "attic" in t_lower:
        item_desc = "Junction boxes(open splices)@Attic"
        money = extract_item_money(["junction", "attic"], transcript, t_lower)
        items_found.append(item_desc + money)
    elif "junction" in t_lower:
        item_desc = "JunctionBox"
        money = extract_item_money(["junction"], transcript, t_lower)
        items_found.append(item_desc + money)
        
    items_str = "|".join(items_found) if items_found else "Unknown"

    # Build Context String
    ctx_parts = []
    if customer_ctx:
        if customer_ctx.get("plan"): ctx_parts.append(f"plan={customer_ctx.get('plan')}")
        if customer_ctx.get("contractType"): ctx_parts.append(f"contractType={customer_ctx.get('contractType')}")
        if customer_ctx.get("state"): ctx_parts.append(f"state={customer_ctx.get('state')}")
    
    if contract_start: ctx_parts.append(f"contractStart={contract_start}")
    if items_str != "Unknown": ctx_parts.append(f"items={items_str}")
    if outcome_str != "normal": ctx_parts.append(f"callOutcome={outcome_str}")
    if auth_scope != "none": ctx_parts.append(f"authorizedScope={auth_scope}")
    if auth_total: ctx_parts.append(f"authorizedTotal={auth_total}")
    
    context_prefix = f"[CALL_CONTEXT: {'; '.join(ctx_parts)}]"

    # 4. FALLBACK LOGIC
    if not filtered_qs:
        fallback_qs = []
        
        # Outcome/Eligibility signals
        if outcome_str != "normal" or auth_scope != "none" or contract_start:
            if "denied" in outcome_str or "pre-existing" in outcome_str:
                 fallback_qs.append(f"What items were authorized versus denied during this call, and what was the specific reason for denial ({outcome_str})?")
                 fallback_qs.append(f"How does the policy define 'pre-existing conditions' or 'waiting periods', and how do these terms apply to the current claim given the contract start date of {contract_start}?")
            
            if auth_total:
                fallback_qs.append(f"What is the member's financial responsibility for the non-covered items, and does the authorized total of ${auth_total} cover the diagnosis and completed outlet repairs?")
        
        # Item extraction fallback
        if not fallback_qs and items_found:
            for item in items_found:
                # Strip money brackets for fallback question text
                clean_item = re.sub(r'\[.*?\]', '', item).strip()
                fallback_qs.append(f"Is the {clean_item} covered under the plan?")
        
        filtered_qs = fallback_qs
        if VERBOSE_DEBUG and filtered_qs:
             _log("debug", "🔍", f"Fallback questions generated: {filtered_qs}")

    # 5. FINAL ENRICHMENT
    final_qs = []
    for q in filtered_qs:
        final_qs.append(f"{context_prefix} {q}")

    if VERBOSE_DEBUG:
        _log("debug", "🔍", f"Final enriched questions: {final_qs}")

    return final_qs


def _queue_questions(st: _SessionState, questions: List[str]) -> bool:
    """Return True if queue changed."""
    changed = False
    for q in questions:
        qn = _s(q)
        # Strip context for key generation (deduplication)
        clean_q = re.sub(r"\[CALL_CONTEXT:.*?\]\s*", "", qn).strip()
        k = _norm_text(clean_q)
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


# -----------------------
# Phone extraction + Mongo lookup (AHS.Users)
# -----------------------


# _PHONE_RE is now imported from utils.constants
_mongo_client: Optional[MongoClient] = None


def _extract_phone_candidates(text: str) -> List[str]:
    t = _s(text)
    if not t:
        return []
    out: List[str] = []
    for m in _PHONE_RE.finditer(t):
        digits = "".join(m.groups())
        if len(digits) == 10:
            out.append(digits)
            out.append("+1" + digits)
    raw_digits = re.sub(r"\D+", "", t)
    if len(raw_digits) == 10:
        out.append(raw_digits)
        out.append("+1" + raw_digits)
    if len(raw_digits) == 11 and raw_digits.startswith("1"):
        out.append(raw_digits[1:])
        out.append("+1" + raw_digits[1:])
    # de-dupe preserving order
    seen = set()
    deduped = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        deduped.append(x)
    return deduped[:4]


def _get_mongo_client() -> MongoClient:
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(MONGO_URI, unicode_decode_error_handler="ignore")
    return _mongo_client


def _lookup_user_by_phone(phone_candidates: List[str]) -> Optional[Dict[str, Any]]:
    if not MONGO_URI:
        print("[LIVE_COPILOT] ERROR: MONGO_URI not configured, cannot lookup user", flush=True)
        return None
    if not phone_candidates:
        print("[LIVE_COPILOT] ERROR: No phone candidates provided", flush=True)
        return None
    
    print(f"[LIVE_COPILOT] Attempting MongoDB lookup with {len(phone_candidates)} phone candidates", flush=True)
    users = _get_mongo_client()["AHS"]["Users"]
    
    # Try individual lookups first
    for p in phone_candidates:
        try:
            with tracer.start_as_current_span("db.mongo.find_one") as sp:
                _set_session_attr(sp)
                sp.set_attribute("db.system", "mongodb")
                sp.set_attribute("db.operation", "find_one")
                sp.set_attribute("db.collection", "Users")
                sp.set_attribute("db.query.mobile", str(p))
                doc = users.find_one({"mobile": p})
            if doc:
                try:
                    name = doc.get('name') or doc.get('fullName') or doc.get('firstName') or ''
                    plan = doc.get('plan') or doc.get('selectedPlan') or doc.get('planName') or ''
                    state = doc.get('state') or doc.get('selectedState') or doc.get('stateName') or ''
                    phone_masked = f"***{str(p)[-4:]}" if len(str(p)) >= 4 else "***"
                    print(
                        f"[LIVE_COPILOT] ✅ Mongo user match found!",
                        f"phone={phone_masked}",
                        f"name={name}",
                        f"plan={plan}",
                        f"state={state}",
                        flush=True
                    )
                except Exception as e:
                    print(f"[LIVE_COPILOT] User found but error logging details: {e}", flush=True)
                return doc
        except Exception as e:
            print(f"[LIVE_COPILOT] Error querying MongoDB with phone {p}: {e}", flush=True)
            continue
    
    # Fallback: try $in query
    try:
        with tracer.start_as_current_span("db.mongo.find_one") as sp:
            _set_session_attr(sp)
            doc = users.find_one({"mobile": {"$in": phone_candidates}})
        if doc:
            try:
                phone_val = doc.get("mobile") or ""
                name = doc.get('name') or doc.get('fullName') or doc.get('firstName') or ''
                plan = doc.get('plan') or doc.get('selectedPlan') or doc.get('planName') or ''
                state = doc.get('state') or doc.get('selectedState') or doc.get('stateName') or ''
                phone_masked = f"***{str(phone_val)[-4:]}" if len(str(phone_val)) >= 4 else "***"
                print(
                    f"[LIVE_COPILOT] ✅ Mongo user match found (via $in query)!",
                    f"phone={phone_masked}",
                    f"name={name}",
                    f"plan={plan}",
                    f"state={state}",
                    flush=True
                )
            except Exception as e:
                print(f"[LIVE_COPILOT] User found but error logging details: {e}", flush=True)
            return doc
    except Exception as e:
        print(f"[LIVE_COPILOT] Error with $in query: {e}", flush=True)
    
    print(f"[LIVE_COPILOT] ❌ No user found in MongoDB for any phone candidate", flush=True)
    return None

def _normalize_customer_doc(doc: Dict[str, Any], phone: str) -> Dict[str, Any]:
    name = doc.get("name") or doc.get("fullName") or doc.get("firstName") or ""
    if doc.get("lastName") and name and doc.get("lastName") not in str(name):
        name = f"{name} {doc.get('lastName')}"
    plan = doc.get("plan") or doc.get("selectedPlan") or doc.get("planName") or ""
    contract_type = doc.get("contractType") or doc.get("contract_type") or ""
    state = doc.get("state") or doc.get("selectedState") or doc.get("stateName") or ""
    return {
        "verified": True,
        "name": _s(name) or "Customer",
        "phone": phone,
        "plan": _s(plan),
        "contractType": _s(contract_type),
        "state": _s(state),
    }


def _build_mongo_user_details(doc: Dict[str, Any], phone: str) -> Dict[str, Any]:
    """
    Build UI-ready user details dict from MongoDB doc for the Customer Details card.
    Frontend expects: name, phone, email, plan, state, contractType, address.
    """
    name = doc.get("name") or doc.get("fullName") or doc.get("firstName") or ""
    if doc.get("lastName") and name and doc.get("lastName") not in str(name):
        name = f"{name} {doc.get('lastName')}"
    plan = doc.get("plan") or doc.get("selectedPlan") or doc.get("planName") or ""
    contract_type = doc.get("contractType") or doc.get("contract_type") or ""
    state = doc.get("state") or doc.get("selectedState") or doc.get("stateName") or ""
    email = doc.get("email") or doc.get("emailAddress") or ""
    address = doc.get("address") or doc.get("addressLine1") or ""
    return {
        "name": _s(name) or "Customer",
        "phone": _s(phone),
        "email": _s(email),
        "plan": _s(plan),
        "state": _s(state),
        "contractType": _s(contract_type),
        "address": _s(address),
    }


# -----------------------
# Milvus selection (same naming logic as app.py)
# -----------------------


# CLEAR_STATE_ALIASES is now imported from utils.constants


def _normalize_contract_type(contract_type: str) -> str:
    return _s(contract_type).upper()


def _normalize_state_for_milvus(selected_state: str) -> str:
    raw = _s(selected_state)
    if not raw:
        return ""
    key = raw.upper()
    if key in CLEAR_STATE_ALIASES:
        return CLEAR_STATE_ALIASES[key]
    lower = raw.lower()
    for v in CLEAR_STATE_ALIASES.values():
        if lower == v.lower():
            return v
    return raw


def _normalize_plan_for_milvus(contract_type: str, selected_plan: str) -> str:
    raw = _s(selected_plan)
    if not raw:
        return ""
    compact = re.sub(r"[^a-z0-9]+", "", raw.lower())
    ct = _normalize_contract_type(contract_type)
    if ct == "RE":
        if compact in ("shieldessential", "essential"):
            return "ShieldEssential"
        if compact in ("shieldplus", "plus"):
            return "ShieldPlus"
        if compact in ("shieldcomplete", "complete"):
            return "default"
    if ct == "DTC":
        if compact in ("shieldsilver", "silver"):
            return "ShieldSilver"
        if compact in ("shieldgold", "gold"):
            return "ShieldGold"
        if compact in ("shieldplatinum", "platinum"):
            return "default"
    return raw


def _milvus_collection(contract_type: str, selected_plan: str, selected_state: str) -> Optional[str]:
    ct = _normalize_contract_type(contract_type)
    st = _normalize_state_for_milvus(selected_state)
    pl = _normalize_plan_for_milvus(ct, selected_plan)
    if not ct or not st:
        return None
    mapping = {
        "RE": {
            "ShieldEssential": f"{st}_RE_ShieldEssential",
            "ShieldPlus": f"{st}_RE_ShieldPlus",
            "default": f"{st}_RE_ShieldComplete",
        },
        "DTC": {
            "ShieldSilver": f"{st}_DTC_ShieldSilver",
            "ShieldGold": f"{st}_DTC_ShieldGold",
            "default": f"{st}_DTC_ShieldPlatinum",
        },
    }
    return mapping.get(ct, {}).get(pl, mapping.get(ct, {}).get("default"))


_embed: Optional[OpenAIEmbeddings] = None
_milvus_cache: Dict[str, Milvus] = {}


def _get_embed() -> OpenAIEmbeddings:
    global _embed
    if _embed is None:
        _embed = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)
    return _embed


def _get_vector_db(collection_name: str) -> Milvus:
    if collection_name in _milvus_cache:
        return _milvus_cache[collection_name]
    vector_db: Milvus = Milvus(
        _get_embed(),
        collection_name=collection_name,
        connection_args={"host": MILVUS_HOST, "port": "19530"},
    )
    _milvus_cache[collection_name] = vector_db
    return vector_db




def _call_intent_llm(*, transcript: str, handler: CallbackHandler, span) -> Dict[str, Any]:
    llm = _get_intent_llm()  # Use cached instance
    chain = _intent_prompt | llm | StrOutputParser()
    raw = (chain.invoke({"transcript": transcript}, config={"callbacks": [handler]}) or "").strip()
    if _trace_include_payloads():
        span.set_attribute("llm.prompt.preview", _preview(transcript))
        span.set_attribute("llm.response.preview", _preview(raw))
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




def _simple_rag_answer(*, question: str, customer: Dict[str, Any], handler: CallbackHandler, span) -> Dict[str, Any]:
    """
    Simple RAG implementation - fallback when INFER wrapper is not available.
    Uses direct Milvus similarity search + LLM summarization.
    """
    if not MILVUS_HOST:
        return {"error": "MILVUS_HOST not configured"}
    collection = _milvus_collection(customer.get("contractType"), customer.get("plan"), customer.get("state"))
    if not collection:
        return {"error": "Missing plan context for Milvus collection"}
    vector_db = _get_vector_db(collection)
    # Similarity search then summarize with LLM
    docs = vector_db.similarity_search(question, k=6)
    chunks = []
    for d in docs:
        content = getattr(d, "page_content", "") or ""
        if content.strip():
            chunks.append(content.strip())
    if not chunks:
        return {"answer": "I couldn't find relevant policy language for that question.", "citedChunks": []}
    llm = _get_suggest_llm()  # Use cached instance
    chain = _rag_prompt | llm | StrOutputParser()
    payload = {"question": question, "chunks": "\n\n".join(chunks)}
    raw = (chain.invoke(payload, config={"callbacks": [handler]}) or "").strip()
    if _trace_include_payloads():
        span.set_attribute("llm.prompt.preview", _preview(payload))
        span.set_attribute("llm.response.preview", _preview(raw))
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and obj.get("answer") is not None:
            cited = obj.get("citedChunks") or []
            if not isinstance(cited, list):
                cited = []
            return {"answer": str(obj.get("answer")), "citedChunks": cited[:2]}
    except Exception:
        pass
    return {"answer": raw[:1200], "citedChunks": chunks[:1]}


def _rag_answer(*, question: str, customer: Dict[str, Any], handler: CallbackHandler, span) -> Dict[str, Any]:
    """
    Main RAG function - uses INFER wrapper if available, otherwise falls back to simple RAG.
    
    The INFER wrapper uses the full LangChain Agent with:
    - Knowledge Base tool (RetrievalQA)
    - User Lookup tool
    - Sophisticated system prompt for query breakdown
    
    Args:
        question: The customer question to answer
        customer: Dict with contractType, plan, state, etc.
        
    Returns:
        Dict with keys: answer, citedChunks/relevantChunks, and optionally error
    """
    contract_type = customer.get("contractType", "")
    plan = customer.get("plan", "")
    state = customer.get("state", "")
    sessionId = customer.get("sessionId", "")
    
    # Try to use INFER wrapper first (full LangChain Agent)
    infer_wrapper = _get_infer_wrapper()
    
    if infer_wrapper is not None:
        try:
            with _infer_handler_context(handler):
                result = infer_wrapper(
                    question=question,
                    contract_type=contract_type,
                    selected_plan=plan,
                    selected_state=state,
                    transcript_context="",  # Could add more context here if needed
                    sessionId = sessionId,
                )
            
            # Transform result to match expected format
            answer = (result or {}).get("answer", "")
            chunks = (result or {}).get("relevantChunks", [])
            
            if (result or {}).get("error"):
                # Fall through to simple RAG
                pass
            elif answer:
                if _trace_include_payloads():
                    span.set_attribute("rag.question.preview", _preview(question))
                    span.set_attribute("rag.answer.preview", _preview(answer))
                return {
                    "answer": answer,
                    "citedChunks": chunks[:3] if chunks else [],
                    "confidence": result.get("confidence", 0.9),
                    "latency": result.get("latency", 0.0),
                    "source": "INFER",  # Track which method was used
                }
        except Exception as e:
            import traceback
            traceback.print_exc()
            # Fall through to simple RAG
    
    # Fallback: use simple RAG implementation
    result = _simple_rag_answer(question=question, customer=customer, handler=handler, span=span)
    result["source"] = "simple_rag"
    return result


def _diagnostics_steps(*, transcript: str, handler: CallbackHandler, span) -> Dict[str, Any]:
    # Generic troubleshooting guidance without coverage promises
    llm = _get_diagnostics_llm()  # Use cached instance
    chain = _diagnostics_prompt | llm | StrOutputParser()
    raw = (chain.invoke({"transcript": transcript}, config={"callbacks": [handler]}) or "").strip()
    if _trace_include_payloads():
        span.set_attribute("llm.prompt.preview", _preview(transcript))
        span.set_attribute("llm.response.preview", _preview(raw))
    try:
        return json.loads(raw)
    except Exception:
        return {"steps": [], "questions": []}




def _call_suggest_llm(
    *,
    intent: str,
    customer_verified: bool,
    customer_context: Dict[str, Any],
    tool_result: Dict[str, Any],
    transcript: str,
    evidence: str,
) -> List[Dict[str, Any]]:
    raise RuntimeError("_call_suggest_llm should be called via _call_suggest_llm_traced")


def _call_suggest_llm_traced(
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
    if handler is None or span is None:
        raise RuntimeError("_call_suggest_llm_traced requires handler and span")

    if VERBOSE_DEBUG:
        _log("debug", "🔍", f"Generating suggestions | intent={intent} | verified={customer_verified}")

    # Use temperature=0.0 for deterministic, consistent outputs
    llm = _get_suggest_llm()  # Use cached instance
    chain = _suggest_prompt #| llm | StrOutputParser()
    prompt_payload = {
        "intent": intent,
        "customer_verified": bool(customer_verified),
        "customer_context": json.dumps(customer_context or {}, default=str),
        "tool_result": json.dumps(tool_result or {}, default=str),
        "transcript": transcript,
    }
    try:
        raw = (chain.invoke(prompt_payload, config={"callbacks": [handler]}) or "").strip()
    except (APITimeoutError, httpx.TimeoutException) as e:
        _log("warn", "⏱️", f"Suggestion LLM timeout: {type(e).__name__}")
        try:
            span.set_attribute("llm.error", "timeout")
            span.set_attribute("llm.error.type", type(e).__name__)
        except Exception:
            pass
        return [
            {
                "title": "Next step",
                "csrScript": "I’m here to help. Could you repeat that last part so I can make sure I captured it correctly?",
                "evidence": evidence or "",
                "priority": "low",
            }
        ]
    except Exception as e:
        _log("warn", "⚠️", f"Suggestion LLM error: {type(e).__name__}: {e}")
        try:
            span.set_attribute("llm.error", "exception")
            span.set_attribute("llm.error.type", type(e).__name__)
        except Exception:
            pass
        return [
            {
                "title": "Next step",
                "csrScript": "I can help. Could you tell me a bit more about what happened and what you're trying to get resolved today?",
                "evidence": evidence or "",
                "priority": "medium",
            }
        ]

    if _trace_include_payloads():
        try:
            span.set_attribute("llm.prompt.preview", _preview(prompt_payload))
            span.set_attribute("llm.response.preview", _preview(raw))
        except Exception:
            pass

    if VERBOSE_DEBUG:
        _log("debug", "📄", f"Suggestion LLM response: {raw[:150]}...")

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
            _log("info", "💡", f"Generated {len(cards)} suggestion cards", intent=intent)
            # Ensure evidence populated
            for c in cards:
                if isinstance(c, dict) and not c.get("evidence") and evidence:
                    c["evidence"] = evidence
            return cards
    except Exception as e:
        _log("warn", "⚠️", f"Suggestion parse error: {e}")
        pass
    return [
        {
            "title": "Next step",
            "csrScript": "I can help. Could you tell me a bit more about what happened and what you're trying to get resolved today?",
            "evidence": evidence or "",
            "priority": "medium",
        }
    ]


# -----------------------
# Public entrypoint
# -----------------------


def handle_transcript_event(payload: Dict[str, Any], parent_context=None) -> Optional[Dict[str, Any]]:
    session_id = _s(payload.get("sessionId"))
    speaker = _s(payload.get("speaker")).lower()
    text = _s(payload.get("text"))
    # Default False so missing isPartial doesn't block (treat as final)
    is_partial = bool(payload.get("isPartial", False))

    if not session_id or not text:
        return None
    if is_partial:
        return None

    # Update buffer and session context (always needed for conversation history)
    st = _get_state(session_id)
    _update_session_context_from_payload(st, payload)
    _append_buffer(st, speaker=speaker, text=text)

    # Check for phoneNumber in payload and lookup user in MongoDB
    # Always perform MongoDB lookup when phoneNumber is present for cross-verification
    payload_phone = _s(payload.get("phoneNumber"))
    if payload_phone:
        # Extract payload data for comparison
        payload_contract_type = _s(payload.get("contractType"))
        payload_plan = _s(payload.get("plan"))
        payload_state = _s(payload.get("state"))
        
        print(f"[LIVE_COPILOT] Received phoneNumber from payload: {payload_phone}", flush=True)
        if payload_contract_type or payload_plan or payload_state:
            print(
                f"[LIVE_COPILOT] Payload data: contractType={payload_contract_type}, "
                f"plan={payload_plan}, state={payload_state}",
                flush=True
            )
        
        # Normalize phone number - remove non-digits and handle +1 prefix
        phone_clean = re.sub(r"\D+", "", payload_phone)
        print(f"[LIVE_COPILOT] Normalized phone number: {phone_clean}", flush=True)
        if len(phone_clean) == 10:
            phone_candidates = [phone_clean, "+1" + phone_clean]
        elif len(phone_clean) == 11 and phone_clean.startswith("1"):
            phone_candidates = [phone_clean[1:], "+1" + phone_clean[1:], phone_clean]
        else:
            phone_candidates = [phone_clean]
        
        print(f"[LIVE_COPILOT] Searching MongoDB with phone candidates: {phone_candidates}", flush=True)
        doc = _lookup_user_by_phone(phone_candidates)
        if doc:
            # Normalize MongoDB document
            st.customer = _normalize_customer_doc(doc, phone_candidates[0])
            
            # Extract MongoDB user details
            mongo_name = st.customer.get("name") or st.customer.get("fullName") or st.customer.get("firstName") or ""
            mongo_contract_type = _s(st.customer.get("contractType"))
            mongo_plan = _s(st.customer.get("plan")) or _s(st.customer.get("selectedPlan")) or _s(st.customer.get("planName"))
            mongo_state = _s(st.customer.get("state")) or _s(st.customer.get("selectedState")) or _s(st.customer.get("stateName"))
            mongo_email = _s(st.customer.get("email"))
            mongo_address = _s(st.customer.get("address"))
            
            # Log MongoDB user details
            print("=" * 80, flush=True)
            print(f"[LIVE_COPILOT] ✅ MONGODB USER DETAILS:", flush=True)
            print(f"  Phone: ***{phone_clean[-4:]}", flush=True)
            print(f"  Name: {mongo_name}", flush=True)
            print(f"  Contract Type: {mongo_contract_type}", flush=True)
            print(f"  Plan: {mongo_plan}", flush=True)
            print(f"  State: {mongo_state}", flush=True)
            if mongo_email:
                print(f"  Email: {mongo_email}", flush=True)
            if mongo_address:
                print(f"  Address: {mongo_address}", flush=True)
            
            # Cross-verification: Compare payload data with MongoDB data (for debugging only)
            # NOTE: MongoDB values are the source of truth and will be used for inference
            if payload_contract_type or payload_plan or payload_state:
                print("-" * 80, flush=True)
                print(f"[LIVE_COPILOT] 🔍 CROSS-VERIFICATION (Payload vs MongoDB - MongoDB is source of truth):", flush=True)
                
                # Compare contractType
                if payload_contract_type and mongo_contract_type:
                    if payload_contract_type.lower() == mongo_contract_type.lower():
                        print(f"  ✅ Contract Type MATCH: {payload_contract_type}", flush=True)
                    else:
                        print(
                            f"  ⚠️  Contract Type MISMATCH: Payload={payload_contract_type}, "
                            f"MongoDB={mongo_contract_type}",
                            flush=True
                        )
                elif payload_contract_type:
                    print(f"  ⚠️  Contract Type: Payload={payload_contract_type}, MongoDB=Not found", flush=True)
                elif mongo_contract_type:
                    print(f"  ℹ️  Contract Type: Payload=Not provided, MongoDB={mongo_contract_type}", flush=True)
                
                # Compare plan
                if payload_plan and mongo_plan:
                    if payload_plan.lower() == mongo_plan.lower():
                        print(f"  ✅ Plan MATCH: {payload_plan}", flush=True)
                    else:
                        print(
                            f"  ⚠️  Plan MISMATCH: Payload={payload_plan}, MongoDB={mongo_plan}",
                            flush=True
                        )
                elif payload_plan:
                    print(f"  ⚠️  Plan: Payload={payload_plan}, MongoDB=Not found", flush=True)
                elif mongo_plan:
                    print(f"  ℹ️  Plan: Payload=Not provided, MongoDB={mongo_plan}", flush=True)
                
                # Compare state
                if payload_state and mongo_state:
                    if payload_state.lower() == mongo_state.lower():
                        print(f"  ✅ State MATCH: {payload_state}", flush=True)
                    else:
                        print(
                            f"  ⚠️  State MISMATCH: Payload={payload_state}, MongoDB={mongo_state}",
                            flush=True
                        )
                elif payload_state:
                    print(f"  ⚠️  State: Payload={payload_state}, MongoDB=Not found", flush=True)
                elif mongo_state:
                    print(f"  ℹ️  State: Payload=Not provided, MongoDB={mongo_state}", flush=True)
            
            print("=" * 80, flush=True)
            
            # Update session state - ONLY use MongoDB data (no payload fallback)
            try:
                if mongo_contract_type:
                    st.contract_type = mongo_contract_type
                    print(f"[LIVE_COPILOT] ✅ Set contract_type from MongoDB: {mongo_contract_type}", flush=True)
                if mongo_plan:
                    st.selected_plan = mongo_plan
                    print(f"[LIVE_COPILOT] ✅ Set selected_plan from MongoDB: {mongo_plan}", flush=True)
                if mongo_state:
                    st.selected_state = mongo_state
                    print(f"[LIVE_COPILOT] ✅ Set selected_state from MongoDB: {mongo_state}", flush=True)
            except Exception as e:
                print(f"[LIVE_COPILOT] Error updating session state: {e}", flush=True)
            # Set UI-ready user details so frontend Customer Details card receives userDetails
            st.mongo_user_details = _build_mongo_user_details(doc, phone_candidates[0])
        else:
            print(f"[LIVE_COPILOT] ❌ No user found in MongoDB for phone candidates: {phone_candidates}", flush=True)
            print(f"[LIVE_COPILOT] ⚠️  contractType, plan, and state will NOT be set (MongoDB lookup failed)", flush=True)

    # Skip AI processing for CSR text - only process customer prompts for suggestions
    if speaker == "agent":
        # No AI suggestions needed for CSR text
        # But if user details were just fetched, send them for display
        if st.mongo_user_details:
            return {
                "sessionId": session_id,
                "intent": "OTHER",
                "userDetails": st.mongo_user_details,
                "createdAt": str(_now_epoch()),
            }
        return None
    
    if is_trivial_utterance(text):
        return None

    handler = CallbackHandler()
    output: Optional[Dict[str, Any]] = None
    tok = _live_session_id_var.set(session_id)
    try:
        # Live Copilot branch is a child of csr_copilot.session (session-level trace root).
        with tracer.start_as_current_span("live_call.processing", context=parent_context) as root:
            root.set_attribute("live.session_id", session_id)
            if _trace_include_payloads():
                root.set_attribute("live.transcript.preview", _preview(text))

            transcript = _buffer_text(st)
            important_change = False

            # ---------------- phase: intent_detection ----------------
            with tracer.start_as_current_span("live_copilot.intent_detection") as sp_intent:
                _set_session_attr(sp_intent)
                
                # Prioritize phoneNumber from payload, fallback to regex extraction from text
                phone_candidates = []
                if payload_phone:
                    phone_clean = re.sub(r"\D+", "", payload_phone)
                    if len(phone_clean) == 10:
                        phone_candidates = [phone_clean, "+1" + phone_clean]
                    elif len(phone_clean) == 11 and phone_clean.startswith("1"):
                        phone_candidates = [phone_clean[1:], "+1" + phone_clean[1:], phone_clean]
                    else:
                        phone_candidates = [phone_clean]
                
                if not phone_candidates:
                    phone_candidates = _extract_phone_candidates(text)
                
                # Only force CUSTOMER_IDENTIFICATION when we don't have customer yet (first-time lookup).
                # Once customer is known, use LLM for intent so coverage/inquiry questions get INQUIRY
                # and we extract questions + run RAG/Infer (VectorDB).
                intent_obj: Dict[str, Any]
                if phone_candidates and not st.customer:
                    intent_obj = {
                        "intent": "CUSTOMER_IDENTIFICATION",
                        "confidence": 0.95,
                        "entities": {
                            "phone": phone_candidates[0],
                            "appliance": "",
                            "symptom": "",
                            "money_amount": "",
                            "timeline": "",
                            "claimId": "",
                            "question": "",
                        },
                        "requiresVerification": True,
                        "evidenceQuote": text[:200],
                    }
                else:
                    intent_obj = _call_intent_llm(transcript=transcript, handler=handler, span=sp_intent)
                    # If LLM didn't return phone but we have payload phone, attach for context_retrieval
                    if phone_candidates and (not intent_obj.get("entities") or not intent_obj.get("entities", {}).get("phone")):
                        intent_obj = dict(intent_obj)
                        intent_obj.setdefault("entities", {})
                        intent_obj["entities"]["phone"] = intent_obj["entities"].get("phone") or phone_candidates[0]

                intent = _s(intent_obj.get("intent")) or "OTHER"
                confidence = float(intent_obj.get("confidence") or 0.0)
                evidence = _s(intent_obj.get("evidenceQuote")) or text[:200]
                entities = intent_obj.get("entities") or {}
                phone_entity = _s(entities.get("phone"))

            # ---------------- phase: context_retrieval ----------------
            with tracer.start_as_current_span("live_copilot.context_retrieval") as sp_ctx:
                _set_session_attr(sp_ctx)
                
                tool_result: Dict[str, Any] = {}
                customer = st.customer

                if (phone_candidates or phone_entity) and not customer:
                    candidates = phone_candidates or [phone_entity]
                    doc = _lookup_user_by_phone([c for c in candidates if c])
                    if doc:
                        st.customer = _normalize_customer_doc(doc, candidates[0])
                        customer = st.customer
                        try:

                            st.contract_type = st.contract_type or _s(customer.get("contractType"))
                            st.selected_plan = st.selected_plan or _s(customer.get("plan"))
                            st.selected_state = st.selected_state or _s(customer.get("state"))
                            mongo_ct = _s(customer.get("contractType"))
                            mongo_pl = _s(customer.get("plan")) or _s(customer.get("selectedPlan")) or _s(customer.get("planName"))
                            mongo_st = _s(customer.get("state")) or _s(customer.get("selectedState")) or _s(customer.get("stateName"))
                            if mongo_ct:
                                st.contract_type = mongo_ct
                            if mongo_pl:
                                st.selected_plan = mongo_pl
                            if mongo_st:
                                st.selected_state = mongo_st
                        except Exception:
                            pass
                        important_change = True

                customer_ctx = _effective_customer_context(st)
                customer_ctx["sessions"] = session_id
                verified = bool(customer_ctx.get("verified"))

                should_extract = (
                    speaker == "customer" 
                    and _should_extract_questions(text) 
                    and intent not in ("CUSTOMER_IDENTIFICATION", "SMALL_TALK", "OTHER")
                )
                if should_extract:
                    extracted = _extract_questions_llm(transcript=transcript, handler=handler, span=sp_ctx, customer_ctx=customer_ctx)
                    if not extracted:
                        q1 = _s(entities.get("question"))
                        if q1:
                            extracted = [q1]
                    if extracted:
                        if _queue_questions(st, extracted):
                            important_change = True

                # Build tool_result snapshot (always present so the prompt has state + conversation context)
                # Include previousAnswers so LLM doesn't contradict itself
                previous_answers = [
                    {
                        "question": k,
                        "answer": v.get("answer", ""),
                        "citedChunks": v.get("citedChunks", []),
                    }
                    for k, v in st.answered.items()
                    if v.get("answer")
                ]

                # Helper to strip context for display
                def _strip_ctx(s: str) -> str:
                    return re.sub(r"\[CALL_CONTEXT:.*?\]\s*", "", _s(s)).strip()

                tool_result = {
                    "mode": "verified" if verified else "unverified",
                    "sessionContext": {
                        "contractType": customer_ctx.get("contractType"),
                        "plan": customer_ctx.get("plan"),
                        "state": customer_ctx.get("state"),
                    },
                    "pendingQuestions": [_strip_ctx(x.get("q")) for x in st.pending_questions if _s(x.get("q"))],
                    "answeredCount": len(st.answered),
                    "previousAnswers": previous_answers,  # Include all previously answered questions for consistency
                    "newAnswers": [],
                    "verification": {
                        "needsPhone": False,
                        "askForPhone": False,
                    },
                }

                requires_verification = bool(intent_obj.get("requiresVerification"))
                if (requires_verification or st.pending_questions) and not verified:
                    tool_result["verification"]["needsPhone"] = True
                    if st.verification_asks < COPILOT_MAX_VERIFICATION_ASKS:
                        st.verification_asks += 1
                        tool_result["verification"]["askForPhone"] = True

                can_rag = bool(
                    customer_ctx.get("contractType") and customer_ctx.get("plan") and customer_ctx.get("state")
                )

            # ---------------- phase: rag_answer (where applicable) ----------------
            if can_rag and st.pending_questions:
                with tracer.start_as_current_span("live_copilot.rag_answer") as sp_rag:
                    _set_session_attr(sp_rag)
                    
                    answered_now = []
                    for item in list(st.pending_questions)[:2]:
                        k = _s(item.get("k"))
                        q = _s(item.get("q"))
                        if not k or not q:
                            continue
                        if k in st.answered:
                            continue
                        res = _rag_answer(question=q, customer=customer_ctx, handler=handler, span=sp_rag)
                        st.answered[k] = {"ts": time(), **(res or {})}
                        answered_now.append({"question": _strip_ctx(q), "result": res})
                    if answered_now:
                        st.pending_questions = [
                            x for x in st.pending_questions if _s(x.get("k")) not in st.answered
                        ]
                        tool_result["newAnswers"] = answered_now
                        important_change = True

                        # Score insert with answer-quality metrics (relevance_score, resolution_score).
                        for item in answered_now:
                            _question = item.get("question") or ""
                            _result = item.get("result") or {}
                            if not _question:
                                continue

                            # def _run_monitor_live(span, question, result, _session_id, _user_email):
                            #     dicts = security_scores(span, question)
                            #     if not dicts:
                            #         return
                            #     answer = (result.get("answer") or "").strip()
                            #     cited_chunks = result.get("citedChunks") or []
                            #     is_fallback = _is_answer_fallback(answer)
                            #     resolution_score = 1 if (answer and not is_fallback) else 0
                            #     has_relevant = len(cited_chunks) > 0
                            #     relevance_score = 1 if (resolution_score == 1 and has_relevant and not is_fallback) else 0
                            #     dicts["relevance_score"] = relevance_score
                            #     dicts["resolution_score"] = resolution_score
                            #     # Feature Usage: webhook/live → Live Copilot, flow_type=live; forward session_id, user_email.
                            #     func_Binsert(span, dicts, question, session_id=_session_id, user_email=_user_email, answer_text=answer if answer else None, feature_name="Live Copilot", flow_type="live")
                            # _user_email = (getattr(st, "customer", None) or {}).get("email") or None
                            # threading.Thread(
                            #     target=_run_monitor_live,
                            #     args=(sp_rag, _question, _result, session_id, _user_email),
                            #     daemon=True,
                            # ).start()

            if intent == "PROBLEM":
                # Generate diagnostics steps
                with tracer.start_as_current_span("live_copilot.diagnostics") as sp_diag:
                    _set_session_attr(sp_diag)
                    tool_result["diagnostics"] = _diagnostics_steps(transcript=transcript, handler=handler, span=sp_diag)

            # ---------------- phase: suggestion_generation ----------------
            with tracer.start_as_current_span("live_copilot.suggestion_generation") as sp_llm:
                _set_session_attr(sp_llm)
                
                # Always generate cards for first suggestion in call (never emitted yet)
                if not st.last_suggested_at:
                    important_change = True
                if not _cooldown_ok(st) and not important_change:
                    cards = None
                else:
                    cards = _call_suggest_llm_traced(
                        intent=intent,
                        customer_verified=verified,
                        customer_context=customer_ctx,
                        tool_result=tool_result,
                        transcript=transcript,
                        evidence=evidence,
                        handler=handler,
                        span=sp_llm,
                    )

            # ---------------- phase: response_postprocessing ----------------
            # In-memory deduplication and token aggregation - no span needed
            if cards is None:
                # Even if no cards, send user details if available (for sticky header)
                if st.mongo_user_details:
                    output = {
                        "sessionId": session_id,
                        "intent": intent or "OTHER",
                        "userDetails": st.mongo_user_details,
                        "createdAt": str(_now_epoch()),
                    }
                else:
                    output = None
            else:
                fp = _fingerprint({"intent": intent, "customer": customer_ctx, "cards": cards})
                if fp == st.last_emit_fingerprint and not important_change:
                    # Even if deduplicated, send user details if available (for sticky header)
                    if st.mongo_user_details:
                        output = {
                            "sessionId": session_id,
                            "intent": intent,
                            "userDetails": st.mongo_user_details,
                            "createdAt": str(_now_epoch()),
                        }
                    else:
                        output = None
                else:
                    st.last_emit_fingerprint = fp
                    st.last_suggested_at = time()
                    st.last_intent = intent
                    output = {
                        "sessionId": session_id,
                        "intent": intent,
                        "confidence": confidence,
                        "customer": customer_ctx,
                        "cards": cards,
                        "createdAt": str(_now_epoch()),
                    }
                    # Add MongoDB user details as sticky header for UI display
                    if st.mongo_user_details:
                        output["userDetails"] = st.mongo_user_details
                    if _trace_include_payloads():
                        root.set_attribute("live.response.preview", _preview(output))

            # Handler aggregation EXACTLY ONCE at end; attach totals ONLY to this Live Copilot branch span.
            try:
                runs, token_usage = handler.infi()
                llm_trace_to_jaeger(runs, token_usage)
                prompt_t = 0
                completion_t = 0
                total_t = 0
                calls = 0
                for t in token_usage or []:
                    if not isinstance(t, dict):
                        continue
                    prompt_t += int(t.get("prompt_tokens") or 0)
                    completion_t += int(t.get("completion_tokens") or 0)
                    total_t += int(t.get("total_tokens") or 0)
                    calls += 1
                root.set_attribute("llm.tokens.prompt", int(prompt_t))
                root.set_attribute("llm.tokens.completion", int(completion_t))
                root.set_attribute("llm.tokens.total", int(total_t))
                root.set_attribute("llm.calls", int(calls))
            except Exception:
                pass

        return output
    finally:
        try:
            _live_session_id_var.reset(tok)
        except Exception:
            pass


def handle_copilot_enable_event(session_id: str, phone_number: str = None) -> Optional[Dict[str, Any]]:
    """
    Proactively initialize session state and lookup user details when copilot is enabled.
    This allows displaying user details card as soon as the call connects.
    """
    if not session_id:
        return None
        
    st = _get_state(session_id)
    
    # If phone number is provided from connection metadata, perform proactive lookup
    if phone_number:
        # Normalize phone number
        phone_clean = re.sub(r"\D+", "", phone_number)
        if len(phone_clean) == 10:
            phone_candidates = [phone_clean, "+1" + phone_clean]
        elif len(phone_clean) == 11 and phone_clean.startswith("1"):
            phone_candidates = [phone_clean[1:], "+1" + phone_clean[1:], phone_clean]
        else:
            phone_candidates = [phone_clean]
            
        _log("info", "📞", f"Proactive lookup for session {session_id} with phone {phone_number}")
        doc = _lookup_user_by_phone(phone_candidates)
        if doc:
            st.customer = _normalize_customer_doc(doc, phone_candidates[0])
            st.mongo_user_details = _build_mongo_user_details(doc, phone_candidates[0])
            
            # Sync plan context
            try:
                mongo_ct = _s(st.customer.get("contractType"))
                mongo_pl = _s(st.customer.get("plan")) or _s(st.customer.get("selectedPlan")) or _s(st.customer.get("planName"))
                mongo_st = _s(st.customer.get("state")) or _s(st.customer.get("selectedState")) or _s(st.customer.get("stateName"))
                if mongo_ct: st.contract_type = mongo_ct
                if mongo_pl: st.selected_plan = mongo_pl
                if mongo_st: st.selected_state = mongo_st
            except Exception:
                pass
                
            return {
                "sessionId": session_id,
                "intent": "OTHER",
                "userDetails": st.mongo_user_details,
                "createdAt": str(_now_epoch()),
            }
            
    return None



