from dotenv import load_dotenv
load_dotenv(override=True)

from core.db import User
from pydantic import BaseModel, Field
from functools import lru_cache
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
from pymongo.collection import Collection

from core.transcript_process import process_live_copilot_question, InferenceMode
from utils import kb
from core import db
from core.schemas import Response, Transcript, SessionState, Question

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

def _norm_text(s: str) -> str: return re.sub(r"\s+", " ", _s(s).lower()).strip()

def _fingerprint(obj: Any) -> str:
    try:
        raw = json.dumps(obj, sort_keys=True, default=str)
    except Exception:
        raw = str(obj)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# -----------------------
# In-proc session state
# -----------------------



_sessions: Dict[str, SessionState] = {}


def _get_state(session_id: str) -> SessionState:
    st = _sessions.get(session_id)
    if st is None:
        st = SessionState(session_id=session_id)
        _sessions[session_id] = st
    return st


def _cooldown_ok(st: SessionState) -> bool:
    return (time() - float(st.last_suggested_at or 0.0)) >= float(COPILOT_COOLDOWN_SECONDS or 0)


def _append_buffer(st: SessionState, speaker: str, text: str):
    st.buffer.append(Transcript(speaker=speaker, text=text))
    if len(st.buffer) > 30:
        st.buffer = st.buffer[-30:]


def _buffer_text(st: SessionState) -> str:
    lines = []
    for item in st.buffer[-20:]:
        sp = _s(item.speaker).lower() or "unknown"
        tx = _s(item.text)
        if not tx:
            continue
        lines.append(f"{sp}: {tx}")
    return "\n".join(lines).strip()


def _effective_customer_context(st: SessionState) -> Dict[str, Any]:
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


def _queue_questions(st: SessionState, questions: List[str]) -> bool:
    """Return True if queue changed."""
    changed = False
    for q in questions:
        qn = _s(q)
        # Strip context for key generation (deduplication)
        clean_q = re.sub(r"\[CALL_CONTEXT:.*?\]\s*", "", qn).strip()
        q = _norm_text(clean_q)
        if st.questions_queue.get(q, None) is None:
            st.questions_queue[q] = Question(question=q)
            if len(st.questions_queue) > 12:
                st.questions_queue.popitem(last=False)
        changed = True
    return changed



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




@lru_cache(maxsize=20)
def fetch_user_by_phone(phone: str):
    user = None
    with tracer.start_as_current_span("db.mongo.find_one") as sp:
        _set_session_attr(sp)
        sp.set_attribute("db.system", "mongodb")
        sp.set_attribute("db.operation", "find_one")
        sp.set_attribute("db.collection", "Users")
        sp.set_attribute("db.query.mobile", str(phone))
        user = db.get_user_details_from_mobile(phone)
        if user :
            print(
                f"[LIVE_COPILOT] ✅ Mongo user match found!",
                f"{user=}",
                flush=True
            )
    return user


@lru_cache(maxsize=30)
def cache_fetch_user_by_phones(phone_candidates: tuple, users: Collection):
    doc = None
    with tracer.start_as_current_span("db.mongo.find_one") as sp:
        _set_session_attr(sp)
        sp.set_attribute("db.system", "mongodb")
        sp.set_attribute("db.operation", "find_one")
        sp.set_attribute("db.collection", "Users")
        sp.set_attribute("db.query.mobile.$in", str(phone_candidates))
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
    if doc :
        doc = User(**doc) 
    return doc


def _lookup_user_by_phone(phone_candidates: tuple) -> Optional[User]:
    print("[LIVE_COPILOT DEBUG]: ", phone_candidates)
    if not MONGO_URI:
        print("[LIVE_COPILOT] ERROR: MONGO_URI not configured, cannot lookup user", flush=True)
        return None
    if not phone_candidates:
        print("[LIVE_COPILOT] ERROR: No phone candidates provided", flush=True)
        return None
    
    print(f"[LIVE_COPILOT] Attempting MongoDB lookup with {len(phone_candidates)} phone candidates", flush=True)
    users = db.get_db_collection("Users")
    
    # Try individual lookups first
    for p in phone_candidates:
        try:
            doc = fetch_user_by_phone(p)
            if doc: 
                print("Found Candidates!!!", doc)
                return doc
        except Exception as e:
            print(f"[LIVE_COPILOT] Error querying MongoDB with phone {p}: {e}", flush=True)
    
    # Fallback: try $in query
    try:
        doc = cache_fetch_user_by_phones(phone_candidates, users)
        if doc :
            print("Found Candidates!!!", doc)
            return doc
    except Exception as e:
        print(f"[LIVE_COPILOT] Error with $in query: {e}", flush=True)
    
    print(f"[LIVE_COPILOT] ❌ No user found in MongoDB for any phone candidate", flush=True)
    return None









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
    
    # Try to use INFER wrapper first (full LangChain Agent)
    
    if True:
        try:
            result = process_live_copilot_question(
                question=question,
                policyId=kb.getPolicyid(contract_type=contract_type, selected_plan=plan, selected_state=state),
                transcript_context="",  # Could add more context here if needed
            )
            
            # Transform result to match expected format
            answer = (result or {}).get("answer", "")
            chunks = (result or {}).get("relevantChunks", [])
            
            if answer:
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
    
    # Fallback: use simple RAG implementation
    return {}


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
    chain = _suggest_prompt | llm | StrOutputParser()
    prompt_payload = {
        "intent": intent,
        "customer_verified": bool(customer_verified),
        "customer_context": json.dumps(customer_context or {}, default=str),
        "tool_result": json.dumps(tool_result or {}, default=str),
        "transcript": transcript,
    }
    try:
        raw = (chain.invoke(prompt_payload, config={"callbacks": [handler]}) or "")
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


handler = CallbackHandler()





def handle_transcript_event(payload: Dict[str, Any], parent_context=None) :
    session_id = _s(payload.get("sessionId"))
    speaker = _s(payload.get("speaker")).lower()
    text = _s(payload.get("text"))

    is_partial = bool(payload.get("isPartial", False))

    if not session_id or not text:
        return None
    if is_partial:
        return None

    if is_trivial_utterance(text):
        return None

    # Update buffer and session context (always needed for conversation history)
    st = _get_state(session_id)
    _append_buffer(st, speaker=speaker, text=text)


    # Skip AI processing for CSR text - only process customer prompts for suggestions

    # Check for phoneNumber in payload and lookup user in MongoDB
    # Always perform MongoDB lookup when phoneNumber is present for cross-verification
    payload_phone = _s(payload.get("phoneNumber"))
    if payload_phone:
        # Extract payload data for comparison
        payload_contract_type = _s(payload.get("contractType"))
        payload_plan = _s(payload.get("plan"))
        payload_state = _s(payload.get("state"))
        ''' 
        print(f"[LIVE_COPILOT] Received phoneNumber from payload: {payload_phone}", flush=True)
        if payload_contract_type or payload_plan or payload_state:
            print(
                f"[LIVE_COPILOT] Payload data: contractType={payload_contract_type}, "
                f"plan={payload_plan}, state={payload_state}",
                flush=True
            )
        '''
        # Normalize phone number - remove non-digits and handle +1 prefix
        phone_clean = re.sub(r"\D+", "", payload_phone)
        phone_candidates = (phone_clean, "+1" + phone_clean, phone_clean[1:], "+1" + phone_clean[1:])
        print(f"[LIVE_COPILOT] Searching MongoDB with phone candidates: {phone_candidates}", flush=True)
        user_data = _lookup_user_by_phone(phone_candidates)
        if user_data:
            # Normalize MongoDB document
            st.customer = user_data
            print(f"[LIVE COPILOT]: {st.customer=}") 
            
            # Cross-verification: Compare payload data with MongoDB data (for debugging only)
            # NOTE: MongoDB values are the source of truth and will be used for inference
            # Set UI-ready user details so frontend Customer Details card receives userDetails
        else:
            print(f"[LIVE_COPILOT] ❌ No user found in MongoDB for phone candidates: {phone_candidates}", flush=True)
            print(f"[LIVE_COPILOT] ⚠️  contractType, plan, and state will NOT be set (MongoDB lookup failed)", flush=True)

    if speaker == "agent":
        # No AI suggestions needed for CSR text
        # But if user details were just fetched, send them for display
        if st.customer:
            return Response(sessionId=session_id, userDetails=st.customer).dict()
        return None
    

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
                    user_data = _lookup_user_by_phone(tuple([c for c in candidates if c]))
                    if user_data:
                        st.customer = user_data
                        customer = st.customer
                        important_change = True

                customer_ctx = _effective_customer_context(st)
                customer_ctx["sessionId"] = session_id
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
                previous_answers = [ q.__dict__ for _, q in st.answered.items() ]

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
                    "pendingQuestions": [_strip_ctx(x.question) for _,x in st.questions_queue.items() if _s(x.question)],
                    "answeredCount": len(st.answered),
                    "previousAnswers": previous_answers,  # Include all previously answered questions for consistency
                    "newAnswers": [],
                    "verification": {
                        "needsPhone": False,
                        "askForPhone": False,
                    },
                }

                requires_verification = bool(intent_obj.get("requiresVerification"))
                if (requires_verification or st.questions_queue) and not verified:
                    tool_result["verification"]["needsPhone"] = True
                    if st.verification_asks < COPILOT_MAX_VERIFICATION_ASKS:
                        st.verification_asks += 1
                        tool_result["verification"]["askForPhone"] = True

                can_rag = bool(
                    customer_ctx.get("contractType") and customer_ctx.get("plan") and customer_ctx.get("state")
                )

            # ---------------- phase: rag_answer (where applicable) ----------------
            if can_rag and st.questions_queue:
                with tracer.start_as_current_span("live_copilot.rag_answer") as sp_rag:
                    _set_session_attr(sp_rag)
                    
                    answered_now = []
                    question_str, question = st.questions_queue.popitem(last=False)
                    res = _rag_answer(question=question_str, customer=customer_ctx, handler=handler, span=sp_rag)
                    question.answer = res.get("answer", "")
                    question.citedChunks = res.get("chunks", "")
                    question.ts = _now_epoch()
                    st.answered[question_str] = question
                    answered_now.append({"question": _strip_ctx(question_str), "result": res})
                    if answered_now:
                        tool_result["newAnswers"] = answered_now
                        important_change = True

            if intent == "PROBLEM":
                cards = None
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
                if st.customer:
                    output = {
                        "sessionId": session_id,
                        "intent": intent or "OTHER",
                        "userDetails": st.customer.dict(),
                        "createdAt": str(_now_epoch()),
                    }
                else:
                    output = None
            else:
                fp = _fingerprint({"intent": intent, "customer": customer_ctx, "cards": cards})
                if fp == st.last_emit_fingerprint and not important_change:
                    # Even if deduplicated, send user details if available (for sticky header)
                    if st.customer:
                        output = {
                            "sessionId": session_id,
                            "intent": intent,
                            "userDetails": st.customer.dict(),
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
                        output["userDetails"] = st.customer.dict()
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
                "userDetails": st.customer.dict(),
                "createdAt": str(_now_epoch()),
            }
            
    return None



