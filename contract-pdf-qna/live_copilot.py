import os
import re
import json
import hashlib
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, List, Optional

from pymongo import MongoClient

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Milvus
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from monitoring_module import tracer, llm_trace_to_jaeger
from token_module import CallbackHandler

_live_session_id_var: ContextVar[str] = ContextVar("live_session_id", default="")
_infer_handler_lock = threading.Lock()


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


"""
Live Copilot Orchestrator (PoC)

Invoked by app.py (in a SocketIO background task) ONLY when:
- ENABLE_LIVE_COPILOT=1, AND
- session is enabled by Analyze Live UI (copilot_enable), AND
- transcript event arrives for that session.

This module must be safe and fail-soft: callers will swallow exceptions so /webhook remains unchanged.

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


# -----------------------
# Env + config
# -----------------------


def _env_int(name: str, default: int) -> int:
    try:
        raw = (os.getenv(name) or "").strip()
        if not raw:
            return default
        v = int(raw)
        return v if v > 0 else default
    except Exception:
        return default


# -----------------------
# Tracing helpers (ADD ONLY)
# -----------------------


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


def _span_common(span, agent_name: str, agent_role: str, from_agent: str) -> None:
    """
    Apply consistent metadata across spans for correlation + agent attribution.
    Do not change span names/hierarchy; this is additive metadata only.
    """
    try:
        _set_session_attr(span)
        if agent_name:
            span.set_attribute("agent.name", agent_name)
        if agent_role:
            span.set_attribute("agent.role", agent_role)
        if from_agent:
            span.set_attribute("agent.from", from_agent)
        span.set_attribute("agent.type", "simulated")
        span.set_attribute("agent.orchestration", "sequential")
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


# Hardcoded: emit suggestions at most once per second (no env needed)
COPILOT_COOLDOWN_SECONDS = 1
COPILOT_MAX_VERIFICATION_ASKS = _env_int("COPILOT_MAX_VERIFICATION_ASKS", 2)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MILVUS_HOST = os.getenv("MILVUS_HOST")
MONGO_URI = os.getenv("MONGO_URI")

MODEL_INTENT = os.getenv("COPILOT_MODEL_INTENT", "gpt-4o")
MODEL_SUGGEST = os.getenv("COPILOT_MODEL_SUGGEST", "gpt-4o")

# Debug mode: set VERBOSE_DEBUG=1 to enable detailed logging
VERBOSE_DEBUG = os.getenv("VERBOSE_DEBUG", "").lower() in ("1", "true", "yes")


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
    
    Payload contains these fields directly from Amazon Connect:
    - contractType: Contract type (RE, DTC)
    - plan / selectedPlan: Plan name (ShieldPlus, ShieldGold, etc.)
    - state / selectedState: State name (Texas, California, etc.)
    - phoneNumber / phone: Customer phone number
    
    Since Amazon Connect provides all necessary info, we auto-verify the user.
    """
    # Extract contract type
    ct = _s(payload.get("contractType"))
    if ct:
        st.contract_type = ct
    
    # Extract plan (check both 'plan' and 'selectedPlan' keys)
    pl = _s(payload.get("plan")) or _s(payload.get("selectedPlan"))
    if pl:
        st.selected_plan = pl
    
    # Extract state (check both 'state' and 'selectedState' keys)
    stt = _s(payload.get("state")) or _s(payload.get("selectedState"))
    if stt:
        st.selected_state = stt
    
    # Extract phone (check both 'phoneNumber' and 'phone' keys)
    phone = _s(payload.get("phoneNumber")) or _s(payload.get("phone"))
    
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
    
    # AUTO-VERIFY: Since Amazon Connect provides phoneNumber + plan context,
    # we consider the user verified without DB lookup
    if phone and ct and pl and stt:
        # Create verified customer context directly from payload
        if not st.customer or not st.customer.get("verified"):
            st.customer = {
                "phone": phone,
                "contractType": ct,
                "plan": pl,
                "state": stt,
                "verified": True,  # Auto-verified from Amazon Connect data
                "name": "Customer",
            }
            # No raw logging here.
    elif phone and not st.customer:
        # Fallback: Try DB lookup if we have phone but missing other context
        doc = _lookup_user_by_phone([phone])
        if doc:
            st.customer = _normalize_customer_doc(doc, phone)
            if not st.contract_type:
                st.contract_type = _s(st.customer.get("contractType"))
            if not st.selected_plan:
                st.selected_plan = _s(st.customer.get("plan"))
            if not st.selected_state:
                st.selected_state = _s(st.customer.get("state"))
            # No raw logging here.


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


# -----------------------
# Question extraction (LLM)
# -----------------------

_question_extract_prompt = ChatPromptTemplate.from_template(
    """
You extract customer-intent questions from a live insurance support call.

Return ONLY valid JSON:
{{"questions":["q1","q2"]}}

Rules:
- Extract ONLY customer-intent questions (coverage, limits, exclusions, service steps/timeline/costs).
- If the customer described a problem but did not ask explicitly, infer a likely question.
- Each question must be specific (include appliance/system + issue) unless it's a general policy/process question.
- Max 3 questions.

Transcript (most recent last):
{transcript}
"""
)


def _extract_questions_llm(*, transcript: str, handler: CallbackHandler, span) -> List[str]:
    llm = ChatOpenAI(temperature=0.0, model=MODEL_SUGGEST)
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
    
    try:
        obj = json.loads(cleaned)
        qs = obj.get("questions") if isinstance(obj, dict) else []
        if not isinstance(qs, list):
            return []
        out: List[str] = []
        for q in qs:
            q = _s(q)
            if q:
                out.append(q)
        return out[:3]
    except Exception as e:
        return []


def _queue_questions(st: _SessionState, questions: List[str]) -> bool:
    """Return True if queue changed."""
    changed = False
    for q in questions:
        qn = _s(q)
        k = _norm_text(qn)
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


_PHONE_RE = re.compile(r"(?:(?:\+?1\s*)?)\(?\s*(\d{3})\s*\)?[\s.-]?(\d{3})[\s.-]?(\d{4})")
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
        return None
    if not phone_candidates:
        return None
    users = _get_mongo_client()["AHS"]["Users"]
    for p in phone_candidates:
        with tracer.start_as_current_span("db.mongo.find_one") as sp:
            _set_session_attr(sp)
            sp.set_attribute("db.system", "mongodb")
            sp.set_attribute("db.operation", "find_one")
            sp.set_attribute("db.collection", "Users")
            doc = users.find_one({"mobile": p})
        if doc:
            return doc
    with tracer.start_as_current_span("db.mongo.find_one") as sp:
        _set_session_attr(sp)
        sp.set_attribute("db.system", "mongodb")
        sp.set_attribute("db.operation", "find_one")
        sp.set_attribute("db.collection", "Users")
        return users.find_one({"mobile": {"$in": phone_candidates}})


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


# -----------------------
# Milvus selection (same naming logic as app.py)
# -----------------------


CLEAR_STATE_ALIASES = {
    "AZ": "Arizona",
    "CA": "California",
    "GA": "Georgia",
    "MD": "Maryland",
    "MN": "Minnesota",
    "NV": "Nevada",
    "TX": "Texas",
    "UT": "Utah",
    "WI": "Wisconsin",
}


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


# -----------------------
# Agent 1: intent classifier
# -----------------------


_intent_prompt = ChatPromptTemplate.from_template(
    """
You are an intent classifier for a live insurance customer support call.\n
Return ONLY valid JSON in exactly this schema:\n
{{
  \"intent\": \"CUSTOMER_IDENTIFICATION|INQUIRY|PROBLEM|CLAIM_STATUS|COMPLAINT|SMALL_TALK|OTHER\",
  \"confidence\": 0.0,
  \"entities\": {{
    \"phone\": \"string_or_empty\",
    \"appliance\": \"string_or_empty\",
    \"symptom\": \"string_or_empty\",
    \"money_amount\": \"string_or_empty\",
    \"timeline\": \"string_or_empty\",
    \"claimId\": \"string_or_empty\",
    \"question\": \"string_or_empty\"
  }},
  \"requiresVerification\": true,
  \"evidenceQuote\": \"verbatim quote from the customer\"
}}
\nRules:\n
- If you see a phone number, intent MUST be CUSTOMER_IDENTIFICATION with confidence >= 0.9 and entities.phone filled.\n
- CLAIM_STATUS means the customer asks about an existing claim status/ETA/scheduling.\n
- COMPLAINT means frustration, threats to cancel, anger, escalation requests.\n
- INQUIRY means coverage/plan/policy/terms questions.\n
- PROBLEM means a malfunction/issue report (\"not working\", \"leaking\", etc.).\n
- SMALL_TALK greetings/thanks/off-topic.\n
- requiresVerification should be true for CLAIM_STATUS and for plan-specific coverage confirmation.\n
\nRecent transcript (most recent last):\n
{transcript}\n
"""
)


def _call_intent_llm(*, transcript: str, handler: CallbackHandler, span) -> Dict[str, Any]:
    llm = ChatOpenAI(temperature=0.0, model=MODEL_INTENT)
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


# -----------------------
# Verified RAG (Milvus) + generic tool results
# -----------------------


_rag_prompt = ChatPromptTemplate.from_template(
    """
You are assisting a customer care executive.\n
Use ONLY the provided policy chunks to answer. If insufficient, say what is missing.\n
Be concise and professional.\n
Question:\n
{question}\n
Policy chunks:\n
{chunks}\n
Return ONLY JSON:\n
{{\"answer\":\"...\",\"citedChunks\":[\"...\"]}}\n
"""
)


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
    llm = ChatOpenAI(temperature=0.0, model=MODEL_SUGGEST)
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
    prompt = ChatPromptTemplate.from_template(
        """
You are a troubleshooting assistant for home appliance/system issues.\n
Return only JSON: {{\"steps\":[\"...\"],\"questions\":[\"...\"]}}\n
Transcript:\n
{transcript}\n
"""
    )
    llm = ChatOpenAI(temperature=0.2, model=MODEL_SUGGEST)
    chain = prompt | llm | StrOutputParser()
    raw = (chain.invoke({"transcript": transcript}, config={"callbacks": [handler]}) or "").strip()
    if _trace_include_payloads():
        span.set_attribute("llm.prompt.preview", _preview(transcript))
        span.set_attribute("llm.response.preview", _preview(raw))
    try:
        return json.loads(raw)
    except Exception:
        return {"steps": [], "questions": []}


# -----------------------
# Agent 2: suggestion generator
# -----------------------


_suggest_prompt = ChatPromptTemplate.from_template(
    """
You are a real-time copilot helping a CSR (Customer Service Representative) during a live home warranty insurance call.

Your role is to generate PROFESSIONAL, CALM, and CONCISE suggestions that the CSR can say directly to the customer.

STRICT GROUNDING RULES (CRITICAL - FOLLOW EXACTLY):
- NEVER invent or guess dollar amounts, fees, limits, or coverage percentages
- ONLY use specific numbers/values that appear in tool_result.newAnswers or tool_result.previousAnswers
- If a specific fee/limit is NOT in tool_result, say "Let me verify the exact amount for your plan"
- If you previously answered a question (check previousAnswers), use THE SAME answer - never contradict yourself
- When tool_result contains an answer, quote the numbers EXACTLY as they appear

OPERATING RULES:
- Use conversation context below (do not ignore earlier customer questions).
- Use tool_result + customer_context as your ground truth; do NOT invent coverage details.
- If plan context (contractType/plan/state) is missing, suggest asking CSR to confirm it before making commitments.
- If customer_context shows "verified": true, DO NOT ask for phone verification - the user is already verified!
- When user is verified, focus on answering their questions using newAnswers from tool_result.
- Do NOT re-answer questions already addressed; reference prior answer and suggest next step.
- Generate 1-3 suggestion cards focused on the customer's actual questions/issues.

CSR SCRIPT TONE REQUIREMENTS:
- Be CALM and reassuring - avoid alarming language
- Be CONCISE - 1-2 sentences maximum
- Be PROFESSIONAL - use polite, helpful language
- Be DIRECT about coverage decisions (Yes, covered / No, not covered / Partially covered)
- Include specific details ONLY when they are in tool_result

EXAMPLES OF GOOD CSR SCRIPTS:
- "Good news! Your plan does cover water heater repairs. [Use exact fee from tool_result], and we can dispatch a technician within 24-48 hours."
- "I understand your concern about the refrigerator. Unfortunately, cosmetic damage to the exterior panel is not covered under your plan, but I can help you with other options."
- "Based on your plan, drain line stoppages are covered. Let me create a service request for you."

Return ONLY valid JSON:
{{
  "cards": [
    {{
      "title": "Coverage Confirmation",
      "csrScript": "The calm, professional sentence CSR says to customer",
      "evidence": "Verbatim customer quote that triggered this",
      "priority": "high|medium|low"
    }}
  ]
}}

intent: {intent}
customer_context: {customer_context}
tool_result: {tool_result}

Conversation context (most recent last):
{transcript}
"""
)


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
    llm = ChatOpenAI(temperature=0.0, model=MODEL_SUGGEST)
    chain = _suggest_prompt | llm | StrOutputParser()
    prompt_payload = {
        "intent": intent,
        "customer_verified": bool(customer_verified),
        "customer_context": json.dumps(customer_context or {}, default=str),
        "tool_result": json.dumps(tool_result or {}, default=str),
        "transcript": transcript,
    }
    raw = (chain.invoke(prompt_payload, config={"callbacks": [handler]}) or "").strip()

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
    is_partial = bool(payload.get("isPartial", True))

    if not session_id or not text:
        return None
    if is_partial:
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

            with tracer.start_as_current_span("orchestrator_agent") as orch:
                _span_common(
                    orch,
                    agent_name="orchestrator-agent.live-infer",
                    agent_role="Coordinates intent detection, retrieval, and response generation",
                    from_agent="live_call.processing",
                )

                st = _get_state(session_id)
                _update_session_context_from_payload(st, payload)
                _append_buffer(st, speaker=speaker, text=text)
                transcript = _buffer_text(st)

                important_change = False

                # ---------------- phase: intent_detection ----------------
                with tracer.start_as_current_span("intent_detection") as sp_intent:
                    _span_common(
                        sp_intent,
                        agent_name="atomic-agent.intent_detection",
                        agent_role="Intent classification + entity extraction",
                        from_agent="orchestrator-agent.live-infer",
                    )

                    phone_candidates = _extract_phone_candidates(text)
                    intent_obj: Dict[str, Any]
                    if speaker == "csr" and _looks_like_verification_request(text):
                        intent_obj = {
                            "intent": "CUSTOMER_IDENTIFICATION",
                            "confidence": 0.9,
                            "entities": {"phone": "", "question": ""},
                            "requiresVerification": True,
                            "evidenceQuote": text[:200],
                        }
                    elif phone_candidates:
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

                    intent = _s(intent_obj.get("intent")) or "OTHER"
                    confidence = float(intent_obj.get("confidence") or 0.0)
                    evidence = _s(intent_obj.get("evidenceQuote")) or text[:200]
                    entities = intent_obj.get("entities") or {}
                    phone_entity = _s(entities.get("phone"))

                # ---------------- phase: context_retrieval ----------------
                with tracer.start_as_current_span("context_retrieval") as sp_ctx:
                    _span_common(
                        sp_ctx,
                        agent_name="atomic-agent.context_retrieval",
                        agent_role="Load customer context, DB lookups, and question extraction",
                        from_agent="atomic-agent.intent_detection",
                    )

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
                            except Exception:
                                pass
                            important_change = True

                    customer_ctx = _effective_customer_context(st)
                    verified = bool(customer_ctx.get("verified"))

                    should_extract = speaker == "customer" and _should_extract_questions(text)
                    if should_extract:
                        extracted = _extract_questions_llm(transcript=transcript, handler=handler, span=sp_ctx)
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

                    tool_result = {
                        "mode": "verified" if verified else "unverified",
                        "sessionContext": {
                            "contractType": customer_ctx.get("contractType"),
                            "plan": customer_ctx.get("plan"),
                            "state": customer_ctx.get("state"),
                        },
                        "pendingQuestions": [x.get("q") for x in st.pending_questions if _s(x.get("q"))],
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
                    with tracer.start_as_current_span("rag_answer") as sp_rag:
                        _span_common(
                            sp_rag,
                            agent_name="atomic-agent.rag_answer",
                            agent_role="Answer queued questions via Infer pipeline (RAG)",
                            from_agent="atomic-agent.context_retrieval",
                        )
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
                            answered_now.append({"question": q, "result": res})
                        if answered_now:
                            st.pending_questions = [
                                x for x in st.pending_questions if _s(x.get("k")) not in st.answered
                            ]
                            tool_result["newAnswers"] = answered_now
                            important_change = True

                if intent == "PROBLEM":
                    # Keep within rag_answer when applicable (same operational bucket: "tools").
                    with tracer.start_as_current_span("rag_answer") as sp_rag:
                        _span_common(
                            sp_rag,
                            agent_name="atomic-agent.rag_answer",
                            agent_role="Generate generic diagnostics steps (non-coverage)",
                            from_agent="atomic-agent.context_retrieval",
                        )
                        tool_result["diagnostics"] = _diagnostics_steps(transcript=transcript, handler=handler, span=sp_rag)

                # ---------------- phase: llm_call ----------------
                with tracer.start_as_current_span("llm_call") as sp_llm:
                    _span_common(
                        sp_llm,
                        agent_name="atomic-agent.llm_call",
                        agent_role="Generate CSR suggestion cards",
                        from_agent="atomic-agent.rag_answer" if can_rag else "atomic-agent.context_retrieval",
                    )

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
                with tracer.start_as_current_span("response_postprocessing") as sp_post:
                    _span_common(
                        sp_post,
                        agent_name="atomic-agent.response_postprocessing",
                        agent_role="Dedupe and finalize response payload",
                        from_agent="atomic-agent.llm_call",
                    )

                    if cards is None:
                        output = None
                    else:
                        fp = _fingerprint({"intent": intent, "customer": customer_ctx, "cards": cards})
                        if fp == st.last_emit_fingerprint and not important_change:
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
                            if _trace_include_payloads():
                                sp_post.set_attribute("live.response.preview", _preview(output))

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


