from dotenv import load_dotenv
load_dotenv(override=True)

from concurrent.futures import ThreadPoolExecutor
from core.db import User
from functools import lru_cache
from contextvars import ContextVar
from time import time
from typing import Any, Dict, List, Optional
from pymongo.collection import Collection

from core import db
from core.schemas import Response, Transcript, SessionState, Question
from core.schemas import QA



from monitoring_module import tracer, llm_trace_to_jaeger, func_Binsert, security_scores, _is_answer_fallback
from token_module import CallbackHandler

from utils.transcript_filters import is_trivial_utterance
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


'''
Changes
- handle_live_copilot will be a synchrnous function.
    1. Append payload data to a queue (with lock).
    2. notify the worker (background task).

- worker.
    - process the transcript event (sequential in nature).
    - push out socketio events.


'''

_live_session_id_var: ContextVar[str] = ContextVar("live_session_id", default="")

import re
from utils.helpers import _s, _norm_text, _trace_include_payloads, _preview, _now_epoch, _fingerprint, _log
from core.call_llms import _call_intent_llm, _call_suggest_llm_traced, _diagnostics_steps, _extract_questions_llm, _rag_answer

QNA_THREAD_POOL_SIZE=5
qna_executor = ThreadPoolExecutor(max_workers=QNA_THREAD_POOL_SIZE)



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





_sessions: Dict[str, SessionState] = {}


def _get_state(session_id: str) -> SessionState:
    st = _sessions.get(session_id)
    if st is None:
        st = SessionState(session_id=session_id)
        _sessions[session_id] = st
    return st








def _effective_customer_context(st: User) -> Dict[str, Any]:
    """
    Prefer verified customer profile when present, but always keep plan context available
    (either from verified user doc or from UI-provided session context).
    """
    base = st.model_dump()
    verified = bool(base.get("verified", True))
    # If unverified, fill plan context from session selections.
    return base



def _should_extract_questions(text: str) -> bool:
    t = _norm_text(text)
    if not t:
        return False
    if "?" in text:
        return True
    # Heuristics: coverage/policy intent
    cues = ["covered", "cover", "limit", "deductible", "fee", "cost", "refund", "cancel", "renew", "service request"]
    return any(c in t for c in cues)



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


handler = CallbackHandler()

from core.schemas import CopilotSessionData
from typing import Literal
from queue import Queue
from collections import deque

text_buffer = deque(maxlen=3)
copilotQueue = Queue()






def process_transcript_event_loop(socketio,parent_context=None):
    global copilotQueue
    print('[LIVE COPILOT][PROCESSING TRANSCRIPT] Working ...')
    while True:
        transcriptChunk = copilotQueue.get()
        if transcriptChunk == None:
            break
        try:
            result = process_transcript_event(transcriptChunk, parent_context)
            socketio.emit("suggestion_update", result, room=transcriptChunk.sessionId)
        except Exception as e:
            print('[LIVE COPILOT][PROCESSING], An Error Occured ', e)
        


questions = []
def process_transcript_event(payload: CopilotSessionData, parent_context=None):
    global questions
    global deque

    session_id = payload.sessionId
    speaker = payload.speaker
    deque.append(payload.text)
    text = "\n".join(deque)
    customer_ctx = None

    is_partial = payload.isPartial

    if not session_id or not text:
        return None


    payload_phone = payload.phoneNumber
    phone_clean = re.sub(r"\D+", "", payload_phone)
    phone_candidates = (phone_clean, "+1" + phone_clean, phone_clean[1:], "+1" + phone_clean[1:])
    print(f"[LIVE_COPILOT] Searching MongoDB with phone candidates: {phone_candidates}", flush=True)
    user_data = _lookup_user_by_phone(phone_candidates)
    if user_data:
        customer_ctx = _effective_customer_context(user_data)
        # Normalize MongoDB document
        print(f"[LIVE COPILOT]: {user_data=}") 

    else:
        print(f"[LIVE_COPILOT] ❌ No user found in MongoDB for phone candidates: {phone_candidates}", flush=True)
        print(f"[LIVE_COPILOT] ⚠️  contractType, plan, and state will NOT be set (MongoDB lookup failed)", flush=True)

    if speaker == "agent":
        # No AI suggestions needed for CSR text
        # But if user details were just fetched, send them for display
        if user_data:
            return Response(sessionId=session_id, userDetails=user_data).model_dump()
        return None
    

    output: Optional[Dict[str, Any]] = None
    tok = _live_session_id_var.set(session_id)
    try:
        # Live Copilot branch is a child of csr_copilot.session (session-level trace root).
        with tracer.start_as_current_span("live_call.processing", context=parent_context) as root:
            root.set_attribute("live.session_id", session_id)
            if _trace_include_payloads():
                root.set_attribute("live.transcript.preview", _preview(text))

            transcript = text#_buffer_text(st)
            important_change = False

            # ---------------- phase: intent_detection ----------------
            with tracer.start_as_current_span("live_copilot.intent_detection") as sp_intent:
                _set_session_attr(sp_intent)
                
                intent_obj: Dict[str, Any]
                intent_obj = _call_intent_llm(transcript=transcript, handler=handler, span=sp_intent)
                # If LLM didn't return phone but we have payload phone, attach for context_retrieval
                intent = _s(intent_obj.get("intent")) or "OTHER"
                confidence = float(intent_obj.get("confidence") or 0.0)
                evidence = _s(intent_obj.get("evidenceQuote")) or text[:200]
                entities = intent_obj.get("entities") or {}
                phone_entity = _s(entities.get("phone"))

            # ---------------- phase: context_retrieval ----------------
            with tracer.start_as_current_span("live_copilot.context_retrieval") as sp_ctx:
                _set_session_attr(sp_ctx)
                
                tool_result: Dict[str, Any] = {}
                customer = user_data
                customer_ctx["sessionId"] = session_id
                verified = bool(customer_ctx.get("verified"))

                should_extract = (_should_extract_questions(text) and intent not in ("CUSTOMER_IDENTIFICATION", "SMALL_TALK", "OTHER"))
                if should_extract:
                    extracted = _extract_questions_llm(transcript=transcript, handler=handler, span=sp_ctx,previous_questions=[x.question for x in questions], customer_ctx=customer_ctx)
                    for ques in extracted: questions.append(QA(question=ques))

                # Build tool_result snapshot (always present so the prompt has state + conversation context)
                # Include previousAnswers so LLM doesn't contradict itself

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
                    "pendingQuestions": [_strip_ctx(x.question) for x in questions if not x.answer],
                    "answeredCount": len([x.answer for x in questions if x.answer]),
                    "previousAnswers": [ x.answer for x in questions if x.answer],  # Include all previously answered questions for consistency
                    "newAnswers": [],
                    "verification": {
                        "needsPhone": False,
                        "askForPhone": False,
                    },
                }

                can_rag = bool(
                    customer_ctx.get("contractType") and customer_ctx.get("plan") and customer_ctx.get("state")
                )

            # ---------------- phase: rag_answer (where applicable) ----------------
            if can_rag and questions:
                with tracer.start_as_current_span("live_copilot.rag_answer") as sp_rag:
                    _set_session_attr(sp_rag)
                    
                    answers = []
                    answers = list(qna_executor.map(lambda q: _rag_answer(question=q, customer=customer_ctx, handler=handler, span=sp_rag), [_.question for _  in questions if not _.answer] ))
                    
                    for index, answer in enumerate(answers):
                        questions[index].answer = answer
                    answers = [x["answer"] for x in answers]
                    if answers:
                        tool_result["newAnswers"] = answers

            if intent == "PROBLEM":
                # Generate diagnostics steps
                with tracer.start_as_current_span("live_copilot.diagnostics") as sp_diag:
                    _set_session_attr(sp_diag)
                    tool_result["diagnostics"] = _diagnostics_steps(transcript=transcript, handler=handler, span=sp_diag)
                    print("[DIAGNOSTICS]: ", tool_result["diagnostics"])

            # ---------------- phase: suggestion_generation ----------------
            with tracer.start_as_current_span("live_copilot.suggestion_generation") as sp_llm:
                _set_session_attr(sp_llm)
                print(f"[LIVE_COPILOT] Calling suggest LLM with intent: {intent}, customer_verified: {verified}", flush=True)
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
            output = {
                "sessionId": session_id,
                "intent": intent or "OTHER",
                "userDetails": user_data.model_dump(),
                "confidence": confidence,
                "customer": customer_ctx,
                "createdAt": str(_now_epoch()),
                "cards": cards or []
            }
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



def handle_transcript_event(payload: CopilotSessionData):
    global copilotQueue
    copilotQueue.put(payload)











def handle_copilot_enable_event(session_id: str, phone_number: Optional[str] = None) -> Optional[Dict[str, Any]]:
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
        phone_candidates = (phone_clean, "+1" + phone_clean, phone_clean[1:], "+1" + phone_clean[1:])
            
        _log("info", "📞", f"Proactive lookup for session {session_id} with phone {phone_number}")
        doc = _lookup_user_by_phone(phone_candidates)
        if doc:
            st.customer = doc
            
            # Sync plan context
                
            return {
                "sessionId": session_id,
                "intent": "OTHER",
                "userDetails": st.customer.model_dump() if st.customer else "",
                "createdAt": str(_now_epoch()),
            }
            
    return None



