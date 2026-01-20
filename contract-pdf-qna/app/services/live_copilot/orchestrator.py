"""Main orchestrator for Live Copilot."""
from typing import Dict, Any, Optional
from time import time
from contextvars import ContextVar
from token_module import CallbackHandler
from utils.transcript_filters import is_trivial_utterance
from monitoring_module import llm_trace_to_jaeger

from .session_state import (
    get_state, cooldown_ok, append_buffer, buffer_text,
    update_session_context_from_payload, effective_customer_context,
    COPILOT_MAX_VERIFICATION_ASKS
)
from .customer_lookup import extract_phone_candidates, lookup_user_by_phone
from .intent_detection import call_intent_llm
from .question_extraction import should_extract_questions, extract_questions_llm, queue_questions
from .rag_handler import rag_answer
from .suggestion_generator import diagnostics_steps, call_suggest_llm_traced
from .infer_service import get_infer_wrapper
from .tracing import get_tracer, trace_include_payloads, preview, span_common, live_session_id
from .utils import s, fingerprint, now_epoch

_live_session_id_var: ContextVar[str] = ContextVar("live_session_id", default="")


def handle_transcript_event(payload: Dict[str, Any], parent_context=None) -> Optional[Dict[str, Any]]:
    """
    Main entry point for processing transcript events.
    
    Invoked by webhook route (in a SocketIO background task) ONLY when:
    - ENABLE_LIVE_COPILOT=1, AND
    - session is enabled by Analyze Live UI (copilot_enable), AND
    - transcript event arrives for that session.
    
    Returns:
        Dict with sessionId, intent, confidence, customer, cards, createdAt
        or None if no suggestion should be emitted
    """
    session_id = s(payload.get("sessionId"))
    speaker = s(payload.get("speaker")).lower()
    text = s(payload.get("text"))
    is_partial = bool(payload.get("isPartial", True))

    if not session_id or not text:
        return None
    if is_partial:
        return None

    # Update buffer and session context (always needed for conversation history)
    st = get_state(session_id)
    from .customer_lookup import normalize_customer_doc
    update_session_context_from_payload(st, payload, lookup_user_by_phone, normalize_customer_doc)
    append_buffer(st, speaker=speaker, text=text)

    # Skip AI processing for CSR text - only process customer prompts for suggestions
    if speaker == "agent":
        # No AI suggestions needed for CSR text
        return None
    
    if is_trivial_utterance(text):
        return None

    handler = CallbackHandler()
    output: Optional[Dict[str, Any]] = None
    tok = _live_session_id_var.set(session_id)
    tracer = get_tracer()
    
    try:
        # Live Copilot branch is a child of csr_copilot.session (session-level trace root).
        with tracer.start_as_current_span("live_call.processing", context=parent_context) as root:
            root.set_attribute("live.session_id", session_id)
            if trace_include_payloads():
                root.set_attribute("live.transcript.preview", preview(text))

            with tracer.start_as_current_span("orchestrator_agent") as orch:
                span_common(
                    orch,
                    agent_name="orchestrator-agent.live-infer",
                    agent_role="Coordinates intent detection, retrieval, and response generation",
                    from_agent="live_call.processing",
                )

                transcript = buffer_text(st)
                important_change = False

                # ---------------- phase: intent_detection ----------------
                with tracer.start_as_current_span("intent_detection") as sp_intent:
                    span_common(
                        sp_intent,
                        agent_name="atomic-agent.intent_detection",
                        agent_role="Intent classification + entity extraction",
                        from_agent="orchestrator-agent.live-infer",
                    )

                    phone_candidates = extract_phone_candidates(text)
                    intent_obj: Dict[str, Any]
                    if phone_candidates:
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
                        intent_obj = call_intent_llm(transcript=transcript, handler=handler, span=sp_intent)

                    intent = s(intent_obj.get("intent")) or "OTHER"
                    confidence = float(intent_obj.get("confidence") or 0.0)
                    evidence = s(intent_obj.get("evidenceQuote")) or text[:200]
                    entities = intent_obj.get("entities") or {}
                    phone_entity = s(entities.get("phone"))

                # ---------------- phase: context_retrieval ----------------
                with tracer.start_as_current_span("context_retrieval") as sp_ctx:
                    span_common(
                        sp_ctx,
                        agent_name="atomic-agent.context_retrieval",
                        agent_role="Load customer context, DB lookups, and question extraction",
                        from_agent="atomic-agent.intent_detection",
                    )

                    tool_result: Dict[str, Any] = {}
                    customer = st.customer

                    if (phone_candidates or phone_entity) and not customer:
                        candidates = phone_candidates or [phone_entity]
                        doc = lookup_user_by_phone([c for c in candidates if c])
                        if doc:
                            from .customer_lookup import normalize_customer_doc
                            st.customer = normalize_customer_doc(doc, candidates[0])
                            customer = st.customer
                            try:
                                st.contract_type = st.contract_type or s(customer.get("contractType"))
                                st.selected_plan = st.selected_plan or s(customer.get("plan"))
                                st.selected_state = st.selected_state or s(customer.get("state"))
                            except Exception:
                                pass
                            important_change = True

                    customer_ctx = effective_customer_context(st)
                    verified = bool(customer_ctx.get("verified"))

                    should_extract = (
                        speaker == "customer" 
                        and should_extract_questions(text) 
                        and intent not in ("CUSTOMER_IDENTIFICATION", "SMALL_TALK", "OTHER")
                    )
                    if should_extract:
                        extracted = extract_questions_llm(transcript=transcript, handler=handler, span=sp_ctx)
                        if not extracted:
                            q1 = s(entities.get("question"))
                            if q1:
                                extracted = [q1]
                        if extracted:
                            if queue_questions(st, extracted):
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
                        "pendingQuestions": [x.get("q") for x in st.pending_questions if s(x.get("q"))],
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
                        span_common(
                            sp_rag,
                            agent_name="atomic-agent.rag_answer",
                            agent_role="Answer queued questions via Infer pipeline (RAG)",
                            from_agent="atomic-agent.context_retrieval",
                        )
                        answered_now = []
                        for item in list(st.pending_questions)[:2]:
                            k = s(item.get("k"))
                            q = s(item.get("q"))
                            if not k or not q:
                                continue
                            if k in st.answered:
                                continue
                            res = rag_answer(question=q, customer=customer_ctx, handler=handler, span=sp_rag, get_infer_wrapper_fn=get_infer_wrapper)
                            st.answered[k] = {"ts": time(), **(res or {})}
                            answered_now.append({"question": q, "result": res})
                        if answered_now:
                            st.pending_questions = [
                                x for x in st.pending_questions if s(x.get("k")) not in st.answered
                            ]
                            tool_result["newAnswers"] = answered_now
                            important_change = True

                if intent == "PROBLEM":
                    # Keep within rag_answer when applicable (same operational bucket: "tools").
                    with tracer.start_as_current_span("rag_answer") as sp_rag:
                        span_common(
                            sp_rag,
                            agent_name="atomic-agent.rag_answer",
                            agent_role="Generate generic diagnostics steps (non-coverage)",
                            from_agent="atomic-agent.context_retrieval",
                        )
                        tool_result["diagnostics"] = diagnostics_steps(transcript=transcript, handler=handler, span=sp_rag)

                # ---------------- phase: llm_call ----------------
                with tracer.start_as_current_span("llm_call") as sp_llm:
                    span_common(
                        sp_llm,
                        agent_name="atomic-agent.llm_call",
                        agent_role="Generate CSR suggestion cards",
                        from_agent="atomic-agent.rag_answer" if can_rag else "atomic-agent.context_retrieval",
                    )

                    if not cooldown_ok(st) and not important_change:
                        cards = None
                    else:
                        cards = call_suggest_llm_traced(
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
                    span_common(
                        sp_post,
                        agent_name="atomic-agent.response_postprocessing",
                        agent_role="Dedupe and finalize response payload",
                        from_agent="atomic-agent.llm_call",
                    )

                    if cards is None:
                        output = None
                    else:
                        fp = fingerprint({"intent": intent, "customer": customer_ctx, "cards": cards})
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
                                "createdAt": str(now_epoch()),
                            }
                            if trace_include_payloads():
                                sp_post.set_attribute("live.response.preview", preview(output))

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
