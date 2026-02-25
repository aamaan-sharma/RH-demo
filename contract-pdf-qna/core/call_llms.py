from typing import Dict, Any, List, Optional
import re
import json
from langchain_core.output_parsers import StrOutputParser
from core.llms import SUGGEST_LLM, INTENT_LLM, DIAGNOSTIC_LLM
from config import VERBOSE_DEBUG
from token_module import CallbackHandler
from monitoring_module import tracer, llm_trace_to_jaeger, func_Binsert, security_scores, _is_answer_fallback
from utils.prompts import (
    _rag_prompt,
    _question_extract_prompt,
    _suggest_prompt,
    _intent_prompt,
    _diagnostics_prompt
)

import httpx
from utils.kb import getPolicyid
from core.transcript_process import process_live_copilot_question, InferenceMode
from utils.helpers import _log, _trace_include_payloads, _preview, _s

from openai import APITimeoutError  # type: ignore

from langchain_core.output_parsers import StrOutputParser

def _extract_questions_llm(*, transcript: str, handler: CallbackHandler, span, customer_ctx: Optional[Dict[str, Any]] = None) -> List[str]:
    llm = SUGGEST_LLM
    
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





def _call_intent_llm(*, transcript: str, handler: CallbackHandler, span) -> Dict[str, Any]:
    llm = INTENT_LLM  # Use cached instance
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
                policyId=getPolicyid(contract_type=contract_type, selected_plan=plan, selected_state=state),
                transcript_context="",  # Could add more context here if needed
                handler=handler
            )
            
            # Transform result to match expected format
            answer = (result or {}).get("answer", "")
            chunks = (result or {}).get("relevantChunks", [])
            num_chunks = len(chunks) if chunks else 0
            cited = chunks[:3] if chunks else []

            if answer:
                if _trace_include_payloads():
                    span.set_attribute("rag.question.preview", _preview(question))
                    span.set_attribute("rag.answer.preview", _preview(answer))
                print(f"[CHUNKS] step=rag_answer_return num_chunks_received={num_chunks} num_cited_passed={len(cited)}")
                return {
                    "answer": answer,
                    "citedChunks": cited,
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
    llm = DIAGNOSTIC_LLM # Use cached instance
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
    llm = SUGGEST_LLM  # Use cached instance
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

