"""RAG handling for Live Copilot."""
import json
from typing import Dict, Any
from langchain_core.output_parsers import StrOutputParser
from token_module import CallbackHandler
from utils.prompts import _rag_prompt
from .llm_cache import get_suggest_llm
from .milvus_manager import get_vector_db, milvus_collection
from .tracing import trace_include_payloads, preview
from app.config.settings import settings


def simple_rag_answer(*, question: str, customer: Dict[str, Any], handler: CallbackHandler, span) -> Dict[str, Any]:
    """
    Simple RAG implementation - fallback when INFER wrapper is not available.
    Uses direct Milvus similarity search + LLM summarization.
    """
    if not settings.MILVUS_HOST:
        return {"error": "MILVUS_HOST not configured"}
    collection = milvus_collection(customer.get("contractType"), customer.get("plan"), customer.get("state"))
    if not collection:
        return {"error": "Missing plan context for Milvus collection"}
    vector_db = get_vector_db(collection)
    # Similarity search then summarize with LLM
    docs = vector_db.similarity_search(question, k=6)
    chunks = []
    for d in docs:
        content = getattr(d, "page_content", "") or ""
        if content.strip():
            chunks.append(content.strip())
    if not chunks:
        return {"answer": "I couldn't find relevant policy language for that question.", "citedChunks": []}
    llm = get_suggest_llm()  # Use cached instance
    chain = _rag_prompt | llm | StrOutputParser()
    payload = {"question": question, "chunks": "\n\n".join(chunks)}
    raw = (chain.invoke(payload, config={"callbacks": [handler]}) or "").strip()
    if trace_include_payloads():
        span.set_attribute("llm.prompt.preview", preview(payload))
        span.set_attribute("llm.response.preview", preview(raw))
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


def rag_answer(*, question: str, customer: Dict[str, Any], handler: CallbackHandler, span, get_infer_wrapper_fn) -> Dict[str, Any]:
    """
    Main RAG function - uses INFER wrapper if available, otherwise falls back to simple RAG.
    
    The INFER wrapper uses the full LangChain Agent with:
    - Knowledge Base tool (RetrievalQA)
    - User Lookup tool
    - Sophisticated system prompt for query breakdown
    
    Args:
        question: The customer question to answer
        customer: Dict with contractType, plan, state, etc.
        handler: Callback handler for token tracking
        span: Tracing span
        get_infer_wrapper_fn: Function to get INFER wrapper
        
    Returns:
        Dict with keys: answer, citedChunks/relevantChunks, and optionally error
    """
    contract_type = customer.get("contractType", "")
    plan = customer.get("plan", "")
    state = customer.get("state", "")
    
    # Try to use INFER wrapper first (full LangChain Agent)
    infer_wrapper = get_infer_wrapper_fn()
    
    if infer_wrapper is not None:
        try:
            result = infer_wrapper(
                question=question,
                contract_type=contract_type,
                selected_plan=plan,
                selected_state=state,
                transcript_context="",  # Could add more context here if needed
                handler=handler,  # Pass handler directly
            )
            
            # Transform result to match expected format
            answer = (result or {}).get("answer", "")
            chunks = (result or {}).get("relevantChunks", [])
            
            if (result or {}).get("error"):
                # Fall through to simple RAG
                pass
            elif answer:
                if trace_include_payloads():
                    span.set_attribute("rag.question.preview", preview(question))
                    span.set_attribute("rag.answer.preview", preview(answer))
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
    result = simple_rag_answer(question=question, customer=customer, handler=handler, span=span)
    result["source"] = "simple_rag"
    return result
