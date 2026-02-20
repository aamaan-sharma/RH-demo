import json
import traceback
from time import time

from utils.prompts import ANSWERING_PROMPT_SEARCH, _agent_system_message
from config import MOTORHEAD_CLIENT_ID, MOTORHEAD_API_KEY
from utils.constants import MILVUS_FALLBACK_K, MILVUS_MAX_RETURN_CHUNKS, MILVUS_RETRIEVER_K

from pymilvus import Milvus


from typing import Dict
from functools import cache
from langchain.tools import tool
import asyncio
from langchain_community.memory.motorhead_memory import MotorheadMemory
from dataclasses import dataclass, field

from langchain.callbacks.base import BaseCallbackHandler

from langchain.chains import RetrievalQA

from langchain.agents import initialize_agent, Tool, AgentType, create_tool_calling_agent, AgentExecutor
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate
from utils.kb import get_vector_db, getPolicyid, getRetriver

from core.db import get_user_details_from_mobile, User
from core.llms import TRANSCRIPT_QA_AGENT, SEARCH_LLM
from enum import Enum
from typing import Optional, List

from token_module import CallbackHandler
from monitoring_module import q_monitor, tracer, llm_trace_to_jaeger, func_Binsert, security_scores, _is_answer_fallback
@dataclass
class Response:
    error: Optional[str] = None
    answer: str =  ""
    confidence: float = 0.0
    latency: float =  0.0
    relevantChunksDetails: list[str] = field(default_factory=list)
    relevantChunks: list[str] = field(default_factory=list)


@cache
async def get_memory_instance(*, session_id):
    memory = MotorheadMemory(
        api_key=MOTORHEAD_API_KEY,
        client_id=MOTORHEAD_CLIENT_ID,
        session_id=session_id,
        memory_key="chat_history",
        return_messages=True,
        input_key="input",
        output_key="output",
    )

    await memory.init()
    return memory



class InferenceMode(Enum):
    SEARCH = "Search"
    INFER = "Infer"



class DocCaptureHandler(BaseCallbackHandler):
    def __init__(self):
        self.docs = []

    def on_retriever_end(self, documents, **kwargs):
        self.docs.extend(documents)







@tool
def fetch_user_by_mobile(mobile_number: str) -> str:
    """
    Fetch user details from the database based on mobile number.


    Useful for fetching user details from the database based on mobile
    number. Use this tool when you need to retrieve customer
    information, user profile, or any user-related data. Input should
    be the mobile number as a string. Returns user details in JSON
    format if found, or an error message if not found.
    
    Args:
        mobile_number: The mobile number to search for
        
    Returns:
        A string containing user details in JSON format, or an error message
    """
    print(f"[TOOL CALL][FETCH USER TOOL]: {mobile_number=}")
    try:
        user = get_user_details_from_mobile(mobile_number) 
        if user :
            return json.dumps(user.__dict__, indent=2, default=str)
        else:
            return f"No user found with mobile number: {mobile_number}"
    except Exception as e:
        return f"Error fetching user details: {str(e)}"


@tool
def knowledge_base_tool(query: str, policyId: str) -> str:
    '''
    tool for extracting relvant document given the query and policyId

    Useful for answering questions related to insurance coverage of
    home appliances, home fixtures, their repairs/replacement, service
    requests, about the renewal, cancellation or refund policies,
    whether a certain service is covered under the contract, permit
    limit, code violation limit, modification limit, limitations and
    exclusions.

    args:
        query: str 
        policyId: str
    
    return:
        Relevant Document Lists

    '''

    print(f"[TOOL CALL][KNOWLEDGE TOOL]: {query=}, {policyId=}")
    docs = getRetriver(policyId.strip()).invoke(query)
    print(f"[TOOL CALL RESULT][KNOWLEDGE TOOL] {len(docs)=}")
    return "\n\n".join([doc.page_content for doc in docs])
    



@cache
def get_agent_instance(policyId, sessionId):
    retriever = getRetriver(policyId)
    tools = [knowledge_base_tool, fetch_user_by_mobile]

    #memory = asyncio.run(get_memory_instance(session_id=current_time))
    sys_prompt = ChatPromptTemplate.from_messages([
        ("system", _agent_system_message),
        ("human", "{input}"),
        ("user", "use this policyId: {policyId}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    agent = create_tool_calling_agent(TRANSCRIPT_QA_AGENT, tools, sys_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)


    return agent_executor




def input_prompt(entered_query, policyId, handler, sessionId):
    # Retriever chain as Tool for agent
    docHandler = DocCaptureHandler()
    agent_executor = get_agent_instance(policyId, sessionId)
    response = agent_executor.invoke({"input": entered_query, "policyId": policyId},callbacks=[handler, docHandler])
    docs = docHandler.docs
    return response, docs


def handle_request_search(question, transcript_context, policyId: str, sessionId: Optional[str] = None, *, handler: CallbackHandler) -> tuple:

    enriched_query = (
        f"{question}\n\nTranscript situation/evidence:\n{transcript_context}".strip()
        if (transcript_context or "").strip()
        else question
    )
    qa_search = RetrievalQA.from_chain_type(
        llm=SEARCH_LLM,
        retriever=getRetriver(policyId),
        verbose=True,
        return_source_documents=True,
        chain_type_kwargs = {"prompt": ANSWERING_PROMPT_SEARCH}
    )
    
    # print("[CHUNKS] process_single_transcript_question: calling QA chain (Search)")
    qa_search_response = qa_search.invoke(
        {"query": enriched_query},
        config={"callbacks": [handler]},
    )
    answer_search = qa_search_response["result"] if isinstance(qa_search_response, dict) else qa_search_response
    print(
        "[CHUNKS] process_single_transcript_question: QA chain completed "
        f"answer_len={len(str(answer_search))}"
    )

    relevant_documents = str(qa_search_response["source_documents"])

    return answer_search, relevant_documents


def handle_request_infer(question: str = "", transcript_context: str = "", policyId: str = "",sessionId: Optional[str]=None, *, handler: CallbackHandler) -> tuple:

    enriched_query = (
        f"{question}\n\nTranscript situation/evidence:\n{transcript_context}".strip()
        if (transcript_context or "").strip()
        else question
    )

    agent_response, docs = input_prompt(enriched_query, policyId,handler, sessionId)
    answer = agent_response["output"]
    print(
        "[CHUNKS] process_single_transcript_question: agent_response received "
        f"answer_len={len(str(answer))}"
    )
    return answer, docs 

def process_single_transcript_question(
    question: str,
    policyId: str,
    inferenceMode: InferenceMode,
    handler: CallbackHandler,
    transcript_context: str = "",
    sessionId: str = "",
) -> Response:
    """
    Process a single question from transcript and return answer with chunks
    Reuses logic from /start endpoint but without conversation context
    """
    try:
        q_start_time = time()
        # No conversation context for transcript questions, but we CAN pass the transcript-derived
        # situation/evidence as part of the query to improve retrieval + answer relevance.
        # Keep the user-visible question unchanged elsewhere; only enrich the internal query.
        print(
            "[CHUNKS] process_single_transcript_question: START "
            f"question='{str(question)[:200]}', "
            f"{policyId}"
        )

        match inferenceMode:
            case InferenceMode.SEARCH: 
                answer, docs = handle_request_search(question, transcript_context, policyId, sessionId, handler=handler)
            case InferenceMode.INFER : 
                answer, docs = handle_request_infer(question, transcript_context, policyId, sessionId, handler=handler)

        
        q_latency = time() - q_start_time
        
        # Build relevantChunks from Milvus docs (always list[str] in the API response)
        # This ensures frontend receives text chunks (not placeholder "[]" / not dict objects).
        chunk_texts = []
        chunk_details = []
        try:

            docs_iter = docs
            if MILVUS_MAX_RETURN_CHUNKS is not None:
                docs_iter = docs[:MILVUS_MAX_RETURN_CHUNKS]

            for doc in docs_iter:
                content = (getattr(doc, "page_content", "") or "").strip()
                metadata = getattr(doc, "metadata", {}) or {}
                if not content:
                    continue
                chunk_texts.append(content)
                chunk_details.append({"content": content, "metadata": metadata})
        except Exception as e:
            print(f"[CHUNKS] process_single_transcript_question: ERROR building chunks: {e}")

        if not chunk_texts:
            # As a last resort, still return a non-empty list (but keep it explicit for debugging).
            # This should be rare; most Milvus collections should return at least some results.
            chunk_texts = ["(No supporting excerpts found)"]
        

        returned_chunks =  chunk_texts[:MILVUS_MAX_RETURN_CHUNKS] if MILVUS_MAX_RETURN_CHUNKS else chunk_texts

        
        relvant_chunk_details = chunk_details[:MILVUS_MAX_RETURN_CHUNKS] if MILVUS_MAX_RETURN_CHUNKS is not None else chunk_details
        return Response(answer=answer, relevantChunks=returned_chunks, relevantChunksDetails=relvant_chunk_details, confidence=0.90,latency=q_latency)
    except Exception as e:
        print(f"Error processing transcript question: {e}")
        traceback.print_exc()
        return Response(error=str(e),answer="Error processing question")



def process_live_copilot_question(
    question: str,
    policyId: str,
    transcript_context: str = "",
    sessionId: str = "",
    *,handler: CallbackHandler) -> Dict:
    """
    Wrapper for Live Copilot to use the existing INFER implementation.
    
    This function initializes Milvus, LLMs, and retriever, then calls
    process_single_transcript_question with gpt_model="Infer" to leverage
    the full LangChain Agent with Knowledge Base and User Lookup tools.
    
    Args:
        question: The customer question to answer
        contract_type: Contract type (RE or DTC)
        selected_plan: Plan name (ShieldPlus, ShieldGold, etc.)
        selected_state: State name (California, Texas, etc.)
        transcript_context: Optional transcript context for enrichment
        
    Returns:
        Dict with keys: answer, relevantChunks, confidence, latency
    """
    with tracer.start_as_current_span("live_copilot.process_question") as span:
        try:
            span.set_attribute("live_copilot.question.preview", question[:200] if question else "")
            span.set_attribute("live_copilot.policyId", policyId)
            
            print(f"[LIVE_COPILOT_INFER] Processing question='{question[:100]}...', {policyId=}")
            
            # Get collection name using utility function
            
            if not policyId:
                # Get normalized values for error logging
                print(f"[LIVE_COPILOT_INFER] Could not determine policy Id name for={policyId=}")
                span.set_attribute("live_copilot.error", "policyId_not_found")
                return Response(answer="Unable to determine the appropriate knowledge base for your query.").__dict__
            
            
            
            result = process_single_transcript_question(
                question=question,
                policyId=policyId,
                inferenceMode=InferenceMode.INFER,
                handler=handler,
                transcript_context=transcript_context,
            )
            
            print(f"[LIVE_COPILOT_INFER] Result: answer_len={len(result.answer)}, chunks={len(result.relevantChunks)}")
            print(f"{result.answer=}")
            
            if result:
                span.set_attribute("live_copilot.answer_length", len(result.answer))
                span.set_attribute("live_copilot.chunks_count", len(result.relevantChunks))
                span.set_attribute("live_copilot.confidence", result.confidence)
            
            return result.__dict__
            
        except Exception as e:
            print(f"[LIVE_COPILOT_INFER] Error: {e}")
            traceback.print_exc()
            span.set_attribute("live_copilot.error", str(e)[:200])
            return Response(answer=f"Error processing question: {str(e)}").__dict__
