"""Transcript processing service for complex transcript operations."""
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from time import time
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Milvus
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks import StdOutCallbackHandler
from app.services.rag_service import RAGService
from app.services.claims_service import ClaimsService
from app.services.transcript_service import TranscriptService
from app.services.agent_service import AgentService
from app.utils.milvus_utils import (
    normalize_state_for_milvus,
    normalize_contract_type,
    normalize_plan_for_milvus
)
from app.utils.chunks import normalize_chunks_with_names, generate_chunk_name, get_placeholder_chunk_values
from app.config.settings import settings


class TranscriptProcessorService:
    """Service for processing transcripts and generating answers."""
    
    def __init__(
        self,
        rag_service: RAGService,
        claims_service: ClaimsService,
        transcript_service: TranscriptService,
        agent_service: AgentService
    ):
        """Initialize transcript processor service."""
        self.rag_service = rag_service
        self.claims_service = claims_service
        self.transcript_service = transcript_service
        self.agent_service = agent_service
        self._PLACEHOLDER_CHUNK_VALUES = get_placeholder_chunk_values()
    
    def process_single_transcript_question(
        self,
        question: str,
        contract_type: str,
        selected_plan: str,
        selected_state: str,
        gpt_model: str,
        vector_db: Milvus,
        llm: ChatOpenAI,
        llm2: ChatOpenAI,
        retriever,
        handler: Optional[Any],
        transcript_context: str = "",
    ) -> Dict[str, Any]:
        """Process a single question from transcript and return answer with chunks."""
        try:
            q_start_time = time()
            enriched_query = (
                f"{question}\n\nTranscript situation/evidence:\n{transcript_context}".strip()
                if (transcript_context or "").strip()
                else question
            )
            
            print(
                "[CHUNKS] process_single_transcript_question: START "
                f"question='{str(question)[:200]}', "
                f"contract_type={contract_type}, selected_plan={selected_plan}, "
                f"selected_state={selected_state}, gpt_model={gpt_model}"
            )
            
            if gpt_model == "Search":
                prompt_template = """
You are a professional insurance claims representative (CSR). Be empathetic, firm, and to the point.

Use ONLY the policy/contract context provided below. Do NOT speculate or invent facts.

Claims can be informational or coverage-related. Act accordingly:
- If the question is about process/timeline/documents/next steps/costs, answer informatively with clear steps.
- If the question is about coverage/limits/exclusions/eligibility, use a universal decision posture and give a clear determination using available facts from the context.

Avoid "if/then" branching answers.

Universal decision posture model (choose ONE):
- ACCEPT_AND_PAY: trigger met, no applicable exclusions, conditions met, scope/valuation supported
- ACCEPT_PARTIAL: some components covered, others excluded/limited; apply deductible/sublimits/depreciation if stated
- DENY: trigger not met, exclusion applies, policy not in force/eligible (if stated)
- REQUEST_INFO: insufficient evidence; request specific items needed and why
- RESERVE_RIGHTS: potential coverage issues; continue investigation while reserving rights (only if the context supports this posture)

If required information is missing to make a determination:
1) State what CAN be concluded from known facts,
2) State what CANNOT be concluded,
3) State exactly what you need next (documents/details) and the next step.

Make the answer accountable so follow-up WH-questions are answerable:
- Include a short "Why" line grounded in the provided context.
- Include a short "Policy basis" line quoting 1–2 exact clause snippets from the provided context.
- Include "Next step" if anything is missing or a process step is required.
Keep these accountability lines short.

Policy/contract context (verbatim):
{context}

Customer question:
{question}

Answer format:
- Answer: (2–6 sentences, decisive, no hypotheticals)
- Why: (1 short sentence)
- Policy basis: (quote 1–2 short clause snippets)
- Next step: (if applicable; otherwise say "No further action needed.")
"""
                
                PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
                chain_type_kwargs = {"prompt": PROMPT}
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever,
                    verbose=True,
                    chain_type_kwargs=chain_type_kwargs
                )
                
                qa_response = qa.invoke(
                    {"query": enriched_query},
                    config={"callbacks": [handler] if handler else []},
                )
                answer = qa_response["result"] if isinstance(qa_response, dict) else qa_response
                print(
                    "[CHUNKS] process_single_transcript_question: QA chain completed "
                    f"answer_len={len(str(answer))}"
                )
                
                relevant_documents = self.rag_service.get_relevant_documents_string(enriched_query, retriever)
                
            elif gpt_model == "Infer":
                qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, verbose=True)
                agent_response = self.agent_service.input_prompt(enriched_query, qa, llm, handler)
                answer = agent_response["output"]
                print(
                    "[CHUNKS] process_single_transcript_question: agent_response received "
                    f"answer_len={len(str(answer))}"
                )
                knowledge_base_thoughts = [
                    item[0].tool_input for item in agent_response["intermediate_steps"] 
                    if item[0].tool == 'Knowledge Base'
                ]
                relevant_documents = ""
                for action_input in knowledge_base_thoughts:
                    print(
                        "[CHUNKS] process_single_transcript_question: calling relevant_docs (Infer) "
                        f"for tool_input='{str(action_input)[:200]}'"
                    )
                    rd = self.rag_service.get_relevant_documents_string(action_input, retriever)
                    print(
                        "[CHUNKS] process_single_transcript_question: returned from relevant_docs (Infer) "
                        f"len={len(rd)}"
                    )
                    relevant_documents += rd
            else:
                return {
                    "error": f"Invalid gpt_model: {gpt_model}",
                    "answer": "",
                    "relevantChunks": [],
                    "confidence": 0.0,
                    "latency": 0.0
                }
            
            q_latency = time() - q_start_time
            
            # Build relevantChunks from Milvus docs
            chunk_texts = []
            chunk_details = []
            try:
                # First attempt: retriever (normal path)
                docs_for_chunks = retriever.get_relevant_documents(enriched_query)
                if not docs_for_chunks:
                    # Fallbacks to ensure we still fetch something from Milvus
                    fallback_queries = [
                        f"{enriched_query} {contract_type} {selected_plan} {selected_state}",
                        f"{contract_type} {selected_plan} contract coverage",
                        "contract coverage",
                    ]
                    for fq in fallback_queries:
                        try:
                            docs_for_chunks = vector_db.similarity_search(fq, k=settings.MILVUS_FALLBACK_K)
                            if docs_for_chunks:
                                break
                        except Exception as e:
                            print(f"[CHUNKS] process_single_transcript_question: fallback similarity_search failed: {e}")
                            continue
                
                docs_for_chunks = docs_for_chunks or []
                print(
                    "[CHUNKS] process_single_transcript_question: docs_for_chunks_count="
                    f"{len(docs_for_chunks)}"
                )
                
                docs_iter = docs_for_chunks
                if settings.MILVUS_MAX_RETURN_CHUNKS is not None:
                    docs_iter = docs_for_chunks[:settings.MILVUS_MAX_RETURN_CHUNKS]
                
                for doc in docs_iter:
                    content = (getattr(doc, "page_content", "") or "").strip()
                    metadata = getattr(doc, "metadata", {}) or {}
                    
                    if not content:
                        continue
                    
                    chunk_name = generate_chunk_name(metadata, len(chunk_texts) + 1)
                    chunk_texts.append(content)
                    chunk_details.append({
                        "content": content,
                        "metadata": metadata,
                        "name": chunk_name
                    })
            except Exception as e:
                print(f"[CHUNKS] process_single_transcript_question: ERROR building chunks: {e}")
            
            if not chunk_texts:
                chunk_texts = ["(No supporting excerpts found)"]
            
            returned_chunks = chunk_texts
            returned_chunk_details = chunk_details
            if settings.MILVUS_MAX_RETURN_CHUNKS is not None:
                returned_chunks = chunk_texts[:settings.MILVUS_MAX_RETURN_CHUNKS]
                returned_chunk_details = chunk_details[:settings.MILVUS_MAX_RETURN_CHUNKS]
            
            # Return chunks as objects with names
            chunks_with_names = []
            for detail in returned_chunk_details:
                if isinstance(detail, dict):
                    chunk_obj = {
                        "content": detail.get("content", ""),
                        "metadata": detail.get("metadata", {}),
                        "name": detail.get("name", f"Clause {len(chunks_with_names) + 1}")
                    }
                    chunks_with_names.append(chunk_obj)
                else:
                    chunks_with_names.append({
                        "content": str(detail) if not isinstance(detail, dict) else detail.get("content", ""),
                        "name": f"Clause {len(chunks_with_names) + 1}"
                    })
            
            return {
                "answer": answer,
                "relevantChunks": chunks_with_names,
                "relevantChunksStrings": returned_chunks,
                "relevantChunksDetail": returned_chunk_details,
                "confidence": 0.90,
                "latency": q_latency
            }
        except Exception as e:
            print(f"Error processing transcript question: {e}")
            import traceback
            traceback.print_exc()
            return {
                "error": str(e),
                "answer": "Error processing question",
                "relevantChunks": [],
                "confidence": 0.0,
                "latency": 0.0
            }
    
    def process_live_copilot_question(
        self,
        question: str,
        contract_type: str,
        selected_plan: str,
        selected_state: str,
        transcript_context: str = "",
        handler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Wrapper for Live Copilot to use the INFER implementation.
        This method initializes all required components and calls process_single_transcript_question.
        
        Args:
            question: The customer question to answer
            contract_type: Contract type (RE or DTC)
            selected_plan: Plan name (ShieldPlus, ShieldGold, etc.)
            selected_state: State name (California, Texas, etc.)
            transcript_context: Optional transcript context for enrichment
            handler: Optional callback handler for token tracking
            
        Returns:
            Dict with keys: answer, relevantChunks, confidence, latency
        """
        from app.services.llm_factory import get_llm_factory
        
        try:
            print(
                f"[LIVE_COPILOT_INFER] Processing question='{question[:100]}...', "
                f"contract_type={contract_type}, plan={selected_plan}, state={selected_state}"
            )
            
            # Get collection name
            collection_name = self.rag_service.get_collection_for_context(
                contract_type, selected_plan, selected_state
            )
            
            if not collection_name:
                print(f"[LIVE_COPILOT_INFER] Could not determine collection name for contract_type={contract_type}, plan={selected_plan}")
                return {
                    "answer": "Unable to determine the appropriate knowledge base for your query.",
                    "relevantChunks": [],
                    "confidence": 0.0,
                    "latency": 0.0,
                }
            
            print(f"[LIVE_COPILOT_INFER] Using Milvus collection: {collection_name}")
            
            # Initialize vector DB and retriever
            vector_db = self.rag_service.get_vector_db(collection_name)
            retriever = self.rag_service.create_retriever(collection_name)
            
            # Initialize LLMs
            llm_factory = get_llm_factory()
            llm = llm_factory.create_chat_llm(model=settings.MODEL_SUGGEST)
            llm2 = llm_factory.create_chat_llm(model=settings.MODEL_INTENT)
            
            # Call the existing process method with Infer mode
            result = self.process_single_transcript_question(
                question=question,
                contract_type=contract_type,
                selected_plan=selected_plan,
                selected_state=selected_state,
                gpt_model="Infer",  # Use INFER mode with LangChain Agent
                vector_db=vector_db,
                llm=llm,
                llm2=llm2,
                retriever=retriever,
                handler=handler,
                transcript_context=transcript_context,
            )
            
            print(f"[LIVE_COPILOT_INFER] Result: answer_len={len(result.get('answer', ''))}, chunks={len(result.get('relevantChunks', []))}")
            
            return result
            
        except Exception as e:
            print(f"[LIVE_COPILOT_INFER] Error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"Error processing question: {str(e)}",
                "relevantChunks": [],
                "confidence": 0.0,
                "latency": 0.0,
            }
