"""Claims endpoints."""
from flask import Blueprint, request, jsonify
from bson.objectid import ObjectId
from datetime import datetime
import uuid
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.utils.auth import token_process
from app.services.claims_service import ClaimsService
from app.models.database import Database

claims_bp = Blueprint('claims', __name__)


def init_claims_routes(claims_service: ClaimsService, database: Database):
    """Initialize claims routes with service dependencies.
    
    Args:
        claims_service: ClaimsService instance
        database: Database service instance
    """
    @claims_bp.route("/claims/followup", methods=["POST"])
    def claims_followup_chat():
        """Claims follow-up chat - full implementation."""
        from monitoring_module import tracer
        
        with tracer.start_as_current_span("api/claims/followup"):
            authorization_header = request.headers.get("Authorization")
            if authorization_header is None:
                return jsonify({"message": "Token is missing"}), 401
            
            if authorization_header:
                token_data = token_process(authorization_header)
                if token_data[1] == 401 or token_data[1] == 403:
                    return (token_data[0].get_json()), token_data[1]
            
            conversation_id = request.args.get("conversation-id")
            if not conversation_id:
                return jsonify({"error": "conversation-id is required"}), 400
            
            data = request.get_json() or {}
            entered_query = (data.get("enteredQuery") or "").strip()
            if not entered_query:
                return jsonify({"error": "enteredQuery is required"}), 400
            
            user_email = token_data[0]["email"]
            qna_collection_user = f"chats_{user_email}"
            qna_collection = database.db[qna_collection_user]
            
            docs = qna_collection.find_one({"_id": ObjectId(conversation_id)}) or {}
            if not docs:
                return jsonify({"error": "Conversation not found"}), 404
            
            if docs.get("doc_type") != "transcript_conversation":
                return jsonify({"error": "claims/followup is only supported for transcript conversations"}), 400
            
            if (docs.get("status") or "").lower() == "inactive":
                return jsonify({"error": "Case is closed. Chat is disabled."}), 403
            
            # Handle contract/plan/state overrides
            try:
                override_contract = (data.get("contractType") or "").strip()
                override_plan = (data.get("selectedPlan") or "").strip()
                override_state = (data.get("selectedState") or "").strip()
                if override_contract or override_plan or override_state:
                    updates = {}
                    if override_contract:
                        updates["contract_type"] = override_contract
                        docs["contract_type"] = override_contract
                    if override_plan:
                        updates["selected_plan"] = override_plan
                        docs["selected_plan"] = override_plan
                    if override_state:
                        updates["selected_state"] = override_state
                        docs["selected_state"] = override_state
                    if updates:
                        updates["updated_at"] = datetime.utcnow()
                        qna_collection.update_one({"_id": ObjectId(conversation_id)}, {"$set": updates})
            except Exception:
                pass
            
            # Build and cache plan overview if needed
            try:
                if claims_service.looks_like_plan_overview_question(entered_query):
                    overview = claims_service.get_or_build_plan_overview_for_claims(docs)
                    if overview:
                        qna_collection.update_one(
                            {"_id": ObjectId(conversation_id)},
                            {"$set": {"plan_overview": overview, "updated_at": datetime.utcnow()}},
                        )
                        docs["plan_overview"] = overview
            except Exception:
                pass
            
            # Build case context and retrieve chunks
            case_context = claims_service.build_claims_case_context_for_llm(docs)
            if not case_context:
                return jsonify({"error": "Missing case context for this conversation"}), 400
            
            policy_chunks, referred_docs_text = claims_service.retrieve_policy_chunks_for_claims(
                docs, entered_query, k=6
            )
            
            # Build policy section
            policy_section = ""
            if policy_chunks:
                lines = ["RETRIEVED POLICY CLAUSES (Vector DB)"]
                for i, ch in enumerate(policy_chunks[:12], start=1):
                    if not isinstance(ch, dict):
                        continue
                    content = str(ch.get("content") or "").strip()
                    if not content:
                        continue
                    chunk_name = ch.get("name") or f"Clause {i}"
                    lines.append(f"- {chunk_name}: {content[:100]}...")
                policy_section = "\n".join(lines).strip()
            
            # Build prompt
            prompt = (
                "You are an insurance claims copilot.\n"
                "Answer the user's question using BOTH the CASE CONTEXT and (when provided) the RETRIEVED POLICY CLAUSES.\n"
                "If the user asks:\n"
                "- what the claim is about: summarize using FINAL ANALYZED ANSWER.\n"
                "- what customer queries were: list/explain from EXTRACTED CUSTOMER QUERIES.\n"
                "- a repeat question: answer consistently, using the context and prior follow-up chat.\n"
                "For policy/coverage questions, use the RETRIEVED POLICY CLAUSES when relevant.\n"
                "If the answer is not in CASE CONTEXT or the RETRIEVED POLICY CLAUSES, say you don't have that information.\n"
                "Do NOT use any external policy lookup beyond the provided clauses.\n"
                "\n"
                f"{case_context}\n"
                "\n"
                f"{policy_section}\n"
                "\n"
                f"USER QUESTION: {entered_query}\n"
                "ANSWER:"
            )
            
            # Generate answer
            llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")
            ai_text = ""
            try:
                ai_text = str(llm.invoke([HumanMessage(content=prompt)]).content or "").strip()
            except Exception as e:
                return jsonify({"error": f"LLM error: {e}"}), 500
            
            # Save chat
            chat_id = str(uuid.uuid4())
            now_ts = datetime.utcnow()
            underlying = docs.get("underlying_model") or "Search"
            try:
                qna_collection.update_one(
                    {"_id": ObjectId(conversation_id)},
                    {
                        "$push": {
                            "chats": {
                                "chat_id": chat_id,
                                "entered_query": entered_query,
                                "response": ai_text,
                                "relevant_chunks": policy_chunks or [],
                                "relevant_docs": referred_docs_text or "",
                                "gpt_model": "Calls",
                                "underlying_model": underlying,
                                "chat_timestamp": now_ts,
                                "latency": 0.0,
                                "confidence": 0.0,
                            }
                        },
                        "$set": {"updated_at": now_ts},
                    },
                )
            except Exception:
                pass
            
            return jsonify({"aiResponse": ai_text, "conversationId": str(conversation_id), "chatId": chat_id}), 200
    
    return claims_bp
