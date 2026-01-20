"""Transcript endpoints."""
import json
from flask import Blueprint, request, jsonify, Response, stream_with_context
from datetime import datetime
from bson.objectid import ObjectId
from pymongo import ReturnDocument
from app.utils.auth import token_process
from app.services.transcript_service import TranscriptService
from app.services.transcript_processor_service import TranscriptProcessorService
from app.utils.gcp_storage import GCPStorageService
from app.utils.sse import format_sse
from app.utils.chunks import normalize_chunks_with_names, get_placeholder_chunk_values
from app.models.database import Database
from app.config.settings import settings

transcripts_bp = Blueprint('transcripts', __name__)


def init_transcripts_routes(
    transcript_service: TranscriptService,
    gcp_storage_service: GCPStorageService,
    database: Database,
    transcript_processor_service: TranscriptProcessorService
):
    """Initialize transcript routes with service dependencies.
    
    Args:
        transcript_service: TranscriptService instance
        gcp_storage_service: GCPStorageService instance
        database: Database service instance
    """
    @transcripts_bp.route("/transcripts", methods=["GET"])
    def list_transcripts():
        from monitoring_module import tracer
        
        with tracer.start_as_current_span('api/transcripts'):
            authorization_header = request.headers.get("Authorization")
            
            if authorization_header is None:
                return jsonify({"message": "Token is missing"}), 401
            
            if authorization_header:
                token_data = token_process(authorization_header)
                
                if token_data[1] == 401 or token_data[1] == 403:
                    return (token_data[0].get_json()), token_data[1]
            
            limit = request.args.get("limit", type=int)
            offset = request.args.get("offset", type=int, default=0)
            search = request.args.get("search") or request.args.get("q")
            
            transcripts, total_count = gcp_storage_service.list_transcript_files(
                limit=limit, offset=offset, search=search
            )
            
            return jsonify({
                "transcripts": transcripts,
                "pagination": {
                    "limit": limit or total_count,
                    "offset": offset,
                    "total": total_count,
                }
            })
    
    @transcripts_bp.route("/transcripts/<filename>", methods=["GET"])
    def get_transcript_content(filename: str):
        authorization_header = request.headers.get("Authorization")
        
        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401
        
        if authorization_header:
            token_data = token_process(authorization_header)
            
            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]
        
        try:
            content, file_metadata = gcp_storage_service.read_transcript_file(filename)
            return jsonify({
                "content": content,
                "metadata": file_metadata
            })
        except FileNotFoundError:
            return jsonify({"error": "Transcript file not found"}), 404
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @transcripts_bp.route("/transcripts/dialogue", methods=["POST"])
    def transcript_dialogue():
        """Fetch transcript and return in chat-like format."""
        from monitoring_module import tracer
        
        with tracer.start_as_current_span("api/transcripts/dialogue"):
            authorization_header = request.headers.get("Authorization")
            
            if authorization_header is None:
                return jsonify({"message": "Token is missing"}), 401
            
            if authorization_header:
                token_data = token_process(authorization_header)
                if token_data[1] == 401 or token_data[1] == 403:
                    return (token_data[0].get_json()), token_data[1]
            
            data = request.get_json() or {}
            transcript_file_name = data.get("transcriptFileName") or data.get("fileName")
            use_llm = bool(data.get("useLLM", False))
            
            if not transcript_file_name:
                return jsonify({"error": "transcriptFileName is required"}), 400
            
            if not gcp_storage_service.fs:
                return jsonify({"error": "GCP Storage not configured or unavailable"}), 500
            
            try:
                transcript_content, file_metadata = gcp_storage_service.read_transcript_file(transcript_file_name)
                
                transcript_data = None
                transcript_text = transcript_content
                try:
                    transcript_data = json.loads(transcript_content)
                    extracted = transcript_service.extract_text_from_transcript_json(transcript_data)
                    if extracted:
                        transcript_text = extracted
                except Exception:
                    transcript_data = None
                
                used_llm = False
                conversation = transcript_service.transcript_to_chat_turns(transcript_text, transcript_data)
                
                if use_llm or (len(conversation) <= 1 and len(transcript_text or "") > 600):
                    llm_turns = transcript_service.llm_segment_transcript(transcript_text)
                    if llm_turns:
                        conversation = llm_turns
                        used_llm = True
                
                transcript_id = transcript_file_name.replace(".json", "").replace(".txt", "")
                
                return jsonify({
                    "transcriptId": transcript_id,
                    "transcriptFileName": transcript_file_name,
                    "transcriptMetadata": file_metadata,
                    "conversation": conversation,
                    "totalTurns": len(conversation),
                    "usedLLM": used_llm,
                }), 200
            except FileNotFoundError:
                return jsonify({"error": f"Transcript file not found: {transcript_file_name}"}), 404
            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({"error": "An error occurred while building transcript dialogue", "details": str(e)}), 500
    
    @transcripts_bp.route("/transcripts/status", methods=["PATCH"])
    def update_transcript_status():
        """Update transcript status in MongoDB."""
        from monitoring_module import tracer
        
        with tracer.start_as_current_span('api/transcripts/status'):
            authorization_header = request.headers.get("Authorization")
            
            if authorization_header is None:
                return jsonify({"message": "Token is missing"}), 401
            
            if authorization_header:
                token_data = token_process(authorization_header)
                if token_data[1] == 401 or token_data[1] == 403:
                    return (token_data[0].get_json()), token_data[1]
            
            user_email = token_data[0]["email"]
            data = request.get_json() or {}
            status = data.get("status")
            transcript_file_name = (
                data.get("transcriptFileName") or data.get("fileName") or data.get("transcriptId")
            )
            
            if not transcript_file_name:
                return jsonify({"error": "transcriptFileName or transcriptId is required"}), 400
            if status not in ("active", "inactive"):
                return jsonify({"error": "status must be 'active' or 'inactive'"}), 400
            
            transcript_id = transcript_file_name.replace(".json", "").replace(".txt", "")
            
            qna_collection_user = f"chats_{user_email}"
            qna_collection = database.db[qna_collection_user]
            
            now_ts = datetime.utcnow()
            doc = qna_collection.find_one_and_update(
                {"doc_type": "transcript_status", "transcript_id": transcript_id},
                {"$set": {
                    "doc_type": "transcript_status",
                    "transcript_id": transcript_id,
                    "transcript_file_name": transcript_file_name,
                    "status": status,
                    "updated_at": now_ts,
                }},
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
            
            return jsonify({
                "transcriptId": transcript_id,
                "transcriptFileName": transcript_file_name,
                "status": doc.get("status"),
                "updatedAt": doc.get("updated_at"),
            }), 200
    
    @transcripts_bp.route("/transcripts/conversations", methods=["GET"])
    def list_transcript_conversations():
        """List transcript conversations for a transcript."""
        from monitoring_module import tracer
        
        with tracer.start_as_current_span('api/transcripts/conversations'):
            authorization_header = request.headers.get("Authorization")
            
            if authorization_header is None:
                return jsonify({"message": "Token is missing"}), 401
            
            if authorization_header:
                token_data = token_process(authorization_header)
                if token_data[1] == 401 or token_data[1] == 403:
                    return (token_data[0].get_json()), token_data[1]
            
            user_email = token_data[0]["email"]
            transcript_file_name = (
                request.args.get("transcriptFileName") or 
                request.args.get("fileName") or 
                request.args.get("transcriptId")
            )
            
            if not transcript_file_name:
                return jsonify({"error": "transcriptFileName or transcriptId is required"}), 400
            
            transcript_id = transcript_file_name.replace(".json", "").replace(".txt", "")
            
            qna_collection_user = f"chats_{user_email}"
            qna_collection = database.db[qna_collection_user]
            
            cursor = qna_collection.find(
                {"doc_type": "transcript_conversation", "transcript_id": transcript_id},
                {"_id": 1, "conversation_name": 1, "status": 1, "query_time": 1, "updated_at": 1},
            ).sort([("updated_at", -1), ("query_time", -1)])
            
            conversations = []
            for doc in cursor:
                conversations.append({
                    "conversationId": str(doc.get("_id")),
                    "conversationName": doc.get("conversation_name") or "",
                    "status": (doc.get("status") or "active"),
                    "createdAt": doc.get("query_time"),
                    "updatedAt": doc.get("updated_at") or doc.get("query_time"),
                })
            
            return jsonify({
                "transcriptId": transcript_id,
                "transcriptFileName": transcript_file_name,
                "conversations": conversations,
            }), 200
    
    @transcripts_bp.route("/transcripts/conversation/stub", methods=["POST"])
    def create_transcript_conversation_stub():
        """Create a processing transcript conversation stub."""
        from monitoring_module import tracer
        
        with tracer.start_as_current_span("api/transcripts/conversation/stub"):
            authorization_header = request.headers.get("Authorization")
            
            if authorization_header is None:
                return jsonify({"message": "Token is missing"}), 401
            
            if authorization_header:
                token_data = token_process(authorization_header)
                if token_data[1] == 401 or token_data[1] == 403:
                    return (token_data[0].get_json()), token_data[1]
            
            user_email = token_data[0]["email"]
            data = request.get_json() or {}
            transcript_file_name = data.get("transcriptFileName") or data.get("fileName")
            contract_type = data.get("contractType")
            selected_plan = data.get("selectedPlan")
            selected_state = data.get("selectedState")
            gpt_model = data.get("gptModel", "Search")
            new_conversation = bool(data.get("newConversation", False))
            requested_conversation_name = data.get("conversationName")
            
            if not transcript_file_name:
                return jsonify({"error": "transcriptFileName is required"}), 400
            if not all([contract_type, selected_plan, selected_state]):
                return jsonify({"error": "contractType, selectedPlan, selectedState are required"}), 400
            
            transcript_id = transcript_file_name.replace(".json", "").replace(".txt", "")
            
            qna_collection_user = f"chats_{user_email}"
            qna_collection = database.db[qna_collection_user]
            
            now_ts = datetime.utcnow()
            status_doc = qna_collection.find_one(
                {"doc_type": "transcript_status", "transcript_id": transcript_id},
                {"_id": 0, "status": 1},
            )
            transcript_status = (status_doc or {}).get("status") or "active"
            
            base_name = (requested_conversation_name or transcript_file_name or "").strip() or transcript_id
            if new_conversation:
                existing_count = qna_collection.count_documents(
                    {"doc_type": "transcript_conversation", "transcript_id": transcript_id}
                )
                conv_name = base_name if existing_count == 0 else f"{base_name} ({existing_count + 1})"
            else:
                conv_name = base_name
            
            stub = {
                "doc_type": "transcript_conversation",
                "conversation_mode": "Calls",
                "underlying_model": gpt_model,
                "conversation_name": conv_name,
                "transcript_id": transcript_id,
                "contract_type": contract_type,
                "selected_plan": selected_plan,
                "selected_state": selected_state,
                "query_time": now_ts,
                "updated_at": now_ts,
                "status": transcript_status,
                "case_disposition": None,
                "closed_at": None,
                "review_comments": None,
                "processing": True,
                "chats": [],
            }
            inserted = qna_collection.insert_one(stub)
            conv_doc_id = inserted.inserted_id
            
            return jsonify({
                "conversationId": str(conv_doc_id),
                "conversationName": conv_name,
                "status": transcript_status,
                "processing": True,
                "transcriptId": transcript_id,
                "transcriptFileName": transcript_file_name,
            }), 200
    
    @transcripts_bp.route("/transcripts/process", methods=["POST"])
    def process_transcript():
        """Process transcript: fetch from GCP, extract questions, and get answers."""
        from monitoring_module import tracer
        from time import time
        from langchain_openai import ChatOpenAI
        from langchain.vectorstores import Milvus
        from langchain.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from app.utils.milvus_utils import (
            normalize_state_for_milvus,
            normalize_contract_type,
            normalize_plan_for_milvus
        )
        from app.services.llm_factory import get_llm_factory
        
        try:
            with tracer.start_as_current_span('api/transcripts/process') as parent0:
                start_time = time()
                extraction_warning = None
                
                # Authorization
                with tracer.start_as_current_span('authorization'):
                    authorization_header = request.headers.get("Authorization")
                    if authorization_header is None:
                        return jsonify({"message": "Token is missing"}), 401
                    if authorization_header:
                        token_data = token_process(authorization_header)
                        if token_data[1] == 401 or token_data[1] == 403:
                            return (token_data[0].get_json()), token_data[1]
                
                # Get request data
                with tracer.start_as_current_span('data-fetching'):
                    data = request.get_json()
                    if not data:
                        return jsonify({"error": "Request body is missing or invalid"}), 400
                    
                    transcript_file_name = data.get("transcriptFileName")
                    contract_type = data.get("contractType")
                    selected_plan = data.get("selectedPlan")
                    selected_state = data.get("selectedState")
                    milvus_state = normalize_state_for_milvus(selected_state)
                    contract_type_norm = normalize_contract_type(contract_type)
                    selected_plan_norm = normalize_plan_for_milvus(contract_type_norm, selected_plan)
                    gpt_model = data.get("gptModel", "Search")
                    extract_questions = data.get("extractQuestions", True)
                    provided_questions = data.get("questions", [])
                    force_reprocess = bool(data.get("forceReprocess", False))
                    new_conversation = bool(data.get("newConversation", False))
                    requested_conversation_name = data.get("conversationName")
                    
                    if not transcript_file_name:
                        return jsonify({"error": "transcriptFileName is required"}), 400
                    if extract_questions and not all([contract_type, selected_plan, selected_state]):
                        return jsonify({
                            "error": "contractType, selectedPlan, selectedState are required when extractQuestions=true"
                        }), 400
                
                user_email = token_data[0]["email"]
                transcript_id = transcript_file_name.replace(".json", "").replace(".txt", "")
                
                # Use the existing per-user chat collection
                qna_collection_user = f"chats_{user_email}"
                qna_collection = database.db[qna_collection_user]
                
                # Handle conversation stub if provided
                requested_conversation_id = data.get("conversationId") or data.get("conversation_id")
                conv_doc_id = None
                if requested_conversation_id:
                    try:
                        conv_doc_id = ObjectId(str(requested_conversation_id))
                        existing = qna_collection.find_one({"_id": conv_doc_id}) or {}
                        if existing.get("doc_type") != "transcript_conversation":
                            return jsonify({"error": "Invalid conversationId for transcript processing"}), 400
                        force_reprocess = True
                        now_ts = datetime.utcnow()
                        qna_collection.update_one(
                            {"_id": conv_doc_id},
                            {
                                "$set": {
                                    "processing": True,
                                    "updated_at": now_ts,
                                    "conversation_mode": "Calls",
                                    "underlying_model": gpt_model,
                                    "transcript_id": transcript_id,
                                    "contract_type": contract_type,
                                    "selected_plan": selected_plan,
                                    "selected_state": selected_state,
                                }
                            },
                        )
                    except Exception:
                        return jsonify({"error": "Invalid conversationId"}), 400
                
                # Check for existing conversation (cache)
                existing_conv = None
                conv_name = None
                _PLACEHOLDER_CHUNK_VALUES = get_placeholder_chunk_values()
                
                if not new_conversation:
                    existing_conv = qna_collection.find_one(
                        {"doc_type": "transcript_conversation", "transcript_id": transcript_id},
                        sort=[("updated_at", -1), ("query_time", -1)],
                    )
                
                if existing_conv and not force_reprocess and not new_conversation:
                    # Check for placeholder chunks
                    try:
                        existing_chats = existing_conv.get("chats") or []
                        has_placeholder_chunks = False
                        for c in existing_chats:
                            if c.get("chat_id") == "final_answer":
                                continue
                            rc = c.get("relevant_chunks") or []
                            if rc and all(
                                (
                                    isinstance(x, dict)
                                    and (str(x.get("content") or "").strip() in _PLACEHOLDER_CHUNK_VALUES)
                                )
                                or (
                                    isinstance(x, str)
                                    and (x.strip() in _PLACEHOLDER_CHUNK_VALUES)
                                )
                                for x in rc
                            ):
                                has_placeholder_chunks = True
                                break
                        if has_placeholder_chunks or not existing_conv.get("final_summary"):
                            conv_doc_id = existing_conv.get("_id")
                            conv_name = existing_conv.get("conversation_name")
                            existing_conv = None
                    except Exception:
                        existing_conv = None
                
                if existing_conv and not force_reprocess and not new_conversation:
                    # Return cached response
                    cached = existing_conv.get("response_payload") or {}
                    cached["conversationId"] = str(existing_conv.get("_id"))
                    cached.setdefault("transcriptId", existing_conv.get("transcript_id"))
                    cached.setdefault("transcriptMetadata", existing_conv.get("transcript_metadata"))
                    cached.setdefault("finalSummary", existing_conv.get("final_summary"))
                    cached.setdefault("status", existing_conv.get("status", "active"))
                    cached.setdefault("conversationName", existing_conv.get("conversation_name"))
                    
                    if not cached.get("questions") and existing_conv.get("chats"):
                        cached["questions"] = [
                            {
                                "questionId": c.get("chat_id"),
                                "question": c.get("entered_query"),
                                "answer": c.get("response"),
                                "relevantChunks": c.get("relevant_chunks", []),
                                "latency": c.get("latency", 0.0),
                                "confidence": c.get("confidence", 0.0),
                            }
                            for c in existing_conv.get("chats", [])
                        ]
                    
                    # Normalize cached chunks
                    try:
                        for q in cached.get("questions", []) or []:
                            rc = q.get("relevantChunks") or []
                            if isinstance(rc, list):
                                rc = [str(x) for x in rc if str(x).strip()]
                            else:
                                rc = []
                            if not rc and q.get("questionId") != "final_answer":
                                rc = ["(No supporting excerpts found)"]
                            if settings.MILVUS_MAX_RETURN_CHUNKS is not None:
                                rc = rc[:settings.MILVUS_MAX_RETURN_CHUNKS]
                            q["relevantChunks"] = rc
                    except Exception:
                        pass
                    
                    cached.setdefault(
                        "finalAnswer",
                        {
                            "question": "Final Answer for transcript",
                            "answer": cached.get("finalSummary") or "",
                        },
                    )
                    
                    return jsonify(cached), 200
                
                # Create/update processing stub
                if existing_conv and (force_reprocess and not new_conversation):
                    conv_doc_id = existing_conv.get("_id")
                    conv_name = existing_conv.get("conversation_name")
                    existing_conv = None
                
                now_ts = datetime.utcnow()
                status_doc = qna_collection.find_one(
                    {"doc_type": "transcript_status", "transcript_id": transcript_id},
                    {"_id": 0, "status": 1},
                )
                transcript_status = (status_doc or {}).get("status") or "active"
                
                if not conv_name:
                    base_name = (requested_conversation_name or transcript_file_name or "").strip() or transcript_id
                    if new_conversation:
                        existing_count = qna_collection.count_documents(
                            {"doc_type": "transcript_conversation", "transcript_id": transcript_id}
                        )
                        conv_name = base_name if existing_count == 0 else f"{base_name} ({existing_count + 1})"
                    else:
                        conv_name = base_name
                
                if conv_doc_id is None:
                    stub = {
                        "doc_type": "transcript_conversation",
                        "conversation_mode": "Calls",
                        "underlying_model": gpt_model,
                        "conversation_name": conv_name,
                        "transcript_id": transcript_id,
                        "contract_type": contract_type,
                        "selected_plan": selected_plan,
                        "selected_state": selected_state,
                        "query_time": now_ts,
                        "updated_at": now_ts,
                        "status": transcript_status,
                        "processing": True,
                        "chats": [],
                    }
                    inserted = qna_collection.insert_one(stub)
                    conv_doc_id = inserted.inserted_id
                else:
                    qna_collection.update_one(
                        {"_id": conv_doc_id},
                        {"$set": {"processing": True, "updated_at": now_ts}},
                    )
                
                # Read transcript from GCP
                with tracer.start_as_current_span('download-transcript'):
                    if not gcp_storage_service.fs:
                        return jsonify({"error": "GCP Storage not configured or unavailable"}), 500
                    
                    try:
                        transcript_content, file_metadata = gcp_storage_service.read_transcript_file(transcript_file_name)
                        try:
                            transcript_data = json.loads(transcript_content)
                            if isinstance(transcript_data, dict):
                                transcript_text = transcript_data.get("text",
                                    transcript_data.get("transcript",
                                    transcript_data.get("content", str(transcript_data))))
                            else:
                                transcript_text = transcript_content
                        except json.JSONDecodeError:
                            transcript_text = transcript_content
                    except FileNotFoundError as e:
                        return jsonify({"error": f"Transcript file not found: {transcript_file_name}"}), 404
                    except Exception as e:
                        return jsonify({"error": f"Error reading transcript file: {str(e)}"}), 500
                
                # Extract questions
                questions = []
                if extract_questions:
                    with tracer.start_as_current_span('extract-questions'):
                        llm_extract = ChatOpenAI(temperature=0.0, model="gpt-4o")
                        questions = transcript_service.extract_relevant_customer_questions(transcript_text, llm_extract)
                        if not questions or len(questions) == 0:
                            questions = transcript_service.extract_questions_with_agent(transcript_text, llm_extract)
                        if not questions:
                            extraction_warning = "No questions could be extracted from transcript; inferring from context."
                            inferred_question = {
                                "question": f"Is this issue covered: {transcript_text[:120]}",
                                "context": transcript_text[:400],
                                "questionType": "coverage",
                                "userIntent": "Customer wants to know if the described issue is covered",
                                "questionId": "q1",
                            }
                            questions = [inferred_question]
                else:
                    questions = provided_questions
                    if not questions:
                        return jsonify({"error": "No questions provided"}), 400
                
                # Initialize vector DB and LLM
                with tracer.start_as_current_span('vector_db-initialization'):
                    collection_mapping = {
                        "RE": {
                            "ShieldEssential": f"{milvus_state}_RE_ShieldEssential",
                            "ShieldPlus": f"{milvus_state}_RE_ShieldPlus",
                            "default": f"{milvus_state}_RE_ShieldComplete",
                        },
                        "DTC": {
                            "ShieldSilver": f"{milvus_state}_DTC_ShieldSilver",
                            "ShieldGold": f"{milvus_state}_DTC_ShieldGold",
                            "default": f"{milvus_state}_DTC_ShieldPlatinum",
                        },
                    }
                    
                    selected_collection_name = collection_mapping.get(contract_type_norm, {}).get(
                        selected_plan_norm, collection_mapping.get(contract_type_norm, {}).get("default")
                    )
                    
                    llm_factory = get_llm_factory()
                    embedding = llm_factory.create_embedding()
                    
                    vector_db1 = Milvus(
                        embedding,
                        collection_name=selected_collection_name,
                        connection_args={"host": settings.MILVUS_HOST, "port": "19530"},
                    )
                    retriever = vector_db1.as_retriever(search_kwargs={"k": settings.MILVUS_RETRIEVER_K})
                    
                    if gpt_model == "Search":
                        llm2 = ChatOpenAI(temperature=0.0, model="ft:gpt-3.5-turbo-0613:mindstix::8YYD56aA")
                        llm = ChatOpenAI(temperature=0.0, model="gpt-4o")
                    elif gpt_model == "Infer":
                        llm3 = ChatOpenAI(temperature=0.0, model="ft:gpt-3.5-turbo-0613:mindstix::8YYD56aA")
                        llm = ChatOpenAI(temperature=0.0, model='gpt-4o')
                        llm2 = ChatOpenAI(temperature=0.0, model='gpt-4o')
                    else:
                        return jsonify({"error": f"Invalid gpt_model: {gpt_model}. Must be 'Search' or 'Infer'"}), 400
                    
                    handler = None  # Can add callback handler if needed
                
                # Process each question
                results = []
                total_latency = 0
                confidences = []
                
                with tracer.start_as_current_span('process-questions'):
                    for question_obj in questions:
                        question_text = question_obj.get("question", "")
                        question_id = question_obj.get("questionId", f"q{len(results) + 1}")
                        
                        result = transcript_processor_service.process_single_transcript_question(
                            question_text, contract_type, selected_plan,
                            selected_state, gpt_model, vector_db1, llm, llm2,
                            retriever, handler,
                            transcript_context=question_obj.get("context", ""),
                        )
                        
                        result["questionId"] = question_id
                        result["question"] = question_text
                        result["context"] = question_obj.get("context", "")
                        result["questionType"] = question_obj.get("questionType", "general")
                        result["userIntent"] = question_obj.get("userIntent", "")
                        
                        # Handle relevantChunks
                        rc = result.get("relevantChunks") or []
                        if isinstance(rc, list) and len(rc) > 0:
                            if isinstance(rc[0], dict):
                                if settings.MILVUS_MAX_RETURN_CHUNKS is not None:
                                    rc = rc[:settings.MILVUS_MAX_RETURN_CHUNKS]
                            else:
                                rc = [{"content": str(x), "name": f"Clause {i+1}"} for i, x in enumerate(rc) if str(x).strip()]
                                if settings.MILVUS_MAX_RETURN_CHUNKS is not None:
                                    rc = rc[:settings.MILVUS_MAX_RETURN_CHUNKS]
                        else:
                            rc = [{"content": "(No supporting excerpts found)", "name": "Clause 1"}]
                        result["relevantChunks"] = rc
                        
                        if "error" not in result:
                            confidences.append(result.get("confidence", 0.0))
                            total_latency += result.get("latency", 0.0)
                        
                        results.append(result)
                
                # Calculate summary
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                
                response = {
                    "transcriptId": transcript_id,
                    "transcriptMetadata": {
                        "fileName": file_metadata["fileName"],
                        "uploadDate": file_metadata["uploadDate"],
                        "fileSize": file_metadata["fileSize"]
                    },
                    "questions": results,
                    "summary": {
                        "totalQuestions": len(questions),
                        "processedQuestions": len([r for r in results if "error" not in r]),
                        "averageConfidence": round(avg_confidence, 2),
                        "totalLatency": round(total_latency, 2)
                    }
                }
                if extraction_warning:
                    response["warning"] = extraction_warning
                
                # Claim decision
                try:
                    all_chunks = []
                    for r in results or []:
                        rc = r.get("relevantChunks") or []
                        if isinstance(rc, list):
                            for chunk in rc:
                                if isinstance(chunk, dict):
                                    all_chunks.append(str(chunk.get("content", "")))
                                else:
                                    all_chunks.append(str(chunk))
                    seen = set()
                    deduped = []
                    for c in all_chunks:
                        if c in seen:
                            continue
                        seen.add(c)
                        deduped.append(c)
                    claims_context = []
                    for r in results or []:
                        if not isinstance(r, dict):
                            continue
                        claims_context.append({
                            "claimId": (r.get("questionId") or ""),
                            "customerClaim": (r.get("question") or ""),
                            "situation": (r.get("context") or ""),
                        })
                    claim_decision = claims_service.generate_claim_decision_from_chunks(
                        deduped, claims_context=claims_context
                    )
                    response["claimDecision"] = claim_decision
                except Exception as e:
                    print(f"Warning: unable to generate claimDecision: {e}")
                
                # Final summary
                final_summary_text = ""
                try:
                    with tracer.start_as_current_span('final-summary'):
                        llm_summary = ChatOpenAI(temperature=0.0, model="gpt-4o")
                        qa_lines = []
                        for r in results or []:
                            if not r:
                                continue
                            q = (r.get("question") or "").strip()
                            if not q:
                                continue
                            ctx = (r.get("context") or "").strip()
                            a = (r.get("answer") or "").strip()
                            if not a:
                                a = "(No answer was generated for this question.)"
                            if ctx:
                                qa_lines.append(f"Q: {q}\nSituation: {ctx}\nA: {a}")
                            else:
                                qa_lines.append(f"Q: {q}\nA: {a}")
                        
                        qa_blob = "\n\n".join(qa_lines)
                        if qa_blob.strip():
                            summary_prompt = PromptTemplate(
                                input_variables=["qa_blob"],
                                template=(
                                    "You are writing the FINAL ANSWER for a claims transcript.\n"
                                    "IMPORTANT: Do NOT present the final answer as a list of each Q&A.\n"
                                    "Instead, synthesize ALL Q&A into an APPLIANCE/ITEM-BASED final answer.\n"
                                    "\n"
                                    "Task:\n"
                                    "- Identify the distinct appliance(s)/item(s)/system(s) mentioned across the Q&A.\n"
                                    "- Group/merge related questions into the correct item section (do not repeat the questions).\n"
                                    "- If the transcript includes multiple items with separate claims, show them as separate sections.\n"
                                    "\n"
                                    "For EACH item section, include in JSON FORMAT:\n"
                                    "- ITEM : <1,2,3...>\n"
                                    "- ITEM: <name> (add 1-line details if available: location/part/symptom)\n"
                                    "- TYPE: Appliance | System | Fixture | Other (infer from wording; if unclear use Other)\n"
                                    "- DECISION: APPROVED | REJECTED | PARTIAL | NEED_HUMAN_ASSISTANCE\n"
                                    "- AMOUNTS (only if mentioned in Q&A):\n"
                                    "  1. Customer quoted/asked: $...\n"
                                    "  2. Company can provide: $... (coverage amount/limit/service fee/deductible as stated in Q&A)\n"
                                    "- Situation: what happened / what customer is claiming (from Situation lines)\n"
                                    "- What's covered (numeric list, if any)\n"
                                    "- What's not covered / limitations (numeric list, if any)\n"
                                    "- Why (1–2 short sentences grounded in the Q&A outcomes; no policy speculation)\n"
                                    "- Next steps (specific actions the customer should take)\n"
                                    "\n"
                                    "CRITICAL DECISION RULES:\n"
                                    "- The DECISION field is MANDATORY and MUST NEVER be left empty for any item.\n"
                                    "- If it is confirmed that there is NO coverage for a particular item, the DECISION MUST be REJECTED.\n"
                                    "- If outcomes are mixed for the same item, use PARTIAL and clearly break down covered vs not covered.\n"
                                    "- If coverage cannot be determined, use NEED_HUMAN_ASSISTANCE.\n"
                                    "- Be concise, decisive, and avoid hypothetical/if-then language.\n"
                                    "- End with a short overall next step (1–2 bullets) if multiple items exist.\n\n"
                                    "{qa_blob}\n"
                                ),
                            )
                            summary_chain = summary_prompt | llm_summary | StrOutputParser()
                            final_summary_text = summary_chain.invoke({"qa_blob": qa_blob}).strip()
                except Exception as e:
                    print(f"Warning: failed to generate final transcript summary: {e}")
                
                if (not final_summary_text.strip()) and (results and len(results) > 0):
                    final_summary_text = "\n".join([
                        f"- {((r.get('answer') or '').strip() or '(No answer was generated for this question.)')}"
                        for r in results
                        if r and (r.get("question") or "").strip()
                    ]).strip()
                
                response["finalSummary"] = final_summary_text
                response["finalAnswer"] = {
                    "question": "Final Answer for transcript",
                    "answer": final_summary_text,
                }
                
                # Persist to MongoDB
                transcript_chats = []
                now_ts = datetime.utcnow()
                for res in results:
                    chunks = res.get("relevantChunks", []) or []
                    chunk_contents = []
                    for c in chunks:
                        if isinstance(c, dict):
                            chunk_contents.append(str(c.get("content", "")))
                        else:
                            chunk_contents.append(str(c))
                    relevant_docs_text = "\n\n---\n\n".join([c for c in chunk_contents if c.strip()])
                    normalized_chunks = normalize_chunks_with_names(chunks)
                    
                    transcript_chats.append({
                        "chat_id": res.get("questionId"),
                        "entered_query": res.get("question", ""),
                        "response": res.get("answer", ""),
                        "relevant_chunks": normalized_chunks,
                        "relevant_docs": relevant_docs_text,
                        "gpt_model": "Calls",
                        "underlying_model": gpt_model,
                        "chat_timestamp": now_ts,
                        "latency": res.get("latency", 0.0),
                        "confidence": res.get("confidence", 0.0),
                    })
                
                transcript_chats.append({
                    "chat_id": "final_answer",
                    "entered_query": "Final Answer for transcript",
                    "response": final_summary_text,
                    "relevant_chunks": [],
                    "relevant_docs": "",
                    "gpt_model": "Calls",
                    "underlying_model": gpt_model,
                    "chat_timestamp": now_ts,
                    "latency": 0.0,
                    "confidence": 0.0,
                })
                
                response["questions"] = (response.get("questions") or []) + [{
                    "questionId": "final_answer",
                    "question": "Final Answer for transcript",
                    "answer": final_summary_text,
                    "relevantChunks": [],
                    "confidence": 0.0,
                    "latency": 0.0,
                }]
                
                transcript_doc = {
                    "doc_type": "transcript_conversation",
                    "conversation_mode": "Calls",
                    "underlying_model": gpt_model,
                    "conversation_name": conv_name or transcript_file_name,
                    "transcript_id": transcript_id,
                    "transcript_metadata": response["transcriptMetadata"],
                    "contract_type": contract_type,
                    "selected_plan": selected_plan,
                    "selected_state": selected_state,
                    "query_time": now_ts,
                    "updated_at": now_ts,
                    "status": transcript_status,
                    "processing": False,
                    "summary": response.get("summary"),
                    "final_summary": final_summary_text,
                    "claim_decision": response.get("claimDecision"),
                    "chats": transcript_chats,
                }
                
                if conv_doc_id is None:
                    inserted = qna_collection.insert_one(transcript_doc)
                    conv_doc_id = inserted.inserted_id
                else:
                    qna_collection.update_one(
                        {"_id": conv_doc_id},
                        {"$set": transcript_doc},
                    )
                
                response["conversationId"] = str(conv_doc_id)
                response["status"] = transcript_status
                response["conversationName"] = conv_name or transcript_file_name
                
                qna_collection.update_one(
                    {"_id": conv_doc_id},
                    {"$set": {"response_payload": response}},
                )
                
                return jsonify(response), 200
                
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in /transcripts/process endpoint: {str(e)}")
            print(f"Traceback: {error_trace}")
            return jsonify({
                "error": "An error occurred while processing transcript",
                "details": str(e)
            }), 500
    
    return transcripts_bp
