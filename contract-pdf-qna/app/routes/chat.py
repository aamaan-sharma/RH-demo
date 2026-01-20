"""Chat endpoints."""
import uuid
import threading
from time import time
from datetime import datetime
from flask import Blueprint, request, jsonify, make_response
from bson.objectid import ObjectId
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from app.utils.auth import token_process
from app.services.rag_service import RAGService
from app.services.agent_service import AgentService
from app.services.memory_service import MemoryService
from app.services.standalone_service import StandaloneService
from app.services.llm_factory import get_llm_factory
from app.utils.milvus_utils import normalize_state_for_milvus, normalize_contract_type, normalize_plan_for_milvus
from app.models.database import Database

chat_bp = Blueprint('chat', __name__)


def init_chat_routes(
    rag_service: RAGService,
    agent_service: AgentService,
    memory_service: MemoryService,
    database: Database
):
    """Initialize chat routes with service dependencies.
    
    Args:
        rag_service: RAGService instance
        agent_service: AgentService instance
        memory_service: MemoryService instance
        database: Database service instance
    """
    standalone_service = StandaloneService()
    llm_factory = get_llm_factory()
    
    @chat_bp.route("/start", methods=["POST"])
    def start():
        """Start a new chat conversation or continue existing one."""
        try:
            from monitoring_module import tracer, q_monitor, llm_trace_to_jaeger
            from token_module import token_calculator, CallbackHandler
            
            with tracer.start_as_current_span('api/start') as parent0:
                handler = CallbackHandler()
                start_time = time()
                
                with tracer.start_as_current_span('authorization'):
                    authorization_header = request.headers.get("Authorization")
                    
                    if authorization_header is None:
                        return jsonify({"message": "Token is missing"}), 401
                    
                    if authorization_header:
                        token_data = token_process(authorization_header)
                        
                        if token_data[1] == 401 or token_data[1] == 403:
                            return (token_data[0].get_json()), token_data[1]
                
                with tracer.start_as_current_span('data-fetching'):
                    data = request.get_json()
                    if not data:
                        return jsonify({"error": "Request body is missing or invalid"}), 400
                    
                    contract_type = data.get("contractType")
                    selected_plan = data.get("selectedPlan")
                    selected_state = data.get("selectedState")
                    milvus_state = normalize_state_for_milvus(selected_state)
                    contract_type_norm = normalize_contract_type(contract_type)
                    selected_plan_norm = normalize_plan_for_milvus(contract_type_norm, selected_plan)
                    gpt_model = data.get("gptModel")
                    entered_query = data.get("enteredQuery")
                    
                    if not all([contract_type, selected_plan, selected_state, gpt_model, entered_query]):
                        return jsonify({
                            "error": "Missing required fields: contractType, selectedPlan, selectedState, gptModel, enteredQuery"
                        }), 400
                    
                    user_email = token_data[0]["email"]
                    conversation_id = request.args.get("conversation-id")
                    
                    collection_name = rag_service.get_collection_for_context(
                        contract_type, selected_plan, selected_state
                    )
                    
                    if not collection_name:
                        return jsonify({"error": "Invalid contract/plan/state combination"}), 400
                
                with tracer.start_as_current_span('vector_db-initialization'):
                    vector_db = rag_service.get_vector_db(collection_name)
                
                # Initialize variables
                agent_resp = None
                relevant_documents = ""
                
                # Create memory for conversation context
                from langchain_core.chat_history import InMemoryChatMessageHistory
                memory1 = InMemoryChatMessageHistory()
                
                # Load conversation memory
                question1, answer1 = memory_service.load_conversation_memory(
                    user_email, conversation_id or "", memory1, max_pairs=3
                )
                
                if gpt_model == "Search":
                    with tracer.start_as_current_span('Search') as parent1:
                        with tracer.start_as_current_span('llm-retriever-initialization'):
                            llm2 = llm_factory.create_standalone_llm()
                            llm = llm_factory.create_chat_llm(model="gpt-4o", temperature=0.0)
                            retriever = rag_service.create_retriever(collection_name)
                        
                        with tracer.start_as_current_span('standalone-prompt-chain') as p:
                            standalone_result = standalone_service.create_standalone_question(
                                current_question=entered_query,
                                previous_question=question1,
                                previous_answer=answer1,
                                mode="Search",
                                handler=handler
                            )
                            
                            # Token tracking
                            res1, tok1 = handler.infi()
                            llm_trace_to_jaeger(res1, tok1)
                            threading.Thread(target=token_calculator, args=(tok1,)).start()
                        
                        with tracer.start_as_current_span('q_monitor') as parentq:
                            threading.Thread(target=q_monitor, args=(parentq, entered_query)).start()
                        
                        with tracer.start_as_current_span('llm-RetrievalQA-chain') as q:
                            prompt_template = """
You are assisting a customer care executive. Your role is to review the contract's contextual information given in the context below.

{context}

Answer the given user inquiry based on context above as truthfully as possible, providing in-depth explanations together with answers to the inquiries.
You may rephrase the final response to make it concise and sound more human-like, but do not go out of context and do not lose important details and meaning.

You'll be asked about repairs, coverage policy and service questions about home appliances, home fixtures, home care, repairs/replacement and cleaning, and also about the renewal, cancellation or refund policies in the contract, whether a certain service is covered under the contract and similar context.

The contract context given will have information about contractual details, terms and conditions, renewals, cancellation, refund and service request policies, the coverage limits, limitation and exclusion policies. You will need to use and infer from all the information available in context to analyze and then respond with the final answer.

If the question is about a square feet limit, make sure to compare the numerical values properly. 
For example,
Question: "will my 800 square feet guest house be covered?"
The answer to this question will be No, as the square feet limit for guest houses is 750 and 800 is greater than 750.

If the inquiry is unrelated to home repair and service, answer with "I don't have the information to answer this question.". For example, questions like "Tell me about space.", "Write a poem for me.", "Where can I buy a refrigerator?", "Hi! How are you?", etc. are out of context.

Always include the appliance name in the answer and provide in depth information.

Make the answer as short as possible with in depth information.

Question: {standalone_result}
Answer: """
                            
                            # Replace standalone_result placeholder in template
                            final_prompt_template = prompt_template.replace("{standalone_result}", standalone_result)
                            PROMPT = PromptTemplate(
                                template=final_prompt_template,
                                input_variables=["context"]
                            )
                            qa_chain = rag_service.create_retrieval_qa(
                                retriever, llm, prompt_template=final_prompt_template
                            )
                            
                            qa_resp = qa_chain.invoke(
                                {"query": standalone_result},
                                config={"callbacks": [handler]},
                            )
                            agent_resp = qa_resp["result"] if isinstance(qa_resp, dict) else qa_resp
                            
                            res2, tok2 = handler.infi()
                            llm_trace_to_jaeger(res2, tok2)
                            threading.Thread(target=token_calculator, args=(tok2,)).start()
                        
                        with tracer.start_as_current_span('relevant_documents'):
                            relevant_documents = rag_service.get_relevant_documents(entered_query, retriever)
                
                elif gpt_model == "Infer":
                    with tracer.start_as_current_span('Infer') as parent1:
                        with tracer.start_as_current_span('llm-retriever-initialization'):
                            llm3 = llm_factory.create_standalone_llm()
                            llm = llm_factory.create_chat_llm(model="gpt-4o", temperature=0.0)
                            llm2 = llm_factory.create_chat_llm(model="gpt-4o", temperature=0.0)
                            retriever = rag_service.create_retriever(collection_name)
                        
                        with tracer.start_as_current_span('standalone-prompt-chain') as p:
                            standalone_result = standalone_service.create_standalone_question(
                                current_question=entered_query,
                                previous_question=question1,
                                previous_answer=answer1,
                                mode="Infer",
                                handler=handler
                            )
                            
                            res1, tok1 = handler.infi()
                            llm_trace_to_jaeger(res1, tok1)
                            threading.Thread(target=token_calculator, args=(tok1,)).start()
                        
                        with tracer.start_as_current_span('q_monitor') as parentq:
                            threading.Thread(target=q_monitor, args=(parentq, entered_query)).start()
                        
                        with tracer.start_as_current_span('llm-RetrievalQA-chain') as q:
                            qa_chain = rag_service.create_retrieval_qa(retriever, llm2)
                            agent = agent_service.create_agent(qa_chain, llm, handler=handler)
                            agent_response = agent_service.run_agent(agent, standalone_result, handler=handler)
                            agent_resp = agent_response["output"]
                            
                            res2, tok2 = handler.infi()
                            llm_trace_to_jaeger(res2, tok2)
                            threading.Thread(target=token_calculator, args=(tok2,)).start()
                        
                        with tracer.start_as_current_span('relevant_documents'):
                            knowledge_base_thoughts = [
                                item[0].tool_input
                                for item in agent_response.get("intermediate_steps", [])
                                if item[0].tool == 'Knowledge Base'
                            ]
                            relevant_documents = ""
                            for action_input in knowledge_base_thoughts:
                                rd = rag_service.get_relevant_documents(action_input, retriever)
                                relevant_documents += rd
                else:
                    return jsonify({
                        "error": f"Invalid gpt_model: {gpt_model}. Must be 'Search' or 'Infer'"
                    }), 400
                
                with tracer.start_as_current_span('output-formating'):
                    if agent_resp is None:
                        return jsonify({
                            "error": "Invalid gpt_model. Must be 'Search' or 'Infer'"
                        }), 400
                    
                    ai_response = agent_resp
                    word_count = len(relevant_documents.split())
                    latency = time() - start_time
                    query_time = datetime.utcnow()
                    
                    chat = {
                        "chat_id": str(uuid.uuid4()),
                        "entered_query": entered_query,
                        "response": ai_response,
                        "relevant_docs": relevant_documents,
                        "gpt_model": gpt_model,
                        "chat_timestamp": query_time,
                        "latency": latency,
                        "word_count": word_count
                    }
                    
                    if conversation_id is None or conversation_id == "":
                        qna_json = {
                            "conversation_name": entered_query,
                            "contract_type": contract_type,
                            "selected_plan": selected_plan,
                            "selected_state": selected_state,
                            "query_time": query_time,
                            "status": "active",
                            "conversation_mode": gpt_model,
                            "chats": [chat],
                        }
                        
                        result = database.insert_qna(qna_json, user_email)
                        conversation_id = str(result.inserted_id)
                    else:
                        database.update_chat(chat, conversation_id, user_email)
                        
                        # Update conversation_mode if not transcript conversation
                        try:
                            qna_collection_user = f"chats_{user_email}"
                            qna_collection = database.db[qna_collection_user]
                            existing = qna_collection.find_one(
                                {"_id": ObjectId(conversation_id)},
                                {"_id": 0, "doc_type": 1, "conversation_mode": 1},
                            ) or {}
                            if existing.get("doc_type") != "transcript_conversation":
                                qna_collection.update_one(
                                    {"_id": ObjectId(conversation_id)},
                                    {"$set": {"conversation_mode": gpt_model}},
                                )
                        except Exception:
                            pass
                    
                    output_json = {
                        "aiResponse": ai_response,
                        "conversationId": str(conversation_id),
                        "chatId": chat.get("chat_id")
                    }
            
            return make_response(jsonify(output_json), 200)
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in /start endpoint: {str(e)}")
            print(f"Traceback: {error_trace}")
            return jsonify({
                "error": "An error occurred while processing your request",
                "details": str(e)
            }), 500
    
    @chat_bp.route("/calls/start", methods=["POST"])
    def calls_start():
        """Start a calls conversation."""
        try:
            authorization_header = request.headers.get("Authorization")
            
            if authorization_header is None:
                return jsonify({"message": "Token is missing"}), 401
            
            if authorization_header:
                token_data = token_process(authorization_header)
                
                if token_data[1] == 401 or token_data[1] == 403:
                    return (token_data[0].get_json()), token_data[1]
            
            data = request.get_json()
            if not data:
                return jsonify({"error": "Request body is missing or invalid"}), 400
            
            contract_type = data.get("contractType")
            selected_plan = data.get("selectedPlan")
            selected_state = data.get("selectedState")
            entered_query = data.get("enteredQuery")
            
            if not all([contract_type, selected_plan, selected_state, entered_query]):
                return jsonify({
                    "error": "Missing required fields: contractType, selectedPlan, selectedState, enteredQuery"
                }), 400
            
            user_email = token_data[0]["email"]
            conversation_id = request.args.get("conversation-id")
            
            if conversation_id is None or conversation_id == "":
                return jsonify({"error": "Calls conversationId is required"}), 400
            
            # Check if calls conversation exists
            # Note: calls_conversations_collection needs to be defined in Database model
            # For now, using direct access
            from pymongo import MongoClient
            from app.config.settings import settings
            mongo_client = MongoClient(settings.MONGO_URI)
            db2 = mongo_client[settings.MONGO_DB_NAME] if settings.MONGO_DB_NAME else None
            calls_conversations_collection = db2.calls_conversations if db2 else None
            if not calls_conversations_collection:
                return jsonify({"error": "Calls conversations collection not available"}), 500
            
            try:
                calls_conversation = calls_conversations_collection.find_one(
                    {"_id": ObjectId(conversation_id), "user_email": user_email}
                )
            except Exception:
                calls_conversation = None
            
            if not calls_conversation:
                return jsonify({"error": "Calls conversation not found"}), 404
            
            query_time = datetime.utcnow()
            
            chat = {
                "chat_id": str(uuid.uuid4()),
                "entered_query": entered_query,
                "response": f"You are in Calls mode. This is a placeholder response for: {entered_query}",
                "gpt_model": "Calls",
                "chat_timestamp": query_time,
            }
            
            calls_conversations_collection.update_one(
                {"_id": ObjectId(conversation_id)},
                {
                    "$push": {"chats": chat},
                    "$set": {
                        "contract_type": contract_type,
                        "selected_plan": selected_plan,
                        "selected_state": selected_state,
                        "updated_at": query_time,
                    },
                },
            )
            
            output_json = {
                "aiResponse": chat["response"],
                "conversationId": str(conversation_id),
                "chatId": chat.get("chat_id"),
            }
            
            return make_response(jsonify(output_json), 200)
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in /calls/start endpoint: {str(e)}")
            print(f"Traceback: {error_trace}")
            return jsonify({
                "error": "An error occurred while processing your request",
                "details": str(e)
            }), 500
    
    return chat_bp
