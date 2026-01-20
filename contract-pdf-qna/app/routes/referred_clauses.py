"""Referred clauses endpoint."""
from flask import Blueprint, request, jsonify
from app.utils.auth import token_process
from app.services.claims_service import ClaimsService
from app.models.database import Database

referred_clauses_bp = Blueprint('referred_clauses', __name__)


def init_referred_clauses_routes(claims_service: ClaimsService, database: Database):
    """Initialize referred clauses routes with service dependencies.
    
    Args:
        claims_service: ClaimsService instance
        database: Database service instance
    """
    @referred_clauses_bp.route("/referred-clauses", methods=["GET"])
    def referred_clauses():
        from monitoring_module import tracer
        
        with tracer.start_as_current_span('api/referred-clauses'):
            authorization_header = request.headers.get("Authorization")
            
            if authorization_header is None:
                return jsonify({"message": "Token is missing"}), 401
            
            if authorization_header:
                token_data = token_process(authorization_header)
                
                if token_data[1] == 401 or token_data[1] == 403:
                    return (token_data[0].get_json()), token_data[1]
            
            conversation_id = request.args.get("conversation-id")
            user_email = token_data[0]["email"]
            
            if not conversation_id:
                return jsonify({"error": "conversation-id is required"}), 400
            
            docs = database.read_qna(user_email, conversation_id)
            if not docs:
                return jsonify({"error": "Conversation not found"}), 404
            
            # Get the last query from the conversation
            chats = docs.get("chats", [])
            last_query = ""
            if chats:
                last_chat = chats[-1]
                last_query = last_chat.get("entered_query", "")
            
            if not last_query:
                return jsonify({"error": "No query found in conversation"}), 404
            
            # Retrieve chunks
            chunks_for_ui, referred_docs_text = claims_service.retrieve_policy_chunks_for_claims(
                docs, last_query, k=6
            )
            
            return jsonify({
                "referredClauses": chunks_for_ui,
                "referredDocs": referred_docs_text,
            })
    
    return referred_clauses_bp
