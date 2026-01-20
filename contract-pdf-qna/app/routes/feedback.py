"""Feedback endpoint."""
from datetime import datetime
from flask import Blueprint, request, jsonify
from app.utils.auth import token_process
from app.models.database import Database

feedback_bp = Blueprint('feedback', __name__)


def init_feedback_routes(database: Database):
    """Initialize feedback routes with database dependency.
    
    Args:
        database: Database service instance
    """
    @feedback_bp.route("/feedback", methods=["POST"])
    def feedback():
        from monitoring_module import tracer
        
        with tracer.start_as_current_span('api/feedback'):
            authorization_header = request.headers.get("Authorization")
            
            if authorization_header is None:
                return jsonify({"message": "Token is missing"}), 401
            
            if authorization_header:
                token_data = token_process(authorization_header)
                
                if token_data[1] == 401 or token_data[1] == 403:
                    return (token_data[0].get_json()), token_data[1]
            
            user_feedback = request.get_json()
            conversation_id = request.args.get("conversation-id")
            chat_id = request.args.get("chat-id")
            reaction = user_feedback.get("reaction")
            response = user_feedback.get("response")
            user_email = token_data[0]["email"]
            
            query_time = datetime.utcnow()
            
            feedback_json = {
                "query_time": query_time,
                "conversation_id": conversation_id,
                "chat_id": chat_id,
                "reaction": reaction,
                "response": response,
            }
            
            database.insert_feedback(feedback_json, user_email)
            return {}
    
    return feedback_bp
