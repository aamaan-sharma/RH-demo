"""Conversation management endpoints."""
from flask import Blueprint, request, jsonify
from bson.objectid import ObjectId
from app.utils.auth import token_process
from app.models.database import Database

conversation_bp = Blueprint('conversation', __name__)


def init_conversation_routes(database: Database):
    """Initialize conversation routes with database dependency.
    
    Args:
        database: Database service instance
    """
    @conversation_bp.route("/history", methods=["GET"])
    def chat_history():
        authorization_header = request.headers.get("Authorization")
        
        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401
        
        if authorization_header:
            token_data = token_process(authorization_header)
            
            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]
        
        user_email = token_data[0]["email"]
        qna_collection_user = f"chats_{user_email}"
        qna_collection = database.db[qna_collection_user]
        
        conversations = list(qna_collection.find().sort("query_time", -1).limit(50))
        
        result = []
        for conv in conversations:
            result.append({
                "conversation_id": str(conv["_id"]),
                "conversation_name": conv.get("conversation_name", ""),
                "query_time": conv.get("query_time").isoformat() if conv.get("query_time") else None,
                "status": conv.get("status", "active"),
            })
        
        return jsonify(result)
    
    @conversation_bp.route("/sidebar", methods=["GET"])
    def sidebar_history():
        authorization_header = request.headers.get("Authorization")
        
        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401
        
        if authorization_header:
            token_data = token_process(authorization_header)
            
            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]
        
        user_email = token_data[0]["email"]
        qna_collection_user = f"chats_{user_email}"
        qna_collection = database.db[qna_collection_user]
        
        conversations = list(qna_collection.find().sort("query_time", -1).limit(20))
        
        result = []
        for conv in conversations:
            result.append({
                "conversation_id": str(conv["_id"]),
                "conversation_name": conv.get("conversation_name", ""),
                "query_time": conv.get("query_time").isoformat() if conv.get("query_time") else None,
            })
        
        return jsonify(result)
    
    @conversation_bp.route("/delete", methods=["DELETE"])
    def delete_conversation():
        authorization_header = request.headers.get("Authorization")
        
        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401
        
        if authorization_header:
            token_data = token_process(authorization_header)
            
            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]
        
        user_email = token_data[0]["email"]
        conversation_id = request.args.get("conversation-id")
        
        if not conversation_id:
            return jsonify({"error": "conversation-id is required"}), 400
        
        qna_collection_user = f"chats_{user_email}"
        qna_collection = database.db[qna_collection_user]
        result = qna_collection.delete_one({"_id": ObjectId(conversation_id)})
        
        return jsonify({"deleted": result.deleted_count > 0})
    
    @conversation_bp.route("/edit-conversation-name", methods=["PATCH"])
    def edit_name():
        authorization_header = request.headers.get("Authorization")
        
        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401
        
        if authorization_header:
            token_data = token_process(authorization_header)
            
            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]
        
        user_email = token_data[0]["email"]
        data = request.get_json()
        conversation_id = data.get("conversation-id")
        new_name = data.get("conversation-name")
        
        if not conversation_id or not new_name:
            return jsonify({"error": "conversation-id and conversation-name are required"}), 400
        
        qna_collection_user = f"chats_{user_email}"
        qna_collection = database.db[qna_collection_user]
        result = qna_collection.update_one(
            {"_id": ObjectId(conversation_id)},
            {"$set": {"conversation_name": new_name}}
        )
        
        return jsonify({"updated": result.modified_count > 0})
    
    @conversation_bp.route("/conversation/authorize", methods=["PATCH"])
    def authorize_conversation_answer():
        authorization_header = request.headers.get("Authorization")
        
        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401
        
        if authorization_header:
            token_data = token_process(authorization_header)
            
            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]
        
        user_email = token_data[0]["email"]
        data = request.get_json()
        conversation_id = data.get("conversation-id")
        authorized_answer = data.get("authorizedFinalAnswer")
        
        if not conversation_id:
            return jsonify({"error": "conversation-id is required"}), 400
        
        qna_collection_user = f"chats_{user_email}"
        qna_collection = database.db[qna_collection_user]
        result = qna_collection.update_one(
            {"_id": ObjectId(conversation_id)},
            {"$set": {"authorized_final_answer": authorized_answer}}
        )
        
        return jsonify({"updated": result.modified_count > 0})
    
    @conversation_bp.route("/conversation/status", methods=["PATCH"])
    def update_conversation_status():
        authorization_header = request.headers.get("Authorization")
        
        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401
        
        if authorization_header:
            token_data = token_process(authorization_header)
            
            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]
        
        user_email = token_data[0]["email"]
        data = request.get_json()
        conversation_id = data.get("conversation-id")
        status = data.get("status")
        
        if not conversation_id or not status:
            return jsonify({"error": "conversation-id and status are required"}), 400
        
        qna_collection_user = f"chats_{user_email}"
        qna_collection = database.db[qna_collection_user]
        result = qna_collection.update_one(
            {"_id": ObjectId(conversation_id)},
            {"$set": {"status": status}}
        )
        
        return jsonify({"updated": result.modified_count > 0})
    
    @conversation_bp.route("/conversation/close", methods=["PATCH"])
    def close_conversation():
        authorization_header = request.headers.get("Authorization")
        
        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401
        
        if authorization_header:
            token_data = token_process(authorization_header)
            
            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]
        
        user_email = token_data[0]["email"]
        data = request.get_json()
        conversation_id = data.get("conversation-id")
        case_disposition = data.get("caseDisposition")
        
        if not conversation_id:
            return jsonify({"error": "conversation-id is required"}), 400
        
        qna_collection_user = f"chats_{user_email}"
        qna_collection = database.db[qna_collection_user]
        update_data = {"status": "closed"}
        if case_disposition:
            update_data["case_disposition"] = case_disposition
        
        result = qna_collection.update_one(
            {"_id": ObjectId(conversation_id)},
            {"$set": update_data}
        )
        
        return jsonify({"updated": result.modified_count > 0})
    
    return conversation_bp
