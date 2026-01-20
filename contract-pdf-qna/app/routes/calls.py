"""Calls endpoints."""
from datetime import datetime
from flask import Blueprint, request, jsonify
from app.utils.auth import token_process
from app.models.database import Database

calls_bp = Blueprint('calls', __name__)


def init_calls_routes(database: Database):
    """Initialize calls routes with database dependency.
    
    Args:
        database: Database service instance
    """
    @calls_bp.route("/calls/transcripts", methods=["GET"])
    def calls_transcripts():
        authorization_header = request.headers.get("Authorization")
        
        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401
        
        if authorization_header:
            token_data = token_process(authorization_header)
            
            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]
        
        q = request.args.get("q", "").strip()
        status = request.args.get("status", "active").lower()
        page = int(request.args.get("page", 1) or 1)
        page_size = int(request.args.get("pageSize", 20) or 20)
        
        query = {}
        if status and status != "all":
            query["status"] = status
        if q:
            query["name"] = {"$regex": q, "$options": "i"}
        
        skip = (page - 1) * page_size
        
        calls_transcripts_collection = database.db2.calls_transcripts if database.db2 else None
        if not calls_transcripts_collection:
            return jsonify({"items": [], "pagination": {"page": page, "pageSize": page_size, "total": 0, "totalPages": 0}})
        
        total = calls_transcripts_collection.count_documents(query)
        cursor = (
            calls_transcripts_collection.find(query)
            .sort("created_at", -1)
            .skip(skip)
            .limit(page_size)
        )
        
        items = []
        for doc in cursor:
            items.append({
                "id": str(doc["_id"]),
                "name": doc.get("name"),
                "stateName": doc.get("state_name"),
                "contractType": doc.get("contract_type"),
                "planName": doc.get("plan_name"),
                "status": doc.get("status", "active"),
                "createdAt": doc.get("created_at").isoformat() if doc.get("created_at") else None,
                "updatedAt": doc.get("updated_at").isoformat() if doc.get("updated_at") else None,
            })
        
        return jsonify({
            "items": items,
            "pagination": {
                "page": page,
                "pageSize": page_size,
                "total": total,
                "totalPages": (total + page_size - 1) // page_size,
            },
        })
    
    return calls_bp
