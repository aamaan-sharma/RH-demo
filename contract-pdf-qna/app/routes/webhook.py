"""Webhook endpoint for transcript events."""
import os
import json
from flask import Blueprint, request, jsonify
from flask_socketio import SocketIO
from app.config.settings import settings
from app.models.database import Database

webhook_bp = Blueprint('webhook', __name__)


def init_webhook_routes(database: Database, socketio: SocketIO):
    """Initialize webhook routes with dependencies.
    
    Args:
        database: Database service instance
        socketio: SocketIO instance
    """
    @webhook_bp.route("/webhook", methods=["POST"])
    def transcript_event():
        data = request.get_json()
        if not data:
            return jsonify({"error": "invalid payload"}), 400
        
        session_id = data.get("sessionId")
        if not session_id:
            return jsonify({"error": "sessionId is required"}), 400
        
        # Deduplication check
        dedup_query = {
            "sessionId": data.get("sessionId"),
            "speaker": data.get("speaker"),
            "text": data.get("text"),
            "beginOffsetMillis": data.get("beginOffsetMillis"),
            "endOffsetMillis": data.get("endOffsetMillis"),
        }
        
        if database.transcripts_collection:
            existing = database.transcripts_collection.find_one(dedup_query)
            if existing:
                return jsonify({"ok": True, "duplicate": True}), 200
            
            transcript_doc = {
                "sessionId": data["sessionId"],
                "contactId": data.get("contactId"),
                "speaker": data.get("speaker"),
                "text": data.get("text"),
                "isPartial": data.get("isPartial", True),
                "beginOffsetMillis": data.get("beginOffsetMillis"),
                "endOffsetMillis": data.get("endOffsetMillis"),
                "createdAt": data.get("createdAt")
            }
            
            database.transcripts_collection.insert_one(transcript_doc)
        
        # Broadcast to UI via websocket
        include_payloads = settings.OTEL_TRACE_INCLUDE_PAYLOADS
        if include_payloads:
            try:
                limit = settings.OTEL_TRACE_PAYLOAD_PREVIEW_CHARS or 500
                print("🔴 TRANSCRIPT RECEIVED (payload):", json.dumps(data, indent=2, default=str)[:max(0, limit)])
            except Exception:
                pass
        else:
            try:
                txt = str(data.get("text") or "")
                print(
                    "🔴 TRANSCRIPT RECEIVED (summary): "
                    f"sessionId={data.get('sessionId')}, speaker={data.get('speaker')}, "
                    f"isPartial={bool(data.get('isPartial', True))}, text_len={len(txt)}"
                )
            except Exception:
                pass
        
        socketio.emit("transcript_update", data)
        socketio.emit("transcript_update", data, room=data["sessionId"])
        
        # Live Copilot integration
        if (
            settings.ENABLE_LIVE_COPILOT
            and not data.get("isPartial", True)
        ):
            try:
                from app.services.live_copilot import handle_transcript_event
                from app.utils.tracing import get_or_create_session_trace_context
                
                copilot_payload = {
                    "sessionId": session_id,
                    "contactId": data.get("contactId"),
                    "speaker": data.get("speaker"),
                    "text": data.get("text"),
                    "isPartial": data.get("isPartial", False),
                    "beginOffsetMillis": data.get("beginOffsetMillis"),
                    "endOffsetMillis": data.get("endOffsetMillis"),
                    "phoneNumber": data.get("phoneNumber") or data.get("phone"),
                    "state": data.get("state"),
                    "contractType": data.get("contractType"),
                    "plan": data.get("plan"),
                }
                
                parent_ctx = get_or_create_session_trace_context(session_id)
                copilot_result = handle_transcript_event(copilot_payload, parent_context=parent_ctx)
                
                if copilot_result:
                    if include_payloads:
                        try:
                            limit = settings.OTEL_TRACE_PAYLOAD_PREVIEW_CHARS or 500
                            print(
                                "🟢 COPILOT SUGGESTION (payload):",
                                json.dumps(copilot_result, indent=2, default=str)[:max(0, limit)],
                            )
                        except Exception:
                            pass
                    else:
                        try:
                            cards = copilot_result.get("cards") or []
                            print(
                                "🟢 COPILOT SUGGESTION (summary): "
                                f"sessionId={copilot_result.get('sessionId')}, intent={copilot_result.get('intent')}, cards={len(cards)}"
                            )
                        except Exception:
                            pass
                    
                    socketio.emit("suggestion_update", copilot_result)
                    socketio.emit("suggestion_update", copilot_result, room=session_id)
            except Exception as e:
                print(f"⚠️ Copilot processing error (non-blocking): {e}")
                import traceback
                traceback.print_exc()
        
        return jsonify({"ok": True}), 200
    
    return webhook_bp
