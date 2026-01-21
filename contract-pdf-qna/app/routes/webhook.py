import json
import os
from datetime import datetime

from flask import Blueprint, jsonify, request

from ..extensions import socketio
from ..services.copilot_service import handle_transcript_event_safe
from ..services.copilot.session import (
    _copilot_enabled_sessions,
    _copilot_session_context,
    _copilot_sessions_lock,
    copilot_session_ttl_seconds,
    flag_enabled,
    get_parent_trace_context,
)
from ..utils.transcript_filters import should_start_copilot

webhook_bp = Blueprint("webhook", __name__)

@webhook_bp.route("/webhook", methods=["POST"])
def transcript_event():
    data = request.get_json()
    if not data:
        return jsonify({"error": "invalid payload"}), 400

    session_id = data.get("sessionId")
    if not session_id:
        return jsonify({"error": "sessionId is required"}), 400

    include_payloads = str(os.getenv("OTEL_TRACE_INCLUDE_PAYLOADS", "0") or "").strip().lower() in (
        "1",
        "true",
        "yes",
        "y",
        "on",
    )
    if include_payloads:
        try:
            limit = int(os.getenv("OTEL_TRACE_PAYLOAD_PREVIEW_CHARS", "500") or 500)
        except Exception:
            limit = 500
        print("🔴 TRANSCRIPT RECEIVED (payload):", json.dumps(data, indent=2, default=str)[: max(0, limit)])
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

    if flag_enabled("ENABLE_LIVE_COPILOT", "0") and should_start_copilot(data):
        def process_copilot_async():
            try:
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

                parent_ctx = get_parent_trace_context(session_id)
                copilot_result = handle_transcript_event_safe(copilot_payload, parent_ctx)
                if copilot_result:
                    if include_payloads:
                        try:
                            limit = int(os.getenv("OTEL_TRACE_PAYLOAD_PREVIEW_CHARS", "500") or 500)
                        except Exception:
                            limit = 500
                        print(
                            "🟢 COPILOT SUGGESTION (payload):",
                            json.dumps(copilot_result, indent=2, default=str)[: max(0, limit)],
                        )
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
            except Exception as exc:  # pragma: no cover - non-blocking path
                print(f"⚠️ Copilot processing error (non-blocking): {exc}")
        import threading
        threading.Thread(target=process_copilot_async, daemon=True).start()

    return jsonify({"ok": True}), 200
