from time import time

from flask import session
from flask_socketio import disconnect, join_room

from ..extensions import db, socketio
from ..utils.auth import token_process
from ..services.copilot.session import (
    _copilot_enabled_sessions,
    _copilot_session_context,
    _copilot_sessions_lock,
    copilot_session_ttl_seconds,
)


@socketio.on("connect")
def on_connect(auth):
    token = None
    if isinstance(auth, dict):
        token = auth.get("token")

    if not token:
        print("❌ No JWT in connect auth")
        disconnect()
        return

    token_data, status = token_process(f"Bearer {token}")
    if status in (401, 403):
        print("❌ Invalid JWT in connect auth")
        disconnect()
        return

    user_email = token_data.get("email")
    if not user_email:
        print("❌ Email missing in JWT")
        disconnect()
        return

    session["user_email"] = user_email
    print(f"✅ Socket connected user: {user_email}")


@socketio.on("join_session")
def on_join_session(data):
    session_id = (data or {}).get("sessionId")
    if not session_id:
        return

    user_email = session.get("user_email")
    if not user_email:
        print("❌ No user_email in socket session")
        return

    join_room(session_id)
    if db:
        db["sessions"].update_one(
            {"contactId": session_id},
            {
                "$setOnInsert": {
                    "contactId": session_id,
                    "email": user_email,
                    "createdAt": time(),
                }
            },
            upsert=True,
        )
    print(f"✅ Session mapped: {session_id} → {user_email}")


@socketio.on("copilot_enable")
def on_copilot_enable(data):
    """Enable Live Copilot for a session when call connects."""
    session_id = data.get("sessionId")
    if session_id:
        with _copilot_sessions_lock:
            _copilot_enabled_sessions[session_id] = time() + copilot_session_ttl_seconds()
            _copilot_session_context[session_id] = {
                "contractType": data.get("contractType", ""),
                "selectedPlan": data.get("selectedPlan", ""),
                "selectedState": data.get("selectedState", ""),
            }
        print(f"🟢 COPILOT ENABLED for session: {session_id}")
        socketio.emit("copilot_status", {"sessionId": session_id, "enabled": True}, room=session_id)


@socketio.on("copilot_disable")
def on_copilot_disable(data):
    """Disable Live Copilot when call ends."""
    session_id = data.get("sessionId")
    if session_id:
        with _copilot_sessions_lock:
            _copilot_enabled_sessions.pop(session_id, None)
            _copilot_session_context.pop(session_id, None)
        print(f"🔴 COPILOT DISABLED for session: {session_id}")
        socketio.emit("copilot_status", {"sessionId": session_id, "enabled": False}, room=session_id)
