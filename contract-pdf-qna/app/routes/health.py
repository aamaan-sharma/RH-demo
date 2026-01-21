from datetime import datetime

from flask import Blueprint, jsonify, make_response

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health():
    """
    Lightweight liveness endpoint.
    Returns 200 if the Flask process is running and able to serve requests.
    """
    payload = {
        "status": "ok",
        "service": "contract-pdf-qna",
        "time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    resp = make_response(jsonify(payload), 200)
    resp.headers["Cache-Control"] = "no-store"
    return resp
