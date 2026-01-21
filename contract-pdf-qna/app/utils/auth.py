import os
from typing import Tuple

from flask import jsonify
from oauth2client import client as oauth_client


def token_process(authorization_header: str, audience: str | None = None) -> Tuple[dict, int]:
    """
    Verify bearer token using oauth2client. Returns (payload, status_code).
    """
    audience = audience or os.getenv("JWT_AUDIENCE")
    parts = authorization_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        bearer_token = parts[1]
        try:
            token = oauth_client.verify_id_token(bearer_token, audience)
            return token, 200
        except Exception as exc:
            if str(exc).split(",")[0] == "Token used too late":
                return jsonify({"message": "Token has expired"}), 403
            return jsonify({"message": "Token is invalid"}), 403
    return jsonify({"message": "Token is missing"}), 401


__all__ = ["token_process"]
