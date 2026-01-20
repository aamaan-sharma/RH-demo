"""Authentication utilities for JWT token verification."""
from functools import wraps
from typing import Tuple, Dict, Any, Optional
from flask import request, jsonify
from oauth2client import client
from app.config.settings import settings


def token_process(authorization_header: str) -> Tuple[Dict[str, Any], int]:
    """Process and verify JWT token from authorization header.
    
    Args:
        authorization_header: Authorization header value (e.g., "Bearer <token>")
        
    Returns:
        Tuple of (token_data_dict, status_code)
        - token_data_dict: Decoded token data if valid, error message dict if invalid
        - status_code: 200 if valid, 401/403 if invalid
    """
    parts = authorization_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        bearer_token = parts[1]
        try:
            token = client.verify_id_token(bearer_token, settings.JWT_AUDIENCE)
            return token, 200
        except Exception as e:
            error_msg = str(e).split(",")[0] if "," in str(e) else str(e)
            if error_msg == "Token used too late":
                return jsonify({"message": "Token has expired"}), 403
            else:
                return jsonify({"message": "Token is invalid"}), 403
    else:
        return jsonify({"message": "Token is missing"}), 401


def require_auth(f):
    """Decorator to require JWT authentication for a route.
    
    Usage:
        @app.route("/protected")
        @require_auth
        def protected_route():
            # token_data is available in request.token_data
            user_email = request.token_data["email"]
            ...
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        authorization_header = request.headers.get("Authorization")
        
        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401
        
        token_data, status_code = token_process(authorization_header)
        
        if status_code != 200:
            return token_data, status_code
        
        # Attach token data to request for use in route handler
        request.token_data = token_data
        return f(*args, **kwargs)
    
    return decorated_function
