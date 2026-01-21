import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

from .config import Settings
from .extensions import init_extensions, socketio, ensure_gcs_fs
from .routes.health import health_bp
from .routes.webhook import webhook_bp
from .routes.transcripts import transcripts_bp
from .routes import socket_handlers  # noqa: F401 - side-effect import to register handlers
from .services.transcript_service import list_transcript_files_gcp, GCP_BUCKET_NAME
from .extensions import gcs_fs


def create_app(settings: Settings | None = None) -> Flask:
    """
    Application factory for the modular Flask backend.
    Loads environment, initializes extensions, and registers blueprints.
    """
    load_dotenv()
    settings = settings or Settings.from_env()

    app = Flask(__name__)
    app.config["SECRET_KEY"] = settings.flask_secret or "dev-secret"
    app.config["JSON_SORT_KEYS"] = False

    # Initialize CORS and shared extensions (SocketIO, Mongo, GCS, embeddings, tracing).
    CORS(app, resources={r"/*": {"origins": "*"}})
    init_extensions(app, settings)

    # Blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(webhook_bp)
    app.register_blueprint(transcripts_bp)

    # Basic before_request to support legacy OPTIONS preflight handling.
    @app.before_request
    def _allow_preflight():
        pass

    return app


# Ensure GCS filesystem is available for modules/tests that import without create_app.
ensure_gcs_fs()


__all__ = [
    "create_app",
    "socketio",
    "Settings",
    "list_transcript_files_gcp",
    "gcs_fs",
    "GCP_BUCKET_NAME",
]
