import os
import ssl
from typing import Optional

from flask_socketio import SocketIO
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Milvus

try:
    import eventlet  # noqa: F401

    eventlet.monkey_patch()
    _DEFAULT_ASYNC_MODE = "eventlet"
except Exception:
    _DEFAULT_ASYNC_MODE = "threading"

try:
    from monitoring_module import tracer, q_monitor, llm_trace_to_jaeger  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tracer = None
    q_monitor = None
    llm_trace_to_jaeger = None

socketio: SocketIO = SocketIO(cors_allowed_origins="*")
mongo_client: Optional[MongoClient] = None
db = None
embed: Optional[OpenAIEmbeddings] = None
milvus_client: Optional[Milvus] = None
gcs_fs = None
ssl_context: Optional[ssl.SSLContext] = None


def init_extensions(app, settings):
    """
    Initialize shared extensions. This keeps globals centralized and testable.
    """
    global mongo_client, db, embed, milvus_client, gcs_fs, ssl_context

    async_mode = settings.async_mode or os.getenv("SOCKETIO_ASYNC_MODE") or _DEFAULT_ASYNC_MODE
    socketio.init_app(app, cors_allowed_origins="*", async_mode=async_mode, manage_session=True)

    if settings.mongo_uri:
        mongo_client = MongoClient(settings.mongo_uri, unicode_decode_error_handler="ignore")
        db = mongo_client["FrontDoorDB"]

    if settings.openai_api_key:
        embed = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=settings.openai_api_key)

    # Configure SSL certificates for macOS compatibility if certifi is available.
    try:
        import certifi

        cert_path = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", cert_path)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", cert_path)
        os.environ.setdefault("AIOHTTP_CA_BUNDLE", cert_path)
        ssl_context = ssl.create_default_context(cafile=cert_path)
    except Exception:
        ssl_context = None

    ensure_gcs_fs(settings)


def ensure_gcs_fs(settings=None):
    """
    Lazily initialize the shared GCS filesystem using fsspec.
    Safe to call multiple times; returns None if unavailable.
    """
    global gcs_fs
    if gcs_fs is not None:
        return gcs_fs
    try:
        import fsspec

        project_id = None
        if settings is not None:
            project_id = getattr(settings, "gcp_project_id", None)
        project_id = project_id or os.getenv("GCP_PROJECT_ID", "generative-ai-390411")
        gcs_fs = fsspec.filesystem("gcs", project=project_id)
        return gcs_fs
    except Exception:
        gcs_fs = None
        return None


__all__ = [
    "socketio",
    "mongo_client",
    "db",
    "embed",
    "milvus_client",
    "gcs_fs",
    "ssl_context",
    "tracer",
    "q_monitor",
    "llm_trace_to_jaeger",
    "init_extensions",
]
