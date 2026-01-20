"""Blueprint registration and route initialization."""
from flask import Flask
from flask_socketio import SocketIO
from app.routes import health, feedback, conversation, calls, claims, referred_clauses, transcripts, webhook, chat
from app.services.rag_service import RAGService
from app.services.agent_service import AgentService
from app.services.memory_service import MemoryService
from app.services.claims_service import ClaimsService
from app.services.transcript_service import TranscriptService
from app.services.transcript_processor_service import TranscriptProcessorService
from app.services.llm_factory import get_llm_factory
from app.models.database import Database
from app.utils.gcp_storage import GCPStorageService
from app.config.settings import settings


def register_blueprints(
    app: Flask,
    socketio: SocketIO,
    database: Database,
    rag_service: RAGService,
    agent_service: AgentService,
    memory_service: MemoryService,
    claims_service: ClaimsService,
    transcript_service: TranscriptService,
    transcript_processor_service: TranscriptProcessorService,
    gcp_storage_service: GCPStorageService
):
    """Register all blueprints with the Flask app.
    
    Args:
        app: Flask application instance
        socketio: SocketIO instance
        database: Database service instance
        rag_service: RAGService instance
        agent_service: AgentService instance
        memory_service: MemoryService instance
        claims_service: ClaimsService instance
        transcript_service: TranscriptService instance
        transcript_processor_service: TranscriptProcessorService instance
        gcp_storage_service: GCPStorageService instance
    """
    # Register simple blueprints
    app.register_blueprint(health.health_bp)
    
    # Register blueprints with dependencies
    app.register_blueprint(feedback.init_feedback_routes(database))
    app.register_blueprint(conversation.init_conversation_routes(database))
    app.register_blueprint(calls.init_calls_routes(database))
    app.register_blueprint(claims.init_claims_routes(claims_service, database))
    app.register_blueprint(referred_clauses.init_referred_clauses_routes(claims_service, database))
    app.register_blueprint(transcripts.init_transcripts_routes(
        transcript_service, gcp_storage_service, database, transcript_processor_service
    ))
    app.register_blueprint(webhook.init_webhook_routes(database, socketio))
    app.register_blueprint(chat.init_chat_routes(rag_service, agent_service, memory_service, database))
