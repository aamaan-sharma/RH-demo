"""Flask application factory."""
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from pymongo import MongoClient
from app.config.settings import settings
from app.models.database import Database
from app.services.llm_factory import get_llm_factory
from app.services.rag_service import RAGService
from app.services.agent_service import AgentService
from app.services.memory_service import MemoryService
from app.services.claims_service import ClaimsService
from app.services.transcript_service import TranscriptService
from app.services.transcript_processor_service import TranscriptProcessorService
from app.utils.gcp_storage import GCPStorageService
from app.routes import register_blueprints


def create_app(config_name: str = None) -> Flask:
    """Create and configure Flask application.
    
    Args:
        config_name: Optional configuration name (not used currently)
        
    Returns:
        Configured Flask application instance
    """
    app = Flask(__name__)
    app.secret_key = settings.FLASK_SECRET_KEY
    
    # CORS configuration
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    # SocketIO initialization
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode=settings.SOCKETIO_ASYNC_MODE,
        logger=False,
        engineio_logger=False,
    )
    
    # Initialize MongoDB
    mongo_client = MongoClient(settings.MONGO_URI)
    database = Database(mongo_client, db_name="FrontDoorDB")
    
    # Initialize services
    llm_factory = get_llm_factory()
    embedding = llm_factory.create_embedding()
    
    rag_service = RAGService(llm_factory, embedding)
    agent_service = AgentService(llm_factory, database)
    memory_service = MemoryService(database)
    claims_service = ClaimsService(rag_service, llm_factory)
    transcript_service = TranscriptService(llm_factory)
    transcript_processor_service = TranscriptProcessorService(
        rag_service, claims_service, transcript_service, agent_service
    )
    
    # Store llm_factory for access in routes
    app.config['llm_factory'] = llm_factory
    
    # Initialize GCP Storage
    gcp_storage_service = GCPStorageService(
        bucket_name=settings.GCP_BUCKET_NAME,
        project_id=settings.GCP_PROJECT_ID,
        service_account_path=settings.GCP_SERVICE_ACCOUNT_PATH
    )
    
    # Register blueprints
    register_blueprints(
        app=app,
        socketio=socketio,
        database=database,
        rag_service=rag_service,
        agent_service=agent_service,
        memory_service=memory_service,
        claims_service=claims_service,
        transcript_service=transcript_service,
        transcript_processor_service=transcript_processor_service,
        gcp_storage_service=gcp_storage_service
    )
    
    # SocketIO event handlers
    @socketio.on("connect")
    def on_connect(auth):
        print("Client connected")
    
    @socketio.on("join_session")
    def on_join_session(data):
        session_id = data.get("sessionId")
        if session_id:
            from flask_socketio import join_room
            join_room(session_id)
            print(f"Client joined session: {session_id}")
    
    @socketio.on("copilot_enable")
    def on_copilot_enable(data):
        session_id = data.get("sessionId")
        print(f"Copilot enabled for session: {session_id}")
    
    @socketio.on("copilot_disable")
    def on_copilot_disable(data):
        session_id = data.get("sessionId")
        print(f"Copilot disabled for session: {session_id}")
    
    # Store services in app context for access in routes if needed
    app.config['database'] = database
    app.config['rag_service'] = rag_service
    app.config['agent_service'] = agent_service
    app.config['memory_service'] = memory_service
    app.config['claims_service'] = claims_service
    app.config['transcript_service'] = transcript_service
    app.config['gcp_storage_service'] = gcp_storage_service
    app.config['socketio'] = socketio
    
    return app
