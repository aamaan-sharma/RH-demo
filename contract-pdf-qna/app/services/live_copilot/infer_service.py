"""INFER service initialization for Live Copilot."""
from typing import Optional, Any
from app.config.settings import settings

_INFER_WRAPPER_AVAILABLE = False
_transcript_processor_service = None


def get_infer_wrapper():
    """
    Get or create TranscriptProcessorService instance for INFER processing.
    Uses lazy initialization to avoid circular dependencies.
    """
    global _INFER_WRAPPER_AVAILABLE, _transcript_processor_service
    
    if _transcript_processor_service is not None:
        return _transcript_processor_service.process_live_copilot_question
    
    try:
        # Lazy import to avoid circular dependencies
        from app.services.transcript_processor_service import TranscriptProcessorService
        from app.services.rag_service import RAGService
        from app.services.claims_service import ClaimsService
        from app.services.transcript_service import TranscriptService
        from app.services.agent_service import AgentService
        from app.services.llm_factory import get_llm_factory
        from app.models.database import Database
        from pymongo import MongoClient
        
        # Initialize services (similar to app/__init__.py)
        llm_factory = get_llm_factory()
        embedding = llm_factory.create_embedding()
        
        rag_service = RAGService(llm_factory, embedding)
        database = Database(MongoClient(settings.MONGO_URI), db_name="FrontDoorDB")
        agent_service = AgentService(llm_factory, database)
        claims_service = ClaimsService(rag_service, llm_factory)
        transcript_service = TranscriptService(llm_factory)
        
        _transcript_processor_service = TranscriptProcessorService(
            rag_service=rag_service,
            claims_service=claims_service,
            transcript_service=transcript_service,
            agent_service=agent_service
        )
        
        _INFER_WRAPPER_AVAILABLE = True
        return _transcript_processor_service.process_live_copilot_question
    except Exception as e:
        print(f"[LIVE_COPILOT] Failed to initialize TranscriptProcessorService: {e}")
        import traceback
        traceback.print_exc()
        _INFER_WRAPPER_AVAILABLE = False
        return None
