---
name: Modular Flask Refactoring
overview: Refactor the monolithic 7436-line app.py into a clean, modular Flask application following SOLID principles and Clean Architecture. Split into routes, services, tools, models, and utilities with proper dependency injection and type hints.
todos:
  - id: config-settings
    content: Create app/config/settings.py - Move all environment variables and configuration logic
    status: completed
  - id: app-factory
    content: Create app/__init__.py with create_app() factory function - Initialize Flask, CORS, SocketIO, register blueprints
    status: completed
    dependencies:
      - config-settings
  - id: milvus-utils
    content: Create app/utils/milvus_utils.py - Move Milvus normalization functions and collection mapping
    status: completed
  - id: gcp-storage
    content: Create app/utils/gcp_storage.py - Move GCP Storage operations and transcript file handling
    status: completed
  - id: auth-utils
    content: Create app/utils/auth.py - Move JWT token processing and create auth decorator
    status: completed
  - id: database-models
    content: Create app/models/database.py - Move all MongoDB CRUD operations with dependency injection
    status: completed
  - id: llm-factory
    content: Create app/services/llm_factory.py - Centralize LLM provider configuration (OpenAI/Anthropic)
    status: completed
    dependencies:
      - config-settings
  - id: tools-knowledge-base
    content: Create app/tools/knowledge_base_tool.py - Extract Knowledge Base tool with DI
    status: completed
    dependencies:
      - llm-factory
  - id: tools-user-lookup
    content: Create app/tools/user_lookup_tool.py - Extract User Lookup tool with DI
    status: completed
    dependencies:
      - database-models
  - id: tools-transcript-extractor
    content: Create app/tools/transcript_extractor_tool.py - Extract transcript extractor tool with DI
    status: completed
    dependencies:
      - llm-factory
  - id: service-agent
    content: Create app/services/agent_service.py - Move agent orchestration logic with DI
    status: completed
    dependencies:
      - tools-knowledge-base
      - tools-user-lookup
      - llm-factory
  - id: service-rag
    content: Create app/services/rag_service.py - Move RAG chain creation and retrieval logic with DI
    status: completed
    dependencies:
      - llm-factory
      - milvus-utils
  - id: service-memory
    content: Create app/services/memory_service.py - Move memory management logic with DI
    status: completed
    dependencies:
      - database-models
  - id: service-transcript
    content: Create app/services/transcript_service.py - Move transcript processing functions with DI
    status: completed
    dependencies:
      - llm-factory
      - gcp-storage
  - id: service-claims
    content: Create app/services/claims_service.py - Move claims processing functions with DI
    status: completed
    dependencies:
      - service-rag
  - id: route-health
    content: Create app/routes/health.py - Move /health endpoint to blueprint
    status: completed
    dependencies:
      - app-factory
  - id: route-chat
    content: Create app/routes/chat.py - Move /start and /calls/start endpoints with service injection
    status: completed
    dependencies:
      - service-rag
      - service-agent
      - service-memory
      - auth-utils
  - id: route-transcripts
    content: Create app/routes/transcripts.py - Move /transcripts/* endpoints with service injection
    status: completed
    dependencies:
      - service-transcript
      - gcp-storage
  - id: route-calls
    content: Create app/routes/calls.py - Move /calls/transcripts endpoint
    status: completed
    dependencies:
      - database-models
  - id: route-claims
    content: Create app/routes/claims.py - Move /claims/followup endpoint with service injection
    status: completed
    dependencies:
      - service-claims
  - id: route-feedback
    content: Create app/routes/feedback.py - Move /feedback endpoint with service injection
    status: completed
    dependencies:
      - database-models
      - auth-utils
  - id: route-webhook
    content: Create app/routes/webhook.py - Move /webhook endpoint with SocketIO integration
    status: completed
    dependencies:
      - app-factory
  - id: route-conversation
    content: Create app/routes/conversation.py - Move conversation management endpoints
    status: completed
    dependencies:
      - database-models
      - auth-utils
  - id: route-referred-clauses
    content: Create app/routes/referred_clauses.py - Move /referred-clauses endpoint
    status: completed
    dependencies:
      - service-claims
  - id: register-blueprints
    content: Create app/routes/__init__.py - Register all blueprints and setup dependencies
    status: completed
    dependencies:
      - route-health
      - route-chat
      - route-transcripts
      - route-calls
      - route-claims
      - route-feedback
      - route-webhook
      - route-conversation
      - route-referred-clauses
  - id: add-type-hints
    content: Add Python type hints to all new functions and classes
    status: pending
    dependencies:
      - service-agent
      - service-rag
      - service-transcript
      - service-claims
  - id: remove-unused-code
    content: Identify and remove unused imports, variables, and dead code
    status: pending
    dependencies:
      - register-blueprints
  - id: update-entry-point
    content: Update entry point (run.py or app.py) to use create_app() factory
    status: completed
    dependencies:
      - app-factory
      - register-blueprints
---

# Flask Application Refactoring Plan

## Overview

Transform the monolithic `app.py` (7436 lines) into a modular, production-grade Flask application using Application Factory pattern, Blueprints, and proper separation of concerns.

## Current State Analysis

### Key Issues Identified:

1. **Single monolithic file** with 7436 lines mixing routes, business logic, and infrastructure
2. **Global state** (mongo_client, db, embed, gcs_fs, handler, socketio, app)
3. **No dependency injection** - services directly access globals
4. **Circular import risk** between `app.py` and `live_copilot.py`
5. **Mixed concerns** - routes contain business logic, database operations, and LLM calls
6. **No type hints** on most functions
7. **Unused code** - commented sections, duplicate functions

### Components to Extract:

- **20+ Flask routes** across multiple domains (chat, transcripts, calls, claims, feedback, webhook)
- **LangChain agents** (Search/Infer modes with Knowledge Base and User Lookup tools)
- **RAG services** (Milvus vector DB operations, retrieval chains)
- **Transcript processing** (question extraction, metadata extraction, GCP storage)
- **MongoDB operations** (CRUD for feedback, Q&A, conversations, transcripts)
- **WebSocket handlers** (SocketIO for live copilot)
- **Authentication** (JWT token verification)
- **Configuration** (environment variables, Milvus collection mapping)

## Proposed File Structure

```
contract-pdf-qna/
├── app/
│   ├── __init__.py                 # Flask app factory (create_app)
│   ├── routes/
│   │   ├── __init__.py            # Blueprint registration
│   │   ├── health.py               # /health endpoint
│   │   ├── chat.py                 # /start, /calls/start endpoints
│   │   ├── transcripts.py          # /transcripts/* endpoints
│   │   ├── calls.py                # /calls/transcripts endpoint
│   │   ├── claims.py               # /claims/followup endpoint
│   │   ├── feedback.py             # /feedback endpoint
│   │   ├── webhook.py              # /webhook endpoint
│   │   ├── conversation.py         # /conversation/*, /history, /sidebar, /delete endpoints
│   │   └── referred_clauses.py     # /referred-clauses endpoint
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_factory.py          # LLM provider configuration (OpenAI/Anthropic)
│   │   ├── rag_service.py          # RAG operations (RetrievalQA, vector DB)
│   │   ├── agent_service.py        # LangChain agent orchestration
│   │   ├── transcript_service.py   # Transcript processing, question extraction
│   │   ├── claims_service.py       # Claims processing, decision generation
│   │   └── memory_service.py        # Conversation memory management
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── knowledge_base_tool.py  # Knowledge Base tool for agents
│   │   ├── user_lookup_tool.py     # User Lookup tool for agents
│   │   └── transcript_extractor_tool.py  # Transcript question extractor tool
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py             # MongoDB models and operations
│   │   └── schemas.py              # Pydantic schemas for validation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── auth.py                 # JWT authentication utilities
│   │   ├── milvus_utils.py         # Milvus collection mapping, normalization
│   │   ├── gcp_storage.py          # GCP Storage operations (fsspec/gcsfs)
│   │   ├── validators.py           # Input validation functions
│   │   └── tracing.py              # OpenTelemetry tracing utilities
│   └── config/
│       ├── __init__.py
│       └── settings.py             # Configuration management (env vars)
├── live_copilot.py                 # Keep separate (already modular)
├── monitoring_module.py            # Keep as-is
├── token_module.py                # Keep as-is
├── config.py                      # Keep for backward compatibility
└── app.py                         # DEPRECATED - will be replaced by app/__init__.py
```

## Implementation Steps

### Phase 1: Foundation (Configuration & Core Infrastructure)

1. **Create `app/config/settings.py`**

   - Move all environment variable loading from `app.py`
   - Create `Settings` class with type hints
   - Centralize configuration access

2. **Create `app/__init__.py` (Application Factory)**

   - Implement `create_app(config_name: str = None)` function
   - Initialize Flask app, CORS, SocketIO
   - Register blueprints
   - Setup database connections (MongoDB, Milvus, GCP)
   - Return configured app instance

3. **Create `app/utils/milvus_utils.py`**

   - Move `normalize_contract_type()`, `normalize_plan_for_milvus()`, `normalize_state_for_milvus()`
   - Move `CLEAR_STATE_ALIASES` constant
   - Create `get_milvus_collection_name()` function

4. **Create `app/utils/gcp_storage.py`**

   - Move GCP Storage initialization logic
   - Move `list_transcript_files_gcp()`, `read_transcript_file_gcp()`
   - Move `extract_transcript_metadata()`
   - Handle SSL certificate configuration

5. **Create `app/utils/auth.py`**

   - Move `token_process()` function
   - Create authentication decorator for routes

### Phase 2: Database Layer

6. **Create `app/models/database.py`**

   - Create `Database` class with dependency injection
   - Move MongoDB CRUD operations:
     - `insert_feedback()`, `read_feedback()`, `update_feedback()`, `delete_feedback()`
     - `insert_qna()`, `read_qna()`, `update_qna()`, `delete_qna()`
     - `update_chat()`
   - Initialize MongoDB client in app factory, inject into services

### Phase 3: LLM & Services Layer

7. **Create `app/services/llm_factory.py`**

   - Create `LLMFactory` class
   - Methods: `create_chat_llm()`, `create_embedding()`, `create_standalone_llm()`
   - Support OpenAI (current) and prepare for Anthropic
   - Cache LLM instances per configuration

8. **Create `app/tools/knowledge_base_tool.py`**

   - Extract Knowledge Base tool creation logic
   - Accept retriever via dependency injection

9. **Create `app/tools/user_lookup_tool.py`**

   - Extract User Lookup tool creation logic
   - Accept MongoDB client via dependency injection

10. **Create `app/tools/transcript_extractor_tool.py`**

    - Extract transcript question extractor tool logic
    - Accept LLM via dependency injection

11. **Create `app/services/agent_service.py`**

    - Move `input_prompt()` function
    - Create `AgentService` class with methods:
      - `create_agent()` - Initialize agent with tools
      - `run_agent()` - Execute agent with query
    - Accept LLM, tools, memory via dependency injection

12. **Create `app/services/rag_service.py`**

    - Move RAG chain creation logic
    - Create `RAGService` class:
      - `create_retrieval_qa()` - For Search mode
      - `create_retriever()` - Vector DB retriever
      - `get_relevant_documents()` - Wrapper for `relevant_docs()`
    - Accept vector DB, LLM via dependency injection

13. **Create `app/services/memory_service.py`**

    - Move memory management logic
    - Create `MemoryService` class:
      - `load_conversation_memory()` - Load from MongoDB
      - `create_motorhead_memory()` - For Infer mode
      - `create_buffer_memory()` - For simple cases

14. **Create `app/services/transcript_service.py`**

    - Move transcript processing functions:
      - `extract_relevant_customer_questions()`
      - `extract_questions_with_agent()`
      - `extract_atomic_questions()`
      - `filter_relevant_customer_questions()`
    - Create `TranscriptService` class with dependency injection

15. **Create `app/services/claims_service.py`**

    - Move claims-related functions:
      - `_build_claims_case_context_for_llm()`
      - `_retrieve_policy_chunks_for_claims()`
      - `generate_claim_decision_from_chunks()`
      - `_format_claim_decision_for_chat()`
    - Create `ClaimsService` class

### Phase 4: Routes Layer

16. **Create `app/routes/health.py`**

    - Move `/health` endpoint
    - Create blueprint

17. **Create `app/routes/chat.py`**

    - Move `/start` and `/calls/start` endpoints
    - Inject services (RAGService, AgentService, MemoryService)
    - Extract business logic to service layer

18. **Create `app/routes/transcripts.py`**

    - Move `/transcripts/*` endpoints
    - Inject TranscriptService, GCPStorageService

19. **Create `app/routes/calls.py`**

    - Move `/calls/transcripts` endpoint

20. **Create `app/routes/claims.py`**

    - Move `/claims/followup` endpoint
    - Inject ClaimsService

21. **Create `app/routes/feedback.py`**

    - Move `/feedback` endpoint
    - Inject Database service

22. **Create `app/routes/webhook.py`**

    - Move `/webhook` endpoint
    - Handle SocketIO integration
    - Integrate with live_copilot module

23. **Create `app/routes/conversation.py`**

    - Move conversation management endpoints:
      - `/history`, `/sidebar`, `/delete`, `/edit-conversation-name`
      - `/conversation/authorize`, `/conversation/status`, `/conversation/close`
    - Inject Database service

24. **Create `app/routes/referred_clauses.py`**

    - Move `/referred-clauses` endpoint

25. **Create `app/routes/__init__.py`**

    - Register all blueprints
    - Setup route dependencies

### Phase 5: WebSocket Integration

26. **Update `app/__init__.py`**

    - Initialize SocketIO in app factory
    - Register SocketIO event handlers:
      - `on_connect`, `on_join_session`, `copilot_enable`, `copilot_disable`

### Phase 6: Cleanup & Type Hints

27. **Add Type Hints**

    - Add type hints to all new functions
    - Use `typing` module (Dict, List, Optional, Tuple)
    - Create Pydantic models in `app/models/schemas.py` for request/response validation

28. **Remove Unused Code**

    - Identify and remove commented code
    - Remove duplicate functions
    - Clean up unused imports

29. **Fix Circular Imports**

    - Ensure import hierarchy: Routes → Services → Tools → Utils
    - Use lazy imports where necessary (e.g., live_copilot)

30. **Update Entry Point**

    - Create new `run.py` or update existing entry point
    - Use `create_app()` to initialize application
    - Update Dockerfile if needed

## Dependency Injection Pattern

### Example: RAGService

```python
class RAGService:
    def __init__(
        self,
        vector_db: Milvus,
        llm: ChatOpenAI,
        embedding: OpenAIEmbeddings,
        config: Settings
    ):
        self.vector_db = vector_db
        self.llm = llm
        self.embedding = embedding
        self.config = config
    
    def create_retriever(self, collection_name: str, k: int = 25):
        # Implementation
        pass
```

### Example: Route with DI

```python
@chat_bp.route("/start", methods=["POST"])
def start(
    rag_service: RAGService = Depends(get_rag_service),
    agent_service: AgentService = Depends(get_agent_service)
):
    # Route implementation
    pass
```

## Import Strategy

### Absolute Imports Only

- Use: `from app.services.rag_service import RAGService`
- Avoid: `from ..services.rag_service import RAGService` (relative imports)

### Import Hierarchy

1. **Routes** import from Services
2. **Services** import from Tools, Models, Utils
3. **Tools** import from Utils, Models
4. **Utils** import only standard library or external packages

## Testing Strategy

1. **Unit Tests** for each service/tool in isolation
2. **Integration Tests** for routes with mocked services
3. **E2E Tests** for critical flows (chat, transcript processing)

## Migration Path

1. **Create new structure** alongside existing `app.py`
2. **Migrate one route at a time** (start with `/health`)
3. **Test each migration** before proceeding
4. **Update imports** incrementally
5. **Deprecate old `app.py`** once all routes migrated
6. **Remove old code** after verification

## Risk Mitigation

1. **Maintain backward compatibility** during migration
2. **Keep existing functionality identical** - only reorganize code
3. **Test thoroughly** before removing old code
4. **Document changes** in migration notes

## Success Criteria

- ✅ All routes moved to blueprints
- ✅ All business logic in services
- ✅ All tools isolated with DI
- ✅ No circular imports
- ✅ Type hints on all functions
- ✅ Unused code removed
- ✅ Application factory pattern implemented
- ✅ All tests passing
- ✅ No functionality regressions