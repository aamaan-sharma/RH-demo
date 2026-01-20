# Migration Notes

## Status: Core Structure Complete ✅

The modular refactoring is **structurally complete** with all major components in place. However, some routes contain simplified implementations that need full migration from `app.py`.

## What's Complete

✅ **Configuration Layer** - Centralized settings management  
✅ **Database Layer** - MongoDB operations with dependency injection  
✅ **Service Layer** - All major services (RAG, Agent, Memory, Claims, Transcript)  
✅ **Tools Layer** - LangChain tools extracted  
✅ **Routes Layer** - All route blueprints created  
✅ **Application Factory** - Flask app factory with dependency injection  
✅ **Blueprint Registration** - Centralized route registration  
✅ **Entry Point** - New `run.py` using application factory  

## What Needs Full Migration

⚠️ **Route Implementations** - Some complex routes still need migration:
- `app/routes/transcripts.py` - Complex processing endpoints still in `app.py`:
  - `/transcripts/process` (POST) - ~600 lines, needs TranscriptProcessorService
  - `/transcripts/process/stream` (POST) - ~600 lines, needs TranscriptProcessorService  
  - `/internal/transcripts/process` (POST) - ~400 lines, needs TranscriptProcessorService
- `app/routes/chat.py` - `/start` and `/calls/start` endpoints fully migrated ✅
- `app/routes/claims.py` - `/claims/followup` endpoint migrated ✅
- `app/routes/webhook.py` - `/webhook` endpoint migrated ✅

✅ **Completed Transcript Routes**:
- `/transcripts` (GET) - List transcripts ✅
- `/transcripts/<filename>` (GET) - Get transcript content ✅
- `/transcripts/dialogue` (POST) - Convert transcript to chat format ✅
- `/transcripts/status` (PATCH) - Update transcript status ✅
- `/transcripts/conversations` (GET) - List transcript conversations ✅
- `/transcripts/conversation/stub` (POST) - Create conversation stub ✅

## Next Steps

1. **Test the application** - Run `python run.py` to verify basic functionality
2. **Migrate remaining routes** - Move full implementations from `app.py` to route modules
3. **Add type hints** - Complete type annotations across all modules
4. **Remove unused code** - Clean up `app.py` after migration verification
5. **Update tests** - Adapt tests to new modular structure

## Running the Application

```bash
# Use the new entry point
python run.py

# Or with gunicorn/uwsgi
gunicorn run:app
```

## Backward Compatibility

The original `app.py` is still present and functional. The new modular structure runs alongside it. Once migration is verified, `app.py` can be deprecated.

## Import Paths

All imports use absolute paths from `app.`:
- ✅ `from app.config.settings import settings`
- ✅ `from app.services.rag_service import RAGService`
- ✅ `from app.routes import health`

This ensures clean separation and avoids circular dependencies.
