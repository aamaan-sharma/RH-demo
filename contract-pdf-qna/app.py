# Set the OpenAI API Keys, embedding model,

import os
import asyncio
from dotenv import load_dotenv
from flask import Flask, request, jsonify, make_response, Response, stream_with_context, session
from flask_socketio import SocketIO, emit, join_room, disconnect
from pymongo import MongoClient, ReturnDocument
from datetime import datetime
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Milvus
from langchain_community.memory.motorhead_memory import MotorheadMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.agents import Tool, initialize_agent, AgentType
from time import time
from bson.objectid import ObjectId
import uuid
from flask_cors import CORS
from oauth2client import client
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
# LangChain v0.3+ expects AgentExecutor.memory to be a BaseMemory implementation.
from langchain.memory import ConversationBufferMemory
# Add imports for transcript processing
import json
import re
from typing import List, Dict
from pathlib import Path

# Live Copilot for real-time AI suggestions during calls
try:
    from live_copilot import handle_transcript_event
    LIVE_COPILOT_AVAILABLE = True
except ImportError:
    LIVE_COPILOT_AVAILABLE = False
    print("Warning: live_copilot module not available - Live Copilot disabled")

# GCP Storage imports using fsspec (unified filesystem interface)
try:
    import fsspec
    import gcsfs
    import certifi
    import os
    import ssl
    
    # Configure SSL certificates for macOS compatibility
    # CRITICAL: Set these BEFORE creating any filesystem objects
    # gcsfs uses aiohttp which requires SSL certificates via env vars
    cert_path = certifi.where()
    
    # Always set these (don't check if already set - ensure they're correct)
    os.environ['SSL_CERT_FILE'] = cert_path
    os.environ['REQUESTS_CA_BUNDLE'] = cert_path
    os.environ['AIOHTTP_CA_BUNDLE'] = cert_path
    
    # Create SSL context with certifi certificates
    ssl_context = ssl.create_default_context(cafile=cert_path)
    
    print(f"✓ SSL certificates configured: {cert_path}")
    
    GCP_STORAGE_AVAILABLE = True
except ImportError:
    print("Warning: fsspec or gcsfs not installed. GCP Storage features disabled.")
    print("Install with: pip install fsspec gcsfs")
    GCP_STORAGE_AVAILABLE = False
    fsspec = None
    gcsfs = None
    certifi = None
    ssl_context = None

# Safe import of monitoring_module - handle missing dependencies gracefully
# try:
from monitoring_module import q_monitor, tracer, llm_trace_to_jaeger
# except ImportError as e:
#     print(f"Warning: Could not import monitoring_module: {e}")
#     print("Monitoring features will be disabled. The app will continue to run.")
#     # Create dummy functions to prevent errors
#     def q_monitor(*args, **kwargs):
#         pass
#     class DummyTracer:
#         def start_span(self, *args, **kwargs):
#             return DummySpan()
#     class DummySpan:
#         def __enter__(self):
#             return self
#         def __exit__(self, *args):
#             pass
#         def __getattr__(self, name):
#             return self
#     tracer = DummyTracer()
#     def llm_trace_to_jaeger(*args, **kwargs):
#         pass

from token_module import token_calculator, CallbackHandler
import threading

# -----------------------------------------------------------------------------
# Session-level trace context (1 trace per live sessionId)
# -----------------------------------------------------------------------------
_session_trace_ctx = {}  # sessionId -> opentelemetry.trace.SpanContext
_session_trace_lock = threading.Lock()


def _get_or_create_session_trace_context(session_id: str):
    """
    Ensure a single trace per sessionId by creating ONE root span:
      csr_copilot.session
    We immediately end it (not long-running), and then use its SpanContext as the
    explicit parent for all subsequent spans for that session.
    """
    if not session_id:
        return None
    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.trace import NonRecordingSpan
    except Exception:
        return None

    with _session_trace_lock:
        existing = _session_trace_ctx.get(session_id)
        if existing is None:
            # Create the root span exactly once per sessionId.
            with tracer.start_as_current_span("csr_copilot.session") as root:
                root.set_attribute("live.session_id", session_id)
            try:
                existing = root.get_span_context()
            except Exception:
                existing = None
            if existing is not None:
                _session_trace_ctx[session_id] = existing

        if existing is None:
            return None

        # Return an explicit parent context for child spans.
        try:
            parent_span = NonRecordingSpan(existing)
            return otel_trace.set_span_in_context(parent_span)
        except Exception:
            return None

# Using new LangChain memory API - InMemoryChatMessageHistory
# Note: This is only used to store previous Q&A for standalone prompt, not used in chains
memory1 = InMemoryChatMessageHistory()
handler = CallbackHandler()
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

_async_mode = "threading"
try:
    import eventlet  # noqa: F401

    # Prefer eventlet when available (recommended for production SocketIO)
    _async_mode = "eventlet"
except Exception:
    _async_mode = os.getenv("SOCKETIO_ASYNC_MODE", "threading")

try:
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode=_async_mode,
        manage_session=True
    )
except Exception:
    raise

JWT_AUDIENCE = os.getenv("JWT_AUDIENCE")
JWKS_URL = os.getenv("JWKS_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MILVUS_HOST = os.getenv("MILVUS_HOST")
MONGO_URI = (
    os.getenv("MONGO_URI")
)

# ----------------------------
# Live Copilot: session gating
# ----------------------------
# IMPORTANT:
# - This is intentionally fail-open and no-op by default.
# - Copilot runs ONLY if:
#   1) ENABLE_LIVE_COPILOT=1 (feature flag), AND
#   2) Analyze Live tab explicitly enables the session via Socket.IO (copilot_enable)
#
# This ensures existing /webhook + transcript_update behavior remains unchanged.
_copilot_enabled_sessions = {}  # sessionId -> expires_at_epoch_seconds
# Persist per-session plan context for Live Copilot (Analyze Live).
# Filled by `copilot_enable` from the UI and attached to webhook payloads.
_copilot_session_context = {}  # sessionId -> {"contractType": "...", "selectedPlan": "...", "selectedState": "..."}
_copilot_sessions_lock = threading.Lock()


def _flag_enabled(var_name: str, default: str = "0") -> bool:
    raw = (os.getenv(var_name, default) or "").strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


def _copilot_session_ttl_seconds() -> int:
    try:
        raw = (os.getenv("COPILOT_SESSION_TTL_SECONDS") or "").strip()
        ttl = int(raw) if raw else 1800
        return ttl if ttl > 0 else 1800
    except Exception:
        return 1800


def _copilot_session_is_enabled(session_id: str) -> bool:
    if not session_id:
        return False
    now = time()
    with _copilot_sessions_lock:
        exp = _copilot_enabled_sessions.get(session_id)
        if exp is None:
            return False
        if exp <= now:
            try:
                del _copilot_enabled_sessions[session_id]
            except Exception:
                pass
            return False
        return True

def _optional_positive_int_env(var_name: str):
    """Return a positive int from env var, otherwise None (unset/invalid/<=0)."""
    raw = (os.getenv(var_name) or "").strip()
    if not raw:
        return None
    try:
        val = int(raw)
        return val if val > 0 else None
    except Exception:
        return None

# Milvus retrieval sizing:
# - MILVUS_RETRIEVER_K controls the vector search top-k used by LangChain retrievers.
# - MILVUS_MAX_RETURN_CHUNKS controls how many chunks we return to the API (None = no cap).
MILVUS_RETRIEVER_K = _optional_positive_int_env("MILVUS_RETRIEVER_K") or 25
MILVUS_FALLBACK_K = _optional_positive_int_env("MILVUS_FALLBACK_K") or MILVUS_RETRIEVER_K
MILVUS_MAX_RETURN_CHUNKS = _optional_positive_int_env("MILVUS_MAX_RETURN_CHUNKS")

CLEAR_STATE_ALIASES = {
    # Abbreviation -> collection prefix used in Milvus
    "AZ": "Arizona",
    "CA": "California",
    "GA": "Georgia",
    "MD": "Maryland",
    "MN": "Minnesota",
    "NV": "Nevada",
    "TX": "Texas",
    "UT": "Utah",
    "WI": "Wisconsin",
}

_PLACEHOLDER_CHUNK_VALUES = {
    "[]",
    "",
    "(No supporting excerpts found)",
}

def normalize_contract_type(contract_type: str) -> str:
    if contract_type is None:
        return ""
    return str(contract_type).strip().upper()

def normalize_plan_for_milvus(contract_type: str, selected_plan: str) -> str:
    """
    Normalize selectedPlan into the keys expected by collection_mapping.
    Handles values like "SHIELDPLUS" / "shield_plus" / "Shield Plus".
    """
    if selected_plan is None:
        return ""
    raw = str(selected_plan).strip()
    if not raw:
        return ""
    compact = re.sub(r"[^a-z0-9]+", "", raw.lower())
    ct = normalize_contract_type(contract_type)

    # RE plan keys
    if ct == "RE":
        if compact in ("shieldessential", "essential"):
            return "ShieldEssential"
        if compact in ("shieldplus", "plus"):
            return "ShieldPlus"
        if compact in ("shieldcomplete", "complete"):
            # Not a direct key; this is the default for RE
            return "default"

    # DTC plan keys
    if ct == "DTC":
        if compact in ("shieldsilver", "silver"):
            return "ShieldSilver"
        if compact in ("shieldgold", "gold"):
            return "ShieldGold"
        if compact in ("shieldplatinum", "platinum"):
            # Not a direct key; this is the default for DTC
            return "default"

    return raw

def normalize_state_for_milvus(selected_state: str) -> str:
    """
    Normalize incoming selectedState into the exact state prefix used in Milvus collection names.

    Example:
      - "AZ" / "az" -> "Arizona"
      - "arizona" -> "Arizona"

    If the input is unknown, returns a trimmed version of the original.
    """
    if selected_state is None:
        return ""
    raw = str(selected_state).strip()
    if not raw:
        return ""
    key = raw.upper()
    if key in CLEAR_STATE_ALIASES:
        return CLEAR_STATE_ALIASES[key]

    # Accept already-provided full names in any casing (e.g., "california")
    lower = raw.lower()
    for v in CLEAR_STATE_ALIASES.values():
        if lower == v.lower():
            return v

    return raw

CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/health", methods=["GET"])
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

mongo_client = MongoClient(MONGO_URI, unicode_decode_error_handler='ignore')
db = mongo_client["FrontDoorDB"]
db2 = mongo_client[os.getenv("MONGO_DB_NAME")]
transcripts_collection = db2.call_transcripts

model_name = "text-embedding-ada-002"
embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

# Initialize GCP Storage using fsspec (with Application Default Credentials)
# Supports multiple authentication methods via environment variables
GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "ahs-demo-transcripts")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "generative-ai-390411")
GCP_SERVICE_ACCOUNT_PATH = os.getenv("GCP_SERVICE_ACCOUNT_PATH", None)  # Optional: path to service account JSON
gcs_fs = None  # fsspec filesystem instance for GCS

# Cache for transcript metadata to avoid re-reading files
transcript_metadata_cache = {}
# Bump this when metadata extraction logic changes to avoid serving stale cached None values.
TRANSCRIPT_METADATA_CACHE_VERSION = "v2"

if GCP_STORAGE_AVAILABLE:
    try:
        # fsspec with gcsfs uses Application Default Credentials automatically
        # Method 1: Use GOOGLE_APPLICATION_CREDENTIALS environment variable (if set)
        # Method 2: Use Application Default Credentials (gcloud auth application-default login)
        # Method 3: Use explicit service account path from env variable
        
        if GCP_SERVICE_ACCOUNT_PATH and os.path.exists(GCP_SERVICE_ACCOUNT_PATH):
            # Use explicit service account file from environment variable
            gcs_fs = fsspec.filesystem('gcs', token=GCP_SERVICE_ACCOUNT_PATH, project=GCP_PROJECT_ID)
            print(f"✓ GCP Storage initialized using fsspec with service account from: {GCP_SERVICE_ACCOUNT_PATH}")
        else:
            # Use Application Default Credentials (ADC)
            # This will use GOOGLE_APPLICATION_CREDENTIALS if set, otherwise ADC
            try:
                # Get Application Default Credentials explicitly and pass to fsspec
                # gcsfs needs explicit credentials to work properly with ADC
                import certifi
                from google.auth import default as google_auth_default
                
                cert_path = certifi.where()
                
                # Get ADC credentials explicitly
                credentials, _ = google_auth_default()
                
                # Create filesystem with explicit ADC credentials
                gcs_fs = fsspec.filesystem('gcs', token=credentials, project=GCP_PROJECT_ID)
                print(f"✓ GCP Storage filesystem created using fsspec")
                print(f"  Bucket: {GCP_BUCKET_NAME}")
                print(f"  Project: {GCP_PROJECT_ID}")
                print(f"  SSL Certificates: {cert_path}")
                print(f"  Using Application Default Credentials")
                
                # Optional: Test connection (but don't fail if it fails - might be SSL/certificate issues)
                try:
                    bucket_path = f"gs://{GCP_BUCKET_NAME}/"
                    # Try to list files to verify connection
                    test_files = gcs_fs.ls(bucket_path, detail=False)
                    print(f"  ✓ Connection test successful - Found {len(test_files)} files in bucket")
                except Exception as test_error:
                    # Warning but don't fail - filesystem object is created, might work on actual use
                    error_msg = str(test_error)
                    if "SSL" in error_msg or "certificate" in error_msg.lower():
                        print(f"  ⚠ SSL certificate issue detected (common on macOS)")
                        print(f"    The filesystem is created and may work despite this warning")
                        print(f"    If you encounter SSL errors, try:")
                        print(f"      /Applications/Python\\ 3.13/Install\\ Certificates.command")
                    else:
                        print(f"  ⚠ Connection test failed: {test_error}")
                        print(f"    The filesystem is created and may work on actual use")
                        print(f"    If issues persist, try:")
                        print(f"      1. Run: gcloud auth application-default login")
                        print(f"      2. Set GOOGLE_APPLICATION_CREDENTIALS env variable")
                        print(f"      3. Set GCP_SERVICE_ACCOUNT_PATH env variable to service account JSON")
            except Exception as e:
                print(f"✗ GCP Storage filesystem creation failed: {e}")
                print(f"  Options:")
                print(f"    1. Run: gcloud auth application-default login")
                print(f"    2. Set GOOGLE_APPLICATION_CREDENTIALS env variable")
                print(f"    3. Set GCP_SERVICE_ACCOUNT_PATH env variable to service account JSON")
                print(f"    4. Install SSL certificates: /Applications/Python\\ 3.13/Install\\ Certificates.command")
                gcs_fs = None
    except Exception as e:
        print(f"✗ GCP Storage initialization failed: {e}")
        print("  GCP Storage features will be disabled.")
        print("  Make sure fsspec and gcsfs are installed: pip install fsspec gcsfs")
        gcs_fs = None


class AgentAction:
    def __init__(self, tool: str, tool_input: str, log: str = None):
        self.tool = tool
        self.tool_input = tool_input
        self.log = log


# Creating a prompt template
prompt_template = """

You are assisting a customer care executive. Your role is to review the contract’s contextual information given in the context below.

{context}

Answer the given user inquiry based on context above as truthfully as possible, providing in-depth explanations together with answers to the inquiries.
You may rephrase the final response to make it concise and sound more human-like, but do not go out of context and do not lose important details and meaning.

You'll be asked about repairs, coverage policy and service questions about home appliances, home fixtures, home care, repairs/replacement and cleaning, and also about the renewal, cancellation or refund policies in the contract, whether a certain service is covered under the contract and similar context.

The contract context given will have information about contractual details, terms and conditions, renewals, cancellation, refund and service request policies, the coverage limits, limitation and exclusion policies. You will need to use and infer from all the information available in context to analyze and then respond with the final answer.

If the question is about a square feet limit, make sure to compare the numerical values properly. 
For example,
Question: "will my 800 square feet guest house be covered?"
The answer to this question will be No, as the square feet limit for guest houses is 750 and 800 is greater than 750.

If the context you have received says that some breakdown is not covered due to "Misuse or Accidental Acts", then say it is not covered.
For example,
Question: "A rat chewed the wires of my ceiling fan. Is the repair covered?"
The answer to this question will be No, as contract clearly mentions that damage due to pests like rat will not be covered.

If the inquiry is unrelated to home repair and service, answer with "I don't have the information to answer this question.". For example, questions like "Tell me about space.", "Write a poem for me.", "Where can I buy a refrigerator?", "Hi! How are you?", etc. are out of context.

Question: {question} Why?
Answer: """

# PROMPT = PromptTemplate(
#     template=prompt_template, input_variables=["context", "question"]
# )
# chain_type_kwargs = {"prompt": PROMPT}

# conversational retrieval QA
PROMPT = PromptTemplate.from_template(prompt_template)

sys_msg = """

You are assisting an AHS customer care executive with home insurance related inquiries from AHS customers. 

You are given a tool named Knowledge Base, always use this tool to answer the questions. 

You also have access to a tool named User Lookup that can fetch user details from the database based on mobile number. Use this tool when you need to retrieve customer information or user profile data. 

The inquiry asked might be subject to some exclusions and limitations which need to be checked for first before answering the rest of the inquiry. 
You have to break down these complex inquiries into multiple subqueries and then use the knowledge base tool multiple times to return the overall answer from the subqueries for the customer's inquiry. 
Make sure to answer to all the subqueries before you return the final answer.

Following are some examples of complex inquiries and how they can be broken down into sub queries.

Example 1:
“My dryer is not drying the clothes properly. It could be because of lint blockage. Will you come to fix it?”.
1. Is the dryer covered by the plan? If yes, Is repair for link blockage in the dryer covered by the plan?

Example 2: 
“I got my refrigerator fixed last week. But there is another issue with it now. What if that problem was caused by the last repair?”
1. Is the refrigerator covered in the plan?
2. If yes, Can another issue with the refrigerator be fixed in a week’s time from the last repair?

Example 3: 
“I purchased a plan from AHS just 5 days ago, and now I want to repair the microwave because it is creating too much noise. Can I get this repair done?” 
1. Is the microwave covered by the plan? If yes, is the repair for noise from the microwave covered?
2. Can I file a service request within 5 days of getting the plan?

Example 4:
“I use my personal washing machine for my daycare business too at my home. The drain pump doesn't seem to be working. Is it covered?”
1. Is washing machine and it's drain pump covered by the plan?
2. Is the breakdown of washing machine due to commercial use covered?

Example 5: 
“My water heater is leaking for some reason. I need to get it fixed. That water leak seeped into the air conditioning system, so that is not working too. So I need to get that fixed too.“
1. Is the water heater covered by the plan?
2. Is the air conditioning system covered by the plan?
3. Is secondary damage to the air conditioning system due to the water heater covered?

Some questions might be simpler and so might not need breaking down. Find response to those questions as it is. Following are examples of such inquiries.

Example 6:
“My microwave is not working. Is it covered?”

Example 7:
“My toilet seat is broken. Will you repair it?”

Do not answer any questions for which information is not provided by the knowledge base tool. 

If the inquiry is unrelated to home repair and service, answer with "I don't have the information to answer this question." For example, questions like "Tell me about space.", "Write a poem for me.", "Where can I buy a refrigerator?", "Hi! How are you?", etc. are out of context.

"""


def fetch_user_by_mobile(mobile_number: str) -> str:
    """
    Fetch user details from the database based on mobile number.
    
    Args:
        mobile_number: The mobile number to search for
        
    Returns:
        A string containing user details in JSON format, or an error message
    """
    try:
        # Access the 'ahs' database and 'Users' collection
        ahs_db = mongo_client["AHS"]
        users_collection = ahs_db["Users"]
        
        # Search for user by mobile number
        user = users_collection.find_one({"mobile": mobile_number})
        
        if user:
            # Convert ObjectId to string for JSON serialization
            if "_id" in user:
                user["_id"] = str(user["_id"])
            # Return user details as JSON string
            return json.dumps(user, indent=2, default=str)
        else:
            return f"No user found with mobile number: {mobile_number}"
    except Exception as e:
        return f"Error fetching user details: {str(e)}"


def input_prompt(entered_query, qa, llm):
    # Retriever chain as Tool for agent
    knowledge_base_tool = Tool(
        name="Knowledge Base",
        func=qa.run,
        description=(
            "Useful for answering questions related to insurance coverage of home appliances, home fixtures, their repairs/replacement, service requests, about the renewal, cancellation or refund policies, whether a certain service is covered under the contract, permit limit, code violation limit, modification limit, limitations and exclusions."
        ),
    )
    
    # User lookup tool
    user_lookup_tool = Tool(
        name="User Lookup",
        func=fetch_user_by_mobile,
        description=(
            "Useful for fetching user details from the database based on mobile number. "
            "Use this tool when you need to retrieve customer information, user profile, or any user-related data. "
            "Input should be the mobile number as a string. Returns user details in JSON format if found, or an error message if not found."
        ),
    )

    tools = [knowledge_base_tool, user_lookup_tool]

    current_time = time()

    MOTORHEAD_API_KEY = os.getenv("MOTORHEAD_API_KEY")
    MOTORHEAD_CLIENT_ID = os.getenv("MOTORHEAD_CLIENT_ID")
    MOTORHEAD_SESSION_ID = str(current_time)
    MOTORHEAD_MEMORY_KEY = "chat_history"

    # Long Term chat memory
    memory = MotorheadMemory(
        api_key=MOTORHEAD_API_KEY,
        client_id=MOTORHEAD_CLIENT_ID,
        session_id=MOTORHEAD_SESSION_ID,
        memory_key=MOTORHEAD_MEMORY_KEY,
        return_messages=True,
        input_key="input",
        output_key="output",
    )

    #
    async def memory_initialize():
        await memory.init()

    asyncio.run(memory_initialize())

    # Initializing agent
    agent = initialize_agent(
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        verbose=True,
        memory=memory,
        early_stopping_method="generate",
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    new_prompt = agent.agent.create_prompt(system_message=sys_msg, tools=tools)

    agent.agent.llm_chain.prompt = new_prompt

    response = agent({"input": entered_query},callbacks=[handler])
    return response


# Function to get relevant documents
def relevant_docs(entered_query, retriever):
    """
    Wrapper around retriever.get_relevant_documents with detailed logging.
    Returns the original stringified format used by the rest of the app.
    """
    try:
        # Log the incoming query
        # print(
        #     "[CHUNKS] relevant_docs: calling retriever for query="
        #     f"'{str(entered_query)[:500].replace(chr(10), ' ')}'"
        # )

        # Get chunks from the vector store
        docs = retriever.get_relevant_documents(entered_query)
        # print(f"[CHUNKS] relevant_docs: got {len(docs)} docs from retriever")

        if docs:
            # Log every chunk we received for this query
            # for idx, doc in enumerate(docs):
            #     content = getattr(doc, "page_content", "") or ""
            #     metadata = getattr(doc, "metadata", {}) or {}
            #     print(
            #         "[CHUNKS] relevant_docs: chunk "
            #         f"index={idx}, "
            #         f"content_len={len(content)}, "
            #         f"content_preview='{content[:500].replace(chr(10), ' ')}', "
            #         f"metadata={metadata}"
            #     )
            pass
        # else:
        #     print("[CHUNKS] relevant_docs: docs list is EMPTY")

        # Preserve existing behavior (stringified docs)
        relevant_document = "Referred Documents: " + str(docs)

        # Log the actual string that will be stored / returned (trimmed for safety)
        # print(
        #     "[CHUNKS] relevant_docs: relevant_document value_preview="
        #     f"'{relevant_document[:2000].replace(chr(10), ' ')}'"
        # )
        # print(
        #     "[CHUNKS] relevant_docs: returning stringified documents, "
        #     f"len(relevant_document)={len(relevant_document)}"
        # )
        return relevant_document
    except Exception as e:
        print(f"[CHUNKS] relevant_docs: ERROR calling retriever: {e}")
        return "Referred Documents: []"


# ==================== TRANSCRIPT PROCESSING HELPER FUNCTIONS ====================

def extract_transcript_metadata(transcript_content: str, file_name: str) -> Dict:
    """
    Extract contractType, planType, and state from transcript file content
    Uses hybrid approach: JSON parsing -> Regex patterns -> LLM (if needed)
    """
    metadata = {
        "contractType": None,
        "planType": None,
        "state": None
    }
    
    try:
        # Method 1: Try parsing as JSON first (fastest)
        try:
            transcript_data = json.loads(transcript_content)
            if isinstance(transcript_data, dict):
                # Check common metadata field locations
                metadata_fields = transcript_data.get("metadata", {})
                if not metadata_fields:
                    metadata_fields = transcript_data
                
                # Extract contractType (case-insensitive keys)
                metadata["contractType"] = (
                    metadata_fields.get("contractType") or 
                    metadata_fields.get("contract_type") or
                    metadata_fields.get("contractType") or
                    metadata_fields.get("type")
                )
                
                # Extract planType
                metadata["planType"] = (
                    metadata_fields.get("planType") or
                    metadata_fields.get("plan_type") or
                    metadata_fields.get("selectedPlan") or
                    metadata_fields.get("selected_plan") or
                    metadata_fields.get("plan")
                )
                
                # Extract state
                metadata["state"] = (
                    metadata_fields.get("state") or
                    metadata_fields.get("selectedState") or
                    metadata_fields.get("selected_state") or
                    metadata_fields.get("stateCode")
                )
                
                # If all found, return early
                if all([metadata["contractType"], metadata["planType"], metadata["state"]]):
                    return metadata
        except json.JSONDecodeError:
            pass  # Not JSON, continue to text parsing
        
        # Method 2: Regex-based text parsing (fast, no LLM needed)
        content_upper = transcript_content.upper()
        
        # Extract contract type
        # Look for RE or Real Estate mentions
        if re.search(r'\bRE\b', content_upper) or "REAL ESTATE" in content_upper:
            metadata["contractType"] = "RE"
        elif re.search(r'\bDTC\b', content_upper) or "DIRECT TO CONSUMER" in content_upper or "DIRECT-TO-CONSUMER" in content_upper:
            metadata["contractType"] = "DTC"
        
        # Extract plan type using regex patterns
        plan_patterns = {
            "ShieldComplete": [
                r"SHIELD\s*COMPLETE",
                r"SHIELDCOMPLETE",
                r"COMPLETE\s*PLAN"
            ],
            "ShieldEssential": [
                r"SHIELD\s*ESSENTIAL",
                r"SHIELDESSENTIAL",
                r"ESSENTIAL\s*PLAN"
            ],
            "ShieldPlus": [
                r"SHIELD\s*PLUS",
                r"SHIELDPLUS",
                r"PLUS\s*PLAN"
            ],
            "ShieldSilver": [
                r"SHIELD\s*SILVER",
                r"SHIELDSILVER",
                r"SILVER\s*PLAN"
            ],
            "ShieldGold": [
                r"SHIELD\s*GOLD",
                r"SHIELDGOLD",
                r"GOLD\s*PLAN"
            ],
            "ShieldPlatinum": [
                r"SHIELD\s*PLATINUM",
                r"SHIELDPLATINUM",
                r"PLATINUM\s*PLAN"
            ]
        }
        
        for plan, patterns in plan_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_upper):
                    metadata["planType"] = plan
                    break
            if metadata["planType"]:
                break
        
        # Extract state codes (two-letter US state codes)
        # First try full state name matching (more accurate)
        state_names = {
            "CA": ["California", "Calif"],
            "NY": ["New York"],
            "TX": ["Texas"],
            "FL": ["Florida"],
            "IL": ["Illinois"],
            "PA": ["Pennsylvania"],
            "OH": ["Ohio"],
            "GA": ["Georgia"],
            "NC": ["North Carolina"],
            "MI": ["Michigan"],
            "NJ": ["New Jersey"],
            "VA": ["Virginia"],
            "WA": ["Washington"],
            "AZ": ["Arizona"],
            "MA": ["Massachusetts"],
            "TN": ["Tennessee"],
            "IN": ["Indiana"],
            "MO": ["Missouri"],
            "MD": ["Maryland"],
            "WI": ["Wisconsin"],
            "NV": ["Nevada"],
            "UT": ["Utah"],
            "HI": ["Hawaii"],
            "AK": ["Alaska"],
            "AR": ["Arkansas"],
            "CO": ["Colorado"],
            "CT": ["Connecticut"],
            "DE": ["Delaware"],
            "HI": ["Hawaii"],
            "ID": ["Idaho"],
            "IA": ["Iowa"],
            "KS": ["Kansas"],
            "KY": ["Kentucky"],
            "LA": ["Louisiana"],
            "ME": ["Maine"],
            "MN": ["Minnesota"],
            "MS": ["Mississippi"],
        }
        
        for state_code, names in state_names.items():
            # content_upper is uppercase, so compare uppercase to avoid missing matches (bug fix).
            if any(str(name).upper() in content_upper for name in names):
                # Prefer full state name for UI dropdown and Milvus naming.
                # CLEAR_STATE_ALIASES contains mappings like "CA" -> "California".
                metadata["state"] = CLEAR_STATE_ALIASES.get(state_code, state_code)
                break
        
        # If not found by name, try state code matching with context
        if not metadata["state"]:
            # Common state codes (prioritize common ones)
            common_state_codes = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI", 
                                 "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI"]
            other_state_codes = ["AL", "AK", "AR", "CO", "CT", "DE", "HI", "ID", "IA", "KS",
                                "KY", "LA", "ME", "MN", "MS", "MT", "NE", "NV", "NH", "NM",
                                "ND", "OK", "OR", "RI", "SC", "SD", "UT", "VT", "WV", "WY", "DC"]
            
            all_state_codes = common_state_codes + other_state_codes
            
            for state_code in all_state_codes:
                # Pattern: state code with word boundaries, but check context
                pattern = r'\b' + state_code + r'\b'
                matches = list(re.finditer(pattern, content_upper))
                
                for match in matches:
                    # Check surrounding context (avoid false positives like "IN" in "calling")
                    start = max(0, match.start() - 15)
                    end = min(len(content_upper), match.end() + 15)
                    context = content_upper[start:end]
                    
                    # Positive indicators
                    positive_keywords = ["STATE", "PLAN", "CONTRACT", "COVERAGE", "POLICY", 
                                       "CALIFORNIA", "TEXAS", "FLORIDA", "NEW YORK", "ILLINOIS"]
                    # Negative indicators (words that might contain state codes)
                    negative_keywords = ["CALLING", "INFORMATION", "INSPECTION", "INSTALLATION"]
                    
                    # Check if context has positive keywords and not negative ones
                    has_positive = any(keyword in context for keyword in positive_keywords)
                    has_negative = any(keyword in context for keyword in negative_keywords)
                    
                    if has_positive or (not has_negative and len(context.strip()) < 30):
                        # Prefer full state name when possible
                        metadata["state"] = CLEAR_STATE_ALIASES.get(state_code, state_code)
                        break
                
                if metadata["state"]:
                    break
    
    except Exception as e:
        print(f"Error extracting metadata from transcript {file_name}: {e}")
    
    return metadata


def list_transcript_files_gcp(limit: int = None, offset: int = 0, search: str = None) -> tuple:
    """
    List transcript files from GCP bucket using fsspec with pagination and search support
    
    IMPORTANT: This function searches through ALL files in the GCS bucket (all 147 files).
    It first lists all files from GCS, then applies search filter, then pagination.
    
    Returns tuple: (paginated_transcripts, total_count)
    - Lists ALL files from GCS bucket first (searches through complete file list)
    - If search is provided, filters ALL files by file name (case-insensitive partial match)
    - Then applies pagination to the filtered results
    - If limit is None, returns all transcripts (for backward compatibility)
    - If limit is set, only reads file contents for the paginated subset (much faster)
    """
    all_file_info = []  # Store basic file info without reading content
    try:
        # Ensure SSL certificates are set (in case function is called before app initialization)
        if GCP_STORAGE_AVAILABLE and certifi:
            cert_path = certifi.where()
            if 'SSL_CERT_FILE' not in os.environ:
                os.environ['SSL_CERT_FILE'] = cert_path
            if 'REQUESTS_CA_BUNDLE' not in os.environ:
                os.environ['REQUESTS_CA_BUNDLE'] = cert_path
            if 'AIOHTTP_CA_BUNDLE' not in os.environ:
                os.environ['AIOHTTP_CA_BUNDLE'] = cert_path
        
        if not gcs_fs:
            print(f"ERROR list_transcript_files_gcp: gcs_fs is None!")
            return ([], 0) if limit else []
        
        print(f"DEBUG list_transcript_files_gcp: Starting, gcs_fs type={type(gcs_fs)}, limit={limit}, offset={offset}, search={search}")
        bucket_path = f"gs://{GCP_BUCKET_NAME}/"
        
        # List files - try both root and transcripts/ prefix
        prefixes = ["transcripts/", ""]
        seen_files = set()  # Track files we've already processed
        
        for prefix in prefixes:
            try:
                # List files with details
                full_path = bucket_path + prefix if prefix else bucket_path
                print(f"DEBUG: Attempting to list files from: {full_path}")
                files = gcs_fs.ls(full_path, detail=True)
                print(f"DEBUG: Found {len(files)} items in {full_path}")
                
                for file_info in files:
                    # file_info can be a dict (detail=True) or string (detail=False)
                    # Handle both cases
                    if isinstance(file_info, str):
                        # If it's a string, it's just the path
                        file_path = file_info
                        file_size = 0
                        time_created = None
                    else:
                        # It's a dict with details
                        file_path = file_info.get('name', '')
                        file_size = file_info.get('size', 0)
                        time_created = file_info.get('timeCreated', None)
                    
                    # Skip directories (they end with /)
                    if file_path.endswith('/'):
                        continue
                    
                    # Only include JSON and TXT files
                    if not (file_path.endswith('.json') or file_path.endswith('.txt')):
                        continue
                    
                    # Extract filename
                    file_name = file_path.split("/")[-1]
                    
                    # Skip if already added
                    if file_name in seen_files:
                        continue
                    seen_files.add(file_name)
                    
                    # Convert timeCreated to ISO format if available
                    upload_date = None
                    if time_created:
                        if isinstance(time_created, str):
                            upload_date = time_created
                        else:
                            # If it's a datetime object, convert to ISO
                            upload_date = time_created.isoformat() if hasattr(time_created, 'isoformat') else str(time_created)
                    
                    # Store basic file info without reading content yet
                    all_file_info.append({
                        "fileName": file_name,
                        "filePath": file_path,
                        "uploadDate": upload_date,
                        "fileSize": file_size if file_size else 0,
                        "timeCreated": time_created
                    })
            except Exception as e:
                # Log the error for debugging
                error_msg = str(e)
                import traceback
                error_trace = traceback.format_exc()
                print(f"ERROR listing files with prefix '{prefix}': {error_msg}")
                print(f"Full traceback:\n{error_trace}")
                if "SSL" in error_msg or "certificate" in error_msg.lower():
                    print(f"  SSL certificate issue detected!")
                    print(f"  SSL_CERT_FILE={os.environ.get('SSL_CERT_FILE', 'NOT SET')}")
                    print(f"  REQUESTS_CA_BUNDLE={os.environ.get('REQUESTS_CA_BUNDLE', 'NOT SET')}")
                    print(f"  AIOHTTP_CA_BUNDLE={os.environ.get('AIOHTTP_CA_BUNDLE', 'NOT SET')}")
                # Continue to next prefix
                continue
        
        # Sort by upload date (newest first)
        all_file_info.sort(key=lambda x: x.get("uploadDate", "") or "", reverse=True)
        
        # Log total files found from GCS before any filtering
        total_files_from_gcs = len(all_file_info)
        print(f"DEBUG: Listed ALL {total_files_from_gcs} files from GCS bucket")
        
        # Store sample file names before filtering (for debugging)
        sample_file_names = []
        if total_files_from_gcs > 0:
            sample_file_names = [f.get("fileName", "") for f in all_file_info[:10]]
            print(f"DEBUG: Sample file names from GCS (first 10): {sample_file_names}")
        
        # Apply search filter if provided (case-insensitive partial match on file name)
        # This searches through ALL files from GCS (all 147 files)
        if search and search.strip():
            search_term = search.strip().lower()
            print(f"DEBUG: Searching through ALL {total_files_from_gcs} files from GCS for: '{search_term}'")
            print(f"DEBUG: Search will match any file name containing '{search_term}' (case-insensitive)")
            
            # Filter: search through all files from GCS
            matching_files = []
            checked_count = 0
            for file_info in all_file_info:
                file_name = file_info.get("fileName", "")
                file_name_lower = file_name.lower()
                
                # Debug: log first few comparisons
                if checked_count < 5:
                    matches = search_term in file_name_lower
                    print(f"DEBUG: Checking file '{file_name}' (lowercase: '{file_name_lower}') - contains '{search_term}'? {matches}")
                    checked_count += 1
                
                if search_term in file_name_lower:
                    matching_files.append(file_info)
            
            all_file_info = matching_files
            print(f"DEBUG: Search complete - Found {len(all_file_info)} matching files out of {total_files_from_gcs} total files in GCS")
            
            # If no matches, show sample file names to help debug
            if len(all_file_info) == 0 and total_files_from_gcs > 0:
                print(f"DEBUG: No matches found for '{search_term}'")
                print(f"DEBUG: Sample file names available in GCS: {sample_file_names}")
                print(f"DEBUG: Tip: Check if any file names contain '{search_term}'. Try calling without search parameter to see all file names.")
        else:
            print(f"DEBUG: No search term provided - returning all {total_files_from_gcs} files from GCS")
        
        total_count = len(all_file_info)
        print(f"DEBUG: Final count after search: {total_count} files, limit={limit}, offset={offset}")
        
        # If limit is None, return all (backward compatibility - read all files)
        if limit is None:
            print("DEBUG: limit is None - returning all transcripts")
            transcripts = []
            for file_info in all_file_info:
                # Extract contract metadata from file content
                transcript_metadata = {
                    "contractType": None,
                    "planType": None,
                    "state": None
                }
                
                # Check cache first
                cache_key = f"{TRANSCRIPT_METADATA_CACHE_VERSION}_{file_info['filePath']}_{file_info['timeCreated']}"
                if cache_key in transcript_metadata_cache:
                    transcript_metadata = transcript_metadata_cache[cache_key]
                else:
                    # Read file content to extract metadata (limit size for performance)
                    try:
                        file_size = file_info.get('fileSize', 0)
                        if file_size and file_size < 50000:  # Only read files < 50KB for metadata extraction
                            with gcs_fs.open(file_info['filePath'], 'r') as f:
                                content = f.read()
                            transcript_metadata = extract_transcript_metadata(content, file_info['fileName'])
                            # Cache the result
                            transcript_metadata_cache[cache_key] = transcript_metadata
                        elif file_size:
                            print(f"Skipping metadata extraction for large file: {file_info['fileName']} ({file_size} bytes)")
                    except Exception as e:
                        print(f"Error reading transcript {file_info['fileName']} for metadata extraction: {e}")
                
                transcripts.append({
                    "fileName": file_info['fileName'],
                    "filePath": file_info['filePath'],
                    "uploadDate": file_info['uploadDate'],
                    "fileSize": file_info['fileSize'],
                    "metadata": {},
                    "contractType": transcript_metadata.get("contractType"),
                    "planType": transcript_metadata.get("planType"),
                    "state": transcript_metadata.get("state")
                })
            return transcripts
        
        # Apply pagination BEFORE reading file contents (optimization)
        print(f"DEBUG: Applying pagination - slicing all_file_info[{offset}:{offset + limit}]")
        paginated_file_info = all_file_info[offset:offset + limit]
        print(f"DEBUG: Paginated to {len(paginated_file_info)} files out of {total_count} total")
        
        # Now read file contents only for the paginated subset
        transcripts = []
        for file_info in paginated_file_info:
            # Extract contract metadata from file content
            transcript_metadata = {
                "contractType": None,
                "planType": None,
                "state": None
            }
            
            # Check cache first
            cache_key = f"{TRANSCRIPT_METADATA_CACHE_VERSION}_{file_info['filePath']}_{file_info['timeCreated']}"
            if cache_key in transcript_metadata_cache:
                transcript_metadata = transcript_metadata_cache[cache_key]
            else:
                # Read file content to extract metadata (limit size for performance)
                try:
                    file_size = file_info.get('fileSize', 0)
                    if file_size and file_size < 50000:  # Only read files < 50KB for metadata extraction
                        with gcs_fs.open(file_info['filePath'], 'r') as f:
                            content = f.read()
                        transcript_metadata = extract_transcript_metadata(content, file_info['fileName'])
                        # Cache the result
                        transcript_metadata_cache[cache_key] = transcript_metadata
                    elif file_size:
                        print(f"Skipping metadata extraction for large file: {file_info['fileName']} ({file_size} bytes)")
                except Exception as e:
                    print(f"Error reading transcript {file_info['fileName']} for metadata extraction: {e}")
            
            transcripts.append({
                "fileName": file_info['fileName'],
                "filePath": file_info['filePath'],
                "uploadDate": file_info['uploadDate'],
                "fileSize": file_info['fileSize'],
                "metadata": {},
                "contractType": transcript_metadata.get("contractType"),
                "planType": transcript_metadata.get("planType"),
                "state": transcript_metadata.get("state")
            })
        
        print(f"DEBUG: Returning {len(transcripts)} transcripts with total_count={total_count}")
        return (transcripts, total_count)
        
    except Exception as e:
        print(f"Error listing transcript files from GCP: {e}")
        import traceback
        traceback.print_exc()
        return ([], 0) if limit else []


def read_transcript_file_gcp(file_name: str) -> tuple:
    """
    Read transcript file content from GCP bucket using fsspec
    Returns: (content, file_metadata_dict)
    """
    try:
        if not gcs_fs:
            raise Exception("GCP Storage not available")
        
        bucket_path = f"gs://{GCP_BUCKET_NAME}/"
        
        # Try different possible paths
        possible_paths = [
            f"{bucket_path}transcripts/{file_name}",
            f"{bucket_path}{file_name}",
        ]
        
        file_path = None
        for path in possible_paths:
            if gcs_fs.exists(path):
                file_path = path
                break
        
        if not file_path:
            raise FileNotFoundError(f"Transcript file not found: {file_name}")
        
        # Read file content using fsspec
        with gcs_fs.open(file_path, 'r') as f:
            content = f.read()
        
        # Get file metadata
        file_info = gcs_fs.info(file_path)
        time_created = file_info.get('timeCreated', None)
        
        # Convert timeCreated to ISO format if available
        upload_date = None
        if time_created:
            if isinstance(time_created, str):
                upload_date = time_created
            else:
                # If it's a datetime object, convert to ISO
                upload_date = time_created.isoformat() if hasattr(time_created, 'isoformat') else str(time_created)
        
        file_metadata = {
            "fileName": file_name,
            "fileSize": file_info.get('size', 0),
            "uploadDate": upload_date,
            "metadata": {}  # fsspec doesn't provide custom metadata
        }
        
        return content, file_metadata
    
    except Exception as e:
        print(f"Error reading transcript file from GCP: {e}")
        raise


def filter_relevant_customer_questions(questions: List[Dict]) -> List[Dict]:
    """
    Filter questions to keep only relevant customer questions related to coverage, damage, repair, or customer problems.
    Excludes customer service representative questions and non-relevant queries.
    
    Args:
        questions: List of question dictionaries with 'question', 'context', and 'questionType' fields
        
    Returns:
        Filtered list of relevant customer questions
    """
    if not questions:
        return []
    
    # Keywords/phrases that indicate customer service rep questions (to exclude)
    rep_question_patterns = [
        'can i have your',
        'what\'s your',
        'may i know',
        'could you please provide',
        'can you tell me your',
        'what is your',
        'do you have',
        'are you',
        # Note: keep this list conservative; generic phrases like "is this" can appear in customer questions.
        'can you confirm',
        'would you like',
        'how can i help',
        'thank you for calling',
        'good morning',
        'good afternoon',
        'good evening'
    ]
    
    filtered_questions = []
    
    for question_obj in questions:
        question_text_raw = (question_obj.get('question', '') or '').strip()
        if not question_text_raw:
            continue
        question_text = question_text_raw.lower()
        context_text = (question_obj.get('context', '') or '').lower()
        combined_text = f"{question_text} {context_text}"
        
        # Check if it's a rep question (exclude these)
        is_rep_question = any(pattern in combined_text for pattern in rep_question_patterns)
        if is_rep_question:
            continue

        # We rely primarily on the LLM prompt to extract only customer-intent items.
        # Keep this filter permissive to avoid dropping valid intents (especially implicit/process questions).
        filtered_questions.append(question_obj)
    
    return filtered_questions


def extract_relevant_customer_questions(transcript_content: str, llm) -> List[Dict]:
    """
    Extract only relevant atomic questions asked by end customers from transcript.
    Focuses on coverage lookup, damage repair, and customer problems.
    Filters out customer service representative questions and non-relevant queries.
    
    This function is specifically designed for the Calls section (/transcripts/process endpoint).
    """

    # Optimized extraction prompt with 3-step process: Understand Intent → Frame Question → Extract
    extraction_prompt_template = """
        You are an expert at analyzing customer service transcripts. Take the customer's perspective to understand what they need, then break that need into clear questions required to fully resolve it. Use a structured 3-step process, and infer questions even when none are explicitly asked (including when described by a technician/representative).

        STEP 1: UNDERSTAND USER INTENT
        - Put yourself in the customer’s shoes; aim for complete customer satisfaction.
        - Accept signals from any speaker (customer, technician, representative) describing the customer’s issue.
        - Include implicit clarifications needed to resolve the situation end-to-end (item, location, cause, limits, costs, approvals).

        Focus on customer statements that express:
        - Intent to understand coverage (e.g., "I want to know if...", "Is this covered?", "Will you repair...")
        - Intent to understand problems (e.g., "My appliance is...", "There's a leak...", "The damage is...")
        - Intent to understand policies (e.g., "What's the limit?", "How much does it cost?", "What's included?")

        EXCLUDE:
        - Customer service representative questions (e.g., "Can I have your name?", "What's your position?", "May I know...", "How can I help you?")
        - Administrative questions
        - Greetings or pleasantries
        - Questions not related to coverage/repair/damage/contract/policy

        STEP 2: FRAME THE QUESTIONS
        For each identified customer intent, frame it as a clear, atomic question that can be answered independently. Frame questions in a way that:
        - Captures the customer's actual concern or problem (in their voice)
        - Is specific and answerable from contract knowledge base
        - Focuses on coverage, damage, repair, policy, and any clarifiers needed to resolve the issue
        - If no explicit questions are present, infer and create them. Ensure at least one question per described issue.

        HARD REQUIREMENTS (Calls mode):
        - Do NOT write generic questions like "Is it covered?", "Is this covered?", "Is that covered?" or "Is it covered or not?".
        - Every question must be CUSTOMER-SPECIFIC: explicitly mention the appliance/system and the specific issue/service (symptom/part/service).
        - Avoid vague pronouns ("it/this/that") unless you immediately clarify the appliance/issue in the same sentence.
        - Generate a compact but complete set of questions that covers WH-style checks as QUESTIONS when implied by the case:
          - What failed / what service is needed
          - Where (location / affected area / on/off premises when relevant)
          - When (timing, waiting period, recent repair when relevant)
          - Why (suspected cause, secondary damage, misuse/commercial use when relevant)
          - How (repair vs replace, diagnostics, service call/trade call, limits/fees)
        - Keep questions clean and professional; no filler, no disclaimers.

        Question types to frame:
        1. Coverage questions (contextual): "Does my plan cover diagnosing/repairing/replacing [appliance/part] for [specific failure mode]?"
        2. Damage/repair questions (contextual): "Does the plan cover [specific repair/service] for [specific damage/failure] and under what limits/fees?"
        3. Policy/limit questions: "What is the [specific limit/policy] for [item]?"
        4. Problem statements: Convert customer problems into questions that include appliance + failure mode + requested service.
        5. Clarifying questions that help resolve the customer’s need (e.g., specifics about the item, location, cause, or limits that determine coverage)

        STEP 3: EXTRACT AND RETRIEVE
        Extract the framed questions with proper context and question type classification.

Transcript:
{transcript}

        Follow this 3-step process:
        1. Identify customer intents (what they want to know/understand)
        2. Frame each intent as a clear, atomic question
        3. Extract and return the questions

        Return ONLY a JSON array of relevant customer questions in this format:
[
  {{
                "question": "Does my plan cover diagnosing and repairing my water heater tank leak described in the transcript, including any covered parts/labor and applicable fees?",
                "context": "Customer mentioned their water heater tank is leaking and causing floor damage; customer wants to know coverage for the leak and related service",
                "questionType": "coverage",
                "userIntent": "Customer wants to understand if the water heater leak they're experiencing is covered by their plan"
            }},
            {{
                "question": "What are the out of pocket costs for uncovered repairs?",
                "context": "Customer asked about homeowner's financial responsibility when repair is not covered",
                "questionType": "coverage",
                "userIntent": "Customer wants to understand their financial responsibility for uncovered repairs"
            }},
            {{
                "question": "Do you cover leak detection for a backyard copper line leak?",
                "context": "Customer/tech described backyard leak with unknown exact source and recommended leak detection",
                "questionType": "coverage",
                "userIntent": "Customer wants to know if leak detection for this scenario is covered"
  }}
]

        IMPORTANT:
        - Extract only questions that reflect customer intent (not rep questions)
        - Frame questions clearly and specifically
        - Include userIntent field to show what the customer is trying to understand
        - Return only valid JSON, no additional text
        - If no relevant customer questions are found, return an empty array []
    """

    extraction_prompt = ChatPromptTemplate.from_template(extraction_prompt_template)
    
    extraction_chain = extraction_prompt | llm | StrOutputParser()
    
    try:
        result = extraction_chain.invoke({"transcript": transcript_content})
        # Clean the result - remove markdown code blocks if present
        result = re.sub(r'```json\n?', '', result)
        result = re.sub(r'```\n?', '', result)
        result = result.strip()
        
        questions = json.loads(result)
        
        # Apply post-extraction filtering for additional safety
        questions = filter_relevant_customer_questions(questions)
        
        # Add question IDs
        for idx, q in enumerate(questions):
            q["questionId"] = f"q{idx + 1}"
        
        return questions
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from LLM: {e}")
        print(f"LLM Response: {result[:500]}")
        return []
    except Exception as e:
        print(f"Error extracting relevant customer questions: {e}")
        return []


def extract_questions_with_agent(transcript_content: str, llm) -> List[Dict]:
    """
    Extract relevant customer questions from transcript using an agent-based approach.
    Uses the same extraction prompt and filtering logic as extract_relevant_customer_questions()
    to ensure consistency with Search/Infer functionality.
    
    This function is specifically designed for the Calls section (/transcripts/process endpoint).
    """
    """
    Extract relevant customer questions from transcript using an agent-based approach.
    Uses the same extraction prompt and filtering logic as extract_relevant_customer_questions()
    to ensure consistency with Search/Infer functionality.
    
    This function is specifically designed for the Calls section (/transcripts/process endpoint).
    """
    # Optimized extraction prompt with 3-step process: Understand Intent → Frame Question → Extract
    extraction_prompt_template = """
        You are an expert at analyzing customer service transcripts. Take the customer's perspective to understand what they need, then break that need into clear questions required to fully resolve it. Use a structured 3-step process, and infer questions even when none are explicitly asked (including when described by a technician/representative).

        STEP 1: UNDERSTAND USER INTENT
        - Put yourself in the customer’s shoes; aim for complete customer satisfaction.
        - Accept signals from any speaker (customer, technician, representative) describing the customer’s issue.
        - Include implicit clarifications needed to resolve the situation end-to-end (item, location, cause, limits, costs, approvals).

        Focus on customer statements that express:
        - Intent to understand coverage (e.g., "I want to know if...", "Is this covered?", "Will you repair...")
        - Intent to understand problems (e.g., "My appliance is...", "There's a leak...", "The damage is...")
        - Intent to understand policies (e.g., "What's the limit?", "How much does it cost?", "What's included?")

        EXCLUDE:
        - Customer service representative questions (e.g., "Can I have your name?", "What's your position?", "May I know...", "How can I help you?")
        - Administrative questions
        - Greetings or pleasantries
        - Questions not related to coverage/repair/damage/contract/policy

        STEP 2: FRAME THE QUESTIONS
        For each identified customer intent, frame it as a clear, atomic question that can be answered independently. Frame questions in a way that:
        - Captures the customer's actual concern or problem (in their voice)
        - Is specific and answerable from contract knowledge base
        - Focuses on coverage, damage, repair, policy, and any clarifiers needed to resolve the issue
        - If no explicit questions are present, infer and create them. Ensure at least one question per described issue.

        HARD REQUIREMENTS (Calls mode):
        - Do NOT write generic questions like "Is it covered?", "Is this covered?", "Is that covered?" or "Is it covered or not?".
        - Every question must be CUSTOMER-SPECIFIC: explicitly mention the appliance/system and the specific issue/service (symptom/part/service).
        - Avoid vague pronouns ("it/this/that") unless you immediately clarify the appliance/issue in the same sentence.
        - Generate a compact but complete set of questions that covers WH-style checks as QUESTIONS when implied by the case:
          - What failed / what service is needed
          - Where (location / affected area / on/off premises when relevant)
          - When (timing, waiting period, recent repair when relevant)
          - Why (suspected cause, secondary damage, misuse/commercial use when relevant)
          - How (repair vs replace, diagnostics, service call/trade call, limits/fees)
        - Keep questions clean and professional; no filler, no disclaimers.

        Question types to frame:
        1. Coverage questions (contextual): "Does my plan cover diagnosing/repairing/replacing [appliance/part] for [specific failure mode]?"
        2. Damage/repair questions (contextual): "Does the plan cover [specific repair/service] for [specific damage/failure] and under what limits/fees?"
        3. Policy/limit questions: "What is the [specific limit/policy] for [item]?"
        4. Problem statements: Convert customer problems into questions that include appliance + failure mode + requested service.
        5. Clarifying questions that help resolve the customer’s need (e.g., specifics about the item, location, cause, or limits that determine coverage)

        STEP 3: EXTRACT AND RETRIEVE
        Extract the framed questions with proper context and question type classification.

Transcript:
{transcript}

        Follow this 3-step process:
        1. Identify customer intents (what they want to know/understand)
        2. Frame each intent as a clear, atomic question
        3. Extract and return the questions

        Return ONLY a JSON array of relevant customer questions in this format:
        [
            {{
                "question": "Does my plan cover diagnosing and repairing my water heater tank leak described in the transcript, including any covered parts/labor and applicable fees?",
                "context": "Customer mentioned their water heater tank is leaking and causing floor damage; customer wants to know coverage for the leak and related service",
                "questionType": "coverage",
                "userIntent": "Customer wants to understand if the water heater leak they're experiencing is covered by their plan"
            }},
            {{
                "question": "What are the out of pocket costs for uncovered repairs?",
                "context": "Customer asked about homeowner's financial responsibility when repair is not covered",
                "questionType": "coverage",
                "userIntent": "Customer wants to understand their financial responsibility for uncovered repairs"
            }},
            {{
                "question": "Do you cover leak detection for a backyard copper line leak?",
                "context": "Customer/tech described backyard leak with unknown exact source and recommended leak detection",
                "questionType": "coverage",
                "userIntent": "Customer wants to know if leak detection for this scenario is covered"
            }}
        ]

        IMPORTANT:
        - Extract only questions that reflect customer intent (not rep questions)
        - Frame questions clearly and specifically
        - Include userIntent field to show what the customer is trying to understand
        - Return only valid JSON, no additional text
        - If no relevant customer questions are found, return an empty array []
    """
    # Optimized extraction prompt with 3-step process: Understand Intent → Frame Question → Extract
#     extraction_prompt_template = """
# You are extracting customer-intent questions for an insurance claim from a transcript.

# Extract items ONLY if they represent a customer intent, need, question, confusion, objection, request, or decision point.
# Include both explicit questions and implicit questions (e.g., “I’m not sure what to do” → “What should I do next?”).

# Do NOT extract agent/CSR questions unless the customer repeats/endorses them as their own concern.
# Do NOT include pleasantries, small talk, or purely informational statements unless they imply a need.

# No speculation. No invented facts.

# You MUST capture the claim situation:
# - Always include at least one canonical item that summarizes the primary claim the customer is calling about (what happened, what is damaged, what the customer wants us to do).
# - If multiple items/damages are involved, include one item per distinct issue AND a primary claim item tying them together.
 
# Coverage-model completeness (extract questions that map to real claim handling):
# - Claim intake/triage: who is reporting, what happened (alleged cause), when, where, how discovered, urgency/safety/ongoing damage.
# - Policy & eligibility gating: correct insured/asset/location, policy in force/waiting period, eligibility, limits & deductibles/sublimits.
# - Coverage trigger: what must be true for coverage to apply (based on policy wording in general terms; do not invent).
# - Causation/mechanism ("because"): sudden vs gradual, wear/tear vs accidental, contributing causes, sequence of events.
# - Exclusions/limitations: identify likely carve-outs the customer is worried about and turn them into questions.
# - Conditions/duties: notice, mitigation, documentation, proof-of-loss, preserve evidence, cooperation.
# - Damages/scope/valuation: what is being claimed (repair/replacement/reimbursement), amounts/estimates, valuation method if implied.
# - Process/timeline: claim filing steps, documents needed, expected timeline, next steps, appeal/dispute options.

# For EACH extracted item:
# - Make it atomic and customer-voiced (what the customer wants to know/do).
# - Include a brief situation summary in context.
# - Include 1–2 verbatim evidence quotes from the transcript inside the context field as:
#   Evidence: “...” / “...”
# - Assign an appropriate questionType.
# - Include userIntent.

# De-duplicate repeated intents into one canonical question (keep the most specific wording).

# Completeness rule:
# - Do NOT limit the number of extracted items. Include ALL distinct customer intents present in the transcript.
#  - If the transcript implies uncertainty that blocks a decision, extract a targeted question to resolve it (exactly what is missing).

# Transcript:
# {transcript}

# Return ONLY valid JSON (no markdown, no extra text) as a JSON array:
# [
#   {{
#     "question": "string",
#     "context": "1–3 sentences situation summary. Evidence: “...” / “...”",
#     "questionType": "coverage|limit|exclusion|eligibility|process|cost|timeline|status|next_steps|repair|damage|policy|claim|other",
#     "userIntent": "string"
#   }}
# ]

# If no relevant customer intents are present, return [].
#     """
    
    # Create a tool that uses the extraction prompt
    def extract_questions_tool(transcript: str) -> str:
        """Tool to extract relevant customer questions from transcript using the standard extraction prompt."""
        extraction_prompt = ChatPromptTemplate.from_template(extraction_prompt_template)
        extraction_chain = extraction_prompt | llm | StrOutputParser()
        
        try:
            result = extraction_chain.invoke({"transcript": transcript})
            # Clean the result - remove markdown code blocks if present
            result = re.sub(r'```json\n?', '', result)
            result = re.sub(r'```\n?', '', result)
            result = result.strip()
            return result
        except Exception as e:
            print(f"Error in extraction tool: {e}")
            return "[]"
    
    # Create the transcript analysis tool
    transcript_analysis_tool = Tool(
        name="Transcript Question Extractor",
        func=extract_questions_tool,
        description=(
            "Useful for extracting relevant customer questions from customer service transcripts using a 3-step process: "
            "1) Understand user intent (what customer wants to know), "
            "2) Frame clear atomic questions from intents, "
            "3) Extract questions with context. "
            "Focuses on coverage lookup, damage/repair issues, coverage limits, and customer problems. "
            "Excludes customer service representative questions and administrative queries. "
            "Returns a JSON array with question, context, questionType, and userIntent fields."
        ),
    )
    
    tools = [transcript_analysis_tool]
    
    # System message for the agent - optimized with 3-step process
    agent_sys_msg = """
You are a claims transcript extraction supervisor.

Use the tool "Transcript Question Extractor" with the full transcript.

Your success criteria:
- Extract ONLY customer intents (explicit or implicit): needs, questions, confusion, objections, requests, decision points.
- Exclude CSR/admin questions unless the customer explicitly adopts them.
- De-duplicate repeated intents into one canonical question.
- Output MUST be ONLY a valid JSON array of objects with:
  question, context (including 1–2 evidence quotes), questionType, userIntent

Hard rule:
- If the tool output contains any non-JSON text, fix it and return ONLY the JSON array.

Return the final JSON array and nothing else.
    """
    
    # LangChain AgentExecutor expects a BaseMemory, not a ChatMessageHistory.
    # Use a simple in-process buffer memory for this one-off extraction run.
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="input",
        output_key="output",
    )
    
    try:
        # Initialize agent
        agent = initialize_agent(
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            tools=tools,
            llm=llm,
            verbose=True,
            memory=memory,
            early_stopping_method="generate",
            handle_parsing_errors=True,
            return_intermediate_steps=True,
        )
        
        # Create prompt with system message
        new_prompt = agent.agent.create_prompt(system_message=agent_sys_msg, tools=tools)
        agent.agent.llm_chain.prompt = new_prompt
        
        # Run agent with transcript
        agent_input = f"Extract relevant customer questions from this transcript:\n\n{transcript_content}"
        print(f"DEBUG: Running agent with transcript length: {len(transcript_content)} characters")
        response = agent.invoke({"input": agent_input})
        
        print(f"DEBUG: Agent response keys: {response.keys()}")
        print(f"DEBUG: Agent output: {response.get('output', '')[:200]}")
        
        # Extract the result from agent response
        result_text = response.get("output", "")
        
        # If agent used the tool, extract from intermediate steps
        if "intermediate_steps" in response and response["intermediate_steps"]:
            print(f"DEBUG: Found {len(response['intermediate_steps'])} intermediate steps")
            # Get the last tool result
            for idx, step in enumerate(reversed(response["intermediate_steps"])):
                print(f"DEBUG: Step {idx}: {type(step)}, length: {len(step) if isinstance(step, (list, tuple)) else 'N/A'}")
                if len(step) > 1 and isinstance(step[1], str):
                    result_text = step[1]
                    print(f"DEBUG: Found tool result in step {idx}: {result_text[:200]}")
                    break
        
        # Clean the result - remove markdown code blocks if present
        result_text = re.sub(r'```json\n?', '', result_text)
        result_text = re.sub(r'```\n?', '', result_text)
        result_text = result_text.strip()
        
        print(f"DEBUG: Cleaned result text length: {len(result_text)}")
        print(f"DEBUG: Cleaned result text (first 500 chars): {result_text[:500]}")
        
        # Parse JSON
        try:
            questions = json.loads(result_text)
            print(f"DEBUG: Successfully parsed {len(questions)} questions from agent")
        except json.JSONDecodeError as json_err:
            print(f"DEBUG: JSON decode error: {json_err}")
            print(f"DEBUG: Attempting to extract JSON from text...")
            # Try to extract JSON array from the text
            json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
            if json_match:
                try:
                    questions = json.loads(json_match.group())
                    print(f"DEBUG: Extracted JSON array with {len(questions)} questions")
                except:
                    print(f"DEBUG: Failed to parse extracted JSON")
                    raise json_err
            else:
                raise json_err
        
        # Apply post-extraction filtering using existing function (same as Search/Infer)
        print(f"DEBUG: Before filtering: {len(questions)} questions")
        questions = filter_relevant_customer_questions(questions)
        print(f"DEBUG: After filtering: {len(questions)} questions")
        
        # If no questions after agent extraction, try direct extraction as fallback
        if not questions or len(questions) == 0:
            print(f"DEBUG: Agent extraction returned no questions, trying direct extraction method...")
            try:
                direct_questions = extract_relevant_customer_questions(transcript_content, llm)
                if direct_questions and len(direct_questions) > 0:
                    print(f"DEBUG: Direct extraction found {len(direct_questions)} questions")
                    return direct_questions
                else:
                    print(f"DEBUG: Direct extraction also returned no questions")
            except Exception as fallback_err:
                print(f"DEBUG: Direct extraction fallback failed: {fallback_err}")
        
        # Add question IDs
        for idx, q in enumerate(questions):
            q["questionId"] = f"q{idx + 1}"
        
        return questions
        
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON parsing failed in agent extraction: {e}")
        print(f"ERROR: Result text (first 1000 chars): {result_text[:1000] if 'result_text' in locals() else 'N/A'}")
        print(f"ERROR: Falling back to direct extraction method...")
        # Fallback to direct extraction if agent fails
        try:
            return extract_relevant_customer_questions(transcript_content, llm)
        except Exception as fallback_err:
            print(f"ERROR: Fallback extraction also failed: {fallback_err}")
            return []
    except Exception as e:
        print(f"ERROR: Exception in agent extraction: {e}")
        import traceback
        traceback.print_exc()
        print(f"ERROR: Falling back to direct extraction method...")
        # Fallback to direct extraction if agent fails
        try:
            return extract_relevant_customer_questions(transcript_content, llm)
        except Exception as fallback_err:
            print(f"ERROR: Fallback extraction also failed: {fallback_err}")
            return []


def extract_atomic_questions(transcript_content: str, llm) -> List[Dict]:
    """
    Extract atomic questions from transcript content using LLM
    """
    extraction_prompt = ChatPromptTemplate.from_template(
        """
        You are an expert at analyzing customer service transcripts and extracting atomic questions.
        
        Analyze the following transcript and extract all atomic questions that customers asked.
        An atomic question is a single, specific question that can be answered independently.
        
        Transcript:
        {transcript}
        
        Return ONLY a JSON array of questions in this format:
        [
            {{
                "question": "Is the refrigerator covered?",
                "context": "Customer mentioned refrigerator issue",
                "questionType": "coverage"
            }},
            {{
                "question": "What is the repair limit?",
                "context": "Customer asked about repair costs",
                "questionType": "limit"
            }}
        ]
        
        Extract all questions. Return only valid JSON, no additional text.
        """
    )
    
    extraction_chain = extraction_prompt | llm | StrOutputParser()
    
    try:
        result = extraction_chain.invoke({"transcript": transcript_content})
        # Clean the result - remove markdown code blocks if present
        result = re.sub(r'```json\n?', '', result)
        result = re.sub(r'```\n?', '', result)
        result = result.strip()
        
        questions = json.loads(result)
        # Add question IDs
        for idx, q in enumerate(questions):
            q["questionId"] = f"q{idx + 1}"
        return questions
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from LLM: {e}")
        print(f"LLM Response: {result[:500]}")
        return []
    except Exception as e:
        print(f"Error extracting questions: {e}")
        return []


def process_single_transcript_question(
    question: str,
    contract_type: str,
    selected_plan: str,
    selected_state: str,
    gpt_model: str,
    vector_db: Milvus,
    llm,
    llm2,
    retriever,
    handler,
    transcript_context: str = "",
) -> Dict:
    """
    Process a single question from transcript and return answer with chunks
    Reuses logic from /start endpoint but without conversation context
    """
    try:
        q_start_time = time()
        # No conversation context for transcript questions, but we CAN pass the transcript-derived
        # situation/evidence as part of the query to improve retrieval + answer relevance.
        # Keep the user-visible question unchanged elsewhere; only enrich the internal query.
        standalone_result = question
        enriched_query = (
            f"{question}\n\nTranscript situation/evidence:\n{transcript_context}".strip()
            if (transcript_context or "").strip()
            else question
        )
        
        print(
            "[CHUNKS] process_single_transcript_question: START "
            f"question='{str(question)[:200]}', "
            f"contract_type={contract_type}, selected_plan={selected_plan}, "
            f"selected_state={selected_state}, gpt_model={gpt_model}"
        )

        if gpt_model == "Search":
            prompt_template = """
You are a professional insurance claims representative (CSR). Be empathetic, firm, and to the point.

Use ONLY the policy/contract context provided below. Do NOT speculate or invent facts.

Claims can be informational or coverage-related. Act accordingly:
- If the question is about process/timeline/documents/next steps/costs, answer informatively with clear steps.
- If the question is about coverage/limits/exclusions/eligibility, use a universal decision posture and give a clear determination using available facts from the context.

Avoid “if/then” branching answers.

Universal decision posture model (choose ONE):
- ACCEPT_AND_PAY: trigger met, no applicable exclusions, conditions met, scope/valuation supported
- ACCEPT_PARTIAL: some components covered, others excluded/limited; apply deductible/sublimits/depreciation if stated
- DENY: trigger not met, exclusion applies, policy not in force/eligible (if stated)
- REQUEST_INFO: insufficient evidence; request specific items needed and why
- RESERVE_RIGHTS: potential coverage issues; continue investigation while reserving rights (only if the context supports this posture)

If required information is missing to make a determination:
1) State what CAN be concluded from known facts,
2) State what CANNOT be concluded,
3) State exactly what you need next (documents/details) and the next step.

Make the answer accountable so follow-up WH-questions are answerable:
- Include a short "Why" line grounded in the provided context.
- Include a short "Policy basis" line quoting 1–2 exact clause snippets from the provided context.
- Include "Next step" if anything is missing or a process step is required.
Keep these accountability lines short.

Policy/contract context (verbatim):
{context}

Customer question:
{question}

Answer format:
- Answer: (2–6 sentences, decisive, no hypotheticals)
- Why: (1 short sentence)
- Policy basis: (quote 1–2 short clause snippets)
- Next step: (if applicable; otherwise say "No further action needed.")
"""
            
            PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            chain_type_kwargs = {"prompt": PROMPT}
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                verbose=True,
                chain_type_kwargs=chain_type_kwargs
            )
            
            # print("[CHUNKS] process_single_transcript_question: calling QA chain (Search)")
            qa_response = qa.invoke(
                {"query": enriched_query},
                config={"callbacks": [handler]},
            )
            answer = qa_response["result"] if isinstance(qa_response, dict) else qa_response
            print(
                "[CHUNKS] process_single_transcript_question: QA chain completed "
                f"answer_len={len(str(answer))}"
            )

            # print("[CHUNKS] process_single_transcript_question: calling relevant_docs (Search)")
            relevant_documents = relevant_docs(enriched_query, retriever=retriever)
            # print(
            #     "[CHUNKS] process_single_transcript_question: relevant_documents string length "
            #     f"len={len(relevant_documents)}"
            # )
            
        elif gpt_model == "Infer":
            # print("[CHUNKS] process_single_transcript_question: building QA chain (Infer)")
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, verbose=True)
            agent_response = input_prompt(enriched_query, qa, llm)
            answer = agent_response["output"]
            print(
                "[CHUNKS] process_single_transcript_question: agent_response received "
                f"answer_len={len(str(answer))}"
            )
            knowledge_base_thoughts = [
                item[0].tool_input for item in agent_response["intermediate_steps"] 
                if item[0].tool == 'Knowledge Base'
            ]
            relevant_documents = ""
            for action_input in knowledge_base_thoughts:
                print(
                    "[CHUNKS] process_single_transcript_question: calling relevant_docs (Infer) "
                    f"for tool_input='{str(action_input)[:200]}'"
                )
                rd = relevant_docs(action_input, retriever)
                print(
                    "[CHUNKS] process_single_transcript_question: returned from relevant_docs (Infer) "
                    f"len={len(rd)}"
                )
                relevant_documents += rd
        else:
            return {
                "error": f"Invalid gpt_model: {gpt_model}",
                "answer": "",
                "relevantChunks": [],
                "confidence": 0.0,
                "latency": 0.0
            }
        
        q_latency = time() - q_start_time
        
        # Build relevantChunks from Milvus docs (always list[str] in the API response)
        # This ensures frontend receives text chunks (not placeholder "[]" / not dict objects).
        chunk_texts = []
        chunk_details = []
        try:
            # First attempt: retriever (normal path)
            docs_for_chunks = retriever.get_relevant_documents(enriched_query)
            if not docs_for_chunks:
                # Fallbacks to ensure we still fetch something from Milvus
                fallback_queries = [
                    f"{enriched_query} {contract_type} {selected_plan} {selected_state}",
                    f"{contract_type} {selected_plan} contract coverage",
                    "contract coverage",
                ]
                for fq in fallback_queries:
                    try:
                        docs_for_chunks = vector_db.similarity_search(fq, k=MILVUS_FALLBACK_K)
                        if docs_for_chunks:
                            break
                    except Exception as e:
                        print(f"[CHUNKS] process_single_transcript_question: fallback similarity_search failed: {e}")
                        continue

            docs_for_chunks = docs_for_chunks or []
            print(
                "[CHUNKS] process_single_transcript_question: docs_for_chunks_count="
                f"{len(docs_for_chunks)}"
            )

            docs_iter = docs_for_chunks
            if MILVUS_MAX_RETURN_CHUNKS is not None:
                docs_iter = docs_for_chunks[:MILVUS_MAX_RETURN_CHUNKS]

            print(f"[CHUNKS_DEBUG] process_single_transcript_question: Processing {len(docs_iter)} documents")
            
            for doc in docs_iter:
                chunk_index = len(chunk_texts) + 1
                print(f"\n[CHUNKS_DEBUG] ========== Chunk {chunk_index} (process_single_transcript_question) ==========")
                
                content = (getattr(doc, "page_content", "") or "").strip()
                metadata = getattr(doc, "metadata", {}) or {}
                
                print(f"[CHUNKS_DEBUG] Content preview (first 100 chars): {content[:100]}...")
                print(f"[CHUNKS_DEBUG] Metadata type: {type(metadata)}")
                
                if not content:
                    print(f"[CHUNKS_DEBUG] SKIPPING: No content in document")
                    continue
                
                # Detailed metadata logging
                if isinstance(metadata, dict):
                    print(f"[CHUNKS_DEBUG] Metadata keys: {list(metadata.keys())}")
                    # Print each metadata field individually for clarity
                    for key, value in metadata.items():
                        print(f"[CHUNKS_DEBUG]   {key}: {value}")
                    print(f"[CHUNKS_DEBUG] Full metadata JSON: {json.dumps(metadata, default=str, indent=2)}")
                else:
                    print(f"[CHUNKS_DEBUG] Metadata value: {metadata}")
                
                print(f"[CHUNKS_DEBUG] ===========================================\n")
                
                # Generate meaningful name from metadata
                chunk_name = _generate_chunk_name(metadata, chunk_index)
                
                # Log the generated name
                print(f"[CHUNKS_DEBUG] Generated chunk name for chunk {chunk_index}: '{chunk_name}'")
                
                chunk_texts.append(content)
                chunk_details.append({
                    "content": content, 
                    "metadata": metadata,
                    "name": chunk_name  # Add meaningful name
                })
        except Exception as e:
            print(f"[CHUNKS] process_single_transcript_question: ERROR building chunks: {e}")

        if not chunk_texts:
            # As a last resort, still return a non-empty list (but keep it explicit for debugging).
            # This should be rare; most Milvus collections should return at least some results.
            chunk_texts = ["(No supporting excerpts found)"]
        
        # print(
        #     "[CHUNKS] process_single_transcript_question: FINAL "
        #     f"chunks_count={len(chunk_texts)}, latency={q_latency}"
        # )

        # Log the exact chunks that will be returned with this question
        returned_chunks = chunk_texts
        returned_chunk_details = chunk_details
        if MILVUS_MAX_RETURN_CHUNKS is not None:
            returned_chunks = chunk_texts[:MILVUS_MAX_RETURN_CHUNKS]
            returned_chunk_details = chunk_details[:MILVUS_MAX_RETURN_CHUNKS]
        
        # Return chunks as objects with names for better UI display
        # Frontend can handle both string arrays (backward compatibility) and object arrays
        chunks_with_names = []
        for detail in returned_chunk_details:
            if isinstance(detail, dict):
                chunk_obj = {
                    "content": detail.get("content", ""),
                    "metadata": detail.get("metadata", {}),
                    "name": detail.get("name", f"Clause {len(chunks_with_names) + 1}")
                }
                chunks_with_names.append(chunk_obj)
                # Log what we're returning
                print(f"[CHUNKS_DEBUG] ✅ Returning chunk object with name: '{chunk_obj.get('name')}'")
            else:
                # Fallback for backward compatibility
                chunks_with_names.append({
                    "content": str(detail) if not isinstance(detail, dict) else detail.get("content", ""),
                    "name": f"Clause {len(chunks_with_names) + 1}"
                })
        
        print(f"[CHUNKS_DEBUG] ✅ Total chunks_with_names: {len(chunks_with_names)}")
        if chunks_with_names:
            print(f"[CHUNKS_DEBUG] ✅ First chunk name: '{chunks_with_names[0].get('name', 'MISSING')}'")
            print(f"[CHUNKS_DEBUG] ✅ First chunk keys: {list(chunks_with_names[0].keys())}")
        
        # print(
        #     "[CHUNKS] process_single_transcript_question: returning relevantChunks="
        #     f"{[c[:200].replace(chr(10), ' ') for c in returned_chunks]}"
        # )

        # Use chunks_with_names as the primary format (objects with name, content, metadata)
        # This ensures MongoDB stores objects with names, not just strings
        return {
            "answer": answer,
            # Return chunks as objects with names (this is what gets stored in MongoDB)
            "relevantChunks": chunks_with_names,  # Changed from returned_chunks (strings) to chunks_with_names (objects)
            # Keep string array for backward compatibility if needed
            "relevantChunksStrings": returned_chunks,
            # Keep details for optional persistence/debugging
            "relevantChunksDetail": returned_chunk_details,
            "confidence": 0.90,  # Default confidence, can be calculated from LLM
            "latency": q_latency
        }
    except Exception as e:
        print(f"Error processing transcript question: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "answer": "Error processing question",
            "relevantChunks": [],
            "confidence": 0.0,
            "latency": 0.0
        }


# -------------------------------------------------------------------
# process_live_copilot_question: Wrapper for Live Copilot INFER
# -------------------------------------------------------------------
def process_live_copilot_question(
    question: str,
    contract_type: str,
    selected_plan: str,
    selected_state: str,
    transcript_context: str = "",
) -> Dict:
    """
    Wrapper for Live Copilot to use the existing INFER implementation.
    
    This function initializes Milvus, LLMs, and retriever, then calls
    process_single_transcript_question with gpt_model="Infer" to leverage
    the full LangChain Agent with Knowledge Base and User Lookup tools.
    
    Args:
        question: The customer question to answer
        contract_type: Contract type (RE or DTC)
        selected_plan: Plan name (ShieldPlus, ShieldGold, etc.)
        selected_state: State name (California, Texas, etc.)
        transcript_context: Optional transcript context for enrichment
        
    Returns:
        Dict with keys: answer, relevantChunks, confidence, latency
    """
    try:
        print(
            f"[LIVE_COPILOT_INFER] Processing question='{question[:100]}...', "
            f"contract_type={contract_type}, plan={selected_plan}, state={selected_state}"
        )
        
        # Normalize inputs
        milvus_state = normalize_state_for_milvus(selected_state)
        contract_type_norm = normalize_contract_type(contract_type)
        selected_plan_norm = normalize_plan_for_milvus(contract_type_norm, selected_plan)
        
        # Build collection name
        collection_mapping = {
            "RE": {
                "ShieldEssential": f"{milvus_state}_RE_ShieldEssential",
                "ShieldPlus": f"{milvus_state}_RE_ShieldPlus",
                "default": f"{milvus_state}_RE_ShieldComplete",
            },
            "DTC": {
                "ShieldSilver": f"{milvus_state}_DTC_ShieldSilver",
                "ShieldGold": f"{milvus_state}_DTC_ShieldGold",
                "default": f"{milvus_state}_DTC_ShieldPlatinum",
            },
        }
        
        selected_collection_name = collection_mapping.get(contract_type_norm, {}).get(
            selected_plan_norm, collection_mapping.get(contract_type_norm, {}).get("default")
        )
        
        if not selected_collection_name:
            print(f"[LIVE_COPILOT_INFER] Could not determine collection name for contract_type={contract_type_norm}, plan={selected_plan_norm}")
            return {
                "answer": "Unable to determine the appropriate knowledge base for your query.",
                "relevantChunks": [],
                "confidence": 0.0,
                "latency": 0.0,
            }
        
        print(f"[LIVE_COPILOT_INFER] Using Milvus collection: {selected_collection_name}")
        
        # Initialize Milvus vector DB
        vector_db1 = Milvus(
            embed,
            collection_name=selected_collection_name,
            connection_args={"host": MILVUS_HOST, "port": "19530"},
        )
        
        # Initialize retriever
        retriever = vector_db1.as_retriever(search_kwargs={"k": MILVUS_RETRIEVER_K})
        
        # Initialize LLMs for Infer mode
        llm = ChatOpenAI(temperature=0.0, model="gpt-4o")
        llm2 = ChatOpenAI(temperature=0.0, model="gpt-4o")
        
        # Call the existing INFER implementation
        result = process_single_transcript_question(
            question=question,
            contract_type=contract_type,
            selected_plan=selected_plan,
            selected_state=selected_state,
            gpt_model="Infer",  # Use INFER mode with LangChain Agent
            vector_db=vector_db1,
            llm=llm,
            llm2=llm2,
            retriever=retriever,
            handler=handler,
            transcript_context=transcript_context,
        )
        
        print(f"[LIVE_COPILOT_INFER] Result: answer_len={len(result.get('answer', ''))}, chunks={len(result.get('relevantChunks', []))}")
        
        return result
        
    except Exception as e:
        print(f"[LIVE_COPILOT_INFER] Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "answer": f"Error processing question: {str(e)}",
            "relevantChunks": [],
            "confidence": 0.0,
            "latency": 0.0,
        }


# Feedback CRUD Operations


# Feedback CRUD Operations
# Create (Insert) operation
def insert_feedback(data, email_id):
    feedbacks_collection_user = f"feedbacks_{email_id}"
    feedbacks_collection = db[feedbacks_collection_user]
    result = feedbacks_collection.insert_one(data)
    print(f"Document inserted with ID: {result.inserted_id}")


# Read operation
def read_feedback(query, email_id):
    feedbacks_collection_user = f"feedbacks_{email_id}"
    feedbacks_collection = db[feedbacks_collection_user]
    search_query = {"entered_query": query}
    documents = (
        feedbacks_collection.find(search_query)
        if search_query
        else feedbacks_collection.find()
    )
    for document in documents:
        print(document)


# Update operation
def update_feedback(query, new_data, email_id):
    feedbacks_collection_user = f"feedbacks_{email_id}"
    feedbacks_collection = db[feedbacks_collection_user]
    search_query = {"entered_query": query}
    result = feedbacks_collection.update_one(search_query, {"$set": new_data})
    print(f"Modified {result.modified_count} document(s)")


# Delete operation
def delete_feedback(query, email_id):
    feedbacks_collection_user = f"feedbacks_{email_id}"
    feedbacks_collection = db[feedbacks_collection_user]
    search_query = {"entered_query": query}
    result = feedbacks_collection.delete_one(search_query)
    print(f"Deleted {result.deleted_count} document(s)")


# Questions and Answers CRUD Operations
# Create (Insert) operation
def insert_qna(data, email_id):
    qna_collection_today = f"chats_{email_id}"
    qna_collection = db[qna_collection_today]
    result = qna_collection.insert_one(data)
    print(f"Document inserted with ID: {result.inserted_id}")
    return result


# Anirudha Read operation
# def read_qna(query, email_id):
#     qna_collection_today = f"chats_{email_id}"
#     qna_collection = db[qna_collection_today]
#     search_query = {"entered_query": query}
#     documents = qna_collection.find(search_query) if search_query else qna_collection.find()
#     for document in documents:
#         print(document)

# def read_qna(email_id,mongo_qna_id=None):


def read_qna(email_id, conversation_id):
    qna_collection_user = f"chats_{email_id}"
    qna_collection = db[qna_collection_user]
    search_query = {"_id": ObjectId(conversation_id)}
    documents = qna_collection.find_one(search_query)
    return documents


# Update operation
def update_qna(query, new_data, email_id):
    qna_collection_today = f"chats_{email_id}"
    qna_collection = db[qna_collection_today]
    search_query = {"entered_query": query}
    result = qna_collection.update_one(search_query, {"$set": new_data})
    print(f"Modified {result.modified_count} document(s)")


# Delete operation
def delete_qna(query, email_id):
    qna_collection_today = f"chats_{email_id}"
    qna_collection = db[qna_collection_today]
    search_query = {"entered_query": query}
    result = qna_collection.delete_one(search_query)
    print(f"Deleted {result.deleted_count} document(s)")


def update_chat(new_data, conversation_id, email_id):
    qna_collection_user = f"chats_{email_id}"
    qna_collection = db[qna_collection_user]
    search_query = {"_id": ObjectId(conversation_id)}
    result = qna_collection.update_one(search_query, {"$push": {"chats": new_data}})
    print(f"Modified {result.modified_count} document(s)")


def token_process(authorization_header):
    parts = authorization_header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        bearer_token = parts[1]
        try:
            token = client.verify_id_token(bearer_token, JWT_AUDIENCE)
            return (token), 200
        except Exception as e:
            if str(e).split(",")[0] == "Token used too late":
                return jsonify({"message": "Token has expired"}), 403
            else:
                return jsonify({"message": "Token is invalid"}), 403
    else:
        return jsonify({"message": "Token is missing"}), 401


@app.before_request
def before_request():
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
    }
    if request.method == "OPTIONS" or request.method == "options":
        return jsonify(headers), 200


@app.route("/feedback", methods=["POST"])
def feedback():
    with tracer.start_as_current_span('api/feedback'):
        authorization_header = request.headers.get("Authorization")

        # case 5: missing token
        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401

        if authorization_header:
            token_data = token_process(authorization_header)

            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]

        user_feedback = request.get_json()

        # extract from query parameters
        conversation_id = request.args.get("conversation-id")
        chat_id = request.args.get("chat-id")

        # extract values from input
        reaction = user_feedback.get("reaction")
        response = user_feedback.get("response")
        user_email = token_data[0]["email"]

        query_time = datetime.utcnow()

        # output to be stored in mongodb collection
        feedback_json = {
            "query_time": query_time,
            "conversation_id": conversation_id,
            "chat_id": chat_id,
            "reaction": reaction,
            "response": response,
        }

        insert_feedback(feedback_json, user_email)
        return {}


@app.route("/calls/transcripts", methods=["GET"])
def calls_transcripts():
    authorization_header = request.headers.get("Authorization")

    if authorization_header is None:
        return jsonify({"message": "Token is missing"}), 401

    if authorization_header:
        token_data = token_process(authorization_header)

        if token_data[1] == 401 or token_data[1] == 403:
            return (token_data[0].get_json()), token_data[1]

    q = request.args.get("q", "").strip()
    status = request.args.get("status", "active").lower()
    page = int(request.args.get("page", 1) or 1)
    page_size = int(request.args.get("pageSize", 20) or 20)

    query = {}
    if status and status != "all":
        query["status"] = status
    if q:
        query["name"] = {"$regex": q, "$options": "i"}

    skip = (page - 1) * page_size

    total = calls_transcripts_collection.count_documents(query)
    cursor = (
        calls_transcripts_collection.find(query)
        .sort("created_at", -1)
        .skip(skip)
        .limit(page_size)
    )

    items = []
    for doc in cursor:
        items.append(
            {
                "id": str(doc["_id"]),
                "name": doc.get("name"),
                "stateName": doc.get("state_name"),
                "contractType": doc.get("contract_type"),
                "planName": doc.get("plan_name"),
                "status": doc.get("status", "active"),
                "createdAt": doc.get("created_at").isoformat()
                if doc.get("created_at")
                else None,
                "updatedAt": doc.get("updated_at").isoformat()
                if doc.get("updated_at")
                else None,
            }
        )

    return jsonify(
        {
            "items": items,
            "pagination": {
                "page": page,
                "pageSize": page_size,
                "total": total,
                "totalPages": (total + page_size - 1) // page_size,
            },
        }
    )


@app.route("/start", methods=["POST"])
def start():
    try:
        with tracer.start_as_current_span('api/start') as parent0:
            with tracer.start_as_current_span('authorization'):
                start_time = time()
                authorization_header = request.headers.get("Authorization")

                if authorization_header is None:
                    return jsonify({"message": "Token is missing"}), 401

                if authorization_header:
                    token_data = token_process(authorization_header)

                    if token_data[1] == 401 or token_data[1] == 403:
                        return (token_data[0].get_json()), token_data[1]
            
            with tracer.start_as_current_span('data-fetching'):
                data = request.get_json()
                if not data:
                    return jsonify({"error": "Request body is missing or invalid"}), 400
                
                contract_type = data.get("contractType")
                selected_plan = data.get("selectedPlan")
                selected_state = data.get("selectedState")
                milvus_state = normalize_state_for_milvus(selected_state)
                contract_type_norm = normalize_contract_type(contract_type)
                selected_plan_norm = normalize_plan_for_milvus(contract_type_norm, selected_plan)
                gpt_model = data.get("gptModel")
                entered_query = data.get("enteredQuery")
                
                # Validate required fields
                if not all([contract_type, selected_plan, selected_state, gpt_model, entered_query]):
                    return jsonify({"error": "Missing required fields: contractType, selectedPlan, selectedState, gptModel, enteredQuery"}), 400
                
                # user_email = "kartik.dabre@mindstix.com"
                user_email = token_data[0]["email"]
                conversation_id = request.args.get("conversation-id")

                collection_mapping = {
                    "RE": {
                        "ShieldEssential": f"{milvus_state}_RE_ShieldEssential",
                        "ShieldPlus": f"{milvus_state}_RE_ShieldPlus",
                        "default": f"{milvus_state}_RE_ShieldComplete",
                    },
                    "DTC": {
                        "ShieldSilver": f"{milvus_state}_DTC_ShieldSilver",
                        "ShieldGold": f"{milvus_state}_DTC_ShieldGold",
                        "default": f"{milvus_state}_DTC_ShieldPlatinum",
                    },
                }

                # Get the collection name based on contract_type and selected_plan
                selected_collection_name = collection_mapping.get(contract_type_norm, {}).get(
                    selected_plan_norm, collection_mapping.get(contract_type_norm, {}).get("default")
                )
                print(
                    "[MILVUS] /start selected_state="
                    f"{selected_state!r} -> milvus_state={milvus_state!r}, "
                    f"contract_type={contract_type!r}->{contract_type_norm!r}, "
                    f"selected_plan={selected_plan!r}->{selected_plan_norm!r}, "
                    f"collection={selected_collection_name!r}"
                )
            with tracer.start_as_current_span('vector_db-initialization'):
                # Selecting collection dynamically
                vector_db1: Milvus = Milvus(
                    embed,
                    collection_name=selected_collection_name,
                    connection_args={"host": MILVUS_HOST, "port": "19530"},
                )
            
            # Initialize variables to prevent undefined errors
            agent_resp = None
            relevant_documents = ""

            if gpt_model == "Search":
                with tracer.start_as_current_span('Search') as parent1:
                    with tracer.start_as_current_span('llm-retriever-initialization'):
                        llm2 = ChatOpenAI(temperature=0.0, model="ft:gpt-3.5-turbo-0613:mindstix::8YYD56aA")
                        llm = ChatOpenAI(temperature=0.0, model="gpt-4o")
                        retriever = vector_db1.as_retriever(search_kwargs={"k": MILVUS_RETRIEVER_K})
                    
                    with tracer.start_as_current_span('memory_update'):
                        memory1.clear()
                        question1 = ""
                        answer1 = ""
                        if conversation_id is not None and conversation_id != "":
                            docs = read_qna(email_id=user_email, conversation_id=conversation_id)
                            if docs and "chats" in docs and len(docs["chats"]) > 0:
                                # Use the last *real* user Q&A pairs as memory. Skip system/synthetic chats.
                                skip_chat_ids = {"final_answer", "claim_decision", "case_closed"}
                                skip_entered = {"Final Answer for transcript"}
                                pairs = []
                                for c in reversed(docs.get("chats") or []):
                                    if not isinstance(c, dict):
                                        continue
                                    cid = str(c.get("chat_id") or "").strip()
                                    q = str(c.get("entered_query") or "").strip()
                                    a = str(c.get("response") or "").strip()
                                    if not q or not a:
                                        continue
                                    if cid in skip_chat_ids or q in skip_entered:
                                        continue
                                    if a == "Loading Response":
                                        continue
                                    pairs.append((q, a))
                                    if len(pairs) >= 3:
                                        break

                                # Keep standalone prompt variables as "previous question/answer"
                                if pairs:
                                    question1, answer1 = pairs[0]

                                # Store in new memory API format (oldest -> newest)
                                for q, a in reversed(pairs):
                                    memory1.add_message(HumanMessage(content=q))
                                    memory1.add_message(AIMessage(content=a))

                    with tracer.start_as_current_span('standalone-prompt-chain') as p:
                        standalone_prompt = ChatPromptTemplate.from_template(
                        """
                        Act as an expert in question rephrasing and create a standalone question in its own language by analyzing previous question, answer to the previous question and current question.
                        If the current question is not related to previous question and answer, then return the current question as standalone question. you have analyze if the component or appliance mentioned in the current question is related to the component or appliance mentioned in the previoius question and answer. based on that create the standalone question.
                        standalone question should always contain the appliance name, unless it is a service related question. questions related to modifications, code violation upgrades and permits are not bound to any appliance, so do not rephrase the question and do not relate this to any appliance related question.
                        previous question: """ + question1 + """
                        answer of previous question: """ + answer1 + """
                        current question: """ + entered_query + """
                        
                        examples:
                        1)  previous question:''
                            answer of previous question: ''
                            current question: is the Fridge covered?
                            standalone question: is the Fridge covered?
                        
                        2)  previous question: is the Air Conditioner system covered?
                            answer of previous question: yes, the air conditioner system is covered under the contract.
                            current question: is the compressor covered?
                            standalone question: is the compressor of the air conditioner system covered?
                        
                        In some of the cases, we will not need rephrasing, for example:
                        
                        3)  previous question: is the kitchen faucet covered?
                            answer of previous question: yes, the kitchen faucet is covered under the contract.
                            current question: is the garbage disposal covered?
                            standalone question: is the garbage disposal covered?
                        
                        4)  previous question: is the washer covered
                            answer of previous question: yes, washer is covered under the contract.
                            current question: there is damage to air conditioning unit because of leak but it is secondary, is it covered?
                            standalone question: there is damage to air conditioning unit because of leak but it is secondary, is it covered?
                        
                        """
                        )
                        start = int(time())
                        standalone_chain = standalone_prompt | llm2 | StrOutputParser()

                        standalone_result = standalone_chain.invoke(
                            {"input": entered_query},
                            config={"callbacks": [handler]},
                        )
                        print(standalone_result)
                        res1, tok1 = handler.infi()
                        llm_trace_to_jaeger(res1, tok1)
                        a = threading.Thread(target=token_calculator, args=(tok1,))
                        a.start()

                        print(f"time taken for standalone = {time() - start}")

                    with tracer.start_as_current_span('q_monitor') as parentq:
                        t = threading.Thread(target=q_monitor, args=(parentq,entered_query,))
                        t.start()
                        # q_monitor(parentq,entered_query)

                    with tracer.start_as_current_span('llm-RetrievalQA-chain') as q:
                        # Creating a prompt template
                        prompt_template = """

                        You are assisting a customer care executive. Your role is to review the contract’s contextual information given in the context below.

                        {context}

                        Answer the given user inquiry based on context above as truthfully as possible, providing in-depth explanations together with answers to the inquiries.
                        You may rephrase the final response to make it concise and sound more human-like, but do not go out of context and do not lose important details and meaning.

                        You'll be asked about repairs, coverage policy and service questions about home appliances, home fixtures, home care, repairs/replacement and cleaning, and also about the renewal, cancellation or refund policies in the contract, whether a certain service is covered under the contract and similar context.

                        The contract context given will have information about contractual details, terms and conditions, renewals, cancellation, refund and service request policies, the coverage limits, limitation and exclusion policies. You will need to use and infer from all the information available in context to analyze and then respond with the final answer.

                        If the question is about a square feet limit, make sure to compare the numerical values properly. 
                        For example,
                        Question: "will my 800 square feet guest house be covered?"
                        The answer to this question will be No, as the square feet limit for guest houses is 750 and 800 is greater than 750.

                        If the inquiry is unrelated to home repair and service, answer with "I don't have the information to answer this question.". For example, questions like "Tell me about space.", "Write a poem for me.", "Where can I buy a refrigerator?", "Hi! How are you?", etc. are out of context.

                        Always include the appliance name in the answer and provide in depth information.

                        Make the answer as short as possible with in depth information.

                        Question: """ + standalone_result + """
                        Answer: """

                        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context"])
                        chain_type_kwargs = {"prompt": PROMPT}
                        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, verbose=True,
                                                        chain_type_kwargs=chain_type_kwargs)

                        qa_resp = qa.invoke(
                            {"query": standalone_result},
                            config={"callbacks": [handler]},
                        )
                        agent_resp = qa_resp["result"] if isinstance(qa_resp, dict) else qa_resp
                        res2, tok2 = handler.infi()
                        llm_trace_to_jaeger(res2, tok2)
                        b = threading.Thread(target=token_calculator, args=(tok2,))
                        b.start()
                    
                    with tracer.start_as_current_span('relevant_documents'):
                        print(
                            "[CHUNKS] /start(Search): calling relevant_docs for entered_query "
                            f"'{str(entered_query)[:200]}'"
                        )
                        relevant_documents = relevant_docs(entered_query, retriever=retriever)
                        print(
                            "[CHUNKS] /start(Search): relevant_documents built "
                            f"len={len(relevant_documents)}"
                        )

            elif gpt_model == "Infer":
                with tracer.start_as_current_span('Infer') as parent1:
                    with tracer.start_as_current_span('llm-retriever-initialization'):
                        llm3 = ChatOpenAI(temperature=0.0, model="ft:gpt-3.5-turbo-0613:mindstix::8YYD56aA")
                        llm = ChatOpenAI(temperature=0.0, model='gpt-4o')
                        llm2 = ChatOpenAI(temperature=0.0, model='gpt-4o')
                        retriever = vector_db1.as_retriever(search_kwargs={"k": MILVUS_RETRIEVER_K})
                        
                    with tracer.start_as_current_span('memory_update'):
                        memory1.clear()
                        question1 = ""
                        answer1 = ""
                        if conversation_id is not None and conversation_id != "":
                            docs = read_qna(email_id=user_email, conversation_id=conversation_id)
                            if docs and "chats" in docs and len(docs["chats"]) > 0:
                                # Use the last *real* user Q&A pairs as memory. Skip system/synthetic chats.
                                skip_chat_ids = {"final_answer", "claim_decision", "case_closed"}
                                skip_entered = {"Final Answer for transcript"}
                                pairs = []
                                for c in reversed(docs.get("chats") or []):
                                    if not isinstance(c, dict):
                                        continue
                                    cid = str(c.get("chat_id") or "").strip()
                                    q = str(c.get("entered_query") or "").strip()
                                    a = str(c.get("response") or "").strip()
                                    if not q or not a:
                                        continue
                                    if cid in skip_chat_ids or q in skip_entered:
                                        continue
                                    if a == "Loading Response":
                                        continue
                                    pairs.append((q, a))
                                    if len(pairs) >= 3:
                                        break

                                if pairs:
                                    question1, answer1 = pairs[0]

                                for q, a in reversed(pairs):
                                    memory1.add_message(HumanMessage(content=q))
                                    memory1.add_message(AIMessage(content=a))

                    with tracer.start_as_current_span('standalone-prompt-chain') as p:
                        standalone_prompt = ChatPromptTemplate.from_template(
                            """
                            Identify if the current question is related to previous question and answer and Create a standalone question in its own language by analyzing previous question, answer to the previous question and current question.
                            If the current question is not related to previous question and answer, then return the current question as standalone question. If the previous question and answer is not available, then return current question as standalone question. you have analyze if the component or appliance mentioned in the current question is related to the component or appliance mentioned in the previoius question and answer. based on that create the standalone question.
                            standalone question should always contain the appliance name, unless it is a service related question. questions related to modifications, code violation upgrades and permits are not bound to any appliance, so do not rephrase the question and do not relate this to any appliance related question.
                            Always only return the output.
                            previous question: """ + question1 + """
                            answer of previous question: """ + answer1 + """
                            current question: """ + entered_query + """
                    
                            examples:
                            If there is no previous question or previous answer, then do not create the standalone question at all.
                            1)  previous question:''
                                answer of previous question: ''
                                current question: is the Fridge covered?
                                standalone question: is the Fridge covered?
                                
                            If there is secondary damage to the appliance being talked, create a standalone question in following way.
                            2)  previous question: my oven caught fire, is the oven covered?
                                answer of the previous question:Yes, your oven is covered by the plan. The plan covers all parts and components of installed ranges, ovens, and cooktops, including burner, display, self-clean, igniter, element, control panel and board, oven heating element, and temperature sensor. However, there are certain limitations and exclusions that apply, so it's important to review the specific terms and conditions of your plan for more details.
                                current question: this fire has damaged the exhaust fan located above it, is it covered?
                                standalone question: is the secondary damaged caused by the fire in the oven to the exhaust fan covered? 
                    
                            In some of the cases, current question wont need rephrasing, for example:
                            
                            3)  previous question: is the washer covered
                                answer of previous question: yes, washer is covered under the contract.
                                current question: there is damage to air conditioning unit because of leak but it is secondary damage, is it covered?
                                standalone question: there is damage to air conditioning unit because of leak but it is secondary damage, is it covered?
                            
                                    """
                        )
                        start = int(time())
                        standalone_chain = standalone_prompt | llm3 | StrOutputParser()

                        standalone_result = standalone_chain.invoke({"input": entered_query})
                        print(standalone_result)
                        res1, tok1 = handler.infi()
                        llm_trace_to_jaeger(res1, tok1)
                        a = threading.Thread(target=token_calculator, args=(tok1,))
                        a.start()

                    with tracer.start_as_current_span('q_monitor') as parentq:
                        t = threading.Thread(target=q_monitor, args=(parentq,entered_query,))
                        t.start()

                    with tracer.start_as_current_span('llm-RetrievalQA-chain') as q:
                        qa = RetrievalQA.from_chain_type(llm=llm2, retriever=retriever, verbose=True)
                        agent_response = input_prompt(standalone_result, qa, llm)
                        agent_resp = agent_response["output"]
                        res2, tok2 = handler.infi()
                        llm_trace_to_jaeger(res2, tok2)
                        b = threading.Thread(target=token_calculator, args=(tok2,))
                        b.start()
                    
                    with tracer.start_as_current_span('relevant_documents'):
                        knowledge_base_thoughts = [
                            item[0].tool_input
                            for item in agent_response["intermediate_steps"]
                            if item[0].tool == 'Knowledge Base'
                        ]
                        print(
                            "[CHUNKS] /start(Infer): knowledge_base_thoughts_count="
                            f"{len(knowledge_base_thoughts)}"
                        )
                        relevant_documents = ""
                        for idx, action_input in enumerate(knowledge_base_thoughts):
                            print(
                                "[CHUNKS] /start(Infer): calling relevant_docs for KB thought "
                                f"index={idx}, input_preview='{str(action_input)[:200]}'"
                            )
                            rd = relevant_docs(action_input, retriever)
                            print(
                                "[CHUNKS] /start(Infer): returned from relevant_docs "
                                f"index={idx}, len={len(rd)}"
                            )
                            relevant_documents += rd
            else:
                return jsonify({"error": f"Invalid gpt_model: {gpt_model}. Must be 'Search' or 'Infer'"}), 400

            with tracer.start_as_current_span('output-formating'):
                # Validate that we have a response
                if agent_resp is None:
                    return jsonify({"error": "Invalid gpt_model. Must be 'Search' or 'Infer'"}), 400
                
                ai_response = agent_resp

                word_count = len(relevant_documents.split())
                latency = time() - start_time

                query_time = datetime.now()

                chat = {
                    "chat_id": str(uuid.uuid4()),
                    "entered_query": entered_query,
                    "response": ai_response,
                    "relevant_docs": relevant_documents,
                    "gpt_model": gpt_model,
                    "chat_timestamp": query_time,
                    "latency": latency,
                    "word_count": word_count
                }

                if conversation_id is None or conversation_id == "":
                    print(
                        "[CHUNKS] /start: creating NEW conversation document with "
                        f"relevant_docs_len={len(relevant_documents)}"
                        f"R_D:  {relevant_documents}"
                    )
                    qna_json = {
                        "conversation_name": entered_query,
                        "contract_type": contract_type,
                        "selected_plan": selected_plan,
                        "selected_state": selected_state,
                        "query_time": query_time,
                        "status": "active",
                        "conversation_mode": gpt_model,
                        "chats": [chat],
                    }

                    conversation_id = insert_qna(email_id=user_email, data=qna_json)
                    conversation_id = conversation_id.inserted_id

                else:
                    print(
                        "[CHUNKS] /start: updating EXISTING conversation "
                        f"{conversation_id} with relevant_docs_len={len(relevant_documents)}"
                    )
                    add_chat = update_chat(
                        new_data=chat, conversation_id=conversation_id, email_id=user_email
                    )
                    # Keep conversation_mode updated for filtering in the sidebar.
                    # IMPORTANT: For transcript conversations, we must NOT change conversation_mode away from "Calls",
                    # otherwise the case disappears from the Calls list when asking follow-up questions.
                    try:
                        qna_collection_user = f"chats_{user_email}"
                        qna_collection = db[qna_collection_user]
                        existing = qna_collection.find_one(
                            {"_id": ObjectId(conversation_id)},
                            {"_id": 0, "doc_type": 1, "conversation_mode": 1},
                        ) or {}
                        if existing.get("doc_type") != "transcript_conversation":
                            qna_collection.update_one(
                                {"_id": ObjectId(conversation_id)},
                                {"$set": {"conversation_mode": gpt_model}},
                            )
                    except Exception:
                        pass

                output_json = {"aiResponse": ai_response, "conversationId": str(conversation_id), "chatId":chat.get("chat_id")}

        return make_response(jsonify(output_json), 200)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in /start endpoint: {str(e)}")
        print(f"Traceback: {error_trace}")
        return jsonify({"error": "An error occurred while processing your request", "details": str(e)}), 500


@app.route("/calls/start", methods=["POST"])
def calls_start():
    try:
        authorization_header = request.headers.get("Authorization")

        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401

        if authorization_header:
            token_data = token_process(authorization_header)

            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]

        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is missing or invalid"}), 400

        contract_type = data.get("contractType")
        selected_plan = data.get("selectedPlan")
        selected_state = data.get("selectedState")
        entered_query = data.get("enteredQuery")

        if not all([contract_type, selected_plan, selected_state, entered_query]):
            return jsonify(
                {
                    "error": "Missing required fields: contractType, selectedPlan, selectedState, enteredQuery"
                }
            ), 400

        user_email = token_data[0]["email"]
        conversation_id = request.args.get("conversation-id")

        if conversation_id is None or conversation_id == "":
            return jsonify({"error": "Calls conversationId is required"}), 400

        try:
            calls_conversation = calls_conversations_collection.find_one(
                {"_id": ObjectId(conversation_id), "user_email": user_email}
            )
        except Exception:
            calls_conversation = None

        if not calls_conversation:
            return jsonify({"error": "Calls conversation not found"}), 404

        query_time = datetime.now()

        chat = {
            "chat_id": str(uuid.uuid4()),
            "entered_query": entered_query,
            "response": f"You are in Calls mode. This is a placeholder response for: {entered_query}",
            "gpt_model": "Calls",
            "chat_timestamp": query_time,
        }

        calls_conversations_collection.update_one(
            {"_id": ObjectId(conversation_id)},
            {
                "$push": {"chats": chat},
                "$set": {
                    "contract_type": contract_type,
                    "selected_plan": selected_plan,
                    "selected_state": selected_state,
                    "updated_at": query_time,
                },
            },
        )

        output_json = {
            "aiResponse": chat["response"],
            "conversationId": str(conversation_id),
            "chatId": chat.get("chat_id"),
        }

        return make_response(jsonify(output_json), 200)
    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        print(f"Error in /calls/start endpoint: {str(e)}")
        print(f"Traceback: {error_trace}")
        return (
            jsonify(
                {
                    "error": "An error occurred while processing your request",
                    "details": str(e),
                }
            ),
            500,
        )


@app.route("/history", methods=["GET"])
def chat_history():
    with tracer.start_as_current_span('api/history'):
        authorization_header = request.headers.get("Authorization")

        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401

        if authorization_header:
            token_data = token_process(authorization_header)

            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]

        conversation_id = request.args.get("conversation-id")
        user_email = token_data[0]["email"]

        docs = read_qna(email_id=user_email, conversation_id=conversation_id)
        if not docs:
            return make_response(
                jsonify({"message": "No data found in the specified conversation"}), 404
            )

        feedback_collection_user = f"feedbacks_{user_email}"
        feedback_collection = db[feedback_collection_user]
        feedback_reaction = feedback_collection.find(
            {"conversation_id": str(conversation_id)}
        )
        feedback_dict = {}

        for doc in feedback_reaction:
            chat_id = str(doc["chat_id"])
            feedback_dict[chat_id] = doc["reaction"]

        chats = docs["chats"]
        # Do not return synthetic/system chats in the UI.
        chats = [
            c
            for c in (chats or [])
            if str((c or {}).get("chat_id") or "") not in ("claim_decision", "case_closed")
        ]
        for chat in chats:
            chat_id = chat.get("chat_id")
            if chat_id in feedback_dict:
                chat["reaction"] = feedback_dict[chat_id]
            # Normalize chunks to ensure they have names
            if "relevant_chunks" in chat:
                print(f"\n[HISTORY_API] ===== Normalizing chunks for chat_id: {chat_id} =====")
                print(f"[HISTORY_API] Original chunks type: {type(chat['relevant_chunks'])}")
                print(f"[HISTORY_API] Original chunks count: {len(chat['relevant_chunks']) if isinstance(chat['relevant_chunks'], list) else 'not_list'}")
                if isinstance(chat["relevant_chunks"], list) and len(chat["relevant_chunks"]) > 0:
                    for idx, c in enumerate(chat["relevant_chunks"][:3], start=1):
                        print(f"[HISTORY_API] Original chunk {idx} type: {type(c)}")
                        if isinstance(c, dict):
                            print(f"[HISTORY_API] Original chunk {idx} keys: {list(c.keys())}")
                            print(f"[HISTORY_API] Original chunk {idx} NAME: '{c.get('name', 'MISSING NAME!')}'")
                            if c.get('metadata'):
                                source = c.get('metadata', {}).get('source', '')
                                print(f"[HISTORY_API] Original chunk {idx} metadata.source: {source}")
                        else:
                            print(f"[HISTORY_API] Original chunk {idx} is STRING: {str(c)[:50]}")
                
                normalized_chunks = _normalize_chunks_with_names(chat["relevant_chunks"])
                chat["relevant_chunks"] = normalized_chunks
                chat["relevantChunks"] = normalized_chunks
                
                print(f"[HISTORY_API] After normalization:")
                for idx, c in enumerate(normalized_chunks[:3], start=1):
                    if isinstance(c, dict):
                        print(f"[HISTORY_API]   Normalized chunk {idx} NAME: '{c.get('name', 'STILL MISSING!')}'")
                    else:
                        print(f"[HISTORY_API]   Normalized chunk {idx} is not dict!")
                print(f"[HISTORY_API] ===========================================\n")
            elif "relevantChunks" in chat:
                normalized_chunks = _normalize_chunks_with_names(chat["relevantChunks"])
                chat["relevantChunks"] = normalized_chunks
            # Normalize chunk fields for frontend consumption (keep backwards-compatible snake_case too)
            if "relevant_chunks" in chat and "relevantChunks" not in chat:
                chat["relevantChunks"] = chat.get("relevant_chunks")
            if "underlying_model" in chat and "underlyingModel" not in chat:
                chat["underlyingModel"] = chat.get("underlying_model")

        # IMPORTANT:
        # Transcript (Claims/Calls) conversations must always be treated as "Calls" mode for UI routing.
        # Older versions of the app could accidentally overwrite conversation_mode to "Search"/"Infer"
        # when asking follow-up questions. We correct for that here so refresh behaves correctly.
        is_transcript_conv = (docs.get("doc_type") == "transcript_conversation") or bool(docs.get("transcript_id"))
        effective_mode = "Calls" if is_transcript_conv else (
            docs.get("conversation_mode") or (chats[0].get("gpt_model") if chats else None)
        )

        output_json = {
            "conversationName": docs.get("conversation_name"),
            "contractType": docs.get("contract_type"),
            "selectedPlan": docs.get("selected_plan"),
            "selectedState": docs.get("selected_state"),
            "status": docs.get("status", "active"),
            # Final disposition for this case (set on Approve/Reject & Proceed flows)
            "caseDisposition": docs.get("case_disposition"),
            "closedAt": (
                (docs.get("closed_at").isoformat() + "Z")
                if docs.get("closed_at")
                else None
            ),
            "reviewComments": docs.get("review_comments"),
            # Transcript conversations can be mid-processing; expose this so the UI can show a loader.
            "processing": bool(docs.get("processing", False)),
            "chats": chats,
            "createdAt": (
                (docs.get("created_at").isoformat() + "Z")
                if docs.get("created_at")
                else None
            ),
            "updatedAt": (
                (docs.get("updated_at").isoformat() + "Z")
                if docs.get("updated_at")
                else None
            ),
            # For transcript conversations we force Calls mode so refresh doesn't jump to Search history.
            # Underlying Search/Infer model is still stored per-chat / in `underlying_model`.
            "gptModel": effective_mode,
            "finalSummary": docs.get("final_summary"),
            "claimDecision": docs.get("claim_decision"),
            "authorizedFinalAnswer": docs.get("authorized_final_answer"),
            "authorizedApprovedAt": (
                (docs.get("authorized_approved_at").isoformat() + "Z")
                if docs.get("authorized_approved_at")
                else None
            ),
            "transcriptId": docs.get("transcript_id"),
            "transcriptMetadata": docs.get("transcript_metadata"),
        }
        return make_response(jsonify(output_json), 200)


@app.route("/conversation/authorize", methods=["PATCH"])
def authorize_conversation_answer():
    """Store an agent-authorized final answer for a conversation and (optionally) close it.

    Query params:
      - conversation-id (str)

    Body:
      - authorizedFinalAnswer (str, required)
      - status (optional): 'inactive' | 'active' (defaults to 'inactive')
      - reviewComments (optional): string
    """
    try:
        with tracer.start_as_current_span("api/conversation/authorize"):
            authorization_header = request.headers.get("Authorization")

            if authorization_header is None:
                return jsonify({"message": "Token is missing"}), 401

            if authorization_header:
                token_data = token_process(authorization_header)
                if token_data[1] == 401 or token_data[1] == 403:
                    return (token_data[0].get_json()), token_data[1]

            conversation_id = request.args.get("conversation-id")
            if not conversation_id:
                return jsonify({"error": "conversation-id is required"}), 400

            data = request.get_json() or {}
            authorized_final_answer = (data.get("authorizedFinalAnswer") or "").strip()
            if not authorized_final_answer:
                return jsonify({"error": "authorizedFinalAnswer is required"}), 400

            status = (data.get("status") or "inactive").strip().lower()
            if status not in ("active", "inactive"):
                return jsonify({"error": "status must be 'active' or 'inactive'"}), 400

            review_comments = (data.get("reviewComments") or "").strip()

            user_email = token_data[0]["email"]
            qna_collection_user = f"chats_{user_email}"
            qna_collection = db[qna_collection_user]

            now_ts = datetime.utcnow()
            now_iso = now_ts.isoformat() + "Z"
            closed_at = now_ts if status == "inactive" else None
            updated = qna_collection.find_one_and_update(
                {"_id": ObjectId(conversation_id)},
                {
                    "$set": {
                        "authorized_final_answer": authorized_final_answer,
                        "authorized_approved_at": now_ts,
                        "status": status,
                        "case_disposition": "approved",
                        "closed_at": closed_at,
                        "review_comments": review_comments if review_comments else None,
                        "updated_at": now_ts,
                    }
                },
                return_document=ReturnDocument.AFTER,
            )
            if not updated:
                return jsonify({"error": "Conversation not found"}), 404

            # Keep cached payload consistent for transcript conversations (if present).
            try:
                if updated.get("response_payload"):
                    qna_collection.update_one(
                        {"_id": ObjectId(conversation_id)},
                        {
                            "$set": {
                                "response_payload.status": status,
                                "response_payload.authorizedFinalAnswer": authorized_final_answer,
                                "response_payload.authorizedApprovedAt": now_iso,
                                "response_payload.caseDisposition": "approved",
                                "response_payload.closedAt": (closed_at.isoformat() + "Z") if closed_at else None,
                                "response_payload.reviewComments": review_comments if review_comments else None,
                            }
                        },
                    )
            except Exception:
                pass

            return (
                jsonify(
                    {
                        "conversationId": conversation_id,
                        "status": status,
                        "caseDisposition": "approved",
                        "closedAt": (closed_at.isoformat() + "Z") if closed_at else None,
                        "reviewComments": review_comments if review_comments else None,
                        "authorizedFinalAnswer": authorized_final_answer,
                        "authorizedApprovedAt": now_iso,
                    }
                ),
                200,
            )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/conversation/status", methods=["PATCH"])
def update_conversation_status():
    """Set a conversation status in MongoDB (per user).

    Query params:
      - conversation-id (str)

    Body:
      - status: 'active' | 'inactive'
    """
    try:
        with tracer.start_as_current_span('api/conversation/status'):
            authorization_header = request.headers.get("Authorization")

            if authorization_header is None:
                return jsonify({"message": "Token is missing"}), 401

            if authorization_header:
                token_data = token_process(authorization_header)
                if token_data[1] == 401 or token_data[1] == 403:
                    return (token_data[0].get_json()), token_data[1]

            conversation_id = request.args.get("conversation-id")
            if not conversation_id:
                return jsonify({"error": "conversation-id is required"}), 400

            data = request.get_json() or {}
            status = (data.get("status") or "").strip().lower()
            if status not in ("active", "inactive"):
                return jsonify({"error": "status must be 'active' or 'inactive'"}), 400

            user_email = token_data[0]["email"]
            qna_collection_user = f"chats_{user_email}"
            qna_collection = db[qna_collection_user]

            updated = qna_collection.find_one_and_update(
                {"_id": ObjectId(conversation_id)},
                {"$set": {"status": status, "updated_at": datetime.utcnow()}},
                return_document=ReturnDocument.AFTER,
            )
            if not updated:
                return jsonify({"error": "Conversation not found"}), 404

            # Keep cached payload consistent for transcript conversations (if present).
            try:
                if updated.get("response_payload"):
                    qna_collection.update_one(
                        {"_id": ObjectId(conversation_id)},
                        {"$set": {"response_payload.status": status}},
                    )
            except Exception:
                pass

            return jsonify({"conversationId": conversation_id, "status": status}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/conversation/close", methods=["PATCH"])
def close_conversation():
    """Close a conversation and persist its final disposition (Approve/Reject & Proceed).

    Query params:
      - conversation-id (str)

    Body:
      - disposition (required): 'approved' | 'rejected'
      - reviewComments (optional): string
    """
    try:
        with tracer.start_as_current_span("api/conversation/close"):
            authorization_header = request.headers.get("Authorization")
            if authorization_header is None:
                return jsonify({"message": "Token is missing"}), 401

            if authorization_header:
                token_data = token_process(authorization_header)
                if token_data[1] == 401 or token_data[1] == 403:
                    return (token_data[0].get_json()), token_data[1]

            conversation_id = request.args.get("conversation-id")
            if not conversation_id:
                return jsonify({"error": "conversation-id is required"}), 400

            data = request.get_json() or {}
            disposition = (data.get("disposition") or "").strip().lower()
            if disposition not in ("approved", "rejected"):
                return jsonify({"error": "disposition must be 'approved' or 'rejected'"}), 400
            review_comments = (data.get("reviewComments") or "").strip()

            user_email = token_data[0]["email"]
            qna_collection_user = f"chats_{user_email}"
            qna_collection = db[qna_collection_user]

            now_ts = datetime.utcnow()
            now_iso = now_ts.isoformat() + "Z"
            updated = qna_collection.find_one_and_update(
                {"_id": ObjectId(conversation_id)},
                {
                    "$set": {
                        "status": "inactive",
                        "case_disposition": disposition,
                        "closed_at": now_ts,
                        "review_comments": review_comments if review_comments else None,
                        "updated_at": now_ts,
                    }
                },
                return_document=ReturnDocument.AFTER,
            )
            if not updated:
                return jsonify({"error": "Conversation not found"}), 404

            # Keep cached payload consistent for transcript conversations (if present).
            try:
                if updated.get("response_payload"):
                    qna_collection.update_one(
                        {"_id": ObjectId(conversation_id)},
                        {
                            "$set": {
                                "response_payload.status": "inactive",
                                "response_payload.caseDisposition": disposition,
                                "response_payload.closedAt": now_iso,
                                "response_payload.reviewComments": review_comments if review_comments else None,
                            }
                        },
                    )
            except Exception:
                pass

            return (
                jsonify(
                    {
                        "conversationId": conversation_id,
                        "status": "inactive",
                        "caseDisposition": disposition,
                        "closedAt": now_iso,
                        "reviewComments": review_comments if review_comments else None,
                    }
                ),
                200,
            )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _build_claims_case_context_for_llm(docs: dict) -> str:
    """Build a compact, LLM-friendly context pack for Claims follow-up chat.

    Goal: allow answering many questions about the case without vector DB retrieval, using only
    transcript processing outputs + prior follow-up chat in this conversation.
    """
    if not isinstance(docs, dict):
        return ""

    chats = docs.get("chats") or []
    final_summary = (docs.get("final_summary") or "").strip()
    authorized_final = (docs.get("authorized_final_answer") or "").strip()
    claim_decision = docs.get("claim_decision")
    transcript_meta = docs.get("transcript_metadata") or {}
    transcript_id = docs.get("transcript_id") or ""
    contract_type = (docs.get("contract_type") or "").strip()
    selected_plan = (docs.get("selected_plan") or "").strip()
    selected_state = (docs.get("selected_state") or "").strip()
    plan_overview = (docs.get("plan_overview") or "").strip()

    extracted = []
    followups = []

    for c in chats:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("chat_id") or "")
        q = str(c.get("entered_query") or "").strip()
        a = str(c.get("response") or "").strip()
        if not q and not a:
            continue

        # Transcript extracted Qs are stored with ids like q1, q2, ...
        if isinstance(cid, str) and re.match(r"^q\d+$", cid, re.IGNORECASE):
            extracted.append((cid, q, a))
            continue

        # Skip final analyzed answer from the follow-up timeline (it will be included separately).
        if cid == "final_answer" or q == "Final Answer for transcript":
            continue

        # Follow-up chat (user asks about the case)
        if q or a:
            followups.append((q, a))

    # Keep only the most useful slices
    extracted = extracted[:25]
    followups = followups[-12:]

    parts = []
    parts.append("CASE CONTEXT (Claims transcript conversation)")
    if transcript_id:
        parts.append(f"- transcriptId: {transcript_id}")
    if contract_type or selected_plan or selected_state:
        # Make plan metadata visible to the LLM (even if retrieval fails), but keep it compact.
        parts.append(
            f"- plan: state={selected_state or '(unknown)'}, contractType={contract_type or '(unknown)'}, selectedPlan={selected_plan or '(unknown)'}"
        )
    if isinstance(transcript_meta, dict) and transcript_meta:
        fn = transcript_meta.get("fileName") or transcript_meta.get("name") or ""
        if fn:
            parts.append(f"- transcriptFileName: {fn}")
        ud = transcript_meta.get("uploadDate")
        if ud:
            parts.append(f"- uploadDate: {ud}")
    parts.append(f"- status: {docs.get('status')}")
    if docs.get("case_disposition"):
        parts.append(f"- disposition: {docs.get('case_disposition')}")
    parts.append("")

    if plan_overview:
        parts.append("PLAN OVERVIEW (CACHED)")
        parts.append(plan_overview)
        parts.append("")

    if final_summary:
        parts.append("FINAL ANALYZED ANSWER")
        parts.append(final_summary)
        parts.append("")

    if authorized_final:
        parts.append("AUTHORIZED FINAL ANSWER (if reviewer edited)")
        parts.append(authorized_final)
        parts.append("")

    if claim_decision is not None:
        parts.append("CLAIM DECISION (JSON)")
        try:
            parts.append(json.dumps(claim_decision, ensure_ascii=False, indent=2))
        except Exception:
            parts.append(str(claim_decision))
        parts.append("")

    if extracted:
        parts.append("EXTRACTED CUSTOMER QUERIES + AI DRAFT ANSWERS")
        for cid, q, a in extracted:
            if q:
                parts.append(f"- {cid}: {q}")
            if a:
                parts.append(f"  answer: {a}")
        parts.append("")

    if followups:
        parts.append("RECENT FOLLOW-UP CHAT HISTORY")
        for q, a in followups:
            if q:
                parts.append(f"- User: {q}")
            if a:
                parts.append(f"  Assistant: {a}")
        parts.append("")

    return "\n".join(parts).strip()


def _retrieve_policy_chunks_for_claims(docs: dict, query: str, k: int = 6):
    """Best-effort policy retrieval from Milvus for a transcript conversation.

    Returns: (chunks_for_ui, referred_docs_text)
      - chunks_for_ui: list[dict] where each dict has {content, metadata}
      - referred_docs_text: a readable text blob for /referred-clauses legacy page
    """
    # Entry point logging
    print(f"\n{'='*80}")
    print(f"[CHUNKS_DEBUG] _retrieve_policy_chunks_for_claims CALLED")
    print(f"[CHUNKS_DEBUG] Query: {query[:100] if query else 'EMPTY'}")
    print(f"[CHUNKS_DEBUG] Docs type: {type(docs)}")
    print(f"{'='*80}\n")
    
    try:
        if not isinstance(docs, dict):
            print(f"[CHUNKS_DEBUG] ERROR: docs is not a dict, returning empty")
            return [], ""
        query = (query or "").strip()
        if not query:
            print(f"[CHUNKS_DEBUG] ERROR: query is empty, returning empty")
            return [], ""

        contract_type = docs.get("contract_type")
        selected_plan = docs.get("selected_plan")
        selected_state = docs.get("selected_state")
        
        # Log the extracted values
        print(f"[CHUNKS_DEBUG] contract_type={contract_type}")
        print(f"[CHUNKS_DEBUG] selected_plan={selected_plan}")
        print(f"[CHUNKS_DEBUG] selected_state={selected_state}")
        
        if not all([contract_type, selected_plan, selected_state]):
            print(f"[CHUNKS_DEBUG] ERROR: Missing required fields (contract_type, selected_plan, or selected_state), returning empty")
            return [], ""

        milvus_state = normalize_state_for_milvus(selected_state)
        contract_type_norm = normalize_contract_type(contract_type)
        selected_plan_norm = normalize_plan_for_milvus(contract_type_norm, selected_plan)

        collection_mapping = {
            "RE": {
                "ShieldEssential": f"{milvus_state}_RE_ShieldEssential",
                "ShieldPlus": f"{milvus_state}_RE_ShieldPlus",
                "default": f"{milvus_state}_RE_ShieldComplete",
            },
            "DTC": {
                "ShieldSilver": f"{milvus_state}_DTC_ShieldSilver",
                "ShieldGold": f"{milvus_state}_DTC_ShieldGold",
                "default": f"{milvus_state}_DTC_ShieldPlatinum",
            },
        }

        selected_collection_name = collection_mapping.get(contract_type_norm, {}).get(
            selected_plan_norm, collection_mapping.get(contract_type_norm, {}).get("default")
        )
        
        # Log collection name
        print(f"[CHUNKS_DEBUG] Normalized: state={milvus_state}, contract={contract_type_norm}, plan={selected_plan_norm}")
        print(f"[CHUNKS_DEBUG] Selected collection: {selected_collection_name}")
        
        if not selected_collection_name:
            print(f"[CHUNKS_DEBUG] ERROR: No collection name found, returning empty")
            return [], ""

        # Lightweight logging to debug "no clauses found" issues in claims follow-up.
        try:
            print(
                "[CLAIMS_FOLLOWUP] Milvus retrieval "
                f"state={selected_state!r}->{milvus_state!r}, "
                f"contract_type={contract_type!r}->{contract_type_norm!r}, "
                f"selected_plan={selected_plan!r}->{selected_plan_norm!r}, "
                f"collection={selected_collection_name!r}, "
                f"k={k}"
            )
        except Exception:
            pass

        print(f"[CHUNKS_DEBUG] Connecting to Milvus...")
        vector_db1: Milvus = Milvus(
            embed,
            collection_name=selected_collection_name,
            connection_args={"host": MILVUS_HOST, "port": "19530"},
        )
        retriever = vector_db1.as_retriever(search_kwargs={"k": max(1, min(int(k), 12))})

        print(f"[CHUNKS_DEBUG] Querying Milvus with query: {query[:200]}...")
        raw_docs = retriever.get_relevant_documents(query)
        
        # Detailed logging after retrieval
        print(f"[CHUNKS_DEBUG] Milvus returned {len(raw_docs or [])} documents")
        print(f"[CHUNKS_DEBUG] raw_docs type: {type(raw_docs)}")
        
        try:
            print(f"[CLAIMS_FOLLOWUP] Milvus returned {len(raw_docs or [])} docs")
        except Exception:
            pass
            
        chunks_for_ui = []
        text_lines = []
        
        # Check if we have any docs
        if not raw_docs or len(raw_docs) == 0:
            print(f"[CHUNKS_DEBUG] WARNING: No documents retrieved from Milvus!")
            return [], ""
        
        print(f"[CHUNKS_DEBUG] Processing {len(raw_docs)} documents...")
        
        for i, d in enumerate(raw_docs or [], start=1):
            print(f"\n[CHUNKS_DEBUG] ---------- Processing Document {i} ----------")
            
            # Check document type and attributes
            print(f"[CHUNKS_DEBUG] Document type: {type(d)}")
            print(f"[CHUNKS_DEBUG] Document attributes: {dir(d)}")
            
            content = (getattr(d, "page_content", "") or "").strip()
            metadata = getattr(d, "metadata", {}) or {}
            
            print(f"[CHUNKS_DEBUG] Content length: {len(content)}")
            print(f"[CHUNKS_DEBUG] Content preview (first 150 chars): {content[:150]}...")
            print(f"[CHUNKS_DEBUG] Metadata type: {type(metadata)}")
            
            if not content:
                print(f"[CHUNKS_DEBUG] SKIPPING: No content in document {i}")
                continue
            
            # Detailed metadata logging
            if isinstance(metadata, dict):
                print(f"[CHUNKS_DEBUG] Metadata keys: {list(metadata.keys())}")
                print(f"[CHUNKS_DEBUG] Full metadata:")
                for key, value in metadata.items():
                    print(f"[CHUNKS_DEBUG]   {key}: {value}")
                print(f"[CHUNKS_DEBUG] Metadata JSON: {json.dumps(metadata, default=str, indent=2)}")
            else:
                print(f"[CHUNKS_DEBUG] Metadata is not a dict: {metadata}")
            
            # Generate meaningful name from metadata
            chunk_name = _generate_chunk_name(metadata, i)
            
            # Log the generated name
            print(f"[CHUNKS_DEBUG] Generated chunk name: '{chunk_name}'")
            
            # Add chunk with name, content, and metadata
            chunk_obj = {
                "content": content, 
                "metadata": metadata,
                "name": chunk_name  # Add meaningful name
            }
            chunks_for_ui.append(chunk_obj)
            
            # Log what we're adding - show the actual name being stored
            print(f"[CHUNKS_DEBUG] Chunk object being added:")
            print(f"[CHUNKS_DEBUG]   name: '{chunk_obj.get('name')}'")
            print(f"[CHUNKS_DEBUG]   content length: {len(chunk_obj.get('content', ''))}")
            print(f"[CHUNKS_DEBUG]   Full object (truncated): {json.dumps({k: (v[:50] + '...' if isinstance(v, str) and len(v) > 50 else v) for k, v in chunk_obj.items() if k != 'metadata'}, default=str)}")
            
            # Build a readable blob for legacy referred clauses page
            # Use the generated name instead of generic "Clause {i}"
            header = chunk_name
            text_lines.append(header)
            text_lines.append(content)
            text_lines.append("")
            
            print(f"[CHUNKS_DEBUG] Added chunk {i} to chunks_for_ui with name: '{chunk_name}'")
            print(f"[CHUNKS_DEBUG] ----------------------------------------\n")

        print(f"[CHUNKS_DEBUG] FINAL: Total chunks_for_ui: {len(chunks_for_ui)}")
        print(f"[CHUNKS_DEBUG] FINAL: Total text_lines: {len(text_lines)}")
        
        referred_docs_text = "\n".join(text_lines).strip()
        return chunks_for_ui, referred_docs_text
    except Exception as e:
        print(f"[CHUNKS_DEBUG] EXCEPTION in _retrieve_policy_chunks_for_claims: {e}")
        import traceback
        traceback.print_exc()
        print(f"Warning: policy retrieval failed for claims followup: {e}")
        return [], ""


def _normalize_chunks_with_names(chunks: list) -> list:
    """
    Normalize chunks to ensure they have a 'name' field.
    If chunks are strings, convert them to objects.
    If chunks are objects without names, generate names from metadata.
    """
    if not isinstance(chunks, list):
        return []
    
    normalized = []
    for idx, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, str):
            # Convert string to object
            normalized.append({
                "content": chunk,
                "name": f"Clause {idx}"
            })
        elif isinstance(chunk, dict):
            # Ensure it has a name field
            if "name" not in chunk or not chunk.get("name"):
                # Generate name from metadata if available
                metadata = chunk.get("metadata", {}) or {}
                chunk["name"] = _generate_chunk_name(metadata, idx)
            normalized.append(chunk)
        else:
            # Fallback for unknown types
            normalized.append({
                "content": str(chunk),
                "name": f"Clause {idx}"
            })
    
    return normalized


def _generate_chunk_name(metadata: dict, index: int) -> str:
    """
    Generate a meaningful name for a chunk based on its metadata.
    Priority order:
    1. Extract filename from source path (most common case) - ONLY filename, not full path
    2. section + clause/page number
    3. heading/title + page number
    4. clause number
    5. Fallback to "Clause {index}"
    """
    if not isinstance(metadata, dict):
        return f"Clause {index}"
    
    import os
    
    # Try to extract meaningful identifiers
    source = metadata.get("source") or metadata.get("file") or metadata.get("document") or metadata.get("Source") or ""
    section = metadata.get("section") or metadata.get("Section") or ""
    clause = metadata.get("clause") or metadata.get("Clause") or metadata.get("clause_number") or ""
    page = metadata.get("page") or metadata.get("Page") or metadata.get("page_number") or ""
    heading = metadata.get("heading") or metadata.get("Heading") or metadata.get("title") or metadata.get("Title") or ""
    
    # Build name parts
    name_parts = []
    
    # Priority 1: Extract ONLY filename from source path (most common case)
    if source:
        source_str = str(source).strip()
        
        # Extract just the filename part (last component of path)
        # Use os.path.basename() for reliable cross-platform extraction
        # Convert Windows backslashes to forward slashes first (os.path.basename handles both, but this ensures consistency)
        normalized_path = source_str.replace("\\", "/")
        source_name = os.path.basename(normalized_path)
        
        # Double-check: if basename still contains separators, manually extract last part
        if "/" in source_name or "\\" in source_name:
            # Manual fallback: take last part after any separator
            if "\\" in source_str:
                parts = [p.strip() for p in source_str.split("\\") if p.strip()]
            elif "/" in source_str:
                parts = [p.strip() for p in source_str.split("/") if p.strip()]
            else:
                parts = [source_str]
            source_name = parts[-1] if parts else source_str
        
        print(f"[CHUNKS_DEBUG] 📁 Extracting filename from: '{source_str[:70]}...'")
        print(f"[CHUNKS_DEBUG] 📁 Extracted filename (before extension removal): '{source_name}'")
        
        # Remove file extension for cleaner display (e.g., .docx, .doc, .pdf)
        original_name = source_name
        if "." in source_name:
            source_name = source_name.rsplit(".", 1)[0]
            print(f"[CHUNKS_DEBUG] 📁 Removed extension: '{original_name}' -> '{source_name}'")
        
        # Replace underscores and hyphens with spaces for better readability
        source_name = source_name.replace("_", " ").replace("-", " ")
        
        # Capitalize first letter of each word
        source_name = " ".join(word.capitalize() for word in source_name.split() if word.strip())
        
        # Log the final result
        print(f"[CHUNKS_DEBUG] 📁 FINAL extracted filename: '{source_name}' (from path: '{source_str[:60]}...')")
        
        name_parts.append(source_name)
    
    # Priority 2: Section + Clause
    if section and clause and not name_parts:
        name_parts.append(f"{section} - Clause {clause}")
    elif section and not name_parts:
        name_parts.append(section)
    
    # Priority 3: Heading/Title
    if heading and not name_parts:
        name_parts.append(heading)
    
    # Add page number if available
    if page:
        name_parts.append(f"Page {page}")
    
    # If we have any meaningful parts, join them
    if name_parts:
        return " · ".join(name_parts)
    
    # Fallback to clause number or index
    if clause:
        return f"Clause {clause}"
    
    return f"Clause {index}"


def _looks_like_plan_overview_question(q: str) -> bool:
    """Heuristic: broad plan questions that benefit from a cached plan overview."""
    q = (q or "").strip().lower()
    if not q:
        return False
    needles = [
        "what is covered",
        "what's covered",
        "whats covered",
        "what all is covered",
        "coverage in the plan",
        "plan cover",
        "covered in the plan",
        "what does my plan cover",
        "plan coverage",
        "coverage summary",
        "coverage overview",
    ]
    return any(n in q for n in needles)


def _get_or_build_plan_overview_for_claims(docs: dict) -> str:
    """Best-effort: build a cached plan overview using Milvus clauses, store in Mongo for reuse.

    This is intended to make broad plan questions answerable even if the user doesn't ask a
    clause-shaped question. If Milvus is unreachable or returns no clauses, returns "".
    """
    if not isinstance(docs, dict):
        return ""
    existing = (docs.get("plan_overview") or "").strip()
    if existing:
        return existing

    contract_type = docs.get("contract_type")
    selected_plan = docs.get("selected_plan")
    selected_state = docs.get("selected_state")
    if not all([contract_type, selected_plan, selected_state]):
        return ""

    # Pull a broader set of clauses (k=12 cap inside retrieval) and summarize.
    overview_query = (
        "Provide an overview of what is covered and not covered in this plan, including key limits, "
        "exclusions, and service fees. Keep it structured and concise."
    )
    chunks, _ = _retrieve_policy_chunks_for_claims(docs, overview_query, k=12)
    if not chunks:
        return ""

    clauses_blob = "\n\n".join(
        [str(c.get("content") or "").strip() for c in (chunks or []) if isinstance(c, dict) and str(c.get("content") or "").strip()]
    ).strip()
    if not clauses_blob:
        return ""
    clauses_blob = clauses_blob[:12_000]  # keep prompt bounded

    llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")
    prompt = (
        "Summarize the plan coverage based ONLY on the clauses below.\n"
        "Output sections:\n"
        "- Covered (bullets)\n"
        "- Not covered / exclusions (bullets)\n"
        "- Limits / caps / service fees (bullets)\n"
        "- Notes (eligibility, waiting periods, claim process pointers if present)\n"
        "Be careful: do not invent coverage.\n\n"
        f"CLAUSES:\n{clauses_blob}\n"
    )
    try:
        return str(llm.invoke([HumanMessage(content=prompt)]).content or "").strip()
    except Exception:
        return ""


@app.route("/claims/followup", methods=["POST"])
def claims_followup_chat():
    """Claims follow-up chat that answers using BOTH:
    - stored case context (final analyzed answer, extracted Q&A, prior follow-ups), and
    - vector DB retrieved policy clauses (Milvus), when contract/plan/state are available.

    Query params:
      - conversation-id (str)

    Body:
      - enteredQuery (str, required)

    Returns:
      { "aiResponse": "...", "conversationId": "...", "chatId": "..." }
    """
    try:
        with tracer.start_as_current_span("api/claims/followup"):
            authorization_header = request.headers.get("Authorization")
            if authorization_header is None:
                return jsonify({"message": "Token is missing"}), 401

            if authorization_header:
                token_data = token_process(authorization_header)
                if token_data[1] == 401 or token_data[1] == 403:
                    return (token_data[0].get_json()), token_data[1]

            conversation_id = request.args.get("conversation-id")
            if not conversation_id:
                return jsonify({"error": "conversation-id is required"}), 400

            data = request.get_json() or {}
            entered_query = (data.get("enteredQuery") or "").strip()
            if not entered_query:
                return jsonify({"error": "enteredQuery is required"}), 400

            user_email = token_data[0]["email"]
            qna_collection_user = f"chats_{user_email}"
            qna_collection = db[qna_collection_user]

            docs = qna_collection.find_one({"_id": ObjectId(conversation_id)}) or {}
            if not docs:
                return jsonify({"error": "Conversation not found"}), 404

            # Only enable this endpoint for transcript (Claims/Calls) conversations
            if docs.get("doc_type") != "transcript_conversation":
                return jsonify({"error": "claims/followup is only supported for transcript conversations"}), 400

            # Respect closed case lock (frontend also blocks, but enforce server-side too)
            if (docs.get("status") or "").lower() == "inactive":
                return jsonify({"error": "Case is closed. Chat is disabled."}), 403

            # Optional overrides from client (frontend knows selected plan/state from /history).
            # This makes follow-up resilient even if an older conversation stub is missing metadata.
            try:
                override_contract = (data.get("contractType") or "").strip()
                override_plan = (data.get("selectedPlan") or "").strip()
                override_state = (data.get("selectedState") or "").strip()
                if override_contract or override_plan or override_state:
                    updates = {}
                    if override_contract:
                        updates["contract_type"] = override_contract
                        docs["contract_type"] = override_contract
                    if override_plan:
                        updates["selected_plan"] = override_plan
                        docs["selected_plan"] = override_plan
                    if override_state:
                        updates["selected_state"] = override_state
                        docs["selected_state"] = override_state
                    if updates:
                        updates["updated_at"] = datetime.utcnow()
                        qna_collection.update_one({"_id": ObjectId(conversation_id)}, {"$set": updates})
            except Exception:
                pass

            # Best-effort: build & cache plan overview in Mongo for broad plan questions.
            # This helps queries like "What is covered in the plan?" even when retrieval is sparse.
            try:
                if _looks_like_plan_overview_question(entered_query):
                    overview = _get_or_build_plan_overview_for_claims(docs)
                    if overview:
                        qna_collection.update_one(
                            {"_id": ObjectId(conversation_id)},
                            {"$set": {"plan_overview": overview, "updated_at": datetime.utcnow()}},
                        )
                        docs["plan_overview"] = overview
            except Exception:
                pass

            case_context = _build_claims_case_context_for_llm(docs)
            if not case_context:
                return jsonify({"error": "Missing case context for this conversation"}), 400

            # Hybrid: retrieve policy clauses from Milvus using the case's stored contract/plan/state
            print(f"\n[CLAIMS_FOLLOWUP] About to call _retrieve_policy_chunks_for_claims")
            print(f"[CLAIMS_FOLLOWUP] entered_query: {entered_query}")
            print(f"[CLAIMS_FOLLOWUP] docs keys: {list(docs.keys()) if isinstance(docs, dict) else 'not_dict'}")
            print(f"[CLAIMS_FOLLOWUP] contract_type: {docs.get('contract_type')}")
            print(f"[CLAIMS_FOLLOWUP] selected_plan: {docs.get('selected_plan')}")
            print(f"[CLAIMS_FOLLOWUP] selected_state: {docs.get('selected_state')}\n")
            
            policy_chunks, referred_docs_text = _retrieve_policy_chunks_for_claims(docs, entered_query, k=6)

            print(f"[CLAIMS_FOLLOWUP] Returned from _retrieve_policy_chunks_for_claims")
            print(f"[CLAIMS_FOLLOWUP] policy_chunks count: {len(policy_chunks) if policy_chunks else 0}")
            print(f"[CLAIMS_FOLLOWUP] referred_docs_text length: {len(referred_docs_text) if referred_docs_text else 0}\n")
            
            # Log the structure of chunks being stored - DETAILED
            if policy_chunks:
                print(f"\n{'='*80}")
                print(f"[CLAIMS_FOLLOWUP] ===== CHUNKS BEING STORED IN MONGODB =====")
                for idx, ch in enumerate(policy_chunks, start=1):
                    print(f"[CLAIMS_FOLLOWUP] Chunk {idx}:")
                    print(f"[CLAIMS_FOLLOWUP]   Type: {type(ch)}")
                    if isinstance(ch, dict):
                        print(f"[CLAIMS_FOLLOWUP]   Keys: {list(ch.keys())}")
                        print(f"[CLAIMS_FOLLOWUP]   NAME: '{ch.get('name', 'MISSING NAME FIELD!')}'")
                        print(f"[CLAIMS_FOLLOWUP]   Content preview: {str(ch.get('content', ''))[:80]}...")
                        if ch.get('metadata'):
                            source = ch.get('metadata', {}).get('source', '')
                            print(f"[CLAIMS_FOLLOWUP]   Metadata.source: {source}")
                    else:
                        print(f"[CLAIMS_FOLLOWUP]   ERROR: Chunk is not a dict! Value: {ch}")
                print(f"{'='*80}\n")
            
            policy_section = ""
            if policy_chunks:
                print(f"[CLAIMS_FOLLOWUP] policy_chunks is not empty, building policy_section")
                lines = ["RETRIEVED POLICY CLAUSES (Vector DB)"]
                for i, ch in enumerate(policy_chunks[:12], start=1):
                    if not isinstance(ch, dict):
                        continue
                    content = str(ch.get("content") or "").strip()
                    if not content:
                        continue
                    # Use the name if available, otherwise fall back to Clause {i}
                    chunk_name = ch.get("name") or f"Clause {i}"
                    lines.append(f"- {chunk_name}: {content[:100]}...")
                policy_section = "\n".join(lines).strip()
            else:
                print(f"[CLAIMS_FOLLOWUP] policy_chunks is EMPTY - no chunks retrieved!")

            prompt = (
                "You are an insurance claims copilot.\n"
                "Answer the user's question using BOTH the CASE CONTEXT and (when provided) the RETRIEVED POLICY CLAUSES.\n"
                "If the user asks:\n"
                "- what the claim is about: summarize using FINAL ANALYZED ANSWER.\n"
                "- what customer queries were: list/explain from EXTRACTED CUSTOMER QUERIES.\n"
                "- a repeat question: answer consistently, using the context and prior follow-up chat.\n"
                "For policy/coverage questions, use the RETRIEVED POLICY CLAUSES when relevant.\n"
                "If the answer is not in CASE CONTEXT or the RETRIEVED POLICY CLAUSES, say you don't have that information.\n"
                "Do NOT use any external policy lookup beyond the provided clauses.\n"
                "\n"
                f"{case_context}\n"
                "\n"
                f"{policy_section}\n"
                "\n"
                f"USER QUESTION: {entered_query}\n"
                "ANSWER:"
            )

            llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")
            ai_text = ""
            try:
                ai_text = str(llm.invoke([HumanMessage(content=prompt)]).content or "").strip()
            except Exception as e:
                return jsonify({"error": f"LLM error: {e}"}), 500

            chat_id = str(uuid.uuid4())
            now_ts = datetime.utcnow()
            underlying = docs.get("underlying_model") or "Search"
            try:
                qna_collection.update_one(
                    {"_id": ObjectId(conversation_id)},
                    {
                        "$push": {
                            "chats": {
                                "chat_id": chat_id,
                                "entered_query": entered_query,
                                "response": ai_text,
                                "relevant_chunks": policy_chunks or [],
                                "relevant_docs": referred_docs_text or "",
                                "gpt_model": "Calls",
                                "underlying_model": underlying,
                                "chat_timestamp": now_ts,
                                "latency": 0.0,
                                "confidence": 0.0,
                            }
                        },
                        "$set": {"updated_at": now_ts},
                    },
                )
            except Exception:
                pass

            return jsonify({"aiResponse": ai_text, "conversationId": str(conversation_id), "chatId": chat_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/sidebar", methods=["GET"])
def sidebar_history():
    with tracer.start_as_current_span('api/sidebar'):
        authorization_header = request.headers.get("Authorization")

        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401

        if authorization_header:
            token_data = token_process(authorization_header)

            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]

        user_email = token_data[0]["email"]
        mode_param = request.args.get("mode")
        mode_param = mode_param.strip() if isinstance(mode_param, str) else None
        mode_param = mode_param if mode_param in ("Search", "Infer", "Calls") else None

        qna_collection_user = f"chats_{user_email}"
        qna_collection = db[qna_collection_user]

        # projection means setting the key name to 1, i.e we want all ids and names from given collection
        # Exclude transcript status-only documents from showing up in the sidebar.
        # Also include conversation_mode (and a lightweight fallback to chats.gpt_model for older docs).
        result = qna_collection.find(
            {"doc_type": {"$ne": "transcript_status"}},
            {
                "_id": 1,
                "doc_type": 1,
                "conversation_name": 1,
                "conversation_mode": 1,
                "chats.gpt_model": 1,
                "status": 1,
                "updated_at": 1,
                "transcript_id": 1,
                "processing": 1,
            },
        )

        output_json = []
        for doc in result:
            # IMPORTANT:
            # Always treat transcript (Claims/Calls) conversations as Calls mode, even if older data
            # was corrupted by overwriting conversation_mode to "Search"/"Infer".
            if doc.get("doc_type") == "transcript_conversation":
                conv_mode = "Calls"
            else:
                conv_mode = doc.get("conversation_mode")
            if not conv_mode:
                try:
                    chats = doc.get("chats") or []
                    conv_mode = chats[0].get("gpt_model") if chats else None
                except Exception:
                    conv_mode = None
            conv_mode = conv_mode or "Search"

            if mode_param and conv_mode != mode_param:
                continue

            output_json.append(
                {
                    "conversationId": str(doc["_id"]),
                    "conversationName": doc.get("conversation_name", ""),
                    "conversationMode": conv_mode,
                    "status": (doc.get("status") or "active"),
                    "updatedAt": (doc.get("updated_at").isoformat() + "Z") if doc.get("updated_at") else None,
                    "transcriptId": doc.get("transcript_id"),
                    # Only relevant for Claims/Calls (transcript conversations)
                    "processing": bool(doc.get("processing", False)) if conv_mode == "Calls" else False,
                }
            )

        output_json = output_json[::-1]

        if output_json:
            return make_response(jsonify(output_json), 200)
        else:
            return make_response(jsonify([]), 200)


@app.route("/delete", methods=["DELETE"])
def delete_conversation():
    with tracer.start_as_current_span('api/delete'):
        authorization_header = request.headers.get("Authorization")

        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401

        if authorization_header:
            token_data = token_process(authorization_header)

            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]

        user_email = token_data[0]["email"]

        qna_collection_user = f"chats_{user_email}"
        qna_collection = db[qna_collection_user]
        conversation_id = request.args.get("conversation-id")

        qna_collection.delete_one({"_id": ObjectId(conversation_id)})
        return {}


@app.route("/edit-conversation-name", methods=["PATCH"])
def edit_name():
    with tracer.start_as_current_span('api/edit-conversation-name'):
        authorization_header = request.headers.get("Authorization")

        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401

        if authorization_header:
            token_data = token_process(authorization_header)

            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]

        user_email = token_data[0]["email"]

        data = request.get_json()
        new_name = data.get("newName")

        conversation_id = request.args.get("conversation-id")

        try:
            qna_collection_user = f"chats_{user_email}"
            qna_collection = db[qna_collection_user]

            qna_collection.update_one(
                {"_id": ObjectId(conversation_id)},
                {"$set": {"conversation_name": new_name}},
            )
            return jsonify({"message": "Conversation name updated successfully"})

        except Exception as e:
            return jsonify({"error": str(e)})


@app.route("/referred-clauses", methods=["GET"])
def referred_clauses():
    with tracer.start_as_current_span('api/referred-clauses'):
        authorization_header = request.headers.get("Authorization")

        if authorization_header is None:
            return jsonify({"message": "Token is missing"}), 401

        if authorization_header:
            token_data = token_process(authorization_header)

            if token_data[1] == 401 or token_data[1] == 403:
                return (token_data[0].get_json()), token_data[1]

        user_email = token_data[0]["email"]
        conversation_id = request.args.get("conversation-id")
        chat_id = request.args.get("chat-id")

        try:
            print(
                "[CHUNKS] /referred-clauses: fetching conversation_id="
                f"{conversation_id}, chat_id={chat_id}"
            )
            docs = read_qna(email_id=user_email, conversation_id=conversation_id)

            chat_ans = docs["chats"]
            chat_obj = None
            for candidate in chat_ans:
                if candidate.get("chat_id") == chat_id:
                    chat_obj = candidate
                    break

            if not chat_obj:
                print(
                    "[CHUNKS] /referred-clauses: NO chat found for given chat_id; "
                    "cannot return referred clauses"
                )
                return jsonify({"error": "Chat not found for given chatId"}), 404

            question = chat_obj.get("entered_query")
            answer = chat_obj.get("response")
            referred_clauses_value = chat_obj.get("relevant_docs", "")

            print(
                "[CHUNKS] /referred-clauses: found chat, "
                f"referred_clauses_len={len(referred_clauses_value) if referred_clauses_value else 0}"
            )

            referred_clauses_json = {
                "contractType": docs["contract_type"],
                "selectedState": docs["selected_state"],
                "selectedPlan": docs["selected_plan"],
                "question": question,
                "answer": answer,
                "referredClauses": referred_clauses_value,
                "gpt_model": chat_obj.get("gpt_model"),
                "latency": chat_obj.get("latency", None),
                "word_count": chat_obj.get("word_count", None)
            }

            return referred_clauses_json

        except Exception as e:
            return jsonify({"error": str(e)}), 404


# ==================== TRANSCRIPT ENDPOINTS ====================

@app.route("/transcripts", methods=["GET"])
def list_transcripts():
    """List transcript files from GCP bucket with pagination and search (default: 10 records per page)
    
    IMPORTANT: When searching, this endpoint searches through ALL files in the GCS bucket (all 147 files).
    It lists all files from GCS first, then filters by search term, then applies pagination.
    
    Query Parameters:
    - limit (int, default: 10): Number of records per page
    - offset (int, default: 0): Number of records to skip
    - search (str, optional): Search term to filter transcripts by file name (case-insensitive partial match)
                             Searches through ALL files from GCS bucket
    - q (str, optional): Alias for 'search' parameter
    """
    try:
        with tracer.start_as_current_span('api/transcripts'):
            authorization_header = request.headers.get("Authorization")
            
            if authorization_header is None:
                return jsonify({"message": "Token is missing"}), 401
            
            if authorization_header:
                token_data = token_process(authorization_header)
                if token_data[1] == 401 or token_data[1] == 403:
                    return (token_data[0].get_json()), token_data[1]

            user_email = token_data[0]["email"]
            
            if not gcs_fs:
                return jsonify({"error": "GCP Storage not configured or unavailable"}), 500
            
            # Get query parameters - default limit is 9 (popup shows 3x3 grid)
            limit_param = request.args.get("limit", "9")
            offset_param = request.args.get("offset", "0")
            search_param = request.args.get("search") or request.args.get("q")  # Support both 'search' and 'q' parameters
            status_param = request.args.get("status")  # optional: active|inactive
            print(f"DEBUG API: Raw params - limit_param='{limit_param}', offset_param='{offset_param}', search_param='{search_param}'")
            
            try:
                limit = int(limit_param) if limit_param else 9
            except (ValueError, TypeError):
                limit = 9
                print(f"DEBUG API: Invalid limit param, using default: 9")
            
            try:
                offset = int(offset_param) if offset_param else 0
            except (ValueError, TypeError):
                offset = 0
                print(f"DEBUG API: Invalid offset param, using default: 0")
            
            # Validate parameters
            if limit < 1:
                print(f"DEBUG API: limit < 1, setting to 9")
                limit = 9
            if offset < 0:
                print(f"DEBUG API: offset < 0, setting to 0")
                offset = 0
            
            # List transcript files from GCP with pagination and search (only reads content for paginated subset)
            print(f"DEBUG API: Calling list_transcript_files_gcp(limit={limit}, offset={offset}, search={search_param}), gcs_fs={gcs_fs is not None}")
            paginated_transcripts, total_count = list_transcript_files_gcp(limit=limit, offset=offset, search=search_param)
            print(f"DEBUG API: Found {len(paginated_transcripts)} transcripts (showing {offset} to {offset + len(paginated_transcripts)} of {total_count} total)")

            # Attach status (stored in MongoDB) to each transcript returned from GCP.
            # We keep status docs in the same per-user collection as chat history, but with doc_type='transcript_status'.
            try:
                qna_collection_user = f"chats_{user_email}"
                qna_collection = db[qna_collection_user]

                transcript_ids = []
                for t in paginated_transcripts:
                    fname = t.get("fileName", "")
                    transcript_ids.append(fname.replace(".json", "").replace(".txt", ""))

                status_map = {}
                if transcript_ids:
                    cursor = qna_collection.find(
                        {"doc_type": "transcript_status", "transcript_id": {"$in": transcript_ids}},
                        {"_id": 0, "transcript_id": 1, "status": 1},
                    )
                    for d in cursor:
                        status_map[d.get("transcript_id")] = d.get("status")

                for t in paginated_transcripts:
                    fname = t.get("fileName", "")
                    tid = fname.replace(".json", "").replace(".txt", "")
                    t["status"] = status_map.get(tid, "active")

                if status_param in ("active", "inactive"):
                    paginated_transcripts = [t for t in paginated_transcripts if t.get("status") == status_param]
            except Exception as e:
                print(f"Warning: unable to attach transcript status from MongoDB: {e}")
            
            return jsonify({
                "transcripts": paginated_transcripts,
                "totalCount": total_count,
                "limit": limit,
                "offset": offset,
                "hasMore": (offset + limit) < total_count,
                "search": search_param if search_param else None,
                "status": status_param if status_param else None,
            }), 200
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in /transcripts endpoint: {str(e)}")
        print(f"Traceback: {error_trace}")
        return jsonify({"error": "An error occurred while fetching transcripts", "details": str(e)}), 500


@app.route("/transcripts/<filename>", methods=["GET"])
def get_transcript_content(filename):
    """Fetch transcript file content from GCS bucket"""
    try:
        with tracer.start_as_current_span('api/transcripts/content'):
            # Authorization
            authorization_header = request.headers.get("Authorization")
            
            if authorization_header is None:
                return jsonify({"message": "Token is missing"}), 401
            
            if authorization_header:
                token_data = token_process(authorization_header)
                if token_data[1] == 401 or token_data[1] == 403:
                    return (token_data[0].get_json()), token_data[1]
            
            # Validate filename
            if not filename:
                return jsonify({"error": "Filename is required"}), 400
            
            # Check if GCS is available
            if not gcs_fs:
                return jsonify({"error": "GCP Storage not configured or unavailable"}), 500
            
            # Read transcript file from GCP
            try:
                transcript_content, file_metadata = read_transcript_file_gcp(filename)
                
                # Try to parse as JSON to provide structured response
                try:
                    transcript_data = json.loads(transcript_content)
                    is_json = True
                except json.JSONDecodeError:
                    transcript_data = None
                    is_json = False
                
                # Build response
                response = {
                    "fileName": file_metadata["fileName"],
                    "fileSize": file_metadata["fileSize"],
                    "uploadDate": file_metadata["uploadDate"],
                    "content": transcript_content,
                    "isJson": is_json
                }
                
                # If JSON, also include parsed data
                if is_json:
                    response["parsedData"] = transcript_data
                    # Try to extract text content if available
                    if isinstance(transcript_data, dict):
                        text_content = (
                            transcript_data.get("text") or
                            transcript_data.get("transcript") or
                            transcript_data.get("content")
                        )
                        if text_content:
                            response["textContent"] = text_content
                
                return jsonify(response), 200
                
            except FileNotFoundError as e:
                return jsonify({
                    "error": f"Transcript file not found: {filename}",
                    "fileName": filename
                }), 404
            except Exception as e:
                return jsonify({
                    "error": f"Error reading transcript file: {str(e)}",
                    "fileName": filename
                }), 500
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in /transcripts/<filename> endpoint: {str(e)}")
        print(f"Traceback: {error_trace}")
        return jsonify({
            "error": "An error occurred while fetching transcript content",
            "details": str(e)
        }), 500


def _normalize_transcript_speaker_label(label: str) -> str:
    """
    Normalize a speaker label to one of:
      - "Customer"
      - "CSR"
      - "Unknown"
    """
    if not label:
        return "Unknown"
    x = str(label).strip().lower()

    # Customer-like labels
    if any(k in x for k in ["customer", "caller", "homeowner", "policyholder", "member"]):
        return "Customer"

    # CSR-like labels
    if any(k in x for k in ["csr", "agent", "rep", "representative", "support", "dispatcher", "employee"]):
        return "CSR"

    # Generic diarization labels: try to infer based on common patterns
    if x.startswith("speaker") or x.startswith("spk") or x.startswith("speaker_"):
        return "Unknown"

    return "Unknown"


def _extract_text_from_transcript_json(transcript_data) -> str:
    """Best-effort extraction of transcript text from known JSON shapes."""
    if transcript_data is None:
        return ""
    if isinstance(transcript_data, str):
        return transcript_data

    # Common shapes
    if isinstance(transcript_data, dict):
        return (
            transcript_data.get("text")
            or transcript_data.get("transcript")
            or transcript_data.get("content")
            or ""
        )
    return ""


def transcript_to_chat_turns(transcript_text: str, transcript_data=None) -> list:
    """
    Convert a transcript into a chat-style list:
      [{"role":"CSR"|"Customer"|"Unknown", "text":"..."}]

    Strategy:
    1) Use structured fields (utterances/segments) if present.
    2) Use regex splitting if speaker labels exist in text.
    3) Fall back to a single "Unknown" turn.
    """
    turns = []

    # 1) Structured diarization-like shapes
    if isinstance(transcript_data, dict):
        for key in ("utterances", "segments", "turns", "dialogue", "dialog"):
            items = transcript_data.get(key)
            if isinstance(items, list) and items:
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    speaker = (
                        it.get("speaker")
                        or it.get("role")
                        or it.get("speakerLabel")
                        or it.get("participant")
                    )
                    text = it.get("text") or it.get("utterance") or it.get("content") or it.get("message")
                    text = (text or "").strip()
                    if not text:
                        continue
                    role = _normalize_transcript_speaker_label(speaker)
                    # If role unknown, try lightweight hinting based on common opening scripts
                    if role == "Unknown" and re.search(r"\bthank you for calling\b|\bhow can i assist\b", text, re.I):
                        role = "CSR"
                    turns.append({"role": role, "text": text})
                if turns:
                    return turns

    # 2) Regex speaker-tag parsing from plain text
    raw = (transcript_text or "").strip()
    if not raw:
        return []

    # Normalize some separators to make splitting easier
    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Patterns like:
    # "Customer: ...", "CSR: ...", "Agent - ...", "[Customer] ..."
    speaker_pattern = re.compile(
        r"(?mi)^\s*(?:\[(?P<bracket>customer|caller|homeowner|policyholder|csr|agent|rep|representative|technician)\]|\b(?P<plain>customer|caller|homeowner|policyholder|csr|agent|rep|representative|technician)\b)\s*[:\-]\s*"
    )

    matches = list(speaker_pattern.finditer(normalized))
    if matches:
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(normalized)
            label = m.group("bracket") or m.group("plain") or ""
            chunk = normalized[start:end].strip()
            if not chunk:
                continue
            role = _normalize_transcript_speaker_label(label)
            turns.append({"role": role, "text": chunk})

        if turns:
            return turns

    # 3) Fallback single block
    return [{"role": "Unknown", "text": normalized}]


def _llm_segment_transcript_to_chat_turns(transcript_text: str) -> list:
    """
    LLM fallback for transcripts that are a single blob with no speaker tags.
    Uses a small/fast model to return JSON array:
      [{"role":"CSR"|"Customer","text":"..."}]
    """
    try:
        llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")
        prompt = ChatPromptTemplate.from_template(
            """
You are given a call transcript as plain text. Convert it into a chat conversation.

Rules:
- Output ONLY valid JSON (no markdown, no extra text).
- Return an array of objects: {{"role":"CSR"|"Customer","text":"..."}}
- Group contiguous lines by the same role.
- Do NOT invent content; only re-segment the provided transcript.
- Keep each "text" concise (ideally <= 240 characters) but preserve meaning.
- If unsure who spoke a line, choose the most likely role based on context.

Transcript:
{transcript}
"""
        )
        chain = prompt | llm | StrOutputParser()
        raw = (chain.invoke({"transcript": transcript_text}) or "").strip()
        data = json.loads(raw)
        if isinstance(data, list):
            cleaned = []
            for it in data:
                if not isinstance(it, dict):
                    continue
                role = it.get("role")
                text = (it.get("text") or "").strip()
                if role not in ("CSR", "Customer") or not text:
                    continue
                cleaned.append({"role": role, "text": text})
            return cleaned
        return []
    except Exception as e:
        print(f"Warning: LLM transcript segmentation failed: {e}")
        return []


@app.route("/transcripts/dialogue", methods=["POST"])
def transcript_dialogue():
    """
    Fetch a transcript from GCS and return it in a chat-like format.

    Body:
      {
        "transcriptFileName": "transcribe_1.txt",
        "useLLM": false        // optional: if true, forces LLM segmentation
      }

    Returns:
      {
        "transcriptId": "...",
        "transcriptFileName": "...",
        "transcriptMetadata": {...},
        "conversation": [{"role":"CSR"|"Customer"|"Unknown","text":"..."}],
        "totalTurns": 12,
        "usedLLM": false
      }
    """
    try:
        with tracer.start_as_current_span("api/transcripts/dialogue"):
            # Authorization
            authorization_header = request.headers.get("Authorization")
            if authorization_header is None:
                return jsonify({"message": "Token is missing"}), 401
            if authorization_header:
                token_data = token_process(authorization_header)
                if token_data[1] == 401 or token_data[1] == 403:
                    return (token_data[0].get_json()), token_data[1]

            data = request.get_json() or {}
            transcript_file_name = data.get("transcriptFileName") or data.get("fileName")
            use_llm = bool(data.get("useLLM", False))

            if not transcript_file_name:
                return jsonify({"error": "transcriptFileName is required"}), 400

            if not gcs_fs:
                return jsonify({"error": "GCP Storage not configured or unavailable"}), 500

            # Fetch file
            transcript_content, file_metadata = read_transcript_file_gcp(transcript_file_name)

            # Parse JSON if possible (for structured diarization)
            transcript_data = None
            transcript_text = transcript_content
            try:
                transcript_data = json.loads(transcript_content)
                # Prefer text extraction from JSON for downstream parsing
                extracted = _extract_text_from_transcript_json(transcript_data)
                if extracted:
                    transcript_text = extracted
            except Exception:
                transcript_data = None

            used_llm = False
            conversation = transcript_to_chat_turns(transcript_text, transcript_data=transcript_data)

            # If it's still essentially a single blob, optionally use LLM to segment
            if use_llm or (len(conversation) <= 1 and len(transcript_text or "") > 600):
                llm_turns = _llm_segment_transcript_to_chat_turns(transcript_text)
                if llm_turns:
                    conversation = llm_turns
                    used_llm = True

            transcript_id = transcript_file_name.replace(".json", "").replace(".txt", "")

            return jsonify(
                {
                    "transcriptId": transcript_id,
                    "transcriptFileName": transcript_file_name,
                    "transcriptMetadata": file_metadata,
                    "conversation": conversation,
                    "totalTurns": len(conversation),
                    "usedLLM": used_llm,
                }
            ), 200
    except FileNotFoundError:
        return jsonify({"error": f"Transcript file not found: {request.get_json().get('transcriptFileName') if request.get_json() else ''}"}), 404
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in /transcripts/dialogue endpoint: {str(e)}")
        print(f"Traceback: {error_trace}")
        return jsonify({"error": "An error occurred while building transcript dialogue", "details": str(e)}), 500


@app.route("/transcripts/status", methods=["PATCH"])
def update_transcript_status():
    """Toggle / set a transcript status in MongoDB (per user).

    Body:
      - transcriptFileName (str) or transcriptId (str) or fileName (str)
      - status: 'active' | 'inactive'
    """
    try:
        with tracer.start_as_current_span('api/transcripts/status'):
            authorization_header = request.headers.get("Authorization")

            if authorization_header is None:
                return jsonify({"message": "Token is missing"}), 401

            if authorization_header:
                token_data = token_process(authorization_header)
                if token_data[1] == 401 or token_data[1] == 403:
                    return (token_data[0].get_json()), token_data[1]

            user_email = token_data[0]["email"]

            data = request.get_json() or {}
            status = data.get("status")
            transcript_file_name = (
                data.get("transcriptFileName")
                or data.get("fileName")
                or data.get("transcriptId")
            )

            if not transcript_file_name:
                return jsonify({"error": "transcriptFileName or transcriptId is required"}), 400
            if status not in ("active", "inactive"):
                return jsonify({"error": "status must be 'active' or 'inactive'"}), 400

            transcript_id = transcript_file_name.replace(".json", "").replace(".txt", "")

            qna_collection_user = f"chats_{user_email}"
            qna_collection = db[qna_collection_user]

            now_ts = datetime.utcnow()
            doc = qna_collection.find_one_and_update(
                {"doc_type": "transcript_status", "transcript_id": transcript_id},
                {"$set": {
                    "doc_type": "transcript_status",
                    "transcript_id": transcript_id,
                    "transcript_file_name": transcript_file_name,
                    "status": status,
                    "updated_at": now_ts,
                }},
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )

            return jsonify({
                "transcriptId": transcript_id,
                "transcriptFileName": transcript_file_name,
                "status": doc.get("status"),
                "updatedAt": doc.get("updated_at"),
            }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/transcripts/conversations", methods=["GET"])
def list_transcript_conversations():
    """List existing transcript conversations for a given transcript (per user).

    Query parameters:
      - transcriptFileName (str) or transcriptId (str) or fileName (str)

    Returns:
      {
        "transcriptId": "...",
        "transcriptFileName": "...",
        "conversations": [
          {"conversationId": "...", "conversationName": "...", "status": "...", "updatedAt": "...", "createdAt": "..."}
        ]
      }
    """
    try:
        with tracer.start_as_current_span('api/transcripts/conversations'):
            authorization_header = request.headers.get("Authorization")

            if authorization_header is None:
                return jsonify({"message": "Token is missing"}), 401

            if authorization_header:
                token_data = token_process(authorization_header)
                if token_data[1] == 401 or token_data[1] == 403:
                    return (token_data[0].get_json()), token_data[1]

            user_email = token_data[0]["email"]
            transcript_file_name = (
                request.args.get("transcriptFileName")
                or request.args.get("fileName")
                or request.args.get("transcriptId")
            )
            if not transcript_file_name:
                return jsonify({"error": "transcriptFileName or transcriptId is required"}), 400

            transcript_id = transcript_file_name.replace(".json", "").replace(".txt", "")

            qna_collection_user = f"chats_{user_email}"
            qna_collection = db[qna_collection_user]

            cursor = qna_collection.find(
                {"doc_type": "transcript_conversation", "transcript_id": transcript_id},
                {"_id": 1, "conversation_name": 1, "status": 1, "query_time": 1, "updated_at": 1},
            ).sort([("updated_at", -1), ("query_time", -1)])

            conversations = []
            for doc in cursor:
                conversations.append(
                    {
                        "conversationId": str(doc.get("_id")),
                        "conversationName": doc.get("conversation_name") or "",
                        "status": (doc.get("status") or "active"),
                        "createdAt": doc.get("query_time"),
                        "updatedAt": doc.get("updated_at") or doc.get("query_time"),
                    }
                )

            return jsonify(
                {
                    "transcriptId": transcript_id,
                    "transcriptFileName": transcript_file_name,
                    "conversations": conversations,
                }
            ), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/transcripts/conversation/stub", methods=["POST"])
def create_transcript_conversation_stub():
    """Create a processing transcript conversation stub early so the sidebar can show it immediately.

    Body:
      - transcriptFileName (str, required)
      - contractType (str, required)
      - selectedPlan (str, required)
      - selectedState (str, required)
      - gptModel (optional): "Search" | "Infer" (underlying model)
      - newConversation (optional, bool): default false
      - conversationName (optional, str)
    """
    try:
        with tracer.start_as_current_span("api/transcripts/conversation/stub"):
            authorization_header = request.headers.get("Authorization")
            if authorization_header is None:
                return jsonify({"message": "Token is missing"}), 401

            if authorization_header:
                token_data = token_process(authorization_header)
                if token_data[1] == 401 or token_data[1] == 403:
                    return (token_data[0].get_json()), token_data[1]

            user_email = token_data[0]["email"]
            data = request.get_json() or {}
            transcript_file_name = data.get("transcriptFileName") or data.get("fileName")
            contract_type = data.get("contractType")
            selected_plan = data.get("selectedPlan")
            selected_state = data.get("selectedState")
            gpt_model = data.get("gptModel", "Search")
            new_conversation = bool(data.get("newConversation", False))
            requested_conversation_name = data.get("conversationName")

            if not transcript_file_name:
                return jsonify({"error": "transcriptFileName is required"}), 400
            if not all([contract_type, selected_plan, selected_state]):
                return jsonify({"error": "contractType, selectedPlan, selectedState are required"}), 400

            transcript_id = transcript_file_name.replace(".json", "").replace(".txt", "")

            qna_collection_user = f"chats_{user_email}"
            qna_collection = db[qna_collection_user]

            now_ts = datetime.utcnow()
            status_doc = qna_collection.find_one(
                {"doc_type": "transcript_status", "transcript_id": transcript_id},
                {"_id": 0, "status": 1},
            )
            transcript_status = (status_doc or {}).get("status") or "active"

            base_name = (requested_conversation_name or transcript_file_name or "").strip() or transcript_id
            if new_conversation:
                existing_count = qna_collection.count_documents(
                    {"doc_type": "transcript_conversation", "transcript_id": transcript_id}
                )
                conv_name = base_name if existing_count == 0 else f"{base_name} ({existing_count + 1})"
            else:
                conv_name = base_name

            stub = {
                "doc_type": "transcript_conversation",
                "conversation_mode": "Calls",
                "underlying_model": gpt_model,
                "conversation_name": conv_name,
                "transcript_id": transcript_id,
                "contract_type": contract_type,
                "selected_plan": selected_plan,
                "selected_state": selected_state,
                "query_time": now_ts,
                "updated_at": now_ts,
                "status": transcript_status,
                "case_disposition": None,
                "closed_at": None,
                "review_comments": None,
                "processing": True,
                "chats": [],
            }
            inserted = qna_collection.insert_one(stub)
            conv_doc_id = inserted.inserted_id

            return jsonify(
                {
                    "conversationId": str(conv_doc_id),
                    "conversationName": conv_name,
                    "status": transcript_status,
                    "processing": True,
                    "transcriptId": transcript_id,
                    "transcriptFileName": transcript_file_name,
                }
            ), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event (SSE) message."""
    try:
        payload = json.dumps(data, ensure_ascii=False)
    except Exception:
        payload = json.dumps({"error": "Failed to encode SSE payload"})
    return f"event: {event}\ndata: {payload}\n\n"


def generate_claim_decision_from_chunks(chunks: List[str], llm=None, claims_context: List[Dict] = None) -> Dict:
    """
    Produce a single claim authorization decision grounded ONLY in provided policy chunks.

    Returns:
      {
        "decision": "APPROVED"|"REJECTED"|"PARTIAL"|"CANNOT_DETERMINE",
        "shortAnswer": "...",
        "reasons": ["...", "..."],
        "citedChunks": ["...", "..."],
        "claims": [
          {
            "claimId": "c1",
            "items": [{"name": "...", "details": "..."}],
            "situation": "...",
            "decision": "APPROVED|REJECTED|PARTIAL|CANNOT_DETERMINE|REQUEST_INFO",
            "decisionSummary": "one sentence",
            "reasons": ["..."],
            "policyBasis": ["short quoted fragment", "..."],
            "nextSteps": ["..."]
          }
        ]
      }
    """
    cleaned = [str(c).strip() for c in (chunks or []) if str(c).strip()]
    # Drop obvious placeholders
    cleaned = [c for c in cleaned if c not in _PLACEHOLDER_CHUNK_VALUES]

    if not cleaned:
        return {
            "decision": "CANNOT_DETERMINE",
            "shortAnswer": "I can’t confirm approval or rejection from the policy text provided.",
            "reasons": ["No relevant policy clauses were retrieved to support a decision."],
            "citedChunks": [],
            "claims": [],
        }

    try:
        if llm is None:
            llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")

        # Normalize claim contexts into a compact blob for the model.
        claim_lines = []
        if isinstance(claims_context, list):
            for i, c in enumerate(claims_context):
                if not isinstance(c, dict):
                    continue
                cid = (c.get("claimId") or f"c{i+1}").strip()
                claim_text = (c.get("customerClaim") or c.get("claim") or c.get("question") or "").strip()
                situation = (c.get("situation") or c.get("context") or "").strip()
                if not (claim_text or situation):
                    continue
                line = f"- claimId: {cid}\n  claim: {claim_text or '(not provided)'}"
                if situation:
                    line += f"\n  situation: {situation}"
                claim_lines.append(line)

        claims_blob = "\n".join(claim_lines).strip()
        if not claims_blob:
            claims_blob = "- claimId: c1\n  claim: (No explicit claim description provided)\n  situation: (Not provided)"

        prompt = ChatPromptTemplate.from_template(
            """
You are a claims adjudication assistant. You must produce an overall decision AND a per-claim breakdown.

Decisions allowed:
- APPROVED: covered as described by the policy chunks
- REJECTED: clearly excluded/not covered by the policy chunks
- PARTIAL: some items/parts/situations are covered while others are excluded/limited
- CANNOT_DETERMINE: policy chunks are insufficient/ambiguous for a firm decision
- REQUEST_INFO (per-claim only): you need specific missing details to decide

CRITICAL RULES:
- Use ONLY the policy chunks provided below as evidence.
- Do NOT assume anything not explicitly stated in the chunks.
- Do NOT speculate or invent facts.
- If the chunks are insufficient/ambiguous to decide, output CANNOT_DETERMINE.
- Keep the language short, professional, and customer-friendly.
- You MUST address EACH claim listed under "Customer claims" (one entry per claimId).
- For each claim, list the item/items being claimed, and the situation/context in which the customer is claiming them.
- If the customer is claiming multiple items or multiple situations, set the per-claim decision to PARTIAL and explain what is/ isn’t covered.
- Provide 2 to 4 short overall reasons.
- Each overall reason MUST directly map to the provided chunks and include a short quoted fragment (in quotes).
- Also return citedChunks: include only the 1–3 chunk strings you relied on most.

Return ONLY valid JSON in exactly this schema:
{{
  "decision": "APPROVED|REJECTED|PARTIAL|CANNOT_DETERMINE",
  "shortAnswer": "one sentence",
  "reasons": ["reason1","reason2"],
  "citedChunks": ["chunk1","chunk2"],
  "claims": [
    {{
      "claimId": "c1",
      "items": [{{"name":"...","details":"..."}}],
      "situation": "short situation description (from claim context)",
      "decision": "APPROVED|REJECTED|PARTIAL|CANNOT_DETERMINE|REQUEST_INFO",
      "decisionSummary": "one sentence",
      "reasons": ["reason1","reason2"],
      "policyBasis": ["quoted fragment 1","quoted fragment 2"],
      "nextSteps": ["specific next step 1"]
    }}
  ]
}}

Customer claims (use this ONLY to understand what is being claimed; policy evidence must come from chunks):
{claims}

Policy chunks (verbatim):
{chunks}
"""
        )

        chain = prompt | llm | StrOutputParser()
        chunks_blob = "\n\n---\n\n".join(cleaned[:12])
        raw = (chain.invoke({"chunks": chunks_blob, "claims": claims_blob}) or "").strip()
        raw = re.sub(r"```json\\n?", "", raw)
        raw = re.sub(r"```\\n?", "", raw)
        raw = raw.strip()
        data = json.loads(raw)

        decision = (data.get("decision") or "").strip().upper()
        if decision not in ("APPROVED", "REJECTED", "PARTIAL", "CANNOT_DETERMINE"):
            decision = "CANNOT_DETERMINE"
        short_answer = (data.get("shortAnswer") or "").strip()
        reasons = data.get("reasons") or []
        cited = data.get("citedChunks") or []
        claims = data.get("claims") or []

        if not isinstance(reasons, list):
            reasons = []
        reasons = [str(r).strip() for r in reasons if str(r).strip()][:4]
        if not reasons:
            reasons = ["The provided policy text is not sufficient to justify a clear decision."]
            decision = "CANNOT_DETERMINE"

        if not isinstance(cited, list):
            cited = []
        cited = [str(c).strip() for c in cited if str(c).strip()]
        if cited:
            cited = cited[:3]
        else:
            # Default to first chunk(s) if model didn't provide citations
            cited = cleaned[:2]

        # Best-effort validation for claims array (keep backwards-compatible shape if model output is off).
        if not isinstance(claims, list):
            claims = []
        cleaned_claims = []
        for i, c in enumerate(claims):
            if not isinstance(c, dict):
                continue
            cid = str(c.get("claimId") or f"c{i+1}").strip()
            items = c.get("items") or []
            if not isinstance(items, list):
                items = []
            normalized_items = []
            for it in items:
                if isinstance(it, dict):
                    nm = (it.get("name") or "").strip()
                    det = (it.get("details") or "").strip()
                    if nm or det:
                        normalized_items.append({"name": nm, "details": det})
                else:
                    s = str(it).strip()
                    if s:
                        normalized_items.append({"name": s, "details": ""})
            per_dec = str(c.get("decision") or "").strip().upper()
            if per_dec not in ("APPROVED", "REJECTED", "PARTIAL", "CANNOT_DETERMINE", "REQUEST_INFO"):
                per_dec = "CANNOT_DETERMINE"
            cleaned_claims.append(
                {
                    "claimId": cid,
                    "items": normalized_items,
                    "situation": str(c.get("situation") or "").strip(),
                    "decision": per_dec,
                    "decisionSummary": str(c.get("decisionSummary") or "").strip(),
                    "reasons": [str(x).strip() for x in (c.get("reasons") or []) if str(x).strip()][:5],
                    "policyBasis": [str(x).strip() for x in (c.get("policyBasis") or []) if str(x).strip()][:5],
                    "nextSteps": [str(x).strip() for x in (c.get("nextSteps") or []) if str(x).strip()][:5],
                }
            )

        if not short_answer:
            if decision == "APPROVED":
                short_answer = "Your claim appears approved based on the policy clauses provided."
            elif decision == "REJECTED":
                short_answer = "Your claim appears rejected based on the policy clauses provided."
            elif decision == "PARTIAL":
                short_answer = "Your claim appears partially covered based on the policy clauses provided."
            else:
                short_answer = "I can’t confirm approval or rejection from the policy text provided."

        return {
            "decision": decision,
            "shortAnswer": short_answer,
            "reasons": reasons,
            "citedChunks": cited,
            "claims": cleaned_claims,
        }
    except Exception as e:
        print(f"Warning: claim decision generation failed: {e}")
        return {
            "decision": "CANNOT_DETERMINE",
            "shortAnswer": "I can’t confirm approval or rejection from the policy text provided.",
            "reasons": ["The system could not generate a grounded decision from the retrieved clauses."],
            "citedChunks": cleaned[:2],
            "claims": [],
        }


def _format_claim_decision_for_chat(claim_decision) -> str:
    """Human-readable summary of the claim decision JSON, suitable for saving as a chat message."""
    if not isinstance(claim_decision, dict):
        return ""

    overall = str(claim_decision.get("decision") or "").strip()
    short_answer = str(claim_decision.get("shortAnswer") or "").strip()
    reasons = claim_decision.get("reasons") or []
    claims = claim_decision.get("claims") or []

    lines = []
    lines.append("Claim decision (grounded in retrieved policy clauses)")
    if overall:
        lines.append(f"- Overall: {overall}")
    if short_answer:
        lines.append(f"- Summary: {short_answer}")
    lines.append("")

    if isinstance(reasons, list) and reasons:
        lines.append("Reasons")
        for r in reasons[:10]:
            rr = str(r or "").strip()
            if rr:
                lines.append(f"- {rr}")
        lines.append("")

    if isinstance(claims, list) and claims:
        lines.append("Per-claim breakdown")
        for idx, c in enumerate(claims[:25], start=1):
            if not isinstance(c, dict):
                continue
            cid = str(c.get("claimId") or "").strip()
            situation = str(c.get("situation") or "").strip()
            decision = str(c.get("decision") or "").strip()
            decision_summary = str(c.get("decisionSummary") or "").strip()
            items = c.get("items") or []

            header = f"- Claim {idx}"
            if cid:
                header += f" ({cid})"
            if decision:
                header += f": {decision}"
            lines.append(header)

            if isinstance(items, list) and items:
                item_names = []
                for it in items:
                    if isinstance(it, dict):
                        nm = str(it.get("name") or "").strip()
                        det = str(it.get("details") or "").strip()
                        if nm and det:
                            item_names.append(f"{nm} ({det})")
                        elif nm:
                            item_names.append(nm)
                        elif det:
                            item_names.append(det)
                    else:
                        s = str(it or "").strip()
                        if s:
                            item_names.append(s)
                if item_names:
                    lines.append(f"  - Items: {', '.join(item_names[:8])}")

            if situation:
                lines.append(f"  - Situation: {situation}")
            if decision_summary:
                lines.append(f"  - Summary: {decision_summary}")

    return "\n".join(lines).strip()


@app.route("/transcripts/process/stream", methods=["POST"])
def process_transcript_stream():
    """
    Stream transcript processing via SSE.

    Input body (same as /transcripts/process):
      {
        "transcriptFileName": "...",
        "contractType": "RE"|"DTC",
        "selectedPlan": "...",
        "selectedState": "...",
        "gptModel": "Search"|"Infer",
        "extractQuestions": true,
        "forceReprocess": false,
        "newConversation": false,
        "conversationName": "..."
      }

    Output: text/event-stream
      - status events for stages
      - answer events for each question result
      - final event for finalSummary
      - done event
      - error event if something fails
    """
    print("process_transcript_stream")
    @stream_with_context
    def generate():
        start_time = time()
        token_data = None
        user_email = None
        qna_collection = None
        conv_doc_id = None
        conv_name = None
        transcript_status = "active"

        try:
            # Authorization
            authorization_header = request.headers.get("Authorization")
            if authorization_header is None:
                yield _sse("error", {"error": "Token is missing"})
                return
            token_data = token_process(authorization_header)
            if token_data[1] == 401 or token_data[1] == 403:
                # token_process returns (flask_response, status)
                try:
                    yield _sse("error", token_data[0].get_json())
                except Exception:
                    yield _sse("error", {"error": "Unauthorized"})
                return

            user_email = token_data[0]["email"]

            data = request.get_json() or {}
            transcript_file_name = data.get("transcriptFileName")
            contract_type = data.get("contractType")
            selected_plan = data.get("selectedPlan")
            selected_state = data.get("selectedState")
            milvus_state = normalize_state_for_milvus(selected_state)
            contract_type_norm = normalize_contract_type(contract_type)
            selected_plan_norm = normalize_plan_for_milvus(contract_type_norm, selected_plan)
            gpt_model = data.get("gptModel", "Search")
            extract_questions = data.get("extractQuestions", True)
            provided_questions = data.get("questions", [])
            force_reprocess = bool(data.get("forceReprocess", False))
            new_conversation = bool(data.get("newConversation", False))
            requested_conversation_name = data.get("conversationName")

            if not transcript_file_name:
                yield _sse("error", {"error": "transcriptFileName is required"})
                return
            if extract_questions and not all([contract_type, selected_plan, selected_state]):
                yield _sse(
                    "error",
                    {
                        "error": "contractType, selectedPlan, selectedState are required when extractQuestions=true"
                    },
                )
                return

            transcript_id = transcript_file_name.replace(".json", "").replace(".txt", "")

            # Mongo handles (same collection as /transcripts/process)
            qna_collection_user = f"chats_{user_email}"
            qna_collection = db[qna_collection_user]

            # If frontend already created a Mongo stub, reuse it so sidebar updates first.
            requested_conversation_id = data.get("conversationId") or data.get("conversation_id")
            if requested_conversation_id:
                try:
                    conv_doc_id = ObjectId(str(requested_conversation_id))
                    existing = qna_collection.find_one({"_id": conv_doc_id}) or {}
                    if existing.get("doc_type") != "transcript_conversation":
                        yield _sse("error", {"error": "Invalid conversationId for transcript processing"})
                        return
                    # Ensure processing is marked true and keep metadata consistent.
                    now_ts = datetime.utcnow()
                    qna_collection.update_one(
                        {"_id": conv_doc_id},
                        {
                            "$set": {
                                "processing": True,
                                "updated_at": now_ts,
                                "conversation_mode": "Calls",
                                "underlying_model": gpt_model,
                                "transcript_id": transcript_id,
                                "contract_type": contract_type,
                                "selected_plan": selected_plan,
                                "selected_state": selected_state,
                            }
                        },
                    )
                    conv_name = existing.get("conversation_name") or requested_conversation_name or transcript_file_name
                    transcript_status = (existing.get("status") or "active")
                    # Emit started AFTER confirming stub exists.
                    yield _sse(
                        "status",
                        {
                            "stage": "started",
                            "transcriptId": transcript_id,
                            "transcriptFileName": transcript_file_name,
                            "gptModel": gpt_model,
                            "conversationId": str(conv_doc_id),
                            "conversationName": conv_name,
                            "status": transcript_status,
                        },
                    )
                    yield _sse(
                        "status",
                        {
                            "stage": "conversation_created",
                            "conversationId": str(conv_doc_id),
                            "conversationName": conv_name,
                            "status": transcript_status,
                        },
                    )
                    # Ensure we don't prematurely return via the cached fast-path for this stub.
                    force_reprocess = True
                except Exception:
                    yield _sse("error", {"error": "Invalid conversationId"})
                    return

            # Cache fast-path: if exists and not force, stream cached answers immediately
            existing_conv = None
            if not new_conversation:
                existing_conv = qna_collection.find_one(
                    {"doc_type": "transcript_conversation", "transcript_id": transcript_id},
                    sort=[("updated_at", -1), ("query_time", -1)],
                )

            if existing_conv and not force_reprocess and not new_conversation:
                cached = existing_conv.get("response_payload") or {}
                conv_doc_id = existing_conv.get("_id")
                yield _sse(
                    "status",
                    {
                        "stage": "cached",
                        "conversationId": str(conv_doc_id),
                        "conversationName": existing_conv.get("conversation_name") or "",
                        "status": (existing_conv.get("status") or "active"),
                    },
                )

                for q in (cached.get("questions") or []):
                    if not isinstance(q, dict):
                        continue
                    qid = q.get("questionId")
                    if qid == "final_answer":
                        continue
                    yield _sse(
                        "answer",
                        {
                            "questionId": qid,
                            "question": q.get("question") or "",
                            "answer": q.get("answer") or "",
                            "relevantChunks": q.get("relevantChunks") or [],
                            "confidence": q.get("confidence", 0.0),
                            "latency": q.get("latency", 0.0),
                            "questionType": q.get("questionType"),
                            "userIntent": q.get("userIntent"),
                        },
                    )

                if isinstance(cached.get("claimDecision"), dict):
                    yield _sse("claimDecision", cached.get("claimDecision"))

                final_summary = cached.get("finalSummary") or ""
                yield _sse("final", {"finalSummary": final_summary})
                yield _sse("done", {"elapsedSec": round(time() - start_time, 2)})
                return

            # Create / update a processing transcript conversation doc early (same as /transcripts/process)
            now_ts = datetime.utcnow()
            status_doc = qna_collection.find_one(
                {"doc_type": "transcript_status", "transcript_id": transcript_id},
                {"_id": 0, "status": 1},
            )
            transcript_status = (status_doc or {}).get("status") or "active"

            base_name = (requested_conversation_name or transcript_file_name or "").strip() or transcript_id
            if new_conversation:
                existing_count = qna_collection.count_documents(
                    {"doc_type": "transcript_conversation", "transcript_id": transcript_id}
                )
                conv_name = base_name if existing_count == 0 else f"{base_name} ({existing_count + 1})"
            else:
                conv_name = base_name

            # Create / update the processing stub (if frontend didn't already create one).
            if conv_doc_id is None:
                stub = {
                    "doc_type": "transcript_conversation",
                    "conversation_mode": "Calls",
                    "underlying_model": gpt_model,
                    "conversation_name": conv_name,
                    "transcript_id": transcript_id,
                    "contract_type": contract_type,
                    "selected_plan": selected_plan,
                    "selected_state": selected_state,
                    "query_time": now_ts,
                    "updated_at": now_ts,
                    "status": transcript_status,
                    "processing": True,
                    "chats": [],
                }
                inserted = qna_collection.insert_one(stub)
                conv_doc_id = inserted.inserted_id

                # Emit a "started" status only AFTER the Mongo stub exists, so the UI can refresh the sidebar
                # and show the in-progress (yellow) case immediately.
                yield _sse(
                    "status",
                    {
                        "stage": "started",
                        "transcriptId": transcript_id,
                        "transcriptFileName": transcript_file_name,
                        "gptModel": gpt_model,
                        "conversationId": str(conv_doc_id),
                        "conversationName": conv_name,
                        "status": transcript_status,
                    },
                )

                yield _sse(
                    "status",
                    {
                        "stage": "conversation_created",
                        "conversationId": str(conv_doc_id),
                        "conversationName": conv_name,
                        "status": transcript_status,
                    },
                )
            else:
                # Frontend stub exists: mark it as processing before continuing.
                qna_collection.update_one(
                    {"_id": conv_doc_id},
                    {"$set": {"processing": True, "updated_at": now_ts}},
                )

            # Read transcript from GCS
            if not gcs_fs:
                yield _sse("error", {"error": "GCP Storage not configured or unavailable"})
                return

            yield _sse("status", {"stage": "transcript_loading"})
            transcript_content, file_metadata = read_transcript_file_gcp(transcript_file_name)
            transcript_text = transcript_content
            try:
                transcript_data = json.loads(transcript_content)
                if isinstance(transcript_data, dict):
                    transcript_text = transcript_data.get(
                        "text",
                        transcript_data.get(
                            "transcript",
                            transcript_data.get("content", str(transcript_data)),
                        ),
                    )
            except Exception:
                transcript_text = transcript_content

            yield _sse(
                "status",
                {
                    "stage": "transcript_loaded",
                    "transcriptMetadata": {
                        "fileName": file_metadata.get("fileName"),
                        "uploadDate": file_metadata.get("uploadDate"),
                        "fileSize": file_metadata.get("fileSize"),
                    },
                },
            )

            # Extract questions
            extraction_warning = None
            questions = []
            if extract_questions:
                yield _sse("status", {"stage": "extracting_questions"})
                llm_extract = ChatOpenAI(temperature=0.0, model="gpt-4o")
                questions = extract_relevant_customer_questions(transcript_text, llm_extract)
                if not questions:
                    questions = extract_questions_with_agent(transcript_text, llm_extract)
                if not questions:
                    extraction_warning = "No questions could be extracted from transcript; inferring from context."
                    inferred_question = {
                        "question": f"Is this issue covered: {transcript_text[:120]}",
                        "context": transcript_text[:400],
                        "questionType": "coverage",
                        "userIntent": "Customer wants to know if the described issue is covered",
                        "questionId": "q1",
                    }
                    questions = [inferred_question]
            else:
                questions = provided_questions
                if not questions:
                    yield _sse("error", {"error": "No questions provided"})
                    return

            yield _sse(
                "status",
                {
                    "stage": "questions_ready",
                    "totalQuestions": len(questions),
                    "warning": extraction_warning,
                },
            )

            # Initialize vector DB + LLMs
            yield _sse("status", {"stage": "initializing_retriever"})
            collection_mapping = {
                "RE": {
                    "ShieldEssential": f"{milvus_state}_RE_ShieldEssential",
                    "ShieldPlus": f"{milvus_state}_RE_ShieldPlus",
                    "default": f"{milvus_state}_RE_ShieldComplete",
                },
                "DTC": {
                    "ShieldSilver": f"{milvus_state}_DTC_ShieldSilver",
                    "ShieldGold": f"{milvus_state}_DTC_ShieldGold",
                    "default": f"{milvus_state}_DTC_ShieldPlatinum",
                },
            }
            selected_collection_name = collection_mapping.get(contract_type_norm, {}).get(
                selected_plan_norm, collection_mapping.get(contract_type_norm, {}).get("default")
            )
            vector_db1 = Milvus(
                embed,
                collection_name=selected_collection_name,
                connection_args={"host": MILVUS_HOST, "port": "19530"},
            )
            retriever = vector_db1.as_retriever(search_kwargs={"k": MILVUS_RETRIEVER_K})

            if gpt_model == "Search":
                llm2 = ChatOpenAI(temperature=0.0, model="ft:gpt-3.5-turbo-0613:mindstix::8YYD56aA")
                llm = ChatOpenAI(temperature=0.0, model="gpt-4o")
            elif gpt_model == "Infer":
                llm3 = ChatOpenAI(temperature=0.0, model="ft:gpt-3.5-turbo-0613:mindstix::8YYD56aA")
                llm = ChatOpenAI(temperature=0.0, model="gpt-4o")
                llm2 = ChatOpenAI(temperature=0.0, model="gpt-4o")
            else:
                yield _sse("error", {"error": f"Invalid gpt_model: {gpt_model}. Must be 'Search' or 'Infer'"})
                return

            yield _sse("status", {"stage": "answering"})

            results = []
            confidences = []
            total_latency = 0.0
            now_ts = datetime.utcnow()

            # Process each question and stream immediately
            for idx, question_obj in enumerate(questions):
                question_text = question_obj.get("question", "")
                question_id = question_obj.get("questionId", f"q{idx + 1}")

                yield _sse(
                    "status",
                    {"stage": "answering_question", "index": idx + 1, "questionId": question_id},
                )

                result = process_single_transcript_question(
                    question_text,
                    contract_type,
                    selected_plan,
                    selected_state,
                    gpt_model,
                    vector_db1,
                    llm,
                    llm2,
                    retriever,
                    handler,
                    transcript_context=question_obj.get("context", ""),
                )

                result["questionId"] = question_id
                result["question"] = question_text
                result["context"] = question_obj.get("context", "")
                result["questionType"] = question_obj.get("questionType", "general")
                result["userIntent"] = question_obj.get("userIntent", "")

                # Enforce API contract: relevantChunks must be a non-empty list[str]
                rc = result.get("relevantChunks") or []
                if isinstance(rc, list):
                    rc = [str(x) for x in rc if str(x).strip()]
                else:
                    rc = []
                if not rc:
                    rc = ["(No supporting excerpts found)"]
                if MILVUS_MAX_RETURN_CHUNKS is not None:
                    rc = rc[:MILVUS_MAX_RETURN_CHUNKS]
                result["relevantChunks"] = rc

                if "error" not in result:
                    confidences.append(result.get("confidence", 0.0))
                    total_latency += float(result.get("latency", 0.0) or 0.0)

                results.append(result)

                # Persist incremental chat to Mongo (so /history can show progress if needed)
                try:
                    chunks = result.get("relevantChunks") or []
                    # Extract content for text blob
                    chunk_contents = []
                    for c in chunks:
                        if isinstance(c, dict):
                            chunk_contents.append(str(c.get("content", "")))
                        else:
                            chunk_contents.append(str(c))
                    relevant_docs_text = "\n\n---\n\n".join([c for c in chunk_contents if c.strip()])
                    
                    # Ensure chunks are objects with names before storing
                    normalized_chunks = _normalize_chunks_with_names(chunks)
                    
                    qna_collection.update_one(
                        {"_id": conv_doc_id},
                        {
                            "$push": {
                                "chats": {
                                    "chat_id": question_id,
                                    "entered_query": question_text,
                                    "response": result.get("answer", ""),
                                    "relevant_chunks": normalized_chunks,
                                    "relevant_docs": relevant_docs_text,
                                    "gpt_model": "Calls",
                                    "underlying_model": gpt_model,
                                    "chat_timestamp": now_ts,
                                    "latency": result.get("latency", 0.0),
                                    "confidence": result.get("confidence", 0.0),
                                }
                            },
                            "$set": {"updated_at": datetime.utcnow()},
                        },
                    )
                except Exception as e:
                    print(f"Warning: failed to persist incremental transcript chat: {e}")

                # Stream this answer immediately
                yield _sse(
                    "answer",
                    {
                        "questionId": question_id,
                        "question": question_text,
                        "answer": result.get("answer", ""),
                        "relevantChunks": result.get("relevantChunks", []),
                        "confidence": result.get("confidence", 0.0),
                        "latency": result.get("latency", 0.0),
                        "questionType": result.get("questionType"),
                        "userIntent": result.get("userIntent"),
                    },
                )

            # Final summary (same logic as /transcripts/process)
            final_summary_text = ""
            try:
                llm_summary = ChatOpenAI(temperature=0.0, model="gpt-4o")
                qa_lines = []
                for r in results or []:
                    if not r:
                        continue
                    q = (r.get("question") or "").strip()
                    if not q:
                        continue
                    ctx = (r.get("context") or "").strip()
                    a = (r.get("answer") or "").strip() or "(No answer was generated for this question.)"
                    # Provide structured evidence for the final summarizer to cluster by appliance/item.
                    if ctx:
                        qa_lines.append(f"Q: {q}\nSituation: {ctx}\nA: {a}")
                    else:
                        qa_lines.append(f"Q: {q}\nA: {a}")
                qa_blob = "\n\n".join(qa_lines)
                if qa_blob.strip():
                    summary_prompt = PromptTemplate(
                        input_variables=["qa_blob"],
                        template=(
                            "You are writing the FINAL ANSWER for a claims transcript.\n"
                            "IMPORTANT: Do NOT present the final answer as a list of each Q&A.\n"
                            "Instead, synthesize ALL Q&A into an APPLIANCE/ITEM-BASED final answer.\n"
                            "\n"
                            "Task:\n"
                            "- Identify the distinct appliance(s)/item(s)/system(s) mentioned across the Q&A.\n"
                            "- Group/merge related questions into the correct item section (do not repeat the questions).\n"
                            "- If the transcript includes multiple items with separate claims, show them as separate sections.\n"
                            "\n"
                            "For EACH item section, include:\n"
                            "- Item : <1,2,3...>\n"
                            "- Item: <name> (add 1-line details if available: location/part/symptom)\n"
                            "- Type: Appliance | System | Fixture | Other (infer from wording; if unclear use Other)\n"
                            "- Related: related parts/components/secondary-damage items (if any)\n"
                            "- Situation: what happened / what customer is claiming (from Situation lines)\n"
                            "- Decision: APPROVED | REJECTED | PARTIAL | NEED_INFO\n"
                            "- What's covered (bullet list, if any)\n"
                            "- What's not covered / limitations (bullet list, if any)\n"
                            "- Amounts (only if mentioned in Q&A):\n"
                            "  - Customer quoted/asked: $...\n"
                            "  - Company can provide: $... (coverage amount/limit/service fee/deductible as stated in Q&A)\n"
                            "- Why (1–2 short sentences grounded in the Q&A outcomes; no policy speculation)\n"
                            "- Next steps (specific actions the customer should take)\n"
                            "\n"
                            "CRITICAL DECISION RULES:\n"
                            "- The Decision field is MANDATORY and MUST NEVER be left empty for any item.\n"
                            "- If it is confirmed that there is NO coverage for a particular item, the Decision MUST be REJECTED.\n"
                            "- If outcomes are mixed for the same item, use PARTIAL and clearly break down covered vs not covered.\n"
                            "- If coverage cannot be determined, use NEED_INFO.\n"
                            "- Be concise, decisive, and avoid hypothetical/if-then language.\n"
                            "- End with a short overall next step (1–2 bullets) if multiple items exist.\n\n"
                            "{qa_blob}\n"
                        ),
                    )
                    summary_chain = summary_prompt | llm_summary | StrOutputParser()
                    final_summary_text = summary_chain.invoke({"qa_blob": qa_blob}).strip()
            except Exception as e:
                print(f"Warning: failed to generate final transcript summary (stream): {e}")

            if (not final_summary_text.strip()) and results:
                final_summary_text = "\n".join(
                    [
                        f"- {((r.get('answer') or '').strip() or '(No answer was generated for this question.)')}"
                        for r in results
                        if r and (r.get("question") or "").strip()
                    ]
                ).strip()

            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            # Claim decision grounded only in retrieved chunks (stream it before final summary UI finishes)
            claim_decision = None
            try:
                all_chunks = []
                for r in results or []:
                    rc = r.get("relevantChunks") or []
                    if isinstance(rc, list):
                        all_chunks.extend([str(x) for x in rc if str(x).strip()])
                seen = set()
                deduped = []
                for c in all_chunks:
                    if c in seen:
                        continue
                    seen.add(c)
                    deduped.append(c)
                claims_context = []
                for r in results or []:
                    if not isinstance(r, dict):
                        continue
                    claims_context.append(
                        {
                            "claimId": (r.get("questionId") or ""),
                            "customerClaim": (r.get("question") or ""),
                            "situation": (r.get("context") or ""),
                        }
                    )
                claim_decision = generate_claim_decision_from_chunks(deduped, claims_context=claims_context)
                yield _sse("claimDecision", claim_decision)
            except Exception as e:
                print(f"Warning: failed to generate/stream claimDecision: {e}")

            # Store final answer as last chat entry and finalize conversation doc
            try:
                qna_collection.update_one(
                    {"_id": conv_doc_id},
                    {
                        "$push": {
                            "chats": {
                                "$each": [
                                    {
                                        "chat_id": "final_answer",
                                        "entered_query": "Final Answer for transcript",
                                        "response": final_summary_text,
                                        "relevant_chunks": [],
                                        "relevant_docs": "",
                                        "gpt_model": "Calls",
                                        "underlying_model": gpt_model,
                                        "chat_timestamp": datetime.utcnow(),
                                        "latency": 0.0,
                                        "confidence": 0.0,
                                    },
                                ]
                            },
                        },
                        "$set": {
                            "processing": False,
                            "updated_at": datetime.utcnow(),
                            "final_summary": final_summary_text,
                            "claim_decision": claim_decision,
                            "summary": {
                                "totalQuestions": len(questions),
                                "processedQuestions": len([r for r in results if "error" not in r]),
                                "averageConfidence": round(avg_confidence, 2),
                                "totalLatency": round(total_latency, 2),
                            },
                            "transcript_metadata": {
                                "fileName": file_metadata.get("fileName"),
                                "uploadDate": file_metadata.get("uploadDate"),
                                "fileSize": file_metadata.get("fileSize"),
                            },
                        },
                    },
                )
            except Exception as e:
                print(f"Warning: failed to finalize transcript conversation doc (stream): {e}")

            yield _sse("final", {"finalSummary": final_summary_text})
            yield _sse(
                "done",
                {
                    "elapsedSec": round(time() - start_time, 2),
                    "conversationId": str(conv_doc_id) if conv_doc_id else "",
                    "conversationName": conv_name or "",
                    "status": transcript_status,
                },
            )
            return

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in /transcripts/process/stream endpoint: {str(e)}")
            print(f"Traceback: {error_trace}")
            yield _sse("error", {"error": "An error occurred while streaming transcript processing", "details": str(e)})
            return

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return Response(generate(), headers=headers)


@app.route("/transcripts/process", methods=["POST"])
def process_transcript():
    """Process transcript: fetch from GCP, extract questions, and get answers"""
    try:
        with tracer.start_as_current_span('api/transcripts/process') as parent0:
            start_time = time()
            extraction_warning = None
            
            # Authorization
            with tracer.start_as_current_span('authorization'):
                authorization_header = request.headers.get("Authorization")
                
                if authorization_header is None:
                    return jsonify({"message": "Token is missing"}), 401
                
                if authorization_header:
                    token_data = token_process(authorization_header)
                    if token_data[1] == 401 or token_data[1] == 403:
                        return (token_data[0].get_json()), token_data[1]
            
            # Get request data
            with tracer.start_as_current_span('data-fetching'):
                data = request.get_json()
                if not data:
                    return jsonify({"error": "Request body is missing or invalid"}), 400
                
                transcript_file_name = data.get("transcriptFileName")
                contract_type = data.get("contractType")
                selected_plan = data.get("selectedPlan")
                selected_state = data.get("selectedState")
                milvus_state = normalize_state_for_milvus(selected_state)
                contract_type_norm = normalize_contract_type(contract_type)
                selected_plan_norm = normalize_plan_for_milvus(contract_type_norm, selected_plan)
                gpt_model = data.get("gptModel", "Search")
                extract_questions = data.get("extractQuestions", True)
                provided_questions = data.get("questions", [])
                force_reprocess = bool(data.get("forceReprocess", False))
                new_conversation = bool(data.get("newConversation", False))
                requested_conversation_name = data.get("conversationName")
                
                # Validate required fields
                if not transcript_file_name:
                    return jsonify({"error": "transcriptFileName is required"}), 400
                
                if extract_questions and not all([contract_type, selected_plan, selected_state]):
                    return jsonify({
                        "error": "contractType, selectedPlan, selectedState are required when extractQuestions=true"
                    }), 400
            
            user_email = token_data[0]["email"]
            transcript_id = transcript_file_name.replace(".json", "").replace(".txt", "")

            # Use the existing per-user chat collection (same as Search/Infer) for transcript conversations.
            qna_collection_user = f"chats_{user_email}"
            qna_collection = db[qna_collection_user]

            # If frontend already created a Mongo stub, reuse it (ensures sidebar updates first).
            requested_conversation_id = data.get("conversationId") or data.get("conversation_id")
            if requested_conversation_id:
                try:
                    conv_doc_id = ObjectId(str(requested_conversation_id))
                    existing = qna_collection.find_one({"_id": conv_doc_id}) or {}
                    if existing.get("doc_type") != "transcript_conversation":
                        return jsonify({"error": "Invalid conversationId for transcript processing"}), 400
                    # Ensure we don't return cached payload for a newly-created stub.
                    force_reprocess = True
                    now_ts = datetime.utcnow()
                    qna_collection.update_one(
                        {"_id": conv_doc_id},
                        {
                            "$set": {
                                "processing": True,
                                "updated_at": now_ts,
                                "conversation_mode": "Calls",
                                "underlying_model": gpt_model,
                                "transcript_id": transcript_id,
                                "contract_type": contract_type,
                                "selected_plan": selected_plan,
                                "selected_state": selected_state,
                            }
                        },
                    )
                except Exception:
                    return jsonify({"error": "Invalid conversationId"}), 400

            # If we have already processed this transcript for this user, return the cached conversation.
            existing_conv = None
            # conv_doc_id may be pre-set above when frontend created a stub
            conv_name = None
            if not new_conversation:
                # Pick the most recently updated conversation for this transcript (if any)
                existing_conv = qna_collection.find_one(
                    {"doc_type": "transcript_conversation", "transcript_id": transcript_id},
                    sort=[("updated_at", -1), ("query_time", -1)],
                )

            if existing_conv and not force_reprocess and not new_conversation:
                # If the existing record was created before we started storing real chunk text,
                # it may contain placeholder chunk content like "[]". In that case, reprocess.
                try:
                    existing_chats = existing_conv.get("chats") or []
                    has_placeholder_chunks = False
                    for c in existing_chats:
                        if c.get("chat_id") == "final_answer":
                            continue
                        rc = c.get("relevant_chunks") or []
                        # Legacy shape: list[dict] with {"content":"[]"}; New shape: list[str] with "[]"
                        if rc and all(
                            (
                                (
                                    isinstance(x, dict)
                                    and (str(x.get("content") or "").strip() in _PLACEHOLDER_CHUNK_VALUES)
                                )
                                or (
                                    isinstance(x, str)
                                    and (x.strip() in _PLACEHOLDER_CHUNK_VALUES)
                                )
                            )
                            for x in rc
                        ):
                            has_placeholder_chunks = True
                            break
                    if has_placeholder_chunks or not existing_conv.get("final_summary"):
                        # We'll reprocess, but keep updating the same conversation document.
                        conv_doc_id = existing_conv.get("_id")
                        conv_name = existing_conv.get("conversation_name")
                        existing_conv = None
                except Exception as e:
                    print(f"Warning: cache validation failed, will reprocess transcript: {e}")
                    existing_conv = None

            if existing_conv and not force_reprocess and not new_conversation:
                cached = existing_conv.get("response_payload") or {}
                # Ensure required fields exist in cached payload
                cached["conversationId"] = str(existing_conv.get("_id"))
                cached.setdefault("transcriptId", existing_conv.get("transcript_id"))
                cached.setdefault("transcriptMetadata", existing_conv.get("transcript_metadata"))
                cached.setdefault("finalSummary", existing_conv.get("final_summary"))
                cached.setdefault("status", existing_conv.get("status", "active"))
                cached.setdefault("conversationName", existing_conv.get("conversation_name"))

                if not cached.get("questions") and existing_conv.get("chats"):
                    cached["questions"] = [
                        {
                            "questionId": c.get("chat_id"),
                            "question": c.get("entered_query"),
                            "answer": c.get("response"),
                            "relevantChunks": c.get("relevant_chunks", []),
                            "latency": c.get("latency", 0.0),
                            "confidence": c.get("confidence", 0.0),
                        }
                        for c in existing_conv.get("chats", [])
                    ]

                # Normalize cached format to required API contract:
                # relevantChunks must be list[str] and non-empty.
                try:
                    for q in cached.get("questions", []) or []:
                        rc = q.get("relevantChunks") or []
                        if isinstance(rc, list):
                            rc = [str(x) for x in rc if str(x).strip()]
                        else:
                            rc = []
                        if not rc and q.get("questionId") != "final_answer":
                            rc = ["(No supporting excerpts found)"]
                        if MILVUS_MAX_RETURN_CHUNKS is not None:
                            rc = rc[:MILVUS_MAX_RETURN_CHUNKS]
                        q["relevantChunks"] = rc
                except Exception as e:
                    print(f"Warning: failed to normalize cached relevantChunks: {e}")

                cached.setdefault(
                    "finalAnswer",
                    {
                        "question": "Final Answer for transcript",
                        "answer": cached.get("finalSummary") or "",
                    },
                )

                return jsonify(cached), 200

            # If we are force reprocessing an existing conversation, update that document rather than creating a new one.
            if existing_conv and (force_reprocess and not new_conversation):
                conv_doc_id = existing_conv.get("_id")
                conv_name = existing_conv.get("conversation_name")
                existing_conv = None

            # Create / update a "processing" transcript conversation document early so the sidebar can show it immediately.
            # This is intentionally done BEFORE downloading the transcript / calling LLMs.
            now_ts = datetime.utcnow()
            # If a status was previously set for this transcript, apply it; otherwise default active.
            status_doc = qna_collection.find_one(
                {"doc_type": "transcript_status", "transcript_id": transcript_id},
                {"_id": 0, "status": 1},
            )
            transcript_status = (status_doc or {}).get("status") or "active"

            if not conv_name:
                base_name = (requested_conversation_name or transcript_file_name or "").strip() or transcript_id
                if new_conversation:
                    existing_count = qna_collection.count_documents(
                        {"doc_type": "transcript_conversation", "transcript_id": transcript_id}
                    )
                    conv_name = base_name if existing_count == 0 else f"{base_name} ({existing_count + 1})"
                else:
                    conv_name = base_name

            if conv_doc_id is None:
                # Create a new conversation doc for this processing run
                stub = {
                    "doc_type": "transcript_conversation",
                    "conversation_mode": "Calls",
                    "underlying_model": gpt_model,
                    "conversation_name": conv_name,
                    "transcript_id": transcript_id,
                    "contract_type": contract_type,
                    "selected_plan": selected_plan,
                    "selected_state": selected_state,
                    "query_time": now_ts,
                    "updated_at": now_ts,
                    "status": transcript_status,
                    "processing": True,
                    "chats": [],
                }
                inserted = qna_collection.insert_one(stub)
                conv_doc_id = inserted.inserted_id
            else:
                # Mark existing conversation as processing
                qna_collection.update_one(
                    {"_id": conv_doc_id},
                    {"$set": {"processing": True, "updated_at": now_ts}},
                )
            
            # Read transcript from GCP bucket
            with tracer.start_as_current_span('download-transcript'):
                if not gcs_fs:
                    return jsonify({"error": "GCP Storage not configured or unavailable"}), 500
                
                try:
                    transcript_content, file_metadata = read_transcript_file_gcp(transcript_file_name)
                    
                    # Parse transcript (assuming JSON format)
                    try:
                        transcript_data = json.loads(transcript_content)
                        if isinstance(transcript_data, dict):
                            transcript_text = transcript_data.get("text", 
                                transcript_data.get("transcript", 
                                transcript_data.get("content", str(transcript_data))))
                        else:
                            transcript_text = transcript_content
                    except json.JSONDecodeError:
                        # If not JSON, treat as plain text
                        transcript_text = transcript_content
                    
                except FileNotFoundError as e:
                    return jsonify({"error": f"Transcript file not found: {transcript_file_name}"}), 404
                except Exception as e:
                    return jsonify({"error": f"Error reading transcript file: {str(e)}"}), 500
            
            # Extract questions
            questions = []
            if extract_questions:
                with tracer.start_as_current_span('extract-questions'):
                    llm_extract = ChatOpenAI(temperature=0.0, model="gpt-4o")
                    # Try direct extraction first (more reliable), then agent if needed
                    print(f"DEBUG: Attempting direct extraction first...")
                    questions = extract_relevant_customer_questions(transcript_text, llm_extract)
                    
                    # If direct extraction fails, try agent-based extraction
                    if not questions or len(questions) == 0:
                        print(f"DEBUG: Direct extraction returned no questions, trying agent-based extraction...")
                        questions = extract_questions_with_agent(transcript_text, llm_extract)
                    
                    if not questions:
                        print(f"ERROR: No questions extracted from transcript '{transcript_file_name}'")
                        print(f"ERROR: Transcript length: {len(transcript_text)} characters")
                        print(f"ERROR: First 500 chars of transcript: {transcript_text[:500]}")
                        extraction_warning = (
                            "No questions could be extracted from transcript; inferring from context."
                        )
                        inferred_question = {
                            "question": f"Is this issue covered: {transcript_text[:120]}",
                            "context": transcript_text[:400],
                            "questionType": "coverage",
                            "userIntent": "Customer wants to know if the described issue is covered",
                            "questionId": "q1",
                        }
                        questions = [inferred_question]
            else:
                questions = provided_questions
                if not questions:
                    return jsonify({"error": "No questions provided"}), 400
            
            # Initialize vector DB and LLM
            with tracer.start_as_current_span('vector_db-initialization'):
                collection_mapping = {
                    "RE": {
                        "ShieldEssential": f"{milvus_state}_RE_ShieldEssential",
                        "ShieldPlus": f"{milvus_state}_RE_ShieldPlus",
                        "default": f"{milvus_state}_RE_ShieldComplete",
                    },
                    "DTC": {
                        "ShieldSilver": f"{milvus_state}_DTC_ShieldSilver",
                        "ShieldGold": f"{milvus_state}_DTC_ShieldGold",
                        "default": f"{milvus_state}_DTC_ShieldPlatinum",
                    },
                }
                
                selected_collection_name = collection_mapping.get(contract_type_norm, {}).get(
                    selected_plan_norm, collection_mapping.get(contract_type_norm, {}).get("default")
                )
                print(
                    "[MILVUS] /transcripts/process selected_state="
                    f"{selected_state!r} -> milvus_state={milvus_state!r}, "
                    f"contract_type={contract_type!r}->{contract_type_norm!r}, "
                    f"selected_plan={selected_plan!r}->{selected_plan_norm!r}, "
                    f"collection={selected_collection_name!r}"
                )
                
                vector_db1 = Milvus(
                    embed,
                    collection_name=selected_collection_name,
                    connection_args={"host": MILVUS_HOST, "port": "19530"},
                )
                
                retriever = vector_db1.as_retriever(search_kwargs={"k": MILVUS_RETRIEVER_K})
                
                if gpt_model == "Search":
                    llm2 = ChatOpenAI(temperature=0.0, model="ft:gpt-3.5-turbo-0613:mindstix::8YYD56aA")
                    llm = ChatOpenAI(temperature=0.0, model="gpt-4o")
                elif gpt_model == "Infer":
                    llm3 = ChatOpenAI(temperature=0.0, model="ft:gpt-3.5-turbo-0613:mindstix::8YYD56aA")
                    llm = ChatOpenAI(temperature=0.0, model='gpt-4o')
                    llm2 = ChatOpenAI(temperature=0.0, model='gpt-4o')
                else:
                    return jsonify({"error": f"Invalid gpt_model: {gpt_model}. Must be 'Search' or 'Infer'"}), 400
            
            # Process each question
            results = []
            total_latency = 0
            confidences = []
            
            with tracer.start_as_current_span('process-questions'):
                for question_obj in questions:
                    question_text = question_obj.get("question", "")
                    question_id = question_obj.get("questionId", f"q{len(results) + 1}")
                    
                    result = process_single_transcript_question(
                        question_text, contract_type, selected_plan, 
                        selected_state, gpt_model, vector_db1, llm, llm2, 
                        retriever, handler,
                        transcript_context=question_obj.get("context", ""),
                    )
                    
                    result["questionId"] = question_id
                    result["question"] = question_text
                    result["context"] = question_obj.get("context", "")
                    result["questionType"] = question_obj.get("questionType", "general")
                    result["userIntent"] = question_obj.get("userIntent", "")  # Include user intent if available

                    # Handle relevantChunks - they should be objects with name, content, metadata
                    rc = result.get("relevantChunks") or []
                    if isinstance(rc, list) and len(rc) > 0:
                        # Check if chunks are objects or strings
                        if isinstance(rc[0], dict):
                            # Already objects with names - use as is
                            if MILVUS_MAX_RETURN_CHUNKS is not None:
                                rc = rc[:MILVUS_MAX_RETURN_CHUNKS]
                        else:
                            # Strings - convert to objects (backward compatibility)
                            # This shouldn't happen with new code, but handle old data
                            print(f"[WARNING] Chunks are strings, converting to objects")
                            rc = [{"content": str(x), "name": f"Clause {i+1}"} for i, x in enumerate(rc) if str(x).strip()]
                            if MILVUS_MAX_RETURN_CHUNKS is not None:
                                rc = rc[:MILVUS_MAX_RETURN_CHUNKS]
                    else:
                        rc = [{"content": "(No supporting excerpts found)", "name": "Clause 1"}]
                    result["relevantChunks"] = rc

                    rc = result.get("relevantChunks", [])
                    # print(
                    #     "[CHUNKS] /transcripts/process: per-question result "
                    #     f"questionId={question_id}, relevantChunks_count={len(rc)}"
                    # )
                    # Log the actual relevantChunks we are about to include in the response
                    # try:
                    #     def _chunk_preview(c):
                    #         # relevantChunks is list[str] (new contract) but support legacy dict chunks too
                    #         if isinstance(c, dict):
                    #             return {
                    #                 "content_preview": (c.get("content", "") or "")[:200].replace(chr(10), " "),
                    #                 "score": c.get("score"),
                    #             }
                    #         return {
                    #             "content_preview": (str(c) or "")[:200].replace(chr(10), " "),
                    #             "score": None,
                    #         }

                    #     print(
                    #         "[CHUNKS] /transcripts/process: per-question relevantChunks_detail="
                    #         f"{[_chunk_preview(c) for c in rc]}"
                    #     )
                    # except Exception as e:
                    #     print(f"[CHUNKS] /transcripts/process: unable to log chunk detail: {e}")
                    
                    if "error" not in result:
                        confidences.append(result.get("confidence", 0.0))
                        total_latency += result.get("latency", 0.0)
                    
                    results.append(result)
            
            # Calculate summary
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            response = {
                "transcriptId": transcript_id,
                "transcriptMetadata": {
                    "fileName": file_metadata["fileName"],
                    "uploadDate": file_metadata["uploadDate"],
                    "fileSize": file_metadata["fileSize"]
                },
                "questions": results,
                "summary": {
                    "totalQuestions": len(questions),
                    "processedQuestions": len([r for r in results if "error" not in r]),
                    "averageConfidence": round(avg_confidence, 2),
                    "totalLatency": round(total_latency, 2)
                }
            }
            if extraction_warning:
                response["warning"] = extraction_warning

            # Claim decision (Approved/Rejected/Cannot determine), grounded only in retrieved policy chunks
            try:
                all_chunks = []
                for r in results or []:
                    rc = r.get("relevantChunks") or []
                    if isinstance(rc, list):
                        all_chunks.extend([str(x) for x in rc if str(x).strip()])
                # de-duplicate while preserving order
                seen = set()
                deduped = []
                for c in all_chunks:
                    if c in seen:
                        continue
                    seen.add(c)
                    deduped.append(c)
                claims_context = []
                for r in results or []:
                    if not isinstance(r, dict):
                        continue
                    claims_context.append(
                        {
                            "claimId": (r.get("questionId") or ""),
                            "customerClaim": (r.get("question") or ""),
                            "situation": (r.get("context") or ""),
                        }
                    )
                claim_decision = generate_claim_decision_from_chunks(deduped, claims_context=claims_context)
                response["claimDecision"] = claim_decision
            except Exception as e:
                print(f"Warning: unable to generate claimDecision: {e}")

            # Build a final answer: combined summary of answers across ALL extracted questions.
            # We intentionally include every Q/A we produced (even if confidence is low),
            # and only skip items that have no question text at all.
            final_summary_text = ""
            try:
                with tracer.start_as_current_span('final-summary'):
                    llm_summary = ChatOpenAI(temperature=0.0, model="gpt-4o")
                    qa_lines = []
                    for r in results or []:
                        if not r:
                            continue
                        q = (r.get("question") or "").strip()
                        if not q:
                            continue
                        ctx = (r.get("context") or "").strip()
                        a = (r.get("answer") or "").strip()
                        # If answer is missing but question exists, keep a placeholder so the final summary
                        # still reflects ALL extracted questions.
                        if not a:
                            a = "(No answer was generated for this question.)"
                        if ctx:
                            qa_lines.append(f"Q: {q}\nSituation: {ctx}\nA: {a}")
                        else:
                            qa_lines.append(f"Q: {q}\nA: {a}")

                    qa_blob = "\n\n".join(qa_lines)
                    if qa_blob.strip():
                        summary_prompt = PromptTemplate(
                            input_variables=["qa_blob"],
                            template=(
                                "You are writing the FINAL ANSWER for a claims transcript.\n"
                                "IMPORTANT: Do NOT present the final answer as a list of each Q&A.\n"
                                "Instead, synthesize ALL Q&A into an APPLIANCE/ITEM-BASED final answer.\n"
                                "\n"
                                "Task:\n"
                                "- Identify the distinct appliance(s)/item(s)/system(s) mentioned across the Q&A.\n"
                                "- Group/merge related questions into the correct item section (do not repeat the questions).\n"
                                "- If the transcript includes multiple items with separate claims, show them as separate sections.\n"
                                "\n"
                                "For EACH item section, include in JSON FORMAT:\n"
                                "- ITEM : <1,2,3...>\n"
                                "- ITEM: <name> (add 1-line details if available: location/part/symptom)\n"
                                "- TYPE: Appliance | System | Fixture | Other (infer from wording; if unclear use Other)\n"
                                "- DECISION: APPROVED | REJECTED | PARTIAL | NEED_HUMAN_ASSISTANCE\n"
                                "- AMOUNTS (only if mentioned in Q&A):\n"
                                "  1. Customer quoted/asked: $...\n"
                                "  2. Company can provide: $... (coverage amount/limit/service fee/deductible as stated in Q&A)\n"
                                "- Situation: what happened / what customer is claiming (from Situation lines)\n"
                                "- What's covered (numeric list, if any)\n"
                                "- What's not covered / limitations (numeric list, if any)\n"
                                "- Why (1–2 short sentences grounded in the Q&A outcomes; no policy speculation)\n"
                                "- Next steps (specific actions the customer should take)\n"
                                "\n"
                                "CRITICAL DECISION RULES:\n"
                                "- The DECISION field is MANDATORY and MUST NEVER be left empty for any item.\n"
                                "- If it is confirmed that there is NO coverage for a particular item, the DECISION MUST be REJECTED.\n"
                                "- If outcomes are mixed for the same item, use PARTIAL and clearly break down covered vs not covered.\n"
                                "- If coverage cannot be determined, use NEED_HUMAN_ASSISTANCE.\n"
                                "- Be concise, decisive, and avoid hypothetical/if-then language.\n"
                                "- End with a short overall next step (1–2 bullets) if multiple items exist.\n\n"
                                "{qa_blob}\n"
                            ),
                        )
                        summary_chain = summary_prompt | llm_summary | StrOutputParser()
                        final_summary_text = summary_chain.invoke({"qa_blob": qa_blob}).strip()
            except Exception as e:
                print(f"Warning: failed to generate final transcript summary: {e}")

            # Ensure Final Answer is always present when we have questions (even if summarization failed).
            if (not final_summary_text.strip()) and (results and len(results) > 0):
                final_summary_text = "\n".join(
                    [
                        f"- {((r.get('answer') or '').strip() or '(No answer was generated for this question.)')}"
                        for r in results
                        if r and (r.get("question") or "").strip()
                    ]
                ).strip()

            response["finalSummary"] = final_summary_text
            response["finalAnswer"] = {
                "question": "Final Answer for transcript",
                "answer": final_summary_text,
            }

            total_chunks = sum(len(r.get("relevantChunks", [])) for r in results)
            print(
                "[CHUNKS] /transcripts/process: DONE "
                f"fileName={file_metadata['fileName']}, "
                f"questions={len(results)}, total_chunks={total_chunks}, "
                f"avg_confidence={round(avg_confidence, 2)}, total_latency={round(total_latency, 2)}"
            )

            # Persist transcript Q&A and chunks in MongoDB (per user) in the existing chat collection
            transcript_chats = []
            now_ts = datetime.utcnow()
            for res in results:
                # chunks should be objects with name, content, metadata
                chunks = res.get("relevantChunks", []) or []
                # Extract content for text blob (legacy /referred-clauses)
                chunk_contents = []
                for c in chunks:
                    if isinstance(c, dict):
                        chunk_contents.append(str(c.get("content", "")))
                    else:
                        chunk_contents.append(str(c))
                relevant_docs_text = "\n\n---\n\n".join([c for c in chunk_contents if c.strip()])
                
                # Ensure chunks are objects with names before storing
                normalized_chunks = _normalize_chunks_with_names(chunks)
                
                transcript_chats.append({
                    "chat_id": res.get("questionId"),
                    "entered_query": res.get("question", ""),
                    "response": res.get("answer", ""),
                    # For UI: keep chunks as objects with names
                    "relevant_chunks": normalized_chunks,
                    # For existing /referred-clauses UI: keep a text version too
                    "relevant_docs": relevant_docs_text,
                    # Conversation is a Calls mode conversation in UI; keep underlying model separately.
                    "gpt_model": "Calls",
                    "underlying_model": gpt_model,
                    "chat_timestamp": now_ts,
                    "latency": res.get("latency", 0.0),
                    "confidence": res.get("confidence", 0.0),
                })

            # Store final answer as a final chat entry in MongoDB, using a fixed question label
            transcript_chats.append({
                "chat_id": "final_answer",
                "entered_query": "Final Answer for transcript",
                "response": final_summary_text,
                "relevant_chunks": [],
                "relevant_docs": "",
                "gpt_model": "Calls",
                "underlying_model": gpt_model,
                "chat_timestamp": now_ts,
                "latency": 0.0,
                "confidence": 0.0,
            })

            # Also include it in the response questions list so the UI can render it as the last Q/A.
            response["questions"] = (response.get("questions") or []) + [{
                "questionId": "final_answer",
                "question": "Final Answer for transcript",
                "answer": final_summary_text,
                "relevantChunks": [],
                "confidence": 0.0,
                "latency": 0.0,
            }]

            transcript_doc = {
                "doc_type": "transcript_conversation",
                "conversation_mode": "Calls",
                "underlying_model": gpt_model,
                "conversation_name": conv_name or transcript_file_name,
                "transcript_id": transcript_id,
                "transcript_metadata": response["transcriptMetadata"],
                "contract_type": contract_type,
                "selected_plan": selected_plan,
                "selected_state": selected_state,
                "query_time": now_ts,
                "updated_at": now_ts,
                "status": transcript_status,
                "processing": False,
                "summary": response.get("summary"),
                "final_summary": final_summary_text,
                "claim_decision": response.get("claimDecision"),
                "chats": transcript_chats,
            }

            # Update the conversation document created earlier (so sidebar shows it during processing).
            if conv_doc_id is None:
                inserted = qna_collection.insert_one(transcript_doc)
                conv_doc_id = inserted.inserted_id
            else:
                qna_collection.update_one(
                    {"_id": conv_doc_id},
                    {"$set": transcript_doc},
                )

            updated_conv = qna_collection.find_one({"_id": conv_doc_id}) or {}

            response["conversationId"] = str(conv_doc_id)
            response["status"] = transcript_status
            response["conversationName"] = updated_conv.get("conversation_name") or transcript_doc["conversation_name"]
            # Persist full response payload for fast future reads
            qna_collection.update_one(
                {"_id": conv_doc_id},
                {"$set": {"response_payload": response}},
            )

            print(
                "[CHUNKS] /transcripts/process: stored transcript processing result "
                f"transcript_id={transcript_id}, conversation_id={response['conversationId']}, "
                f"questions={len(results)}, total_chunks={total_chunks}"
            )

            return jsonify(response), 200
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in /transcripts/process endpoint: {str(e)}")
        print(f"Traceback: {error_trace}")
        return jsonify({
            "error": "An error occurred while processing transcript", 
            "details": str(e)
        }), 500


@app.route("/internal/transcripts/process", methods=["POST"])
def process_transcript_internal():
    """
    Internal transcript processor (streaming)
    (Cloud Run / system-triggered)
    - No JWT required
    - Resolves user_email via DB mapping (contactId/session OR agentName mapping)
    - Stores results into chats_<user_email> same as normal flow
    - Streams responses via SSE like /transcripts/process/stream
    """
    print("process_transcript_internal: Starting streaming transcript processing")
    
    @stream_with_context
    def generate():
        start_time = time()
        user_email = None
        qna_collection = None
        conv_doc_id = None
        conv_name = None
        transcript_status = "active"
        extraction_warning = None
        
        try:
            # Note: `contactId` is used as the live session correlation key in this internal processor.
            # We establish a single trace per sessionId by parenting this branch off csr_copilot.session.
            parent0 = None
            parent_ctx = None
            # We'll read JSON in claims.data_fetching span below; placeholder here.

            # Pre-read JSON once so we can correlate this request to the live session trace root.
            # This does NOT change behavior; auth is still enforced before any processing.
            try:
                _pre_data = request.get_json() or {}
            except Exception:
                _pre_data = {}
            _session_id = (_pre_data.get("contactId") or _pre_data.get("sessionId") or "").strip()
            parent_ctx = _get_or_create_session_trace_context(_session_id) if _session_id else None

            with tracer.start_as_current_span("claims.transcript_processing", context=parent_ctx) as parent0:
                if _session_id:
                    parent0.set_attribute("live.session_id", str(_session_id))
                parent0.set_attribute("agent.name", "claims-transcript-processor")
                parent0.set_attribute("agent.type", "system")
                parent0.set_attribute("agent.orchestration", "sequential")

                # --- Internal auth (simple shared secret) ---
                with tracer.start_as_current_span("claims.internal_auth") as sp:
                    if _session_id:
                        sp.set_attribute("live.session_id", str(_session_id))
                    sp.set_attribute("agent.name", "claims-transcript-processor")
                    sp.set_attribute("agent.type", "system")
                    sp.set_attribute("agent.orchestration", "sequential")
                    sp.set_attribute("claims.stage", "security")
                    sp.set_attribute("claims.operation", "internal_auth")
                    expected = os.getenv("INTERNAL_PROCESS_SECRET")
                    got = request.headers.get("X-Internal-Auth")
                    if not expected or got != expected:
                        yield _sse("error", {"error": "unauthorized"})
                        return

                # --- Request body ---
                with tracer.start_as_current_span("claims.data_fetching") as sp:
                    if _session_id:
                        sp.set_attribute("live.session_id", str(_session_id))
                    sp.set_attribute("agent.name", "claims-transcript-processor")
                    sp.set_attribute("agent.type", "system")
                    sp.set_attribute("agent.orchestration", "sequential")
                    sp.set_attribute("claims.stage", "enrichment")
                    sp.set_attribute("claims.operation", "data_fetch")
                    data = _pre_data or request.get_json()
                    if not data:
                        yield _sse("error", {"error": "Request body is missing or invalid"})
                        return

                    transcript_file_name = data.get("transcriptFileName")
                    contract_type = data.get("contractType")
                    selected_plan = data.get("selectedPlan")
                    selected_state = data.get("selectedState")

                    gpt_model = data.get("gptModel", "Search")
                    extract_questions = data.get("extractQuestions", True)
                    provided_questions = data.get("questions", [])
                    force_reprocess = bool(data.get("forceReprocess", False))
                    new_conversation = bool(data.get("newConversation", False))
                    requested_conversation_name = data.get("conversationName")

                    contact_id = data.get("contactId")
                    agent_name = data.get("agentName")
                    # session correlation (prefer contactId)
                    session_id = (contact_id or data.get("sessionId") or "").strip()

                    # Validate required fields
                    if not transcript_file_name:
                        yield _sse("error", {"error": "transcriptFileName is required"})
                        return

                    if extract_questions and not all([contract_type, selected_plan, selected_state]):
                        yield _sse("error", {
                            "error": "contractType, selectedPlan, selectedState are required when extractQuestions=true"
                        })
                        return

                transcript_id = transcript_file_name.replace(".json", "").replace(".txt", "")
                
                yield _sse(
                    "status",
                    {
                        "stage": "started",
                        "transcriptId": transcript_id,
                        "transcriptFileName": transcript_file_name,
                        "gptModel": gpt_model,
                    },
                )

                # --- Resolve user email from mappings ---
                with tracer.start_as_current_span('claims.resolve_email') as sp:
                    if session_id:
                        sp.set_attribute("live.session_id", str(session_id))
                    sp.set_attribute("agent.name", "claims-transcript-processor")
                    sp.set_attribute("agent.type", "system")
                    sp.set_attribute("agent.orchestration", "sequential")
                    sp.set_attribute("claims.stage", "enrichment")
                    sp.set_attribute("claims.operation", "resolve_email")
                    user_email = None

                    # (A) Best: resolve from sessions mapping (contactId -> email)
                    sessions_collection = db.get_collection("sessions")  # safer

                    if contact_id:
                        sess = sessions_collection.find_one(
                            {"contactId": contact_id},
                            {"_id": 0, "email": 1}
                        )
                        if sess and sess.get("email"):
                            user_email = sess["email"]

                    # (B) Fallback: resolve from agent_email_mapping (agentName -> email)
                    if not user_email and agent_name:
                        agent_email_mapping = db.get_collection("agent_email_mapping")
                        m = agent_email_mapping.find_one(
                            {"agentName": agent_name},
                            {"_id": 0, "email": 1}
                        )
                        if m and m.get("email"):
                            user_email = m["email"]

                    if not user_email:
                        yield _sse("error", {
                            "error": "Unable to resolve user email",
                            "details": {
                                "contactId": contact_id,
                                "agentName": agent_name
                            }
                        })
                        return

                # --- Now same as your existing logic, starting from here ---
                milvus_state = normalize_state_for_milvus(selected_state)
                contract_type_norm = normalize_contract_type(contract_type)
                selected_plan_norm = normalize_plan_for_milvus(contract_type_norm, selected_plan)

                # Use the existing per-user chat collection (same as Search/Infer) for transcript conversations.
                qna_collection_user = f"chats_{user_email}"
                qna_collection = db[qna_collection_user]

                # If we have already processed this transcript for this user, stream the cached conversation.
                existing_conv = None
                conv_doc_id = None
                conv_name = None

                if not new_conversation:
                    existing_conv = qna_collection.find_one(
                        {"doc_type": "transcript_conversation", "transcript_id": transcript_id},
                        sort=[("updated_at", -1), ("query_time", -1)],
                    )

                if existing_conv and not force_reprocess and not new_conversation:
                    # cache validation same as your original
                    try:
                        existing_chats = existing_conv.get("chats") or []
                        has_placeholder_chunks = False
                        for c in existing_chats:
                            if c.get("chat_id") == "final_answer":
                                continue
                            rc = c.get("relevant_chunks") or []
                            if rc and all(
                                (
                                    (
                                        isinstance(x, dict)
                                        and (str(x.get("content") or "").strip() in _PLACEHOLDER_CHUNK_VALUES)
                                    )
                                    or (
                                        isinstance(x, str)
                                        and (x.strip() in _PLACEHOLDER_CHUNK_VALUES)
                                    )
                                )
                                for x in rc
                            ):
                                has_placeholder_chunks = True
                                break
                        if has_placeholder_chunks or not existing_conv.get("final_summary"):
                            conv_doc_id = existing_conv.get("_id")
                            conv_name = existing_conv.get("conversation_name")
                            existing_conv = None
                    except Exception as e:
                        print(f"Warning: cache validation failed, will reprocess transcript: {e}")
                        existing_conv = None

                if existing_conv and not force_reprocess and not new_conversation:
                    cached = existing_conv.get("response_payload") or {}
                    conv_doc_id = existing_conv.get("_id")
                    yield _sse(
                        "status",
                        {
                            "stage": "cached",
                            "conversationId": str(conv_doc_id),
                            "conversationName": existing_conv.get("conversation_name") or "",
                            "status": (existing_conv.get("status") or "active"),
                        },
                    )

                    for q in (cached.get("questions") or existing_conv.get("chats") or []):
                        if not isinstance(q, dict):
                            continue
                        qid = q.get("questionId") or q.get("chat_id")
                        if qid == "final_answer":
                            continue
                        yield _sse(
                            "answer",
                            {
                                "questionId": qid,
                                "question": q.get("question") or q.get("entered_query") or "",
                                "answer": q.get("answer") or q.get("response") or "",
                                "relevantChunks": q.get("relevantChunks") or q.get("relevant_chunks") or [],
                                "confidence": q.get("confidence", 0.0),
                                "latency": q.get("latency", 0.0),
                                "questionType": q.get("questionType"),
                                "userIntent": q.get("userIntent"),
                            },
                        )

                    if isinstance(existing_conv.get("claim_decision"), dict):
                        yield _sse("claimDecision", existing_conv.get("claim_decision"))

                    final_summary = existing_conv.get("final_summary") or cached.get("finalSummary") or ""
                    yield _sse("final", {"finalSummary": final_summary})
                    yield _sse("done", {"elapsedSec": round(time() - start_time, 2)})
                    return

                # If we are force reprocessing an existing conversation, update that document rather than creating a new one.
                if existing_conv and (force_reprocess and not new_conversation):
                    conv_doc_id = existing_conv.get("_id")
                    conv_name = existing_conv.get("conversation_name")
                    existing_conv = None

                # Create / update a "processing" transcript conversation document early
                now_ts = datetime.utcnow()

                status_doc = qna_collection.find_one(
                    {"doc_type": "transcript_status", "transcript_id": transcript_id},
                    {"_id": 0, "status": 1},
                )
                transcript_status = (status_doc or {}).get("status") or "active"

                if not conv_name:
                    base_name = (requested_conversation_name or transcript_file_name or "").strip() or transcript_id
                    if new_conversation:
                        existing_count = qna_collection.count_documents(
                            {"doc_type": "transcript_conversation", "transcript_id": transcript_id}
                        )
                        conv_name = base_name if existing_count == 0 else f"{base_name} ({existing_count + 1})"
                    else:
                        conv_name = base_name

                if conv_doc_id is None:
                    stub = {
                        "doc_type": "transcript_conversation",
                        "conversation_mode": "Calls",
                        "underlying_model": gpt_model,
                        "conversation_name": conv_name,
                        "transcript_id": transcript_id,
                        "contract_type": contract_type,
                        "selected_plan": selected_plan,
                        "selected_state": selected_state,
                        "query_time": now_ts,
                        "updated_at": now_ts,
                        "status": transcript_status,
                        "processing": True,
                        "chats": [],
                        # Helpful for debugging / audit
                        "internal_trigger": True,
                        "contact_id": contact_id,
                        "agent_name": agent_name,
                        "user_email": user_email,
                    }
                    inserted = qna_collection.insert_one(stub)
                    conv_doc_id = inserted.inserted_id
                else:
                    qna_collection.update_one(
                        {"_id": conv_doc_id},
                        {"$set": {"processing": True, "updated_at": now_ts}},
                    )

                yield _sse(
                    "status",
                    {
                        "stage": "conversation_created",
                        "conversationId": str(conv_doc_id),
                        "conversationName": conv_name,
                        "status": transcript_status,
                    },
                )

                # Read transcript from GCS
                if not gcs_fs:
                    yield _sse("error", {"error": "GCP Storage not configured or unavailable"})
                    return

                yield _sse("status", {"stage": "transcript_loading"})
                file_metadata = None
                try:
                    transcript_content, file_metadata = read_transcript_file_gcp(transcript_file_name)
                    transcript_text = transcript_content
                    try:
                        transcript_data = json.loads(transcript_content)
                        if isinstance(transcript_data, dict):
                            transcript_text = transcript_data.get(
                                "text",
                                transcript_data.get(
                                    "transcript",
                                    transcript_data.get("content", str(transcript_data)),
                                ),
                            )
                    except Exception:
                        transcript_text = transcript_content
                except FileNotFoundError as e:
                    error_msg = f"Transcript file not found: {transcript_file_name}"
                    print(f"ERROR: {error_msg}")
                    yield _sse("error", {"error": error_msg, "stage": "transcript_loading"})
                    return
                except Exception as e:
                    error_msg = f"Error reading transcript file: {str(e)}"
                    print(f"ERROR: {error_msg}")
                    import traceback
                    traceback.print_exc()
                    yield _sse("error", {"error": error_msg, "stage": "transcript_loading", "details": str(e)})
                    return
                
                if not file_metadata:
                    error_msg = f"Failed to retrieve file metadata for: {transcript_file_name}"
                    print(f"ERROR: {error_msg}")
                    yield _sse("error", {"error": error_msg, "stage": "transcript_loading"})
                    return

                yield _sse(
                    "status",
                    {
                        "stage": "transcript_loaded",
                        "transcriptMetadata": {
                            "fileName": file_metadata.get("fileName"),
                            "uploadDate": file_metadata.get("uploadDate"),
                            "fileSize": file_metadata.get("fileSize"),
                        },
                    },
                )

                # Extract questions
                if extract_questions:
                    yield _sse("status", {"stage": "extracting_questions"})
                    llm_extract = ChatOpenAI(temperature=0.0, model="gpt-4o")
                    questions = extract_relevant_customer_questions(transcript_text, llm_extract)
                    if not questions:
                        questions = extract_questions_with_agent(transcript_text, llm_extract)
                    if not questions:
                        extraction_warning = "No questions could be extracted from transcript; inferring from context."
                        inferred_question = {
                            "question": f"Is this issue covered: {transcript_text[:120]}",
                            "context": transcript_text[:400],
                            "questionType": "coverage",
                            "userIntent": "Customer wants to know if the described issue is covered",
                            "questionId": "q1",
                        }
                        questions = [inferred_question]
                else:
                    questions = provided_questions
                    if not questions:
                        yield _sse("error", {"error": "No questions provided"})
                        return

                yield _sse(
                    "status",
                    {
                        "stage": "questions_ready",
                        "totalQuestions": len(questions),
                        "warning": extraction_warning,
                    },
                )

                # Initialize vector DB + LLMs
                yield _sse("status", {"stage": "initializing_retriever"})
                collection_mapping = {
                    "RE": {
                        "ShieldEssential": f"{milvus_state}_RE_ShieldEssential",
                        "ShieldPlus": f"{milvus_state}_RE_ShieldPlus",
                        "default": f"{milvus_state}_RE_ShieldComplete",
                    },
                    "DTC": {
                        "ShieldSilver": f"{milvus_state}_DTC_ShieldSilver",
                        "ShieldGold": f"{milvus_state}_DTC_ShieldGold",
                        "default": f"{milvus_state}_DTC_ShieldPlatinum",
                    },
                }
                selected_collection_name = collection_mapping.get(contract_type_norm, {}).get(
                    selected_plan_norm, collection_mapping.get(contract_type_norm, {}).get("default")
                )
                vector_db1 = Milvus(
                    embed,
                    collection_name=selected_collection_name,
                    connection_args={"host": MILVUS_HOST, "port": "19530"},
                )
                retriever = vector_db1.as_retriever(search_kwargs={"k": MILVUS_RETRIEVER_K})

                if gpt_model == "Search":
                    llm2 = ChatOpenAI(temperature=0.0, model="ft:gpt-3.5-turbo-0613:mindstix::8YYD56aA")
                    llm = ChatOpenAI(temperature=0.0, model="gpt-4o")
                elif gpt_model == "Infer":
                    llm3 = ChatOpenAI(temperature=0.0, model="ft:gpt-3.5-turbo-0613:mindstix::8YYD56aA")
                    llm = ChatOpenAI(temperature=0.0, model="gpt-4o")
                    llm2 = ChatOpenAI(temperature=0.0, model="gpt-4o")
                else:
                    yield _sse("error", {"error": f"Invalid gpt_model: {gpt_model}. Must be 'Search' or 'Infer'"})
                    return

                yield _sse("status", {"stage": "answering"})

                results = []
                confidences = []
                total_latency = 0.0

                # Process each question and stream immediately
                for idx, question_obj in enumerate(questions):
                    question_text = question_obj.get("question", "")
                    question_id = question_obj.get("questionId", f"q{idx + 1}")

                    yield _sse(
                        "status",
                        {"stage": "answering_question", "index": idx + 1, "questionId": question_id},
                    )

                    result = process_single_transcript_question(
                        question_text,
                        contract_type,
                        selected_plan,
                        selected_state,
                        gpt_model,
                        vector_db1,
                        llm,
                        llm2,
                        retriever,
                        handler,
                        transcript_context=question_obj.get("context", ""),
                    )

                    result["questionId"] = question_id
                    result["question"] = question_text
                    result["context"] = question_obj.get("context", "")
                    result["questionType"] = question_obj.get("questionType", "general")
                    result["userIntent"] = question_obj.get("userIntent", "")

                    # Handle relevantChunks - they should be objects with name, content, metadata
                    rc = result.get("relevantChunks") or []
                    if isinstance(rc, list) and len(rc) > 0:
                        # Check if chunks are objects or strings
                        if isinstance(rc[0], dict):
                            # Already objects with names - use as is
                            if MILVUS_MAX_RETURN_CHUNKS is not None:
                                rc = rc[:MILVUS_MAX_RETURN_CHUNKS]
                        else:
                            # Strings - convert to objects (backward compatibility)
                            print(f"[WARNING] Chunks are strings, converting to objects")
                            rc = [{"content": str(x), "name": f"Clause {i+1}"} for i, x in enumerate(rc) if str(x).strip()]
                            if MILVUS_MAX_RETURN_CHUNKS is not None:
                                rc = rc[:MILVUS_MAX_RETURN_CHUNKS]
                    else:
                        rc = [{"content": "(No supporting excerpts found)", "name": "Clause 1"}]
                    result["relevantChunks"] = rc

                    if "error" not in result:
                        confidences.append(result.get("confidence", 0.0))
                        total_latency += float(result.get("latency", 0.0) or 0.0)

                    results.append(result)

                    # Persist incremental chat to Mongo
                    try:
                        chunks = result.get("relevantChunks") or []
                        # Extract content for text blob
                        chunk_contents = []
                        for c in chunks:
                            if isinstance(c, dict):
                                chunk_contents.append(str(c.get("content", "")))
                            else:
                                chunk_contents.append(str(c))
                        relevant_docs_text = "\n\n---\n\n".join([c for c in chunk_contents if c.strip()])
                        
                        # Ensure chunks are objects with names before storing
                        normalized_chunks = _normalize_chunks_with_names(chunks)
                        
                        qna_collection.update_one(
                            {"_id": conv_doc_id},
                            {
                                "$push": {
                                    "chats": {
                                        "chat_id": question_id,
                                        "entered_query": question_text,
                                        "response": result.get("answer", ""),
                                        "relevant_chunks": normalized_chunks,
                                        "relevant_docs": relevant_docs_text,
                                        "gpt_model": "Calls",
                                        "underlying_model": gpt_model,
                                        "chat_timestamp": datetime.utcnow(),
                                        "latency": result.get("latency", 0.0),
                                        "confidence": result.get("confidence", 0.0),
                                    }
                                },
                                "$set": {"updated_at": datetime.utcnow()},
                            },
                        )
                    except Exception as e:
                        print(f"Warning: failed to persist incremental transcript chat: {e}")

                    # Stream this answer immediately
                    yield _sse(
                        "answer",
                        {
                            "questionId": question_id,
                            "question": question_text,
                            "answer": result.get("answer", ""),
                            "relevantChunks": result.get("relevantChunks", []),
                            "confidence": result.get("confidence", 0.0),
                            "latency": result.get("latency", 0.0),
                            "questionType": result.get("questionType"),
                            "userIntent": result.get("userIntent"),
                        },
                    )

                # Final summary (same logic as /transcripts/process/stream)
                final_summary_text = ""
                try:
                    llm_summary = ChatOpenAI(temperature=0.0, model="gpt-4o")
                    qa_lines = []
                    for r in results or []:
                        if not r:
                            continue
                        q = (r.get("question") or "").strip()
                        if not q:
                            continue
                        ctx = (r.get("context") or "").strip()
                        a = (r.get("answer") or "").strip() or "(No answer was generated for this question.)"
                        if ctx:
                            qa_lines.append(f"Q: {q}\nSituation: {ctx}\nA: {a}")
                        else:
                            qa_lines.append(f"Q: {q}\nA: {a}")
                    qa_blob = "\n\n".join(qa_lines)
                    if qa_blob.strip():
                        summary_prompt = PromptTemplate(
                            input_variables=["qa_blob"],
                            template=(
                                "You are writing the FINAL ANSWER for a claims transcript.\n"
                                "IMPORTANT: Do NOT present the final answer as a list of each Q&A.\n"
                                "Instead, synthesize ALL Q&A into an APPLIANCE/ITEM-BASED final answer.\n"
                                "\n"
                                "Task:\n"
                                "- Identify the distinct appliance(s)/item(s)/system(s) mentioned across the Q&A.\n"
                                "- Group/merge related questions into the correct item section (do not repeat the questions).\n"
                                "- If the transcript includes multiple items with separate claims, show them as separate sections.\n"
                                "\n"
                                "For EACH item section, include:\n"
                                "- Item : <1,2,3...>\n"
                                "- Item: <name> (add 1-line details if available: location/part/symptom)\n"
                                "- Type: Appliance | System | Fixture | Other (infer from wording; if unclear use Other)\n"
                                "- Related: related parts/components/secondary-damage items (if any)\n"
                                "- Situation: what happened / what customer is claiming (from Situation lines)\n"
                                "- Decision: APPROVED | REJECTED | PARTIAL | NEED_INFO\n"
                                "- What's covered (bullet list, if any)\n"
                                "- What's not covered / limitations (bullet list, if any)\n"
                                "- Amounts (only if mentioned in Q&A):\n"
                                "  - Customer quoted/asked: $...\n"
                                "  - Company can provide: $... (coverage amount/limit/service fee/deductible as stated in Q&A)\n"
                                "- Why (1–2 short sentences grounded in the Q&A outcomes; no policy speculation)\n"
                                "- Next steps (specific actions the customer should take)\n"
                                "\n"
                                "CRITICAL DECISION RULES:\n"
                                "- The Decision field is MANDATORY and MUST NEVER be left empty for any item.\n"
                                "- If it is confirmed that there is NO coverage for a particular item, the Decision MUST be REJECTED.\n"
                                "- If outcomes are mixed for the same item, use PARTIAL and clearly break down covered vs not covered.\n"
                                "- If coverage cannot be determined, use NEED_INFO.\n"
                                "- Be concise, decisive, and avoid hypothetical/if-then language.\n"
                                "- End with a short overall next step (1–2 bullets) if multiple items exist.\n\n"
                                "{qa_blob}\n"
                            ),
                        )
                        summary_chain = summary_prompt | llm_summary | StrOutputParser()
                        final_summary_text = summary_chain.invoke({"qa_blob": qa_blob}).strip()
                except Exception as e:
                    print(f"Warning: failed to generate final transcript summary (stream): {e}")

                if (not final_summary_text.strip()) and results:
                    final_summary_text = "\n".join(
                        [
                            f"- {((r.get('answer') or '').strip() or '(No answer was generated for this question.)')}"
                            for r in results
                            if r and (r.get("question") or "").strip()
                        ]
                    ).strip()

                avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

                # Claim decision grounded only in retrieved chunks (stream it before final summary UI finishes)
                claim_decision = None
                try:
                    all_chunks = []
                    for r in results or []:
                        rc = r.get("relevantChunks") or []
                        if isinstance(rc, list):
                            all_chunks.extend([str(x) for x in rc if str(x).strip()])
                    seen = set()
                    deduped = []
                    for c in all_chunks:
                        if c in seen:
                            continue
                        seen.add(c)
                        deduped.append(c)
                    claims_context = []
                    for r in results or []:
                        if not isinstance(r, dict):
                            continue
                        claims_context.append(
                            {
                                "claimId": (r.get("questionId") or ""),
                                "customerClaim": (r.get("question") or ""),
                                "situation": (r.get("context") or ""),
                            }
                        )
                    claim_decision = generate_claim_decision_from_chunks(deduped, claims_context=claims_context)
                    yield _sse("claimDecision", claim_decision)
                except Exception as e:
                    print(f"Warning: failed to generate/stream claimDecision: {e}")

                # Store final answer as last chat entry and finalize conversation doc
                try:
                    qna_collection.update_one(
                        {"_id": conv_doc_id},
                        {
                            "$push": {
                                "chats": {
                                    "chat_id": "final_answer",
                                    "entered_query": "Final Answer for transcript",
                                    "response": final_summary_text,
                                    "relevant_chunks": [],
                                    "relevant_docs": "",
                                    "gpt_model": "Calls",
                                    "underlying_model": gpt_model,
                                    "chat_timestamp": datetime.utcnow(),
                                    "latency": 0.0,
                                    "confidence": 0.0,
                                }
                            },
                            "$set": {
                                "processing": False,
                                "updated_at": datetime.utcnow(),
                                "final_summary": final_summary_text,
                                "claim_decision": claim_decision,
                                "summary": {
                                    "totalQuestions": len(questions),
                                    "processedQuestions": len([r for r in results if "error" not in r]),
                                    "averageConfidence": round(avg_confidence, 2),
                                    "totalLatency": round(total_latency, 2),
                                },
                                "transcript_metadata": {
                                    "fileName": file_metadata.get("fileName"),
                                    "uploadDate": file_metadata.get("uploadDate"),
                                    "fileSize": file_metadata.get("fileSize"),
                                },
                            },
                        },
                    )
                except Exception as e:
                    print(f"Warning: failed to finalize transcript conversation doc (stream): {e}")

                yield _sse("final", {"finalSummary": final_summary_text})
                yield _sse(
                    "done",
                    {
                        "elapsedSec": round(time() - start_time, 2),
                        "conversationId": str(conv_doc_id) if conv_doc_id else "",
                        "conversationName": conv_name or "",
                        "status": transcript_status,
                    },
                )
                return

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Error in /internal/transcripts/process endpoint: {str(e)}")
            print(f"Traceback: {error_trace}")
            yield _sse("error", {"error": "An error occurred while streaming transcript processing (internal)", "details": str(e)})
            return

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return Response(generate(), headers=headers)

@app.route("/webhook", methods=["POST"])
def transcript_event():
    # simple shared-secret auth
    # auth = request.headers.get("authorization", "")
    # if auth != f"Bearer {os.getenv('FLASK_AUTH_TOKEN')}":
    #     return {"error": "unauthorized"}, 401

    data = request.get_json()
    if not data:
        return jsonify({"error": "invalid payload"}), 400
    
    session_id = data.get("sessionId")
    if not session_id:
        return jsonify({"error": "sessionId is required"}), 400

    # Deduplication: Check if this exact transcript already exists
    dedup_query = {
        "sessionId": data.get("sessionId"),
        "speaker": data.get("speaker"),
        "text": data.get("text"),
        "beginOffsetMillis": data.get("beginOffsetMillis"),
        "endOffsetMillis": data.get("endOffsetMillis"),
    }
    
    existing = transcripts_collection.find_one(dedup_query)
    if existing:
        # Already received this transcript, skip logging and storing
        return jsonify({"ok": True, "duplicate": True}), 200

    transcript_doc = {
        "sessionId": data["sessionId"],
        "contactId": data.get("contactId"),
        "speaker": data.get("speaker"),
        "text": data.get("text"),
        "isPartial": data.get("isPartial", True),
        "beginOffsetMillis": data.get("beginOffsetMillis"),
        "endOffsetMillis": data.get("endOffsetMillis"),
        "createdAt": data.get("createdAt")
    }

    transcripts_collection.insert_one(transcript_doc)

    # broadcast to UI via websocket
    # 🔥 LOG TRANSCRIPT EVENT
    include_payloads = str(os.getenv("OTEL_TRACE_INCLUDE_PAYLOADS", "0") or "").strip().lower() in ("1", "true", "yes", "y", "on")
    if include_payloads:
        try:
            limit = int(os.getenv("OTEL_TRACE_PAYLOAD_PREVIEW_CHARS", "500") or 500)
        except Exception:
            limit = 500
        print("🔴 TRANSCRIPT RECEIVED (payload):", json.dumps(data, indent=2, default=str)[: max(0, limit)])
    else:
        try:
            txt = str(data.get("text") or "")
            print(
                "🔴 TRANSCRIPT RECEIVED (summary): "
                f"sessionId={data.get('sessionId')}, speaker={data.get('speaker')}, "
                f"isPartial={bool(data.get('isPartial', True))}, text_len={len(txt)}"
            )
        except Exception:
            pass
    socketio.emit("transcript_update", data)
    socketio.emit("transcript_update", data, room=data["sessionId"])

    # ========== LIVE COPILOT: Real-time AI suggestions ==========
    # Process through Live Copilot if:
    # 1. Module is available
    # 2. Feature flag is enabled
    # 3. Session has copilot enabled (via copilot_enable from UI)
    # 4. Transcript is complete (not partial)
    if (
        LIVE_COPILOT_AVAILABLE
        and _flag_enabled("ENABLE_LIVE_COPILOT", "0")
        and not data.get("isPartial", True)
    ):
        try:
            # Build copilot payload with session context
            # Include phone, state, plan, contractType from transcript payload
            copilot_payload = {
                "sessionId": session_id,
                "contactId": data.get("contactId"),
                "speaker": data.get("speaker"),
                "text": data.get("text"),
                "isPartial": data.get("isPartial", False),
                "beginOffsetMillis": data.get("beginOffsetMillis"),
                "endOffsetMillis": data.get("endOffsetMillis"),
                # New fields from transcript for session context
                # Support both 'phoneNumber' (Amazon Connect) and 'phone' keys
                "phoneNumber": data.get("phoneNumber") or data.get("phone"),
                "state": data.get("state"),
                "contractType": data.get("contractType"),
                "plan": data.get("plan"),
            }
            
            # Call Live Copilot to process transcript under the session trace root (1 trace per sessionId)
            parent_ctx = _get_or_create_session_trace_context(session_id)
            copilot_result = handle_transcript_event(copilot_payload, parent_context=parent_ctx)
            
            if copilot_result:
                if include_payloads:
                    try:
                        limit = int(os.getenv("OTEL_TRACE_PAYLOAD_PREVIEW_CHARS", "500") or 500)
                    except Exception:
                        limit = 500
                    print(
                        "🟢 COPILOT SUGGESTION (payload):",
                        json.dumps(copilot_result, indent=2, default=str)[: max(0, limit)],
                    )
                else:
                    try:
                        cards = copilot_result.get("cards") or []
                        print(
                            "🟢 COPILOT SUGGESTION (summary): "
                            f"sessionId={copilot_result.get('sessionId')}, intent={copilot_result.get('intent')}, cards={len(cards)}"
                        )
                    except Exception:
                        pass
                # Emit suggestion to UI
                socketio.emit("suggestion_update", copilot_result)
                socketio.emit("suggestion_update", copilot_result, room=session_id)
        except Exception as e:
            print(f"⚠️ Copilot processing error (non-blocking): {e}")
            import traceback
            traceback.print_exc()
    # =============================================================

    return jsonify({"ok": True}), 200

@socketio.on("connect")
def on_connect(auth):
    # auth is whatever you passed from frontend: { token: "..." }
    token = None
    if isinstance(auth, dict):
        token = auth.get("token")

    if not token:
        print("❌ No JWT in connect auth")
        disconnect()
        return

    token_data, status = token_process(f"Bearer {token}")
    if status in (401, 403):
        print("❌ Invalid JWT in connect auth")
        disconnect()
        return

    user_email = token_data.get("email")
    if not user_email:
        print("❌ Email missing in JWT")
        disconnect()
        return

    # ✅ store for later events
    session["user_email"] = user_email
    print(f"✅ Socket connected user: {user_email}")

@socketio.on("join_session")
def on_join_session(data):
    session_id = (data or {}).get("sessionId")
    if not session_id:
        return

    user_email = session.get("user_email")
    if not user_email:
        print("❌ No user_email in socket session")
        return

    join_room(session_id)

    db["sessions"].update_one(
        {"contactId": session_id},
        {"$setOnInsert": {
            "contactId": session_id,
            "email": user_email,
            "createdAt": datetime.utcnow()
        }},
        upsert=True
    )

    print(f"✅ Session mapped: {session_id} → {user_email}")

@socketio.on("copilot_enable")
def on_copilot_enable(data):
    """Enable Live Copilot for a session when call connects."""
    session_id = data.get("sessionId")
    if session_id:
        with _copilot_sessions_lock:
            _copilot_enabled_sessions[session_id] = time() + _copilot_session_ttl_seconds()
            _copilot_session_context[session_id] = {
                "contractType": data.get("contractType", ""),
                "selectedPlan": data.get("selectedPlan", ""),
                "selectedState": data.get("selectedState", ""),
            }
        print(f"🟢 COPILOT ENABLED for session: {session_id}")
        # Emit status back to UI
        socketio.emit("copilot_status", {
            "sessionId": session_id,
            "enabled": True
        }, room=session_id)

@socketio.on("copilot_disable")
def on_copilot_disable(data):
    """Disable Live Copilot when call ends."""
    session_id = data.get("sessionId")
    if session_id:
        with _copilot_sessions_lock:
            _copilot_enabled_sessions.pop(session_id, None)
            _copilot_session_context.pop(session_id, None)
        print(f"🔴 COPILOT DISABLED for session: {session_id}")
        # Emit status back to UI
        socketio.emit("copilot_status", {
            "sessionId": session_id,
            "enabled": False
        }, room=session_id)

if __name__ == "__main__":
    # use_reloader=False to avoid Windows socket errors during reload
    port = int(os.getenv("PORT", "5000"))
    debug = str(os.getenv("FLASK_DEBUG", "0")).lower() in ("1", "true", "yes")

    # Flask-SocketIO blocks Werkzeug in production by default; allow it if we ever
    # fall back to "threading" mode (eventlet/gevent are preferred when available).
    socketio.run(
        app,
        host="0.0.0.0",
        port=port,
        debug=debug,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
