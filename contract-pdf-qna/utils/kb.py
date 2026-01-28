from dotenv import load_dotenv
import time

load_dotenv()

from pymilvus import Collection, utility, connections
from functools import lru_cache
from langchain_community.vectorstores import Milvus
from utils.constants import (
    MILVUS_RETRIEVER_K,
    MILVUS_FALLBACK_K,
    MILVUS_MAX_RETURN_CHUNKS,
    CLEAR_STATE_ALIASES,
    _PLACEHOLDER_CHUNK_VALUES,
    GCP_SERVICE_ACCOUNT_PATH,
    TRANSCRIPT_METADATA_CACHE_VERSION,
)
from utils.milvus_utils import (
    normalize_contract_type,
    normalize_plan_for_milvus,
    normalize_state_for_milvus,
    get_milvus_collection_name,
)

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from config import MILVUS_HOST

collections_name = ["Arizona_RE_ShieldEssential", "Arizona_RE_ShieldPlus", "Arizona_DTC_ShieldPlatinum", "Arizona_DTC_ShieldSilver", "Arizona_DTC_ShieldGold", "California_RE_ShieldEssential", "California_RE_ShieldPlus", "California_DTC_ShieldPlatinum", "California_DTC_ShieldSilver", "California_DTC_ShieldGold", "Georgia_RE_ShieldEssential", "Georgia_RE_ShieldPlus", "Georgia_DTC_ShieldPlatinum", "Georgia_DTC_ShieldSilver", "Georgia_DTC_ShieldGold", "Maryland_RE_ShieldEssential", "Maryland_RE_ShieldPlus", "Maryland_DTC_ShieldPlatinum", "Maryland_DTC_ShieldSilver", "Maryland_DTC_ShieldGold", "Minnesota_RE_ShieldEssential", "Minnesota_RE_ShieldPlus", "Minnesota_DTC_ShieldPlatinum", "Minnesota_DTC_ShieldSilver", "Minnesota_DTC_ShieldGold", "Nevada_RE_ShieldEssential", "Nevada_RE_ShieldPlus", "Nevada_DTC_ShieldPlatinum", "Nevada_DTC_ShieldSilver", "Nevada_DTC_ShieldGold", "Texas_RE_ShieldEssential", "Texas_RE_ShieldPlus", "Texas_DTC_ShieldPlatinum", "Texas_DTC_ShieldSilver", "Texas_DTC_ShieldGold", "Utah_RE_ShieldEssential", "Utah_RE_ShieldPlus", "Utah_DTC_ShieldPlatinum", "Utah_DTC_ShieldSilver", "Utah_DTC_ShieldGold", "Wisconsin_RE_ShieldEssential", "Wisconsin_RE_ShieldPlus", "Wisconsin_DTC_ShieldPlatinum", "Wisconsin_DTC_ShieldSilver", "Wisconsin_DTC_ShieldGold", "Arizona_RE_ShieldComplete"] 
connections.connect(host=MILVUS_HOST, port="19530")
collections_vector_db = {}


model_name = "text-embedding-ada-002"
embed = OpenAIEmbeddings(model=model_name)

def preloadCollections():
    for collection_name in collections_name:
        vector_db1: Milvus = Milvus(
            embed,
            collection_name=collection_name,
            connection_args={"host": MILVUS_HOST, "port": "19530"},
        )
        collections_vector_db[collection_name] = vector_db1

#will load collections on import (only once)
preloadCollections()


def get_vector_db(collection_name: str) -> Milvus:
    vector_db1 = collections_vector_db.get(collection_name, None)
    if vector_db1 ==  None:
        raise Exception(f"Milvus collection {collection_name!r} not found")
    return vector_db1


#will have to reduce the maxsize as the return docs are large
#help with repeated questions
@lru_cache
def cache_fetch_chunks(selected_collection_name, query, k):
    vector_db1 = get_vector_db(selected_collection_name)
    retriever = vector_db1.as_retriever(search_kwargs={"k": max(1, min(int(k), 12))})

    raw_docs = retriever.get_relevant_documents(query)
    return raw_docs

def _retrieve_policy_chunks_for_claims(docs: dict, query: str, k: int = 6):
    """Best-effort policy retrieval from Milvus for a transcript conversation.

    Returns: (chunks_for_ui, referred_docs_text)
      - chunks_for_ui: list[dict] where each dict has {content, metadata}
      - referred_docs_text: a readable text blob for /referred-clauses legacy page
    """
    try:
        if not isinstance(docs, dict):
            return [], ""
        query = (query or "").strip()
        if not query:
            return [], ""

        contract_type = docs.get("contract_type")
        selected_plan = docs.get("selected_plan")
        selected_state = docs.get("selected_state")
        if not all([contract_type, selected_plan, selected_state]):
            return [], ""

        # Get collection name using utility function
        selected_collection_name = get_milvus_collection_name(
            contract_type=contract_type,
            selected_plan=selected_plan,
            selected_state=selected_state
        )
        if not selected_collection_name:
            return [], ""

        # Get normalized values for logging
        milvus_state = normalize_state_for_milvus(selected_state)
        contract_type_norm = normalize_contract_type(contract_type)
        selected_plan_norm = normalize_plan_for_milvus(contract_type_norm, selected_plan)

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

        s = time.time()
        raw_docs = cache_fetch_chunks(selected_collection_name, query, k)
        print('[CLAIM_FOLLOWUP][TIME DURATION] Chunk Extracted', time.time() - s, len(raw_docs))
        try:
            print(f"[CLAIMS_FOLLOWUP] Milvus returned {len(raw_docs or [])} docs")
        except Exception:
            pass
        chunks_for_ui = []
        text_lines = []
        for i, d in enumerate(raw_docs or [], start=1):
            content = (getattr(d, "page_content", "") or "").strip()
            metadata = getattr(d, "metadata", {}) or {}
            if not content:
                continue
            chunks_for_ui.append({"content": content, "metadata": metadata})
            # Build a readable blob for legacy referred clauses page
            src = ""
            if isinstance(metadata, dict):
                src = metadata.get("source") or metadata.get("file") or metadata.get("document") or ""
            header = f"Clause {i}"
            if src:
                header += f" ({src})"
            text_lines.append(header)
            text_lines.append(content)
            text_lines.append("")

        referred_docs_text = "\n".join(text_lines).strip()
        return chunks_for_ui, referred_docs_text
    except Exception as e:
        print(f"Warning: policy retrieval failed for claims followup: {e}")
        return [], ""

