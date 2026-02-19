from dotenv import load_dotenv
import time

load_dotenv()

from pymilvus import Collection, utility, connections
from functools import lru_cache, cache
from typing import Optional
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

from langchain_openai import OpenAIEmbeddings
from config import MILVUS_HOST
from tqdm import tqdm

collections_name = ["policies"] 
connections.connect(host=MILVUS_HOST, port="19530")
collections_vector_db = {}


model_name = "text-embedding-ada-002"
embed = OpenAIEmbeddings(model=model_name)

def preloadCollections():
    print('Intializing Vector DB...')
    for collection_name in tqdm(collections_name):
        vector_db1: Milvus = Milvus(
            embed,
            collection_name=collection_name,
            connection_args={"host": MILVUS_HOST, "port": "19530"},
        )
        collections_vector_db[collection_name] = vector_db1

#will load collections on import (only once)
preloadCollections()

def getPolicyid(*,
    contract_type: str,
    selected_plan: str,
    selected_state: str
) -> str:
    """
    Get the Milvus collection name based on contract type, plan, and state.
    
    Args:
        contract_type: Contract type (RE or DTC)
        selected_plan: Plan name (e.g., "ShieldPlus", "ShieldGold")
        selected_state: State name or abbreviation (e.g., "California", "CA")
        
    Returns:
        Milvus collection name (e.g., "California_RE_ShieldPlus"), or None if invalid
    """
    milvus_state = normalize_state_for_milvus(selected_state)
    contract_type_norm = normalize_contract_type(contract_type)
    selected_plan_norm = normalize_plan_for_milvus(contract_type_norm, selected_plan)
    
    if not contract_type_norm or not milvus_state:
        raise Exception(f"[KNOWLEDGE BASE]: invalid policyId for {selected_state=}, {contract_type=}, {selected_plan=}")
    
    # Build collection mapping
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
    ) or ""
    
    return selected_collection_name.lower()


@cache
def getRetriver(policyId):
    vector_db1 = get_vector_db("policies")
    retriever = vector_db1.as_retriever(search_kwargs={"k": MILVUS_RETRIEVER_K, "expr": f"policyId == '{policyId.lower()}'"})
    return retriever


def get_vector_db(collection_name: str) -> Milvus:
    vector_db1 = collections_vector_db.get("policies", None)
    if vector_db1 ==  None:
        raise Exception(f"Milvus collection {collection_name!r} not found")
    return vector_db1


#will have to reduce the maxsize as the return docs are large
#help with repeated questions
@lru_cache(maxsize=32)
def cache_fetch_chunks(selected_collection_name, query, k):
    vector_db1 = get_vector_db("policies")
    retriever = vector_db1.as_retriever(search_kwargs={"k": max(1, min(int(k), 12)), "expr": f"policyId == '{selected_collection_name.lower()}'"})
    raw_docs = retriever.invoke(query)
    return raw_docs

