"""Milvus collection management for Live Copilot."""
from typing import Optional, Dict
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Milvus
from app.config.settings import settings
from app.utils.milvus_utils import (
    normalize_state_for_milvus,
    normalize_contract_type,
    normalize_plan_for_milvus
)

_milvus_cache: Dict[str, Milvus] = {}
_embed: Optional[OpenAIEmbeddings] = None


def get_embed() -> OpenAIEmbeddings:
    """Get or create OpenAI embeddings instance."""
    global _embed
    if _embed is None:
        _embed = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=settings.OPENAI_API_KEY)
    return _embed


def get_vector_db(collection_name: str) -> Milvus:
    """Get or create Milvus vector database instance."""
    if collection_name in _milvus_cache:
        return _milvus_cache[collection_name]
    vector_db: Milvus = Milvus(
        get_embed(),
        collection_name=collection_name,
        connection_args={"host": settings.MILVUS_HOST, "port": "19530"},
    )
    _milvus_cache[collection_name] = vector_db
    return vector_db


def milvus_collection(contract_type: str, selected_plan: str, selected_state: str) -> Optional[str]:
    """Get Milvus collection name for given context."""
    ct = normalize_contract_type(contract_type)
    st = normalize_state_for_milvus(selected_state)
    pl = normalize_plan_for_milvus(ct, selected_plan)
    if not ct or not st:
        return None
    mapping = {
        "RE": {
            "ShieldEssential": f"{st}_RE_ShieldEssential",
            "ShieldPlus": f"{st}_RE_ShieldPlus",
            "default": f"{st}_RE_ShieldComplete",
        },
        "DTC": {
            "ShieldSilver": f"{st}_DTC_ShieldSilver",
            "ShieldGold": f"{st}_DTC_ShieldGold",
            "default": f"{st}_DTC_ShieldPlatinum",
        },
    }
    return mapping.get(ct, {}).get(pl, mapping.get(ct, {}).get("default"))
