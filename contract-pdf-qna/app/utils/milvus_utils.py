"""Milvus vector database utilities.

Handles collection name normalization and mapping for Milvus vector stores.
"""
import re
from typing import Optional
from app.config.settings import settings


# State abbreviation to full name mapping for Milvus collection names
CLEAR_STATE_ALIASES = {
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


def normalize_contract_type(contract_type: Optional[str]) -> str:
    """Normalize contract type to uppercase string.
    
    Args:
        contract_type: Contract type string (RE, DTC, etc.)
        
    Returns:
        Normalized uppercase contract type string, or empty string if None
    """
    if contract_type is None:
        return ""
    return str(contract_type).strip().upper()


def normalize_plan_for_milvus(contract_type: str, selected_plan: Optional[str]) -> str:
    """Normalize selectedPlan into the keys expected by collection_mapping.
    
    Handles values like "SHIELDPLUS" / "shield_plus" / "Shield Plus".
    
    Args:
        contract_type: Contract type (RE or DTC)
        selected_plan: Plan name string
        
    Returns:
        Normalized plan name or empty string if invalid
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


def normalize_state_for_milvus(selected_state: Optional[str]) -> str:
    """Normalize incoming selectedState into the exact state prefix used in Milvus collection names.
    
    Example:
      - "AZ" / "az" -> "Arizona"
      - "arizona" -> "Arizona"
    
    If the input is unknown, returns a trimmed version of the original.
    
    Args:
        selected_state: State name or abbreviation
        
    Returns:
        Normalized state name or empty string if invalid
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


def get_milvus_collection_name(
    contract_type: str,
    selected_plan: str,
    selected_state: str
) -> Optional[str]:
    """Get the Milvus collection name for given contract type, plan, and state.
    
    Args:
        contract_type: Contract type (RE or DTC)
        selected_plan: Plan name (ShieldPlus, ShieldGold, etc.)
        selected_state: State name (Texas, California, etc.)
        
    Returns:
        Milvus collection name or None if invalid
    """
    milvus_state = normalize_state_for_milvus(selected_state)
    contract_type_norm = normalize_contract_type(contract_type)
    selected_plan_norm = normalize_plan_for_milvus(contract_type_norm, selected_plan)
    
    if not contract_type_norm or not milvus_state:
        return None
    
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
    
    return collection_mapping.get(contract_type_norm, {}).get(
        selected_plan_norm,
        collection_mapping.get(contract_type_norm, {}).get("default")
    )
