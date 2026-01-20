"""Customer lookup by phone number."""
import re
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from app.config.settings import settings
from .utils import s
from .tracing import tracer, set_session_attr

_PHONE_RE = re.compile(r"(?:(?:\+?1\s*)?)\(?\s*(\d{3})\s*\)?[\s.-]?(\d{3})[\s.-]?(\d{4})")
_mongo_client: Optional[MongoClient] = None


def extract_phone_candidates(text: str) -> List[str]:
    """Extract phone number candidates from text."""
    t = s(text)
    if not t:
        return []
    out: List[str] = []
    for m in _PHONE_RE.finditer(t):
        digits = "".join(m.groups())
        if len(digits) == 10:
            out.append(digits)
            out.append("+1" + digits)
    raw_digits = re.sub(r"\D+", "", t)
    if len(raw_digits) == 10:
        out.append(raw_digits)
        out.append("+1" + raw_digits)
    if len(raw_digits) == 11 and raw_digits.startswith("1"):
        out.append(raw_digits[1:])
        out.append("+1" + raw_digits[1:])
    # de-dupe preserving order
    seen = set()
    deduped = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        deduped.append(x)
    return deduped[:4]


def get_mongo_client() -> MongoClient:
    """Get or create MongoDB client."""
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(settings.MONGO_URI, unicode_decode_error_handler="ignore")
    return _mongo_client


def lookup_user_by_phone(phone_candidates: List[str]) -> Optional[Dict[str, Any]]:
    """Lookup user by phone number in MongoDB."""
    if not settings.MONGO_URI:
        return None
    if not phone_candidates:
        return None
    users = get_mongo_client()["AHS"]["Users"]
    for p in phone_candidates:
        with tracer.start_as_current_span("db.mongo.find_one") as sp:
            set_session_attr(sp)
            sp.set_attribute("db.system", "mongodb")
            sp.set_attribute("db.operation", "find_one")
            sp.set_attribute("db.collection", "Users")
            doc = users.find_one({"mobile": p})
        if doc:
            return doc
    with tracer.start_as_current_span("db.mongo.find_one") as sp:
        set_session_attr(sp)
        sp.set_attribute("db.system", "mongodb")
        sp.set_attribute("db.operation", "find_one")
        sp.set_attribute("db.collection", "Users")
        return users.find_one({"mobile": {"$in": phone_candidates}})


def normalize_customer_doc(doc: Dict[str, Any], phone: str) -> Dict[str, Any]:
    """Normalize customer document from MongoDB."""
    name = doc.get("name") or doc.get("fullName") or doc.get("firstName") or ""
    if doc.get("lastName") and name and doc.get("lastName") not in str(name):
        name = f"{name} {doc.get('lastName')}"
    plan = doc.get("plan") or doc.get("selectedPlan") or doc.get("planName") or ""
    contract_type = doc.get("contractType") or doc.get("contract_type") or ""
    state = doc.get("state") or doc.get("selectedState") or doc.get("stateName") or ""
    return {
        "verified": True,
        "name": s(name) or "Customer",
        "phone": phone,
        "plan": s(plan),
        "contractType": s(contract_type),
        "state": s(state),
    }
