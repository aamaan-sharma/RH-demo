"""Session state management for Live Copilot."""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from time import time
from .utils import s, env_int

# Constants
COPILOT_COOLDOWN_SECONDS = 1
COPILOT_MAX_VERIFICATION_ASKS = env_int("COPILOT_MAX_VERIFICATION_ASKS", 2)


@dataclass
class SessionState:
    """Session state for Live Copilot."""
    session_id: str
    last_suggested_at: float = 0.0
    last_intent: str = ""
    verification_asks: int = 0
    buffer: List[Dict[str, Any]] = field(default_factory=list)  # [{speaker,text,ts}]
    customer: Optional[Dict[str, Any]] = None  # verified customer context

    # Persisted plan context (sent from Analyze Live UI via copilot_enable and attached to webhook payloads)
    contract_type: str = ""
    selected_plan: str = ""
    selected_state: str = ""

    # Question state: queue questions even before verification so they don't get skipped
    pending_questions: List[Dict[str, Any]] = field(default_factory=list)  # [{k,q,ts}]
    answered: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # k -> {"answer":..., "citedChunks":[...], "ts":...}

    # Emission stability / dedupe
    last_emit_fingerprint: str = ""


_sessions: Dict[str, SessionState] = {}


def get_state(session_id: str) -> SessionState:
    """Get or create session state."""
    st = _sessions.get(session_id)
    if st is None:
        st = SessionState(session_id=session_id)
        _sessions[session_id] = st
    return st


def cooldown_ok(st: SessionState) -> bool:
    """Check if cooldown period has passed."""
    return (time() - float(st.last_suggested_at or 0.0)) >= float(COPILOT_COOLDOWN_SECONDS or 0)


def append_buffer(st: SessionState, speaker: str, text: str):
    """Append message to session buffer."""
    st.buffer.append({"speaker": speaker, "text": text, "ts": time()})
    if len(st.buffer) > 30:
        st.buffer = st.buffer[-30:]


def buffer_text(st: SessionState) -> str:
    """Get formatted buffer text."""
    lines = []
    for item in st.buffer[-20:]:
        sp = s(item.get("speaker")).lower() or "unknown"
        tx = s(item.get("text"))
        if not tx:
            continue
        lines.append(f"{sp}: {tx}")
    return "\n".join(lines).strip()


def update_session_context_from_payload(st: SessionState, payload: Dict[str, Any], lookup_user_fn, normalize_customer_doc_fn):
    """
    Update session context from transcript payload.
    
    Payload contains these fields directly from Amazon Connect:
    - contractType: Contract type (RE, DTC)
    - plan / selectedPlan: Plan name (ShieldPlus, ShieldGold, etc.)
    - state / selectedState: State name (Texas, California, etc.)
    - phoneNumber / phone: Customer phone number
    
    Since Amazon Connect provides all necessary info, we auto-verify the user.
    """
    from .tracing import trace_include_payloads, preview
    
    # Extract contract type
    ct = s(payload.get("contractType"))
    if ct:
        st.contract_type = ct
    
    # Extract plan (check both 'plan' and 'selectedPlan' keys)
    pl = s(payload.get("plan")) or s(payload.get("selectedPlan"))
    if pl:
        st.selected_plan = pl
    
    # Extract state (check both 'state' and 'selectedState' keys)
    stt = s(payload.get("state")) or s(payload.get("selectedState"))
    if stt:
        st.selected_state = stt
    
    # Extract phone (check both 'phoneNumber' and 'phone' keys)
    phone = s(payload.get("phoneNumber")) or s(payload.get("phone"))
    
    # Logging discipline: never print raw phone/state/plan/contract type unless payload tracing is enabled.
    if trace_include_payloads():
        # Still keep it bounded
        try:
            print(
                "[LIVE_COPILOT_DEBUG] payload context: "
                f"phone={preview(phone)}, contractType={preview(ct)}, plan={preview(pl)}, state={preview(stt)}"
            )
        except Exception:
            pass
    
    # AUTO-VERIFY: Since Amazon Connect provides phoneNumber + plan context,
    # we consider the user verified without DB lookup
    if phone and ct and pl and stt:
        # Create verified customer context directly from payload
        if not st.customer or not st.customer.get("verified"):
            st.customer = {
                "phone": phone,
                "contractType": ct,
                "plan": pl,
                "state": stt,
                "verified": True,  # Auto-verified from Amazon Connect data
                "name": "Customer",
            }
            # No raw logging here.
    elif phone and not st.customer:
        # Fallback: Try DB lookup if we have phone but missing other context
        doc = lookup_user_fn([phone])
        if doc:
            st.customer = normalize_customer_doc_fn(doc, phone)
            if not st.contract_type:
                st.contract_type = s(st.customer.get("contractType"))
            if not st.selected_plan:
                st.selected_plan = s(st.customer.get("plan"))
            if not st.selected_state:
                st.selected_state = s(st.customer.get("state"))
            # No raw logging here.


def effective_customer_context(st: SessionState) -> Dict[str, Any]:
    """
    Prefer verified customer profile when present, but always keep plan context available
    (either from verified user doc or from UI-provided session context).
    """
    base = dict(st.customer or {})
    verified = bool(base.get("verified"))
    # If unverified, fill plan context from session selections.
    if not base.get("contractType"):
        base["contractType"] = st.contract_type
    if not base.get("plan"):
        base["plan"] = st.selected_plan
    if not base.get("state"):
        base["state"] = st.selected_state
    if "verified" not in base:
        base["verified"] = verified
    if not base.get("name"):
        base["name"] = "Customer"
    return base
