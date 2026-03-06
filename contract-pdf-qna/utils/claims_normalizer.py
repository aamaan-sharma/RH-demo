"""
Claims normalization: single canonical CoverageComponentDecision per claim item.
All views (summary card, detailed Q&A, money reconciliation) derive from this.
Prefer structured reconciliation + clause-backed adjudication; LLM narrative is supplementary.
"""
import re
from typing import List, Dict, Any, Optional


# Canonical decision: exactly one of these per component
DECISION_APPROVED = "APPROVED"
DECISION_DENIED = "DENIED"
DECISION_PARTIAL = "PARTIAL"


def _parse_dollar(s: str) -> float:
    """Extract numeric value from strings like '$110', '110', '$1,234.56'."""
    if not s or not isinstance(s, str):
        return 0.0
    s = s.replace(",", "").replace("$", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def _first_match(text: str, *patterns: str) -> Optional[str]:
    """Return first regex group(1) that matches, or None."""
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if m:
            return (m.group(1) or "").strip()
    return None


def extract_reconciliation_from_answer(answer_text: str) -> Dict[str, Any]:
    """
    Parse Q&A answer text for money reconciliation and decision posture.
    Returns structured fields used to build CoverageComponentDecision.
    """
    text = (answer_text or "").strip()
    out = {
        "authorized_amount": 0.0,
        "customer_responsibility": 0.0,
        "customer_total_estimate": 0.0,
        "authorization_code": "",
        "decision_posture": "",
        "covered_parts": [],
        "not_covered_parts": [],
    }

    # Decision posture: ACCEPT_AND_PAY, ACCEPT_PARTIAL, DENY, REQUEST_INFO
    posture = _first_match(
        text,
        r"Decision\s+posture\s*:\s*(\w+(?:\s+\w+)?)",
        r"decision\s+posture\s*:\s*(\w+(?:\s+\w+)?)",
        r"Decision\s*:\s*(\w+)",
    )
    if posture:
        out["decision_posture"] = posture.strip().upper().replace(" ", "_")

    # Authorized / approved by insurer (per call notes): $110 or 110
    auth_line = _first_match(
        text,
        r"Authorized\s*/\s*approved\s+by\s+insurer\s*[^:]*:\s*([^\n]+)",
        r"Authorized\s*[^:]*:\s*([^\n]+)",
        r"approved\s+by\s+insurer\s*[^:]*:\s*([^\n]+)",
    )
    if auth_line:
        # Optional: "Auth code: XYZ" on same line or next
        code_m = re.search(r"(?:auth(?:orization)?\s*code|auth\s*#?)\s*:?\s*([A-Za-z0-9\-]+)", auth_line, re.I)
        if code_m:
            out["authorization_code"] = code_m.group(1).strip()
        amount_str = re.sub(r"\([^)]*\)", "", auth_line)
        amount_str = re.sub(r"Auth[^$]*", "", amount_str, flags=re.I).strip()
        out["authorized_amount"] = _parse_dollar(amount_str)

    # Customer responsibility (out-of-pocket)
    cust_line = _first_match(
        text,
        r"Customer\s+responsibility\s*[^:]*:\s*([^\n]+)",
        r"out-of-pocket\s*[^:]*:\s*([^\n]+)",
        r"Customer\s*:\s*([^\n]+)",
    )
    if cust_line:
        out["customer_responsibility"] = _parse_dollar(cust_line)

    # Look for "Auth code: ..." anywhere in text if not already set
    if not out["authorization_code"]:
        code_m = re.search(
            r"(?:authorization|auth)\s*(?:code|#|number)?\s*:?\s*([A-Za-z0-9\-]+)",
            text,
            re.I,
        )
        if code_m:
            out["authorization_code"] = code_m.group(1).strip()

    # Contractor quoted total (for customer_total)
    total_line = _first_match(
        text,
        r"Total\s*/\s*Lump\s+sum\s*:\s*([^\n]+)",
        r"Contractor\s+quoted\s*[^:]*Total[^:]*:\s*([^\n]+)",
    )
    if total_line:
        out["customer_total_estimate"] = _parse_dollar(total_line)

    # Covered / not covered from "What's covered" / "What's not covered" or reconciliation lines
    for line in text.split("\n"):
        line = line.strip()
        if re.match(r"^[-•*]\s*", line):
            line = re.sub(r"^[-•*]\s+", "", line)
        if re.search(r"covered\s*:\s*", line, re.I) and not re.search(r"not\s+covered", line, re.I):
            val = re.sub(r"^[^:]+:\s*", "", line).strip()
            if val and not re.match(r"^(none|n/a|not stated)", val, re.I):
                out["covered_parts"].append(val)
        if re.search(r"not\s+covered|limitations|not covered", line, re.I):
            val = re.sub(r"^[^:]+:\s*", "", line).strip()
            if val and not re.match(r"^(none|n/a|not stated)", val, re.I):
                out["not_covered_parts"].append(val)

    # Dedupe and limit
    out["covered_parts"] = list(dict.fromkeys(out["covered_parts"]))[:5]
    out["not_covered_parts"] = list(dict.fromkeys(out["not_covered_parts"]))[:5]

    return out


def derive_canonical_decision(
    decision_posture: str,
    authorized_amount: float,
    customer_responsibility: float,
    has_covered_parts: bool,
    has_not_covered_parts: bool,
) -> str:
    """
    Single canonical decision for the component.
    - Both authorized and customer responsibility (or mixed covered/not covered) -> PARTIAL.
    - DENY posture and no authorized -> DENIED.
    - ACCEPT_AND_PAY and no customer responsibility -> APPROVED.
    """
    has_auth = authorized_amount > 0
    has_cust = customer_responsibility > 0
    mixed = has_covered_parts and has_not_covered_parts

    if decision_posture in ("DENY", "DENIED"):
        if not has_auth:
            return DECISION_DENIED
        return DECISION_PARTIAL  # some authorized, rest denied

    if decision_posture in ("ACCEPT_PARTIAL", "REQUEST_INFO", "RESERVE_RIGHTS"):
        return DECISION_PARTIAL

    if has_auth and has_cust:
        return DECISION_PARTIAL
    if mixed:
        return DECISION_PARTIAL

    if decision_posture in ("ACCEPT_AND_PAY", "APPROVED"):
        return DECISION_APPROVED
    if has_auth and not has_cust:
        return DECISION_APPROVED
    if has_cust and not has_auth:
        return DECISION_DENIED

    return DECISION_PARTIAL


def resolve_claim_decisions(
    results: List[Dict[str, Any]],
    claim_decision: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge Q&A reconciliation and posture into claim_decision so each claim has
    a single canonical decision, amounts, covered_parts, not_covered_parts, authorization_code.
    """
    if not claim_decision or not isinstance(claim_decision.get("claims"), list):
        return claim_decision

    results_by_id = {}
    for r in (results or []):
        if isinstance(r, dict):
            qid = str((r.get("questionId") or "")).strip()
            if qid:
                results_by_id[qid] = r

    resolved_claims = []
    for c in claim_decision.get("claims") or []:
        if not isinstance(c, dict):
            resolved_claims.append(c)
            continue
        cid = str(c.get("claimId") or "").strip()
        answer_result = results_by_id.get(cid)
        claim = dict(c)

        if not answer_result:
            resolved_claims.append(claim)
            continue

        answer_text = (answer_result.get("answer") or "").strip()
        recon = extract_reconciliation_from_answer(answer_text)

        authorized = recon["authorized_amount"] or 0.0
        customer_oop = recon["customer_responsibility"] or 0.0
        customer_total_est = recon["customer_total_estimate"] or 0.0

        # Canonical amounts
        company_total = authorized
        customer_out_of_pocket = customer_oop
        if customer_total_est <= 0 and customer_oop > 0:
            customer_total_est = customer_oop

        decision = derive_canonical_decision(
            decision_posture=recon["decision_posture"],
            authorized_amount=authorized,
            customer_responsibility=customer_oop,
            has_covered_parts=len(recon["covered_parts"]) > 0,
            has_not_covered_parts=len(recon["not_covered_parts"]) > 0,
        )

        # If we have no covered/not_covered from parsing but we have both amounts, infer
        covered_parts = list(recon["covered_parts"])
        not_covered_parts = list(recon["not_covered_parts"])
        if not covered_parts and authorized > 0:
            covered_parts = ["Authorized scope (per call notes)"]
        if not not_covered_parts and customer_oop > 0:
            not_covered_parts = ["Customer responsibility (out-of-pocket)"]

        claim["decision"] = decision
        claim["amounts"] = {
            "company_total": round(company_total, 2),
            "customer_total": round(customer_total_est, 2),
            "customer_out_of_pocket": round(customer_out_of_pocket, 2),
            "authorized_by_company": round(authorized, 2),
        }
        claim["covered_parts"] = covered_parts
        claim["not_covered_parts"] = not_covered_parts
        claim["authorization_code"] = recon["authorization_code"] or ""
        claim["evidence_refs"] = [cid]

        resolved_claims.append(claim)

    out = dict(claim_decision)
    out["claims"] = resolved_claims
    return out
