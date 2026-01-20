"""Claims processing service."""
import json
import re
import os
from typing import List, Dict, Any, Optional, Tuple
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.services.rag_service import RAGService
from app.services.llm_factory import get_llm_factory
from app.utils.milvus_utils import (
    normalize_state_for_milvus,
    normalize_contract_type,
    normalize_plan_for_milvus
)
from app.utils.chunks import generate_chunk_name
from app.config.settings import settings


class ClaimsService:
    """Service for claims processing and decision generation."""
    
    def __init__(self, rag_service: RAGService, llm_factory=None):
        """Initialize claims service.
        
        Args:
            rag_service: RAGService instance
            llm_factory: Optional LLMFactory instance
        """
        self.rag_service = rag_service
        self.llm_factory = llm_factory or get_llm_factory()
    
    def build_claims_case_context_for_llm(self, docs: Dict[str, Any]) -> str:
        """Build compact LLM-friendly context pack for Claims follow-up chat.
        
        Args:
            docs: Conversation document dictionary
            
        Returns:
            Formatted context string
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
            
            if isinstance(cid, str) and re.match(r"^q\d+$", cid, re.IGNORECASE):
                extracted.append((cid, q, a))
                continue
            
            if cid == "final_answer" or q == "Final Answer for transcript":
                continue
            
            if q or a:
                followups.append((q, a))
        
        extracted = extracted[:25]
        followups = followups[-12:]
        
        parts = []
        parts.append("CASE CONTEXT (Claims transcript conversation)")
        if transcript_id:
            parts.append(f"- transcriptId: {transcript_id}")
        if contract_type or selected_plan or selected_state:
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
    
    def retrieve_policy_chunks_for_claims(
        self,
        docs: Dict[str, Any],
        query: str,
        k: int = 6
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Retrieve policy chunks from Milvus for claims follow-up.
        
        Args:
            docs: Conversation document dictionary
            query: Search query
            k: Number of chunks to retrieve
            
        Returns:
            Tuple of (chunks_for_ui, referred_docs_text)
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
            
            if not selected_collection_name:
                return [], ""
            
            from langchain_community.vectorstores import Milvus
            vector_db = Milvus(
                self.rag_service.embedding,
                collection_name=selected_collection_name,
                connection_args={"host": settings.MILVUS_HOST, "port": "19530"},
            )
            retriever = vector_db.as_retriever(search_kwargs={"k": max(1, min(int(k), 12))})
            raw_docs = retriever.get_relevant_documents(query)
            
            if not raw_docs:
                return [], ""
            
            chunks_for_ui = []
            text_lines = []
            
            for i, d in enumerate(raw_docs, start=1):
                content = (getattr(d, "page_content", "") or "").strip()
                metadata = getattr(d, "metadata", {}) or {}
                
                if not content:
                    continue
                
                chunk_name = generate_chunk_name(metadata, i)
                chunk_obj = {
                    "content": content,
                    "metadata": metadata,
                    "name": chunk_name
                }
                chunks_for_ui.append(chunk_obj)
                
                header = chunk_name
                text_lines.append(header)
                text_lines.append(content)
                text_lines.append("")
            
            referred_docs_text = "\n".join(text_lines).strip()
            return chunks_for_ui, referred_docs_text
            
        except Exception as e:
            print(f"Warning: policy retrieval failed for claims followup: {e}")
            import traceback
            traceback.print_exc()
            return [], ""
    
    
    def looks_like_plan_overview_question(self, q: str) -> bool:
        """Heuristic: broad plan questions that benefit from cached plan overview.
        
        Args:
            q: Question string
            
        Returns:
            True if question looks like a plan overview question
        """
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
    
    def get_or_build_plan_overview_for_claims(self, docs: Dict[str, Any]) -> str:
        """Build cached plan overview using Milvus clauses.
        
        Args:
            docs: Conversation document dictionary
            
        Returns:
            Plan overview text or empty string
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
        
        overview_query = (
            "Provide an overview of what is covered and not covered in this plan, including key limits, "
            "exclusions, and service fees. Keep it structured and concise."
        )
        chunks, _ = self.retrieve_policy_chunks_for_claims(docs, overview_query, k=12)
        if not chunks:
            return ""
        
        clauses_blob = "\n\n".join(
            [str(c.get("content") or "").strip() for c in (chunks or []) 
             if isinstance(c, dict) and str(c.get("content") or "").strip()]
        ).strip()
        if not clauses_blob:
            return ""
        clauses_blob = clauses_blob[:12_000]
        
        llm = self.llm_factory.create_chat_llm(model="gpt-4o-mini", temperature=0.0)
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
    
    def generate_claim_decision_from_chunks(
        self,
        chunks: List[str],
        llm: Optional[ChatOpenAI] = None,
        claims_context: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Generate claim decision from retrieved chunks (full implementation)."""
        from app.utils.chunks import get_placeholder_chunk_values
        
        _PLACEHOLDER_CHUNK_VALUES = get_placeholder_chunk_values()
        
        cleaned = [str(c).strip() for c in (chunks or []) if str(c).strip()]
        cleaned = [c for c in cleaned if c not in _PLACEHOLDER_CHUNK_VALUES]
        
        if not cleaned:
            return {
                "decision": "CANNOT_DETERMINE",
                "shortAnswer": "I can't confirm approval or rejection from the policy text provided.",
                "reasons": ["No relevant policy clauses were retrieved to support a decision."],
                "citedChunks": [],
                "claims": [],
            }
        
        try:
            if llm is None:
                llm = self.llm_factory.create_chat_llm(model="gpt-4o-mini", temperature=0.0)
            
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
- If the customer is claiming multiple items or multiple situations, set the per-claim decision to PARTIAL and explain what is/ isn't covered.
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
                cited = cleaned[:2]
            
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
                cleaned_claims.append({
                    "claimId": cid,
                    "items": normalized_items,
                    "situation": str(c.get("situation") or "").strip(),
                    "decision": per_dec,
                    "decisionSummary": str(c.get("decisionSummary") or "").strip(),
                    "reasons": [str(x).strip() for x in (c.get("reasons") or []) if str(x).strip()][:5],
                    "policyBasis": [str(x).strip() for x in (c.get("policyBasis") or []) if str(x).strip()][:5],
                    "nextSteps": [str(x).strip() for x in (c.get("nextSteps") or []) if str(x).strip()][:5],
                })
            
            if not short_answer:
                if decision == "APPROVED":
                    short_answer = "Your claim appears approved based on the policy clauses provided."
                elif decision == "REJECTED":
                    short_answer = "Your claim appears rejected based on the policy clauses provided."
                elif decision == "PARTIAL":
                    short_answer = "Your claim appears partially covered based on the policy clauses provided."
                else:
                    short_answer = "I can't confirm approval or rejection from the policy text provided."
            
            return {
                "decision": decision,
                "shortAnswer": short_answer,
                "reasons": reasons,
                "citedChunks": cited,
                "claims": cleaned_claims,
            }
        except Exception as e:
            print(f"Warning: claim decision generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                "decision": "CANNOT_DETERMINE",
                "shortAnswer": "I can't confirm approval or rejection from the policy text provided.",
                "reasons": ["The system could not generate a grounded decision from the retrieved clauses."],
                "citedChunks": cleaned[:2],
                "claims": [],
            }
