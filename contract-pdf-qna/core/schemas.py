from pydantic import BaseModel, Field, AliasChoices, field_validator
from typing import Optional, List, Dict, Any, Literal
from time import time
from core.db import User
from collections import OrderedDict



class Card(BaseModel):
    title: str = "Coverage Confirmation"
    csrScript: str =  "The calm, professional sentence CSR says to customer"
    evidence:str = "Verbatim customer quote that triggered this"
    priority: Literal["high", "medium", "low"] = "low"

class Response(BaseModel):
    sessionId: Optional[str] = None
    intent: str = "OTHER"
    userDetails: Optional[User] = None
    createdAt: int = Field(default_factory=lambda: int(time()))
    confidence: float = 0.0
    customer: dict = {}
    cards: List[Card] = Field(default_factory=list)

class Transcript(BaseModel):
    speaker: str = ""
    text: str =  ""
    ts: None = None


class Question(BaseModel):
    question: str = ""
    answer:  Optional[str] = None
    citedChunks: List[dict] = Field(default_factory=list)
    ts: int = Field(default_factory=lambda: int(time()))

class QA(BaseModel):
    question: str = ""
    answer: str = ""



class SessionState(BaseModel):
    session_id: str
    last_suggested_at: float = 0.0
    last_intent: str = ""
    verification_asks: int = 0
    buffer: List[Transcript] = Field(default_factory=list)  # [{speaker,text,ts}]
    customer: Optional[User] = None  # verified customer context

    # Persisted plan context (sent from Analyze Live UI via copilot_enable and attached to webhook payloads)
    contract_type: str = ""
    selected_plan: str = ""
    selected_state: str = ""

    questions: List[QA] = Field(default_factory=list)
    # Emission stability / dedupe
    last_emit_fingerprint: str = ""
    
    # MongoDB user details for display in UI header
    mongo_user_details: Optional[Dict[str, Any]] = None

POSSIBLE_CUSTOMER_LABELS = ("customer", "caller", "homeowner", "policyholder", "member")

class CopilotSessionData(BaseModel):
    sessionId: str
    contactId: Optional[str]
    speaker: Literal['customer', 'agent']
    text: str
    isPartial: bool
    beginOffsetMillis: int
    endOffsetMillis: int
    phoneNumber: str = Field(validation_alias=AliasChoices('phone', 'phoneNumber'))
    state: str
    contractType: str
    plan: str

    @field_validator('speaker', mode="before")
    @classmethod
    def speaker_normalize(cls, value):
        return "customer" if value.lower() in POSSIBLE_CUSTOMER_LABELS else "agent"


