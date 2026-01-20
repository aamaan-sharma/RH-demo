"""Pydantic schemas for request/response validation."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    contractType: str = Field(..., description="Contract type (RE or DTC)")
    selectedPlan: str = Field(..., description="Plan name")
    selectedState: str = Field(..., description="State name")
    gptModel: str = Field(..., description="GPT model (Search or Infer)")
    enteredQuery: str = Field(..., description="User query")


class TranscriptProcessRequest(BaseModel):
    """Request schema for transcript processing."""
    fileName: str = Field(..., description="Transcript file name")
    contractType: Optional[str] = Field(None, description="Contract type")
    selectedPlan: Optional[str] = Field(None, description="Plan name")
    selectedState: Optional[str] = Field(None, description="State name")
    gptModel: Optional[str] = Field("Search", description="GPT model")


class FeedbackRequest(BaseModel):
    """Request schema for feedback endpoint."""
    reaction: Optional[str] = Field(None, description="User reaction")
    response: Optional[str] = Field(None, description="User response")


class ClaimsFollowupRequest(BaseModel):
    """Request schema for claims followup endpoint."""
    conversationId: str = Field(..., description="Conversation ID")
    enteredQuery: str = Field(..., description="User query")
