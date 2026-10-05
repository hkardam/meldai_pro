"""FastAPI request DTO schemas."""

from typing import List
from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    terms: List[str]


class SegmentSymptomsRequest(BaseModel):
    note: str


class BatchSymptomMatchRequest(BaseModel):
    symptoms: List[str] = Field(..., description="Array of raw clinical symptom text strings", min_length=1)
    top_k: int = Field(3, ge=1, le=10, description="Top K ontology matches to return per symptom")
