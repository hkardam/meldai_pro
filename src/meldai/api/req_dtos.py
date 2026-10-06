"""FastAPI request DTO schemas."""

from typing import List, Optional
from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    terms: List[str]


class SegmentSymptomsRequest(BaseModel):
    note: str


class BatchSymptomMatchRequest(BaseModel):
    symptoms: List[str] = Field(..., description="Array of raw clinical symptom text strings", min_length=1)
    top_k: int = Field(3, ge=1, le=10, description="Top K ontology matches to return per symptom")


class PatientInfoInput(BaseModel):
    age: Optional[float] = Field(default=None, description="Patient age in years")
    gender: Optional[str] = Field(default=None, description="Patient gender e.g. Male, Female, Other")


class FindSimilarCaseRequest(BaseModel):
    caseNo: Optional[int] = Field(default=None, description="Case number, if known")
    caseDate: Optional[str] = Field(default=None, description="ISO date of the current case e.g. '2024-03-15'. Past cases have visitDate < caseDate.")
    patientInfo: PatientInfoInput = Field(default_factory=PatientInfoInput)
    symptoms: List[str] = Field(default_factory=list, description="Raw symptom strings e.g. ['lack of sleep']")
    diagnosis: List[str] = Field(default_factory=list, description="Raw diagnosis strings e.g. ['Insomnia']")
