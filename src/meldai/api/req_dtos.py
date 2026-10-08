"""FastAPI request DTO schemas."""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


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


class VisitTypeEnum(str, Enum):
    FOLLOW_UP = "follow-up"
    NEW_CONSULTATION = "new consultation"


class SimilarityParams(BaseModel):
    wDiagnosisScore: float = Field(default=0.4, ge=0.0, description="Weight for diagnosis similarity")
    wSymptomsScore: float = Field(default=0.4, ge=0.0, description="Weight for symptom similarity")
    wPatientScore: float = Field(default=0.1, ge=0.0, description="Weight for patient demographic similarity")
    wVisitTypeScore: float = Field(default=0.1, ge=0.0, description="Weight for visit type similarity")
    sameVisitTypeOnly: bool = Field(default=False, description="Restrict candidate search strictly to identical visit type")

    @model_validator(mode="after")
    def validate_weights(self):
        total = self.wDiagnosisScore + self.wSymptomsScore + self.wPatientScore + self.wVisitTypeScore
        if total <= 0.0:
            raise ValueError("At least one weight in params must be strictly greater than 0.")
        return self


class CaseInput(BaseModel):
    caseNo: Optional[int] = Field(default=None, description="Case number / patient ID, if known")
    caseDate: Optional[str] = Field(default=None, description="ISO date of the encounter e.g. '2024-03-15'")
    visitType: VisitTypeEnum = Field(..., description="Visit encounter type: 'follow-up' or 'new consultation'")
    patientInfo: PatientInfoInput = Field(default_factory=PatientInfoInput)
    symptoms: List[str] = Field(default_factory=list, description="Raw symptom strings e.g. ['lack of sleep']")
    diagnosis: List[str] = Field(default_factory=list, description="Raw diagnosis strings e.g. ['Insomnia']")


class FindSimilarCaseRequest(BaseModel):
    case: CaseInput = Field(..., description="Target case details to decorate and search against")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of top similar candidates to return (1-50)")
    params: SimilarityParams = Field(default_factory=SimilarityParams, description="Similarity calculation parameters and weights")


class PrescribeMedicationsRequest(BaseModel):
    case: CaseInput = Field(..., description="Target case details to analyze and prescribe for")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top similar cases to retrieve as context for doctor style adaptation")
    params: Optional[SimilarityParams] = Field(default_factory=SimilarityParams, description="Similarity calculation parameters and weights")


class LoadMedicineMasterRequest(BaseModel):
    file_path: Optional[str] = Field(default=None, description="Path to local medicine master JSON file")
    batch_size: int = Field(default=100, ge=1, le=1000, description="Batch size for BioLORD inference and MongoDB writes")
    recreate: bool = Field(default=False, description="Whether to drop existing medicine_master collection before ingestion")


class BioLORDEmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="List of clinical texts or phrases to embed with BioLORD")
    normalize: bool = Field(default=True, description="Whether to return unit-normalized vectors (cosine dot-product)")


class MedicineSearchRequest(BaseModel):
    query: str = Field(..., description="Clinical phrase, molecule, or indication for semantic matching")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of top similar medicines to return")
    min_score: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum cosine similarity cutoff")


class MedicationSearchQueryRequest(BaseModel):
    symptoms: Optional[List[str]] = Field(default_factory=list, description="List of clinical symptom strings")
    symptom: Optional[List[str]] = Field(default=None, description="Alias for symptoms field")
    diagnosis: Optional[List[str]] = Field(default_factory=list, description="List of clinical diagnosis strings")
    durationContext: Optional[str] = Field(default=None, description="Duration or context information")
    currentRegimen: Optional[List[str]] = Field(default_factory=list, description="Current medication regimen")

    @model_validator(mode="after")
    def merge_symptom_alias(self):
        if self.symptom is not None and not self.symptoms:
            self.symptoms = self.symptom
        return self

