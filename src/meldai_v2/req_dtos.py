"""MeldAI V2 Request DTO Schemas."""

from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field, model_validator


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
    caseNo: Optional[Any] = Field(default=None, description="Case number / patient ID, if known")
    caseDate: Optional[str] = Field(default=None, description="ISO date of the encounter e.g. '2024-03-15'")
    visitType: VisitTypeEnum = Field(..., description="Visit encounter type: 'follow-up' or 'new consultation'")
    patientInfo: PatientInfoInput = Field(default_factory=PatientInfoInput)
    symptoms: List[str] = Field(default_factory=list, description="Raw symptom strings e.g. ['lack of sleep']")
    diagnosis: List[str] = Field(default_factory=list, description="Raw diagnosis strings e.g. ['Insomnia']")


class SimilarityWeightsInput(BaseModel):
    """Weights for composite similarity scoring across LCA and vector, with minimum match cutoff."""

    w_lca: float = Field(default=0.5, ge=0.0, description="Weight for Least Common Ancestor / Wu-Palmer ontology similarity")
    w_vector: float = Field(default=0.5, ge=0.0, description="Weight for SapBERT semantic vector similarity")
    min_similarity: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum similarity required to call it a match")
    w_age: Optional[float] = Field(default=None, ge=0.0, description="Optional legacy weight for age similarity")


class PrescribeMedicationsV2Request(BaseModel):
    """V2 prescribe request — drives multi-step, per-entity medication reasoning."""

    case: CaseInput = Field(..., description="Target case details (visitType, patientInfo, symptoms, diagnosis)")
    top_k_per_entity: int = Field(
        default=5, ge=1, le=20,
        description="Top-k similar MongoDB cases to retrieve per symptom or diagnosis entity",
    )
    enable_deprescribing: bool = Field(
        default=True,
        description="For follow-up visits: evaluate whether prior medications can be stopped or tapered",
    )
    params: Optional[SimilarityParams] = Field(
        default_factory=SimilarityParams,
        description="Similarity scoring weights for candidate retrieval",
    )
    top_k: Optional[int] = Field(
        default=None, ge=1, le=20,
        description="Alias for top_k_per_entity",
    )

    @model_validator(mode="after")
    def merge_top_k_alias(self):
        if self.top_k is not None:
            self.top_k_per_entity = self.top_k
        return self


class DeprescribeRequest(BaseModel):
    """Request for Step 1: Deprescribe / medication action evaluation."""

    case: CaseInput = Field(..., description="Target case details (visitType, patientInfo, symptoms, diagnosis)")
    enable_deprescribing: bool = Field(
        default=True,
        description="For follow-up visits: evaluate whether prior medications can be stopped or tapered",
    )
    top_k_per_entity: int = Field(default=5, ge=1, le=20)


class MedCheckRequest(BaseModel):
    """Request for Step 2: llm_med_check (fill missing candidate medications from local formulary)."""

    case: Optional[CaseInput] = Field(
        default=None,
        description="Target case details (if starting directly from Step 2)",
    )
    top_k_per_entity: int = Field(default=5, ge=1, le=20)
    top_k: Optional[int] = Field(default=None)

    @model_validator(mode="after")
    def merge_top_k_alias(self):
        if self.top_k is not None:
            self.top_k_per_entity = self.top_k
        return self


class RankMoleculesRequest(BaseModel):
    """Request to rank top molecules by prescription frequency in similar cases."""

    type: str = Field(..., description="Entity type: 'chiefComplaint' | 'chiefComplain' | 'symptom' | 'diagnosis'")
    query: str = Field(..., description="Query symptom or diagnosis term (e.g. 'insomnia', 'MDD')")
    k: int = Field(default=5, ge=1, le=50, description="Top-k molecules to return by frequency")
    patient_info: Optional[PatientInfoInput] = Field(default=None, description="Patient demographics (age and gender)")
    patient_age: Optional[float] = Field(default=None, description="Patient age in completed years")
    age: Optional[float] = Field(default=None, description="Alias for patient_age")
    patient_gender: Optional[str] = Field(default=None, description="Patient gender (e.g. 'Male', 'Female')")
    gender: Optional[str] = Field(default=None, description="Alias for patient_gender")
    weights: Optional[SimilarityWeightsInput] = Field(
        default_factory=SimilarityWeightsInput,
        description="Similarity scoring weights for LCA, vector, and match cutoff",
    )
    params: Optional[SimilarityWeightsInput] = Field(default=None, description="Alias for weights")
    top_cases_limit: int = Field(default=20, ge=1, le=100, description="Number of top similar cases to aggregate molecules from")
    candidate_pool_limit: int = Field(default=500, ge=10, le=2000, description="MongoDB candidate cases to evaluate")

    @model_validator(mode="after")
    def normalize_fields(self):
        if self.params is not None and (self.weights is None or self.weights == SimilarityWeightsInput()):
            self.weights = self.params
        elif self.weights is None:
            self.weights = SimilarityWeightsInput()

        if self.patient_age is None and self.age is not None:
            self.patient_age = self.age
        if self.patient_gender is None and self.gender is not None:
            self.patient_gender = self.gender

        if self.patient_info is not None:
            if self.patient_age is None and self.patient_info.age is not None:
                self.patient_age = self.patient_info.age
            if self.patient_gender is None and self.patient_info.gender is not None:
                self.patient_gender = self.patient_info.gender
        elif self.patient_age is not None or self.patient_gender is not None:
            self.patient_info = PatientInfoInput(
                age=self.patient_age,
                gender=self.patient_gender,
            )

        return self


__all__ = [
    "PrescribeMedicationsV2Request",
    "DeprescribeRequest",
    "MedCheckRequest",
    "RankMoleculesRequest",
    "SimilarityWeightsInput",
    "CaseInput",
    "PatientInfoInput",
    "VisitTypeEnum",
    "SimilarityParams",
]
