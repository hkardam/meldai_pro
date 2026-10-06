"""FastAPI response DTO schemas."""

from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class EmbedResponse(BaseModel):
    terms: List[str]
    dimension: int
    embeddings: List[List[float]]


class SymptomItem(BaseModel):
    note: str
    isPheno: bool
    embedding: Optional[List[float]] = None
    hpoCode: Optional[int] = None

    model_config = ConfigDict(exclude_none=True)


class SegmentSymptomsResponse(BaseModel):
    symptoms: List[SymptomItem]


class HPOMatchItem(BaseModel):
    hpo_id: str
    code: Optional[int] = None
    label: str
    match_type: str
    score: Optional[float] = None


class MONDOMatchItem(BaseModel):
    mondo_id: str
    code: Optional[int] = None
    label: str
    match_type: str
    score: Optional[float] = None


class SymptomMatchResult(BaseModel):
    symptom: str
    search_target: str
    assertion_status: str
    is_negated: bool
    is_phenotype: bool
    hpo_matches: List[HPOMatchItem]
    mondo_matches: List[MONDOMatchItem]


class BatchSymptomMatchResponse(BaseModel):
    total_symptoms: int
    matches: List[SymptomMatchResult]


class PatientVisitMigrationResponse(BaseModel):
    status: str
    batch_size: int
    total_records_processed: int
    batches_processed: int
    upserted_count: int
    modified_count: int
    matched_count: int


class PatientDiagnosisMigrationResponse(BaseModel):
    status: str
    batch_size: int
    total_encounters_processed: int
    batches_processed: int
    modified_count: int
    matched_count: int
    unique_terms_indexed: int
    execution_time_seconds: float


class PatientMedicationMigrationResponse(BaseModel):
    status: str
    batch_size: int
    total_encounters_processed: int
    batches_processed: int
    modified_count: int
    matched_count: int
    execution_time_seconds: float


class PatientDemographicsMigrationResponse(BaseModel):
    status: str
    batch_size: int
    total_records_processed: int
    batches_processed: int
    modified_count: int
    matched_count: int
    execution_time_seconds: float



class ChiefComplaintsRunStatusResponse(BaseModel):
    status: str
    batch_size: int
    current_batch: int
    total_batches: int
    total_rows_processed: int
    documents_updated: int
    documents_skipped: int
    current_step: str
    start_time: Optional[float] = None
    elapsed_seconds: float
    error: Optional[str] = None


class ChiefComplaintsActionResponse(BaseModel):
    status: str
    message: str
    batch_size: Optional[int] = None


class DecoratedSymptomItem(BaseModel):
    term: str = Field(..., description="Original clinical symptom string")
    hpoTerm: Optional[str] = Field(default=None, description="Matched HPO concept label")
    hpoCode: Optional[int] = Field(default=None, description="Numeric HPO code e.g. 2360")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding")
    isNegation: bool = Field(default=False, description="Whether symptom is negated")
    isPheno: bool = Field(default=True, description="Whether entity represents a clinical phenotype")
    similarity: Optional[float] = Field(default=None, description="Similarity score")


class DecoratedDiagnosisItem(BaseModel):
    term: str = Field(..., description="Original clinical diagnosis string")
    mondoTerm: Optional[str] = Field(default=None, description="Matched MONDO concept label")
    mondoCode: Optional[int] = Field(default=None, description="Numeric MONDO code e.g. 8807")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding")
    similarity: Optional[float] = Field(default=None, description="Similarity score")


class DecoratedPatientInfo(BaseModel):
    age: Optional[float] = Field(default=None, description="Patient age in years")
    gender: Optional[str] = Field(default=None, description="Patient gender")


class DecoratedCase(BaseModel):
    caseNo: Optional[int] = Field(default=None, description="Case number or null")
    patientInfo: DecoratedPatientInfo = Field(default_factory=DecoratedPatientInfo)
    symptoms: List[DecoratedSymptomItem] = Field(default_factory=list)
    diagnosis: List[DecoratedDiagnosisItem] = Field(default_factory=list)


class FindSimilarCaseResponse(BaseModel):
    decoratedCase: DecoratedCase
    pastCases: List[Any] = Field(default_factory=list, description="Prior visits with matching caseNo and visitDate < caseDate")
    similarCases: List[Any] = Field(default_factory=list, description="List of similar cases found")
