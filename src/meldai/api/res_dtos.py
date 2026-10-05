"""FastAPI response DTO schemas."""

from typing import List, Optional
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
