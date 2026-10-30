"""FastAPI response DTO schemas."""

from typing import Any, Dict, List, Optional, Union
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


class PatientMigrationResponse(BaseModel):
    status: str = Field(default="success")
    batch_size: int = Field(default=20)
    total_records_processed: int
    batches_processed: int
    upserted_count: int
    modified_count: int
    matched_count: int
    execution_time_seconds: float


class PatientMigrationStatusResponse(BaseModel):
    status: str
    batch_size: int
    current_batch: int
    total_batches: int
    total_records_processed: int
    documents_saved: int
    current_step: str
    start_time: Optional[float] = None
    elapsed_seconds: float
    error: Optional[str] = None


class PatientMigrationActionResponse(BaseModel):
    status: str
    message: str
    batch_size: Optional[int] = None


# Backward-compatible aliases
PatientVisitMigrationResponse = PatientMigrationResponse
PatientDiagnosisMigrationResponse = PatientMigrationResponse
PatientMedicationMigrationResponse = PatientMigrationResponse
PatientDemographicsMigrationResponse = PatientMigrationResponse
ChiefComplaintsRunStatusResponse = PatientMigrationStatusResponse
ChiefComplaintsActionResponse = PatientMigrationActionResponse


class DecoratedSymptomItem(BaseModel):
    term: str = Field(..., description="Original clinical symptom string")
    hpoTerm: Optional[str] = Field(default=None, description="Matched HPO concept label")
    hpoCode: Optional[int | str] = Field(default=None, description="Numeric HPO code or CURIE e.g. 2360 or HP:0002360")
    embedding: Optional[List[float]] = Field(default=None, description="Dense embedding vector")
    isNegation: bool = Field(default=False, description="Negation flag from medspaCy ConText")
    isPheno: bool = Field(default=True, description="Phenotype relevance flag")
    similarity: Optional[float] = Field(default=None, description="Optional similarity score")


class DecoratedDiagnosisItem(BaseModel):
    term: str = Field(..., description="Original clinical diagnosis string")
    mondoTerm: Optional[str] = Field(default=None, description="Matched MONDO concept label")
    mondoCode: Optional[int | str] = Field(default=None, description="Numeric MONDO code or CURIE e.g. 8807 or MONDO:0008807")
    embedding: Optional[List[float]] = Field(default=None, description="Dense embedding vector")
    similarity: Optional[float] = Field(default=None, description="Optional similarity score")


class DecoratedPatientInfo(BaseModel):
    age: Optional[float] = Field(default=None, description="Patient age in years")
    gender: Optional[str] = Field(default=None, description="Patient gender (e.g. 'm', 'f', 'other')")


class DecoratedCase(BaseModel):
    caseNo: Optional[int] = Field(default=None, description="Case number / patient ID")
    caseDate: Optional[str] = Field(default=None, description="ISO date string e.g. '2024-01-15'")
    visitType: Optional[str] = Field(default=None, description="Visit type")
    patientInfo: DecoratedPatientInfo = Field(default_factory=DecoratedPatientInfo)
    symptoms: List[DecoratedSymptomItem] = Field(default_factory=list)
    diagnosis: List[DecoratedDiagnosisItem] = Field(default_factory=list)


class ComponentScores(BaseModel):
    diagnosis: Optional[float] = Field(default=None, description="Diagnosis set similarity [0, 1]")
    symptoms: Optional[float] = Field(default=None, description="Symptom set similarity [0, 1]")
    patient: Optional[float] = Field(default=None, description="Patient age similarity [0, 1]")
    visitType: Optional[float] = Field(default=None, description="Visit type match score (1.0 or 0.5)")


class SimilarCaseItem(BaseModel):
    case: Dict[str, Any] = Field(..., description="Candidate case document from knowledge base")
    finalScore: float = Field(..., ge=0.0, le=1.0, description="Composite normalized similarity score")
    components: ComponentScores = Field(..., description="Breakdown of individual component scores")


class FindSimilarCaseResponse(BaseModel):
    decoratedCase: DecoratedCase
    pastCases: List[Any] = Field(default_factory=list, description="Prior visits with matching caseNo and visitDate < caseDate")
    similarCases: List[SimilarCaseItem] = Field(default_factory=list, description="List of similar cases ranked by finalScore")


class PrescribeMedicationsResponse(BaseModel):
    medications: List[Union[Dict[str, Any], str]] = Field(default_factory=list, description="Recommended prescription medication master objects or names")
    explanation: str = Field(..., description="Clinical reasoning and explanation adapting to doctor's prescription style and patient history")
    pastCasesCount: int = Field(default=0, description="Number of prior visits used in context")
    similarCasesCount: int = Field(default=0, description="Number of similar cases used in context")
    responseTimeSeconds: Optional[float] = Field(default=None, description="LLM inference response time in seconds")
    tokensUsed: Optional[Dict[str, int]] = Field(default=None, description="Tokens used breakdown")
    totalTokens: Optional[int] = Field(default=None, description="Total tokens used by LLM call")
    costUsd: Optional[float] = Field(default=None, description="Estimated inference cost in USD")
    runId: Optional[str] = Field(default=None, description="Run ID identifier for prompt/response files in tmp/")
    run_id: Optional[str] = Field(default=None, description="Snake case alias for runId")
    cost_usd: Optional[float] = Field(default=None, description="Snake case alias for costUsd")
    tokens_used: Optional[Dict[str, int]] = Field(default=None, description="Snake case alias for tokensUsed")
    response_time_seconds: Optional[float] = Field(default=None, description="Snake case alias for responseTimeSeconds")
    raw_response: Optional[str] = Field(default=None, description="Raw unparsed LLM text output")


class LoadMedicineMasterResponse(BaseModel):
    status: str = Field(default="success")
    collection: str = Field(default="medicine_master")
    total_records: int = Field(..., description="Number of records parsed from JSON dataset")
    upserted_count: int = Field(..., description="Number of newly inserted documents")
    modified_count: int = Field(..., description="Number of existing documents updated")
    total_in_db: int = Field(..., description="Current total document count in medicine_master")
    batches_processed: int = Field(..., description="Total batch chunks processed")
    duration_seconds: float = Field(..., description="Total ingestion and embedding time in seconds")


class BioLORDEmbedResponse(BaseModel):
    texts: List[str] = Field(..., description="Input clinical texts")
    dimension: int = Field(default=768, description="BioLORD vector dimensionality")
    embeddings: List[List[float]] = Field(..., description="Dense 768-d BioLORD embedding vectors")


class MedicineSearchResponse(BaseModel):
    query: str = Field(..., description="Original search query")
    total_matches: int = Field(..., description="Number of matching medicine records returned")
    matches: List[Dict[str, Any]] = Field(default_factory=list, description="Ranked medicine master records with similarity scores")


class MedicationSearchQueryResponse(BaseModel):
    vector_semantic_queries: List[str] = Field(default_factory=list, description="Vector search queries matching clinical_dosing_indication")
    bm25_keywords: List[str] = Field(default_factory=list, description="BM25 search keywords across brand names, molecules, and drug classes")


class PrescribePipelineActionResponse(BaseModel):
    status: str = Field(..., description="Action outcome: started | stopping | conflict | idle | completed | failed | stopped")
    message: str = Field(..., description="Human-readable description of the action outcome")
    run_id: Optional[str] = Field(default=None, description="Unique run identifier; present when pipeline is started")
    result_file: Optional[str] = Field(default=None, description="Absolute path to the incremental JSON results file in tmp/")


class PrescribePipelineStatusResponse(BaseModel):
    status: str = Field(..., description="Pipeline execution state: idle | running | stopping | completed | failed | stopped")
    run_id: Optional[str] = Field(default=None, description="Run identifier for the active or last run")
    total_cases: int = Field(default=0, description="Total number of cases loaded from the cases file")
    processed_cases: int = Field(default=0, description="Number of cases processed so far")
    passed_cases: int = Field(default=0, description="Cases that completed without error")
    failed_cases: int = Field(default=0, description="Cases that raised an exception")
    current_case_no: Optional[int] = Field(default=None, description="Case number currently being processed")
    result_file: Optional[str] = Field(default=None, description="Path to the incremental JSON results file")
    elapsed_seconds: float = Field(default=0.0, description="Wall-clock seconds since the pipeline started")
    error: Optional[str] = Field(default=None, description="Top-level fatal error message if status is 'failed'")


# ---------------------------------------------------------------------------
# V2 Prescribe Medications — Multi-Step Reasoning (Canonical: meldai_v2.res_dtos)
# ---------------------------------------------------------------------------
from meldai_v2.res_dtos import (  # noqa: E402
    CandidateMolecule,
    ClinicalDeltaSummary,
    DeprescribeResponse,
    DeprescribingRecommendation,
    EntityMoleculeRow,
    LLMStepCall,
    MedCheckResponse,
    PrescribeMedicationsV2Response,
    PrescriptionMemoryState,
    RankMoleculesResponse,
    RankedMoleculeItem,
)









