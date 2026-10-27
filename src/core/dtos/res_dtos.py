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
# V2 Prescribe Medications — Multi-Step Reasoning
# ---------------------------------------------------------------------------

class EntityMoleculeRow(BaseModel):
    """Top candidate molecules/FDCs the doctor historically used for a single symptom or diagnosis entity."""

    entity: str = Field(..., description="Symptom or diagnosis term")
    entity_type: str = Field(..., description="'symptom' or 'diagnosis'")
    curie: Optional[str] = Field(default=None, description="HPO or MONDO CURIE for the entity")
    delta_status: Optional[str] = Field(
        default=None,
        description="Clinical change classification: 'new' | 'maintained' | 'decreased' | 'increased' | 'stop'",
    )
    top_molecules: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ranked molecules/FDCs with doctor frequency, similarity score, and brand examples",
    )
    attributed_molecules: List[str] = Field(
        default_factory=list,
        description="Molecules/FDCs attributed by the LLM as targeting this entity",
    )


class DeprescribingRecommendation(BaseModel):
    """Recommendation for a medication when evaluating clinical entity delta."""

    medicine: str = Field(..., description="Medication name or molecule")
    target_symptom: str = Field(..., description="Symptom or diagnosis entity it targets")
    status_change: str = Field(
        ...,
        description="'stop' | 'maintained' | 'decreased' | 'increased' | 'new' (or 'resolved')",
    )
    action: str = Field(
        ...,
        description="'stop' | 'maintained' | 'decreased' | 'increased' | 'new'",
    )
    rationale: str = Field(..., description="Clinical reasoning for the recommended action")
    taper_plan: Optional[str] = Field(
        default=None,
        description="Step-down schedule, dose adjustment instruction, or null",
    )


class ClinicalDeltaSummary(BaseModel):
    """Clinical change delta between prior visits and current encounter."""

    new_symptoms: List[str] = Field(default_factory=list, description="Newly presenting symptoms")
    maintained_symptoms: List[str] = Field(default_factory=list, description="Persistent symptoms with stable severity")
    decreased_symptoms: List[str] = Field(default_factory=list, description="Improving/milder symptoms")
    increased_symptoms: List[str] = Field(default_factory=list, description="Worsening/more severe symptoms")
    stop_symptoms: List[str] = Field(default_factory=list, description="Resolved symptoms absent in current visit")
    new_diagnoses: List[str] = Field(default_factory=list, description="Newly identified diagnoses")
    maintained_diagnoses: List[str] = Field(default_factory=list, description="Continued diagnoses")
    stop_diagnoses: List[str] = Field(default_factory=list, description="Resolved diagnoses")
    prior_medications: List[str] = Field(default_factory=list, description="Distinct medications from prior visits")
    clinical_notes: str = Field(default="", description="Clinical synthesis note detailing change trajectory")


class LLMStepCall(BaseModel):
    """Container for prompt and output of a workflow LLM step."""

    prompt: Optional[str] = Field(default=None, description="Complete prompt sent to the LLM")
    output: Any = Field(default=None, description="Structured parsed response or output received from the LLM")


class PrescribeMedicationsV2Response(BaseModel):
    """Full v2 clinical summary and encounter progression response."""

    clinical_summary: str = Field(default="", description="Human-readable text of clinical summary for the case")
    llmMedCheck: Optional[LLMStepCall] = Field(
        default=None,
        description="Step 1: Check medications targeting entities and select top molecules for empty entities",
    )
    llmMedPrescribe: Optional[LLMStepCall] = Field(
        default=None,
        description="Step 2: Prescribing synthesis based on active_prompt.md",
    )
    medications: List[Union[Dict[str, Any], str]] = Field(
        default_factory=list,
        description="Final recommended medication master objects or molecule/brand name strings",
    )
    entity_matrix: List[EntityMoleculeRow] = Field(
        default_factory=list,
        description="Per-entity top molecule breakdown updated after Step 1 LLM check",
    )
    history_delta: Optional[Union[ClinicalDeltaSummary, Dict[str, Any]]] = Field(
        default=None,
        description="Summary of what changed between past visits and the current encounter",
    )
    explanation: str = Field(default="", description="Clinical justification for the final regimen")
    past_cases_count: int = Field(default=0, description="Number of prior patient visits used")
    run_id: Optional[str] = Field(default=None, description="Run ID for prompt/response trace files in tmp/")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="LLM token usage, cost, and response time")
    workflow_prompts_file: Optional[str] = Field(
        default=None,
        description="Absolute file path where all prompts used across the workflow were stored",
    )
    llm_log_file: Optional[str] = Field(
        default=None,
        description="Absolute file path where all LLM calls (req + res) for this run were logged",
    )


class RankedMoleculeItem(BaseModel):
    """Ranked molecule or combination prescribed for similar cases."""

    canonical_molecule: str = Field(..., description="Canonical molecule or FDC combination name")
    is_fdc: bool = Field(default=False, description="True if fixed-dose combination")
    score: float = Field(default=0.0, description="Cumulative similarity score across matching cases")
    frequency: int = Field(..., description="Number of top similar cases where this was prescribed")
    percentage: float = Field(..., description="Percentage of top similar cases prescribing this molecule")
    avg_similarity: float = Field(..., description="Average composite similarity score of prescribing cases")
    doctor_brands: List[str] = Field(default_factory=list, description="Brands the doctor used in those cases")


class CandidateMolecule(BaseModel):
    canonical_molecule: str
    frequency: int
    score: float = 0.0


class RankMoleculesResponse(BaseModel):
    """Top molecules ranked by prescription frequency across similar cases."""

    type: str = Field(..., description="Normalized entity type ('chiefComplaint' or 'diagnosis')")
    query: str = Field(..., description="Query symptom or diagnosis term")
    k: int = Field(..., description="Top-k limit returned")
    curie: Optional[str] = Field(default=None, description="Resolved HPO or MONDO CURIE (e.g. HP:0002360)")
    matched_term: Optional[str] = Field(default=None, description="Matched ontology term label")
    similar_cases_count: int = Field(..., description="Number of top similar cases analyzed")
    weights_applied: Dict[str, float] = Field(default_factory=dict, description="Normalized weights applied across components")
    molecules: List[RankedMoleculeItem] = Field(default_factory=list, description="Ranked top-k molecules by frequency")
