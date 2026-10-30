"""MeldAI V2 Response DTO Schemas."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator

from meldai_v2.req_dtos import CaseInput


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
    prior_medications: List[str] = Field(
        default_factory=list,
        description="Distinct medications from prior visits (historical alias)",
    )
    historical_prescription: List[str] = Field(
        default_factory=list,
        description="Medications prescribed across multiple prior encounters",
    )
    latest_documented_prescription: List[str] = Field(
        default_factory=list,
        description="Medications prescribed in the latest recorded encounter",
    )
    patient_reported_current_use: List[str] = Field(
        default_factory=list,
        description="Medicines the patient is actually taking today based on clinical documentation",
    )
    actual_discontinuation: List[str] = Field(
        default_factory=list,
        description="Medications that have been discontinued or stopped",
    )
    clinical_notes: str = Field(default="", description="Clinical synthesis note detailing change trajectory")


class LLMStepCall(BaseModel):
    """Container for prompt and output of a workflow LLM step."""

    prompt: Optional[str] = Field(default=None, description="Complete prompt sent to the LLM")
    output: Any = Field(default=None, description="Structured parsed response or output received from the LLM")


class PrescribeMedicationsV2Response(BaseModel):
    """Full v2 clinical summary and encounter progression response.

    Acts as the single unified memory state across the independent LLM steps:
      Step 1: Deprescribe (evaluates medication continuation/taper/stop actions)
      Step 2: Med Check (batch fills missing candidate medications for entities with 0 meds)
      Step 3: Prescribe (final synthesis reproducing Indian psychiatrist prescribing patterns)
    """

    case: Optional[CaseInput] = Field(default=None, description="Encounter case input")
    clinical_summary: str = Field(default="", description="Human-readable text of clinical summary for the case")
    whats_changed: str = Field(default="", description="Clinical and medication changes reported by LLM")
    whats_changed_med_check: str = Field(default="", description="What changed reported by LLM in med-check step")
    triage_delta: Optional[List[Any]] = Field(
        default=None,
        description="Per-complaint delta: heuristic pre-mark vs. LLM confirmed category/status and what changed",
    )

    # Step 1: Deprescribe
    deprescribe: Optional[LLMStepCall] = Field(
        default=None,
        description="Step 1: Deprescribe review call and decisions",
    )
    deprescribing_recommendations: List[DeprescribingRecommendation] = Field(
        default_factory=list,
        description="Recommendations for stopping, maintaining, or adjusting prior medications",
    )

    # Medication History & Current Use Breakdown
    historical_prescription: List[str] = Field(
        default_factory=list,
        description="Medications prescribed across multiple prior encounters",
    )
    latest_documented_prescription: List[str] = Field(
        default_factory=list,
        description="Medications prescribed in the latest recorded encounter",
    )
    patient_reported_current_use: List[str] = Field(
        default_factory=list,
        description="Medicines the patient is actually taking today based on clinical documentation",
    )
    actual_discontinuation: List[str] = Field(
        default_factory=list,
        description="Medications that have been discontinued or stopped",
    )

    # Step 2: Med Check
    entity_matrix: List[EntityMoleculeRow] = Field(
        default_factory=list,
        description="Per-entity top molecule breakdown updated after Step 2 LLM check",
    )
    llmMedCheck: Optional[LLMStepCall] = Field(
        default=None,
        description="Step 2: Check medications targeting entities and select top molecules for empty entities",
    )
    filled_entities: List[str] = Field(
        default_factory=list,
        description="Entities that had 0 medications and were filled from local formulary",
    )

    # Step 3: Prescribe
    llmMedPrescribe: Optional[LLMStepCall] = Field(
        default=None,
        description="Step 3: Prescribing synthesis based on active_prompt.md",
    )
    medications: List[Union[Dict[str, Any], str]] = Field(
        default_factory=list,
        description="Final recommended medication master objects or molecule/brand name strings",
    )
    explanation: str = Field(default="", description="Clinical justification for the final regimen")

    # Context & Traces
    history_delta: Optional[Union[ClinicalDeltaSummary, Dict[str, Any]]] = Field(
        default=None,
        description="Summary of what changed between past visits and the current encounter",
    )
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

    @model_validator(mode="before")
    @classmethod
    def _migrate_triage_prompt(cls, data: Any) -> Any:
        if isinstance(data, dict) and "triage_prompt" in data:
            prompt_val = data.pop("triage_prompt")
            if prompt_val:
                dep = data.get("deprescribe")
                if dep is None:
                    data["deprescribe"] = {"prompt": prompt_val}
                elif isinstance(dep, dict) and not dep.get("prompt"):
                    dep["prompt"] = prompt_val
                elif hasattr(dep, "prompt") and not dep.prompt:
                    dep.prompt = prompt_val
        return data

    @property
    def triage_prompt(self) -> Optional[str]:
        """Backward-compatible accessor for deprescribe.prompt without duplicate serialization."""
        return self.deprescribe.prompt if self.deprescribe else None

    @triage_prompt.setter
    def triage_prompt(self, value: Optional[str]) -> None:
        if value:
            if self.deprescribe is None:
                self.deprescribe = LLMStepCall(prompt=value)
            else:
                self.deprescribe.prompt = value

    @property
    def historical_prescriptions(self) -> List[str]:
        return self.historical_prescription

    @property
    def prior_medications(self) -> List[str]:
        return self.historical_prescription

    @property
    def discontinued_medications(self) -> List[str]:
        return self.actual_discontinuation


# Single memory state alias
PrescriptionMemoryState = PrescribeMedicationsV2Response


class DeprescribeResponse(BaseModel):
    """Response returned by Step 1: Triage & Deprescribe API."""

    whats_changed: str = Field(..., description="What changed reported by LLM as a string field")
    llm_response: Any = Field(default=None, description="Raw / parsed LLM response from Step 1 prompt")
    deprescribing_recommendations: List[DeprescribingRecommendation] = Field(
        default_factory=list,
        description="Parsed deprescribing recommendations",
    )
    triage_delta: Optional[List[Any]] = Field(
        default=None,
        description="Per-complaint triage delta and change reasoning from Step 1",
    )
    clinical_summary: str = Field(default="", description="Clinical summary for the case")
    state: PrescribeMedicationsV2Response = Field(..., description="Updated single memory state DTO")

    # Medication History & Current Use Breakdown
    historical_prescription: List[str] = Field(
        default_factory=list,
        description="Medications prescribed across multiple prior encounters",
    )
    latest_documented_prescription: List[str] = Field(
        default_factory=list,
        description="Medications prescribed in the latest recorded encounter",
    )
    patient_reported_current_use: List[str] = Field(
        default_factory=list,
        description="Medicines the patient is actually taking today based on clinical documentation",
    )
    actual_discontinuation: List[str] = Field(
        default_factory=list,
        description="Medications that have been discontinued or stopped",
    )

    @property
    def historical_prescriptions(self) -> List[str]:
        return self.historical_prescription

    @property
    def prior_medications(self) -> List[str]:
        return self.historical_prescription

    @property
    def discontinued_medications(self) -> List[str]:
        return self.actual_discontinuation


class MedCheckResponse(BaseModel):
    """Response returned by Step 2: llm_med_check API."""

    whats_changed: str = Field(..., description="Report of missing medications filled for entities as a string field")
    llm_response: Any = Field(default=None, description="Raw / parsed output from llm_med_check prompt")
    filled_entities: List[str] = Field(
        default_factory=list,
        description="Entities with 0 medications that were filled from medication master",
    )
    entity_matrix: List[EntityMoleculeRow] = Field(
        default_factory=list,
        description="Updated entity-molecule matrix",
    )
    state: PrescribeMedicationsV2Response = Field(..., description="Updated single memory state DTO")


class RankedMoleculeItem(BaseModel):
    """Ranked molecule or combination prescribed for similar cases."""

    canonical_molecule: str = Field(..., description="Canonical molecule or FDC combination name")
    is_fdc: bool = Field(default=False, description="True if fixed-dose combination")
    score: float = Field(default=0.0, description="Cumulative similarity score across matching cases")
    frequency: int = Field(..., description="Number of top similar cases where this was prescribed")
    percentage: float = Field(..., description="Percentage of top similar cases prescribing this molecule")
    avg_similarity: float = Field(default=0.0, description="Average composite similarity score of prescribing cases")
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


class PrescribePipelineActionResponse(BaseModel):
    status: str
    message: str
    run_id: Optional[str] = None
    result_file: Optional[str] = None


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


__all__ = [
    "EntityMoleculeRow",
    "DeprescribingRecommendation",
    "ClinicalDeltaSummary",
    "LLMStepCall",
    "PrescribeMedicationsV2Response",
    "PrescriptionMemoryState",
    "DeprescribeResponse",
    "MedCheckResponse",
    "RankedMoleculeItem",
    "CandidateMolecule",
    "RankMoleculesResponse",
    "PrescribePipelineActionResponse",
    "PrescribePipelineStatusResponse",
]
