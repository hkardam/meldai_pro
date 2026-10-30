"""Prescription Service V2 — Per-Entity Molecule Ranking Orchestrator.

Pipeline:
  Step 1: Deprescribe / medication action evaluation (independent step / API).
  Step 2: llmMedCheck (find entities with 0 medications, batch fill from local formulary).
  Step 3: llmMedPrescribe (prescribing synthesis reproducing doctor patterns).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from meldai_v2.req_dtos import (
    CaseInput,
    DeprescribeRequest,
    MedCheckRequest,
    PrescribeMedicationsV2Request,
    RankMoleculesRequest,
)
from meldai_v2.res_dtos import (
    DeprescribeResponse,
    DeprescribingRecommendation,
    EntityMoleculeRow,
    LLMStepCall,
    MedCheckResponse,
    PrescribeMedicationsV2Response,
    PrescriptionMemoryState,
)
from core.config import Settings, get_settings
from core.db.mongodb import MongoKnowledgeBase
from core.nlp.sapbert import get_sapbert_embedder
from core.services.case_service import CaseService
from meldai_v2.services.clinical_delta_service import (
    ChiefComplaintChangeService,
    EntityRankingService,
    compute_delta,
    generate_clinical_summary,
    heuristic_triage_complaint,
)
from meldai_v2.services.entity_matrix_service import rank_meds
from core.services.gemini_service import (
    GeminiService,
    get_gemini_service,
    get_run_log_file_paths,
    set_llm_run_context,
)
from core.terminology.hpo import get_hpo_service
from core.terminology.mondo import get_mondo_service
from core.terminology.taxonomy import get_hpo_taxonomy, get_mondo_taxonomy
from core.utils import (
    clean_diagnosis_term,
    generate_run_id,
    parse_clinical_date,
    save_prompt_to_tmp,
    save_response_to_tmp,
)

logger = logging.getLogger(__name__)


def _find_repo_root() -> Path:
    """Dynamically determine repository root by finding project anchors."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "src" / "core").is_dir():
            return parent
    return Path(__file__).resolve().parents[4]


REPO_ROOT = _find_repo_root()


def _build_entity_ranking_service(settings: Settings) -> EntityRankingService:
    """Factory: wire up and return an EntityRankingService from app settings."""
    embedder = get_sapbert_embedder(settings)
    hpo_svc = get_hpo_service(settings.hpo_obo_path)
    mondo_svc = get_mondo_service(settings.mondo_obo_path)

    try:
        hpo_taxonomy = get_hpo_taxonomy(settings.hpo_obo_path)
    except Exception:
        hpo_taxonomy = None
    try:
        mondo_taxonomy = get_mondo_taxonomy(settings.mondo_obo_path)
    except Exception:
        mondo_taxonomy = None

    return EntityRankingService(
        hpo_service=hpo_svc,
        mondo_service=mondo_svc,
        hpo_taxonomy=hpo_taxonomy,
        mondo_taxonomy=mondo_taxonomy,
        embedder=embedder,
    )


# ---------------------------------------------------------------------------
# Prompt Builders & Helpers — Specialized for Indian Psychiatry
# ---------------------------------------------------------------------------

def strip_special_characters(term: str) -> str:
    """Clean clinical term by stripping punctuation, symbols, brackets, and extraneous characters."""
    if not term:
        return ""
    cleaned = re.sub(r"\([^)]*\)", " ", term)
    cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else re.sub(r"[^\w\s]", " ", term).strip()


def format_entity_matrix_for_prompt(entity_matrix: List[EntityMoleculeRow]) -> str:
    """Format the per-entity medication matrix into a clear clinical prompt section
    with prescribing frequencies, percentages, scores, and Indian doctor brand names.
    """
    if not entity_matrix:
        return "No entity-wise historical medication data available.\n"

    lines = []
    lines.append("### HISTORICAL MEDICATION RELATION MATRIX (ENTITY-WISE PRESCRIPTION FREQUENCY):")
    lines.append(
        "For each clinical entity (symptoms & diagnoses), below are the candidate molecules/FDCs historically prescribed "
        "by psychiatrists for matching clinical presentations, ranked by doctor prescribing frequency and similarity score:\n"
    )

    for row in entity_matrix:
        ent_name = row.entity
        ent_type = (row.entity_type or "symptom").capitalize()
        status_str = f", Clinical Trajectory: {row.delta_status.upper()}" if row.delta_status else ""
        curie_str = f" [{row.curie}]" if row.curie else ""

        lines.append(f"- Entity: \"{ent_name}\" (Type: {ent_type}{status_str}{curie_str})")

        if not row.top_molecules:
            lines.append("  * [NO HISTORICAL MEDICATION MATCH FOUND IN DATABASE - CANDIDATE GAP]")
            lines.append("")
            continue

        for mol in row.top_molecules:
            canon = mol.get("canonical_molecule") or mol.get("name") or "Unknown"
            is_fdc_str = "[FDC]" if mol.get("is_fdc") else "[Single]"
            freq = mol.get("frequency", 0)
            pct = mol.get("percentage", 0.0)
            score = mol.get("score", 0.0)
            brands = mol.get("doctor_brands") or []
            brand_str = f" | Doctor Brands: {', '.join(brands[:4])}" if brands else ""

            lines.append(
                f"  * {canon} {is_fdc_str} "
                f"(Prescribed in {freq} matching cases, {pct:.1f}% frequency, Score: {score:.2f}){brand_str} "
            )
        lines.append("")

    return "\n".join(lines)


def save_workflow_prompts_to_file(
    *,
    run_id: str,
    case_no: Any,
    visit_type: str,
    prompts: List[Dict[str, str]],
) -> str:
    """Save all prompts executed across the clinical workflow into a single structured audit file."""
    lines = [
        "=" * 80,
        "MELDAI CLINICAL INTELLIGENCE ENGINE — WORKFLOW PROMPTS AUDIT LOG",
        f"Case Number : {case_no}",
        f"Visit Type  : {visit_type}",
        f"Run ID      : {run_id}",
        f"Timestamp   : {datetime.now(timezone.utc).isoformat()}",
        "=" * 80,
        "",
    ]

    for idx, p in enumerate(prompts, start=1):
        title = p.get("title", f"Step {idx}")
        content = (p.get("prompt") or "").strip()
        lines.extend([
            "=" * 80,
            f"PROMPT #{idx}: {title.upper()}",
            "=" * 80,
            content if content else "[No prompt content]",
            "",
            "-" * 80,
            "",
        ])

    combined_text = "\n".join(lines)
    target_dir = REPO_ROOT / "tmp"
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / f"{run_id}_all_prompts.txt"
    try:
        file_path.write_text(combined_text, encoding="utf-8")
        logger.info("Saved all workflow prompts (%d chars) to %s", len(combined_text), file_path)
    except Exception as exc:
        logger.warning("Failed writing workflow prompts file to %s: %s", file_path, exc)
    return str(file_path)


PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
TRIAGE_DEPRESCRIBE_PROMPT_PATH = PROMPTS_DIR / "triage_deprescribe_prompt.md"
LLM_MED_CHECK_PROMPT_PATH = PROMPTS_DIR / "llm_med_check_prompt.md"
LLM_MED_PRESCRIBE_PROMPT_PATH = PROMPTS_DIR / "llm_med_prescribe_prompt.md"


def _extract_meds_for_prompt(case_doc: Dict[str, Any]) -> List[str]:
    """Extract medication name strings from a MongoDB case document (for prompt formatting)."""
    meds = case_doc.get("medications") or case_doc.get("medication") or []
    result = []
    for m in meds:
        if isinstance(m, str) and m.strip():
            result.append(m.strip())
        elif isinstance(m, dict):
            val = (
                m.get("term")
                or m.get("original_term")
                or m.get("brand_name")
                or m.get("name")
            )
            if val:
                result.append(str(val).strip())
    return result


def build_unified_triage_deprescribe_prompt(
    keyword_results: List[Dict[str, Any]],
    past_cases: List[Dict[str, Any]],
    prior_medications: Optional[List[str]],
    patient_info: Any = None,
    visit_type: Optional[str] = None,
    current_date: Optional[str] = None,
) -> str:
    """Render the unified triage + deprescribe prompt (single LLM call replacing two).

    Reads triage_deprescribe_prompt.md and fills in all placeholders.
    Falls back to an inline version if the file is missing.
    """
    template = ""
    if TRIAGE_DEPRESCRIBE_PROMPT_PATH.is_file():
        template = TRIAGE_DEPRESCRIBE_PROMPT_PATH.read_text(encoding="utf-8")

    # --- Patient context ---
    age_str = "Unknown"
    gender_str = "Unknown"
    if patient_info:
        age = getattr(patient_info, "age", None) if not isinstance(patient_info, dict) else patient_info.get("age")
        gender = getattr(patient_info, "gender", None) if not isinstance(patient_info, dict) else patient_info.get("gender")
        if age is not None:
            age_str = f"{age} yrs"
        if gender:
            gender_str = str(gender)
    vt = visit_type or ("follow-up" if past_cases else "new consultation")
    date_str = current_date or "Unknown"

    # --- Current intake with heuristic pre-marks ---
    intake_lines = []
    for item in keyword_results:
        matched = f", past match: '{item['matched_past']}'" if item["matched_past"] != "None" else ""
        intake_lines.append(
            f"- \"{item['complaint']}\" [heuristic: {item['category']}, mark: {item['display_kw']}{matched}]"
        )
    intake_str = "\n".join(intake_lines) if intake_lines else "- (no intake items)"

    # --- Past visit history ---
    if past_cases:
        hist_lines = []
        for idx, visit in enumerate(past_cases):
            v_date = visit.get("caseDate") or visit.get("visitDate") or visit.get("date") or "Unknown"
            v_type = visit.get("visitType") or "visit"
            syms = visit.get("chiefComplaints") or visit.get("symptoms") or []
            s_list = []
            for s in syms:
                t = (s.get("term") or s.get("text") or s.get("note") or "") if isinstance(s, dict) else str(s)
                if t.strip():
                    s_list.append(t.strip())
            dxs = visit.get("diagnosis") or []
            d_list = []
            for d in dxs:
                t = (d.get("term") or d.get("text") or "") if isinstance(d, dict) else str(d)
                if t.strip():
                    d_list.append(t.strip())
            meds = _extract_meds_for_prompt(visit)
            hist_lines.append(f"Visit #{idx + 1} — {v_date} [{v_type}]:")
            hist_lines.append(f"  Chief complaints: {', '.join(s_list) if s_list else 'None'}")
            hist_lines.append(f"  Diagnoses: {', '.join(d_list) if d_list else 'None'}")
            hist_lines.append(f"  Medications prescribed: {', '.join(meds) if meds else 'None'}")
        history_str = "\n".join(hist_lines)
    else:
        history_str = "No prior encounters on record."

    if template:
        return (
            template
            .replace("{visit_type}", vt)
            .replace("{patient_age}", age_str)
            .replace("{patient_gender}", gender_str)
            .replace("{encounter_date}", date_str)
            .replace("{current_intake_with_heuristics}", intake_str)
            .replace("{past_visit_history}", history_str)
        )

    # Inline fallback
    return f"""You are a clinical pharmacologist and triage physician.

Visit Type: {vt} | Age: {age_str} | Gender: {gender_str} | Date: {date_str}

CURRENT INTAKE:
{intake_str}

PAST VISITS:
{history_str}

RULES:
1. Absence of mention ≠ resolved. Only classify a symptom as resolved if explicitly denied in current intake.
2. Distinguish past prescriptions in visit history from current medication use:
   - Past visits record what was prescribed in the past.
   - Patient-reported current use only includes medications explicitly reported as taken today.
   - Actual discontinuation includes medications confirmed stopped or discontinued.

Return a single JSON object with keys: patient_condition, symptom_notes, whats_changed, triaged_items (array), deprescribing_recommendations (array), patient_reported_current_use (array), actual_discontinuation (array).
"""


def build_llm_med_check_prompt(
    clinical_summary: str,
    entity_matrix_text: str,
    medicine_master: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build the Step 2 prompt for checking direct/indirect entity targeting and selecting top-k meds for empty entities."""
    template = ""
    if LLM_MED_CHECK_PROMPT_PATH.is_file():
        template = LLM_MED_CHECK_PROMPT_PATH.read_text(encoding="utf-8")

    master_json_str = (
        json.dumps(medicine_master, indent=2)
        if medicine_master
        else "[]"
    )

    if template:
        return (
            template.replace("{clinical_summary}", clinical_summary.strip())
            .replace("{entity_matrix_text}", entity_matrix_text.strip())
            .replace("{medicine_master_json}", master_json_str)
        )

    return f"""ROLE
You are an expert clinical psychopharmacology assistant reviewing a patient's clinical entities (symptoms and diagnoses) and their candidate medication associations.

GOAL
For entities identified as having 0 candidate medications, select top-k clinically appropriate candidate molecules or Fixed-Dose Combinations (FDCs) from the local medication master catalog in batch, and report what changed.

<clinical-summary>
{clinical_summary.strip()}
</clinical-summary>

<entities-needing-medications>
{entity_matrix_text.strip()}
</entities-needing-medications>

<medication-master>
{master_json_str}
</medication-master>

Respond strictly in valid JSON format with "whats_changed", "entity_medication_check", and "summary".
"""


def build_llm_med_prescribe_prompt(
    clinical_summary: str,
    updated_entity_matrix_text: str,
) -> str:
    """Build the Step 3 prompt for prescribing final medications adapting Indian Psychiatrist pattern with updated entity matrix."""
    template = ""
    if LLM_MED_PRESCRIBE_PROMPT_PATH.is_file():
        template = LLM_MED_PRESCRIBE_PROMPT_PATH.read_text(encoding="utf-8")

    if template:
        return (
            template.replace("{clinical_summary}", clinical_summary.strip())
            .replace("{updated_entity_matrix_text}", updated_entity_matrix_text.strip())
        )

    return f"""ROLE
You are an assistant system for a psychiatrist practicing in India.

GOAL
Predict which medicines this physician would prescribe for the current patient, reproducing their own prescribing pattern.

<data>
### CLINICAL SUMMARY:
{clinical_summary.strip()}
</data>

<updated-entity-matrix>
{updated_entity_matrix_text.strip()}
</updated-entity-matrix>

Respond strictly in valid JSON with "medications" and "explanation".
"""


class PrescriptionServiceV2:
    """Orchestrator for multi-step clinical reasoning:
      Step 1: Deprescribe / medication action evaluation.
      Step 2: Med Check (batch fill missing candidate medications for entities with 0 meds).
      Step 3: Final Prescribing Synthesis.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        case_service: Optional[CaseService] = None,
        mongo_kb: Optional[MongoKnowledgeBase] = None,
        gemini_service: Optional[GeminiService] = None,
        entity_ranking_service: Optional[EntityRankingService] = None,
        chief_complaint_service: Optional[ChiefComplaintChangeService] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._case_service = case_service or CaseService(settings=self._settings)
        self._mongo_kb = mongo_kb
        self._gemini_service = gemini_service or get_gemini_service(self._settings)
        self._chief_complaint_service = chief_complaint_service or ChiefComplaintChangeService(
            gemini_service=self._gemini_service,
            settings=self._settings,
        )
        self._entity_ranking_svc = (
            entity_ranking_service or _build_entity_ranking_service(self._settings)
        )

    def _get_mongo(self) -> MongoKnowledgeBase:
        if self._mongo_kb is not None:
            return self._mongo_kb
        return MongoKnowledgeBase(self._settings)

    def load_medicine_master(self) -> List[Dict[str, Any]]:
        """Load local medicine master catalog from JSON."""
        target_path = Path(self._settings.medicine_master_json_path)
        if not target_path.is_file():
            candidate = REPO_ROOT / target_path
            if candidate.is_file():
                target_path = candidate
        if target_path.is_file():
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception as exc:
                logger.warning("Failed to load local medicine master: %s", exc)
        return []

    # ------------------------------------------------------------------
    # Context Preparation (Case decoration, delta, clinical summary)
    # ------------------------------------------------------------------

    def _prepare_encounter_context(
        self,
        case: CaseInput,
        run_id: Optional[str] = None,
    ) -> Tuple[Any, Any, str, List[Dict[str, Any]], Any, str]:
        mongo_kb = self._get_mongo()
        if not run_id:
            run_id = generate_run_id(case_no=case.caseNo)
        set_llm_run_context(run_id)

        # Normalize & clean diagnosis terms
        if case.diagnosis:
            cleaned_diag = [clean_diagnosis_term(d) for d in case.diagnosis]
            case.diagnosis = [d for d in cleaned_diag if d]

        # 1. Fetch past visits
        past_cases: List[Dict[str, Any]] = []
        case_date = case.caseDate or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if case.caseNo is not None:
            try:
                past_cases = mongo_kb.find_past_cases(
                    case_no=case.caseNo,
                    before_date=case_date,
                )
            except Exception as exc:
                logger.warning("Failed fetching past cases: %s", exc)

        # 2. Unified triage + deprescribe: single LLM call returns both triaged_items
        #    (for SapBERT/HPO embedding) and deprescribing_recommendations.
        prior_meds_across_visits: List[str] = []
        for v in past_cases:
            for m in _extract_meds_for_prompt(v):
                if m not in prior_meds_across_visits:
                    prior_meds_across_visits.append(m)

        change_assessment = self._chief_complaint_service.assess_and_deprescribe(
            current_symptoms=case.symptoms,
            past_cases=past_cases,
            prior_medications=prior_meds_across_visits,
            patient_info=case.patientInfo,
            visit_type=case.visitType.value,
            current_date=case_date,
        )

        clean_symptom_entities: List[str] = (
            change_assessment.active_symptoms
            if change_assessment and change_assessment.active_symptoms is not None
            else [s for s in case.symptoms if heuristic_triage_complaint(s)[0] == "symptom"]
        )

        # 3. Decorate current case (HPO + MONDO + SapBERT)
        decorated_case = self._case_service.decorate_case(
            case_no=case.caseNo,
            patient_info=case.patientInfo,
            symptoms=clean_symptom_entities,
            diagnosis=case.diagnosis,
            visit_type=case.visitType.value,
        )

        # 4. Clinical delta & clinical summary (uses already-triaged assessment)
        delta = compute_delta(
            current_symptoms=case.symptoms,
            current_diagnoses=case.diagnosis,
            past_cases=past_cases,
            change_assessment=change_assessment,
        )

        clinical_summary = generate_clinical_summary(
            current_visit_type=case.visitType.value,
            patient_info=case.patientInfo,
            current_symptoms=case.symptoms,
            current_diagnoses=case.diagnosis,
            past_cases=past_cases,
            current_date=case_date,
            change_assessment=change_assessment,
        )

        save_response_to_tmp(clinical_summary, run_id=f"{run_id}_clinical_summary")
        if change_assessment and change_assessment.symptom_notes:
            save_response_to_tmp(change_assessment.symptom_notes, run_id=f"{run_id}_symptom_notes")

        return decorated_case, delta, clinical_summary, past_cases, change_assessment, run_id

    # ------------------------------------------------------------------
    # Step 1: Deprescribe Step (Independent API & Method)
    # ------------------------------------------------------------------

    def deprescribe_step(
        self,
        request: Union[DeprescribeRequest, PrescribeMedicationsV2Request, PrescribeMedicationsV2Response],
    ) -> DeprescribeResponse:
        """Step 1: Deprescribe / medication action evaluation.

        Evaluates current symptoms/diagnoses and prior medications to determine
        stop | maintained | decreased | increased | new actions.
        Returns the LLM's response + what's changed string field + updated state.
        """
        if isinstance(request, PrescribeMedicationsV2Response):
            state = request
            case = state.case
            if not case:
                raise ValueError("Encounter case missing from memory state.")
        elif hasattr(request, "case") and getattr(request, "case") is not None:
            case = request.case
            state = getattr(request, "state", None) or PrescribeMedicationsV2Response(case=case)
        else:
            raise ValueError("Invalid request format for deprescribe_step.")

        run_id = state.run_id or generate_run_id(case_no=case.caseNo)
        state.run_id = run_id
        set_llm_run_context(run_id)

        # Prepare context if clinical_summary or delta not yet prepared
        if not state.clinical_summary or not state.history_delta:
            decorated_case, delta, clinical_summary, past_cases, change_assessment, run_id = (
                self._prepare_encounter_context(case=case, run_id=run_id)
            )
            state.clinical_summary = clinical_summary
            state.history_delta = delta
            state.past_cases_count = len(past_cases)
            if change_assessment and getattr(change_assessment, "llm_prompt", None):
                state.triage_prompt = change_assessment.llm_prompt
            if change_assessment and getattr(change_assessment, "triage_delta", None) is not None:
                state.triage_delta = change_assessment.triage_delta
            # Store unified deprescribing result into state immediately so the caching
            # check below finds it without a second LLM call.
            unified_recs = getattr(change_assessment, "deprescribing_recommendations", None)
            unified_whats_changed = getattr(change_assessment, "whats_changed", None)
            if unified_recs is not None and state.deprescribe is None:
                state.deprescribe = LLMStepCall(
                    prompt=getattr(change_assessment, "llm_prompt", None),
                    output={
                        "whats_changed": unified_whats_changed or "",
                        "deprescribing_recommendations": unified_recs,
                        "source": "unified_triage_deprescribe",
                    },
                )
        else:
            clinical_summary = state.clinical_summary
            delta = state.history_delta
            past_cases = []  # already stored counts; no re-fetch needed for prompt

        prior_meds: List[str] = (
            delta.prior_medications
            if delta and hasattr(delta, "prior_medications")
            else []
        )
        delta_notes: str = (
            delta.clinical_notes
            if delta and hasattr(delta, "clinical_notes")
            else ""
        )

        # The unified triage+deprescribe LLM call already ran inside _prepare_encounter_context.
        # If the change_assessment already carries deprescribing_recommendations, use them directly
        # without a second LLM call. Only fall back to a standalone deprescribe LLM call when:
        #   - the state was passed in pre-built (no access to change_assessment), OR
        #   - it was an explicitly requested DeprescribeRequest with no prior unified result.
        _cached_deprescribe_recs = None
        _cached_whats_changed = None
        _cached_deprescribe_prompt = state.deprescribe.prompt if state.deprescribe else None

        # Try to extract deprescribing result from the unified call cached in state
        if state.deprescribe is not None and state.deprescribe.output:
            _out = state.deprescribe.output
            if isinstance(_out, dict):
                _cached_whats_changed = _out.get("whats_changed") or ""
                _cached_deprescribe_recs = _out.get("deprescribing_recommendations") or []

        # Check if prior meds exist OR there are past cases; skip if neither (initial visit, no meds)
        has_context = bool(prior_meds) or bool(getattr(state, "past_cases_count", 0))

        whats_changed: str = ""
        recs_data: List[Any] = []

        if _cached_deprescribe_recs is not None:
            # Unified call result is available — no second LLM call needed
            deprescribe_prompt = _cached_deprescribe_prompt or ""
            whats_changed = _cached_whats_changed or ""
            recs_data = _cached_deprescribe_recs or []
            llm_result = state.deprescribe.output if isinstance(state.deprescribe.output, dict) else {}
        elif not has_context and not isinstance(request, DeprescribeRequest):
            # No prior meds or history — initial visit shortcut
            whats_changed = "No prior medications recorded; initial evaluation with all clinical symptoms considered new."
            llm_result = {
                "whats_changed": whats_changed,
                "deprescribing_recommendations": [],
                "status": "skipped",
            }
            deprescribe_prompt = _cached_deprescribe_prompt or ""
        else:
            # Run the unified triage + deprescribing step (Step 1)
            assessment = self._chief_complaint_service.assess_and_deprescribe(
                current_symptoms=case.symptoms,
                past_cases=past_cases,
                prior_medications=prior_meds,
                patient_info=case.patientInfo,
                visit_type=case.visitType.value,
                current_date=getattr(case, "caseDate", None),
            )
            deprescribe_prompt = assessment.llm_prompt or ""
            state.triage_delta = assessment.triage_delta
            recs_data = [
                r if isinstance(r, dict) else r.model_dump()
                for r in (assessment.deprescribing_recommendations or [])
            ]
            whats_changed = assessment.whats_changed or ""
            llm_result = {
                "whats_changed": whats_changed,
                "deprescribing_recommendations": recs_data,
                "triage_delta": assessment.triage_delta,
            }

        # Extract what's changed and recommendations
        if not whats_changed:
            if isinstance(llm_result, dict):
                whats_changed = (
                    llm_result.get("whats_changed")
                    or llm_result.get("summary")
                    or llm_result.get("clinical_summary")
                    or ""
                )
                if not recs_data:
                    recs_data = (
                        llm_result.get("deprescribing_recommendations")
                        or llm_result.get("recommendations")
                        or llm_result.get("actions")
                        or []
                    )
            elif isinstance(llm_result, list):
                recs_data = llm_result

        recommendations: List[DeprescribingRecommendation] = []
        for r in recs_data:
            if isinstance(r, DeprescribingRecommendation):
                recommendations.append(r)
            elif isinstance(r, dict):
                try:
                    recommendations.append(
                        DeprescribingRecommendation(
                            medicine=r.get("medicine") or r.get("drug") or "Unknown",
                            target_symptom=r.get("target_symptom") or r.get("symptom") or "General presentation",
                            status_change=r.get("status_change") or r.get("status") or "maintained",
                            action=r.get("action") or "maintained",
                            rationale=r.get("rationale") or "",
                            taper_plan=r.get("taper_plan"),
                        )
                    )
                except Exception as exc:
                    logger.debug("Failed parsing DeprescribingRecommendation: %s", exc)

        if not whats_changed:
            if recommendations:
                actions_summary = ", ".join(
                    f"{r.action} {r.medicine} for {r.target_symptom}"
                    for r in recommendations[:4]
                )
                whats_changed = f"Medication actions recommended: {actions_summary}."
            else:
                whats_changed = "No medication changes required; continuing current clinical management."

        # Update unified memory state
        state.deprescribe = LLMStepCall(
            prompt=deprescribe_prompt,
            output={k: v for k, v in llm_result.items() if k not in ("raw_response", "prompt")}
            if isinstance(llm_result, dict)
            else llm_result,
        )
        state.deprescribing_recommendations = recommendations
        state.whats_changed = whats_changed

        # Populate separate medication history & current use fields
        state.historical_prescription = (
            getattr(delta, "historical_prescription", None)
            or (delta.get("historical_prescription") if isinstance(delta, dict) else [])
            or []
        )
        state.latest_documented_prescription = (
            getattr(delta, "latest_documented_prescription", None)
            or (delta.get("latest_documented_prescription") if isinstance(delta, dict) else [])
            or []
        )
        state.patient_reported_current_use = (
            getattr(delta, "patient_reported_current_use", None)
            or (delta.get("patient_reported_current_use") if isinstance(delta, dict) else [])
            or []
        )
        actual_disc = list(
            getattr(delta, "actual_discontinuation", None)
            or (delta.get("actual_discontinuation") if isinstance(delta, dict) else [])
            or []
        )
        for r in recommendations:
            if getattr(r, "action", "") == "stop" or getattr(r, "status_change", "") in ("resolved", "stop"):
                med_name = getattr(r, "medicine", "")
                if med_name and med_name not in actual_disc:
                    actual_disc.append(med_name)
        state.actual_discontinuation = actual_disc

        return DeprescribeResponse(
            whats_changed=whats_changed,
            llm_response=llm_result,
            deprescribing_recommendations=recommendations,
            triage_delta=state.triage_delta,
            clinical_summary=clinical_summary,
            state=state,
            historical_prescription=state.historical_prescription,
            latest_documented_prescription=state.latest_documented_prescription,
            patient_reported_current_use=state.patient_reported_current_use,
            actual_discontinuation=state.actual_discontinuation,
        )

    # ------------------------------------------------------------------
    # Step 2: Med Check Step (Independent API & Method)
    # ------------------------------------------------------------------

    def med_check_step(
        self,
        request: Union[MedCheckRequest, PrescribeMedicationsV2Request, PrescribeMedicationsV2Response],
    ) -> MedCheckResponse:
        """Step 2: llm_med_check.

        Performs deprescribe step internally if not yet performed, builds candidate entity matrix,
        finds entities with 0 medications, asks LLM in batch using local_medicine_master.json
        to fill missing medications for those specific entities, reports the change,
        and updates and returns the new state.
        """
        # 1. Ensure deprescribe step is run internally
        if isinstance(request, PrescribeMedicationsV2Response) and request.deprescribe is not None:
            state = request
            case = state.case
        else:
            deprescribe_res = self.deprescribe_step(request)
            state = deprescribe_res.state
            case = state.case

        if not case:
            raise ValueError("Case details missing for med_check_step.")

        run_id = state.run_id or generate_run_id(case_no=case.caseNo)
        state.run_id = run_id
        set_llm_run_context(run_id)

        # 2. Build initial entity matrix if not present
        if not state.entity_matrix:
            mongo_kb = self._get_mongo()
            candidate_pool_limit = 500
            case_date = case.caseDate or datetime.now(timezone.utc).strftime("%Y-%m-%d")
            try:
                raw_candidates = mongo_kb.find_similar_candidates(
                    exclude_case_no=case.caseNo,
                    limit=candidate_pool_limit,
                )
            except Exception as exc:
                logger.warning("Failed fetching candidate pool: %s", exc)
                raw_candidates = []

            before_dt = parse_clinical_date(case_date)
            candidates: List[Dict[str, Any]] = []
            for c in raw_candidates:
                c_date_val = c.get("visitDate") or c.get("caseDate")
                if c_date_val and before_dt:
                    c_dt = parse_clinical_date(c_date_val)
                    if c_dt and c_dt.date() >= before_dt.date():
                        continue
                candidates.append(c)

            patient_age: Optional[float] = case.patientInfo.age if case.patientInfo else None
            patient_gender: Optional[str] = case.patientInfo.gender if case.patientInfo else None
            top_k = getattr(request, "top_k_per_entity", 5)

            clean_symptoms = [
                s for s in case.symptoms if heuristic_triage_complaint(s)[0] == "symptom"
            ] or case.symptoms

            decorated_case = self._case_service.decorate_case(
                case_no=case.caseNo,
                patient_info=case.patientInfo,
                symptoms=clean_symptoms,
                diagnosis=case.diagnosis,
                visit_type=case.visitType.value,
            )

            state.entity_matrix = self._entity_ranking_svc.build_entity_matrix(
                decorated_case=decorated_case,
                candidates=candidates,
                delta=state.history_delta,
                past_cases=[],
                patient_age=patient_age,
                patient_gender=patient_gender,
                top_k=top_k,
            )

        entity_matrix = state.entity_matrix

        # 3. Find entities with 0 medications
        empty_entities = [
            row for row in entity_matrix
            if not row.top_molecules or len(row.top_molecules) == 0
        ]

        master_list = self.load_medicine_master()
        whats_changed = ""
        step1_result: Any = None

        if empty_entities:
            # Batch fill missing medications for those specific entities
            empty_entities_text = format_entity_matrix_for_prompt(empty_entities)
            step1_prompt = build_llm_med_check_prompt(
                clinical_summary=state.clinical_summary,
                entity_matrix_text=empty_entities_text,
                medicine_master=master_list,
            )
            save_prompt_to_tmp(step1_prompt, run_id=f"{run_id}_step2_llm_med_check_prompt")
            logger.info(
                "Prescription V2: executing Step 2 llmMedCheck for %d entities with 0 meds (caseNo=%s)",
                len(empty_entities), case.caseNo,
            )

            step1_result = self._gemini_service.generate_prescription(
                step1_prompt,
                run_id=run_id,
                call_name="llm_med_check",
            )

            resp_time = step1_result.get("response_time_seconds", 0.0)
            tokens = step1_result.get("tokens_used", {})
            cost = step1_result.get("cost_usd", 0.0)
            state.metrics["llmMedCheck"] = {
                "response_time_seconds": resp_time,
                "tokens_used": tokens,
                "cost_usd": cost,
            }

            whats_changed = (
                step1_result.get("whats_changed")
                or step1_result.get("summary")
                or ""
            )

            # Update entity matrix with newly recommended molecules from master
            entity_check_list: List[Dict[str, Any]] = (
                step1_result.get("entity_medication_check")
                or step1_result.get("entity_checks")
                or []
            )
            if isinstance(entity_check_list, list):
                check_map: Dict[str, Dict[str, Any]] = {}
                for item in entity_check_list:
                    if isinstance(item, dict):
                        ent_k = (item.get("entity") or "").strip().lower()
                        if ent_k:
                            check_map[ent_k] = item

                for row in entity_matrix:
                    ent_k = row.entity.strip().lower()
                    if ent_k in check_map:
                        c_info = check_map[ent_k]
                        attr_mols = c_info.get("attributed_molecules") or []
                        if attr_mols and isinstance(attr_mols, list):
                            row.attributed_molecules = [str(m) for m in attr_mols if str(m).strip()]

                        rec_mols = c_info.get("recommended_top_molecules") or []
                        if rec_mols and isinstance(rec_mols, list):
                            existing_mol_names = {
                                (m.get("canonical_molecule") or "").strip().lower()
                                for m in row.top_molecules
                                if isinstance(m, dict)
                            }
                            for r_mol in rec_mols:
                                if isinstance(r_mol, dict):
                                    c_name = r_mol.get("canonical_molecule") or r_mol.get("name")
                                    if c_name and str(c_name).strip().lower() not in existing_mol_names:
                                        row.top_molecules.append({
                                            "canonical_molecule": c_name,
                                            "is_fdc": bool(r_mol.get("is_fdc", False)),
                                            "frequency": r_mol.get("frequency", 1),
                                            "percentage": r_mol.get("percentage", 100.0),
                                            "score": r_mol.get("score", 1.0),
                                            "doctor_brands": r_mol.get("doctor_brands") or [],
                                        })
                                        existing_mol_names.add(str(c_name).strip().lower())

            if not whats_changed:
                filled_names = [e.entity for e in empty_entities]
                whats_changed = f"Assigned candidate formulary medications for {len(filled_names)} gap entities: {', '.join(filled_names)}."

            state.filled_entities = [e.entity for e in empty_entities]
            state.llmMedCheck = LLMStepCall(
                prompt=step1_prompt,
                output={k: v for k, v in step1_result.items() if k not in ("raw_response", "prompt")},
            )
        else:
            whats_changed = "All clinical entities have existing candidate medication matches; no missing medications required from formulary."
            step1_result = {"status": "skipped", "message": whats_changed}
            state.filled_entities = []
            state.llmMedCheck = LLMStepCall(prompt=None, output=step1_result)

        # Update state
        state.entity_matrix = entity_matrix
        state.whats_changed = whats_changed
        state.whats_changed_med_check = whats_changed

        return MedCheckResponse(
            whats_changed=whats_changed,
            llm_response=step1_result,
            filled_entities=state.filled_entities,
            entity_matrix=entity_matrix,
            state=state,
        )

    # ------------------------------------------------------------------
    # Step 3: Prescribe Step / Full Pipeline Orchestrator
    # ------------------------------------------------------------------

    def prescribe_medications_v2(
        self,
        request: Union[PrescribeMedicationsV2Request, PrescribeMedicationsV2Response],
    ) -> PrescribeMedicationsV2Response:
        """Step 3: Prescribing synthesis and full orchestrator.

        Executes Step 1 (deprescribe) and Step 2 (med_check) if not already completed,
        then executes Step 3 llmMedPrescribe final synthesis reproducing doctor patterns
        and enriches medications with local medicine master metadata.
        """
        if isinstance(request, PrescribeMedicationsV2Response) and request.llmMedCheck is not None:
            state = request
            case = state.case
        else:
            med_check_res = self.med_check_step(request)
            state = med_check_res.state
            case = state.case

        if not case:
            raise ValueError("Case details missing for prescribe_medications_v2.")

        run_id = state.run_id or generate_run_id(case_no=case.caseNo)
        state.run_id = run_id
        set_llm_run_context(run_id)

        # ---------------------------------------------------------------
        # Step 3 LLM Call — llmMedPrescribe (Final Synthesis)
        # ---------------------------------------------------------------
        updated_entity_matrix_text = format_entity_matrix_for_prompt(state.entity_matrix)
        step2_prompt = build_llm_med_prescribe_prompt(
            clinical_summary=state.clinical_summary,
            updated_entity_matrix_text=updated_entity_matrix_text,
        )
        save_prompt_to_tmp(step2_prompt, run_id=f"{run_id}_step3_llm_med_prescribe_prompt")
        logger.info("Prescription V2: executing Step 3 llmMedPrescribe for caseNo=%s", case.caseNo)

        step2_result = self._gemini_service.generate_prescription(
            step2_prompt,
            run_id=run_id,
            call_name="llm_med_prescribe",
        )

        resp_time = step2_result.get("response_time_seconds", 0.0)
        tokens = step2_result.get("tokens_used", {})
        cost = step2_result.get("cost_usd", 0.0)
        state.metrics["llmMedPrescribe"] = {
            "response_time_seconds": resp_time,
            "tokens_used": tokens,
            "cost_usd": cost,
        }
        state.metrics["two_step_execution"] = True

        final_medications = step2_result.get("medications") or []
        final_explanation = step2_result.get("explanation") or ""

        # ---------------------------------------------------------------
        # Local Medicine Master Enrichment & Formatting
        # ---------------------------------------------------------------
        master_list = self.load_medicine_master()
        master_by_brand: Dict[str, Dict[str, Any]] = {}
        master_by_molecule: Dict[str, Dict[str, Any]] = {}
        for m in master_list:
            b = (m.get("brand_name") or "").strip().lower()
            if b and b not in master_by_brand:
                master_by_brand[b] = m
            mol = (m.get("canonical_molecule") or "").strip().lower()
            if mol and mol not in master_by_molecule:
                master_by_molecule[mol] = m

        enriched_medications: List[Dict[str, Any]] = []
        for med in final_medications:
            if isinstance(med, dict):
                m_copy = dict(med)
                b_key = (med.get("brand_name") or "").strip().lower()
                b_clean = re.sub(r"\b\d+\s*(?:mg|ml|mcg|g)\b", "", b_key).strip()
                mol_key = (med.get("canonical_molecule") or "").strip().lower()

                match = (
                    master_by_brand.get(b_key)
                    or master_by_brand.get(b_clean)
                    or master_by_molecule.get(mol_key)
                )
                if match:
                    for field in (
                        "brand_name",
                        "canonical_molecule",
                        "is_fdc",
                        "dosage_group_id",
                        "release_form",
                        "dosage_tier",
                        "constituent_strengths_mg",
                        "modal_strength_mg",
                        "clinical_dosing_indication",
                    ):
                        if field in match and field not in m_copy:
                            m_copy[field] = match[field]

                if not m_copy.get("canonical_molecule"):
                    if match and match.get("canonical_molecule"):
                        m_copy["canonical_molecule"] = match["canonical_molecule"]
                    else:
                        m_copy["canonical_molecule"] = m_copy.get("brand_name") or "Unknown"

                if "is_fdc" in m_copy:
                    if isinstance(m_copy["is_fdc"], str):
                        m_copy["is_fdc"] = m_copy["is_fdc"].strip().lower() in ("true", "1", "yes")
                    else:
                        m_copy["is_fdc"] = bool(m_copy["is_fdc"])
                elif match and "is_fdc" in match:
                    m_copy["is_fdc"] = bool(match["is_fdc"])
                else:
                    mol_str = str(m_copy.get("canonical_molecule") or "")
                    m_copy["is_fdc"] = "+" in mol_str

                enriched_medications.append(m_copy)
            elif isinstance(med, str):
                b_clean = med.strip()
                match = master_by_brand.get(b_clean.lower())
                if match:
                    m_dict = dict(match)
                else:
                    m_dict = {
                        "brand_name": b_clean,
                        "canonical_molecule": b_clean,
                        "is_fdc": "+" in b_clean,
                    }
                enriched_medications.append(m_dict)

        # ---------------------------------------------------------------
        # Audit Log Prompts
        # ---------------------------------------------------------------
        prompts_to_log: List[Dict[str, str]] = []
        step1_prompt = state.deprescribe.prompt if state.deprescribe else state.triage_prompt
        if step1_prompt:
            prompts_to_log.append({
                "title": "Triage & Symptom Change Detection Prompt",
                "prompt": step1_prompt,
            })
        if state.llmMedCheck and state.llmMedCheck.prompt:
            prompts_to_log.append({
                "title": "Step 1: llmMedCheck Prompt",
                "prompt": state.llmMedCheck.prompt,
            })
        prompts_to_log.append({
            "title": "Step 2: llmMedPrescribe Prompt",
            "prompt": step2_prompt,
        })

        workflow_prompts_file = save_workflow_prompts_to_file(
            run_id=run_id,
            case_no=case.caseNo,
            visit_type=case.visitType.value,
            prompts=prompts_to_log,
        )
        logs_file, _ = get_run_log_file_paths(run_id)

        # Update final state
        state.llmMedPrescribe = LLMStepCall(
            prompt=step2_prompt,
            output={k: v for k, v in step2_result.items() if k not in ("raw_response", "prompt")},
        )
        state.medications = enriched_medications
        state.explanation = final_explanation
        state.workflow_prompts_file = workflow_prompts_file
        state.llm_log_file = str(logs_file)

        return state


# Class alias for clean imports
PrescriptionService = PrescriptionServiceV2
