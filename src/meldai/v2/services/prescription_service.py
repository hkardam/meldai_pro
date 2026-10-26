"""Prescription Service V2 — Per-Entity Molecule Ranking Orchestrator.

Pipeline:
  Step 1:  Decorate current case (HPO + MONDO + SapBERT embeddings).
  Step 1b: Fetch patient's own past visits from MongoDB (delta context).
  Step 2:  Compute clinical delta + generate human-readable clinical summary.
  Step 3:  Fetch candidate pool from MongoDB — excluding the current patient
           (by caseNo) and any records whose visitDate >= current visit date.
  Step 4:  Delegate per-entity ontology resolution, change classification, and
           molecule ranking to EntityRankingService (SRP / DIP).
  Step 7:  LLM Step 1: llmMedCheck (Check & Target Entities / Fill Empty Meds)
  Step 8:  Update Entity Matrix with Step 1 check results
  Step 9:  LLM Step 2: llmMedPrescribe (Final Synthesis)
  Step 10: Local medicine master enrichment & formatting
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from meldai.api.req_dtos import PrescribeMedicationsV2Request, RankMoleculesRequest
from meldai.api.res_dtos import (
    EntityMoleculeRow,
    LLMStepCall,
    PrescribeMedicationsV2Response,
)
from meldai.config import Settings, get_settings
from meldai.db.mongodb import MongoKnowledgeBase
from meldai.nlp.sapbert import get_sapbert_embedder
from meldai.services.case_service import CaseService
from meldai.services.clinical_delta_service import (
    ChiefComplaintChangeService,
    EntityRankingService,
    compute_delta,
    generate_clinical_summary,
    heuristic_triage_complaint,
)
from meldai.services.entity_matrix_service import rank_meds
from meldai.services.gemini_service import (
    GeminiService,
    get_gemini_service,
    get_run_log_file_paths,
    set_llm_run_context,
)
from meldai.terminology.hpo import get_hpo_service
from meldai.terminology.mondo import get_mondo_service
from meldai.terminology.taxonomy import get_hpo_taxonomy, get_mondo_taxonomy
from meldai.utils import (
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
        if (parent / "src" / "meldai").is_dir():
            return parent
    return Path(__file__).resolve().parents[4]


REPO_ROOT = _find_repo_root()


def _build_entity_ranking_service(settings: Settings) -> EntityRankingService:
    """Factory: wire up and return an EntityRankingService from app settings.

    Isolated here (OCP) so that callers can swap any dependency without
    touching PrescriptionServiceV2 itself.
    """
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
    # Strip any content inside parenthesis or brackets if it's descriptive/severity notes
    cleaned = re.sub(r"\([^)]*\)", " ", term)
    cleaned = re.sub(r"\[[^\]]*\]", " ", cleaned)
    # Replace non-alphanumeric (except space)
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


PROMPTS_DIR = REPO_ROOT / "src" / "meldai" / "services" / "prompts" / "medication_prediction"
LLM_MED_CHECK_PROMPT_PATH = PROMPTS_DIR / "llm_med_check_prompt.md"
LLM_MED_PRESCRIBE_PROMPT_PATH = PROMPTS_DIR / "llm_med_prescribe_prompt.md"


def build_llm_med_check_prompt(
    clinical_summary: str,
    entity_matrix_text: str,
    medicine_master: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build the Step 1 prompt for checking direct/indirect entity targeting and selecting top-k meds for empty entities."""
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

    # Fallback inline prompt if file is missing
    return f"""ROLE
You are an expert clinical psychopharmacology assistant reviewing a patient's clinical entities (symptoms and diagnoses) and their candidate medication associations.

GOAL
You need to check medications if they can directly or indirectly target the entities present, or select medications for an entity that has empty top meds from the local medication master catalog (select top-k molecules/FDCs).

<clinical-summary>
{clinical_summary.strip()}
</clinical-summary>

<entity-matrix>
{entity_matrix_text.strip()}
</entity-matrix>

<medication-master>
{master_json_str}
</medication-master>

Respond strictly in valid JSON format with "entity_medication_check" and "summary".
"""


def build_llm_med_prescribe_prompt(
    clinical_summary: str,
    updated_entity_matrix_text: str,
) -> str:
    """Build the Step 2 prompt for prescribing final medications adapting Indian Psychiatrist pattern with updated entity matrix."""
    template = ""
    if LLM_MED_PRESCRIBE_PROMPT_PATH.is_file():
        template = LLM_MED_PRESCRIBE_PROMPT_PATH.read_text(encoding="utf-8")

    if template:
        return (
            template.replace("{clinical_summary}", clinical_summary.strip())
            .replace("{updated_entity_matrix_text}", updated_entity_matrix_text.strip())
        )

    # Fallback inline prompt if file is missing
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
    """Thin orchestrator — coordinates infrastructure calls and delegates
    per-entity ranking to EntityRankingService (DIP).
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
        # Injected or lazily built (DIP — callers can pass a mock for tests)
        self._entity_ranking_svc = (
            entity_ranking_service or _build_entity_ranking_service(self._settings)
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
    # Main public entry point
    # ------------------------------------------------------------------

    def prescribe_medications_v2(
        self,
        request: PrescribeMedicationsV2Request,
    ) -> PrescribeMedicationsV2Response:
        """Run the full v2 multi-step prescribing pipeline.

        Steps:
          1. Fetch patient's past visits from MongoDB (delta context).
          2. Chief complaints change detection, 4-way triage & noise filtering.
          3. Decorate current case (HPO + MONDO + SapBERT) using ONLY clean active symptoms.
          4. Compute clinical delta & generate human-readable clinical summary.
          5. Fetch candidate pool (excluding current patient + future visit dates).
          6. Delegate to EntityRankingService.build_entity_matrix() — no LLM.
          7. Step 1 LLM: llmMedCheck.
          8. Update Entity Matrix.
          9. Step 2 LLM: llmMedPrescribe.
          10. Local medicine master enrichment.
        """
        mongo_kb = self._get_mongo()
        run_id = generate_run_id(case_no=request.case.caseNo)
        set_llm_run_context(run_id)
        vt_str = (
            request.case.visitType.value
            if hasattr(request.case.visitType, "value")
            else str(request.case.visitType)
        ).lower()
        is_followup = vt_str in ("follow-up", "follow_up", "followup")

        # Normalize & clean diagnosis terms: remove non-letter characters before letter starts / after letter ends and trim
        if request.case.diagnosis:
            cleaned_diag = [clean_diagnosis_term(d) for d in request.case.diagnosis]
            request.case.diagnosis = [d for d in cleaned_diag if d]

        # ---------------------------------------------------------------
        # Step 1: Fetch past visits (for delta and clinical summary)
        # ---------------------------------------------------------------
        past_cases: List[Dict[str, Any]] = []
        case_date = request.case.caseDate or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if request.case.caseNo is not None:
            try:
                past_cases = mongo_kb.find_past_cases(
                    case_no=request.case.caseNo,
                    before_date=case_date,
                )
            except Exception as exc:
                logger.warning("Failed fetching past cases: %s", exc)

        logger.info(
            "V2 prescribe: caseNo=%s, visit=%s, %d past visits",
            request.case.caseNo, request.case.visitType.value, len(past_cases),
        )

        # ---------------------------------------------------------------
        # Step 2: Chief complaints triage & change detection (Keywords -> LLM)
        # Filters out administrative metadata, to-do items, global status,
        # and isolates genuine active symptoms from noise.
        # ---------------------------------------------------------------
        change_assessment = self._chief_complaint_service.assess_symptom_changes(
            current_symptoms=request.case.symptoms,
            past_cases=past_cases,
            patient_info=request.case.patientInfo,
            visit_type=request.case.visitType.value,
            current_date=case_date,
        )

        # Extract only genuine, active symptom entities (isolating HPO/SapBERT from noise)
        clean_symptom_entities: List[str] = (
            change_assessment.active_symptoms
            if change_assessment and change_assessment.active_symptoms is not None
            else [s for s in request.case.symptoms if heuristic_triage_complaint(s)[0] == "symptom"]
        )

        # ---------------------------------------------------------------
        # Step 3: Decorate current case (HPO + MONDO + SapBERT embeddings)
        # Using ONLY clean active symptom entities
        # ---------------------------------------------------------------
        decorated_case = self._case_service.decorate_case(
            case_no=request.case.caseNo,
            patient_info=request.case.patientInfo,
            symptoms=clean_symptom_entities,
            diagnosis=request.case.diagnosis,
            visit_type=request.case.visitType.value,
        )

        # ---------------------------------------------------------------
        # Step 4: Compute clinical delta & generate human-readable clinical summary
        # ---------------------------------------------------------------
        delta = compute_delta(
            current_symptoms=request.case.symptoms,
            current_diagnoses=request.case.diagnosis,
            past_cases=past_cases,
            change_assessment=change_assessment,
        )

        clinical_summary = generate_clinical_summary(
            current_visit_type=request.case.visitType.value,
            patient_info=request.case.patientInfo,
            current_symptoms=request.case.symptoms,
            current_diagnoses=request.case.diagnosis,
            past_cases=past_cases,
            current_date=case_date,
            change_assessment=change_assessment,
        )

        save_response_to_tmp(clinical_summary, run_id=f"{run_id}_clinical_summary")
        if change_assessment and change_assessment.symptom_notes:
            save_response_to_tmp(change_assessment.symptom_notes, run_id=f"{run_id}_symptom_notes")
        logger.info("Clinical summary and symptom change assessment for caseNo=%s generated", request.case.caseNo)

        # ---------------------------------------------------------------
        # Step 5: Fetch candidate pool — excluding current patient's case
        #         and any case whose visitDate >= current visit date.
        # ---------------------------------------------------------------
        candidate_pool_limit = getattr(request, "candidate_pool_limit", 500)
        try:
            raw_candidates = mongo_kb.find_similar_candidates(
                exclude_case_no=request.case.caseNo,
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

        logger.info(
            "V2 prescribe: candidate pool size=%d (filtered from %d raw)",
            len(candidates), len(raw_candidates),
        )

        # ---------------------------------------------------------------
        # Step 6: Per-entity molecule ranking — delegated to EntityRankingService
        # ---------------------------------------------------------------
        patient_age: Optional[float] = (
            request.case.patientInfo.age if request.case.patientInfo else None
        )
        patient_gender: Optional[str] = (
            request.case.patientInfo.gender if request.case.patientInfo else None
        )

        entity_matrix: List[EntityMoleculeRow] = self._entity_ranking_svc.build_entity_matrix(
            decorated_case=decorated_case,
            candidates=candidates,
            delta=delta,
            past_cases=past_cases,
            patient_age=patient_age,
            patient_gender=patient_gender,
            top_k=request.top_k_per_entity,
        )

        # ---------------------------------------------------------------
        # Step 7: Step 1 LLM Call — llmMedCheck (Check & Target Entities / Fill Empty Meds)
        # ---------------------------------------------------------------
        master_list = self.load_medicine_master()
        entity_matrix_text = format_entity_matrix_for_prompt(entity_matrix)
        step1_prompt = build_llm_med_check_prompt(
            clinical_summary=clinical_summary,
            entity_matrix_text=entity_matrix_text,
            medicine_master=master_list,
        )

        save_prompt_to_tmp(step1_prompt, run_id=f"{run_id}_step1_llm_med_check_prompt")
        logger.info("Prescription V2: executing Step 1 llmMedCheck for caseNo=%s", request.case.caseNo)

        step1_result = self._gemini_service.generate_prescription(
            step1_prompt,
            run_id=run_id,
            call_name="llm_med_check",
        )

        step1_time = step1_result.get("response_time_seconds", 0.0)
        step1_tokens = step1_result.get("tokens_used", {})
        step1_cost = step1_result.get("cost_usd", 0.0)

        # ---------------------------------------------------------------
        # Step 8: Update Entity Matrix with Step 1 Check Results
        # ---------------------------------------------------------------
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
                    # Update attributed molecules if provided
                    attr_mols = c_info.get("attributed_molecules") or []
                    if attr_mols and isinstance(attr_mols, list):
                        row.attributed_molecules = [str(m) for m in attr_mols if str(m).strip()]

                    # If entity had empty or weak top_molecules, update with newly recommended molecules from master
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

        updated_entity_matrix_text = format_entity_matrix_for_prompt(entity_matrix)

        # ---------------------------------------------------------------
        # Step 9: Step 2 LLM Call — llmMedPrescribe (Final Synthesis based on active_prompt.md)
        # ---------------------------------------------------------------
        step2_prompt = build_llm_med_prescribe_prompt(
            clinical_summary=clinical_summary,
            updated_entity_matrix_text=updated_entity_matrix_text,
        )
        save_prompt_to_tmp(step2_prompt, run_id=f"{run_id}_step2_llm_med_prescribe_prompt")
        logger.info("Prescription V2: executing Step 2 llmMedPrescribe for caseNo=%s", request.case.caseNo)

        step2_result = self._gemini_service.generate_prescription(
            step2_prompt,
            run_id=run_id,
            call_name="llm_med_prescribe",
        )

        step2_time = step2_result.get("response_time_seconds", 0.0)
        step2_tokens = step2_result.get("tokens_used", {})
        step2_cost = step2_result.get("cost_usd", 0.0)

        final_medications = step2_result.get("medications") or []
        final_deprescribe = (
            step2_result.get("deprescribing_recommendations")
            or step2_result.get("deprescribed_meds")
            or []
        )
        final_explanation = step2_result.get("explanation") or ""

        # ---------------------------------------------------------------
        # Step 9.5: Save All Workflow Prompts to Audit File
        # ---------------------------------------------------------------
        prompts_to_log: List[Dict[str, str]] = []
        if change_assessment and change_assessment.llm_prompt:
            prompts_to_log.append({
                "title": "Triage & Symptom Change Detection Prompt",
                "prompt": change_assessment.llm_prompt,
            })
        prompts_to_log.append({
            "title": "Step 1: llmMedCheck Prompt",
            "prompt": step1_prompt,
        })
        prompts_to_log.append({
            "title": "Step 2: llmMedPrescribe Prompt",
            "prompt": step2_prompt,
        })

        workflow_prompts_file = save_workflow_prompts_to_file(
            run_id=run_id,
            case_no=request.case.caseNo,
            visit_type=request.case.visitType.value,
            prompts=prompts_to_log,
        )
        logs_file, _ = get_run_log_file_paths(run_id)
        llm_log_file = str(logs_file)

        # ---------------------------------------------------------------
        # Step 10: Local Medicine Master Enrichment & Formatting
        # ---------------------------------------------------------------
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

                match = master_by_brand.get(b_key) or master_by_brand.get(b_clean) or master_by_molecule.get(mol_key)
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

                # Ensure canonical_molecule is populated
                if not m_copy.get("canonical_molecule"):
                    if match and match.get("canonical_molecule"):
                        m_copy["canonical_molecule"] = match["canonical_molecule"]
                    else:
                        m_copy["canonical_molecule"] = m_copy.get("brand_name") or "Unknown"

                # Ensure is_fdc is strictly boolean
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

        # Calculate aggregated metrics
        total_resp_time = step1_time + step2_time
        p_tok = (step1_tokens.get("prompt_tokens", 0) if isinstance(step1_tokens, dict) else 0) + (
            step2_tokens.get("prompt_tokens", 0) if isinstance(step2_tokens, dict) else 0
        )
        c_tok = (step1_tokens.get("candidates_tokens", 0) if isinstance(step1_tokens, dict) else 0) + (
            step2_tokens.get("candidates_tokens", 0) if isinstance(step2_tokens, dict) else 0
        )
        tot_tok = (step1_tokens.get("total_tokens", 0) if isinstance(step1_tokens, dict) else 0) + (
            step2_tokens.get("total_tokens", 0) if isinstance(step2_tokens, dict) else 0
        )
        cost = step1_cost + step2_cost

        metrics = {
            "response_time_seconds": round(total_resp_time, 4),
            "tokens_used": {
                "prompt_tokens": p_tok,
                "candidates_tokens": c_tok,
                "total_tokens": tot_tok,
            },
            "cost_usd": round(cost, 6),
            "two_step_execution": True,
        }

        # Build step 1 & step 2 LLMStepCall DTOs
        step1_call_dto = LLMStepCall(
            prompt=step1_prompt,
            output={k: v for k, v in step1_result.items() if k not in ("raw_response", "prompt")},
        )
        step2_call_dto = LLMStepCall(
            prompt=step2_prompt,
            output={k: v for k, v in step2_result.items() if k not in ("raw_response", "prompt")},
        )

        return PrescribeMedicationsV2Response(
            clinical_summary=clinical_summary,
            llmMedCheck=step1_call_dto,
            llmMedPrescribe=step2_call_dto,
            medications=enriched_medications,
            entity_matrix=entity_matrix,
            history_delta=delta if is_followup else None,
            explanation=final_explanation or delta.get("clinical_notes") or clinical_summary,
            past_cases_count=len(past_cases),
            run_id=run_id,
            metrics=metrics,
            workflow_prompts_file=workflow_prompts_file,
            llm_log_file=llm_log_file,
        )


# Class alias for clean imports
PrescriptionService = PrescriptionServiceV2
