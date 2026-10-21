"""Prescription Service V2 — Multi-Step Medication Reasoning.

Orchestrates four sequential LLM reasoning steps:

  Step 1: Encounter diff (delta engine, pure Python — no LLM).
  Step 2: Per-entity MongoDB retrieval + molecule frequency matrix.
  Step 3: Deprescribing review (LLM — follow-up only).
  Step 4: Regimen synthesis (LLM — final medication list).

Similarities are computed against MongoDB cases using the existing
hybrid LCA + SapBERT vector scorer from similarity_engine.py.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from meldai.api.req_dtos import PrescribeMedicationsV2Request, SimilarityParams
from meldai.api.res_dtos import (
    DeprescribingRecommendation,
    EntityMoleculeRow,
    PrescribeMedicationsV2Response,
)
from meldai.config import Settings, get_settings
from meldai.db.mongodb import MongoKnowledgeBase
from meldai.services.case_service import CaseService
from meldai.services.clinical_delta_service import compute_delta, generate_clinical_summary
from meldai.services.entity_matrix_service import build_entity_matrix
from meldai.services.gemini_service import GeminiService, get_gemini_service
from meldai.terminology.taxonomy import get_hpo_taxonomy, get_mondo_taxonomy
from meldai.utils import generate_run_id, save_prompt_to_tmp, save_response_to_tmp

logger = logging.getLogger(__name__)

_DEPRESCRIBE_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "medication_prediction" / "deprescribe_prompt.md"
)
_SYNTHESIS_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "medication_prediction" / "regimen_synthesis_v2.md"
)
_MEDICINE_MASTER_PATH_DEFAULT = "data/medication/local_medicine_master.json"


def _read_prompt(path: Path) -> str:
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    return ""


class PrescriptionServiceV2:
    """Multi-step prescribing service."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        case_service: Optional[CaseService] = None,
        mongo_kb: Optional[MongoKnowledgeBase] = None,
        gemini_service: Optional[GeminiService] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._case_service = case_service or CaseService(settings=self._settings)
        self._mongo_kb = mongo_kb
        self._gemini_service = gemini_service or get_gemini_service(self._settings)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_mongo(self) -> MongoKnowledgeBase:
        if self._mongo_kb is not None:
            return self._mongo_kb
        return MongoKnowledgeBase(self._settings)

    def _load_medicine_master(self) -> List[Dict[str, Any]]:
        """Load the local medicine master JSON catalogue."""
        path = Path(self._settings.medicine_master_json_path)
        if not path.is_file():
            # Try from repo root
            repo_root = Path(__file__).resolve().parent.parent.parent.parent
            path = repo_root / path
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception as exc:
                logger.warning("Failed to load medicine master: %s", exc)
        return []

    # ------------------------------------------------------------------
    # Step 3: Deprescribing LLM
    # ------------------------------------------------------------------

    def _run_deprescribing_step(
        self,
        entity_prior_meds: Dict[str, Any],  # {entity_term: {"status": ..., "prior_meds": [...]}} or {entity_term: [prior_meds]}
        run_id: str,
    ) -> Tuple[List[DeprescribingRecommendation], Optional[str]]:
        """Ask LLM to determine medication actions (stop | maintained | decreased | increased | new)."""
        if not entity_prior_meds:
            return [], None

        system_prompt = _read_prompt(_DEPRESCRIBE_PROMPT_PATH)

        # Build data section with entity, status, and prior medications
        lines = ["### SYMPTOMS / DIAGNOSES AND PRIOR MEDICATIONS REVIEW:\n"]
        for entity, info in entity_prior_meds.items():
            if isinstance(info, dict):
                status = info.get("status", "unknown")
                meds = info.get("prior_meds", [])
            else:
                status = "unknown"
                meds = info if isinstance(info, list) else []
            meds_str = ", ".join(meds) if meds else "None recorded"
            lines.append(f"- Entity: {entity} (Clinical Status: {status})")
            lines.append(f"  Prior medications: {meds_str}\n")

        data_section = "\n".join(lines)
        full_prompt = f"{system_prompt}\n\n{data_section}"

        save_prompt_to_tmp(full_prompt, run_id=f"{run_id}_deprescribe")

        try:
            raw_text, metrics = self._gemini_service.generate_content_with_metadata(full_prompt)
            save_response_to_tmp(raw_text, run_id=f"{run_id}_deprescribe")
        except Exception as exc:
            logger.error("Deprescribing LLM call failed: %s", exc)
            save_response_to_tmp(f"ERROR: {exc}", run_id=f"{run_id}_deprescribe")
            return [], full_prompt

        # Parse JSON array
        try:
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```", 2)[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip().rstrip("`").strip()
            parsed = json.loads(cleaned)
            if not isinstance(parsed, list):
                parsed = parsed.get("deprescribing", []) if isinstance(parsed, dict) else []
        except Exception as exc:
            logger.warning("Failed to parse deprescribing LLM response: %s", exc)
            return []

        recommendations = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            raw_action = str(item.get("action", "maintained")).strip().lower()
            if "stop" in raw_action:
                norm_action = "stop"
            elif "decreas" in raw_action or "taper" in raw_action:
                norm_action = "decreased"
            elif "increas" in raw_action or "escalat" in raw_action:
                norm_action = "increased"
            elif "new" in raw_action or "initiat" in raw_action:
                norm_action = "new"
            else:
                norm_action = "maintained"

            status_change = str(item.get("status_change", norm_action)).strip().lower()

            try:
                recommendations.append(DeprescribingRecommendation(
                    medicine=item.get("medicine", ""),
                    target_symptom=item.get("target_symptom", ""),
                    status_change=status_change,
                    action=norm_action,
                    rationale=item.get("rationale", ""),
                    taper_plan=item.get("taper_plan"),
                ))
            except Exception as exc:
                logger.debug("Skipping malformed deprescribing item: %s — %s", item, exc)

        logger.info("Deprescribing step: %d recommendations generated", len(recommendations))
        return recommendations, full_prompt

    # ------------------------------------------------------------------
    # Step 4: Regimen Synthesis LLM
    # ------------------------------------------------------------------

    def _build_synthesis_prompt(
        self,
        request: PrescribeMedicationsV2Request,
        entity_matrix: List[EntityMoleculeRow],
        deprescribed: List[DeprescribingRecommendation],
        delta: Dict[str, Any],
        past_cases_count: int,
        medicine_master: List[Dict[str, Any]],
    ) -> str:
        """Assemble the final synthesis prompt from all v2 components."""
        system_text = _read_prompt(_SYNTHESIS_PROMPT_PATH)
        lines: List[str] = []

        if system_text:
            lines.append(system_text)
            lines.append("")

        # --- Current encounter ---
        case = request.case
        is_followup = case.visitType.value == "follow-up"
        lines.append("<data>")
        lines.append("### CURRENT ENCOUNTER:")
        lines.append(f"- Visit Type: {case.visitType.value}")
        if case.caseNo:
            lines.append(f"- Case No: {case.caseNo}")
        age = case.patientInfo.age
        gender = case.patientInfo.gender or "Unknown"
        lines.append(f"- Demographics: Age: {age or 'Unknown'} years, Gender: {gender}")
        lines.append(f"- Symptoms: {', '.join(case.symptoms) if case.symptoms else 'None'}")
        lines.append(f"- Diagnoses: {', '.join(case.diagnosis) if case.diagnosis else 'None'}")
        lines.append("")

        # --- History delta (follow-up only) ---
        if is_followup and delta:
            lines.append("### HISTORY DELTA (what changed since last visit):")
            lines.append(f"  {delta.get('summary_text', 'No changes detected.')}")
            lines.append("")

        # --- Entity medication matrix ---
        lines.append("### ENTITY MEDICATION MATRIX (doctor's historical prescribing by entity):")
        if entity_matrix:
            for row in entity_matrix:
                header = f"- [{row.entity_type.upper()}] {row.entity}"
                if row.curie:
                    header += f" ({row.curie})"
                lines.append(header)
                if row.top_molecules:
                    for mol in row.top_molecules[:8]:  # cap at 8 per entity
                        fdc_flag = " [FDC]" if mol.get("is_fdc") else ""
                        brands = ", ".join(mol.get("doctor_brands", []))
                        brands_note = f" | brands: {brands}" if brands else ""
                        lines.append(
                            f"    * {mol['canonical_molecule']}{fdc_flag} "
                            f"| freq: {mol['frequency']} | similarity: {mol['avg_similarity']}"
                            f"{brands_note}"
                        )
                else:
                    lines.append("    * No historical data found for this entity.")
                lines.append("")
        else:
            lines.append("  No entity matrix data available.")
            lines.append("")

        # --- Deprescribing / medication review plan ---
        if deprescribed:
            lines.append("### MEDICATION REVIEW & DEPRESCRIBING PLAN (stop | maintained | decreased | increased | new):")
            for rec in deprescribed:
                lines.append(f"- {rec.medicine}: {rec.action} (target entity: {rec.target_symptom}, status: {rec.status_change})")
                lines.append(f"  Rationale: {rec.rationale}")
                if rec.taper_plan:
                    lines.append(f"  Adjustment/Taper: {rec.taper_plan}")
            lines.append("")

        lines.append("</data>")
        lines.append("")

        # --- Local formulary ---
        if medicine_master:
            lines.append("<medication-master>")
            lines.append("```json")
            lines.append(json.dumps(medicine_master, indent=2))
            lines.append("```")
            lines.append("</medication-master>")
            lines.append("")

        return "\n".join(lines)

    def _parse_synthesis_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse the regimen synthesis LLM JSON response."""
        cleaned = raw_text.strip()
        import re
        # Strip markdown code fences
        block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        candidate = block.group(1).strip() if block else cleaned
        # Strip stray fences
        if candidate.startswith("```"):
            candidate = candidate[3:]
        if candidate.startswith("json"):
            candidate = candidate[4:]
        candidate = candidate.strip().rstrip("`").strip()

        try:
            return json.loads(candidate)
        except Exception:
            outer = re.search(r"(\{[\s\S]*\})", candidate)
            if outer:
                try:
                    return json.loads(outer.group(1))
                except Exception:
                    pass
        logger.warning("Could not parse synthesis response as JSON.")
        return {}

    # ------------------------------------------------------------------
    # Main public entry point
    # ------------------------------------------------------------------

    def prescribe_medications_v2(
        self,
        request: PrescribeMedicationsV2Request,
    ) -> PrescribeMedicationsV2Response:
        """Run the full v2 multi-step prescribing pipeline."""
        mongo_kb = self._get_mongo()
        run_id = generate_run_id(case_no=request.case.caseNo)
        is_followup = request.case.visitType.value == "follow-up"

        # ---------------------------------------------------------------
        # Step 1: Decorate current case (HPO + MONDO + SapBERT embeddings)
        # ---------------------------------------------------------------
        decorated_case = self._case_service.decorate_case(
            case_no=request.case.caseNo,
            patient_info=request.case.patientInfo,
            symptoms=request.case.symptoms,
            diagnosis=request.case.diagnosis,
            visit_type=request.case.visitType.value,
        )

        # ---------------------------------------------------------------
        # Step 1b: Fetch past visits (for follow-up delta and context)
        # ---------------------------------------------------------------
        past_cases: List[Dict[str, Any]] = []
        if request.case.caseNo is not None:
            case_date = request.case.caseDate or datetime.now(timezone.utc).strftime("%Y-%m-%d")
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
        # Step 2: Clinical delta & Human-Readable Clinical Summary
        # ---------------------------------------------------------------
        delta = compute_delta(
            current_symptoms=request.case.symptoms,
            current_diagnoses=request.case.diagnosis,
            past_cases=past_cases,
        )

        case_date = request.case.caseDate or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        clinical_summary = generate_clinical_summary(
            current_visit_type=request.case.visitType.value,
            patient_info=request.case.patientInfo,
            current_symptoms=request.case.symptoms,
            current_diagnoses=request.case.diagnosis,
            past_cases=past_cases,
            current_date=case_date,
        )

        save_response_to_tmp(clinical_summary, run_id=f"{run_id}_clinical_summary")
        logger.info("Generated clinical summary for caseNo=%s:\n%s", request.case.caseNo, clinical_summary)

        return PrescribeMedicationsV2Response(
            clinical_summary=clinical_summary,
            consultation_type=request.case.visitType.value,
            medications=[],
            deprescribed_meds=[],
            entity_matrix=[],
            history_delta=delta if is_followup else None,
            explanation=clinical_summary,
            past_cases_count=len(past_cases),
            run_id=run_id,
            metrics={},
            end_prompt=clinical_summary,
            deprescribe_prompt=None,
        )
