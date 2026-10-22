"""Prescription Service V2 — Per-Entity Molecule Ranking Orchestrator.

Pipeline:
  Step 1:  Decorate current case (HPO + MONDO + SapBERT embeddings).
  Step 1b: Fetch patient's own past visits from MongoDB (delta context).
  Step 2:  Compute clinical delta + generate human-readable clinical summary.
  Step 3:  Fetch candidate pool from MongoDB — excluding the current patient
           (by caseNo) and any records whose visitDate >= current visit date.
  Step 4:  Delegate per-entity ontology resolution, change classification, and
           molecule ranking to EntityRankingService (SRP / DIP).
  -> Return entity_matrix directly — no LLM is called.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from meldai.api.req_dtos import PrescribeMedicationsV2Request
from meldai.api.res_dtos import (
    EntityMoleculeRow,
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
from meldai.services.gemini_service import GeminiService, get_gemini_service
from meldai.terminology.hpo import get_hpo_service
from meldai.terminology.mondo import get_mondo_service
from meldai.terminology.taxonomy import get_hpo_taxonomy, get_mondo_taxonomy
from meldai.utils import generate_run_id, parse_clinical_date, save_response_to_tmp

logger = logging.getLogger(__name__)


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
        """
        mongo_kb = self._get_mongo()
        run_id = generate_run_id(case_no=request.case.caseNo)
        vt_str = (
            request.case.visitType.value
            if hasattr(request.case.visitType, "value")
            else str(request.case.visitType)
        ).lower()
        is_followup = vt_str in ("follow-up", "follow_up", "followup")

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

        return PrescribeMedicationsV2Response(
            clinical_summary=clinical_summary,
            consultation_type=request.case.visitType.value,
            medications=[],
            deprescribed_meds=[],
            entity_matrix=entity_matrix,
            history_delta=delta if is_followup else None,
            explanation=delta.get("clinical_notes") or clinical_summary,
            past_cases_count=len(past_cases),
            run_id=run_id,
            metrics={},
            end_prompt=None,
            deprescribe_prompt=None,
        )
