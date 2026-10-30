"""MeldAI V2 API Router."""

import logging
from typing import Any
from fastapi import APIRouter, HTTPException

from meldai_v2.req_dtos import (
    DeprescribeRequest,
    MedCheckRequest,
    PrescribeMedicationsV2Request,
)
from meldai_v2.res_dtos import (
    DeprescribeResponse,
    MedCheckResponse,
    PrescribeMedicationsV2Response,
    PrescribePipelineActionResponse,
    PrescribePipelineStatusResponse,
)
from core.config import get_settings
from meldai_v2.services.prescription_service import PrescriptionServiceV2
from meldai_v2.services.prescribe_pipeline_runner import get_prescribe_pipeline_v2_runner

logger = logging.getLogger(__name__)

v2_router = APIRouter(tags=["v2"])


def _dep(name: str, fallback: Any) -> Any:
    """Dynamically resolve dependency from meldai_v2.router (enables mock patching)."""
    import sys
    mod = sys.modules.get("meldai_v2.router")
    if mod and hasattr(mod, name):
        val = getattr(mod, name)
        if val is not fallback:
            return val
    return fallback


def _get_prescription_service_v2() -> PrescriptionServiceV2:
    settings = get_settings()
    return PrescriptionServiceV2(settings=settings)


# ---------------------------------------------------------------------------
# Step 1: Triage & Deprescribe API
# ---------------------------------------------------------------------------

@v2_router.post(
    "/cases/deprescribe",
    response_model=DeprescribeResponse,
    summary="[V2] Step 1: Unified triage of intake & deprescribing evaluation",
)
def deprescribe_v2(payload: DeprescribeRequest) -> DeprescribeResponse:
    """Step 1: Unified Triage & Deprescribing review.
    Evaluates current clinical intake against prior encounters to detect symptom
    status changes (triage_delta: active, resolved, worsening, persistent, improving, new)
    and determine medication actions (stop | maintained | decreased | increased | new)
    using triage_deprescribe_prompt.md.
    Returns what's changed + deprescribing recommendations + triage_delta + updated memory state.
    """
    try:
        svc_fn = _dep("_get_prescription_service_v2", _get_prescription_service_v2)
        svc = svc_fn()
        return svc.deprescribe_step(payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Step 1 Triage & Deprescribe failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Step 1 Triage & Deprescribe failed: {str(exc)}"
        ) from exc

# ---------------------------------------------------------------------------
# Step 2: Med-Check API (Fill missing medications for entities with 0 meds)
# ---------------------------------------------------------------------------

@v2_router.post(
    "/cases/med-check",
    response_model=MedCheckResponse,
    summary="[V2] Step 2: Perform deprescribe internally and batch fill missing candidate meds for entities with 0 meds",
)
def med_check_v2(payload: MedCheckRequest) -> MedCheckResponse:
    """Step 2: llm_med_check.
    Performs Step 1 (triage & deprescribing) internally if not already executed, finds entities
    with 0 candidate medications in the matrix, asks the LLM in batch to fill missing
    medications using local_medicine_master.json, reports what changed, and updates and returns new state.
    """
    try:
        svc_fn = _dep("_get_prescription_service_v2", _get_prescription_service_v2)
        svc = svc_fn()
        return svc.med_check_step(payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Med-check step failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Med-check step failed: {str(exc)}"
        ) from exc


# ---------------------------------------------------------------------------
# Step 3 / Full Orchestrator: Prescribe Medications
# ---------------------------------------------------------------------------

@v2_router.post(
    "/cases/prescribe-medications",
    response_model=PrescribeMedicationsV2Response,
    summary="[V2] Multi-step medication reasoning with per-entity matrix, deprescribing, and formulary enrichment",
)
def prescribe_medications_v2(
    payload: PrescribeMedicationsV2Request,
) -> PrescribeMedicationsV2Response:
    """Multi-step prescribing pipeline V2:
    1. Decorate the case (HPO / MONDO / SapBERT embeddings).
    2. Fetch patient past visits from MongoDB.
    3. Run Step 1: Unified Triage & Deprescribe (evaluate symptom status changes, triage_delta, and prior medication actions).
    4. Run per-entity similarity retrieval against candidate pool and build entity matrix.
    5. Run Step 2: llmMedCheck (find entities with 0 medications and batch fill from local_medicine_master.json).
    6. Run Step 3: llmMedPrescribe (prescribing synthesis reproducing doctor patterns).
    7. Enrich prescribed medications with local medicine master metadata.
    """
    try:
        svc_fn = _dep("_get_prescription_service_v2", _get_prescription_service_v2)
        svc = svc_fn()
        return svc.prescribe_medications_v2(payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Prescribe medications v2 failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Prescribe medications v2 failed: {str(exc)}"
        ) from exc



# ---------------------------------------------------------------------------
# Background Batch Pipeline Endpoints
# ---------------------------------------------------------------------------

@v2_router.post(
    "/cases/prescribe-pipeline/start",
    response_model=PrescribePipelineActionResponse,
    summary="[V2] Start the V2 prescribe-medications batch pipeline in the background",
)
def start_prescribe_pipeline_v2() -> PrescribePipelineActionResponse:
    """Kick off the background pipeline that runs V2 multi-step reasoning for all cases
    in prescribe-request-all-cases.json and writes streaming results to tmp/."""
    runner = _dep("get_prescribe_pipeline_v2_runner", get_prescribe_pipeline_v2_runner)()
    res = runner.start()
    if res.get("status") == "conflict":
        raise HTTPException(status_code=409, detail=res["message"])
    return PrescribePipelineActionResponse(
        status=res["status"],
        message=res["message"],
        run_id=res.get("run_id"),
        result_file=res.get("result_file"),
    )


@v2_router.post(
    "/cases/prescribe-pipeline/stop",
    response_model=PrescribePipelineActionResponse,
    summary="[V2] Request graceful stop of the active V2 prescribe-medications pipeline",
)
def stop_prescribe_pipeline_v2() -> PrescribePipelineActionResponse:
    """Signal the running V2 pipeline to stop after the current case finishes."""
    runner = _dep("get_prescribe_pipeline_v2_runner", get_prescribe_pipeline_v2_runner)()
    res = runner.stop()
    return PrescribePipelineActionResponse(
        status=res["status"],
        message=res["message"],
    )


@v2_router.get(
    "/cases/prescribe-pipeline/status",
    response_model=PrescribePipelineStatusResponse,
    summary="[V2] Poll the current state and progress of the V2 prescribe-medications pipeline",
)
def get_prescribe_pipeline_v2_status() -> PrescribePipelineStatusResponse:
    """Returns progress counters, current case, elapsed time, and path to the result file for V2."""
    runner = _dep("get_prescribe_pipeline_v2_runner", get_prescribe_pipeline_v2_runner)()
    return PrescribePipelineStatusResponse(**runner.get_status())
