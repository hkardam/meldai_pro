"""MeldAI V2 API Router."""

import logging
from fastapi import APIRouter, HTTPException

from meldai.api.req_dtos import PrescribeMedicationsV2Request
from meldai.api.res_dtos import (
    PrescribeMedicationsV2Response,
    PrescribePipelineActionResponse,
    PrescribePipelineStatusResponse,
)
from meldai.config import get_settings
from meldai.v2.services.prescription_service import PrescriptionServiceV2
from meldai.v2.services.prescribe_pipeline_runner import get_prescribe_pipeline_v2_runner

logger = logging.getLogger(__name__)

v2_router = APIRouter(tags=["v2"])


def _get_prescription_service_v2() -> PrescriptionServiceV2:
    settings = get_settings()
    return PrescriptionServiceV2(settings=settings)


@v2_router.post(
    "/cases/prescribe-medications",
    response_model=PrescribeMedicationsV2Response,
    summary="[V2] Multi-step medication reasoning with per-entity matrix and deprescribing",
)
def prescribe_medications_v2(
    payload: PrescribeMedicationsV2Request,
) -> PrescribeMedicationsV2Response:
    """Multi-step prescribing pipeline V2:
    1. Decorate the case (HPO / MONDO / SapBERT embeddings).
    2. Fetch patient past visits from MongoDB.
    3. Compute clinical delta (new / persistent / improving / resolved entities).
    4. Run per-entity similarity retrieval against the full MongoDB candidate pool.
    5. Build entity-to-molecule frequency matrix.
    6. LLM Step 1: llmMedCheck (validate targeting & fill gap entities from master).
    7. LLM Step 2: llmMedPrescribe (prescribing synthesis reproducing doctor patterns).
    8. Enrich prescribed medications with local medicine master metadata.
    """
    try:
        svc = _get_prescription_service_v2()
        return svc.prescribe_medications_v2(payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Prescribe medications v2 failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Prescribe medications v2 failed: {str(exc)}"
        ) from exc


@v2_router.post(
    "/cases/prescribe-medications/v2",
    response_model=PrescribeMedicationsV2Response,
    summary="[V2] Alias for /cases/prescribe-medications",
    include_in_schema=False,
)
def prescribe_medications_v2_alias(
    payload: PrescribeMedicationsV2Request,
) -> PrescribeMedicationsV2Response:
    return prescribe_medications_v2(payload)


@v2_router.post(
    "/cases/prescribe-pipeline/start",
    response_model=PrescribePipelineActionResponse,
    summary="[V2] Start the V2 prescribe-medications batch pipeline in the background",
)
def start_prescribe_pipeline_v2() -> PrescribePipelineActionResponse:
    """Kick off the background pipeline that runs V2 multi-step reasoning for all 95 cases
    in prescribe-request-all-cases.json and writes streaming results to tmp/."""
    runner = get_prescribe_pipeline_v2_runner()
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
    runner = get_prescribe_pipeline_v2_runner()
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
    runner = get_prescribe_pipeline_v2_runner()
    return PrescribePipelineStatusResponse(**runner.get_status())
