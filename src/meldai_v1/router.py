"""MeldAI V1 API Router."""

import logging
from typing import Any
from fastapi import APIRouter, HTTPException

from meldai_v1.req_dtos import PrescribeMedicationsRequest
from meldai_v1.res_dtos import (
    PrescribeMedicationsResponse,
    PrescribePipelineActionResponse,
    PrescribePipelineStatusResponse,
)
from core.config import get_settings
from meldai_v1.services.postgres_prescription_service import PostgresPrescriptionService
from meldai_v1.services.prescribe_pipeline_runner import get_prescribe_pipeline_runner

logger = logging.getLogger(__name__)

v1_router = APIRouter(tags=["v1"])

def _dep(name: str, fallback: Any) -> Any:
    """Dynamically resolve dependency from meldai_v1.router (enables mock patching)."""
    import sys
    mod = sys.modules.get("meldai_v1.router")
    if mod and hasattr(mod, name):
        val = getattr(mod, name)
        if val is not fallback:
            return val
    return fallback


def _get_postgres_prescription_service() -> PostgresPrescriptionService:
    settings = get_settings()
    return PostgresPrescriptionService(settings=settings)


@v1_router.post(
    "/cases/prescribe-medications",
    response_model=PrescribeMedicationsResponse,
    summary="[V1] Recommend prescribed medications for a clinical case using Gemini LLM",
)
def prescribe_medications(
    payload: PrescribeMedicationsRequest,
) -> PrescribeMedicationsResponse:
    """Enrich case with HPO/MONDO codes, retrieve past encounters and similar cases from PostgreSQL,
    and construct clinical prompt."""
    try:
        svc_fn = _dep("_get_postgres_prescription_service", _get_postgres_prescription_service)
        prescription_svc = svc_fn()
        return prescription_svc.prescribe_medications(payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Prescribe medications (v1) failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Prescribe medications failed: {str(exc)}"
        ) from exc


@v1_router.post(
    "/cases/prescribe-pipeline/start",
    response_model=PrescribePipelineActionResponse,
    summary="[V1] Start the prescribe-medications batch pipeline in the background",
)
def start_prescribe_pipeline() -> PrescribePipelineActionResponse:
    """Kick off the background pipeline that runs V1 prescribe_medications for every
    case in prescribe-request-all-cases.json and writes results to tmp/."""
    runner = _dep("get_prescribe_pipeline_runner", get_prescribe_pipeline_runner)()
    res = runner.start()
    if res.get("status") == "conflict":
        raise HTTPException(status_code=409, detail=res["message"])
    return PrescribePipelineActionResponse(
        status=res["status"],
        message=res["message"],
        run_id=res.get("run_id"),
        result_file=res.get("result_file"),
    )


@v1_router.post(
    "/cases/prescribe-pipeline/stop",
    response_model=PrescribePipelineActionResponse,
    summary="[V1] Request graceful stop of the active prescribe-medications pipeline",
)
def stop_prescribe_pipeline() -> PrescribePipelineActionResponse:
    """Signal the running pipeline to stop after the current case finishes."""
    runner = _dep("get_prescribe_pipeline_runner", get_prescribe_pipeline_runner)()
    res = runner.stop()
    return PrescribePipelineActionResponse(
        status=res["status"],
        message=res["message"],
    )


@v1_router.get(
    "/cases/prescribe-pipeline/status",
    response_model=PrescribePipelineStatusResponse,
    summary="[V1] Poll the current state and progress of the prescribe-medications pipeline",
)
def get_prescribe_pipeline_status() -> PrescribePipelineStatusResponse:
    """Returns progress counters, current case, elapsed time, and path to the result file."""
    runner = _dep("get_prescribe_pipeline_runner", get_prescribe_pipeline_runner)()
    return PrescribePipelineStatusResponse(**runner.get_status())


# ---------------------------------------------------------------------------
# Compatibility Aliases for V2 Pipeline under /api/v1 prefix
# ---------------------------------------------------------------------------

@v1_router.post(
    "/cases/prescribe-pipeline/v2/start",
    response_model=PrescribePipelineActionResponse,
    summary="[V2 Alias] Start the V2 prescribe-medications batch pipeline",
    include_in_schema=False,
)
def start_prescribe_pipeline_v2_compat() -> PrescribePipelineActionResponse:
    from meldai_v2.services.prescribe_pipeline_runner import get_prescribe_pipeline_v2_runner
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


@v1_router.post(
    "/cases/prescribe-pipeline/v2/stop",
    response_model=PrescribePipelineActionResponse,
    summary="[V2 Alias] Stop the active V2 prescribe-medications pipeline",
    include_in_schema=False,
)
def stop_prescribe_pipeline_v2_compat() -> PrescribePipelineActionResponse:
    from meldai_v2.services.prescribe_pipeline_runner import get_prescribe_pipeline_v2_runner
    runner = _dep("get_prescribe_pipeline_v2_runner", get_prescribe_pipeline_v2_runner)()
    res = runner.stop()
    return PrescribePipelineActionResponse(
        status=res["status"],
        message=res["message"],
    )


@v1_router.get(
    "/cases/prescribe-pipeline/v2/status",
    response_model=PrescribePipelineStatusResponse,
    summary="[V2 Alias] Poll the current state and progress of the V2 prescribe-medications pipeline",
    include_in_schema=False,
)
def get_prescribe_pipeline_v2_status_compat() -> PrescribePipelineStatusResponse:
    from meldai_v2.services.prescribe_pipeline_runner import get_prescribe_pipeline_v2_runner
    runner = _dep("get_prescribe_pipeline_v2_runner", get_prescribe_pipeline_v2_runner)()
    return PrescribePipelineStatusResponse(**runner.get_status())
