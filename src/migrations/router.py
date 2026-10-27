"""FastAPI router for patient and clinical case data migrations."""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from core.config import get_settings
from core.db.mongodb import MongoKnowledgeBase
from core.db.postgres import PostgresSource
from core.dtos.res_dtos import (
    PatientMigrationActionResponse,
    PatientMigrationResponse,
    PatientMigrationStatusResponse,
)
from core.migration import MigrationService, get_patient_migration_runner
from core.nlp.sapbert import get_sapbert_embedder
from core.services.assertion_service import ClinicalAssertionService
from core.terminology.hpo import get_hpo_service
from core.terminology.mondo import get_mondo_service

logger = logging.getLogger(__name__)

migration_router = APIRouter()


def _dep(name: str, fallback: Any) -> Any:
    """Dynamically resolve dependency from migrations.router, meldai.api.router, or core.router (enables mock patching)."""
    import sys
    for mod_name in ("migrations.router", "meldai.api.router", "core.router"):
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, name):
            return getattr(mod, name)
    return fallback


def _get_migration_service() -> MigrationService:
    settings = get_settings()
    pg = PostgresSource(settings)
    mongo = MongoKnowledgeBase(settings)
    assertion = ClinicalAssertionService()
    hpo = _dep("get_hpo_service", get_hpo_service)(settings.hpo_obo_path)
    mondo = _dep("get_mondo_service", get_mondo_service)(settings.mondo_obo_path)
    embedder = _dep("get_sapbert_embedder", get_sapbert_embedder)(settings)
    return MigrationService(
        settings=settings,
        postgres_source=pg,
        mongo_kb=mongo,
        assertion_service=assertion,
        hpo_service=hpo,
        mondo_service=mondo,
        embedder=embedder,
    )


# ---------------------------------------------------------------------------
# Unified Patient Migration (PostgreSQL -> Decorate -> MongoDB)
# ---------------------------------------------------------------------------

@migration_router.post(
    "/cases/migrate",
    response_model=PatientMigrationResponse,
    summary="Run synchronous patient migration in batches of 20",
)
def migrate_patients(
    batch_size: int = Query(20, ge=1, le=1000, description="Batch size for processing records (default: 20)"),
    limit: Optional[int] = Query(None, ge=1, description="Optional maximum number of records to migrate"),
) -> PatientMigrationResponse:
    """Execute the clinical patient migration logical flow."""
    try:
        migration_svc = _dep("_get_migration_service", _get_migration_service)()
        result = migration_svc.run_patient_migration(batch_size=batch_size, limit=limit)
        return PatientMigrationResponse(**result)
    except Exception as exc:
        logger.error("Patient migration failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Patient migration failed: {str(exc)}")


@migration_router.post(
    "/cases/migrate/start",
    response_model=PatientMigrationActionResponse,
    summary="Start asynchronous patient migration in background thread",
)
def start_migrate_patients(
    batch_size: int = Query(20, ge=1, le=1000, description="Batch size for processing records (default: 20)"),
    limit: Optional[int] = Query(None, ge=1, description="Optional maximum number of records to migrate"),
) -> PatientMigrationActionResponse:
    """Start asynchronous background execution of the patient migration pipeline."""
    runner = _dep("get_patient_migration_runner", get_patient_migration_runner)()
    res = runner.start(batch_size=batch_size, limit=limit)
    if res.get("status") == "conflict":
        raise HTTPException(status_code=409, detail=res["message"])
    return PatientMigrationActionResponse(**res)


@migration_router.post(
    "/cases/migrate/stop",
    response_model=PatientMigrationActionResponse,
    summary="Stop active patient migration pipeline",
)
def stop_migrate_patients() -> PatientMigrationActionResponse:
    """Request graceful stop of the active patient migration runner."""
    runner = _dep("get_patient_migration_runner", get_patient_migration_runner)()
    res = runner.stop()
    return PatientMigrationActionResponse(**res)


@migration_router.get(
    "/cases/migrate/status",
    response_model=PatientMigrationStatusResponse,
    summary="Get status and metrics of patient migration runner",
)
def get_migrate_patients_status() -> PatientMigrationStatusResponse:
    """Get the current run status, progress, and step metrics of the patient migration."""
    runner = _dep("get_patient_migration_runner", get_patient_migration_runner)()
    status = runner.get_status()
    return PatientMigrationStatusResponse(**status)
