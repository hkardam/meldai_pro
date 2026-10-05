"""FastAPI router — HTTP endpoints for terminology, NLP, and clinical data migrations."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from meldai.api.req_dtos import (
    BatchSymptomMatchRequest,
    EmbedRequest,
    FindSimilarCaseRequest,
    SegmentSymptomsRequest,
)
from meldai.api.res_dtos import (
    BatchSymptomMatchResponse,
    ChiefComplaintsActionResponse,
    ChiefComplaintsRunStatusResponse,
    EmbedResponse,
    FindSimilarCaseResponse,
    HPOMatchItem,
    MONDOMatchItem,
    PatientDemographicsMigrationResponse,
    PatientDiagnosisMigrationResponse,
    PatientMedicationMigrationResponse,
    PatientVisitMigrationResponse,
    SegmentSymptomsResponse,
    SymptomItem,
    SymptomMatchResult,
)

from meldai.config import get_settings
from meldai.db.mongodb import MongoKnowledgeBase
from meldai.db.postgres import PostgresSource
from meldai.nlp.medspacy_nlp import get_medspacy_service
from meldai.nlp.sapbert import SapBERTEmbedder, get_sapbert_embedder
from meldai.services.assertion_service import ClinicalAssertionService
from meldai.services.case_service import CaseService
from meldai.services.chief_complaints_runner import get_chief_complaints_runner
from meldai.services.migration_service import MigrationService
from meldai.services.symptom_service import SymptomService
from meldai.services.terminology_service import TerminologySearchService
from meldai.terminology.hpo import HPOResult, get_hpo_service
from meldai.terminology.mondo import MONDOResult, get_mondo_service


logger = logging.getLogger(__name__)

api_router = APIRouter(prefix="/api/v1")





# ---------------------------------------------------------------------------
# Service Factories (wired to router dependencies for testing/mocking)
# ---------------------------------------------------------------------------

def _get_terminology_service() -> TerminologySearchService:
    settings = get_settings()
    nlp = get_medspacy_service()
    hpo = get_hpo_service(settings.hpo_obo_path)
    mondo = get_mondo_service(settings.mondo_obo_path)
    assertion = ClinicalAssertionService(nlp_service=nlp)
    return TerminologySearchService(
        settings=settings,
        assertion_service=assertion,
        hpo_service=hpo,
        mondo_service=mondo,
    )


def _get_symptom_service() -> SymptomService:
    settings = get_settings()
    nlp = get_medspacy_service()
    hpo = get_hpo_service(settings.hpo_obo_path)
    mondo = get_mondo_service(settings.mondo_obo_path)
    assertion = ClinicalAssertionService(nlp_service=nlp)
    return SymptomService(
        settings=settings,
        nlp_service=nlp,
        assertion_service=assertion,
        hpo_service=hpo,
        mondo_service=mondo,
    )


def _get_case_service() -> CaseService:
    settings = get_settings()
    nlp = get_medspacy_service()
    hpo = get_hpo_service(settings.hpo_obo_path)
    mondo = get_mondo_service(settings.mondo_obo_path)
    assertion = ClinicalAssertionService(nlp_service=nlp)
    embedder = get_sapbert_embedder(settings)
    return CaseService(
        settings=settings,
        assertion_service=assertion,
        hpo_service=hpo,
        mondo_service=mondo,
        embedder=embedder,
    )


def _get_migration_service() -> MigrationService:
    settings = get_settings()
    pg = PostgresSource(settings)
    mongo = MongoKnowledgeBase(settings)
    symptom_svc = _get_symptom_service()
    return MigrationService(
        settings=settings,
        postgres_source=pg,
        mongo_kb=mongo,
        symptom_service=symptom_svc,
    )


# ---------------------------------------------------------------------------
# System & Health
# ---------------------------------------------------------------------------

@api_router.get("/health")
def health_check() -> Dict[str, Any]:
    """Verify connectivity to PostgreSQL, MongoDB, and ontology paths."""
    settings = get_settings()
    pg = PostgresSource(settings)
    mongo = MongoKnowledgeBase(settings)

    return {
        "status": "online",
        "environment": settings.environment,
        "services": {
            "postgres": pg.ping(),
            "mongodb": mongo.ping(),
        },
        "ontologies": {
            "hpo": settings.hpo_obo_path,
            "mondo": settings.mondo_obo_path,
        },
    }


# ---------------------------------------------------------------------------
# Terminology (HPO & MONDO)
# ---------------------------------------------------------------------------

@api_router.get("/hpo/search", response_model=List[HPOResult])
def search_hpo(
    term: str = Query(..., min_length=2, description="Phenotype / symptom term to look up"),
    limit: int = Query(5, ge=1, le=50, description="Maximum number of results"),
    filter_negated: bool = Query(False, description="If true, exclude concepts detected as negated"),
) -> List[HPOResult]:
    """Search Human Phenotype Ontology (HPO) with clinical assertion metadata."""
    try:
        service = _get_terminology_service()
        results = service.search_hpo(term=term, limit=limit, filter_negated=filter_negated)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("HPO search failed for '%s': %s", term, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"HPO search error: {str(exc)}")

    if not results and not (filter_negated and term):
        raise HTTPException(status_code=404, detail=f"No HPO concept found for '{term}'")

    return results


@api_router.get("/mondo/search", response_model=List[MONDOResult])
def search_mondo(
    term: str = Query(..., min_length=2, description="Disease / disorder term to look up"),
    limit: int = Query(5, ge=1, le=50, description="Maximum number of results"),
    filter_negated: bool = Query(False, description="If true, exclude concepts detected as negated"),
) -> List[MONDOResult]:
    """Search MONDO Disease Ontology with clinical assertion metadata."""
    try:
        service = _get_terminology_service()
        results = service.search_mondo(term=term, limit=limit, filter_negated=filter_negated)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("MONDO search failed for '%s': %s", term, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"MONDO search error: {str(exc)}")

    if not results and not (filter_negated and term):
        raise HTTPException(status_code=404, detail=f"No MONDO concept found for '{term}'")

    return results


# ---------------------------------------------------------------------------
# SapBERT Embeddings
# ---------------------------------------------------------------------------

@api_router.post("/embed", response_model=EmbedResponse)
def generate_embeddings(payload: EmbedRequest) -> EmbedResponse:
    """Generate 768-d SapBERT dense clinical embeddings for provided terms."""
    try:
        embedder = get_sapbert_embedder()
        vectors = embedder.embed_entities(payload.terms)
        return EmbedResponse(
            terms=payload.terms,
            dimension=vectors.shape[1] if len(vectors) > 0 else 768,
            embeddings=vectors.tolist(),
        )
    except Exception as exc:
        logger.error("Embedding generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(exc)}")


# ---------------------------------------------------------------------------
# Symptom Segmentation
# ---------------------------------------------------------------------------

@api_router.post(
    "/utils/segment-symptoms",
    response_model=SegmentSymptomsResponse,
    response_model_exclude_none=True,
)
def segment_symptoms(payload: SegmentSymptomsRequest) -> SegmentSymptomsResponse:
    """Segment an unstructured clinical note into phenotype and non-phenotype symptoms."""
    try:
        service = _get_symptom_service()
        items = service.segment_note_symptoms(payload.note)
        return SegmentSymptomsResponse(symptoms=[SymptomItem(**item) for item in items])
    except RuntimeError as exc:
        logger.error("Clinical NLP service unavailable: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail=f"Clinical NLP service unavailable: {str(exc)}")
    except Exception as exc:
        logger.error("Symptom segmentation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Symptom segmentation error: {str(exc)}")


# ---------------------------------------------------------------------------
# Patient Visit Migration (PostgreSQL -> MongoDB)
# ---------------------------------------------------------------------------

@api_router.post("/cases/migrate-patient-visits", response_model=PatientVisitMigrationResponse)
def migrate_patient_visits(
    batch_size: int = Query(1000, ge=1, le=10000, description="Batch size for extracting and pushing records"),
) -> PatientVisitMigrationResponse:
    """Stream patient visits from PostgreSQL and upsert/patch into MongoDB 'cases' collection."""
    try:
        migration_svc = _get_migration_service()
        result = migration_svc.migrate_patient_visits(batch_size=batch_size)
        return PatientVisitMigrationResponse(**result)
    except Exception as exc:
        logger.error("Patient visit migration failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Patient visit migration failed: {str(exc)}")


# ---------------------------------------------------------------------------
# Patient Diagnosis Migration (PostgreSQL -> MongoDB)
# ---------------------------------------------------------------------------

@api_router.post("/cases/migrate-patient-diagnoses", response_model=PatientDiagnosisMigrationResponse)
def migrate_patient_diagnoses(
    batch_size: int = Query(1000, ge=1, le=10000, description="Batch size for extracting and pushing records"),
) -> PatientDiagnosisMigrationResponse:
    """Stream grouped patient diagnoses from PostgreSQL, resolve embeddings, and patch MongoDB documents.

    Non-destructive patching:
    Patches only the 'diagnosis' field using MongoDB $set operator, preserving existing
    fields (like 'symptoms' and 'visitReason').
    """
    try:
        migration_svc = _get_migration_service()
        result = migration_svc.migrate_patient_diagnoses(batch_size=batch_size)
        return PatientDiagnosisMigrationResponse(**result)
    except Exception as exc:
        logger.error("Patient diagnosis migration failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Patient diagnosis migration failed: {str(exc)}")


# ---------------------------------------------------------------------------
# Patient Medication/Prescription Migration (PostgreSQL -> MongoDB)
# ---------------------------------------------------------------------------

@api_router.post("/cases/migrate-patient-prescriptions", response_model=PatientMedicationMigrationResponse)
def migrate_patient_prescriptions(
    batch_size: int = Query(1000, ge=1, le=10000, description="Batch size for extracting and pushing records"),
) -> PatientMedicationMigrationResponse:
    """Stream grouped patient prescription/medication data from PostgreSQL and patch MongoDB case documents.

    Non-destructive patching:
    Patches only the 'medications' field (list of medicine names) using MongoDB $set operator,
    matching documents on compound key (caseNo + visitDate).
    """
    try:
        migration_svc = _get_migration_service()
        result = migration_svc.migrate_patient_prescriptions(batch_size=batch_size)
        return PatientMedicationMigrationResponse(**result)
    except Exception as exc:
        logger.error("Patient prescription migration failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Patient prescription migration failed: {str(exc)}")


# ---------------------------------------------------------------------------
# Patient Demographics Migration (PostgreSQL -> MongoDB)
# ---------------------------------------------------------------------------

@api_router.post("/cases/migrate-patient-demographics", response_model=PatientDemographicsMigrationResponse)
def migrate_patient_demographics(
    batch_size: int = Query(1000, ge=1, le=10000, description="Batch size for extracting and pushing records"),
) -> PatientDemographicsMigrationResponse:
    """Stream patient demographic data from PostgreSQL and patch MongoDB case documents with patientInfo field.

    Non-destructive patching:
    Patches only the 'patientInfo' field ({age: float, gender: 'm' | 'f'}) using MongoDB $set operator,
    matching case documents on 'caseNo'.
    """
    try:
        migration_svc = _get_migration_service()
        result = migration_svc.migrate_patient_demographics(batch_size=batch_size)
        return PatientDemographicsMigrationResponse(**result)
    except Exception as exc:
        logger.error("Patient demographics migration failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Patient demographics migration failed: {str(exc)}")




# ---------------------------------------------------------------------------
# Patient Chief Complaints Migration (PostgreSQL -> MongoDB - Background Runner)
# ---------------------------------------------------------------------------



@api_router.post(
    "/cases/migrate-patient-chief-complaints/start",
    response_model=ChiefComplaintsActionResponse,
)
def start_migrate_patient_chief_complaints(
    batch_size: int = Query(100, ge=1, le=10000, description="Batch size for extracting and pushing records"),
) -> ChiefComplaintsActionResponse:
    """Start asynchronous background migration of patient chief complaints."""
    runner = get_chief_complaints_runner()
    res = runner.start(batch_size=batch_size)
    if res.get("status") == "conflict":
        raise HTTPException(status_code=409, detail=res["message"])
    return ChiefComplaintsActionResponse(**res)


@api_router.post(
    "/cases/migrate-patient-chief-complaints/stop",
    response_model=ChiefComplaintsActionResponse,
)
def stop_migrate_patient_chief_complaints() -> ChiefComplaintsActionResponse:
    """Request graceful stop of the active chief complaints migration runner."""
    runner = get_chief_complaints_runner()
    res = runner.stop()
    return ChiefComplaintsActionResponse(**res)


@api_router.get(
    "/cases/migrate-patient-chief-complaints/status",
    response_model=ChiefComplaintsRunStatusResponse,
)
def get_migrate_patient_chief_complaints_status() -> ChiefComplaintsRunStatusResponse:
    """Get the current run status and metrics of the chief complaints migration."""
    runner = get_chief_complaints_runner()
    status = runner.get_status()
    return ChiefComplaintsRunStatusResponse(**status)


# ---------------------------------------------------------------------------
# Batch Symptom Ontology Matcher (medspaCy Pipe + HPO + MONDO)
# ---------------------------------------------------------------------------

@api_router.post(
    "/symptoms/match-batch",
    response_model=BatchSymptomMatchResponse,
    summary="Batch match clinical symptoms to top-K HPO and MONDO concepts",
)
def match_symptoms_batch(
    payload: BatchSymptomMatchRequest,
) -> BatchSymptomMatchResponse:
    """Batch process a list of raw symptom strings through medspaCy assertion NLP pipeline and return top 3 HPO & MONDO concept matches."""
    try:
        symptom_svc = _get_symptom_service()
        results = symptom_svc.match_symptoms_batch(payload.symptoms, top_k=payload.top_k)
        return BatchSymptomMatchResponse(
            total_symptoms=len(results),
            matches=[SymptomMatchResult(**item) for item in results],
        )
    except Exception as exc:
        logger.error("Batch symptom matching failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch symptom matching failed: {str(exc)}")


# ---------------------------------------------------------------------------
# Similar Cases Search & Case Decoration
# ---------------------------------------------------------------------------

@api_router.post(
    "/cases/find-similar",
    response_model=FindSimilarCaseResponse,
    summary="Decorate a clinical case and find similar cases",
)
def find_similar_cases(
    payload: FindSimilarCaseRequest,
) -> FindSimilarCaseResponse:
    """Enrich case symptoms and diagnoses with HPO/MONDO terms and retrieve similar cases."""
    try:
        case_svc = _get_case_service()
        return case_svc.find_similar_cases(payload)
    except Exception as exc:
        logger.error("Find similar cases failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Find similar cases failed: {str(exc)}")



