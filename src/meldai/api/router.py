"""FastAPI router — HTTP endpoints for terminology, NLP, and clinical data migrations."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from meldai.api.req_dtos import (
    BatchSymptomMatchRequest,
    BioLORDEmbedRequest,
    EmbedRequest,
    FindSimilarCaseRequest,
    LoadMedicineMasterRequest,
    MedicationSearchQueryRequest,
    MedicineSearchRequest,
    PrescribeMedicationsRequest,
    PrescribeMedicationsV2Request,
    RankMoleculesRequest,
    SegmentSymptomsRequest,
)
from meldai.api.res_dtos import (
    BatchSymptomMatchResponse,
    BioLORDEmbedResponse,
    EmbedResponse,
    FindSimilarCaseResponse,
    LoadMedicineMasterResponse,
    MedicationSearchQueryResponse,
    MedicineSearchResponse,
    PatientMigrationActionResponse,
    PatientMigrationResponse,
    PatientMigrationStatusResponse,
    PrescribeMedicationsResponse,
    PrescribeMedicationsV2Response,
    PrescribePipelineActionResponse,
    PrescribePipelineStatusResponse,
    RankMoleculesResponse,
    SegmentSymptomsResponse,
    SymptomItem,
    SymptomMatchResult,
)

from meldai.config import get_settings
from meldai.db.mongodb import MongoKnowledgeBase
from meldai.db.postgres import PostgresSource
from meldai.nlp.biolord import get_biolord_embedder
from meldai.nlp.medspacy_nlp import get_medspacy_service
from meldai.nlp.sapbert import get_sapbert_embedder
from meldai.services.assertion_service import ClinicalAssertionService
from meldai.services.case_service import CaseService
from meldai.services.chief_complaints_runner import get_patient_migration_runner
from meldai.services.prescribe_pipeline_runner import get_prescribe_pipeline_runner
from meldai.services.gemini_service import get_gemini_service
from meldai.services.entity_matrix_service import rank_meds
from meldai.services.medicine_master_service import MedicineMasterService
from meldai.services.migration_service import MigrationService
from meldai.services.postgres_prescription_service import PostgresPrescriptionService
from meldai.services.prescription_service import PrescriptionService
from meldai.services.prescription_service_v2 import PrescriptionServiceV2
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
    assertion = ClinicalAssertionService()
    hpo = get_hpo_service(settings.hpo_obo_path)
    mondo = get_mondo_service(settings.mondo_obo_path)
    embedder = get_sapbert_embedder(settings)
    return MigrationService(
        settings=settings,
        postgres_source=pg,
        mongo_kb=mongo,
        assertion_service=assertion,
        hpo_service=hpo,
        mondo_service=mondo,
        embedder=embedder,
    )


def _get_prescription_service() -> PrescriptionService:
    settings = get_settings()
    case_svc = _get_case_service()
    mongo = MongoKnowledgeBase(settings)
    gemini = get_gemini_service(settings)
    return PrescriptionService(
        settings=settings,
        case_service=case_svc,
        mongo_kb=mongo,
        gemini_service=gemini,
    )


def _get_postgres_prescription_service() -> PostgresPrescriptionService:
    settings = get_settings()
    case_svc = _get_case_service()
    pg = PostgresSource(settings)
    gemini = get_gemini_service(settings)
    return PostgresPrescriptionService(
        settings=settings,
        case_service=case_svc,
        postgres_source=pg,
        gemini_service=gemini,
    )



def _get_medicine_master_service() -> MedicineMasterService:
    settings = get_settings()
    mongo = MongoKnowledgeBase(settings)
    biolord = get_biolord_embedder(settings)
    return MedicineMasterService(
        settings=settings,
        mongo_kb=mongo,
        biolord_embedder=biolord,
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
# Unified Patient Migration (PostgreSQL -> Decorate -> MongoDB)
# ---------------------------------------------------------------------------

@api_router.post(
    "/cases/migrate",
    response_model=PatientMigrationResponse,
    summary="Run synchronous patient migration in batches of 20",
)
def migrate_patients(
    batch_size: int = Query(20, ge=1, le=1000, description="Batch size for processing records (default: 20)"),
    limit: Optional[int] = Query(None, ge=1, description="Optional maximum number of records to migrate"),
) -> PatientMigrationResponse:
    """Execute the clinical patient migration logical flow:
    start -> query docs from psql -> decorate -> save to mongodb -> extended
    Repeated in batches of 20 (or specified batch size).
    """
    try:
        migration_svc = _get_migration_service()
        result = migration_svc.run_patient_migration(batch_size=batch_size, limit=limit)
        return PatientMigrationResponse(**result)
    except Exception as exc:
        logger.error("Patient migration failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Patient migration failed: {str(exc)}")


@api_router.post(
    "/cases/migrate/start",
    response_model=PatientMigrationActionResponse,
    summary="Start asynchronous patient migration in background thread",
)
def start_migrate_patients(
    batch_size: int = Query(20, ge=1, le=1000, description="Batch size for processing records (default: 20)"),
    limit: Optional[int] = Query(None, ge=1, description="Optional maximum number of records to migrate"),
) -> PatientMigrationActionResponse:
    """Start asynchronous background execution of the patient migration pipeline."""
    runner = get_patient_migration_runner()
    res = runner.start(batch_size=batch_size, limit=limit)
    if res.get("status") == "conflict":
        raise HTTPException(status_code=409, detail=res["message"])
    return PatientMigrationActionResponse(**res)


@api_router.post(
    "/cases/migrate/stop",
    response_model=PatientMigrationActionResponse,
    summary="Stop active patient migration pipeline",
)
def stop_migrate_patients() -> PatientMigrationActionResponse:
    """Request graceful stop of the active patient migration runner."""
    runner = get_patient_migration_runner()
    res = runner.stop()
    return PatientMigrationActionResponse(**res)


@api_router.get(
    "/cases/migrate/status",
    response_model=PatientMigrationStatusResponse,
    summary="Get status and metrics of patient migration runner",
)
def get_migrate_patients_status() -> PatientMigrationStatusResponse:
    """Get the current run status, progress, and step metrics of the patient migration."""
    runner = get_patient_migration_runner()
    status = runner.get_status()
    return PatientMigrationStatusResponse(**status)


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
# Rank Molecules by Similarity (LCA + Vector + Patient Age Group)
# ---------------------------------------------------------------------------

@api_router.post(
    "/cases/rank-meds",
    response_model=RankMoleculesResponse,
    summary="Rank top-k molecules by prescription frequency from similar cases",
)
def rank_medications(
    payload: RankMoleculesRequest,
) -> RankMoleculesResponse:
    """Rank molecules by doctor prescribing frequency and similarity score across similar cases.

    Pipeline:
      1. Clinical similarity matching: LCA (Wu-Palmer) + Vector (SapBERT dense cosine).
         Candidates below `min_similarity` cutoff (default 0.5) are filtered out.
      2. Post-match ranking: Matched cases are sorted by combined case similarity and
         patient demographic similarity (discrete age bands + Gaussian log-age kernel + gender match).
      3. Cumulative molecule scoring: Prescribed molecules accumulate case scores and are ranked.
    """
    try:
        return rank_meds(payload)
    except Exception as exc:
        logger.error("Rank meds failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Rank meds failed: {str(exc)}") from exc


# ---------------------------------------------------------------------------
# Prescribe Medications (Clinical Case Enrichment + Gemini LLM)
# ---------------------------------------------------------------------------

@api_router.post(
    "/cases/prescribe-medications",
    response_model=PrescribeMedicationsResponse,
    summary="Recommend prescribed medications for a clinical case using Gemini LLM",
)
def prescribe_medications(
    payload: PrescribeMedicationsRequest,
) -> PrescribeMedicationsResponse:
    """Enrich case with HPO/MONDO codes, retrieve past encounters and similar cases from PostgreSQL,
    and construct clinical prompt."""
    try:
        prescription_svc = _get_postgres_prescription_service()
        return prescription_svc.prescribe_medications(payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Prescribe medications failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prescribe medications failed: {str(exc)}") from exc


# ---------------------------------------------------------------------------
# Prescribe Medications V2 (Multi-Step Reasoning — per-entity matrix + deprescribing)
# ---------------------------------------------------------------------------

def _get_prescription_service_v2() -> PrescriptionServiceV2:
    """Lazy factory for PrescriptionServiceV2 — instantiated per request."""
    settings = get_settings()
    return PrescriptionServiceV2(settings=settings)


@api_router.post(
    "/cases/prescribe-medications/v2",
    response_model=PrescribeMedicationsV2Response,
    summary="V2: Multi-step medication reasoning with per-entity matrix and deprescribing",
)
def prescribe_medications_v2(
    payload: PrescribeMedicationsV2Request,
) -> PrescribeMedicationsV2Response:
    """Multi-step prescribing pipeline:

    1. Decorate the case (HPO / MONDO / SapBERT embeddings).
    2. Fetch patient past visits from MongoDB.
    3. Compute clinical delta (new / persistent / improving / resolved entities).
    4. Run per-entity similarity retrieval against the full MongoDB candidate pool.
    5. Build entity-to-molecule frequency matrix.
    6. For follow-up visits: call the deprescribing LLM on resolved/improving entities.
    7. Call the regimen synthesis LLM with the matrix + deprescribing plan + formulary.
    """
    try:
        svc = _get_prescription_service_v2()
        return svc.prescribe_medications_v2(payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Prescribe medications v2 failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prescribe medications v2 failed: {str(exc)}") from exc


# ---------------------------------------------------------------------------
# Medicine Master Ingestion & BioLORD Vector Embeddings
# ---------------------------------------------------------------------------

@api_router.post(
    "/medicines/load-master",
    response_model=LoadMedicineMasterResponse,
    summary="Load and embed medicine master dataset into MongoDB",
)
def load_medicine_master(
    payload: Optional[LoadMedicineMasterRequest] = None,
) -> LoadMedicineMasterResponse:
    """Ingest local_medicine_master.json, generate BioLORD embeddings for synthesized clinical sentences,
    and store in MongoDB 'medicine_master' in batches of 100."""
    try:
        req = payload or LoadMedicineMasterRequest()
        svc = _get_medicine_master_service()
        result = svc.load_and_embed_master(
            file_path=req.file_path,
            batch_size=req.batch_size,
            recreate=req.recreate,
        )
        return LoadMedicineMasterResponse(**result)
    except Exception as exc:
        logger.error("Medicine master loading failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Medicine master loading failed: {str(exc)}")


@api_router.post(
    "/biolord/embed",
    response_model=BioLORDEmbedResponse,
    summary="Generate BioLORD vector embeddings for clinical texts",
)
def generate_biolord_embeddings(
    payload: BioLORDEmbedRequest,
) -> BioLORDEmbedResponse:
    """Generate 768-d BioLORD dense vector representations using multi-core inference and mean pooling."""
    try:
        embedder = get_biolord_embedder()
        vectors = embedder.embed_texts(payload.texts, normalize=payload.normalize)
        return BioLORDEmbedResponse(
            texts=payload.texts,
            dimension=vectors.shape[1] if len(vectors) > 0 else 768,
            embeddings=vectors.tolist(),
        )
    except Exception as exc:
        logger.error("BioLORD embedding generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"BioLORD embedding error: {str(exc)}")


@api_router.post(
    "/medicines/search",
    response_model=MedicineSearchResponse,
    summary="Semantic search over medicine master collection using BioLORD embeddings",
)
def search_medicines(
    payload: MedicineSearchRequest,
) -> MedicineSearchResponse:
    """Search medicine master records by clinical symptom, molecule, or indication using BioLORD semantic cosine similarity."""
    try:
        svc = _get_medicine_master_service()
        matches = svc.search_similar_medicines(
            query=payload.query,
            top_k=payload.top_k,
            min_score=payload.min_score,
        )
        return MedicineSearchResponse(
            query=payload.query,
            total_matches=len(matches),
            matches=matches,
        )
    except Exception as exc:
        logger.error("Medicine search failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Medicine search error: {str(exc)}")


@api_router.post(
    "/cases/medication-search-queries",
    response_model=MedicationSearchQueryResponse,
    summary="Construct vector semantic queries and BM25 keywords for medication search using active LLM prompt",
)
def generate_medication_search_queries(
    payload: MedicationSearchQueryRequest,
) -> MedicationSearchQueryResponse:
    """Analyze patient symptoms and diagnosis to construct vector_semantic_queries and bm25_keywords via active_prompt.md."""
    try:
        gemini_svc = get_gemini_service()
        res = gemini_svc.generate_medication_search_queries(
            symptoms=payload.symptoms,
            diagnosis=payload.diagnosis,
            duration_context=payload.durationContext,
            current_regimen=payload.currentRegimen,
        )
        return MedicationSearchQueryResponse(
            vector_semantic_queries=res.get("vector_semantic_queries", []),
            bm25_keywords=res.get("bm25_keywords", []),
        )
    except Exception as exc:
        logger.error("Medication search query generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Medication search query generation failed: {str(exc)}") from exc


# ---------------------------------------------------------------------------
# Prescribe Pipeline (batch run all cases through predict_medications)
# ---------------------------------------------------------------------------

@api_router.post(
    "/cases/prescribe-pipeline/start",
    response_model=PrescribePipelineActionResponse,
    summary="Start the prescribe-medications batch pipeline in the background",
)
def start_prescribe_pipeline() -> PrescribePipelineActionResponse:
    """Kick off the background pipeline that runs prescribe_medications for every
    case in prescribe-request-all-cases.json and writes results to tmp/."""
    runner = get_prescribe_pipeline_runner()
    res = runner.start()
    if res.get("status") == "conflict":
        raise HTTPException(status_code=409, detail=res["message"])
    return PrescribePipelineActionResponse(
        status=res["status"],
        message=res["message"],
        run_id=res.get("run_id"),
        result_file=res.get("result_file"),
    )


@api_router.post(
    "/cases/prescribe-pipeline/stop",
    response_model=PrescribePipelineActionResponse,
    summary="Request graceful stop of the active prescribe-medications pipeline",
)
def stop_prescribe_pipeline() -> PrescribePipelineActionResponse:
    """Signal the running pipeline to stop after the current case finishes."""
    runner = get_prescribe_pipeline_runner()
    res = runner.stop()
    return PrescribePipelineActionResponse(
        status=res["status"],
        message=res["message"],
    )


@api_router.get(
    "/cases/prescribe-pipeline/status",
    response_model=PrescribePipelineStatusResponse,
    summary="Poll the current state and progress of the prescribe-medications pipeline",
)
def get_prescribe_pipeline_status() -> PrescribePipelineStatusResponse:
    """Returns progress counters, current case, elapsed time, and path to the result file."""
    runner = get_prescribe_pipeline_runner()
    return PrescribePipelineStatusResponse(**runner.get_status())
