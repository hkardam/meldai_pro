"""FastAPI router — HPO search, MONDO search, SapBERT embeddings."""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from meldai.config import get_settings
from meldai.db.postgres import PostgresSource
from meldai.db.mongodb import MongoKnowledgeBase
from meldai.nlp.sapbert import SapBERTEmbedder
from meldai.terminology.hpo import HPOResult, get_hpo_service
from meldai.terminology.mondo import MONDOResult, get_mondo_service

api_router = APIRouter(prefix="/api/v1")


# ---------------------------------------------------------------------------
# Embed request / response models
# ---------------------------------------------------------------------------

class EmbedRequest(BaseModel):
    terms: List[str]


class EmbedResponse(BaseModel):
    terms: List[str]
    dimension: int
    embeddings: List[List[float]]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@api_router.get("/health")
def health_check() -> Dict[str, Any]:
    """Check connectivity to PostgreSQL and MongoDB."""
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
# HPO search
# ---------------------------------------------------------------------------

@api_router.get("/hpo/search", response_model=List[HPOResult])
def search_hpo(
    term: str = Query(..., min_length=2, description="Phenotype / symptom term to look up"),
    limit: int = Query(5, ge=1, le=50, description="Maximum number of results"),
) -> List[HPOResult]:
    """Search the Human Phenotype Ontology (HPO) for a given term.

    Returns matching HPO concepts sorted by match quality
    (exact label → synonym → substring).
    """
    settings = get_settings()
    service = get_hpo_service(settings.hpo_obo_path)
    try:
        results = service.search(term, limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if not results:
        raise HTTPException(status_code=404, detail=f"No HPO concept found for '{term}'")
    return results


# ---------------------------------------------------------------------------
# MONDO search
# ---------------------------------------------------------------------------

@api_router.get("/mondo/search", response_model=List[MONDOResult])
def search_mondo(
    term: str = Query(..., min_length=2, description="Disease / disorder term to look up"),
    limit: int = Query(5, ge=1, le=50, description="Maximum number of results"),
) -> List[MONDOResult]:
    """Search the MONDO Disease Ontology for a given term.

    Returns matching MONDO concepts sorted by match quality
    (exact label → synonym → substring).
    """
    settings = get_settings()
    service = get_mondo_service(settings.mondo_obo_path)
    try:
        results = service.search(term, limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if not results:
        raise HTTPException(status_code=404, detail=f"No MONDO concept found for '{term}'")
    return results


# ---------------------------------------------------------------------------
# SapBERT embeddings
# ---------------------------------------------------------------------------

@api_router.post("/embed", response_model=EmbedResponse)
def generate_embeddings(payload: EmbedRequest) -> EmbedResponse:
    """Generate dense SapBERT clinical embeddings for given medical phrases."""
    embedder = SapBERTEmbedder()
    vectors = embedder.embed_entities(payload.terms)
    return EmbedResponse(
        terms=payload.terms,
        dimension=vectors.shape[1] if len(vectors) > 0 else 768,
        embeddings=vectors.tolist(),
    )


# ---------------------------------------------------------------------------
# Migration / Data Ingestion API
# ---------------------------------------------------------------------------

class PatientVisitMigrationResponse(BaseModel):
    status: str
    batch_size: int
    total_records_processed: int
    batches_processed: int
    upserted_count: int
    modified_count: int
    matched_count: int


@api_router.post("/cases/migrate-patient-visits", response_model=PatientVisitMigrationResponse)
def migrate_patient_visits(
    batch_size: int = Query(1000, ge=1, le=10000, description="Batch size for extracting and pushing records"),
) -> PatientVisitMigrationResponse:
    """
    Pull patient visit data from PostgreSQL (temp_migrations.patient_visit_data)
    in batches of `batch_size` (default 1000) and upsert/patch into MongoDB 'cases' collection
    using a unique compound index on (caseNo, visitDate).
    """
    settings = get_settings()
    pg = PostgresSource(settings)
    mongo = MongoKnowledgeBase(settings)

    # Initialize / verify unique compound index on (caseNo, visitDate)
    mongo.setup_patient_visits_index()

    total_processed = 0
    batches_count = 0
    total_upserted = 0
    total_modified = 0
    total_matched = 0

    try:
        for batch in pg.stream_patient_visit_data(batch_size=batch_size):
            if not batch:
                continue
            res = mongo.upsert_patient_visits_batch(batch)
            total_processed += len(batch)
            batches_count += 1
            total_upserted += res.get("upserted_count", 0)
            total_modified += res.get("modified_count", 0)
            total_matched += res.get("matched_count", 0)

        return PatientVisitMigrationResponse(
            status="success",
            batch_size=batch_size,
            total_records_processed=total_processed,
            batches_processed=batches_count,
            upserted_count=total_upserted,
            modified_count=total_modified,
            matched_count=total_matched,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Patient visit migration failed: {str(exc)}")


# ---------------------------------------------------------------------------
# Patient Diagnosis Migration API
# ---------------------------------------------------------------------------

class PatientDiagnosisMigrationResponse(BaseModel):
    status: str
    batch_size: int
    total_encounters_processed: int
    batches_processed: int
    modified_count: int
    matched_count: int
    unique_terms_indexed: int
    execution_time_seconds: float


@api_router.post("/cases/migrate-patient-diagnoses", response_model=PatientDiagnosisMigrationResponse)
def migrate_patient_diagnoses(
    batch_size: int = Query(1000, ge=1, le=10000, description="Batch size for extracting and pushing records"),
) -> PatientDiagnosisMigrationResponse:
    """
    Pull patient diagnosis data grouped by (caseNo, visitDate) from PostgreSQL
    (temp_migrations.patient_diagnosis_data) using ARRAY_AGG(DISTINCT "Diagnosis Name").
    Resolves MONDO codes and SapBERT clinical embeddings (cached in memory)
    and replaces the 'diagnosis' array on matching MongoDB 'cases' documents.
    """
    import re
    import time

    start_time = time.time()
    settings = get_settings()
    pg = PostgresSource(settings)
    mongo = MongoKnowledgeBase(settings)
    mondo_service = get_mondo_service(settings.mondo_obo_path)
    embedder = SapBERTEmbedder()

    term_cache: Dict[str, Dict[str, Any]] = {}

    def _ensure_terms_cached(terms: List[str]) -> None:
        uncached = list(dict.fromkeys([t.strip() for t in terms if t and t.strip() and t.strip() not in term_cache]))
        if not uncached:
            return

        # 1. MONDO lookups
        mondo_codes: Dict[str, Optional[int]] = {}
        for t in uncached:
            code = None
            try:
                res = mondo_service.search(t, limit=1)
                if res:
                    m = re.search(r"\d+", res[0].mondo_id)
                    if m:
                        code = int(m.group())
            except Exception:
                code = None
            mondo_codes[t] = code

        # 2. Batch SapBERT embedding
        vectors = embedder.embed_entities(uncached)
        for i, t in enumerate(uncached):
            emb = vectors[i].tolist() if i < len(vectors) else []
            term_cache[t] = {
                "term": t,
                "mondoCode": mondo_codes[t],
                "embedding": emb,
            }

    total_encounters = 0
    batches_count = 0
    total_modified = 0
    total_matched = 0

    try:
        for batch in pg.stream_grouped_patient_diagnoses(batch_size=batch_size):
            if not batch:
                continue

            # Extract terms and batch cache
            batch_terms = [t for enc in batch for t in enc.get("diagnosisNames", []) if t and t.strip()]
            _ensure_terms_cached(batch_terms)

            # Build mongo update payload
            mongo_batch = []
            for enc in batch:
                case_no = enc["caseNo"]
                visit_date = enc["visitDate"]
                diag_names = enc.get("diagnosisNames", [])
                diag_payloads = [term_cache[name.strip()] for name in diag_names if name and name.strip() in term_cache]
                mongo_batch.append({
                    "caseNo": case_no,
                    "visitDate": visit_date,
                    "diagnosis": diag_payloads,
                })

            res = mongo.replace_case_diagnoses_batch(mongo_batch)
            total_encounters += len(batch)
            batches_count += 1
            total_modified += res.get("modified_count", 0)
            total_matched += res.get("matched_count", 0)

        exec_time = round(time.time() - start_time, 3)

        return PatientDiagnosisMigrationResponse(
            status="success",
            batch_size=batch_size,
            total_encounters_processed=total_encounters,
            batches_processed=batches_count,
            modified_count=total_modified,
            matched_count=total_matched,
            unique_terms_indexed=len(term_cache),
            execution_time_seconds=exec_time,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Patient diagnosis migration failed: {str(exc)}")


