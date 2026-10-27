"""Core FastAPI router — HTTP endpoints for terminology, NLP, embeddings, and clinical data migrations."""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from core.config import get_settings
from core.db.mongodb import MongoKnowledgeBase
from core.db.postgres import PostgresSource
from core.dtos.req_dtos import (
    BatchSymptomMatchRequest,
    BioLORDEmbedRequest,
    EmbedRequest,
    LoadMedicineMasterRequest,
    MedicationSearchQueryRequest,
    MedicineSearchRequest,
    RankMoleculesRequest,
    SegmentSymptomsRequest,
)
from core.dtos.res_dtos import (
    BatchSymptomMatchResponse,
    BioLORDEmbedResponse,
    EmbedResponse,
    LoadMedicineMasterResponse,
    MedicationSearchQueryResponse,
    MedicineSearchResponse,
    RankMoleculesResponse,
    SegmentSymptomsResponse,
    SymptomItem,
    SymptomMatchResult,
)
from core.nlp.biolord import get_biolord_embedder
from core.nlp.medspacy_nlp import get_medspacy_service
from core.nlp.sapbert import get_sapbert_embedder
from core.services.assertion_service import ClinicalAssertionService
from core.services.case_service import CaseService
from core.services.gemini_service import get_gemini_service
from core.services.medicine_master_service import MedicineMasterService
from core.services.symptom_service import SymptomService
from core.services.terminology_service import TerminologySearchService
from core.terminology.hpo import HPOResult, get_hpo_service
from core.terminology.mondo import MONDOResult, get_mondo_service
from meldai_v2.services.entity_matrix_service import rank_meds

logger = logging.getLogger(__name__)

core_router = APIRouter()


# ---------------------------------------------------------------------------
# Service Factories (wired to router dependencies for testing/mocking)
# ---------------------------------------------------------------------------

def _dep(name: str, fallback: Any) -> Any:
    """Dynamically resolve dependency from meldai.api.router or core.router (enables mock patching)."""
    import sys
    for mod_name in ("meldai.api.router", "core.router"):
        mod = sys.modules.get(mod_name)
        if mod and hasattr(mod, name):
            return getattr(mod, name)
    return fallback


def _get_terminology_service() -> TerminologySearchService:
    settings = get_settings()
    nlp = _dep("get_medspacy_service", get_medspacy_service)()
    hpo = _dep("get_hpo_service", get_hpo_service)(settings.hpo_obo_path)
    mondo = _dep("get_mondo_service", get_mondo_service)(settings.mondo_obo_path)
    assertion = ClinicalAssertionService(nlp_service=nlp)
    return TerminologySearchService(
        settings=settings,
        assertion_service=assertion,
        hpo_service=hpo,
        mondo_service=mondo,
    )


def _get_symptom_service() -> SymptomService:
    settings = get_settings()
    nlp = _dep("get_medspacy_service", get_medspacy_service)()
    hpo = _dep("get_hpo_service", get_hpo_service)(settings.hpo_obo_path)
    mondo = _dep("get_mondo_service", get_mondo_service)(settings.mondo_obo_path)
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
    nlp = _dep("get_medspacy_service", get_medspacy_service)()
    hpo = _dep("get_hpo_service", get_hpo_service)(settings.hpo_obo_path)
    mondo = _dep("get_mondo_service", get_mondo_service)(settings.mondo_obo_path)
    assertion = ClinicalAssertionService(nlp_service=nlp)
    embedder = _dep("get_sapbert_embedder", get_sapbert_embedder)(settings)
    return CaseService(
        settings=settings,
        assertion_service=assertion,
        hpo_service=hpo,
        mondo_service=mondo,
        embedder=embedder,
    )




def _get_medicine_master_service() -> MedicineMasterService:
    settings = get_settings()
    mongo = MongoKnowledgeBase(settings)
    biolord = _dep("get_biolord_embedder", get_biolord_embedder)(settings)
    return MedicineMasterService(
        settings=settings,
        mongo_kb=mongo,
        biolord_embedder=biolord,
    )


# ---------------------------------------------------------------------------
# System & Health
# ---------------------------------------------------------------------------

@core_router.get("/health")
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

@core_router.get("/hpo/search", response_model=List[HPOResult])
@core_router.get("/terminology/hpo/search", response_model=List[HPOResult], include_in_schema=False)
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


@core_router.get("/mondo/search", response_model=List[MONDOResult])
@core_router.get("/terminology/mondo/search", response_model=List[MONDOResult], include_in_schema=False)
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

@core_router.post("/embed", response_model=EmbedResponse)
def generate_embeddings(payload: EmbedRequest) -> EmbedResponse:
    """Generate 768-d SapBERT dense clinical embeddings for provided terms."""
    try:
        embedder = _dep("get_sapbert_embedder", get_sapbert_embedder)()
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

@core_router.post(
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
# Batch Symptom Ontology Matcher (medspaCy Pipe + HPO + MONDO)
# ---------------------------------------------------------------------------

@core_router.post(
    "/symptoms/match-batch",
    response_model=BatchSymptomMatchResponse,
    summary="Batch match clinical symptoms to top-K HPO and MONDO concepts",
)
def match_symptoms_batch(
    payload: BatchSymptomMatchRequest,
) -> BatchSymptomMatchResponse:
    """Batch process raw symptom strings through medspaCy assertion and return concept matches."""
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

@core_router.post(
    "/cases/rank-meds",
    response_model=RankMoleculesResponse,
    summary="Rank top-k molecules by prescription frequency from similar cases",
)
def rank_medications(
    payload: RankMoleculesRequest,
) -> RankMoleculesResponse:
    """Rank molecules by doctor prescribing frequency and similarity score across similar cases."""
    try:
        rank_fn = _dep("rank_meds", rank_meds)
        return rank_fn(payload)
    except Exception as exc:
        logger.error("Rank meds failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Rank meds failed: {str(exc)}") from exc


# ---------------------------------------------------------------------------
# Medicine Master Formulary Loader
# ---------------------------------------------------------------------------

@core_router.post(
    "/medicine-master/load",
    response_model=LoadMedicineMasterResponse,
    summary="Load and vector embed local medicine master records",
)
@core_router.post(
    "/medicines/load-master",
    response_model=LoadMedicineMasterResponse,
    summary="Load and vector embed local medicine master records (alias)",
    include_in_schema=False,
)
def load_medicine_master_records(
    payload: Optional[LoadMedicineMasterRequest] = None,
) -> LoadMedicineMasterResponse:
    """Load JSON records, generate BioLORD embeddings, and store in MongoDB 'medicine_master'."""
    try:
        req = payload or LoadMedicineMasterRequest()
        svc = _dep("_get_medicine_master_service", _get_medicine_master_service)()
        result = svc.load_and_embed_master(
            file_path=req.file_path,
            batch_size=req.batch_size,
            recreate=req.recreate,
        )
        return LoadMedicineMasterResponse(**result)
    except Exception as exc:
        logger.error("Medicine master loading failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Medicine master loading failed: {str(exc)}")


# ---------------------------------------------------------------------------
# BioLORD Embeddings
# ---------------------------------------------------------------------------

@core_router.post(
    "/biolord/embed",
    response_model=BioLORDEmbedResponse,
    summary="Generate BioLORD vector embeddings for clinical texts",
)
def generate_biolord_embeddings(
    payload: BioLORDEmbedRequest,
) -> BioLORDEmbedResponse:
    """Generate 768-d BioLORD dense vector representations using mean pooling."""
    try:
        embedder = _dep("get_biolord_embedder", get_biolord_embedder)()
        vectors = embedder.embed_texts(payload.texts, normalize=payload.normalize)
        return BioLORDEmbedResponse(
            texts=payload.texts,
            dimension=vectors.shape[1] if len(vectors) > 0 else 768,
            embeddings=vectors.tolist(),
        )
    except Exception as exc:
        logger.error("BioLORD embedding generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"BioLORD embedding error: {str(exc)}")


# ---------------------------------------------------------------------------
# Medicine Semantic Search
# ---------------------------------------------------------------------------

@core_router.post(
    "/medicines/search",
    response_model=MedicineSearchResponse,
    summary="Semantic search over medicine master collection using BioLORD embeddings",
)
def search_medicines(
    payload: MedicineSearchRequest,
) -> MedicineSearchResponse:
    """Search medicine master records by clinical symptom, molecule, or indication."""
    try:
        svc = _dep("_get_medicine_master_service", _get_medicine_master_service)()
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


# ---------------------------------------------------------------------------
# LLM Medication Search Queries
# ---------------------------------------------------------------------------

@core_router.post(
    "/cases/medication-search-queries",
    response_model=MedicationSearchQueryResponse,
    summary="Construct vector semantic queries and BM25 keywords for medication search",
)
def generate_medication_search_queries(
    payload: MedicationSearchQueryRequest,
) -> MedicationSearchQueryResponse:
    """Analyze patient symptoms and diagnosis to construct vector_semantic_queries and bm25_keywords."""
    try:
        gemini_svc = _dep("get_gemini_service", get_gemini_service)()
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
