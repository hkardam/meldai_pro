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
