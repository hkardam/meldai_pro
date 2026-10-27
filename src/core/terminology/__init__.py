"""Offline ontology terminology services — HPO and MONDO."""

from core.terminology.hpo import HPOResult, HPOService, get_hpo_service
from core.terminology.mondo import MONDOResult, MONDOService, get_mondo_service
from core.terminology.taxonomy import (
    TaxonomyIndex,
    get_hpo_taxonomy,
    get_mondo_taxonomy,
)

__all__ = [
    "HPOResult",
    "HPOService",
    "get_hpo_service",
    "MONDOResult",
    "MONDOService",
    "get_mondo_service",
    "TaxonomyIndex",
    "get_hpo_taxonomy",
    "get_mondo_taxonomy",
]
