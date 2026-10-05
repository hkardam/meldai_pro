"""Terminology search service — coordinates ontology search with clinical assertion."""

import logging
from typing import List, Optional

from meldai.config import Settings, get_settings
from meldai.services.assertion_service import ClinicalAssertionService
from meldai.terminology.hpo import HPOResult, HPOService, get_hpo_service
from meldai.terminology.mondo import MONDOResult, MONDOService, get_mondo_service

logger = logging.getLogger(__name__)


class TerminologySearchService:
    """Provides clinical assertion-aware ontology search for HPO and MONDO."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        assertion_service: Optional[ClinicalAssertionService] = None,
        hpo_service: Optional[HPOService] = None,
        mondo_service: Optional[MONDOService] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._assertion_service = assertion_service or ClinicalAssertionService()
        self._hpo_service = hpo_service
        self._mondo_service = mondo_service

    def _get_hpo(self) -> HPOService:
        if self._hpo_service is not None:
            return self._hpo_service
        return get_hpo_service(self._settings.hpo_obo_path)

    def _get_mondo(self) -> MONDOService:
        if self._mondo_service is not None:
            return self._mondo_service
        return get_mondo_service(self._settings.mondo_obo_path)

    def search_hpo(
        self,
        term: str,
        limit: int = 5,
        filter_negated: bool = False,
    ) -> List[HPOResult]:
        """Search Human Phenotype Ontology (HPO) for a clinical term with assertion tags."""
        # 1. Analyze assertion and extract targeted clinical entity
        assertion = self._assertion_service.analyze(term)

        # 2. Return empty results if negated and filtering is requested
        if assertion.is_negated and filter_negated:
            return []

        # 3. Ontology query (try specific entity target first, fallback to raw term)
        hpo = self._get_hpo()
        results = hpo.search(assertion.search_target, limit=limit)
        if not results and assertion.search_target != term:
            results = hpo.search(term, limit=limit)

        # 4. Attach assertion metadata to all matched results
        for r in results:
            r.is_negated = assertion.is_negated
            r.is_phenotype = assertion.is_phenotype
            r.assertion_status = assertion.assertion_status

        return results

    def search_mondo(
        self,
        term: str,
        limit: int = 5,
        filter_negated: bool = False,
    ) -> List[MONDOResult]:
        """Search MONDO Disease Ontology for a disease term with assertion tags."""
        # 1. Analyze assertion and extract targeted clinical entity
        assertion = self._assertion_service.analyze(term)

        # 2. Return empty results if negated and filtering is requested
        if assertion.is_negated and filter_negated:
            return []

        # 3. Ontology query (try specific entity target first, fallback to raw term)
        mondo = self._get_mondo()
        results = mondo.search(assertion.search_target, limit=limit)
        if not results and assertion.search_target != term:
            results = mondo.search(term, limit=limit)

        # 4. Attach assertion metadata to all matched results
        for r in results:
            r.is_negated = assertion.is_negated
            r.is_phenotype = assertion.is_phenotype
            r.assertion_status = assertion.assertion_status

        return results
