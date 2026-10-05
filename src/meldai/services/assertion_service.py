"""Clinical assertion service — extracts negation, uncertainty, and temporality via medspaCy."""

import logging
from dataclasses import dataclass
from typing import List, Optional

from meldai.nlp.medspacy_nlp import MedspaCyService, get_medspacy_service

logger = logging.getLogger(__name__)


@dataclass
class AssertionResult:
    """Clinical assertion result for a medical term or phrase."""
    term: str
    search_target: str
    is_negated: bool
    is_phenotype: bool
    assertion_status: str  # "affirmed", "negated", "historical", "uncertain", "family"


class ClinicalAssertionService:
    """Extracts clinical assertions and entity targets using medspaCy's ConText framework."""

    def __init__(self, nlp_service: Optional[MedspaCyService] = None) -> None:
        self._nlp_service = nlp_service

    def _get_nlp(self) -> MedspaCyService:
        if self._nlp_service is not None:
            return self._nlp_service
        return get_medspacy_service()

    def _doc_to_assertion(self, doc, clean_term: str) -> AssertionResult:
        is_negated = False
        is_historical = False
        is_uncertain = False
        is_family = False
        search_target = clean_term

        # Document-level context graph modifiers
        doc_categories = set()
        context_graph = getattr(doc._, "context_graph", None)
        if context_graph and hasattr(context_graph, "modifiers"):
            doc_categories = {m.category for m in context_graph.modifiers}

        # Entity-level context attributes
        ent_negated = any(getattr(e._, "is_negated", False) for e in doc.ents)
        ent_historical = any(getattr(e._, "is_historical", False) for e in doc.ents)
        ent_uncertain = any(getattr(e._, "is_uncertain", False) for e in doc.ents)
        ent_family = any(getattr(e._, "is_family", False) for e in doc.ents)

        is_negated = "NEGATED_EXISTENCE" in doc_categories or ent_negated
        is_historical = "HISTORICAL" in doc_categories or ent_historical
        is_uncertain = bool(doc_categories.intersection({"POSSIBLE_EXISTENCE", "HYPOTHETICAL"})) or ent_uncertain
        is_family = "FAMILY" in doc_categories or ent_family

        # Best entity match for ontology lookup (longest entity text, fallback to raw term)
        if doc.ents:
            search_target = max((e.text for e in doc.ents), key=len)

        is_phenotype = not is_negated and not is_historical and not is_uncertain and not is_family

        if is_negated:
            assertion_status = "negated"
        elif is_historical:
            assertion_status = "historical"
        elif is_uncertain:
            assertion_status = "uncertain"
        elif is_family:
            assertion_status = "family"
        else:
            assertion_status = "affirmed"

        return AssertionResult(
            term=clean_term,
            search_target=search_target,
            is_negated=is_negated,
            is_phenotype=is_phenotype,
            assertion_status=assertion_status,
        )

    def analyze(self, term: str) -> AssertionResult:
        """Analyze a clinical term to determine negation, temporality, and primary entity."""
        clean_term = term.strip()
        try:
            nlp = self._get_nlp()
            doc = nlp.analyze(clean_term)
            return self._doc_to_assertion(doc, clean_term)
        except Exception as exc:
            logger.warning("Clinical assertion analysis failed for '%s': %s", term, exc)
            return AssertionResult(
                term=clean_term,
                search_target=clean_term,
                is_negated=False,
                is_phenotype=True,
                assertion_status="affirmed",
            )

    def analyze_batch(self, terms: List[str], batch_size: int = 64) -> List[AssertionResult]:
        """Batch analyze clinical terms using medspaCy's batch pipeline."""
        if not terms:
            return []
        clean_terms = [t.strip() for t in terms]
        nlp = self._get_nlp()
        if hasattr(nlp, "analyze_batch"):
            try:
                docs = nlp.analyze_batch(clean_terms, batch_size=batch_size)
                return [self._doc_to_assertion(doc, clean_terms[i]) for i, doc in enumerate(docs)]
            except Exception as exc:
                logger.warning("Batch assertion failed, falling back to sequential: %s", exc)

        return [self.analyze(term) for term in clean_terms]

