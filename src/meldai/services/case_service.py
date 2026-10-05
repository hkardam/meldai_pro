"""Case service — handles case decoration and similar case retrieval."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from meldai.api.req_dtos import FindSimilarCaseRequest, PatientInfoInput
from meldai.api.res_dtos import (
    DecoratedCase,
    DecoratedDiagnosisItem,
    DecoratedPatientInfo,
    DecoratedSymptomItem,
    FindSimilarCaseResponse,
)
from meldai.config import Settings, get_settings
from meldai.nlp.sapbert import SapBERTEmbedder, get_sapbert_embedder
from meldai.services.assertion_service import ClinicalAssertionService
from meldai.terminology.hpo import HPOService, get_hpo_service
from meldai.terminology.mondo import MONDOService, get_mondo_service

logger = logging.getLogger(__name__)


class CaseService:
    """Clinical case enrichment and similarity matching service."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        assertion_service: Optional[ClinicalAssertionService] = None,
        hpo_service: Optional[HPOService] = None,
        mondo_service: Optional[MONDOService] = None,
        embedder: Optional[SapBERTEmbedder] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._assertion_service = assertion_service or ClinicalAssertionService()
        self._hpo_service = hpo_service
        self._mondo_service = mondo_service
        self._embedder = embedder

    def _get_hpo(self) -> HPOService:
        if self._hpo_service is not None:
            return self._hpo_service
        return get_hpo_service(self._settings.hpo_obo_path)

    def _get_mondo(self) -> MONDOService:
        if self._mondo_service is not None:
            return self._mondo_service
        return get_mondo_service(self._settings.mondo_obo_path)

    def _get_embedder(self) -> SapBERTEmbedder:
        if self._embedder is not None:
            return self._embedder
        return get_sapbert_embedder(self._settings)

    def decorate_case(
        self,
        case_no: Optional[int],
        patient_info: Optional[PatientInfoInput],
        symptoms: List[str],
        diagnosis: List[str],
    ) -> DecoratedCase:
        """Decorate a clinical case by resolving symptoms to HPO, diagnoses to MONDO, and generating per-term dense embeddings."""
        # 1. Process symptoms
        decorated_symptoms: List[DecoratedSymptomItem] = []
        if symptoms:
            clean_symptoms = [s.strip() for s in symptoms if s and s.strip()]
            if clean_symptoms:
                assertions = self._assertion_service.analyze_batch(clean_symptoms)
                hpo = self._get_hpo()
                embedder = self._get_embedder()
                hpo_cache: Dict[str, Tuple[Optional[str], Optional[int]]] = {}

                resolved_assertions = []
                symptom_terms_to_embed = []

                for assertion in assertions:
                    target = assertion.search_target
                    raw_text = assertion.term
                    symptom_terms_to_embed.append(raw_text)

                    if target in hpo_cache:
                        hpo_term, hpo_code = hpo_cache[target]
                    else:
                        hits = hpo.search(target, limit=1)
                        if not hits and target != raw_text:
                            hits = hpo.search(raw_text, limit=1)
                        if hits:
                            hit = hits[0]
                            hpo_term = hit.label
                            m = re.search(r"\d+", hit.hpo_id)
                            hpo_code = int(m.group()) if m else None
                        else:
                            hpo_term = None
                            hpo_code = None
                        hpo_cache[target] = (hpo_term, hpo_code)

                    resolved_assertions.append((assertion, hpo_term, hpo_code))

                # Batch embed symptom terms
                vectors = embedder.embed_entities(symptom_terms_to_embed)

                for i, (assertion, hpo_term, hpo_code) in enumerate(resolved_assertions):
                    emb = vectors[i].tolist() if i < len(vectors) else None
                    decorated_symptoms.append(
                        DecoratedSymptomItem(
                            term=assertion.term,
                            hpoTerm=hpo_term,
                            hpoCode=hpo_code,
                            embedding=emb,
                            isNegation=assertion.is_negated,
                            isPheno=assertion.is_phenotype,
                            similarity=None,
                        )
                    )

        # 2. Process diagnoses
        decorated_diagnoses: List[DecoratedDiagnosisItem] = []
        if diagnosis:
            clean_diagnoses = [d.strip() for d in diagnosis if d and d.strip()]
            if clean_diagnoses:
                mondo = self._get_mondo()
                embedder = self._get_embedder()
                mondo_cache: Dict[str, Tuple[Optional[str], Optional[int]]] = {}

                resolved_diagnoses = []
                for d_text in clean_diagnoses:
                    if d_text in mondo_cache:
                        mondo_term, mondo_code = mondo_cache[d_text]
                    else:
                        hits = mondo.search(d_text, limit=1)
                        if not hits:
                            assertion = self._assertion_service.analyze(d_text)
                            if assertion.search_target != d_text:
                                hits = mondo.search(assertion.search_target, limit=1)
                        if hits:
                            hit = hits[0]
                            mondo_term = hit.label
                            m = re.search(r"\d+", hit.mondo_id)
                            mondo_code = int(m.group()) if m else None
                        else:
                            mondo_term = None
                            mondo_code = None
                        mondo_cache[d_text] = (mondo_term, mondo_code)

                    resolved_diagnoses.append((d_text, mondo_term, mondo_code))

                # Batch embed diagnosis terms
                vectors = embedder.embed_entities(clean_diagnoses)

                for i, (d_text, mondo_term, mondo_code) in enumerate(resolved_diagnoses):
                    emb = vectors[i].tolist() if i < len(vectors) else None
                    decorated_diagnoses.append(
                        DecoratedDiagnosisItem(
                            term=d_text,
                            mondoTerm=mondo_term,
                            mondoCode=mondo_code,
                            embedding=emb,
                            similarity=None,
                        )
                    )

        p_info = DecoratedPatientInfo(
            age=patient_info.age if patient_info else None,
            gender=patient_info.gender if patient_info else None,
        )

        return DecoratedCase(
            caseNo=case_no,
            patientInfo=p_info,
            symptoms=decorated_symptoms,
            diagnosis=decorated_diagnoses,
        )

    def find_similar_cases(self, request: FindSimilarCaseRequest) -> FindSimilarCaseResponse:
        """Enrich the input case and query for similar cases."""
        decorated_case = self.decorate_case(
            case_no=request.caseNo,
            patient_info=request.patientInfo,
            symptoms=request.symptoms,
            diagnosis=request.diagnosis,
        )
        return FindSimilarCaseResponse(
            decoratedCase=decorated_case,
            similarCases=[],
        )
