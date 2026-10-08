"""Case service — handles case decoration and similar case retrieval."""

from datetime import datetime, timezone
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from meldai.api.req_dtos import FindSimilarCaseRequest, PatientInfoInput
from meldai.api.res_dtos import (
    ComponentScores,
    DecoratedCase,
    DecoratedDiagnosisItem,
    DecoratedPatientInfo,
    DecoratedSymptomItem,
    FindSimilarCaseResponse,
    SimilarCaseItem,
)
from meldai.config import Settings, get_settings
from meldai.db.mongodb import MongoKnowledgeBase
from meldai.nlp.sapbert import SapBERTEmbedder, get_sapbert_embedder
from meldai.services.assertion_service import ClinicalAssertionService
from meldai.services.similarity_engine import (
    EntityItem,
    compute_composite_score,
    patient_similarity,
    set_similarity,
    visit_type_similarity,
)
from meldai.terminology.hpo import HPOService, get_hpo_service
from meldai.terminology.mondo import MONDOService, get_mondo_service
from meldai.terminology.taxonomy import TaxonomyIndex, get_hpo_taxonomy, get_mondo_taxonomy

logger = logging.getLogger(__name__)

ABSTAIN_THRESHOLD: float = 0.70


def _check_score(hit: Any, threshold: float) -> bool:
    """Safely verify if hit matches above threshold, rejecting definition matches."""
    if hit is None:
        return False
    score = getattr(hit, "score", None)
    match_type = str(getattr(hit, "match_type", ""))
    if match_type.endswith("_definition"):
        return False
    if score is None:
        return True
    try:
        return float(score) >= threshold
    except (TypeError, ValueError):
        return True


class CaseService:
    """Clinical case enrichment and similarity matching service."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        assertion_service: Optional[ClinicalAssertionService] = None,
        hpo_service: Optional[HPOService] = None,
        mondo_service: Optional[MONDOService] = None,
        embedder: Optional[SapBERTEmbedder] = None,
        hpo_taxonomy: Optional[TaxonomyIndex] = None,
        mondo_taxonomy: Optional[TaxonomyIndex] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._assertion_service = assertion_service or ClinicalAssertionService()
        self._hpo_service = hpo_service
        self._mondo_service = mondo_service
        self._embedder = embedder
        self._hpo_taxonomy = hpo_taxonomy
        self._mondo_taxonomy = mondo_taxonomy

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

    def _get_hpo_taxonomy(self) -> Optional[TaxonomyIndex]:
        if self._hpo_taxonomy is not None:
            return self._hpo_taxonomy
        try:
            return get_hpo_taxonomy(self._settings.hpo_obo_path)
        except Exception as exc:
            logger.warning("HPO taxonomy loading failed: %s", exc)
            return None

    def _get_mondo_taxonomy(self) -> Optional[TaxonomyIndex]:
        if self._mondo_taxonomy is not None:
            return self._mondo_taxonomy
        try:
            return get_mondo_taxonomy(self._settings.mondo_obo_path)
        except Exception as exc:
            logger.warning("MONDO taxonomy loading failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Case decoration
    # ------------------------------------------------------------------

    def decorate_case(
        self,
        case_no: Optional[int],
        patient_info: Optional[PatientInfoInput],
        symptoms: List[str],
        diagnosis: List[str],
        visit_type: Optional[str] = None,
    ) -> DecoratedCase:
        """Decorate a clinical case by resolving symptoms to HPO, diagnoses to MONDO,
        and generating per-term dense SapBERT embeddings."""
        from meldai.utils import to_curie

        # 1. Process symptoms
        decorated_symptoms: List[DecoratedSymptomItem] = []
        if symptoms:
            clean_symptoms = [s.strip() for s in symptoms if s and s.strip()]
            if clean_symptoms:
                assertions = self._assertion_service.analyze_batch(clean_symptoms)
                hpo = self._get_hpo()
                embedder = self._get_embedder()
                hpo_cache: Dict[str, Tuple[Optional[str], Optional[int], Optional[str]]] = {}

                resolved_assertions = []
                symptom_terms_to_embed = []

                for assertion in assertions:
                    target = assertion.search_target
                    raw_text = assertion.term
                    symptom_terms_to_embed.append(raw_text)

                    if target in hpo_cache:
                        hpo_term, hpo_code, hpo_curie = hpo_cache[target]
                    else:
                        hits = hpo.search(target, limit=1)
                        if not hits and target != raw_text:
                            hits = hpo.search(raw_text, limit=1)

                        hit = hits[0] if hits else None

                        # Abstain threshold: reject low confidence or definition matches
                        if hit and _check_score(hit, ABSTAIN_THRESHOLD):
                            hpo_term = str(hit.label) if hasattr(hit, "label") and isinstance(hit.label, str) else None
                            hit_id = hit.hpo_id if hasattr(hit, "hpo_id") and isinstance(hit.hpo_id, str) else None
                            hpo_curie = to_curie("HP", hit_id)
                            m = re.search(r"\d+", hit_id) if hit_id else None
                            hpo_code = int(m.group()) if m else None
                        else:
                            hpo_term = None
                            hpo_code = None
                            hpo_curie = None
                        hpo_cache[target] = (hpo_term, hpo_code, hpo_curie)

                    resolved_assertions.append((assertion, hpo_term, hpo_code, hpo_curie))

                # Batch embed all symptom terms in a single SapBERT call
                vectors = embedder.embed_entities(symptom_terms_to_embed)

                for i, (assertion, hpo_term, hpo_code, hpo_curie) in enumerate(resolved_assertions):
                    emb = vectors[i].tolist() if i < len(vectors) else None
                    decorated_symptoms.append(
                        DecoratedSymptomItem(
                            term=assertion.term,
                            hpoTerm=hpo_term,
                            hpoCode=hpo_code,
                            hpoCurie=hpo_curie,
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
                mondo_cache: Dict[str, Tuple[Optional[str], Optional[int], Optional[str]]] = {}

                resolved_diagnoses = []
                for d_text in clean_diagnoses:
                    if d_text in mondo_cache:
                        mondo_term, mondo_code, mondo_curie = mondo_cache[d_text]
                    else:
                        hits = mondo.search(d_text, limit=1)
                        if not hits:
                            assertion = self._assertion_service.analyze(d_text)
                            if assertion.search_target != d_text:
                                hits = mondo.search(assertion.search_target, limit=1)

                        hit = hits[0] if hits else None

                        # Abstain threshold: reject low confidence or definition matches
                        if hit and _check_score(hit, ABSTAIN_THRESHOLD):
                            mondo_term = str(hit.label) if hasattr(hit, "label") and isinstance(hit.label, str) else None
                            hit_id = hit.mondo_id if hasattr(hit, "mondo_id") and isinstance(hit.mondo_id, str) else None
                            mondo_curie = to_curie("MONDO", hit_id)
                            m = re.search(r"\d+", hit_id) if hit_id else None
                            mondo_code = int(m.group()) if m else None
                        else:
                            mondo_term = None
                            mondo_code = None
                            mondo_curie = None
                        mondo_cache[d_text] = (mondo_term, mondo_code, mondo_curie)

                    resolved_diagnoses.append((d_text, mondo_term, mondo_code, mondo_curie))

                # Batch embed all diagnosis terms in a single SapBERT call
                vectors = embedder.embed_entities(clean_diagnoses)

                for i, (d_text, mondo_term, mondo_code, mondo_curie) in enumerate(resolved_diagnoses):
                    emb = vectors[i].tolist() if i < len(vectors) else None
                    decorated_diagnoses.append(
                        DecoratedDiagnosisItem(
                            term=d_text,
                            mondoTerm=mondo_term,
                            mondoCode=mondo_code,
                            mondoCurie=mondo_curie,
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
            visitType=visit_type,
            patientInfo=p_info,
            symptoms=decorated_symptoms,
            diagnosis=decorated_diagnoses,
        )

    # ------------------------------------------------------------------
    # Past cases
    # ------------------------------------------------------------------

    def _fetch_past_cases(
        self,
        case_no: int,
        case_date: str,
        mongo_kb: MongoKnowledgeBase,
    ) -> List[Dict[str, Any]]:
        """Query MongoDB for all visits of caseNo with visitDate < case_date."""
        try:
            docs = mongo_kb.find_past_cases(case_no=case_no, before_date=case_date)
            logger.info(
                "Past cases for caseNo=%s before %s: %d found",
                case_no, case_date, len(docs),
            )
            return docs
        except Exception as exc:
            logger.warning("Past case query failed for caseNo=%s: %s", case_no, exc)
            return []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def find_similar_cases(
        self,
        request: FindSimilarCaseRequest,
        mongo_kb: Optional[MongoKnowledgeBase] = None,
    ) -> FindSimilarCaseResponse:
        """Enrich the input case, fetch past visits, and retrieve & score similar cases."""

        case_input = request.case
        visit_type_str = case_input.visitType.value if hasattr(case_input.visitType, "value") else str(case_input.visitType)

        decorated_case = self.decorate_case(
            case_no=case_input.caseNo,
            patient_info=case_input.patientInfo,
            symptoms=case_input.symptoms,
            diagnosis=case_input.diagnosis,
            visit_type=visit_type_str,
        )

        # 1. Fetch past cases when caseNo is not null (defaults to today's UTC date if caseDate omitted)
        past_cases: List[Dict[str, Any]] = []
        if case_input.caseNo is not None and mongo_kb is not None:
            effective_case_date = case_input.caseDate or datetime.now(timezone.utc).strftime("%Y-%m-%d")
            past_cases = self._fetch_past_cases(
                case_no=case_input.caseNo,
                case_date=effective_case_date,
                mongo_kb=mongo_kb,
            )

        # 2. Retrieve candidates and score similar cases
        similar_cases: List[SimilarCaseItem] = []
        if mongo_kb is not None:
            required_vt = visit_type_str if request.params.sameVisitTypeOnly else None
            candidates = mongo_kb.find_similar_candidates(
                exclude_case_no=case_input.caseNo,
                required_visit_type=required_vt,
                limit=500,
            )

            # Prepare query entities for set comparison (using pure integer codes)
            query_symptoms = [
                EntityItem(term=s.term, embedding=s.embedding, ontology_code=s.hpoCode)
                for s in decorated_case.symptoms
            ]
            query_diagnoses = [
                EntityItem(term=d.term, embedding=d.embedding, ontology_code=d.mondoCode)
                for d in decorated_case.diagnosis
            ]

            hpo_taxonomy = self._get_hpo_taxonomy()
            mondo_taxonomy = self._get_mondo_taxonomy()

            query_age = case_input.patientInfo.age if case_input.patientInfo else None

            # Pre-normalize query visit type once outside the loop
            from meldai.services.similarity_engine import normalize_visit_type
            norm_query_vt = normalize_visit_type(visit_type_str)

            def _parse_hpo_code(item: Dict[str, Any]) -> Optional[int]:
                code = item.get("hpoCode")
                if isinstance(code, int):
                    return code
                if code is not None and str(code).isdigit():
                    return int(code)
                hpo_id = item.get("hpoId")
                if hpo_id and isinstance(hpo_id, str):
                    digits = "".join(filter(str.isdigit, hpo_id))
                    return int(digits) if digits else None
                return None

            def _parse_mondo_code(item: Dict[str, Any]) -> Optional[int]:
                code = item.get("mondoCode")
                if isinstance(code, int):
                    return code
                if code is not None and str(code).isdigit():
                    return int(code)
                mondo_id = item.get("mondoId")
                if mondo_id and isinstance(mondo_id, str):
                    digits = "".join(filter(str.isdigit, mondo_id))
                    return int(digits) if digits else None
                return None

            scored_candidates = []
            for cand in candidates:
                # Extract candidate symptoms with native int hpoCode
                cand_symptoms: List[EntityItem] = []
                for s in cand.get("symptoms", []):
                    if isinstance(s, dict):
                        cand_symptoms.append(
                            EntityItem(
                                term=s.get("term") or s.get("text") or "",
                                embedding=s.get("embedding"),
                                ontology_code=_parse_hpo_code(s),
                            )
                        )
                    elif isinstance(s, str):
                        cand_symptoms.append(EntityItem(term=s))

                # Extract candidate diagnoses with native int mondoCode
                cand_diagnoses: List[EntityItem] = []
                for d in cand.get("diagnosis", []):
                    if isinstance(d, dict):
                        cand_diagnoses.append(
                            EntityItem(
                                term=d.get("term") or d.get("text") or "",
                                embedding=d.get("embedding"),
                                ontology_code=_parse_mondo_code(d),
                            )
                        )
                    elif isinstance(d, str):
                        cand_diagnoses.append(EntityItem(term=d))

                # Extract candidate age
                cand_age = None
                p_info = cand.get("patientInfo")
                if isinstance(p_info, dict) and "age" in p_info:
                    cand_age = p_info["age"]
                elif isinstance(cand.get("patient"), dict):
                    age_obj = cand["patient"].get("age")
                    if isinstance(age_obj, dict):
                        cand_age = age_obj.get("year")

                # Extract and score candidate visit reason directly
                cand_vt = cand.get("visitReason")
                norm_cand_vt = normalize_visit_type(cand_vt)
                visit_type_score = 1.0 if (norm_query_vt and norm_cand_vt and norm_query_vt == norm_cand_vt) else 0.5

                # Component scores
                diagnosis_score = set_similarity(
                    query_set=query_diagnoses,
                    candidate_set=cand_diagnoses,
                    taxonomy=mondo_taxonomy,
                )
                symptoms_score = set_similarity(
                    query_set=query_symptoms,
                    candidate_set=cand_symptoms,
                    taxonomy=hpo_taxonomy,
                )
                patient_score = patient_similarity(age_q=query_age, age_c=cand_age)

                component_scores = {
                    "diagnosis": diagnosis_score,
                    "symptoms": symptoms_score,
                    "patient": patient_score,
                    "visitType": visit_type_score,
                }
                weights = {
                    "diagnosis": request.params.wDiagnosisScore,
                    "symptoms": request.params.wSymptomsScore,
                    "patient": request.params.wPatientScore,
                    "visitType": request.params.wVisitTypeScore,
                }

                final_score = compute_composite_score(component_scores, weights)
                cand_visit_date = str(cand.get("visitDate", ""))

                scored_candidates.append((
                    final_score,
                    diagnosis_score if diagnosis_score is not None else -1.0,
                    symptoms_score if symptoms_score is not None else -1.0,
                    cand_visit_date,
                    cand,
                    component_scores,
                ))

            # Rank candidates: finalScore desc, diagnosisScore desc, symptomsScore desc, visitDate desc
            scored_candidates.sort(
                key=lambda x: (x[0], x[1], x[2], x[3]),
                reverse=True,
            )

            # Filter and select top_k candidates with per-patient frequency capping (max floor(k/3) per caseNo)
            import math
            from collections import defaultdict

            top_k = request.top_k
            max_per_patient = max(1, math.floor(top_k / 3))
            patient_counts: Dict[Any, int] = defaultdict(int)

            for item in scored_candidates:
                f_score, _, _, _, cand_doc, comp_scores = item
                cand_case_no = cand_doc.get("caseNo")
                if cand_case_no is not None:
                    if patient_counts[cand_case_no] >= max_per_patient:
                        continue
                    patient_counts[cand_case_no] += 1

                similar_cases.append(
                    SimilarCaseItem(
                        case=cand_doc,
                        finalScore=round(float(f_score), 4),
                        components=ComponentScores(
                            diagnosis=round(float(comp_scores["diagnosis"]), 4) if comp_scores["diagnosis"] is not None else None,
                            symptoms=round(float(comp_scores["symptoms"]), 4) if comp_scores["symptoms"] is not None else None,
                            patient=round(float(comp_scores["patient"]), 4) if comp_scores["patient"] is not None else None,
                            visitType=round(float(comp_scores["visitType"]), 4) if comp_scores["visitType"] is not None else None,
                        ),
                    )
                )
                if len(similar_cases) == top_k:
                    break

        return FindSimilarCaseResponse(
            decoratedCase=decorated_case,
            pastCases=past_cases,
            similarCases=similar_cases,
        )
