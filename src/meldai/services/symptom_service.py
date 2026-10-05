"""Symptom service — handles note segmentation, medspaCy NLP pipeline, and ontology resolution."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from meldai.config import Settings, get_settings
from meldai.nlp.medspacy_nlp import MedspaCyService, get_medspacy_service
from meldai.services.assertion_service import ClinicalAssertionService
from meldai.terminology.hpo import HPOService, get_hpo_service
from meldai.terminology.mondo import MONDOService, get_mondo_service
from meldai.utils import segment

logger = logging.getLogger(__name__)


class SymptomService:
    """Clinical symptom segmentation, assertion analysis, and ontology mapping."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        nlp_service: Optional[MedspaCyService] = None,
        assertion_service: Optional[ClinicalAssertionService] = None,
        hpo_service: Optional[HPOService] = None,
        mondo_service: Optional[MONDOService] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._nlp_service = nlp_service
        self._assertion_service = assertion_service or ClinicalAssertionService(self._nlp_service)
        self._hpo_service = hpo_service
        self._mondo_service = mondo_service

    def _get_nlp(self) -> MedspaCyService:
        if self._nlp_service is not None:
            return self._nlp_service
        return get_medspacy_service()

    def _get_hpo(self) -> HPOService:
        if self._hpo_service is not None:
            return self._hpo_service
        return get_hpo_service(self._settings.hpo_obo_path)

    def _get_mondo(self) -> MONDOService:
        if self._mondo_service is not None:
            return self._mondo_service
        return get_mondo_service(self._settings.mondo_obo_path)

    def segment_note_symptoms(self, note: str) -> List[Dict[str, Any]]:
        """Split a clinical note into symptom segments and process them via medspaCy pipeline.

        Returns a list of dicts suitable for building SymptomItem responses.
        """
        segments = segment(note)
        if not segments:
            return []

        nlp = self._get_nlp()
        nlp_results = nlp.process(segments)

        items: List[Dict[str, Any]] = []
        for result in nlp_results:
            if result.is_phenotype:
                items.append({
                    "note": result.text,
                    "isPheno": True,
                    "embedding": result.embedding if result.embedding else None,
                    "hpoCode": result.hpo_code,
                })
            else:
                items.append({
                    "note": result.text,
                    "isPheno": False,
                })
        return items

    def resolve_symptom(
        self,
        sym_text: str,
        cache: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Resolve a symptom string to HPO code, MONDO code, and clinical assertion metadata.

        If a cache dictionary is provided, checks and populates it to avoid redundant lookups.
        """
        clean_text = sym_text.strip()
        if cache is not None and clean_text in cache:
            return cache[clean_text]

        # 1. Clinical assertion & entity extraction
        assertion = self._assertion_service.analyze(clean_text)

        # 2. HPO mapping
        hpo = self._get_hpo()
        hpo_res = hpo.search(assertion.search_target, limit=1)
        if not hpo_res and assertion.search_target != clean_text:
            hpo_res = hpo.search(clean_text, limit=1)

        hpo_id = hpo_res[0].hpo_id if hpo_res else None
        hpo_code: Optional[int] = None
        if hpo_id:
            m = re.search(r"\d+", hpo_id)
            if m:
                hpo_code = int(m.group())

        # 3. MONDO mapping
        mondo = self._get_mondo()
        mondo_res = mondo.search(assertion.search_target, limit=1)
        if not mondo_res and assertion.search_target != clean_text:
            mondo_res = mondo.search(clean_text, limit=1)

        mondo_id = mondo_res[0].mondo_id if mondo_res else None
        mondo_code: Optional[int] = None
        if mondo_id:
            m = re.search(r"\d+", mondo_id)
            if m:
                mondo_code = int(m.group())

        entry = {
            "note": clean_text,
            "hpoCode": hpo_code,
            "hpoId": hpo_id,
            "mondoCode": mondo_code,
            "mondoId": mondo_id,
            "is_negated": assertion.is_negated,
            "is_phenotype": assertion.is_phenotype,
            "assertion_status": assertion.assertion_status,
        }

        if cache is not None:
            cache[clean_text] = entry

        return entry

    def resolve_symptoms_batch(
        self,
        sym_texts: List[str],
        cache: Optional[Dict[str, Dict[str, Any]]] = None,
        max_workers: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Batch resolve clinical symptom strings to HPO/MONDO codes with parallel processing."""
        if not sym_texts:
            return {}

        results: Dict[str, Dict[str, Any]] = {}
        uncached: List[str] = []

        for raw in sym_texts:
            clean = raw.strip() if raw else ""
            if not clean:
                continue
            if cache is not None and clean in cache:
                results[clean] = cache[clean]
            elif clean not in results:
                uncached.append(clean)

        if not uncached:
            return results

        # 1. Batch NLP assertion analysis
        assertions = self._assertion_service.analyze_batch(uncached)
        hpo = self._get_hpo()
        mondo = self._get_mondo()

        hpo_target_cache: Dict[str, Tuple[Optional[str], Optional[int]]] = {}
        mondo_target_cache: Dict[str, Tuple[Optional[str], Optional[int]]] = {}

        def _lookup_hpo(target: str, raw_text: str) -> Tuple[Optional[str], Optional[int]]:
            if target in hpo_target_cache:
                return hpo_target_cache[target]
            res = hpo.search(target, limit=1)
            if not res and target != raw_text:
                res = hpo.search(raw_text, limit=1)
            hpo_id = res[0].hpo_id if res else None
            hpo_code = None
            if hpo_id:
                m = re.search(r"\d+", hpo_id)
                if m:
                    hpo_code = int(m.group())
            val = (hpo_id, hpo_code)
            hpo_target_cache[target] = val
            return val

        def _lookup_mondo(target: str, raw_text: str) -> Tuple[Optional[str], Optional[int]]:
            if target in mondo_target_cache:
                return mondo_target_cache[target]
            res = mondo.search(target, limit=1)
            if not res and target != raw_text:
                res = mondo.search(raw_text, limit=1)
            mondo_id = res[0].mondo_id if res else None
            mondo_code = None
            if mondo_id:
                m = re.search(r"\d+", mondo_id)
                if m:
                    mondo_code = int(m.group())
            val = (mondo_id, mondo_code)
            mondo_target_cache[target] = val
            return val

        # 2. Parallel ontology resolution across available CPU cores
        import os
        from concurrent.futures import ThreadPoolExecutor

        workers = max_workers or max(1, min(os.cpu_count() or 2, 4))

        def _resolve_one(assertion) -> Tuple[str, Dict[str, Any]]:
            clean_text = assertion.term
            target = assertion.search_target
            hpo_id, hpo_code = _lookup_hpo(target, clean_text)
            mondo_id, mondo_code = _lookup_mondo(target, clean_text)
            entry = {
                "note": clean_text,
                "hpoCode": hpo_code,
                "hpoId": hpo_id,
                "mondoCode": mondo_code,
                "mondoId": mondo_id,
                "is_negated": assertion.is_negated,
                "is_phenotype": assertion.is_phenotype,
                "assertion_status": assertion.assertion_status,
            }
            return clean_text, entry

        if len(assertions) > 1 and workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                resolved_items = list(executor.map(_resolve_one, assertions))
        else:
            resolved_items = [_resolve_one(a) for a in assertions]

        for clean_text, entry in resolved_items:
            results[clean_text] = entry
            if cache is not None:
                cache[clean_text] = entry

        return results

    def match_symptoms_batch(
        self,
        symptoms: List[str],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Batch match clinical symptom strings to top-K HPO and MONDO concepts with assertion analysis."""
        if not symptoms:
            return []

        clean_symptoms = [s.strip() for s in symptoms if s and s.strip()]
        if not clean_symptoms:
            return []

        # 1. Batch NLP assertion parsing using medspaCy pipe
        assertions = self._assertion_service.analyze_batch(clean_symptoms)
        hpo = self._get_hpo()
        mondo = self._get_mondo()

        results: List[Dict[str, Any]] = []
        for assertion in assertions:
            raw_text = assertion.term
            target = assertion.search_target

            # Search HPO top_k
            hpo_results = hpo.search(target, limit=top_k)
            if not hpo_results and target != raw_text:
                hpo_results = hpo.search(raw_text, limit=top_k)

            hpo_matches = []
            for item in hpo_results[:top_k]:
                code = None
                m = re.search(r"\d+", item.hpo_id)
                if m:
                    code = int(m.group())
                hpo_matches.append({
                    "hpo_id": item.hpo_id,
                    "code": code,
                    "label": item.label,
                    "match_type": item.match_type,
                    "score": item.score,
                })

            # Search MONDO top_k
            mondo_results = mondo.search(target, limit=top_k)
            if not mondo_results and target != raw_text:
                mondo_results = mondo.search(raw_text, limit=top_k)

            mondo_matches = []
            for item in mondo_results[:top_k]:
                code = None
                m = re.search(r"\d+", item.mondo_id)
                if m:
                    code = int(m.group())
                mondo_matches.append({
                    "mondo_id": item.mondo_id,
                    "code": code,
                    "label": item.label,
                    "match_type": item.match_type,
                    "score": item.score,
                })

            results.append({
                "symptom": raw_text,
                "search_target": target,
                "assertion_status": assertion.assertion_status,
                "is_negated": assertion.is_negated,
                "is_phenotype": assertion.is_phenotype,
                "hpo_matches": hpo_matches,
                "mondo_matches": mondo_matches,
            })

        return results
    def resolve_hpo_batch(
        self,
        terms: List[str],
        cache: Optional[Dict[str, Dict[str, Any]]] = None,
        max_workers: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Batch resolve clinical symptom strings to HPO codes only (MONDO intentionally skipped).

        Returns a dict mapping clean_term -> symptom payload dict with shape:
            {
                "term": str,
                "hpoTerm": str | None,
                "hpoId": str | None,
                "hpoCode": int | None,
                "isNegation": bool,
                "isPheno": bool,
                "assertionStatus": str,
                "matchScore": float | None,
                "matchType": str | None,
            }

        If a persistent cross-batch *cache* is provided, already-resolved terms are
        returned immediately without re-running NLP or HPO search.
        """
        if not terms:
            return {}

        results: Dict[str, Dict[str, Any]] = {}
        uncached: List[str] = []

        for raw in terms:
            clean = raw.strip() if raw else ""
            if not clean:
                continue
            if cache is not None and clean in cache:
                results[clean] = cache[clean]
            elif clean not in results:
                uncached.append(clean)

        if not uncached:
            return results

        # 1. Batch NLP assertion analysis
        assertions = self._assertion_service.analyze_batch(uncached)
        hpo = self._get_hpo()

        # Term-level HPO target cache to avoid duplicate HPO searches within this call
        hpo_target_cache: Dict[str, tuple] = {}

        def _lookup_hpo(target: str, raw_text: str):
            if target in hpo_target_cache:
                return hpo_target_cache[target]
            res = hpo.search(target, limit=1)
            if not res and target != raw_text:
                res = hpo.search(raw_text, limit=1)
            if res:
                hit = res[0]
                hpo_id = hit.hpo_id
                hpo_code = None
                m = re.search(r"\d+", hpo_id)
                if m:
                    hpo_code = int(m.group())
                val = (hit.label, hpo_id, hpo_code, hit.score, hit.match_type)
            else:
                val = (None, None, None, None, None)
            hpo_target_cache[target] = val
            return val

        def _resolve_one(assertion) -> tuple:
            clean_text = assertion.term
            target = assertion.search_target
            hpo_term, hpo_id, hpo_code, match_score, match_type = _lookup_hpo(target, clean_text)
            entry = {
                "term": clean_text,
                "hpoTerm": hpo_term,
                "hpoId": hpo_id,
                "hpoCode": hpo_code,
                "isNegation": assertion.is_negated,
                "isPheno": assertion.is_phenotype,
                "assertionStatus": assertion.assertion_status,
                "matchScore": round(match_score, 4) if match_score is not None else None,
                "matchType": match_type,
            }
            return clean_text, entry

        import os
        from concurrent.futures import ThreadPoolExecutor

        workers = max_workers or max(1, min(os.cpu_count() or 2, 4))

        if len(assertions) > 1 and workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                resolved_items = list(executor.map(_resolve_one, assertions))
        else:
            resolved_items = [_resolve_one(a) for a in assertions]

        for clean_text, entry in resolved_items:
            results[clean_text] = entry
            if cache is not None:
                cache[clean_text] = entry

        return results

