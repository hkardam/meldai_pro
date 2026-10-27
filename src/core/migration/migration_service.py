"""Unified Patient Migration Service — extracts, decorates, and ingests clinical cases from PostgreSQL into MongoDB."""

from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.config import Settings, get_settings
from core.db.mongodb import MongoKnowledgeBase
from core.db.postgres import PostgresSource
from core.nlp.sapbert import SapBERTEmbedder, get_sapbert_embedder
from core.services.assertion_service import ClinicalAssertionService
from core.terminology.hpo import HPOService, get_hpo_service
from core.terminology.mondo import MONDOService, get_mondo_service
from core.utils import to_curie

logger = logging.getLogger(__name__)


def clean_medicine_token(term: str) -> str:
    """Strip dosage forms and common clinical prefixes/suffixes from a medication string."""
    t = term.strip()
    return re.sub(r"\b(tab|cap|inj|syp|tablet|capsule|syrup|injection)\b\.?", "", t, flags=re.IGNORECASE).strip()


def search_medicine_molecules(term: str, master_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Search local_medicine_master.json entries matching a raw medicine string.
    Tries exact brand/molecule match first, then word-boundary/prefix, then substring.
    """
    if not term or not term.strip():
        return []

    c_term = clean_medicine_token(term)
    term_upper = c_term.upper()
    term_clean = re.sub(r"[\s\-_]+", "", term_upper)

    if not term_clean:
        return []

    # 1. Exact match on brand_name or canonical_molecule
    exact_matches = []
    for item in master_records:
        b = re.sub(r"[\s\-_]+", "", (item.get("brand_name") or "").upper())
        m = re.sub(r"[\s\-_]+", "", (item.get("canonical_molecule") or "").upper())
        if term_clean == b or term_clean == m:
            exact_matches.append(item)
    if exact_matches:
        return exact_matches

    # 2. Word boundary or prefix match on brand name
    brand_prefix_matches = []
    for item in master_records:
        b = (item.get("brand_name") or "").upper()
        if not b:
            continue
        if re.search(r"\b" + re.escape(b) + r"\b", term_upper) or term_upper.startswith(b):
            brand_prefix_matches.append(item)
    if brand_prefix_matches:
        return brand_prefix_matches

    # 3. Canonical molecule word match
    mol_matches = []
    for item in master_records:
        m = (item.get("canonical_molecule") or "").upper()
        if not m:
            continue
        if re.search(r"\b" + re.escape(m) + r"\b", term_upper) or term_upper.startswith(m):
            mol_matches.append(item)
    if mol_matches:
        return mol_matches

    # 4. Partial substring fallback
    substring_matches = []
    for item in master_records:
        b = re.sub(r"[\s\-_]+", "", (item.get("brand_name") or "").upper())
        m = re.sub(r"[\s\-_]+", "", (item.get("canonical_molecule") or "").upper())
        if (b and b in term_clean) or (m and m in term_clean) or (len(term_clean) >= 4 and (term_clean in b or term_clean in m)):
            substring_matches.append(item)
    return substring_matches


class MigrationService:
    """
    Executes the unified clinical patient migration loop:
    start -> query docs from psql -> decorate -> save to mongodb -> extended
    Repeated in batches of 20 (or configured batch_size).
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        postgres_source: Optional[PostgresSource] = None,
        mongo_kb: Optional[MongoKnowledgeBase] = None,
        assertion_service: Optional[ClinicalAssertionService] = None,
        hpo_service: Optional[HPOService] = None,
        mondo_service: Optional[MONDOService] = None,
        embedder: Optional[SapBERTEmbedder] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._pg = postgres_source or PostgresSource(self._settings)
        self._mongo = mongo_kb or MongoKnowledgeBase(self._settings)
        self._assertion_service = assertion_service or ClinicalAssertionService()
        self._hpo_service = hpo_service
        self._mondo_service = mondo_service
        self._embedder = embedder
        self._medicine_master: Optional[List[Dict[str, Any]]] = None

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

    def _get_medicine_master(self) -> List[Dict[str, Any]]:
        """Load and cache local_medicine_master.json dataset."""
        if self._medicine_master is not None:
            return self._medicine_master

        path = Path(self._settings.medicine_master_json_path)
        if not path.is_file():
            repo_root = Path(__file__).resolve().parents[3]
            candidate = repo_root / path
            if candidate.is_file():
                path = candidate

        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._medicine_master = json.load(f)
                    logger.info("Loaded %d medicine master records from %s", len(self._medicine_master), path)
            except Exception as exc:
                logger.warning("Failed to load medicine master from %s: %s", path, exc)
                self._medicine_master = []
        else:
            logger.warning("Medicine master JSON file not found at %s", path)
            self._medicine_master = []

        return self._medicine_master

    def decorate_symptoms(
        self,
        chief_complaints: Optional[str],
        symptom_cache: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Decorate symptoms:
        string -> split by new line or '●' character -> array of symptoms object with
        original term + hbo Term + SapBERT vector, negation detection flag.
        """
        if not chief_complaints or not str(chief_complaints).strip():
            return []

        raw_str = str(chief_complaints).strip()
        tokens = [t.strip() for t in re.split(r"[\r\n●]+", raw_str) if t and t.strip()]
        if not tokens:
            return []

        hpo_svc = self._get_hpo()
        embedder = self._get_embedder()

        # Identify uncached terms
        uncached = [t for t in dict.fromkeys(tokens) if t not in symptom_cache]
        if uncached:
            embeddings = embedder.embed_entities(uncached)
            for i, term in enumerate(uncached):
                assertion = self._assertion_service.analyze(term)
                hits = hpo_svc.search(assertion.search_target, limit=1)
                if not hits and assertion.search_target != term:
                    hits = hpo_svc.search(term, limit=1)

                hit = hits[0] if hits else None
                hpo_term = hit.label if hit else None
                raw_hpo_id = hit.hpo_id if hit else None
                hpo_curie = to_curie("HP", raw_hpo_id) if raw_hpo_id else None
                m_code = re.search(r"\d+", raw_hpo_id) if raw_hpo_id else None
                hpo_code = int(m_code.group()) if m_code else None

                vec = embeddings[i].tolist() if i < len(embeddings) else []

                symptom_cache[term] = {
                    "term": term,
                    "hboTerm": hpo_term,
                    "hpoTerm": hpo_term,
                    "hpoCode": hpo_code,
                    "hpoId": hpo_curie,
                    "embedding": vec,
                    "is_negated": assertion.is_negated,
                    "isNegation": assertion.is_negated,
                    "isPheno": assertion.is_phenotype,
                }

        return [dict(symptom_cache[t]) for t in tokens if t in symptom_cache]

    def decorate_diagnosis(
        self,
        diagnosis_names: Optional[str],
        diagnosis_cache: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Decorate diagnosis:
        string -> split by ',' -> array of diagnosis object with
        original term + mondo Term + SapBERT vector, negation detection flag.
        """
        if not diagnosis_names or not str(diagnosis_names).strip():
            return []

        raw_str = str(diagnosis_names).strip()
        tokens = [t.strip() for t in raw_str.split(",") if t and t.strip()]
        if not tokens:
            return []

        mondo_svc = self._get_mondo()
        embedder = self._get_embedder()

        uncached = [t for t in dict.fromkeys(tokens) if t not in diagnosis_cache]
        if uncached:
            embeddings = embedder.embed_entities(uncached)
            for i, term in enumerate(uncached):
                assertion = self._assertion_service.analyze(term)
                hits = mondo_svc.search(assertion.search_target, limit=1)
                if not hits and assertion.search_target != term:
                    hits = mondo_svc.search(term, limit=1)

                hit = hits[0] if hits else None
                mondo_term = hit.label if hit else None
                raw_mondo_id = hit.mondo_id if hit else None
                mondo_curie = to_curie("MONDO", raw_mondo_id) if raw_mondo_id else None
                m_code = re.search(r"\d+", raw_mondo_id) if raw_mondo_id else None
                mondo_code = int(m_code.group()) if m_code else None

                vec = embeddings[i].tolist() if i < len(embeddings) else []

                diagnosis_cache[term] = {
                    "term": term,
                    "mondoTerm": mondo_term,
                    "mondoCode": mondo_code,
                    "mondoId": mondo_curie,
                    "embedding": vec,
                    "is_negated": assertion.is_negated,
                    "isNegation": assertion.is_negated,
                }

        return [dict(diagnosis_cache[t]) for t in tokens if t in diagnosis_cache]

    def decorate_medicines(
        self,
        medicine_names: Optional[str],
        medicine_cache: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Decorate medicines:
        string -> split by ',' -> array of medicine object with:
        term, original_term, brand_name, canonical_molecule (split by '+' and trimmed),
        is_fdc, constituent_strengths_mg, modal_strength_mg, release_form, dosage_tier,
        dosage_group_id, clinical_note.
        """
        if not medicine_names or not str(medicine_names).strip():
            return []

        raw_str = str(medicine_names).strip()
        tokens = [t.strip() for t in raw_str.split(",") if t and t.strip()]
        if not tokens:
            return []

        master = self._get_medicine_master()

        for term in dict.fromkeys(tokens):
            if term not in medicine_cache:
                matches = search_medicine_molecules(term, master)
                if matches:
                    best = matches[0]
                    brand_name = best.get("brand_name")
                    raw_mol = str(best.get("canonical_molecule") or "")
                    canonical_molecules = [p.strip() for p in raw_mol.split("+") if p and p.strip()]
                    is_fdc = bool(best.get("is_fdc", False))
                    constituent_strengths = best.get("constituent_strengths_mg") or {}
                    modal_strength = best.get("modal_strength_mg")
                    release_form = best.get("release_form")
                    dosage_tier = best.get("dosage_tier")
                    dosage_group_id = best.get("dosage_group_id")
                    clinical_note = best.get("clinical_dosing_indication", "")
                else:
                    brand_name = None
                    canonical_molecules = []
                    is_fdc = False
                    constituent_strengths = {}
                    modal_strength = None
                    release_form = None
                    dosage_tier = None
                    dosage_group_id = None
                    clinical_note = ""

                medicine_cache[term] = {
                    "term": term,
                    "original_term": term,
                    "brand_name": brand_name,
                    "canonical_molecule": canonical_molecules,
                    "is_fdc": is_fdc,
                    "constituent_strengths_mg": constituent_strengths,
                    "modal_strength_mg": modal_strength,
                    "release_form": release_form,
                    "dosage_tier": dosage_tier,
                    "dosage_group_id": dosage_group_id,
                    "clinical_note": clinical_note,
                }

        return [dict(medicine_cache[t]) for t in tokens if t in medicine_cache]

    def run_patient_migration(
        self,
        batch_size: int = 20,
        limit: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Dict[str, Any]:
        """
        Logical flow:
        start -> query docs from psql -> decorate -> save to mongodb -> extended
        Repeated in batches of 20 (or specified batch_size).
        """
        start_time = time.time()
        self._mongo.setup_patient_visits_index()

        # Step 0: start
        logger.info("Patient migration logical flow started. Batch size: %d, limit: %s", batch_size, limit)
        if progress_callback:
            progress_callback({
                "status": "running",
                "current_step": "start",
                "batch_size": batch_size,
                "current_batch": 0,
                "total_records_processed": 0,
                "documents_saved": 0,
            })

        total_pg_records = 0
        try:
            total_pg_records = self._pg.get_patient_migration_count()
            if limit is not None and limit > 0:
                total_pg_records = min(total_pg_records, limit)
        except Exception as exc:
            logger.warning("Could not determine total patient migration count: %s", exc)

        total_batches = math.ceil(total_pg_records / batch_size) if total_pg_records > 0 else 0

        # In-memory cross-batch caches to avoid duplicate lookups
        symptom_cache: Dict[str, Dict[str, Any]] = {}
        diagnosis_cache: Dict[str, Dict[str, Any]] = {}
        medicine_cache: Dict[str, Dict[str, Any]] = {}

        total_processed = 0
        total_upserted = 0
        total_modified = 0
        total_matched = 0
        batches_count = 0

        # Step 1: Query docs from psql (streamed in batches of 20)
        stream = self._pg.stream_patient_migration_records(batch_size=batch_size, limit=limit)

        for batch_rows in stream:
            if not batch_rows:
                continue

            batches_count += 1
            logger.info("Processing migration batch %d (size: %d)", batches_count, len(batch_rows))

            if progress_callback and not progress_callback({
                "status": "running",
                "current_step": "query docs from psql",
                "current_batch": batches_count,
                "total_batches": total_batches,
                "total_records_processed": total_processed,
                "documents_saved": total_upserted + total_modified,
            }):
                logger.info("Migration stopped by request during psql query step.")
                break

            # Step 2: decorate
            if progress_callback and not progress_callback({
                "status": "running",
                "current_step": "decorate",
                "current_batch": batches_count,
                "total_batches": total_batches,
                "total_records_processed": total_processed,
                "documents_saved": total_upserted + total_modified,
            }):
                logger.info("Migration stopped by request during decorate step.")
                break

            decorated_cases: List[Dict[str, Any]] = []
            for row in batch_rows:
                dec_symptoms = self.decorate_symptoms(row.get("chiefComplaints"), symptom_cache)
                dec_diagnosis = self.decorate_diagnosis(row.get("diagnosisNames"), diagnosis_cache)
                dec_medicines = self.decorate_medicines(row.get("medicineNames"), medicine_cache)

                patient_info = {
                    "age": row.get("age", 0.0),
                    "gender": row.get("gender", "m"),
                    "ageDay": row.get("ageDay"),
                    "ageMonth": row.get("ageMonth"),
                    "ageYear": row.get("ageYear"),
                }

                case_doc = {
                    "visitDate": row.get("visitDate"),
                    "caseNo": row.get("caseNo"),
                    "diagnosis": dec_diagnosis,
                    "investigationNames": row.get("investigationNames"),
                    "isLatest": (row.get("isLatest") == 1),
                    "is_latest": row.get("isLatest", 1),
                    "medications": dec_medicines,
                    "migratedAt": datetime.now(timezone.utc).isoformat(),
                    "patientInfo": patient_info,
                    "chiefComplaints": dec_symptoms,
                    "visitReason": row.get("visitReason"),
                }
                decorated_cases.append(case_doc)

            # Step 3: save to mongodb
            if progress_callback and not progress_callback({
                "status": "running",
                "current_step": "save to mongodb",
                "current_batch": batches_count,
                "total_batches": total_batches,
                "total_records_processed": total_processed,
                "documents_saved": total_upserted + total_modified,
            }):
                logger.info("Migration stopped by request during save to mongodb step.")
                break

            write_res = self._mongo.upsert_patient_cases_batch(decorated_cases)
            total_upserted += write_res.get("upserted_count", 0)
            total_modified += write_res.get("modified_count", 0)
            total_matched += write_res.get("matched_count", 0)
            total_processed += len(batch_rows)

            # Step 4: extended (continue loop to next batch)
            logger.info(
                "Batch %d complete (extended) — %d docs saved so far (upserted: %d, modified: %d)",
                batches_count,
                total_upserted + total_modified,
                total_upserted,
                total_modified,
            )

            if progress_callback and not progress_callback({
                "status": "running",
                "current_step": "extended",
                "current_batch": batches_count,
                "total_batches": total_batches,
                "total_records_processed": total_processed,
                "documents_saved": total_upserted + total_modified,
            }):
                logger.info("Migration stopped by request during extended step.")
                break

        exec_time = round(time.time() - start_time, 3)
        logger.info(
            "Patient migration finished: processed=%d batches=%d upserted=%d modified=%d elapsed=%.3fs",
            total_processed,
            batches_count,
            total_upserted,
            total_modified,
            exec_time,
        )

        return {
            "status": "success",
            "batch_size": batch_size,
            "total_records_processed": total_processed,
            "batches_processed": batches_count,
            "upserted_count": total_upserted,
            "modified_count": total_modified,
            "matched_count": total_matched,
            "execution_time_seconds": exec_time,
        }
