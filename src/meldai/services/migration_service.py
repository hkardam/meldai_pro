"""Migration and ETL service — orchestrates streaming from PostgreSQL to MongoDB with non-destructive patching."""

import logging
import math
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from meldai.config import Settings, get_settings
from meldai.db.mongodb import MongoKnowledgeBase
from meldai.db.postgres import PostgresSource
from meldai.nlp.sapbert import SapBERTEmbedder
from meldai.services.symptom_service import SymptomService
from meldai.terminology.mondo import get_mondo_service
from meldai.utils import segment

logger = logging.getLogger(__name__)


class MigrationService:
    """Orchestrates data migration pipelines between PostgreSQL and MongoDB."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        postgres_source: Optional[PostgresSource] = None,
        mongo_kb: Optional[MongoKnowledgeBase] = None,
        symptom_service: Optional[SymptomService] = None,
        embedder: Optional[SapBERTEmbedder] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._pg = postgres_source or PostgresSource(self._settings)
        self._mongo = mongo_kb or MongoKnowledgeBase(self._settings)
        self._symptom_service = symptom_service or SymptomService(settings=self._settings)
        self._embedder = embedder




    def migrate_patient_visits(self, batch_size: int = 1000) -> Dict[str, Any]:
        """Pull patient visit data from PostgreSQL and upsert/patch into MongoDB 'cases' collection."""
        self._mongo.setup_patient_visits_index()

        total_processed = 0
        batches_count = 0
        total_upserted = 0
        total_modified = 0
        total_matched = 0

        total_records = 0
        try:
            count_val = self._pg.get_patient_visit_count()
            if isinstance(count_val, int):
                total_records = count_val
        except Exception as exc:
            logger.warning("Could not determine total patient visit count: %s", exc)

        batches = self._pg.stream_patient_visit_data(batch_size=batch_size)
        if total_records > 0:
            total_batches = math.ceil(total_records / batch_size)
        elif isinstance(batches, (list, tuple)):
            total_batches = len(batches)
        else:
            total_batches = 0

        for batch in batches:
            if not batch:
                continue
            res = self._mongo.upsert_patient_visits_batch(batch)
            total_processed += len(batch)
            batches_count += 1
            total_upserted += res.get("upserted_count", 0)
            total_modified += res.get("modified_count", 0)
            total_matched += res.get("matched_count", 0)

            display_total = max(batches_count, total_batches) if total_batches > 0 else batches_count
            logger.info(
                "Patient visits migration: completed batch %d of %d (%d records processed)",
                batches_count,
                display_total,
                len(batch),
            )

        return {
            "status": "success",
            "batch_size": batch_size,
            "total_records_processed": total_processed,
            "batches_processed": batches_count,
            "upserted_count": total_upserted,
            "modified_count": total_modified,
            "matched_count": total_matched,
        }

    def migrate_patient_diagnoses(self, batch_size: int = 1000) -> Dict[str, Any]:
        """Pull grouped patient diagnoses from PostgreSQL, resolve embeddings, and patch MongoDB documents."""
        start_time = time.time()
        mondo_service = get_mondo_service(self._settings.mondo_obo_path)
        embedder = self._embedder or SapBERTEmbedder(self._settings)

        term_cache: Dict[str, Dict[str, Any]] = {}

        def _ensure_terms_cached(terms: List[str]) -> None:
            uncached = list(dict.fromkeys([t.strip() for t in terms if t and t.strip() and t.strip() not in term_cache]))
            if not uncached:
                return

            # MONDO lookups (resolving canonical mondoTerm and integer mondoCode)
            mondo_info: Dict[str, Tuple[Optional[str], Optional[int]]] = {}
            for t in uncached:
                m_term = None
                m_code = None
                try:
                    res = mondo_service.search(t, limit=1)
                    if res:
                        m_term = res[0].label
                        m = re.search(r"\d+", res[0].mondo_id)
                        if m:
                            m_code = int(m.group())
                except Exception:
                    m_term = None
                    m_code = None
                mondo_info[t] = (m_term, m_code)

            # SapBERT embeddings
            vectors = embedder.embed_entities(uncached)
            for i, t in enumerate(uncached):
                emb = vectors[i].tolist() if i < len(vectors) else []
                m_term, m_code = mondo_info[t]
                term_cache[t] = {
                    "term": t,
                    "mondoTerm": m_term,
                    "mondoCode": m_code,
                    "embedding": emb,
                }

        total_encounters = 0
        batches_count = 0
        total_modified = 0
        total_matched = 0

        total_records = 0
        try:
            count_val = self._pg.get_grouped_patient_diagnoses_count()
            if isinstance(count_val, int):
                total_records = count_val
        except Exception as exc:
            logger.warning("Could not determine total grouped patient diagnoses count: %s", exc)

        batches = self._pg.stream_grouped_patient_diagnoses(batch_size=batch_size)
        if total_records > 0:
            total_batches = math.ceil(total_records / batch_size)
        elif isinstance(batches, (list, tuple)):
            total_batches = len(batches)
        else:
            total_batches = 0

        for batch in batches:
            if not batch:
                continue

            batch_terms = [t for enc in batch for t in enc.get("diagnosisNames", []) if t and t.strip()]
            _ensure_terms_cached(batch_terms)

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

            # Non-destructively patch ONLY the 'diagnosis' field using MongoDB $set operator
            res = self._mongo.replace_case_diagnoses_batch(mongo_batch)
            total_encounters += len(batch)
            batches_count += 1
            total_modified += res.get("modified_count", 0)
            total_matched += res.get("matched_count", 0)

            display_total = max(batches_count, total_batches) if total_batches > 0 else batches_count
            logger.info(
                "Patient diagnoses migration: completed batch %d of %d (%d encounters processed)",
                batches_count,
                display_total,
                len(batch),
            )

        exec_time = round(time.time() - start_time, 3)
        return {
            "status": "success",
            "batch_size": batch_size,
            "total_encounters_processed": total_encounters,
            "batches_processed": batches_count,
            "modified_count": total_modified,
            "matched_count": total_matched,
            "unique_terms_indexed": len(term_cache),
            "execution_time_seconds": exec_time,
        }

    def load_chief_complaints(
        self,
        batch_size: int = 100,
        max_workers: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Dict[str, Any]:
        """Migrate chief complaints from PostgreSQL into MongoDB 'cases.symptoms' field.

        Pipeline per batch:
          1.1 Fetch `batch_size` rows from temp_migrations.patient_chiefcomplains_data
          1.2 Clean & split each "Chief Complaints" string via utils.segment()
          1.3 Bulk HPO-only resolution (MONDO skipped) with cross-batch persistent cache
          1.4 Merge resolved results back to per-row symptom payloads
          1.5 Batch-fetch MongoDB _ids, then bulk $set symptoms field on _id match

        Args:
            batch_size:         Rows fetched per PostgreSQL batch (default 100).
            max_workers:        Thread-pool size for parallel HPO resolution.
            progress_callback:  Called after each batch with a state dict.
                                Returns True to continue, False to stop.
        """
        start_time = time.time()
        embedder = self._embedder or SapBERTEmbedder(self._settings)

        # Persistent cross-batch HPO cache: term → resolved payload
        # Shared across all batches so identical symptoms are resolved only once.
        hpo_cache: Dict[str, Dict[str, Any]] = {}

        total_rows_processed = 0
        total_documents_updated = 0
        total_documents_skipped = 0
        batches_processed = 0

        total_rows = 0
        count_fn = getattr(self._pg, "get_patient_chief_complaints_count", None) or getattr(self._pg, "get_chief_complaints_count", None)
        if count_fn:
            try:
                count_res = count_fn()
                if isinstance(count_res, (int, float)):
                    total_rows = int(count_res)
            except Exception as exc:
                logger.warning("Could not determine chief complaints row count: %s", exc)

        total_batches = math.ceil(total_rows / batch_size) if total_rows > 0 else 0
        logger.info(
            "Chief complaints migration starting: total_rows=%d total_batches=%d batch_size=%d",
            total_rows, total_batches, batch_size,
        )

        stream_fn = getattr(self._pg, "stream_patient_chief_complaints", None) or getattr(self._pg, "stream_chief_complaints", None)
        for batch in stream_fn(batch_size=batch_size):
            if not batch:
                continue

            batch_start = time.time()
            batches_processed += 1

            # ------------------------------------------------------------------
            # Step 1.2 — Clean & split using utils.segment()
            # ------------------------------------------------------------------
            # row_tokens: index → list of symptom token strings for that row
            row_tokens: Dict[int, List[str]] = {}
            batch_unique: Dict[str, None] = {}

            for i, row in enumerate(batch):
                raw_cc = row.get("chiefComplaint") or row.get("chiefComplaints") or ""
                tokens = segment(raw_cc)
                tokens = [t for t in tokens if t]  # drop empties after sanitize
                row_tokens[i] = tokens
                for t in tokens:
                    batch_unique[t] = None

            # ------------------------------------------------------------------
            # Step 1.3 — Bulk HPO-only search (stop-aware)
            # ------------------------------------------------------------------
            if progress_callback and not progress_callback({
                "current_batch": batches_processed,
                "total_batches": total_batches,
                "total_rows_processed": total_rows_processed,
                "documents_updated": total_documents_updated,
                "documents_skipped": total_documents_skipped,
                "current_step": f"resolving_hpo_batch_{batches_processed}",
            }):
                logger.info("Chief complaints migration stop requested before HPO resolve (batch %d).", batches_processed)
                break

            unique_terms = list(batch_unique)
            hpo_results: Dict[str, Dict[str, Any]] = {}
            if unique_terms:
                for t in unique_terms:
                    res_item = self._symptom_service.resolve_symptom(t, cache=hpo_cache)
                    if isinstance(res_item, dict):
                        hpo_results[t] = res_item

            hpo_resolved = sum(1 for v in hpo_results.values() if v.get("hpoId") is not None)
            hpo_hit_rate = (hpo_resolved / len(unique_terms) * 100) if unique_terms else 0.0

            # ------------------------------------------------------------------
            # Step 1.3b — Batch SapBERT vector embeddings (cached across batches)
            # ------------------------------------------------------------------
            uncached_embed_terms = [
                t for t in unique_terms
                if t in hpo_results and "embedding" not in hpo_results[t]
            ]
            if uncached_embed_terms:
                if progress_callback and not progress_callback({
                    "current_batch": batches_processed,
                    "total_batches": total_batches,
                    "total_rows_processed": total_rows_processed,
                    "documents_updated": total_documents_updated,
                    "documents_skipped": total_documents_skipped,
                    "current_step": f"embedding_symptoms_batch_{batches_processed}",
                }):
                    logger.info("Chief complaints migration stop requested before embedding (batch %d).", batches_processed)
                    break

                vectors = embedder.embed_entities(uncached_embed_terms)
                for i, term in enumerate(uncached_embed_terms):
                    emb = vectors[i].tolist() if i < len(vectors) else []
                    hpo_results[term]["embedding"] = emb
                    if term in hpo_cache:
                        hpo_cache[term]["embedding"] = emb

            # ------------------------------------------------------------------
            # Step 1.4 — Merge HPO results back to per-row symptom payloads
            # ------------------------------------------------------------------
            mongo_batch: List[Dict[str, Any]] = []
            for i, row in enumerate(batch):
                symptoms_payload = []
                for token in row_tokens[i]:
                    resolved = hpo_results.get(token)
                    if resolved:
                        symptoms_payload.append(resolved)
                mongo_batch.append({
                    "caseNo": row["caseNo"],
                    "visitDate": row["visitDate"],
                    "symptoms": symptoms_payload,
                })

            # ------------------------------------------------------------------
            # Step 1.5a — Bulk fetch MongoDB _ids (single $or round-trip)
            # ------------------------------------------------------------------
            if progress_callback and not progress_callback({
                "current_batch": batches_processed,
                "total_batches": total_batches,
                "total_rows_processed": total_rows_processed,
                "documents_updated": total_documents_updated,
                "documents_skipped": total_documents_skipped,
                "current_step": f"fetching_mongo_ids_batch_{batches_processed}",
            }):
                logger.info("Chief complaints migration stop requested before Mongo write (batch %d).", batches_processed)
                break

            encounters = [(r["caseNo"], r["visitDate"]) for r in mongo_batch]
            docs_map: Dict[tuple, Any] = {}
            if hasattr(self._mongo, "find_cases_by_encounters_batch"):
                try:
                    res_map = self._mongo.find_cases_by_encounters_batch(encounters)
                    if isinstance(res_map, dict):
                        docs_map = dict(res_map)
                except Exception:
                    pass

            if not docs_map and hasattr(self._mongo, "find_case_by_encounter"):
                for r in mongo_batch:
                    key = (r["caseNo"], str(r["visitDate"]))
                    doc = self._mongo.find_case_by_encounter(r["caseNo"], str(r["visitDate"]))
                    if doc:
                        docs_map[key] = doc

            docs_found = len(docs_map)
            mongo_match_rate = (docs_found / len(batch) * 100) if batch else 0.0

            if mongo_match_rate < 50.0:
                logger.warning(
                    "Low MongoDB match rate in batch %d: %.1f%% (%d/%d docs found). "
                    "Chief complaints rows may precede patient visit migration.",
                    batches_processed, mongo_match_rate, docs_found, len(batch),
                )

            # ------------------------------------------------------------------
            # Step 1.5b — Bulk $set symptoms on _id (ordered=False)
            # ------------------------------------------------------------------
            updates: List[tuple] = []
            batch_skipped = 0

            for r in mongo_batch:
                key = (r["caseNo"], str(r["visitDate"]))
                doc = docs_map.get(key)
                if doc:
                    pv_id = doc.get("patient_visit_id")
                    symptoms_with_pv = []
                    for s in r["symptoms"]:
                        s_dict = dict(s)
                        if pv_id and "patient_visit_id" not in s_dict:
                            s_dict["patient_visit_id"] = pv_id
                        symptoms_with_pv.append(s_dict)
                    updates.append((doc["_id"], symptoms_with_pv))
                else:
                    batch_skipped += 1
                    logger.debug(
                        "No MongoDB doc found for caseNo=%s visitDate=%s — skipping.",
                        r["caseNo"], r["visitDate"],
                    )

            batch_updated = 0
            if updates:
                if hasattr(self._mongo, "update_case_symptoms"):
                    for doc_id, syms in updates:
                        self._mongo.update_case_symptoms(doc_id, syms)
                    batch_updated = len(updates)
                elif hasattr(self._mongo, "update_cases_symptoms_bulk"):
                    try:
                        res = self._mongo.update_cases_symptoms_bulk(updates)
                        if isinstance(res, dict):
                            batch_updated = res.get("matched_count", 0) or res.get("modified_count", 0)
                        elif isinstance(res, (int, float)):
                            batch_updated = int(res)
                        else:
                            batch_updated = len(updates)
                    except Exception:
                        pass

            total_rows_processed += len(batch)
            total_documents_updated += batch_updated
            total_documents_skipped += batch_skipped

            batch_elapsed = round(time.time() - batch_start, 2)
            logger.info(
                "Patient chief complaints migration: completed batch %d of %d",
                batches_processed,
                total_batches or batches_processed,
            )
            logger.info(
                "Chief complaints batch %d/%d: rows=%d updated=%d skipped=%d "
                "hpo_hit=%.1f%% mongo_match=%.1f%% cache_size=%d elapsed=%.2fs",
                batches_processed,
                max(batches_processed, total_batches),
                len(batch),
                batch_updated,
                batch_skipped,
                hpo_hit_rate,
                mongo_match_rate,
                len(hpo_cache),
                batch_elapsed,
            )

            # ------------------------------------------------------------------
            # Progress callback — returns False when stop is requested
            # ------------------------------------------------------------------
            if progress_callback and not progress_callback({
                "current_batch": batches_processed,
                "total_batches": total_batches,
                "total_rows_processed": total_rows_processed,
                "documents_updated": total_documents_updated,
                "documents_skipped": total_documents_skipped,
                "current_step": f"batch_{batches_processed}_complete",
            }):
                logger.info("Chief complaints migration stop requested after batch %d.", batches_processed)
                break

        exec_time = round(time.time() - start_time, 3)
        logger.info(
            "Chief complaints migration finished: total_rows=%d updated=%d skipped=%d "
            "batches=%d unique_terms_cached=%d elapsed=%.3fs",
            total_rows_processed,
            total_documents_updated,
            total_documents_skipped,
            batches_processed,
            len(hpo_cache),
            exec_time,
        )

        return {
            "status": "success",
            "batch_size": batch_size,
            "total_rows_processed": total_rows_processed,
            "documents_updated": total_documents_updated,
            "documents_skipped": total_documents_skipped,
            "batches_processed": batches_processed,
            "unique_terms_cached": len(hpo_cache),
            "execution_time_seconds": exec_time,
        }

    def migrate_patient_prescriptions(self, batch_size: int = 1000) -> Dict[str, Any]:
        """
        Pull patient prescription/medication data from PostgreSQL in batches,
        match case documents in MongoDB using caseNo + visitDate, and patch the
        'medications' field with the raw medicine name strings directly.
        """
        start_time = time.time()
        total_encounters = 0
        batches_count = 0
        total_modified = 0
        total_matched = 0

        total_records = 0
        try:
            count_val = self._pg.get_patient_prescriptions_count()
            if isinstance(count_val, int):
                total_records = count_val
        except Exception as exc:
            logger.warning("Could not determine total patient prescriptions count: %s", exc)

        batches = self._pg.stream_patient_prescriptions(batch_size=batch_size)
        if total_records > 0:
            total_batches = math.ceil(total_records / batch_size)
        elif isinstance(batches, (list, tuple)):
            total_batches = len(batches)
        else:
            total_batches = 0

        for batch in batches:
            if not batch:
                continue

            batches_count += 1
            total_encounters += len(batch)

            encounters = [(enc["caseNo"], enc["visitDate"]) for enc in batch]
            docs_map = self._mongo.find_cases_by_encounters_batch(encounters)

            updates: List[Tuple[Any, List[str]]] = []
            for enc in batch:
                key = (enc["caseNo"], str(enc["visitDate"]))
                doc = docs_map.get(key)
                if doc:
                    updates.append((doc["_id"], enc.get("medications", [])))

            if updates:
                res = self._mongo.update_cases_medications_bulk(updates)
                total_modified += res.get("modified_count", 0)
                total_matched += res.get("matched_count", 0)

            logger.info(
                "Completed patient prescriptions migration batch %d/%d (batch_size=%d, matched=%d, modified=%d)",
                batches_count,
                total_batches or batches_count,
                len(batch),
                len(updates),
                total_modified,
            )

        exec_time = round(time.time() - start_time, 3)
        logger.info(
            "Patient prescriptions migration finished: total_encounters=%d batches=%d modified=%d matched=%d elapsed=%.3fs",
            total_encounters,
            batches_count,
            total_modified,
            total_matched,
            exec_time,
        )

        return {
            "status": "success",
            "batch_size": batch_size,
            "total_encounters_processed": total_encounters,
            "batches_processed": batches_count,
            "modified_count": total_modified,
            "matched_count": total_matched,
            "execution_time_seconds": exec_time,
        }

    def migrate_patient_demographics(self, batch_size: int = 1000) -> Dict[str, Any]:
        """
        Pull patient demographics from PostgreSQL in batches, match case documents in
        MongoDB using caseNo, and patch the 'patientInfo' field: {age: float, gender: 'm' | 'f'}.
        """
        start_time = time.time()
        total_records = 0
        batches_count = 0
        total_modified = 0
        total_matched = 0

        total_pg_records = 0
        try:
            count_val = self._pg.get_patient_demographics_count()
            if isinstance(count_val, int):
                total_pg_records = count_val
        except Exception as exc:
            logger.warning("Could not determine total patient demographics count: %s", exc)

        batches = self._pg.stream_patient_demographics(batch_size=batch_size)
        if total_pg_records > 0:
            total_batches = math.ceil(total_pg_records / batch_size)
        elif isinstance(batches, (list, tuple)):
            total_batches = len(batches)
        else:
            total_batches = 0

        for batch in batches:
            if not batch:
                continue

            batches_count += 1
            total_records += len(batch)

            case_nos = [rec["caseNo"] for rec in batch]
            docs_map = self._mongo.find_cases_by_case_nos_batch(case_nos)

            updates: List[Tuple[Any, Dict[str, Any]]] = []
            for rec in batch:
                c_no = rec["caseNo"]
                matched_docs = docs_map.get(c_no, [])
                for doc in matched_docs:
                    updates.append((doc["_id"], rec["patientInfo"]))

            if updates:
                res = self._mongo.update_cases_demographics_bulk(updates)
                total_modified += res.get("modified_count", 0)
                total_matched += res.get("matched_count", 0)

            logger.info(
                "Completed patient demographics migration batch %d/%d (batch_size=%d, matched=%d, modified=%d)",
                batches_count,
                total_batches or batches_count,
                len(batch),
                len(updates),
                total_modified,
            )

        exec_time = round(time.time() - start_time, 3)
        logger.info(
            "Patient demographics migration finished: total_records=%d batches=%d modified=%d matched=%d elapsed=%.3fs",
            total_records,
            batches_count,
            total_modified,
            total_matched,
            exec_time,
        )

        return {
            "status": "success",
            "batch_size": batch_size,
            "total_records_processed": total_records,
            "batches_processed": batches_count,
            "modified_count": total_modified,
            "matched_count": total_matched,
            "execution_time_seconds": exec_time,
        }

    # Backward-compatible alias
    migrate_patient_chief_complaints = load_chief_complaints


