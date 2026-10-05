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
    ) -> None:
        self._settings = settings or get_settings()
        self._pg = postgres_source or PostgresSource(self._settings)
        self._mongo = mongo_kb or MongoKnowledgeBase(self._settings)
        self._symptom_service = symptom_service or SymptomService(settings=self._settings)




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
        embedder = SapBERTEmbedder()

        term_cache: Dict[str, Dict[str, Any]] = {}

        def _ensure_terms_cached(terms: List[str]) -> None:
            uncached = list(dict.fromkeys([t.strip() for t in terms if t and t.strip() and t.strip() not in term_cache]))
            if not uncached:
                return

            # MONDO lookups
            mondo_codes: Dict[str, Optional[int]] = {}
            for t in uncached:
                code = None
                try:
                    res = mondo_service.search(t, limit=1)
                    if res:
                        m = re.search(r"\d+", res[0].mondo_id)
                        if m:
                            code = int(m.group())
                except Exception:
                    code = None
                mondo_codes[t] = code

            # SapBERT embeddings
            vectors = embedder.embed_entities(uncached)
            for i, t in enumerate(uncached):
                emb = vectors[i].tolist() if i < len(vectors) else []
                term_cache[t] = {
                    "term": t,
                    "mondoCode": mondo_codes[t],
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

        # Persistent cross-batch HPO cache: term → resolved payload
        # Shared across all batches so identical symptoms are resolved only once.
        hpo_cache: Dict[str, Dict[str, Any]] = {}

        total_rows_processed = 0
        total_documents_updated = 0
        total_documents_skipped = 0
        batches_processed = 0

        # --- Determine total row count for progress tracking ---
        total_rows = 0
        try:
            total_rows = self._pg.get_chief_complaints_count()
        except Exception as exc:
            logger.warning("Could not determine chief complaints row count: %s", exc)

        total_batches = math.ceil(total_rows / batch_size) if total_rows > 0 else 0
        logger.info(
            "Chief complaints migration starting: total_rows=%d total_batches=%d batch_size=%d",
            total_rows, total_batches, batch_size,
        )

        for batch in self._pg.stream_chief_complaints(batch_size=batch_size):
            if not batch:
                continue

            batch_start = time.time()
            batches_processed += 1

            # ------------------------------------------------------------------
            # Step 1.2 — Clean & split using utils.segment()
            # ------------------------------------------------------------------
            # row_tokens: index → list of symptom token strings for that row
            row_tokens: Dict[int, List[str]] = {}
            batch_unique: set = set()

            for i, row in enumerate(batch):
                tokens = segment(row["chiefComplaint"])
                tokens = [t for t in tokens if t]  # drop empties after sanitize
                row_tokens[i] = tokens
                batch_unique.update(tokens)

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
                hpo_results = self._symptom_service.resolve_hpo_batch(
                    terms=unique_terms,
                    cache=hpo_cache,
                    max_workers=max_workers,
                )

            hpo_resolved = sum(1 for v in hpo_results.values() if v.get("hpoId") is not None)
            hpo_hit_rate = (hpo_resolved / len(unique_terms) * 100) if unique_terms else 0.0

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
            docs_map = self._mongo.find_cases_by_encounters_batch(encounters)

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
                    updates.append((doc["_id"], r["symptoms"]))
                else:
                    batch_skipped += 1
                    logger.debug(
                        "No MongoDB doc found for caseNo=%s visitDate=%s — skipping.",
                        r["caseNo"], r["visitDate"],
                    )

            batch_updated = 0
            if updates:
                res = self._mongo.update_cases_symptoms_bulk(updates)
                batch_updated = res.get("modified_count", 0) + res.get("matched_count", 0)
                # Count unique _ids successfully matched (modified_count can be 0 when
                # symptoms payload is unchanged; matched_count captures those too).
                batch_updated = res.get("matched_count", 0)

            total_rows_processed += len(batch)
            total_documents_updated += batch_updated
            total_documents_skipped += batch_skipped

            batch_elapsed = round(time.time() - batch_start, 2)
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

