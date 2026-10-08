"""Medicine Master Service — Ingestion, BioLORD vector embedding, and MongoDB synchronization."""

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import numpy as np

from meldai.config import Settings, get_settings
from meldai.db.mongodb import MongoKnowledgeBase
from meldai.nlp.biolord import BioLORDEmbedder, get_biolord_embedder

logger = logging.getLogger(__name__)


def generate_digest_text(item: Dict[str, Any]) -> str:
    """
    Synthesize a clean, descriptive clinical sentence for BioLORD embedding digestion:
    '{canonical_molecule}, marketed as {brand_name}, is used as {clinical_dosing_indication}.'

    Handles acronyms, capitalization, multi-molecule combinations, and missing fields.
    """
    molecule = (item.get("canonical_molecule") or "").strip()
    brand = (item.get("brand_name") or "").strip()
    indication = (item.get("clinical_dosing_indication") or "").strip()

    if indication:
        # Preserve leading acronyms like SNRI, SSRI, ADHD; otherwise lower-case initial letter
        first_word = indication.split()[0]
        if first_word.isupper() and len(first_word) > 1:
            formatted_ind = indication
        else:
            formatted_ind = indication[0].lower() + indication[1:]

        if formatted_ind.endswith("."):
            formatted_ind = formatted_ind[:-1]

        if molecule and brand:
            text = f"{molecule}, marketed as {brand}, is used as {formatted_ind}."
        elif molecule:
            text = f"{molecule} is used as {formatted_ind}."
        elif brand:
            text = f"{brand} is used as {formatted_ind}."
        else:
            text = f"Medication used as {formatted_ind}."
    else:
        if molecule and brand:
            text = f"{molecule}, marketed as {brand}."
        elif molecule:
            text = f"{molecule}."
        elif brand:
            text = f"{brand}."
        else:
            text = "Medication."

    return text


class MedicineMasterService:
    """Service to load local medicine master datasets, generate BioLORD embeddings, and sync to MongoDB."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        mongo_kb: Optional[MongoKnowledgeBase] = None,
        biolord_embedder: Optional[BioLORDEmbedder] = None,
    ):
        self.settings = settings or get_settings()
        self.mongo_kb = mongo_kb or MongoKnowledgeBase(self.settings)
        self.biolord_embedder = biolord_embedder or get_biolord_embedder(self.settings)

    def load_and_embed_master(
        self,
        file_path: Optional[str] = None,
        batch_size: int = 100,
        recreate: bool = False,
    ) -> Dict[str, Any]:
        """
        Load medicine master JSON array, digest into clinical sentences,
        generate BioLORD vector embeddings, and ingest into MongoDB in batches of 100.

        Args:
            file_path: Optional override path to local_medicine_master.json.
            batch_size: Batch size for database writes and chunking (default: 100).
            recreate: If True, clears existing medicine_master collection before ingestion.

        Returns:
            Dict containing ingestion statistics, timing, and count metrics.
        """
        start_time = time.perf_counter()
        target_path = Path(file_path or self.settings.medicine_master_json_path)
        if not target_path.is_file():
            # Try resolving relative to workspace root if not found directly
            repo_root = Path(__file__).resolve().parent.parent.parent.parent
            candidate = repo_root / target_path
            if candidate.is_file():
                target_path = candidate
            else:
                raise FileNotFoundError(f"Medicine master file not found at: {target_path}")

        logger.info("Reading medicine master JSON dataset from %s", target_path)
        with open(target_path, "r", encoding="utf-8") as f:
            raw_records: List[Dict[str, Any]] = json.load(f)

        if not isinstance(raw_records, list):
            raise ValueError(f"Expected a JSON array of objects, got {type(raw_records)}")

        total_records = len(raw_records)
        logger.info("Found %d medicine master records to process.", total_records)

        # Ensure collection indexes are present
        self.mongo_kb.setup_medicine_master_indexes()

        if recreate:
            deleted_count = self.mongo_kb.clear_medicine_master()
            logger.info("Cleared existing medicine_master collection (%d documents removed).", deleted_count)

        total_upserted = 0
        total_modified = 0
        batches_processed = 0

        # Process in batches of 100 (or configured batch_size)
        for i in range(0, total_records, batch_size):
            batch_items = raw_records[i : i + batch_size]
            batches_processed += 1

            # 1. Synthesize digest texts for BioLORD
            batch_texts = [generate_digest_text(item) for item in batch_items]

            # 2. Compute BioLORD dense vector representations
            batch_vectors = self.biolord_embedder.embed_texts(batch_texts, normalize=True)

            # 3. Attach embedding and digest_text at document root
            prepared_docs = []
            for item, text, vector in zip(batch_items, batch_texts, batch_vectors):
                doc = dict(item)  # Preserve all original fields
                doc["digest_text"] = text
                doc["embedding"] = vector.tolist()
                prepared_docs.append(doc)

            # 4. Upsert batch into MongoDB medicine_master
            stats = self.mongo_kb.upsert_medicine_master_batch(prepared_docs)
            total_upserted += stats["upserted_count"]
            total_modified += stats["modified_count"]

            logger.info(
                "Batch %d/%d completed: %d documents processed.",
                batches_processed,
                (total_records + batch_size - 1) // batch_size,
                len(prepared_docs),
            )

        elapsed = round(time.perf_counter() - start_time, 2)
        final_count = self.mongo_kb.get_medicine_master_count()

        logger.info(
            "Medicine master ingestion finished in %.2fs. Total in DB: %d",
            elapsed,
            final_count,
        )

        return {
            "status": "success",
            "collection": "medicine_master",
            "total_records": total_records,
            "upserted_count": total_upserted,
            "modified_count": total_modified,
            "total_in_db": final_count,
            "batches_processed": batches_processed,
            "duration_seconds": elapsed,
        }

    def search_similar_medicines(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic cosine similarity search across medicine_master collection.
        Uses vectorized NumPy dot-product ranking over unit-normalized BioLORD embeddings,
        or MongoDB Atlas $vectorSearch pipeline when available.

        Args:
            query: Clinical query, molecule, or indication phrase.
            top_k: Maximum number of matches to return.
            min_score: Minimum cosine similarity threshold.

        Returns:
            Ranked list of matching medicine records with similarity scores.
        """
        if not query.strip():
            return []

        # 1. Embed query text with BioLORD (unit-normalized)
        query_vec = self.biolord_embedder.embed_texts(query, normalize=True)[0]

        # 2. Try Atlas $vectorSearch pipeline first
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_vec.tolist(),
                        "numCandidates": max(top_k * 5, 20),
                        "limit": top_k,
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "embedding": 0,
                        "similarity_score": {"$meta": "vectorSearchScore"},
                    }
                },
            ]
            results = list(self.mongo_kb.medicine_master.aggregate(pipeline))
            if results:
                return [r for r in results if r.get("similarity_score", 0.0) >= min_score]
        except Exception:
            # Atlas Search not enabled; fallback to in-memory vectorized cosine ranking
            pass

        # 3. Vectorized NumPy cosine similarity engine (for MongoDB Community / Standalone)
        cursor = self.mongo_kb.medicine_master.find(
            {"embedding": {"$exists": True}},
            {"_id": 0},
        )
        docs = list(cursor)
        if not docs:
            return []

        embeddings_list = [d["embedding"] for d in docs]
        emb_matrix = np.array(embeddings_list, dtype=np.float32)

        # Dot product against all unit-normalized embeddings gives exact cosine similarities
        cosine_scores = np.dot(emb_matrix, query_vec)

        matches = []
        for doc, score in zip(docs, cosine_scores):
            sim = float(score)
            if sim >= min_score:
                item = dict(doc)
                item["similarity_score"] = round(sim, 4)
                item.pop("embedding", None)
                matches.append(item)

        matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        return matches[:top_k]

    def search_text_medicines(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Full-text search using MongoDB text index on brand_name, canonical_molecule,
        and clinical_dosing_indication.

        Args:
            query: Keyword search string.
            top_k: Maximum number of matches to return.

        Returns:
            Ranked list of matching medicine records with MongoDB textScores.
        """
        if not query.strip():
            return []

        cursor = self.mongo_kb.medicine_master.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}, "embedding": 0, "_id": 0},
        ).sort([("score", {"$meta": "textScore"})]).limit(top_k)

        matches = []
        for doc in cursor:
            item = dict(doc)
            item["text_score"] = round(float(item.pop("score", 0.0)), 4)
            matches.append(item)
        return matches
