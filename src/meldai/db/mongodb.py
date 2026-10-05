"""MongoDB Knowledge Base & Data Sink for Medical Cases and Embeddings."""

import logging
from typing import Any, Dict, List, Optional, Tuple
import uuid
from pydantic import BaseModel, Field
from pymongo import MongoClient, ASCENDING, UpdateOne
from pymongo.database import Database
from pymongo.collection import Collection

from meldai.config import Settings, get_settings

logger = logging.getLogger(__name__)


class PatientAge(BaseModel):
    """Granular patient age specification."""
    year: int = Field(default=0, ge=0, description="Age in years")
    month: int = Field(default=0, ge=0, description="Age in months")
    day: int = Field(default=0, ge=0, description="Age in days")


class PatientInfo(BaseModel):
    """Patient demographic details."""
    name: str = Field(..., description="Full name or pseudonymized identifier")
    gender: str = Field(..., description="Gender: Male, Female, Other, Unknown")
    age: PatientAge = Field(default_factory=PatientAge)


class ClinicalEntityItem(BaseModel):
    """Clinical finding or diagnosis stored as raw text + SapBERT embedding vector.

    Ontology codes (HPO / MONDO) are resolved independently via the
    /api/v1/hpo/search and /api/v1/mondo/search endpoints.
    """
    text: str = Field(..., description="Raw clinical text surface form")
    embedding: List[float] = Field(..., description="768-dimensional SapBERT embedding vector")


class DiagnosisItem(BaseModel):
    """Diagnosis entity item patched into MongoDB case documents."""
    term: str = Field(..., description="Diagnosis surface form from database")
    mondoCode: Optional[int] = Field(default=None, description="Numeric MONDO ontology code")
    embedding: List[float] = Field(default_factory=list, description="768-dimensional SapBERT embedding vector")



class CaseDocument(BaseModel):
    """
    Standard Medical Case Document stored in MongoDB knowledge base.
    Matches exact clinical schema specification.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique Case UUID")
    caseNo: int = Field(..., description="Sequential or institutional case number")
    visitDate: str = Field(..., description="Date of clinical encounter (e.g. YYYY-MM-DD)")
    patient: PatientInfo
    symptoms: List[ClinicalEntityItem] = Field(default_factory=list)
    diagnosis: List[ClinicalEntityItem] = Field(default_factory=list)


class MongoKnowledgeBase:
    """Manages the MongoDB sink and clinical cases repository."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._client: Optional[MongoClient] = None

    @property
    def client(self) -> MongoClient:
        """Lazy MongoClient instance."""
        if self._client is None:
            logger.info("Connecting to MongoDB at %s:%s", self.settings.mongo_host, self.settings.mongo_port)
            self._client = MongoClient(
                self.settings.resolved_mongo_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
            )
        return self._client

    @property
    def db(self) -> Database:
        """Active MongoDB database."""
        return self.client[self.settings.mongo_db_name]

    @property
    def cases(self) -> Collection:
        """Collection storing medical cases."""
        return self.db["cases"]

    def setup_indexes(self) -> None:
        """Initialize indexes for fast lookups and uniqueness."""
        self.cases.create_index([("id", ASCENDING)], unique=True)
        self.setup_patient_visits_index()
        self.cases.create_index([("patient.name", ASCENDING)])
        logger.info("MongoDB 'cases' indexes verified.")

    def setup_patient_visits_index(self) -> None:
        """Initialize unique compound index on (caseNo, visitDate)."""
        try:
            index_info = self.cases.index_information()
            for idx_name, idx_spec in index_info.items():
                keys = idx_spec.get("key", [])
                if keys == [("caseNo", 1)] and idx_spec.get("unique"):
                    self.cases.drop_index(idx_name)
                    logger.info("Dropped legacy single-field unique index: %s", idx_name)
        except Exception as exc:
            logger.warning("Index cleanup warning: %s", exc)

        self.cases.create_index(
            [("caseNo", ASCENDING), ("visitDate", ASCENDING)],
            unique=True,
            name="uniq_caseNo_visitDate",
        )
        logger.info("MongoDB 'cases' compound unique index (caseNo, visitDate) verified.")

    def ping(self) -> bool:
        """Check if MongoDB sink is healthy and responsive."""
        try:
            res = self.client.admin.command("ping")
            return res.get("ok") == 1.0
        except Exception as exc:
            logger.warning("MongoDB ping failed: %s", exc)
            return False

    def insert_case(self, case: CaseDocument) -> str:
        """Insert a single medical case document into MongoDB."""
        doc = case.model_dump()
        result = self.cases.insert_one(doc)
        logger.debug("Inserted medical case id=%s, caseNo=%d", case.id, case.caseNo)
        return case.id

    def insert_cases(self, cases: List[CaseDocument]) -> List[str]:
        """Bulk insert multiple medical cases into MongoDB."""
        if not cases:
            return []
        docs = [c.model_dump() for c in cases]
        self.cases.insert_many(docs)
        inserted_ids = [c.id for c in cases]
        logger.info("Successfully ingested %d cases into MongoDB", len(inserted_ids))
        return inserted_ids

    def find_by_case_no(self, case_no: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific case by caseNo."""
        return self.cases.find_one({"caseNo": case_no}, {"_id": 0})

    def upsert_patient_visits_batch(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Bulk upsert/patch patient visit case documents in MongoDB.
        Uses unique compound key (caseNo, visitDate).
        Document schema:
        {
            "_id": uuid,
            "caseNo": int/str,
            "visitDate": date/str,
            "visitReason": str
        }
        """
        if not records:
            return {"upserted_count": 0, "modified_count": 0, "matched_count": 0}

        operations = []
        for rec in records:
            case_no = rec.get("caseNo")
            visit_date = rec.get("visitDate")
            visit_reason = rec.get("visitReason")

            filter_doc = {"caseNo": case_no, "visitDate": visit_date}
            update_doc = {
                "$set": {
                    "caseNo": case_no,
                    "visitDate": visit_date,
                    "visitReason": visit_reason,
                },
                "$setOnInsert": {
                    "_id": str(uuid.uuid4())
                },
            }
            operations.append(UpdateOne(filter_doc, update_doc, upsert=True))

        result = self.cases.bulk_write(operations, ordered=False)
        logger.info(
            "Batch bulk_write completed: upserted=%d, modified=%d, matched=%d",
            result.upserted_count,
            result.modified_count,
            result.matched_count,
        )
        return {
            "upserted_count": result.upserted_count,
            "modified_count": result.modified_count,
            "matched_count": result.matched_count,
        }

    def replace_case_diagnoses_batch(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Bulk patch/replace the 'diagnosis' array for matched patient visit case documents in MongoDB.
        Uses unique compound key (caseNo, visitDate).
        """
        if not records:
            return {"modified_count": 0, "matched_count": 0}

        operations = []
        for rec in records:
            case_no = rec.get("caseNo")
            visit_date = rec.get("visitDate")
            diagnosis = rec.get("diagnosis", [])

            filter_doc = {"caseNo": case_no, "visitDate": visit_date}
            update_doc = {
                "$set": {
                    "diagnosis": diagnosis,
                }
            }
            operations.append(UpdateOne(filter_doc, update_doc))

        result = self.cases.bulk_write(operations, ordered=False)
        logger.info(
            "Diagnosis batch replace completed: modified=%d, matched=%d",
            result.modified_count,
            result.matched_count,
        )
        return {
            "modified_count": result.modified_count,
            "matched_count": result.matched_count,
        }

    def find_case_by_encounter(self, case_no: Any, visit_date: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific case document by compound key (caseNo, visitDate)."""
        return self.cases.find_one({"caseNo": case_no, "visitDate": visit_date})

    def update_case_symptoms(self, case_id: Any, symptoms: List[Dict[str, Any]]) -> bool:
        """Non-destructively patch symptoms array on a specific case document by _id.

        Uses the MongoDB $set operator so ONLY the 'symptoms' field is updated.
        All other existing fields (such as 'diagnosis', 'visitReason', 'caseNo',
        and 'visitDate') are strictly preserved and untouched.
        """
        res = self.cases.update_one(
            {"_id": case_id},
            {"$set": {"symptoms": symptoms}},
        )
        return res.modified_count > 0 or res.matched_count > 0

    def find_cases_by_encounters_batch(
        self, encounters: List[Tuple[Any, Any]]
    ) -> Dict[Tuple[Any, str], Dict[str, Any]]:
        """Fetch multiple case documents in a single round-trip using an $or compound query.

        Returns a dictionary mapping (caseNo, str(visitDate)) -> doc.
        """
        if not encounters:
            return {}
        or_clauses = [{"caseNo": c, "visitDate": v} for c, v in encounters if c is not None and v is not None]
        if not or_clauses:
            return {}
        cursor = self.cases.find(
            {"$or": or_clauses},
            {"_id": 1, "caseNo": 1, "visitDate": 1, "patient_visit_id": 1}
        )
        result: Dict[Tuple[Any, str], Dict[str, Any]] = {}
        for doc in cursor:
            result[(doc.get("caseNo"), str(doc.get("visitDate")))] = doc
        return result

    def update_cases_symptoms_bulk(
        self, updates: List[Tuple[Any, List[Dict[str, Any]]]]
    ) -> Dict[str, int]:
        """Bulk non-destructively patch symptoms array on multiple case documents.

        Uses the MongoDB $set operator so ONLY the 'symptoms' field is updated.
        Existing fields (diagnosis, visitReason, etc.) are strictly preserved.
        """
        if not updates:
            return {"modified_count": 0, "matched_count": 0}

        operations = [
            UpdateOne({"_id": doc_id}, {"$set": {"symptoms": symptoms}})
            for doc_id, symptoms in updates
        ]
        result = self.cases.bulk_write(operations, ordered=False)
        return {
            "modified_count": result.modified_count,
            "matched_count": result.matched_count,
        }

    def update_cases_medications_bulk(
        self, updates: List[Tuple[Any, List[str]]]
    ) -> Dict[str, int]:
        """Bulk non-destructively patch medications array on multiple case documents.

        Uses the MongoDB $set operator so ONLY the 'medications' field is updated.
        Existing fields (diagnosis, symptoms, visitReason, etc.) are strictly preserved.
        """
        if not updates:
            return {"modified_count": 0, "matched_count": 0}

        operations = [
            UpdateOne({"_id": doc_id}, {"$set": {"medications": medications}})
            for doc_id, medications in updates
        ]
        result = self.cases.bulk_write(operations, ordered=False)
        return {
            "modified_count": result.modified_count,
            "matched_count": result.matched_count,
        }

    def close(self) -> None:
        """Close MongoDB connection pool."""
        if self._client:
            self._client.close()
            self._client = None


