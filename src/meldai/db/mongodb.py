"""MongoDB Knowledge Base & Data Sink for Medical Cases and Embeddings."""

import logging
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field
from pymongo import MongoClient, ASCENDING
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
        self.cases.create_index([("caseNo", ASCENDING)], unique=True)
        self.cases.create_index([("visitDate", ASCENDING)])
        self.cases.create_index([("patient.name", ASCENDING)])
        logger.info("MongoDB 'cases' indexes verified.")

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

    def close(self) -> None:
        """Close MongoDB connection pool."""
        if self._client:
            self._client.close()
            self._client = None
