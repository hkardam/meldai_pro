"""Unit tests for pipeline orchestration and case schema validation."""

from unittest.mock import MagicMock
import numpy as np
import pandas as pd
from meldai.db.mongodb import CaseDocument, PatientInfo, PatientAge, ClinicalEntityItem
from meldai.pipelines.medical_analysis import MedicalAnalysisPipeline


def test_case_document_schema():
    """Verify CaseDocument schema conforms strictly to required structure."""
    case = CaseDocument(
        id="a65979bb-ec6c-4860-913a-a19ad6bb07bf",
        caseNo=101,
        visitDate="2026-09-25",
        patient=PatientInfo(
            name="John Doe",
            gender="Male",
            age=PatientAge(year=58, month=6, day=14),
        ),
        symptoms=[
            ClinicalEntityItem(
                text="substernal chest pressure",
                embedding=[0.1] * 768,
            )
        ],
        diagnosis=[
            ClinicalEntityItem(
                text="acute myocardial infarction",
                embedding=[0.2] * 768,
            )
        ],
    )

    data = case.model_dump()
    assert data["id"] == "a65979bb-ec6c-4860-913a-a19ad6bb07bf"
    assert data["caseNo"] == 101
    assert data["visitDate"] == "2026-09-25"
    assert data["patient"]["name"] == "John Doe"
    assert data["patient"]["gender"] == "Male"
    assert data["patient"]["age"]["year"] == 58
    assert data["patient"]["age"]["month"] == 6
    assert data["patient"]["age"]["day"] == 14
    assert len(data["symptoms"]) == 1
    assert data["symptoms"][0]["text"] == "substernal chest pressure"
    assert len(data["symptoms"][0]["embedding"]) == 768
    assert data["diagnosis"][0]["text"] == "acute myocardial infarction"


def test_pipeline_orchestration_with_mocks():
    mock_pg = MagicMock()
    mock_pg.load_training_records.return_value = pd.DataFrame([
        {
            "id": 1,
            "patient_id": "PT-01",
            "encounter_id": "ENC-01",
            "clinical_note": "Patient diagnosed with essential hypertension.",
            "suspected_condition": "essential hypertension",
            "created_at": "2026-09-25",
        }
    ])

    mock_embedder = MagicMock()
    mock_embedder.embed_entities.return_value = np.zeros((1, 768), dtype=np.float32)

    mock_mongo = MagicMock()
    mock_mongo.insert_cases.return_value = ["case_uuid_123"]

    pipeline = MedicalAnalysisPipeline(
        postgres_source=mock_pg,
        mongo_sink=mock_mongo,
        sapbert_embedder=mock_embedder,
    )

    inserted_ids = pipeline.run(limit=1)
    assert inserted_ids == ["case_uuid_123"]
    assert mock_mongo.insert_cases.called
    assert mock_embedder.embed_entities.called
