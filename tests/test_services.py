"""Unit tests for MeldAI service layer and non-destructive MongoDB patching."""

from unittest.mock import MagicMock
import pytest

from meldai.nlp.medspacy_nlp import SegmentResult
from meldai.services.assertion_service import ClinicalAssertionService
from meldai.services.migration_service import MigrationService
from meldai.services.symptom_service import SymptomService
from meldai.services.terminology_service import TerminologySearchService
from meldai.terminology.hpo import HPOResult
from meldai.terminology.mondo import MONDOResult


def _make_mock_doc(text: str, ent_text: str = "", is_negated: bool = False, modifier_category: str = ""):
    doc = MagicMock()
    doc.text = text
    if ent_text:
        ent = MagicMock()
        ent.text = ent_text
        ent._.is_negated = is_negated
        ent._.is_historical = False
        ent._.is_uncertain = False
        ent._.is_family = False
        doc.ents = [ent]
    else:
        doc.ents = []

    mod = MagicMock()
    mod.category = modifier_category or ("NEGATED_EXISTENCE" if is_negated else "")
    doc._.context_graph.modifiers = [mod] if (modifier_category or is_negated) else []
    return doc


# ---------------------------------------------------------------------------
# 1. ClinicalAssertionService tests
# ---------------------------------------------------------------------------

def test_assertion_service_affirmed():
    mock_nlp = MagicMock()
    mock_nlp.analyze.return_value = _make_mock_doc("headache", ent_text="headache", is_negated=False)

    service = ClinicalAssertionService(nlp_service=mock_nlp)
    res = service.analyze("headache")

    assert res.term == "headache"
    assert res.search_target == "headache"
    assert res.is_negated is False
    assert res.is_phenotype is True
    assert res.assertion_status == "affirmed"


def test_assertion_service_negated():
    mock_nlp = MagicMock()
    mock_nlp.analyze.return_value = _make_mock_doc("no fever", ent_text="fever", is_negated=True)

    service = ClinicalAssertionService(nlp_service=mock_nlp)
    res = service.analyze("no fever")

    assert res.term == "no fever"
    assert res.search_target == "fever"
    assert res.is_negated is True
    assert res.is_phenotype is False
    assert res.assertion_status == "negated"


# ---------------------------------------------------------------------------
# 2. TerminologySearchService tests
# ---------------------------------------------------------------------------

def test_terminology_service_hpo_search():
    mock_nlp = MagicMock()
    mock_nlp.analyze.return_value = _make_mock_doc("fatigue", ent_text="fatigue", is_negated=False)
    mock_assertion = ClinicalAssertionService(nlp_service=mock_nlp)

    mock_hpo = MagicMock()
    mock_hpo.search.return_value = [
        HPOResult(
            hpo_id="HP:0012378",
            label="Fatigue",
            synonyms=["Tiredness"],
            match_type="exact_label",
            score=1.0,
        )
    ]

    service = TerminologySearchService(
        assertion_service=mock_assertion,
        hpo_service=mock_hpo,
    )
    results = service.search_hpo("fatigue", limit=5)

    assert len(results) == 1
    assert results[0].hpo_id == "HP:0012378"
    assert results[0].is_negated is False
    assert results[0].is_phenotype is True
    assert results[0].assertion_status == "affirmed"


def test_terminology_service_mondo_search_filtered():
    mock_nlp = MagicMock()
    mock_nlp.analyze.return_value = _make_mock_doc("denies diabetes", ent_text="diabetes", is_negated=True)
    mock_assertion = ClinicalAssertionService(nlp_service=mock_nlp)

    mock_mondo = MagicMock()
    mock_mondo.search.return_value = [
        MONDOResult(
            mondo_id="MONDO:0005015",
            label="diabetes mellitus",
            synonyms=[],
            match_type="exact_label",
            score=1.0,
        )
    ]

    service = TerminologySearchService(
        assertion_service=mock_assertion,
        mondo_service=mock_mondo,
    )
    results = service.search_mondo("denies diabetes", limit=5, filter_negated=True)
    assert results == []


# ---------------------------------------------------------------------------
# 3. SymptomService tests
# ---------------------------------------------------------------------------

def test_symptom_service_resolve_symptom():
    mock_nlp = MagicMock()
    mock_nlp.analyze.return_value = _make_mock_doc("migraine", ent_text="migraine", is_negated=False)

    mock_hpo = MagicMock()
    mock_hpo.search.return_value = [
        HPOResult(hpo_id="HP:0002076", label="Migraine", synonyms=[], match_type="exact", score=1.0)
    ]
    mock_mondo = MagicMock()
    mock_mondo.search.return_value = [
        MONDOResult(mondo_id="MONDO:0005280", label="migraine disorder", synonyms=[], match_type="exact", score=1.0)
    ]

    service = SymptomService(
        nlp_service=mock_nlp,
        hpo_service=mock_hpo,
        mondo_service=mock_mondo,
    )
    cache = {}
    item = service.resolve_symptom("migraine", cache=cache)

    assert item["note"] == "migraine"
    assert item["hpoCode"] == 2076
    assert item["mondoCode"] == 5280
    assert item["is_negated"] is False
    assert "migraine" in cache


# ---------------------------------------------------------------------------
# 4. MigrationService & Non-Destructive Patching tests
# ---------------------------------------------------------------------------

def test_migration_service_preserves_diagnosis_and_other_fields():
    """Verify that loading chief complaints patches $set: {'symptoms': ...}

    and DOES NOT remove 'diagnosis', 'visitReason', or any existing fields.
    """
    mock_pg = MagicMock()
    mock_pg.stream_patient_chief_complaints.return_value = [
        [
            {
                "caseNo": "M999",
                "visitDate": "2024-03-01",
                "chiefComplaints": "● chronic insomnia\n● mild dizziness",
            }
        ]
    ]

    # Pre-existing MongoDB case document with diagnosis and visitReason
    existing_mongo_doc = {
        "_id": "case-uuid-999",
        "caseNo": "M999",
        "visitDate": "2024-03-01",
        "visitReason": "Annual Checkup",
        "patient_visit_id": "visit-uuid-999",
        "diagnosis": [
            {"term": "Essential hypertension", "mondoCode": 7777}
        ],
    }

    mock_mongo = MagicMock()
    mock_mongo.find_case_by_encounter.return_value = existing_mongo_doc
    mock_mongo.update_case_symptoms.return_value = True

    mock_symptom_svc = MagicMock()
    mock_symptom_svc.resolve_symptom.side_effect = lambda note, cache: {
        "note": note,
        "hpoCode": 1234,
        "mondoCode": 5678,
        "is_negated": False,
        "is_phenotype": True,
        "assertion_status": "affirmed",
    }

    migration_svc = MigrationService(
        postgres_source=mock_pg,
        mongo_kb=mock_mongo,
        symptom_service=mock_symptom_svc,
    )

    result = migration_svc.load_chief_complaints(batch_size=50)

    assert result["status"] == "success"
    assert result["documents_updated"] == 1
    assert result["documents_skipped"] == 0

    # Ensure update_case_symptoms was called with case-uuid-999
    mock_mongo.update_case_symptoms.assert_called_once()
    called_id, symptoms = mock_mongo.update_case_symptoms.call_args[0]
    assert called_id == "case-uuid-999"
    assert len(symptoms) == 2
    assert symptoms[0]["patient_visit_id"] == "visit-uuid-999"
    assert symptoms[0]["note"] == "chronic insomnia"
    assert symptoms[1]["note"] == "mild dizziness"

    # Verify that existing_mongo_doc retains its 'diagnosis' and 'visitReason' fields!
    assert "diagnosis" in existing_mongo_doc
    assert existing_mongo_doc["diagnosis"][0]["term"] == "Essential hypertension"
    assert existing_mongo_doc["visitReason"] == "Annual Checkup"


def test_migration_service_batch_progress_logging(caplog):
    """Verify batch progress logging ('completed batch X of Y') across all migrate-patient services."""
    import logging

    mock_pg = MagicMock()
    mock_pg.get_patient_visit_count.return_value = 2500
    mock_pg.stream_patient_visit_data.return_value = [
        [{"caseNo": 1, "visitDate": "2024-01-01", "visitReason": "checkup"}],
        [{"caseNo": 2, "visitDate": "2024-01-02", "visitReason": "fever"}],
    ]

    mock_pg.get_grouped_patient_diagnoses_count.return_value = 2000
    mock_pg.stream_grouped_patient_diagnoses.return_value = [
        [{"caseNo": 1, "visitDate": "2024-01-01", "diagnosisNames": ["flu"]}],
    ]

    mock_pg.get_patient_chief_complaints_count.return_value = 200
    mock_pg.stream_patient_chief_complaints.return_value = [
        [{"caseNo": 1, "visitDate": "2024-01-01", "chiefComplaints": "headache"}],
    ]

    mock_mongo = MagicMock()
    mock_mongo.upsert_patient_visits_batch.return_value = {"upserted_count": 1}
    mock_mongo.replace_case_diagnoses_batch.return_value = {"modified_count": 1}
    mock_mongo.find_case_by_encounter.return_value = {"_id": "c1", "patient_visit_id": "v1"}
    mock_mongo.update_case_symptoms.return_value = True

    mock_symptom_svc = MagicMock()
    mock_symptom_svc.resolve_symptom.return_value = {"note": "headache", "hpoCode": 1, "mondoCode": 2}

    migration_svc = MigrationService(
        postgres_source=mock_pg,
        mongo_kb=mock_mongo,
        symptom_service=mock_symptom_svc,
    )

    with caplog.at_level(logging.INFO):
        # 1. Test migrate_patient_visits
        res_visits = migration_svc.migrate_patient_visits(batch_size=1000)
        assert res_visits["batches_processed"] == 2
        assert "Patient visits migration: completed batch 1 of 3" in caplog.text
        assert "Patient visits migration: completed batch 2 of 3" in caplog.text

        caplog.clear()

        # 2. Test migrate_patient_diagnoses
        res_diag = migration_svc.migrate_patient_diagnoses(batch_size=1000)
        assert res_diag["batches_processed"] == 1
        assert "Patient diagnoses migration: completed batch 1 of 2" in caplog.text

        caplog.clear()

        # 3. Test load_chief_complaints
        res_cc = migration_svc.load_chief_complaints(batch_size=100)
        assert res_cc["batches_processed"] == 1
        assert "Patient chief complaints migration: completed batch 1 of 2" in caplog.text

