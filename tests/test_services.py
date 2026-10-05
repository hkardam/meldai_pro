"""Unit tests for MeldAI service layer and non-destructive MongoDB patching."""

from unittest.mock import MagicMock
import numpy as np
import pytest

from meldai.api.req_dtos import FindSimilarCaseRequest, PatientInfoInput
from meldai.nlp.medspacy_nlp import SegmentResult
from meldai.services.assertion_service import ClinicalAssertionService
from meldai.services.case_service import CaseService
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

    mock_embedder = MagicMock()
    mock_embedder.embed_entities.return_value = np.zeros((2, 768))

    migration_svc = MigrationService(
        postgres_source=mock_pg,
        mongo_kb=mock_mongo,
        symptom_service=mock_symptom_svc,
        embedder=mock_embedder,
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

    mock_embedder = MagicMock()
    mock_embedder.embed_entities.return_value = np.zeros((1, 768))

    migration_svc = MigrationService(
        postgres_source=mock_pg,
        mongo_kb=mock_mongo,
        symptom_service=mock_symptom_svc,
        embedder=mock_embedder,
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


# ---------------------------------------------------------------------------
# 6. CaseService tests
# ---------------------------------------------------------------------------

def test_case_service_decorate_and_find_similar():
    mock_nlp = MagicMock()
    # Mock NLP analyze for symptom "lack of sleep" and "no fever"
    def mock_analyze(term):
        if "no fever" in term:
            return _make_mock_doc(term, ent_text="fever", is_negated=True)
        return _make_mock_doc(term, ent_text=term, is_negated=False)

    mock_nlp.analyze.side_effect = mock_analyze
    assertion_svc = ClinicalAssertionService(nlp_service=mock_nlp)

    mock_hpo = MagicMock()
    mock_hpo.search.side_effect = lambda term, limit=1: [
        HPOResult(
            hpo_id="HP:0002360",
            label="Sleep disturbance",
            match_type="fuzzy_label",
            score=0.9,
        )
    ] if "sleep" in term else []

    mock_mondo = MagicMock()
    mock_mondo.search.side_effect = lambda term, limit=1: [
        MONDOResult(
            mondo_id="MONDO:0008807",
            label="insomnia (disease)",
            match_type="fuzzy_label",
            score=0.95,
        )
    ] if "insomia" in term.lower() or "insomnia" in term.lower() else []

    mock_embedder = MagicMock()
    mock_embedder.embed_entities.side_effect = lambda terms: np.ones((len(terms), 768), dtype=np.float32)

    case_svc = CaseService(
        assertion_service=assertion_svc,
        hpo_service=mock_hpo,
        mondo_service=mock_mondo,
        embedder=mock_embedder,
    )

    req = FindSimilarCaseRequest(
        caseNo=101,
        patientInfo=PatientInfoInput(age=45.0, gender="Male"),
        symptoms=["lack of sleep", "no fever"],
        diagnosis=["Insomia"],
    )

    res = case_svc.find_similar_cases(req)
    assert res.similarCases == []

    dec_case = res.decoratedCase
    assert dec_case.caseNo == 101
    assert dec_case.patientInfo.age == 45.0
    assert dec_case.patientInfo.gender == "Male"

    # Symptom 1: lack of sleep
    assert len(dec_case.symptoms) == 2
    s1 = dec_case.symptoms[0]
    assert s1.term == "lack of sleep"
    assert s1.hpoTerm == "Sleep disturbance"
    assert s1.hpoCode == 2360
    assert isinstance(s1.embedding, list)
    assert len(s1.embedding) == 768
    assert s1.isNegation is False
    assert s1.isPheno is True
    assert s1.similarity is None

    # Symptom 2: no fever (negated)
    s2 = dec_case.symptoms[1]
    assert s2.term == "no fever"
    assert len(s2.embedding) == 768
    assert s2.isNegation is True
    assert s2.isPheno is False

    # Diagnosis: Insomia
    assert len(dec_case.diagnosis) == 1
    d1 = dec_case.diagnosis[0]
    assert d1.term == "Insomia"
    assert d1.mondoTerm == "insomnia (disease)"
    assert d1.mondoCode == 8807
    assert isinstance(d1.embedding, list)
    assert len(d1.embedding) == 768
    assert d1.similarity is None

