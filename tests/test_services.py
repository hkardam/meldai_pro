"""Unit tests for MeldAI service layer and non-destructive MongoDB patching."""

from unittest.mock import MagicMock
import numpy as np
import pytest

from meldai.api.req_dtos import (
    CaseInput,
    FindSimilarCaseRequest,
    PatientInfoInput,
    SimilarityParams,
    VisitTypeEnum,
)
from meldai.nlp.medspacy_nlp import SegmentResult
from meldai.services.assertion_service import ClinicalAssertionService
from meldai.services.case_service import CaseService
from meldai.services.migration_service import MigrationService
from meldai.services.similarity_engine import (
    EntityItem,
    compute_composite_score,
    patient_similarity,
    set_similarity,
    term_similarity,
    visit_type_similarity,
)
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


def test_migrate_patient_diagnoses_payload_structure():
    """Verify that migrate_patient_diagnoses formats diagnosis payloads with term, mondoTerm, mondoCode, and embedding."""
    mock_pg = MagicMock()
    mock_pg.get_grouped_patient_diagnoses_count.return_value = 1
    mock_pg.stream_grouped_patient_diagnoses.return_value = [
        [{"caseNo": 100, "visitDate": "2024-05-01", "diagnosisNames": ["Essential Hypertension"]}],
    ]

    mock_mongo = MagicMock()
    mock_mongo.replace_case_diagnoses_batch.return_value = {"modified_count": 1, "matched_count": 1}

    mock_embedder = MagicMock()
    mock_embedder.embed_entities.return_value = np.ones((1, 768), dtype=np.float32)

    migration_svc = MigrationService(
        postgres_source=mock_pg,
        mongo_kb=mock_mongo,
        embedder=mock_embedder,
    )

    res = migration_svc.migrate_patient_diagnoses(batch_size=100)
    assert res["status"] == "success"
    mock_mongo.replace_case_diagnoses_batch.assert_called_once()
    called_batch = mock_mongo.replace_case_diagnoses_batch.call_args[0][0]
    assert len(called_batch) == 1
    enc_doc = called_batch[0]
    assert enc_doc["caseNo"] == 100
    assert enc_doc["visitDate"] == "2024-05-01"
    assert len(enc_doc["diagnosis"]) == 1
    diag_item = enc_doc["diagnosis"][0]
    assert "term" in diag_item
    assert "mondoTerm" in diag_item
    assert "mondoCode" in diag_item
    assert "embedding" in diag_item
    assert diag_item["term"] == "Essential Hypertension"
    assert len(diag_item["embedding"]) == 768


# ---------------------------------------------------------------------------
# 6. CaseService tests
# ---------------------------------------------------------------------------

def test_case_service_decorate_and_find_similar():
    mock_nlp = MagicMock()
    def mock_analyze(term):
        if "no fever" in term:
            return _make_mock_doc(term, ent_text="fever", is_negated=True)
        return _make_mock_doc(term, ent_text=term, is_negated=False)

    mock_nlp.analyze.side_effect = mock_analyze
    mock_nlp.analyze_batch.side_effect = lambda terms, **kwargs: [mock_analyze(t) for t in terms]
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
        case=CaseInput(
            caseNo=101,
            visitType=VisitTypeEnum.FOLLOW_UP,
            patientInfo=PatientInfoInput(age=45.0, gender="Male"),
            symptoms=["lack of sleep", "no fever"],
            diagnosis=["Insomia"],
        )
    )

    res = case_svc.find_similar_cases(req, mongo_kb=None)
    assert res.pastCases == []
    assert res.similarCases == []

    dec_case = res.decoratedCase
    assert dec_case.caseNo == 101
    assert dec_case.visitType == "follow-up"
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


def test_case_service_past_cases_populated():
    """When caseNo + caseDate are provided, past visits with visitDate < caseDate are returned."""
    mock_nlp = MagicMock()
    mock_nlp.analyze.side_effect = lambda term: _make_mock_doc(term, ent_text=term, is_negated=False)
    mock_nlp.analyze_batch.side_effect = lambda terms, **kwargs: [_make_mock_doc(t, ent_text=t, is_negated=False) for t in terms]
    assertion_svc = ClinicalAssertionService(nlp_service=mock_nlp)

    mock_embedder = MagicMock()
    mock_embedder.embed_entities.side_effect = lambda terms: np.ones((len(terms), 768), dtype=np.float32)

    case_svc = CaseService(
        assertion_service=assertion_svc,
        hpo_service=MagicMock(),
        mondo_service=MagicMock(),
        embedder=mock_embedder,
    )

    past_doc_1 = {"caseNo": 101, "visitDate": "2023-01-10", "visitReason": "fever", "symptoms": [], "diagnosis": []}
    past_doc_2 = {"caseNo": 101, "visitDate": "2023-06-20", "visitReason": "headache", "symptoms": [], "diagnosis": []}

    mock_mongo = MagicMock()
    mock_mongo.find_past_cases.return_value = [past_doc_1, past_doc_2]
    mock_mongo.find_similar_candidates.return_value = []

    req = FindSimilarCaseRequest(
        case=CaseInput(
            caseNo=101,
            caseDate="2024-01-01",
            visitType=VisitTypeEnum.FOLLOW_UP,
            patientInfo=PatientInfoInput(age=45.0, gender="Male"),
            symptoms=[],
            diagnosis=[],
        )
    )

    res = case_svc.find_similar_cases(req, mongo_kb=mock_mongo)

    mock_mongo.find_past_cases.assert_called_once_with(case_no=101, before_date="2024-01-01")
    assert len(res.pastCases) == 2
    assert res.pastCases[0]["visitDate"] == "2023-01-10"
    assert res.pastCases[1]["visitDate"] == "2023-06-20"


def test_case_service_past_cases_defaults_to_today_when_no_case_date():
    """When caseDate is omitted, current date is used to query prior visits."""
    mock_nlp = MagicMock()
    mock_nlp.analyze.side_effect = lambda term: _make_mock_doc(term, ent_text=term, is_negated=False)
    mock_nlp.analyze_batch.side_effect = lambda terms, **kwargs: [_make_mock_doc(t, ent_text=t, is_negated=False) for t in terms]
    assertion_svc = ClinicalAssertionService(nlp_service=mock_nlp)

    mock_embedder = MagicMock()
    mock_embedder.embed_entities.side_effect = lambda terms: np.ones((len(terms), 768), dtype=np.float32)

    mock_mongo = MagicMock()
    mock_mongo.find_past_cases.return_value = []
    mock_mongo.find_similar_candidates.return_value = []

    case_svc = CaseService(
        assertion_service=assertion_svc,
        hpo_service=MagicMock(),
        mondo_service=MagicMock(),
        embedder=mock_embedder,
    )

    req = FindSimilarCaseRequest(
        case=CaseInput(
            caseNo=101,
            caseDate=None,
            visitType=VisitTypeEnum.FOLLOW_UP,
            symptoms=[],
            diagnosis=[],
        )
    )

    case_svc.find_similar_cases(req, mongo_kb=mock_mongo)
    assert mock_mongo.find_past_cases.called
    call_kwargs = mock_mongo.find_past_cases.call_args[1]
    assert call_kwargs["case_no"] == 101
    assert len(call_kwargs["before_date"]) == 10  # YYYY-MM-DD


# ---------------------------------------------------------------------------
# 7. Mathematical Properties & Invariants Tests
# ---------------------------------------------------------------------------

def test_identical_sets_score_one_and_symmetric():
    """Identical sets score 1.0, and score(A, B) == score(B, A) when BETA = 0.5."""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]

    set_a = [
        EntityItem(term="insomnia", embedding=vec1, ontology_code="MONDO:0008807"),
        EntityItem(term="headache", embedding=vec2, ontology_code="MONDO:0005101"),
    ]
    set_b = [
        EntityItem(term="insomnia", embedding=vec1, ontology_code="MONDO:0008807"),
        EntityItem(term="headache", embedding=vec2, ontology_code="MONDO:0005101"),
    ]

    score_ab = set_similarity(set_a, set_b, beta=0.5)
    score_ba = set_similarity(set_b, set_a, beta=0.5)

    assert pytest.approx(score_ab, rel=1e-5) == 1.0
    assert pytest.approx(score_ba, rel=1e-5) == 1.0
    assert pytest.approx(score_ab, rel=1e-5) == score_ba


def test_weak_match_cutoff_monotony():
    """Adding symptoms whose best match is <= TAU (0.3) to a candidate never raises its score."""
    q_vec = [1.0, 0.0, 0.0]
    c_good_vec = [1.0, 0.0, 0.0]      # identical -> raw = 1.0, s* = 1.0
    c_weak_vec = [0.25, 0.968, 0.0]   # cos ~ 0.25 <= TAU=0.3 -> s* = 0.0

    query = [EntityItem(term="cough", embedding=q_vec)]
    candidate_base = [EntityItem(term="cough", embedding=c_good_vec)]

    # Candidate with extra weak symptom
    candidate_with_weak = [
        EntityItem(term="cough", embedding=c_good_vec),
        EntityItem(term="unrelated", embedding=c_weak_vec),
    ]

    score_base = set_similarity(query, candidate_base)
    score_with_weak = set_similarity(query, candidate_with_weak)

    assert score_base is not None and score_with_weak is not None
    # Extra symptom with s* = 0 reduces C -> Q coverage average, so score must not increase
    assert score_with_weak <= score_base


def test_exact_plus_unrelated_beats_weak_matches():
    """1 identical + 1 unrelated beats 10 symptoms each at raw 0.5 for query of 2 symptoms."""
    # Query: 2 symptoms
    q1 = EntityItem(term="fever", embedding=[1.0, 0.0, 0.0])
    q2 = EntityItem(term="cough", embedding=[0.0, 1.0, 0.0])
    query = [q1, q2]

    # Candidate 1: 1 identical + 1 unrelated (orthogonal)
    c1_good = EntityItem(term="fever", embedding=[1.0, 0.0, 0.0])
    c1_unrel = EntityItem(term="fracture", embedding=[0.0, 0.0, 1.0])
    candidate_1 = [c1_good, c1_unrel]

    # Candidate 2: 10 symptoms each with cosine = 0.5 (below 1.0, sharpened = (0.5 - 0.3) / 0.7 = 0.285)
    # Cosine = 0.5 vector with respect to q1 and q2: [0.5, 0.5, 0.707]
    c2_items = [
        EntityItem(term=f"weak_symptom_{i}", embedding=[0.5, 0.5, 0.7071])
        for i in range(10)
    ]

    score_cand1 = set_similarity(query, candidate_1, tau=0.3, alpha=1.0)
    score_cand2 = set_similarity(query, c2_items, tau=0.3, alpha=1.0)

    assert score_cand1 is not None and score_cand2 is not None
    assert score_cand1 > score_cand2


def test_set_deduplication():
    """Duplicating a term in a set does not change the score."""
    q_vec = [1.0, 0.0, 0.0]
    query = [EntityItem(term="nausea", embedding=q_vec)]

    cand_unique = [
        EntityItem(term="nausea", embedding=q_vec),
        EntityItem(term="vomiting", embedding=[0.0, 1.0, 0.0]),
    ]
    cand_duplicated = [
        EntityItem(term="nausea", embedding=q_vec),
        EntityItem(term="nausea", embedding=q_vec),  # Duplicate
        EntityItem(term="vomiting", embedding=[0.0, 1.0, 0.0]),
    ]

    s_unique = set_similarity(query, cand_unique)
    s_dup = set_similarity(query, cand_duplicated)

    assert s_unique is not None
    assert pytest.approx(s_unique, rel=1e-5) == s_dup


def test_patient_score_bands_and_gaussian():
    """Patient score evaluates 0.7 * bandScore + 0.3 * gaussScore."""
    # Same age (same band, 0 diff -> band 1.0, gauss 1.0 -> 1.0)
    assert pytest.approx(patient_similarity(30.0, 30.0), rel=1e-5) == 1.0

    # 1 band apart: 20 (band 4: 18-25) vs 30 (band 5: 26-40) -> bandScore 0.7
    score_1_apart = patient_similarity(20.0, 30.0)
    assert 0.5 < score_1_apart < 1.0

    # 2 bands apart: 20 (band 4) vs 50 (band 6: 41-60) -> bandScore 0.1
    score_2_apart = patient_similarity(20.0, 50.0)
    assert 0.0 < score_2_apart < score_1_apart

    # Missing age returns None
    assert patient_similarity(None, 30.0) is None
    assert patient_similarity(30.0, None) is None


def test_visit_type_score_exact_and_different():
    """Visit type score is 1.0 for exact match and 0.5 for different."""
    assert visit_type_similarity("follow-up", "follow-up") == 1.0
    assert visit_type_similarity("follow-up", "Follow-up Visit") == 1.0
    assert visit_type_similarity("follow-up", "new consultation") == 0.5
    assert visit_type_similarity("new consultation", "Follow-up Visit") == 0.5


def test_composite_score_weight_normalization():
    """Composite score dynamically normalizes weights over available components."""
    # When all 4 available
    scores = {"diagnosis": 1.0, "symptoms": 0.5, "patient": 1.0, "visitType": 0.5}
    weights = {"diagnosis": 0.4, "symptoms": 0.4, "patient": 0.1, "visitType": 0.1}
    expected = (0.4 * 1.0 + 0.4 * 0.5 + 0.1 * 1.0 + 0.1 * 0.5) / 1.0
    assert pytest.approx(compute_composite_score(scores, weights), rel=1e-5) == expected

    # When diagnosis is missing (None), weights re-normalize over available components
    scores_partial = {"diagnosis": None, "symptoms": 1.0, "patient": 0.5, "visitType": 1.0}
    expected_partial = (0.4 * 1.0 + 0.1 * 0.5 + 0.1 * 1.0) / (0.4 + 0.1 + 0.1)
    assert pytest.approx(compute_composite_score(scores_partial, weights), rel=1e-5) == expected_partial


def test_case_service_candidate_ranking_and_filters():
    """Verify Candidate retrieval, scoring, tie-breaking, and Top-K limit."""
    mock_nlp = MagicMock()
    mock_nlp.analyze.side_effect = lambda term: _make_mock_doc(term, ent_text=term, is_negated=False)
    mock_nlp.analyze_batch.side_effect = lambda terms, **kwargs: [_make_mock_doc(t, ent_text=t, is_negated=False) for t in terms]
    assertion_svc = ClinicalAssertionService(nlp_service=mock_nlp)

    mock_embedder = MagicMock()
    mock_embedder.embed_entities.side_effect = lambda terms: np.ones((len(terms), 768), dtype=np.float32)

    case_svc = CaseService(
        assertion_service=assertion_svc,
        hpo_service=MagicMock(),
        mondo_service=MagicMock(),
        embedder=mock_embedder,
    )

    # 3 candidates in MongoDB knowledge base
    cand1 = {
        "caseNo": 201,
        "visitDate": "2024-02-01",
        "visitReason": "Follow-up Visit",
        "medication": ["paracetamol"],
        "patientInfo": {"age": 45.0, "gender": "m"},
        "diagnosis": [{"term": "insomnia", "embedding": [1.0] * 768}],
        "symptoms": [{"term": "fatigue", "embedding": [1.0] * 768}],
    }
    cand2 = {
        "caseNo": 202,
        "visitDate": "2024-01-15",
        "visitReason": "New Consultation",
        "medication": ["ibuprofen"],
        "patientInfo": {"age": 12.0, "gender": "f"},
        "diagnosis": [{"term": "asthma", "embedding": [0.0] * 768}],
        "symptoms": [{"term": "fever", "embedding": [0.0] * 768}],
    }

    mock_mongo = MagicMock()
    mock_mongo.find_past_cases.return_value = []
    mock_mongo.find_similar_candidates.return_value = [cand1, cand2]

    req = FindSimilarCaseRequest(
        case=CaseInput(
            caseNo=101,
            visitType=VisitTypeEnum.FOLLOW_UP,
            patientInfo=PatientInfoInput(age=45.0, gender="m"),
            symptoms=["fatigue"],
            diagnosis=["insomnia"],
        ),
        top_k=5,
        params=SimilarityParams(
            wDiagnosisScore=0.4,
            wSymptomsScore=0.4,
            wPatientScore=0.1,
            wVisitTypeScore=0.1,
        ),
    )

    res = case_svc.find_similar_cases(req, mongo_kb=mock_mongo)
    assert len(res.similarCases) == 2
    # Candidate 1 has exact diagnosis, symptom, age, and visitType -> highest score
    first = res.similarCases[0]
    second = res.similarCases[1]
    assert first.case["caseNo"] == 201
    assert first.finalScore > second.finalScore
    assert first.components.visitType == 1.0
    assert second.components.visitType == 0.5


def test_taxonomy_integer_lookups(tmp_path):
    """Test integer-based TaxonomyIndex Wu-Palmer with both int codes and CURIE strings."""
    from meldai.terminology.taxonomy import TaxonomyIndex

    obo_file = tmp_path / "test.obo"
    obo_file.write_text(
        "[Term]\nid: HP:0000001\nname: Root\n\n"
        "[Term]\nid: HP:0000002\nname: Level1\nis_a: HP:0000001\n\n"
        "[Term]\nid: HP:0000003\nname: SpecificCommon\nis_a: HP:0000002\n\n"
        "[Term]\nid: HP:0000004\nname: LeafA\nis_a: HP:0000003\n\n"
        "[Term]\nid: HP:0000005\nname: LeafB\nis_a: HP:0000003\n\n"
        "[Term]\nid: HP:0000006\nname: AltLeafB\nalt_id: HP:0000007\nis_a: HP:0000003\n"
    )

    tax = TaxonomyIndex(obo_file, prefix="HP", generic_depth=2)

    # Identical codes -> 1.0
    assert tax.wu_palmer(4, 4) == 1.0
    assert tax.wu_palmer("HP:0000004", 4) == 1.0

    # Common ancestor is 3 (depth 3 > generic_depth 2). Leaf depth = 4.
    # wu_palmer(4, 5) = 2 * 3 / (4 + 4) = 6 / 8 = 0.75
    score_int = tax.wu_palmer(4, 5)
    score_str = tax.wu_palmer("HP:0000004", "HP:0000005")
    assert pytest.approx(score_int, rel=1e-5) == 0.75
    assert pytest.approx(score_str, rel=1e-5) == 0.75

    # Alt ID mapping: 7 -> 6
    score_alt = tax.wu_palmer(4, 7)
    score_canonical = tax.wu_palmer(4, 6)
    assert pytest.approx(score_alt, rel=1e-5) == score_canonical

    # Missing term -> 0.0
    assert tax.wu_palmer(4, 99999) == 0.0


def test_similar_cases_patient_frequency_capping():
    """Verify that similar case selection caps cases from any single patient to floor(k/3)."""
    mock_nlp = MagicMock()
    mock_nlp.analyze.side_effect = lambda term: _make_mock_doc(term, ent_text=term, is_negated=False)
    mock_nlp.analyze_batch.side_effect = lambda terms, **kwargs: [_make_mock_doc(t, ent_text=t, is_negated=False) for t in terms]
    assertion_svc = ClinicalAssertionService(nlp_service=mock_nlp)

    mock_embedder = MagicMock()
    mock_embedder.embed_entities.side_effect = lambda terms: np.ones((len(terms), 768), dtype=np.float32)

    # 5 candidate cases for caseNo=200, 3 candidate cases for caseNo=300, 2 for caseNo=400
    candidates = [
        {"caseNo": 200, "visitDate": f"2024-01-0{i}", "symptoms": ["headache"], "diagnosis": ["fever"], "patientInfo": {"age": 30.0}}
        for i in range(1, 6)
    ] + [
        {"caseNo": 300, "visitDate": f"2024-02-0{i}", "symptoms": ["headache"], "diagnosis": ["fever"], "patientInfo": {"age": 30.0}}
        for i in range(1, 4)
    ] + [
        {"caseNo": 400, "visitDate": f"2024-03-0{i}", "symptoms": ["headache"], "diagnosis": ["fever"], "patientInfo": {"age": 30.0}}
        for i in range(1, 3)
    ]

    mock_mongo = MagicMock()
    mock_mongo.find_past_cases.return_value = []
    mock_mongo.find_similar_candidates.return_value = candidates

    case_svc = CaseService(
        assertion_service=assertion_svc,
        hpo_service=MagicMock(),
        mondo_service=MagicMock(),
        embedder=mock_embedder,
    )

    # Test top_k = 6 -> max_per_patient = floor(6/3) = 2
    req = FindSimilarCaseRequest(
        case=CaseInput(
            caseNo=100,
            visitType=VisitTypeEnum.FOLLOW_UP,
            symptoms=["headache"],
            diagnosis=["fever"],
            patientInfo=PatientInfoInput(age=30.0),
        ),
        top_k=6,
    )

    res = case_svc.find_similar_cases(req, mongo_kb=mock_mongo)
    assert len(res.similarCases) == 6
    case_no_counts = {}
    for item in res.similarCases:
        c_no = item.case["caseNo"]
        case_no_counts[c_no] = case_no_counts.get(c_no, 0) + 1
        assert case_no_counts[c_no] <= 2

    # Test top_k = 5 -> max_per_patient = floor(5/3) = 1
    req5 = FindSimilarCaseRequest(
        case=CaseInput(
            caseNo=100,
            visitType=VisitTypeEnum.FOLLOW_UP,
            symptoms=["headache"],
            diagnosis=["fever"],
            patientInfo=PatientInfoInput(age=30.0),
        ),
        top_k=5,
    )

    res5 = case_svc.find_similar_cases(req5, mongo_kb=mock_mongo)
    assert len(res5.similarCases) == 3  # Only 3 unique patients available (200, 300, 400)
    case_no_counts5 = {}
    for item in res5.similarCases:
        c_no = item.case["caseNo"]
        case_no_counts5[c_no] = case_no_counts5.get(c_no, 0) + 1
        assert case_no_counts5[c_no] <= 1



