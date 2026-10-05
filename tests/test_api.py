from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from meldai.main import create_web_app
from meldai.nlp.medspacy_nlp import SegmentResult
from meldai.terminology.hpo import HPOResult
from meldai.terminology.mondo import MONDOResult

client = TestClient(create_web_app())


def _make_segment_results(pheno_text, non_pheno_text):
    """Helper to produce mock SegmentResult objects."""
    return [
        SegmentResult(
            text=pheno_text,
            is_phenotype=True,
            embedding=[0.1] * 96,
            hpo_code=745,
        ),
        SegmentResult(
            text=non_pheno_text,
            is_phenotype=False,
            embedding=[],
            hpo_code=None,
        ),
    ]


def test_segment_symptoms_endpoint():
    mock_service = MagicMock()
    mock_service.process.return_value = _make_segment_results(
        "anxiety while driving", "60% better mood"
    )

    with patch("meldai.api.router.get_medspacy_service", return_value=mock_service):
        payload = {"note": "● anxiety while driving\n● 60% better mood"}
        response = client.post("/api/v1/utils/segment-symptoms", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "symptoms" in data
    assert len(data["symptoms"]) == 2

    # Phenotype segment
    pheno_item = data["symptoms"][0]
    assert pheno_item["note"] == "anxiety while driving"
    assert pheno_item["isPheno"] is True
    assert isinstance(pheno_item["embedding"], list)
    assert len(pheno_item["embedding"]) == 96
    assert pheno_item["hpoCode"] == 745

    # Non-phenotype segment
    non_pheno_item = data["symptoms"][1]
    assert non_pheno_item["note"] == "60% better mood"
    assert non_pheno_item["isPheno"] is False
    assert "embedding" not in non_pheno_item
    assert "hpoCode" not in non_pheno_item


def test_segment_symptoms_empty_note():
    response = client.post("/api/v1/utils/segment-symptoms", json={"note": ""})
    assert response.status_code == 200
    assert response.json()["symptoms"] == []


def test_segment_symptoms_nlp_error_raises_503():
    mock_service = MagicMock()
    mock_service.process.side_effect = RuntimeError("model not loaded")

    with patch("meldai.api.router.get_medspacy_service", return_value=mock_service):
        payload = {"note": "● headache"}
        response = client.post("/api/v1/utils/segment-symptoms", json=payload)

    assert response.status_code == 503


# ---------------------------------------------------------------------------
# HPO search with negation tests
# ---------------------------------------------------------------------------

def _make_mock_nlp_doc(text, ent_text="video games", is_negated=False, modifier_category=None):
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


def test_hpo_search_affirmed_endpoint():
    mock_nlp = MagicMock()
    mock_nlp.analyze.return_value = _make_mock_nlp_doc("playing video games", is_negated=False)

    mock_hpo = MagicMock()
    mock_hpo.search.return_value = [
        HPOResult(
            hpo_id="HP:5200336",
            label="Addictive video game use",
            synonyms=["Excessive video game playing"],
            match_type="token_label",
            score=0.83,
        )
    ]

    with patch("meldai.api.router.get_medspacy_service", return_value=mock_nlp), \
         patch("meldai.api.router.get_hpo_service", return_value=mock_hpo):
        resp = client.get("/api/v1/hpo/search", params={"term": "playing video games"})

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["hpo_id"] == "HP:5200336"
    assert data[0]["is_negated"] is False
    assert data[0]["is_phenotype"] is True
    assert data[0]["assertion_status"] == "affirmed"


def test_hpo_search_negated_filtered_endpoint():
    mock_nlp = MagicMock()
    mock_nlp.analyze.return_value = _make_mock_nlp_doc("not playing video games", is_negated=True)

    with patch("meldai.api.router.get_medspacy_service", return_value=mock_nlp):
        resp = client.get(
            "/api/v1/hpo/search",
            params={"term": "not playing video games", "filter_negated": "true"},
        )

    assert resp.status_code == 200
    assert resp.json() == []


def test_hpo_search_negated_default_endpoint():
    mock_nlp = MagicMock()
    mock_nlp.analyze.return_value = _make_mock_nlp_doc("not playing video games", is_negated=True)

    mock_hpo = MagicMock()
    mock_hpo.search.return_value = [
        HPOResult(
            hpo_id="HP:5200336",
            label="Addictive video game use",
            synonyms=["Excessive video game playing"],
            match_type="token_label",
            score=0.83,
        )
    ]

    with patch("meldai.api.router.get_medspacy_service", return_value=mock_nlp), \
         patch("meldai.api.router.get_hpo_service", return_value=mock_hpo):
        resp = client.get(
            "/api/v1/hpo/search",
            params={"term": "not playing video games"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["hpo_id"] == "HP:5200336"
    assert data[0]["is_negated"] is True
    assert data[0]["is_phenotype"] is False
    assert data[0]["assertion_status"] == "negated"


# ---------------------------------------------------------------------------
# Load Chief Complaints endpoint tests
# ---------------------------------------------------------------------------

def test_load_chief_complaints_success():
    mock_pg = MagicMock()
    mock_pg.stream_patient_chief_complaints.return_value = [
        [
            {
                "caseNo": "M1001",
                "visitDate": "2024-01-01",
                "chiefComplaints": "● headache\n● no fever",
            }
        ]
    ]

    mock_mongo = MagicMock()
    mock_mongo.find_case_by_encounter.return_value = {
        "_id": "case-uuid-1",
        "caseNo": "M1001",
        "patient_visit_id": "visit-uuid-1",
    }
    mock_mongo.update_case_symptoms.return_value = True

    mock_nlp = MagicMock()
    doc_affirmed = _make_mock_nlp_doc("headache", ent_text="headache", is_negated=False)
    doc_negated = _make_mock_nlp_doc("no fever", ent_text="fever", is_negated=True)
    mock_nlp.analyze.side_effect = [doc_affirmed, doc_negated]

    mock_hpo = MagicMock()
    mock_hpo.search.return_value = [
        HPOResult(
            hpo_id="HP:0002315",
            label="Headache",
            synonyms=["Cephalgia"],
            match_type="exact_label",
            score=1.0,
        )
    ]

    mock_mondo = MagicMock()
    mock_mondo.search.return_value = [
        MONDOResult(
            mondo_id="MONDO:0005555",
            label="Headache disorder",
            synonyms=[],
            match_type="exact_label",
            score=1.0,
        )
    ]

    with patch("meldai.api.router.PostgresSource", return_value=mock_pg), \
         patch("meldai.api.router.MongoKnowledgeBase", return_value=mock_mongo), \
         patch("meldai.api.router.get_medspacy_service", return_value=mock_nlp), \
         patch("meldai.api.router.get_hpo_service", return_value=mock_hpo), \
         patch("meldai.api.router.get_mondo_service", return_value=mock_mondo):
        resp = client.post("/api/v1/cases/migrate-patient-chief-complaints", params={"batch_size": 50})

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["batch_size"] == 50
    assert data["total_rows_processed"] == 1
    assert data["documents_updated"] == 1
    assert data["documents_skipped"] == 0
    assert data["batches_processed"] == 1

    # Verify update_case_symptoms was called with patient_visit_id
    mock_mongo.update_case_symptoms.assert_called_once()
    call_args = mock_mongo.update_case_symptoms.call_args[0]
    assert call_args[0] == "case-uuid-1"
    symptoms = call_args[1]
    assert len(symptoms) == 2
    assert symptoms[0]["note"] == "headache"
    assert symptoms[0]["patient_visit_id"] == "visit-uuid-1"
    assert symptoms[0]["hpoCode"] == 2315
    assert symptoms[0]["mondoCode"] == 5555
    assert symptoms[0]["is_negated"] is False
    assert symptoms[1]["note"] == "no fever"
    assert symptoms[1]["is_negated"] is True


def test_migrate_patient_chief_complaints_skipped_rows():
    mock_pg = MagicMock()
    mock_pg.stream_patient_chief_complaints.return_value = [
        [
            # Case 1: Empty note
            {"caseNo": "M1001", "visitDate": "2024-01-01", "chiefComplaints": "   "},
            # Case 2: MongoDB document not found
            {"caseNo": "M1002", "visitDate": "2024-01-02", "chiefComplaints": "headache"},
        ]
    ]

    mock_mongo = MagicMock()
    mock_mongo.find_case_by_encounter.return_value = None

    with patch("meldai.api.router.PostgresSource", return_value=mock_pg), \
         patch("meldai.api.router.MongoKnowledgeBase", return_value=mock_mongo):
        resp = client.post("/api/v1/cases/migrate-patient-chief-complaints", params={"batch_size": 100})

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["total_rows_processed"] == 2
    assert data["documents_updated"] == 0
    assert data["documents_skipped"] == 2


def test_migrate_patient_chief_complaints_failure_raises_500():
    mock_pg = MagicMock()
    mock_pg.stream_patient_chief_complaints.side_effect = RuntimeError("PostgreSQL connection lost")

    with patch("meldai.api.router.PostgresSource", return_value=mock_pg), \
         patch("meldai.api.router.MongoKnowledgeBase", return_value=MagicMock()):
        resp = client.post("/api/v1/cases/migrate-patient-chief-complaints", params={"batch_size": 100})

    assert resp.status_code == 500
    assert "PostgreSQL connection lost" in resp.json()["detail"]


def test_find_similar_cases_endpoint():
    mock_case_svc = MagicMock()
    mock_case_svc.find_similar_cases.return_value = {
        "decoratedCase": {
            "caseNo": 999,
            "patientInfo": {"age": 50, "gender": "Female"},
            "symptoms": [
                {
                    "term": "lack of sleep",
                    "hpoTerm": "Insomnia",
                    "hpoCode": 2360,
                    "embedding": [0.1] * 768,
                    "isNegation": False,
                    "isPheno": True,
                    "similarity": None,
                }
            ],
            "diagnosis": [
                {
                    "term": "Insomia",
                    "mondoTerm": "insomnia",
                    "mondoCode": 8807,
                    "embedding": [0.1] * 768,
                    "similarity": None,
                }
            ],
        },
        "similarCases": [],
    }

    with patch("meldai.api.router._get_case_service", return_value=mock_case_svc):
        payload = {
            "caseNo": 999,
            "patientInfo": {"age": 50, "gender": "Female"},
            "symptoms": ["lack of sleep"],
            "diagnosis": ["Insomia"],
        }
        resp = client.post("/api/v1/cases/find-similar", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert "decoratedCase" in data
    assert data["decoratedCase"]["caseNo"] == 999
    assert data["decoratedCase"]["patientInfo"]["age"] == 50
    assert data["decoratedCase"]["patientInfo"]["gender"] == "Female"
    assert len(data["decoratedCase"]["symptoms"]) == 1
    assert data["decoratedCase"]["symptoms"][0]["term"] == "lack of sleep"
    assert data["decoratedCase"]["symptoms"][0]["hpoCode"] == 2360
    assert len(data["decoratedCase"]["symptoms"][0]["embedding"]) == 768
    assert len(data["decoratedCase"]["diagnosis"]) == 1
    assert data["decoratedCase"]["diagnosis"][0]["term"] == "Insomia"
    assert data["decoratedCase"]["diagnosis"][0]["mondoCode"] == 8807
    assert len(data["decoratedCase"]["diagnosis"][0]["embedding"]) == 768
    assert data["similarCases"] == []

