"""Unit tests for GeminiService, PrescriptionService, and /cases/prescribe-medications endpoint."""

import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from meldai.api.req_dtos import (
    CaseInput,
    PatientInfoInput,
    PrescribeMedicationsRequest,
    VisitTypeEnum,
)
from meldai.api.res_dtos import (
    ComponentScores,
    DecoratedCase,
    DecoratedDiagnosisItem,
    DecoratedPatientInfo,
    DecoratedSymptomItem,
    FindSimilarCaseResponse,
    SimilarCaseItem,
)
from meldai.main import create_web_app
from meldai.services.gemini_service import GeminiService
from meldai.services.prescription_service import PrescriptionService

# ---------------------------------------------------------------------------
# 1. GeminiService Tests
# ---------------------------------------------------------------------------

def test_gemini_service_mock_when_not_configured():
    """GeminiService produces deterministic mock response when no API key is provided."""
    service = GeminiService(api_key=None)
    assert not service.is_configured

    result = service.generate_prescription("Patient has severe insomnia and fatigue.")
    assert "medications" in result
    assert isinstance(result["medications"], list)
    assert len(result["medications"]) > 0
    assert "explanation" in result
    assert isinstance(result["explanation"], str)


def test_gemini_service_parses_json_with_markdown_fence():
    """GeminiService strips ```json ... ``` formatting safely."""
    service = GeminiService(api_key="mock_key")
    markdown_json = (
        '```json\n'
        '{\n'
        '  "medications": ["Amoxicillin 500mg", "Paracetamol 650mg"],\n'
        '  "explanation": "Standard antimicrobial regimen for bacterial pharyngitis."\n'
        '}\n'
        '```'
    )
    with patch.object(service, "generate_content", return_value=markdown_json):
        res = service.generate_prescription("sample prompt")
        assert res["medications"] == ["Amoxicillin 500mg", "Paracetamol 650mg"]
        assert "antimicrobial regimen" in res["explanation"]


def test_gemini_service_live_call_success():
    """GeminiService calls google.genai client.models.generate_content correctly."""
    service = GeminiService(api_key="valid-test-key")
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({
        "medications": ["Metformin 500mg"],
        "explanation": "Initiation for type 2 diabetes.",
    })

    with patch.object(service.client.models, "generate_content", return_value=mock_resp) as mock_gen:
        res = service.generate_prescription("prompt text")
        assert res["medications"] == ["Metformin 500mg"]
        assert res["explanation"] == "Initiation for type 2 diabetes."
        mock_gen.assert_called_once()
        args, kwargs = mock_gen.call_args
        assert kwargs["model"] == service._model_name
        assert kwargs["contents"] == "prompt text"
        assert kwargs["config"].response_mime_type == "application/json"


# ---------------------------------------------------------------------------
# 2. PrescriptionService Prompt Builder Tests (No Embeddings Attached)
# ---------------------------------------------------------------------------

def test_prepare_prompt_omits_dense_embeddings():
    """Verify that dense vector embeddings are strictly excluded from the LLM prompt."""
    service = PrescriptionService()

    dummy_vector = [0.1234, -0.5678, 0.9999] + [0.0] * 765

    decorated_case = DecoratedCase(
        caseNo=101,
        visitType="follow-up",
        patientInfo=DecoratedPatientInfo(age=45.0, gender="Male"),
        symptoms=[
            DecoratedSymptomItem(
                term="frequent nighttime waking",
                hpoTerm="Sleep disturbance",
                hpoCode=2360,
                embedding=dummy_vector,
                isNegation=False,
                isPheno=True,
            ),
            DecoratedSymptomItem(
                term="no fever",
                hpoTerm="Fever",
                hpoCode=1945,
                embedding=dummy_vector,
                isNegation=True,
                isPheno=False,
            ),
        ],
        diagnosis=[
            DecoratedDiagnosisItem(
                term="Chronic Insomnia",
                mondoTerm="insomnia (disease)",
                mondoCode=8807,
                embedding=dummy_vector,
            )
        ],
    )

    past_cases = [
        {
            "visitDate": "2024-01-10",
            "visitReason": "Initial Consult",
            "diagnosis": [{"term": "Insomnia", "mondoCode": 8807, "embedding": dummy_vector}],
            "symptoms": [{"term": "difficulty sleeping", "hpoCode": 2360, "embedding": dummy_vector}],
            "medications": ["Melatonin 5mg"],
        }
    ]

    similar_cases = [
        SimilarCaseItem(
            case={
                "caseNo": 999,
                "visitReason": "follow-up",
                "patientInfo": {"age": 42.0, "gender": "m"},
                "diagnosis": [{"term": "Insomnia", "mondoCode": 8807, "embedding": dummy_vector}],
                "symptoms": [{"term": "insomnia", "hpoCode": 2360, "embedding": dummy_vector}],
                "medications": ["Zolpidem 5mg", "Melatonin 3mg"],
            },
            finalScore=0.96,
            components=ComponentScores(diagnosis=0.98, symptoms=0.94, patient=0.95, visitType=1.0),
        )
    ]

    prompt = service.prepare_prompt(
        decorated_case=decorated_case,
        past_cases=past_cases,
        similar_cases=similar_cases,
    )

    # 1. Critical assertion: NO embedding vector floats in the prompt
    assert "0.1234" not in prompt
    assert "-0.5678" not in prompt
    assert "embedding" not in prompt.lower()

    # 2. Key information MUST be present with canonical CURIEs and no patient identifiers
    assert "Case Number" not in prompt
    assert "Patient ID" not in prompt
    assert "101" not in prompt  # Dropped patient identifier
    assert "Chronic Insomnia" in prompt
    assert "MONDO:0008807" in prompt
    assert "HP:0002360" in prompt
    assert "Pertinent Negatives:" in prompt
    assert "NEGATED" in prompt
    assert "Unmatched" not in prompt
    assert "Melatonin 5mg" in prompt  # Past medication
    assert "Zolpidem 5mg" in prompt   # Similar case medication
    assert "0.96" in prompt           # Similarity score


def test_prescription_service_flow():
    """Verify end-to-end execution of prescription service with mocked dependencies."""
    mock_case_svc = MagicMock()
    mock_gemini = MagicMock()

    mock_gemini.generate_prescription.return_value = {
        "medications": ["Zolpidem 5mg", "Melatonin 3mg"],
        "explanation": "Consistent with patient's prior sleep management and similar cases.",
    }

    dummy_decorated = DecoratedCase(
        caseNo=50,
        visitType="follow-up",
        patientInfo=DecoratedPatientInfo(age=35.0, gender="Female"),
        symptoms=[
            DecoratedSymptomItem(term="headache", hpoTerm="Headache", hpoCode=2315, embedding=None)
        ],
        diagnosis=[
            DecoratedDiagnosisItem(term="Migraine", mondoTerm="migraine", mondoCode=5432, embedding=None)
        ],
    )

    mock_case_svc.find_similar_cases.return_value = FindSimilarCaseResponse(
        decoratedCase=dummy_decorated,
        pastCases=[{"visitDate": "2024-01-01", "medications": ["Sumatriptan 50mg"]}],
        similarCases=[
            SimilarCaseItem(
                case={"caseNo": 80, "medications": ["Sumatriptan 50mg", "Naproxen 500mg"]},
                finalScore=0.92,
                components=ComponentScores(),
            )
        ],
    )

    presc_service = PrescriptionService(
        case_service=mock_case_svc,
        gemini_service=mock_gemini,
        mongo_kb=MagicMock(),
    )

    req = PrescribeMedicationsRequest(
        case=CaseInput(
            caseNo=50,
            visitType=VisitTypeEnum.FOLLOW_UP,
            patientInfo=PatientInfoInput(age=35.0, gender="Female"),
            symptoms=["headache"],
            diagnosis=["Migraine"],
        ),
        top_k=3,
    )

    res = presc_service.prescribe_medications(req)

    assert res.medications == ["Zolpidem 5mg", "Melatonin 3mg"]
    assert "patient's prior sleep management" in res.explanation
    assert res.pastCasesCount == 1
    assert res.similarCasesCount == 1

    mock_case_svc.find_similar_cases.assert_called_once()
    mock_gemini.generate_prescription.assert_called_once()


# ---------------------------------------------------------------------------
# 3. FastAPI Endpoint Tests
# ---------------------------------------------------------------------------

def test_prescribe_medications_endpoint():
    """Verify HTTP POST /api/v1/cases/prescribe-medications."""
    app = create_web_app()
    client = TestClient(app)

    mock_presc_svc = MagicMock()
    from meldai.api.res_dtos import PrescribeMedicationsResponse

    mock_presc_svc.prescribe_medications.return_value = PrescribeMedicationsResponse(
        medications=["Metformin 500mg"],
        explanation="First-line glycemic management.",
        pastCasesCount=0,
        similarCasesCount=2,
    )

    with patch("meldai.api.router._get_prescription_service", return_value=mock_presc_svc):
        payload = {
            "case": {
                "caseNo": 1234,
                "visitType": "new consultation",
                "patientInfo": {"age": 55, "gender": "Male"},
                "symptoms": ["increased thirst"],
                "diagnosis": ["Type 2 Diabetes"],
            },
            "top_k": 5,
        }
        resp = client.post("/api/v1/cases/prescribe-medications", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data["medications"] == ["Metformin 500mg"]
    assert "First-line glycemic management" in data["explanation"]
    assert data["pastCasesCount"] == 0
    assert data["similarCasesCount"] == 2


def test_prescription_service_reads_active_prompt_file():
    """Verify that PrescriptionService dynamically reads and embeds active_prompt.md."""
    from meldai.services.prescription_service import ACTIVE_PROMPT_PATH, get_active_prompt

    active_text = get_active_prompt()
    assert len(active_text) > 0
    assert "psychiatrist practicing in India" in active_text
    assert "<data>" in active_text

    service = PrescriptionService()
    decorated_case = DecoratedCase(
        caseNo=777,
        visitType="follow-up",
        patientInfo=DecoratedPatientInfo(age=30.0, gender="Female"),
        symptoms=[],
        diagnosis=[],
    )
    prompt = service.prepare_prompt(decorated_case=decorated_case, past_cases=[], similar_cases=[])
    assert "psychiatrist practicing in India" in prompt
    assert "<data>" in prompt
    assert "</data>" in prompt
