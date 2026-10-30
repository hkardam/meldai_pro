"""Unit tests for MeldAI V2 independent 3-step LLM pipeline and DTO relocation."""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

# Verify DTOs defined in meldai_v2
from meldai_v2.req_dtos import (
    CaseInput,
    DeprescribeRequest,
    MedCheckRequest,
    PatientInfoInput,
    PrescribeMedicationsV2Request,
    RankMoleculesRequest,
    SimilarityParams,
    SimilarityWeightsInput,
    VisitTypeEnum,
)
from meldai_v2.res_dtos import (
    ClinicalDeltaSummary,
    DeprescribeResponse,
    DeprescribingRecommendation,
    EntityMoleculeRow,
    LLMStepCall,
    MedCheckResponse,
    PrescribeMedicationsV2Response,
    PrescriptionMemoryState,
    RankMoleculesResponse,
    RankedMoleculeItem,
)

# Verify backward-compatibility re-exports from core.dtos
from core.dtos.req_dtos import (
    PrescribeMedicationsV2Request as CorePrescribeMedicationsV2Request,
    SimilarityWeightsInput as CoreSimilarityWeightsInput,
)
from core.dtos.res_dtos import (
    DeprescribingRecommendation as CoreDeprescribingRecommendation,
    EntityMoleculeRow as CoreEntityMoleculeRow,
    PrescribeMedicationsV2Response as CorePrescribeMedicationsV2Response,
)

from meldai_v2.services.prescription_service import (
    PrescriptionServiceV2,
    build_unified_triage_deprescribe_prompt,
    build_llm_med_check_prompt,
)
from main import create_web_app


def test_dto_relocation_and_compatibility():
    """Verify that V2 DTOs are canonically defined in meldai_v2 and re-exported in core.dtos."""
    assert PrescribeMedicationsV2Request is CorePrescribeMedicationsV2Request
    assert SimilarityWeightsInput is CoreSimilarityWeightsInput
    assert PrescribeMedicationsV2Response is CorePrescribeMedicationsV2Response
    assert EntityMoleculeRow is CoreEntityMoleculeRow
    assert DeprescribingRecommendation is CoreDeprescribingRecommendation
    assert PrescriptionMemoryState is PrescribeMedicationsV2Response


def test_build_unified_triage_deprescribe_prompt():
    """Verify build_unified_triage_deprescribe_prompt populates intake, history, and prior medications."""
    keyword_results = [
        {
            "complaint": "Sadness of mood improved",
            "category": "symptom",
            "display_kw": "improved",
            "matched_past": "None",
            "canonical_entity": "Sadness of mood",
        }
    ]
    past_cases = [
        {
            "caseDate": "2026-01-01",
            "visitType": "new consultation",
            "chiefComplaints": [{"term": "Sadness of mood"}],
            "diagnosis": [{"term": "MDD"}],
            "medications": ["Escitalopram 10mg"],
        }
    ]
    prompt = build_unified_triage_deprescribe_prompt(
        keyword_results=keyword_results,
        past_cases=past_cases,
        prior_medications=["Escitalopram 10mg"],
        patient_info={"age": 40, "gender": "Male"},
        visit_type="follow-up",
        current_date="2026-01-24",
    )
    assert "<current-intake>" in prompt
    assert "Sadness of mood improved" in prompt
    assert "<past-visit-history>" in prompt
    assert "Escitalopram 10mg" in prompt
    assert "triaged_items" in prompt
    assert "deprescribing_recommendations" in prompt


def test_deprescribe_step_returns_llm_response_and_whats_changed():
    """Step 1: Verify deprescribe_step triggers the unified LLM call, returns LLM response,
    reports whats_changed as a string field, and returns the unified state.
    """
    mock_settings = MagicMock()
    mock_settings.medicine_master_json_path = "data/medication/local_medicine_master.json"

    mock_gemini = MagicMock()
    mock_gemini.is_configured = True
    mock_gemini.generate_prescription.return_value = {
        "whats_changed": "Discontinued Paracetamol as headache has fully resolved; decreased Clonazepam to taper.",
        "deprescribing_recommendations": [
            {
                "medicine": "Paracetamol",
                "target_symptom": "headache",
                "status_change": "stop",
                "action": "stop",
                "rationale": "Headache completely resolved.",
                "taper_plan": None,
            },
            {
                "medicine": "Clonazepam",
                "target_symptom": "insomnia",
                "status_change": "decreased",
                "action": "decreased",
                "rationale": "Sleep improved, taper benzodiazepine to avoid dependence.",
                "taper_plan": "Reduce to 0.25mg for 7 days then discontinue.",
            },
        ],
        "response_time_seconds": 0.45,
        "tokens_used": {"prompt_tokens": 80, "candidates_tokens": 40, "total_tokens": 120},
        "cost_usd": 0.0008,
    }

    mock_mongo = MagicMock()
    mock_mongo.find_past_cases.return_value = [
        {
            "caseNo": 201,
            "caseDate": "2025-12-01",
            "medications": ["Clonazepam", "Paracetamol"],
            "symptoms": ["headache", "insomnia"],
        }
    ]

    mock_case_svc = MagicMock()
    decorated_case = MagicMock()
    decorated_case.symptoms = []
    decorated_case.diagnosis = []
    mock_case_svc.decorate_case.return_value = decorated_case

    mock_chief_svc = MagicMock()
    assessment = MagicMock()
    assessment.active_symptoms = ["insomnia"]
    assessment.symptom_notes = ""
    assessment.patient_condition = ""
    assessment.triaged_items = []
    assessment.llm_prompt = "mock-unified-prompt"
    assessment.triage_delta = None
    # Unified result: deprescribing already returned by assess_and_deprescribe
    assessment.deprescribing_recommendations = [
        {
            "medicine": "Paracetamol",
            "target_symptom": "headache",
            "status_change": "stop",
            "action": "stop",
            "rationale": "Headache completely resolved.",
            "taper_plan": None,
        },
        {
            "medicine": "Clonazepam",
            "target_symptom": "insomnia",
            "status_change": "decreased",
            "action": "decreased",
            "rationale": "Sleep improved, taper benzodiazepine to avoid dependence.",
            "taper_plan": "Reduce to 0.25mg for 7 days then discontinue.",
        },
    ]
    assessment.whats_changed = "Discontinued Paracetamol as headache has fully resolved; decreased Clonazepam to taper."
    mock_chief_svc.assess_and_deprescribe.return_value = assessment

    svc = PrescriptionServiceV2(
        settings=mock_settings,
        case_service=mock_case_svc,
        mongo_kb=mock_mongo,
        gemini_service=mock_gemini,
        chief_complaint_service=mock_chief_svc,
    )

    req = DeprescribeRequest(
        case=CaseInput(
            caseNo=201,
            patientInfo=PatientInfoInput(age=40, gender="Male"),
            symptoms=["insomnia"],
            diagnosis=["chronic insomnia"],
            visitType=VisitTypeEnum.FOLLOW_UP,
        )
    )

    resp = svc.deprescribe_step(req)

    assert isinstance(resp, DeprescribeResponse)
    assert isinstance(resp.whats_changed, str)
    assert "Discontinued Paracetamol" in resp.whats_changed
    assert resp.llm_response is not None
    assert len(resp.deprescribing_recommendations) == 2
    assert resp.deprescribing_recommendations[0].action == "stop"
    assert resp.deprescribing_recommendations[1].action == "decreased"

    # Verify state DTO is populated
    state = resp.state
    assert isinstance(state, PrescribeMedicationsV2Response)
    assert state.whats_changed == resp.whats_changed
    assert state.deprescribe is not None
    assert len(state.deprescribing_recommendations) == 2


def test_med_check_step_performs_deprescribe_and_fills_empty_entities():
    """Step 2: Verify med_check_step uses the unified assess_and_deprescribe call,
    identifies entities with 0 medications, batch fills them from local formulary,
    reports what changed as a string field, and updates the state.
    """
    mock_settings = MagicMock()
    mock_settings.medicine_master_json_path = "data/medication/local_medicine_master.json"

    mock_gemini = MagicMock()
    mock_gemini.is_configured = True
    # Step 2 med check LLM response (batch fills the 0-med entity "panic attacks")
    med_check_llm = {
        "whats_changed": "Assigned Clonazepam from formulary for entity 'panic attacks' which had 0 candidate matches.",
        "entity_medication_check": [
            {
                "entity": "panic attacks",
                "entity_type": "symptom",
                "directly_targeted": True,
                "indirectly_targeted": False,
                "attributed_molecules": ["Clonazepam"],
                "recommended_top_molecules": [
                    {
                        "canonical_molecule": "Clonazepam",
                        "is_fdc": False,
                        "doctor_brands": ["LONAZEP"],
                        "frequency": 12,
                        "score": 9.2,
                        "percentage": 85.0,
                        "targeting_type": "direct",
                        "rationale": "High-potency benzodiazepine for acute panic episodes.",
                    }
                ],
            }
        ],
        "summary": "Panic attacks covered from formulary catalog.",
        "response_time_seconds": 0.5,
        "tokens_used": {"prompt_tokens": 90, "candidates_tokens": 45, "total_tokens": 135},
        "cost_usd": 0.0009,
    }

    # Unified call result returned from generate_content_with_metadata is not used
    # because we mock assess_and_deprescribe directly.
    mock_gemini.generate_prescription.side_effect = [med_check_llm]

    mock_mongo = MagicMock()
    mock_mongo.find_past_cases.return_value = [
        {"caseNo": 305, "medications": ["Sertraline"], "symptoms": ["depression"]}
    ]
    mock_mongo.find_similar_candidates.return_value = []

    mock_case_svc = MagicMock()
    decorated_case = MagicMock()
    decorated_case.symptoms = []
    decorated_case.diagnosis = []
    mock_case_svc.decorate_case.return_value = decorated_case

    mock_chief_svc = MagicMock()
    assessment = MagicMock()
    assessment.active_symptoms = ["depression", "panic attacks"]
    assessment.symptom_notes = ""
    assessment.patient_condition = ""
    assessment.triaged_items = []
    assessment.llm_prompt = "mock-unified-prompt"
    assessment.triage_delta = None
    # Unified path: deprescribing recs come from assess_and_deprescribe
    assessment.deprescribing_recommendations = []
    assessment.whats_changed = "Maintained baseline regimen; no drug discontinuation needed."
    mock_chief_svc.assess_and_deprescribe.return_value = assessment

    # Entity matrix: "depression" has 1 match; "panic attacks" has 0 matches!
    mock_entity_ranking_svc = MagicMock()
    mock_entity_ranking_svc.build_entity_matrix.return_value = [
        EntityMoleculeRow(
            entity="depression",
            entity_type="symptom",
            top_molecules=[
                {
                    "canonical_molecule": "Sertraline",
                    "frequency": 15,
                    "percentage": 75.0,
                    "score": 8.5,
                    "doctor_brands": ["SERTA"],
                }
            ],
        ),
        EntityMoleculeRow(
            entity="panic attacks",
            entity_type="symptom",
            top_molecules=[],  # 0 medications!
        ),
    ]

    svc = PrescriptionServiceV2(
        settings=mock_settings,
        case_service=mock_case_svc,
        mongo_kb=mock_mongo,
        gemini_service=mock_gemini,
        entity_ranking_service=mock_entity_ranking_svc,
        chief_complaint_service=mock_chief_svc,
    )

    req = MedCheckRequest(
        case=CaseInput(
            caseNo=305,
            patientInfo=PatientInfoInput(age=29, gender="Female"),
            symptoms=["depression", "panic attacks"],
            diagnosis=["Major Depressive Disorder with Panic Features"],
            visitType=VisitTypeEnum.FOLLOW_UP,
        )
    )

    resp = svc.med_check_step(req)

    assert isinstance(resp, MedCheckResponse)
    assert isinstance(resp.whats_changed, str)
    assert "Assigned Clonazepam" in resp.whats_changed
    assert "panic attacks" in resp.filled_entities

    # Verify entity matrix was updated with newly filled molecule
    panic_row = next(r for r in resp.entity_matrix if r.entity == "panic attacks")
    assert len(panic_row.top_molecules) > 0
    assert panic_row.top_molecules[0]["canonical_molecule"] == "Clonazepam"

    # Verify unified memory state
    state = resp.state
    assert isinstance(state, PrescribeMedicationsV2Response)
    assert state.deprescribe is not None  # Deprescribe was run internally
    assert state.llmMedCheck is not None
    assert "panic attacks" in state.filled_entities


def test_api_endpoints_deprescribe_and_med_check():
    """Verify HTTP API endpoints:
      POST /api/v2/cases/deprescribe
      POST /api/v2/cases/med-check
    """
    mock_svc = MagicMock()
    mock_dep_resp = DeprescribeResponse(
        whats_changed="Discontinued Paracetamol as headache has fully resolved.",
        llm_response={"test": "ok"},
        deprescribing_recommendations=[],
        clinical_summary="Patient 42M with MDD.",
        state=PrescribeMedicationsV2Response(clinical_summary="Patient 42M with MDD."),
    )
    mock_check_resp = MedCheckResponse(
        whats_changed="Filled candidate medications for severe insomnia from master formulary.",
        llm_response={"test": "ok"},
        filled_entities=["Severe insomnia"],
        entity_matrix=[],
        state=PrescribeMedicationsV2Response(clinical_summary="Patient 42M with MDD."),
    )
    mock_svc.deprescribe_step.return_value = mock_dep_resp
    mock_svc.med_check_step.return_value = mock_check_resp

    case_payload = {
        "case": {
            "caseNo": 999,
            "caseDate": "2026-02-01",
            "visitType": "new consultation",
            "patientInfo": {"age": 42, "gender": "Male"},
            "symptoms": ["Sadness of mood", "Severe insomnia"],
            "diagnosis": ["MDD"],
        }
    }

    with patch("meldai_v2.router._get_prescription_service_v2", return_value=mock_svc):
        app = create_web_app()
        client = TestClient(app)

        # 1. Test POST /api/v2/cases/deprescribe
        dep_res = client.post("/api/v2/cases/deprescribe", json=case_payload)
        assert dep_res.status_code == 200, dep_res.text
        dep_data = dep_res.json()
        assert "whats_changed" in dep_data
        assert "Discontinued Paracetamol" in dep_data["whats_changed"]
        assert "state" in dep_data
        assert "deprescribing_recommendations" in dep_data

        # 2. Test POST /api/v2/cases/med-check
        check_res = client.post("/api/v2/cases/med-check", json=case_payload)
        assert check_res.status_code == 200, check_res.text
        check_data = check_res.json()
        assert "whats_changed" in check_data
        assert "Filled candidate medications" in check_data["whats_changed"]
        assert "state" in check_data
        assert "entity_matrix" in check_data
