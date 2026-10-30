"""Unit tests for /deprescribe state DTO prompt de-duplication and 4-way medication distinction.

Verifies:
1. The prompt is stored only once at state.deprescribe.prompt and not duplicated at state.triage_prompt.
2. Distinguishes previous prescriptions from current medication use:
   - historical_prescription: distinct medicines prescribed across multiple encounters.
   - latest_documented_prescription: medications prescribed in the most recent prior encounter.
   - patient_reported_current_use: medications the patient actually reports taking today.
   - actual_discontinuation: medications identified as stopped or discontinued.
"""

from unittest.mock import MagicMock
import pytest

from meldai_v2.req_dtos import (
    CaseInput,
    DeprescribeRequest,
    PatientInfoInput,
    VisitTypeEnum,
)
from meldai_v2.res_dtos import (
    ClinicalDeltaSummary,
    DeprescribeResponse,
    DeprescribingRecommendation,
    LLMStepCall,
    PrescribeMedicationsV2Response,
)
from meldai_v2.services.prescription_service import PrescriptionServiceV2
from meldai_v2.services.clinical_delta_service import (
    ChiefComplaintChangeService,
    ClinicalChangeAssessment,
    compute_delta,
    _extract_patient_reported_meds,
    _extract_discontinued_meds,
)


def test_state_dto_no_duplicate_prompt_serialization():
    """Verify state DTO stores prompt once at state.deprescribe.prompt and does not serialize triage_prompt twice."""
    large_prompt = "ROLE: You are an expert clinical physician reviewing a patient encounter..." * 100
    state = PrescribeMedicationsV2Response()
    state.triage_prompt = large_prompt

    # Accessors work
    assert state.triage_prompt == large_prompt
    assert state.deprescribe is not None
    assert state.deprescribe.prompt == large_prompt

    # Serialization does NOT duplicate triage_prompt
    dumped = state.model_dump()
    assert "triage_prompt" not in dumped, "triage_prompt should NOT be serialized as a duplicate top-level field"
    assert dumped["deprescribe"]["prompt"] == large_prompt

    # Deserialization migrates triage_prompt into deprescribe.prompt
    migrated = PrescribeMedicationsV2Response.model_validate({"triage_prompt": large_prompt})
    assert migrated.deprescribe.prompt == large_prompt
    assert "triage_prompt" not in migrated.model_dump()


def test_extract_patient_reported_meds_and_discontinuation():
    """Verify heuristic extraction detects patient active intake and discontinuation."""
    intake = [
        "currently taking tab clonazepam 0.5mg hs",
        "on escitalopram 10mg",
        "stopped fluoxetine 20mg 2 weeks ago",
        "not taking any sleeping pills",
        "tension headache",
    ]
    current_use = _extract_patient_reported_meds(intake)
    assert any("clonazepam" in m.lower() for m in current_use)
    assert any("escitalopram" in m.lower() for m in current_use)
    # Stopped or negated meds should not be in current use
    assert not any("fluoxetine" in m.lower() for m in current_use)
    assert not any("sleeping" in m.lower() for m in current_use)

    discontinued = _extract_discontinued_meds(intake)
    assert any("fluoxetine" in m.lower() for m in discontinued)


def test_compute_delta_distinguishes_medication_fields():
    """Verify compute_delta separates historical, latest documented, patient-reported, and discontinuation."""
    past_cases = [
        {
            "caseNo": 301,
            "caseDate": "2025-10-01",
            "visitType": "new consultation",
            "chiefComplaints": [{"term": "depression"}],
            "medications": ["Fluoxetine 20mg", "Clonazepam 0.5mg"],
        },
        {
            "caseNo": 301,
            "caseDate": "2025-12-15",
            "visitType": "follow-up",
            "chiefComplaints": [{"term": "depression improved"}],
            "medications": ["Escitalopram 10mg"],
        },
    ]

    current_symptoms = [
        "taking escitalopram 10mg hs",
        "stopped fluoxetine",
        "mild anxiety",
    ]

    assessment = ClinicalChangeAssessment(
        active_symptoms=["mild anxiety"],
        deprescribing_recommendations=[
            {
                "medicine": "Clonazepam 0.5mg",
                "action": "stop",
                "status_change": "resolved",
                "target_symptom": "anxiety",
                "rationale": "Anxiety resolved; stop benzodiazepine.",
            }
        ],
    )

    delta = compute_delta(
        current_symptoms=current_symptoms,
        current_diagnoses=["Major Depressive Disorder"],
        past_cases=past_cases,
        change_assessment=assessment,
    )

    # 1. Historical prescriptions contains all past prescribed meds
    assert "Fluoxetine 20mg" in delta["historical_prescription"]
    assert "Clonazepam 0.5mg" in delta["historical_prescription"]
    assert "Escitalopram 10mg" in delta["historical_prescription"]

    # 2. Latest documented prescription contains ONLY the latest visit subset
    assert delta["latest_documented_prescription"] == ["Escitalopram 10mg"]

    # 3. Patient-reported current use contains what patient is taking today
    assert any("escitalopram" in m.lower() for m in delta["patient_reported_current_use"])
    assert not any("clonazepam" in m.lower() for m in delta["patient_reported_current_use"])

    # 4. Actual discontinuation contains both intake report and deprescribing stop recs
    assert any("fluoxetine" in m.lower() for m in delta["actual_discontinuation"])
    assert any("clonazepam" in m.lower() for m in delta["actual_discontinuation"])


def test_deprescribe_step_service_populates_all_four_fields():
    """Verify deprescribe_step populates all 4 fields on both DeprescribeResponse and state."""
    mock_settings = MagicMock()
    mock_settings.medicine_master_json_path = "data/medication/local_medicine_master.json"

    mock_gemini = MagicMock()
    mock_gemini.is_configured = True

    mock_mongo = MagicMock()
    mock_mongo.find_past_cases.return_value = [
        {
            "caseNo": 501,
            "caseDate": "2025-06-01",
            "medications": ["Sertraline 50mg", "Alprazolam 0.25mg"],
            "symptoms": ["panic attacks", "depression"],
        },
        {
            "caseNo": 501,
            "caseDate": "2025-09-01",
            "medications": ["Sertraline 100mg"],
            "symptoms": ["depression"],
        },
    ]

    mock_case_svc = MagicMock()
    decorated_case = MagicMock()
    decorated_case.symptoms = []
    decorated_case.diagnosis = []
    mock_case_svc.decorate_case.return_value = decorated_case

    mock_chief_svc = MagicMock()
    assessment = ClinicalChangeAssessment(
        active_symptoms=["mild sadness"],
        resolved_symptoms=["panic attacks"],
        deprescribing_recommendations=[
            {
                "medicine": "Alprazolam 0.25mg",
                "action": "stop",
                "status_change": "resolved",
                "target_symptom": "panic attacks",
                "rationale": "Panic attacks resolved; stop benzodiazepine.",
                "taper_plan": None,
            }
        ],
        whats_changed="Alprazolam discontinued; Sertraline maintained.",
        historical_prescription=["Sertraline 50mg", "Alprazolam 0.25mg", "Sertraline 100mg"],
        latest_documented_prescription=["Sertraline 100mg"],
        patient_reported_current_use=["Sertraline 100mg"],
        actual_discontinuation=["Alprazolam 0.25mg"],
        llm_prompt="UNIFIED PROMPT 6.3 KB CONTENT",
    )
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
            caseNo="501",
            patientInfo=PatientInfoInput(age=32, gender="Female"),
            symptoms=["currently taking sertraline 100mg", "denies panic attacks"],
            diagnosis=["Major Depressive Disorder"],
            visitType=VisitTypeEnum.FOLLOW_UP,
        )
    )

    res = svc.deprescribe_step(req)

    # Verify DeprescribeResponse fields
    assert "Sertraline 50mg" in res.historical_prescription
    assert "Alprazolam 0.25mg" in res.historical_prescription
    assert res.latest_documented_prescription == ["Sertraline 100mg"]
    assert any("sertraline" in m.lower() for m in res.patient_reported_current_use)
    assert any("alprazolam" in m.lower() for m in res.actual_discontinuation)

    # Verify State DTO fields
    assert res.state.historical_prescription == res.historical_prescription
    assert res.state.latest_documented_prescription == res.latest_documented_prescription
    assert res.state.patient_reported_current_use == res.patient_reported_current_use
    assert res.state.actual_discontinuation == res.actual_discontinuation

    # Verify no duplicate prompt in serialized state DTO
    state_dump = res.state.model_dump()
    assert "triage_prompt" not in state_dump
    assert state_dump["deprescribe"]["prompt"] == "UNIFIED PROMPT 6.3 KB CONTENT"

