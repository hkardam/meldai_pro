"""Unit tests for PrescriptionServiceV2 Indian Psychiatrist two-step prescribing pipeline."""

import json
from unittest.mock import MagicMock, patch

import pytest
from meldai_v2.req_dtos import (
    CaseInput,
    PatientInfoInput,
    PrescribeMedicationsV2Request,
    VisitTypeEnum,
)
from meldai_v2.res_dtos import EntityMoleculeRow
from meldai_v2.services.prescription_service import (
    PrescriptionServiceV2,
    build_llm_med_check_prompt,
    build_llm_med_prescribe_prompt,
    format_entity_matrix_for_prompt,
    save_workflow_prompts_to_file,
    strip_special_characters,
)
from pathlib import Path


def test_strip_special_characters():
    assert strip_special_characters("headache (severe) - daily?") == "headache daily"
    assert strip_special_characters("anxiety/panic attacks; cannot sleep!") == "anxiety panic attacks cannot sleep"
    assert strip_special_characters("nausea & vomiting (+++)") == "nausea vomiting"
    assert strip_special_characters("!obsessive thoughts...") == "obsessive thoughts"
    assert strip_special_characters("?GTCS [onset 2 days]") == "GTCS"
    assert strip_special_characters("") == ""


def test_format_entity_matrix_for_prompt():
    row1 = EntityMoleculeRow(
        entity="insomnia",
        entity_type="symptom",
        curie="HP:0002360",
        delta_status="maintained",
        top_molecules=[
            {
                "canonical_molecule": "Clonazepam",
                "is_fdc": False,
                "frequency": 12,
                "percentage": 60.0,
                "score": 9.8,
                "doctor_brands": ["Clonafit", "Zapiz"],
            }
        ],
    )
    row2 = EntityMoleculeRow(
        entity="tension headache",
        entity_type="symptom",
        curie="HP:0002315",
        delta_status="new",
        top_molecules=[],
    )

    formatted = format_entity_matrix_for_prompt([row1, row2])
    assert "HISTORICAL MEDICATION RELATION MATRIX" in formatted
    assert 'Entity: "insomnia"' in formatted
    assert "Clonazepam [Single]" in formatted
    assert "Doctor Brands: Clonafit, Zapiz" in formatted
    assert 'Entity: "tension headache"' in formatted
    assert "NO HISTORICAL MEDICATION MATCH FOUND IN DATABASE - CANDIDATE GAP" in formatted


def test_save_workflow_prompts_to_file(tmp_path):
    prompts = [
        {"title": "Triage Prompt", "prompt": "Identify symptom changes."},
        {"title": "Step 1 Prompt", "prompt": "Prescribe initial regimen."},
    ]
    file_path = save_workflow_prompts_to_file(
        run_id="test_run_audit",
        case_no="CASE-999",
        visit_type="follow-up",
        prompts=prompts,
    )
    p = Path(file_path)
    assert p.exists()
    content = p.read_text(encoding="utf-8")
    assert "MELDAI CLINICAL INTELLIGENCE ENGINE — WORKFLOW PROMPTS AUDIT LOG" in content
    assert "Case Number : CASE-999" in content
    assert "PROMPT #1: TRIAGE PROMPT" in content
    assert "Identify symptom changes." in content
    assert "PROMPT #2: STEP 1 PROMPT" in content
    assert "Prescribe initial regimen." in content


def test_build_llm_med_check_prompt():
    master_sample = [
        {
            "brand_name": "RTC-LS",
            "canonical_molecule": "Escitalopram + Clonazepam",
            "is_fdc": True,
            "constituent_strengths_mg": {"Escitalopram": 10.0, "Clonazepam": 0.5},
            "modal_strength_mg": 10.0,
            "release_form": "FDC",
            "dosage_tier": "STANDARD_MAINTENANCE",
            "dosage_group_id": "GRP_FDC_ESCITALOPRAM_10_CLONAZEPAM_0.5",
            "clinical_dosing_indication": "Fixed dose combination",
        }
    ]
    prompt = build_llm_med_check_prompt(
        clinical_summary="Patient 35M with generalized anxiety.",
        entity_matrix_text="### HISTORICAL MEDICATION MATRIX\n- Entity: anxiety",
        medicine_master=master_sample,
    )
    assert "directly or indirectly target" in prompt
    assert "<clinical-summary>" in prompt
    assert "Patient 35M with generalized anxiety." in prompt
    assert "<entity-matrix>" in prompt
    assert "<medication-master>" in prompt
    assert "RTC-LS" in prompt
    assert "Escitalopram + Clonazepam" in prompt
    assert "entity_medication_check" in prompt


def test_build_llm_med_prescribe_prompt():
    prompt = build_llm_med_prescribe_prompt(
        clinical_summary="case type: follow up visit\nPatient 35M with anxiety and headache.",
        updated_entity_matrix_text="- Entity: headache (Type: Symptom)",
    )
    assert "assistant system for a psychiatrist practicing in India" in prompt
    assert "reproducing their own prescribing pattern" in prompt
    assert "<data>" in prompt
    assert "case type: follow up visit" in prompt
    assert "<updated-entity-matrix>" in prompt
    assert "- Entity: headache (Type: Symptom)" in prompt
    assert "### VISIT ENCOUNTER TYPE:" not in prompt
    assert "<medication-master>" not in prompt


def test_prescribe_medications_v2_two_step_execution():
    """Verify that Step 1 llmMedCheck runs, updates entity matrix, Step 2 llmMedPrescribe runs,
    prompts are persisted to file, and local medicine master FDC schema is validated.
    """
    mock_settings = MagicMock()
    mock_settings.medicine_master_json_path = "data/medication/local_medicine_master.json"

    mock_gemini = MagicMock()
    # Step 1 response: llmMedCheck checking targeting and adding top molecules for gap
    step1_resp = {
        "entity_medication_check": [
            {
                "entity": "anxiety",
                "entity_type": "symptom",
                "directly_targeted": True,
                "indirectly_targeted": False,
                "attributed_molecules": ["Escitalopram"],
                "recommended_top_molecules": [],
            },
            {
                "entity": "tension headache",
                "entity_type": "symptom",
                "directly_targeted": True,
                "indirectly_targeted": False,
                "attributed_molecules": ["Escitalopram + Clonazepam"],
                "recommended_top_molecules": [
                    {
                        "canonical_molecule": "Escitalopram + Clonazepam",
                        "is_fdc": True,
                        "doctor_brands": ["RTC-LS"],
                        "frequency": 8,
                        "score": 6.5,
                        "percentage": 40.0,
                        "targeting_type": "direct",
                        "rationale": "FDC targeting tension and autonomic distress",
                    }
                ],
            },
        ],
        "summary": "Check complete",
        "response_time_seconds": 0.5,
        "tokens_used": {"prompt_tokens": 100, "candidates_tokens": 50, "total_tokens": 150},
        "cost_usd": 0.001,
    }

    # Step 2 response: llmMedPrescribe produces final medications
    step2_resp = {
        "medications": [
            {
                "brand_name": "ESPRIDE",
                "canonical_molecule": "Escitalopram",
                "is_fdc": False,
                "constituent_strengths_mg": {"Escitalopram": 10.0},
                "modal_strength_mg": 10.0,
                "release_form": "Regular",
                "dosage_tier": "STANDARD_MAINTENANCE",
                "dosage_group_id": "GRP_ESCITALOPRAM_STANDARD",
                "clinical_dosing_indication": "SSRI therapy for anxiety",
                "target_entities": ["anxiety"],
                "clinical_rationale": "SSRI therapy",
            },
            {
                "brand_name": "RTC-LS",
                "canonical_molecule": "Escitalopram + Clonazepam",
                "is_fdc": True,
                "constituent_strengths_mg": {"Escitalopram": 10.0, "Clonazepam": 0.5},
                "modal_strength_mg": 10.0,
                "release_form": "FDC",
                "dosage_tier": "STANDARD_MAINTENANCE",
                "dosage_group_id": "GRP_FDC_ESCITALOPRAM_10_CLONAZEPAM_0.5",
                "clinical_dosing_indication": "Fixed dose combination for severe anxiety with somatic distress",
                "target_entities": ["tension headache"],
                "clinical_rationale": "FDC combining SSRI and benzodiazepine for somatic symptom relief",
            },
        ],
        "deprescribing_recommendations": [],
        "explanation": "Final comprehensive regimen addressing anxiety and tension headache.",
        "final_confidence_score": 0.95,
        "response_time_seconds": 0.4,
        "tokens_used": {"prompt_tokens": 120, "candidates_tokens": 60, "total_tokens": 180},
        "cost_usd": 0.0012,
    }

    mock_gemini.generate_prescription.side_effect = [step1_resp, step2_resp]

    mock_mongo = MagicMock()
    mock_mongo.find_past_cases.return_value = []
    mock_mongo.find_similar_candidates.return_value = []

    mock_case_svc = MagicMock()
    decorated_case = MagicMock()
    decorated_case.symptoms = []
    decorated_case.diagnosis = []
    mock_case_svc.decorate_case.return_value = decorated_case

    mock_chief_svc = MagicMock()
    assessment = MagicMock()
    assessment.active_symptoms = ["anxiety", "tension headache"]
    assessment.symptom_notes = ""
    assessment.patient_condition = ""
    assessment.triaged_items = []
    assessment.status_changes = {}
    assessment.recovered_symptoms = []
    assessment.new_symptoms = []
    assessment.persistent_symptoms = []
    assessment.worsening_symptoms = []
    assessment.improving_symptoms = []
    assessment.llm_prompt = "TRIAGE & SYMPTOM CHANGE DETECTION PROMPT TEXT"
    assessment.triage_delta = None
    assessment.deprescribing_recommendations = []
    assessment.whats_changed = "No prior medications; initial evaluation."
    mock_chief_svc.assess_and_deprescribe.return_value = assessment

    mock_entity_ranking_svc = MagicMock()
    mock_entity_ranking_svc.build_entity_matrix.return_value = [
        EntityMoleculeRow(
            entity="anxiety",
            entity_type="symptom",
            top_molecules=[
                {
                    "canonical_molecule": "Escitalopram",
                    "frequency": 10,
                    "percentage": 50.0,
                    "score": 8.0,
                    "doctor_brands": ["ESPRIDE"],
                }
            ],
        ),
        EntityMoleculeRow(
            entity="tension headache",
            entity_type="symptom",
            top_molecules=[],
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

    req = PrescribeMedicationsV2Request(
        case=CaseInput(
            caseNo="101",
            patientInfo=PatientInfoInput(age=35, gender="Male"),
            symptoms=["anxiety", "tension headache (severe)?"],
            diagnosis=["generalized anxiety disorder"],
            visitType=VisitTypeEnum.FOLLOW_UP,
        )
    )

    res = svc.prescribe_medications_v2(req)

    # Verify both LLM calls occurred (Step 1 llmMedCheck and Step 2 llmMedPrescribe)
    assert mock_gemini.generate_prescription.call_count == 2
    assert res.metrics["two_step_execution"] is True
    assert len(res.medications) == 2

    # Check new fields
    assert res.llmMedCheck is not None
    assert res.llmMedCheck.prompt is not None
    assert "directly or indirectly target" in res.llmMedCheck.prompt
    assert res.llmMedCheck.output is not None

    assert res.llmMedPrescribe is not None
    assert res.llmMedPrescribe.prompt is not None
    assert "<updated-entity-matrix>" in res.llmMedPrescribe.prompt
    assert res.llmMedPrescribe.output is not None

    # Check that tension headache in entity_matrix was updated from Step 1 result
    th_row = next(r for r in res.entity_matrix if r.entity == "tension headache")
    assert len(th_row.top_molecules) > 0
    assert th_row.top_molecules[0]["canonical_molecule"] == "Escitalopram + Clonazepam"

    # Verify workflow prompts file was saved and contains all three prompts
    assert res.workflow_prompts_file is not None
    prompt_file = Path(res.workflow_prompts_file)
    assert prompt_file.exists()
    file_content = prompt_file.read_text(encoding="utf-8")
    assert "MELDAI CLINICAL INTELLIGENCE ENGINE — WORKFLOW PROMPTS AUDIT LOG" in file_content
    assert "PROMPT #1: TRIAGE & SYMPTOM CHANGE DETECTION PROMPT" in file_content
    assert "TRIAGE & SYMPTOM CHANGE DETECTION PROMPT TEXT" in file_content
    assert "PROMPT #2: STEP 1: LLMMEDCHECK PROMPT" in file_content
    assert "PROMPT #3: STEP 2: LLMMEDPRESCRIBE PROMPT" in file_content

    # Verify medications schema and FDC prediction
    med_non_fdc = res.medications[0]
    assert med_non_fdc["brand_name"] == "ESPRIDE"
    assert med_non_fdc["canonical_molecule"] == "Escitalopram"
    assert med_non_fdc["is_fdc"] is False
    assert isinstance(med_non_fdc["is_fdc"], bool)
    assert med_non_fdc["release_form"] == "Regular"
    assert med_non_fdc["dosage_tier"] == "STANDARD_MAINTENANCE"

    med_fdc = res.medications[1]
    assert med_fdc["brand_name"] == "RTC-LS"
    assert med_fdc["canonical_molecule"] == "Escitalopram + Clonazepam"
    assert med_fdc["is_fdc"] is True
    assert isinstance(med_fdc["is_fdc"], bool)
    assert med_fdc["release_form"] == "FDC"
    assert med_fdc["dosage_tier"] == "STANDARD_MAINTENANCE"

    # Verify per-run LLM log file was created
    assert res.llm_log_file is not None
    assert Path(res.llm_log_file).parent.name in ("logs", "tmp")


def test_log_llm_call_console_and_file(capsys):
    """Verify log_llm_call writes both request and response to console and per-run log files."""
    import uuid
    from core.services.gemini_service import get_run_log_file_paths, log_llm_call

    run_id = f"test_run_llm_audit_{uuid.uuid4().hex[:8]}"
    logs_file, tmp_file = get_run_log_file_paths(run_id)
    logs_file.unlink(missing_ok=True)
    tmp_file.unlink(missing_ok=True)

    logs_path, tmp_path = log_llm_call(
        run_id=run_id,
        call_name="test_clinical_call",
        model_name="gemini-2.5-flash",
        temperature=0.2,
        prompt="Sample clinical prompt for patient evaluation",
        response_text='{"status": "ok", "recommendation": "ESPRIDE"}',
        elapsed_seconds=1.234,
        metrics={"tokens_used": {"prompt_tokens": 15, "candidates_tokens": 10, "total_tokens": 25}, "cost_usd": 0.00005},
    )

    # Check console output (captured via stdout)
    captured = capsys.readouterr()
    assert "🤖 [LLM REQUEST]" in captured.out
    assert run_id in captured.out
    assert "test_clinical_call" in captured.out
    assert "Sample clinical prompt for patient evaluation" in captured.out
    assert "✅ [LLM RESPONSE]" in captured.out
    assert '{"status": "ok", "recommendation": "ESPRIDE"}' in captured.out
    assert "Latency     : 1.234s" in captured.out

    # Check file output in both logs/ and tmp/
    for p_str in (logs_path, tmp_path):
        p = Path(p_str)
        assert p.exists()
        content = p.read_text(encoding="utf-8")
        assert "MELDAI ENGINE — LLM CALLS AUDIT LOG" in content
        assert f"Run ID      : {run_id}" in content
        assert "Call Name   : test_clinical_call" in content
        assert "Sample clinical prompt for patient evaluation" in content
        assert '{"status": "ok", "recommendation": "ESPRIDE"}' in content
        assert "1.234s" in content


def test_gemini_service_offline_mock_logs_to_file():
    """Verify that GeminiService.generate_content_with_metadata records calls to the per-run log file."""
    import uuid
    from core.services.gemini_service import GeminiService, get_run_log_file_paths

    mock_settings = MagicMock()
    mock_settings.gemini_api_key = ""
    mock_settings.gemini_model_name = "gemini-2.5-flash"
    mock_settings.gemini_temperature = 0.0

    service = GeminiService(settings=mock_settings)
    run_id = f"test_run_offline_mock_{uuid.uuid4().hex[:8]}"

    raw_text, metrics = service.generate_content_with_metadata(
        prompt="You are a Senior Consultant Indian Psychiatrist. Prescribe medications.",
        run_id=run_id,
        call_name="step1_psychiatrist_prescription",
    )

    assert raw_text is not None
    assert len(raw_text) > 0

    logs_file, tmp_file = get_run_log_file_paths(run_id)
    assert logs_file.exists()
    content = logs_file.read_text(encoding="utf-8")
    assert f"Run ID      : {run_id}" in content
    assert "Call Name   : step1_psychiatrist_prescription" in content
    assert "Senior Consultant Indian Psychiatrist" in content


def test_gemini_service_multiple_calls_append_same_run():
    """Verify that multiple sequential LLM calls in the same run append to the single run log file."""
    import uuid
    from core.services.gemini_service import GeminiService, get_run_log_file_paths

    mock_settings = MagicMock()
    mock_settings.gemini_api_key = ""
    mock_settings.gemini_model_name = "gemini-2.5-flash"
    mock_settings.gemini_temperature = 0.0

    service = GeminiService(settings=mock_settings)
    run_id = f"test_multi_call_run_{uuid.uuid4().hex[:8]}"
    logs_file, tmp_file = get_run_log_file_paths(run_id)
    logs_file.unlink(missing_ok=True)
    tmp_file.unlink(missing_ok=True)

    # Call 1: Triage
    service.generate_content_with_metadata(
        prompt="Evaluate changes in patient symptoms and intake items.",
        run_id=f"{run_id}_triage",
        call_name="triage_symptom_changes",
    )

    # Call 2: Step 1
    service.generate_prescription(
        prompt="Senior Consultant Indian Psychiatrist provisional prescription.",
        run_id=f"{run_id}_step1_response",
        call_name="step1_psychiatrist_prescription",
    )

    # Call 3: Step 2
    service.generate_prescription(
        prompt="Step 2: Fill clinical gaps and finalize regimen.",
        run_id=f"{run_id}_step2_response",
        call_name="step2_fill_gap_prescription",
    )

    assert logs_file.exists()
    content = logs_file.read_text(encoding="utf-8")

    # All three calls should be present in the exact same log file!
    assert "Call Name   : triage_symptom_changes" in content
    assert "Call Name   : step1_psychiatrist_prescription" in content
    assert "Call Name   : step2_fill_gap_prescription" in content
    assert content.count("[LLM CALL]") == 3

