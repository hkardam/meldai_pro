"""Unit tests for PrescribePipelineV2Runner and V2 accuracy matrix compatibility."""

import json
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from meldai.api.res_dtos import (
    EntityMoleculeRow,
    LLMStepCall,
    PrescribeMedicationsV2Response,
)
from meldai.v2.services.prescribe_pipeline_runner import (
    PrescribePipelineV2Runner,
    _StreamingJsonArrayWriter,
)
from scripts.analyzer.accuracy_matrix import evaluate_single_case


def test_prescribe_pipeline_v2_runner_initial_state():
    runner = PrescribePipelineV2Runner()
    status = runner.get_status()
    assert status["status"] == "idle"
    assert status["version"] == "v2"
    assert status["processed_cases"] == 0
    assert status["passed_cases"] == 0
    assert status["failed_cases"] == 0
    assert status["current_case_no"] is None


def test_streaming_json_array_writer(tmp_path: Path):
    target_file = tmp_path / "test_stream.json"
    with _StreamingJsonArrayWriter(target_file) as writer:
        writer.append({"caseNo": 1, "status": "ok"})
        writer.append({"caseNo": 2, "status": "ok"})

    assert target_file.is_file()
    data = json.loads(target_file.read_text(encoding="utf-8"))
    assert len(data) == 2
    assert data[0]["caseNo"] == 1
    assert data[1]["caseNo"] == 2


def test_v2_pipeline_output_accuracy_matrix_compatibility():
    """Verify that records emitted by PrescribePipelineV2Runner match what accuracy_matrix.py expects."""
    mock_record = {
        "runId": "test_run_123",
        "caseNo": 332,
        "req": {
            "case": {
                "caseNo": 332,
                "visitType": "new consultation",
                "patientInfo": {"age": 37.0, "gender": "Male"},
                "symptoms": ["Sadness of mood", "Severe insomnia"],
                "diagnosis": ["MDD WITH ANXIOUS DISTRESS", "INSOMNIA"],
            },
            "top_k": 5,
        },
        "expectedMeds": [
            {
                "brand_name": "LERITHIUM",
                "canonical_molecule": "Lithium Carbonate",
                "is_fdc": False,
                "constituent_strengths_mg": {"Lithium Carbonate": 400.0},
            },
            {
                "brand_name": "REVEN XR",
                "canonical_molecule": "Venlafaxine",
                "is_fdc": False,
                "constituent_strengths_mg": {"Venlafaxine": 75.0},
            },
        ],
        "prompts": {
            "step1_llm_med_check_prompt": "Step 1 Prompt content...",
            "step2_llm_med_prescribe_prompt": "Step 2 Prompt content...",
            "workflow_prompts_file": "/tmp/test_run_123_all_prompts.txt",
        },
        "res": {
            "medications": [
                {
                    "brand_name": "LERITHIUM",
                    "canonical_molecule": "Lithium Carbonate",
                    "is_fdc": False,
                    "constituent_strengths_mg": {"Lithium Carbonate": 400.0},
                    "modal_strength_mg": 400.0,
                    "release_form": "SR",
                },
                {
                    "brand_name": "REVEN XR",
                    "canonical_molecule": "Venlafaxine",
                    "is_fdc": False,
                    "constituent_strengths_mg": {"Venlafaxine": 75.0},
                    "modal_strength_mg": 75.0,
                    "release_form": "ER",
                },
            ],
            "explanation": "Patient presents with treatment-resistant MDD. Lithium is continued as a mood stabilizer anchor alongside Venlafaxine.",
            "clinical_summary": "37yo Male presenting with persistent depressive symptoms.",
            "metrics": {"response_time_seconds": 12.5, "cost_usd": 0.003},
        },
    }

    eval_result = evaluate_single_case(mock_record, match_by="molecule")
    assert eval_result.case_no == 332
    assert eval_result.expected_count == 2
    assert eval_result.prescribed_count == 2
    assert eval_result.matched_count == 2
    assert eval_result.accuracy == 100.0


def test_v2_runner_start_conflict():
    runner = PrescribePipelineV2Runner()
    runner._state["status"] = "running"
    res = runner.start()
    assert res["status"] == "conflict"


def test_api_v1_and_v2_pipeline_status_endpoints():
    from fastapi.testclient import TestClient
    from meldai.main import create_web_app

    app = create_web_app()
    client = TestClient(app)

    # V1 status
    res_v1 = client.get("/api/v1/cases/prescribe-pipeline/status")
    assert res_v1.status_code == 200
    assert "status" in res_v1.json()

    # V2 status via /api/v2
    res_v2 = client.get("/api/v2/cases/prescribe-pipeline/status")
    assert res_v2.status_code == 200
    assert "status" in res_v2.json()

    # V2 status via /api/v1/cases/prescribe-pipeline/v2/status
    res_v1_v2 = client.get("/api/v1/cases/prescribe-pipeline/v2/status")
    assert res_v1_v2.status_code == 200
    assert "status" in res_v1_v2.json()

