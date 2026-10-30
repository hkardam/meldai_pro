"""Unit tests for chief complaint triage, noise filtering, and canonical negation mapping."""

from meldai_v2.services.clinical_delta_service import (
    ChiefComplaintChangeService,
    ClinicalChangeAssessment,
    TriagedComplaintItem,
    compute_delta,
    extract_canonical_from_negation,
    generate_clinical_summary,
    heuristic_triage_complaint,
)


def test_heuristic_triage_non_symptoms():
    non_symptoms = [
        "Informant: patient",
        "informant: mother",
        "more history to be taken from wife",
        "history to be taken from attendant",
        "follow up in 2 weeks",
        "review in 1 month",
        "referred to psychiatry",
        "labs advised",
        "blood test to be done",
        "next visit with brother",
    ]
    for text in non_symptoms:
        cat, canon = heuristic_triage_complaint(text)
        assert cat == "non_symptom", f"Expected '{text}' to be categorized as non_symptom, got {cat}"
        assert canon is None


def test_heuristic_triage_global_status():
    global_statuses = [
        "Feels much better",
        "feeling a lot better",
        "doing well",
        "No active symptoms reported",
        "no complaints reported",
        "asymptomatic",
        "symptom free",
        "completely normal",
    ]
    for text in global_statuses:
        cat, canon = heuristic_triage_complaint(text)
        assert cat == "global_status", f"Expected '{text}' to be categorized as global_status, got {cat}"
        assert canon is None


def test_heuristic_triage_negated_symptoms():
    cases = [
        ("denies paranoia currently", "paranoia"),
        ("denied auditory hallucinations", "auditory hallucinations"),
        ("no panic attacks", "panic attacks"),
        ("free of tremors", "tremors"),
        ("insomnia resolved", "insomnia"),
        ("headache absent", "headache"),
    ]
    for text, expected_canon in cases:
        cat, canon = heuristic_triage_complaint(text)
        assert cat == "negated_symptom", f"Expected '{text}' to be negated_symptom, got {cat}"
        assert canon == expected_canon, f"Expected canon '{expected_canon}', got '{canon}'"


def test_heuristic_triage_genuine_symptoms():
    symptoms = [
        "headache",
        "moderate insomnia",
        "severe anxiety",
        "loss of appetite",
        "low mood",
    ]
    for text in symptoms:
        cat, canon = heuristic_triage_complaint(text)
        assert cat == "symptom", f"Expected '{text}' to be symptom, got {cat}"
        assert canon == text


def test_compute_delta_excludes_noise_and_maps_negation():
    past_cases = [
        {
            "caseDate": "2026-09-01",
            "visitType": "new consultation",
            "chiefComplaints": [{"term": "paranoia"}, {"term": "severe anxiety"}],
            "diagnosis": [{"term": "generalized anxiety disorder"}],
            "medications": ["Clonazepam 0.5mg"],
        }
    ]
    current_symptoms = [
        "Informant: patient",
        "Feels much better",
        "No active symptoms reported",
        "denies paranoia currently",
        "more history to be taken from wife",
    ]

    svc = ChiefComplaintChangeService()
    assessment = svc.assess_symptom_changes(
        current_symptoms=current_symptoms,
        past_cases=past_cases,
        patient_info={"age": 35, "gender": "Female"},
        visit_type="follow-up",
        current_date="2026-10-22",
    )

    assert assessment.active_symptoms == []
    assert "paranoia" in assessment.resolved_symptoms

    delta = compute_delta(
        current_symptoms=current_symptoms,
        current_diagnoses=["generalized anxiety disorder"],
        past_cases=past_cases,
        change_assessment=assessment,
    )

    assert delta["new_symptoms"] == []
    assert delta["decreased_symptoms"] == []
    assert delta["maintained_symptoms"] == []
    assert "paranoia" in delta["stop_symptoms"]
    # Severe anxiety was not mentioned in current intake; absence of mention != resolved,
    # so compute_delta should NOT programmatically auto-classify it as stopped.
    assert "severe anxiety" not in delta["stop_symptoms"]
    assert "denies paranoia currently" not in delta["stop_symptoms"]
    assert "Informant: patient" not in delta["stop_symptoms"]
    assert "more history to be taken from wife" not in delta["stop_symptoms"]


def test_generate_clinical_summary_formatting():
    past_cases = [
        {
            "caseDate": "2026-09-01",
            "visitType": "new consultation",
            "chiefComplaints": [{"term": "paranoia"}, {"term": "severe anxiety"}],
            "diagnosis": [{"term": "generalized anxiety disorder"}],
        }
    ]
    current_symptoms = [
        "Informant: patient",
        "Feels much better",
        "No active symptoms reported",
        "denies paranoia currently",
        "more history to be taken from wife",
    ]

    svc = ChiefComplaintChangeService()
    assessment = svc.assess_symptom_changes(
        current_symptoms=current_symptoms,
        past_cases=past_cases,
        patient_info={"age": 35, "gender": "Female"},
        visit_type="follow-up",
        current_date="2026-10-22",
    )

    summary = generate_clinical_summary(
        current_visit_type="follow-up",
        patient_info={"age": 35, "gender": "Female"},
        current_symptoms=current_symptoms,
        current_diagnoses=["generalized anxiety disorder"],
        past_cases=past_cases,
        current_date="2026-10-22",
        change_assessment=assessment,
    )

    # Informant in header
    assert "informant: patient" in summary
    # Chief complaints section shows only genuine/resolved status, no noise
    assert "more history to be taken from wife" not in summary
    assert "feels much better" not in summary.split("clinical notes:")[0]
    assert "- paranoia (resolved / denied)" in summary


def test_clinical_date_parsing_and_past_cases_order():
    from core.utils import parse_clinical_date
    from meldai_v2.services.clinical_delta_service import _format_relative_time
    from datetime import datetime

    # 1. Verify DD/MM/YYYY vs ISO parsing
    dt1 = parse_clinical_date("12/09/2025")
    assert dt1 == datetime(2025, 9, 12)
    dt2 = parse_clinical_date("2025-10-08")
    assert dt2 == datetime(2025, 10, 8)
    dt3 = parse_clinical_date("02/09/2025")
    assert dt3 == datetime(2025, 9, 2)

    # 2. Verify relative time formatting does not treat 12/09/2025 as future Dec 9
    rel_str = _format_relative_time("12/09/2025", "2025-10-08")
    assert rel_str == "3 weeks ago"

    # 3. Verify ordering: 02/09/2025 -> 12/09/2025 (chronological ascending)
    dates = ["12/09/2025", "02/09/2025"]
    sorted_dates = sorted(dates, key=lambda d: parse_clinical_date(d) or datetime.min)
    assert sorted_dates == ["02/09/2025", "12/09/2025"]

