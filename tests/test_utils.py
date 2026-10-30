from core.utils import sanitize, segment


def test_sanitize_unicode_and_leading_bullets():
    raw = "★ • 60% cough, ++fever, ?GTCS"
    sanitized = sanitize(raw)
    assert sanitized == "60% cough, ++fever, ?GTCS"


def test_sanitize_keeps_clinical_symbols():
    text = "+fever\n?GTCS\n-cough"
    assert sanitize(text) == "+fever\n?GTCS\n-cough"


def test_segment_splits_only_newlines_and_bullets():
    note = "● Fever, chills; sweating\n● Cough with sputum (mild, dry)\n● Severe headache. Dizziness"
    result = segment(note)
    assert result == [
        "Fever, chills; sweating",
        "Cough with sputum (mild, dry)",
        "Severe headache. Dizziness",
    ]


def test_segment_with_sanitization():
    note = "★ • High grade fever (+39C)\n■ persistent cough (dry)\n● ?GTCS"
    result = segment(note)
    assert result == [
        "High grade fever (+39C)",
        "persistent cough (dry)",
        "?GTCS",
    ]


def test_segment_empty_or_whitespace():
    assert segment("") == []
    assert segment(" .  \n ●   ") == []


def test_clean_diagnosis_term():
    from core.utils import clean_diagnosis_term

    # Target test case requested by user
    assert clean_diagnosis_term("? mdd with anxious distress") == "mdd with anxious distress"

    # Various leading and trailing special characters
    assert clean_diagnosis_term("?? MDD with anxious distress??") == "MDD with anxious distress"
    assert clean_diagnosis_term("- ? mdd with anxious distress . ") == "mdd with anxious distress"
    assert clean_diagnosis_term("(bipolar affective disorder)") == "bipolar affective disorder"
    assert clean_diagnosis_term("!schizophrenia...") == "schizophrenia"
    assert clean_diagnosis_term("  major depressive disorder  ") == "major depressive disorder"

    # Single letter or empty / non-string
    assert clean_diagnosis_term("a") == "a"
    assert clean_diagnosis_term("? a !") == "a"
    assert clean_diagnosis_term("???") == ""
    assert clean_diagnosis_term("") == ""
    assert clean_diagnosis_term(None) == ""
