from meldai.utils import segment


def test_segment_bullets_and_lines():
    note = "● Fever\n● Cough\n● Headache"
    result = segment(note)
    assert result == ["Fever", "Cough", "Headache"]


def test_segment_commas_and_semicolons():
    note = "Fever, cough; shortness of breath"
    result = segment(note)
    assert result == ["Fever", "cough", "shortness of breath"]


def test_segment_parentheses_ignored():
    note = "Fever (mild, non-persistent), cough (dry; nocturnal), fatigue"
    result = segment(note)
    assert result == [
        "Fever (mild, non-persistent)",
        "cough (dry; nocturnal)",
        "fatigue",
    ]


def test_segment_sentence_ends():
    note = "Patient has severe headache. Also reports nausea and dizziness."
    result = segment(note)
    assert result == [
        "Patient has severe headache",
        "Also reports nausea",
        "dizziness",
    ]


def test_segment_empty_or_whitespace():
    assert segment("") == []
    assert segment(" .  \n ●   ") == []
