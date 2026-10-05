"""Unit tests for the MedspaCyService — NER, ConText assertion, HPO linking, embedding."""

from unittest.mock import MagicMock, patch
from meldai.nlp.medspacy_nlp import MedspaCyService, SegmentResult


def _make_mock_ent(text, is_negated=False, is_historical=False, is_uncertain=False, is_family=False, kb_ents=None):
    """Helper to create a mock spaCy entity with medspaCy ConText attributes."""
    ent = MagicMock()
    ent.text = text
    ent._.is_negated = is_negated
    ent._.is_historical = is_historical
    ent._.is_uncertain = is_uncertain
    ent._.is_family = is_family
    ent._.kb_ents = kb_ents or []
    return ent


def _make_mock_doc(text, ents=None, vector=None, has_vector=True):
    """Helper to create a mock spaCy Doc."""
    doc = MagicMock()
    doc.text = text
    doc.ents = ents or []
    doc.vector = vector if vector is not None else [0.1] * 96
    doc.has_vector = has_vector
    return doc


def _build_service_with_mock_nlp(mock_nlp):
    """Build a MedspaCyService and inject a mock _nlp."""
    service = MedspaCyService.__new__(MedspaCyService)
    service._nlp = mock_nlp
    return service


def test_affirmed_entity_is_phenotype():
    ent = _make_mock_ent("anxiety", kb_ents=[("HP:0000745", 0.95)])
    doc = _make_mock_doc("anxiety while driving", ents=[ent])

    mock_nlp = MagicMock()
    mock_nlp.pipe.return_value = iter([doc])

    service = _build_service_with_mock_nlp(mock_nlp)
    results = service.process(["anxiety while driving"])

    assert len(results) == 1
    r = results[0]
    assert r.is_phenotype is True
    assert r.hpo_code == 745
    assert len(r.embedding) == 96


def test_negated_entity_is_not_phenotype():
    ent = _make_mock_ent("fever", is_negated=True, kb_ents=[("HP:0001945", 0.90)])
    doc = _make_mock_doc("no fever", ents=[ent])

    mock_nlp = MagicMock()
    mock_nlp.pipe.return_value = iter([doc])

    service = _build_service_with_mock_nlp(mock_nlp)
    results = service.process(["no fever"])

    assert results[0].is_phenotype is False
    assert results[0].hpo_code is None


def test_historical_entity_is_not_phenotype():
    ent = _make_mock_ent("seizures", is_historical=True, kb_ents=[("HP:0001250", 0.88)])
    doc = _make_mock_doc("h/o seizures", ents=[ent])

    mock_nlp = MagicMock()
    mock_nlp.pipe.return_value = iter([doc])

    service = _build_service_with_mock_nlp(mock_nlp)
    results = service.process(["h/o seizures"])

    assert results[0].is_phenotype is False


def test_no_entity_is_not_phenotype():
    doc = _make_mock_doc("follow up in 2 weeks", ents=[])
    mock_nlp = MagicMock()
    mock_nlp.pipe.return_value = iter([doc])

    service = _build_service_with_mock_nlp(mock_nlp)
    results = service.process(["follow up in 2 weeks"])

    assert results[0].is_phenotype is False
    assert results[0].hpo_code is None


def test_no_vector_returns_empty_embedding():
    ent = _make_mock_ent("cough", kb_ents=[("HP:0012735", 0.85)])
    doc = _make_mock_doc("cough", ents=[ent], has_vector=False)

    mock_nlp = MagicMock()
    mock_nlp.pipe.return_value = iter([doc])

    service = _build_service_with_mock_nlp(mock_nlp)
    results = service.process(["cough"])

    assert results[0].embedding == []


def test_batch_processes_multiple_segments():
    ents_a = [_make_mock_ent("headache", kb_ents=[("HP:0002315", 0.9)])]
    ents_b = []  # no entity — non-phenotype

    docs = [
        _make_mock_doc("severe headache", ents=ents_a),
        _make_mock_doc("60% better mood", ents=ents_b),
    ]

    mock_nlp = MagicMock()
    mock_nlp.pipe.return_value = iter(docs)

    service = _build_service_with_mock_nlp(mock_nlp)
    results = service.process(["severe headache", "60% better mood"])

    assert results[0].is_phenotype is True
    assert results[0].hpo_code == 2315
    assert results[1].is_phenotype is False


def test_analyze_affirmed():
    ent = _make_mock_ent("video games", is_negated=False)
    doc = _make_mock_doc("playing video games", ents=[ent])

    mock_nlp = MagicMock()
    mock_nlp.return_value = doc

    service = _build_service_with_mock_nlp(mock_nlp)
    analyzed_doc = service.analyze("playing video games")

    assert analyzed_doc == doc
    assert analyzed_doc.ents[0].text == "video games"
    assert analyzed_doc.ents[0]._.is_negated is False


def test_analyze_negated():
    ent = _make_mock_ent("video games", is_negated=True)
    doc = _make_mock_doc("not playing video games", ents=[ent])

    mock_nlp = MagicMock()
    mock_nlp.return_value = doc

    service = _build_service_with_mock_nlp(mock_nlp)
    analyzed_doc = service.analyze("not playing video games")

    assert analyzed_doc == doc
    assert analyzed_doc.ents[0].text == "video games"
    assert analyzed_doc.ents[0]._.is_negated is True

