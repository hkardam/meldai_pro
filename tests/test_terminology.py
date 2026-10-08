"""Unit tests for offline HPO and MONDO terminology services."""

import pytest
from meldai.config import get_settings
from meldai.terminology.hpo import get_hpo_service
from meldai.terminology.mondo import get_mondo_service


@pytest.fixture(scope="module")
def hpo_service():
    settings = get_settings()
    return get_hpo_service(settings.hpo_obo_path)


@pytest.fixture(scope="module")
def mondo_service():
    settings = get_settings()
    return get_mondo_service(settings.mondo_obo_path)


def test_hpo_exact_search(hpo_service):
    results = hpo_service.search("Seizure", limit=5)
    assert len(results) > 0
    top = results[0]
    assert top.hpo_id == "HP:0001250"
    assert top.label == "Seizure"
    assert top.match_type == "exact_label"
    assert top.score == 1.0


def test_hpo_token_and_fuzzy_search(hpo_service):
    # Tests matching "Less social interactions" via token/fuzzy matching
    results = hpo_service.search("Less social interactions", limit=5)
    assert len(results) > 0
    top = results[0]
    assert top.hpo_id == "HP:5200310"
    assert top.label == "Diminishment of social interactions"
    assert top.match_type in ("token_synonym", "token_label")
    assert top.score > 0.6


def test_hpo_get_by_id(hpo_service):
    res = hpo_service.get_by_id("HP:0001250")
    assert res is not None
    assert res.label == "Seizure"
    assert res.match_type == "id_lookup"


def test_mondo_exact_search(mondo_service):
    results = mondo_service.search("diabetes mellitus", limit=5)
    assert len(results) > 0
    top = results[0]
    assert top.label.lower() == "diabetes mellitus"
    assert top.match_type == "exact_label"
    assert top.score == 1.0


def test_mondo_token_and_fuzzy_search(mondo_service):
    results = mondo_service.search("depressive disorders", limit=5)
    assert len(results) > 0
    assert any("depressive" in r.label.lower() for r in results)
    assert results[0].score > 0.6


def test_hpo_word_boundary_prevents_subword_match(hpo_service):
    """Word boundary matching ensures 'sleep' does not match 'Microsleep' (HP:5200289)."""
    results = hpo_service.search("sleep", limit=10)
    for r in results:
        if r.hpo_id == "HP:5200289":
            assert r.match_type != "substring_label"


def test_hpo_lack_of_sleep_resolves_to_sleep_disturbance(hpo_service):
    """Clinical alias maps 'Lack of sleep' to 'Sleep disturbance' (HP:0002360), not Microsleep."""
    results = hpo_service.search("Lack of sleep", limit=5)
    assert len(results) > 0
    top = results[0]
    assert top.hpo_id == "HP:0002360"
    assert "sleep" in top.label.lower()


def test_to_curie_formatting():
    """Verify canonical CURIE formatting with 7 digits and standard prefixes."""
    from meldai.utils import to_curie

    assert to_curie("HP", 13600) == "HP:0013600"
    assert to_curie("HP", "HPO:2189") == "HP:0002189"
    assert to_curie("HP", "HP:5200289") == "HP:5200289"
    assert to_curie("MONDO", 8807) == "MONDO:0008807"
    assert to_curie("MONDO", "MONDO:0008807") == "MONDO:0008807"
    assert to_curie("HP", None) is None
    assert to_curie("HP", "unmatched") is None


def test_deficit_symptom_not_negated():
    """Verify that deficit phrases like 'Lack of sleep' are treated as affirmed phenotypes."""
    from unittest.mock import MagicMock
    from meldai.services.assertion_service import ClinicalAssertionService

    mock_nlp = MagicMock()
    mock_doc = MagicMock()
    mock_doc.ents = []
    mock_doc._.context_graph = MagicMock(modifiers=[])
    mock_nlp.analyze.return_value = mock_doc

    service = ClinicalAssertionService(nlp_service=mock_nlp)
    res = service.analyze("Lack of sleep")

    assert res.is_negated is False
    assert res.is_phenotype is True
    assert res.assertion_status == "affirmed"
    assert res.search_target == "Lack of sleep"

