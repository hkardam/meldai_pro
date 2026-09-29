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
