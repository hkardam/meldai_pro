"""Test age-based candidate and medication filtering and sorting."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from meldai_v2.req_dtos import (
    CaseInput,
    PatientInfoInput,
    PrescribeMedicationsV2Request,
    RankMoleculesRequest,
    SimilarityWeightsInput,
)
from meldai_v2.services.entity_matrix_service import rank_meds, rank_meds_for_entity
from core.services.similarity_engine import patient_info_similarity, patient_similarity


def test_patient_info_similarity_zero_age_sim():
    """When age similarity is 0.0 (incompatible age bands across 3+ groups), demographic sim is 0.0."""
    # Age 13 (adolescent) vs Age 55 (middle age adult): age_sim is 0.0
    age_sim = patient_similarity(13.0, 55.0)
    assert age_sim == 0.0

    # Even if genders match, demographic similarity must be 0.0
    p_sim = patient_info_similarity(age_q=13.0, age_c=55.0, gender_q="Male", gender_c="Male")
    assert p_sim == 0.0


def test_rank_meds_filters_incompatible_age_candidates():
    """Candidates with incompatible age (age_sim <= 0.0) must be filtered out before aggregating meds."""
    # cand1: Adolescent (age 14), clinical sim match
    cand1 = {
        "_id": "case_teen_1",
        "chiefComplaints": [{"term": "depression", "hpoId": "HP:0000716"}],
        "patientInfo": {"age": 14.0, "gender": "Male"},
        "medications": [
            {"canonical_molecule": "Escitalopram", "name": "ESPRIDE"},
        ],
    }

    # cand2: Older adult (age 52), clinical sim match, but incompatible age (age_sim = 0.0 vs age 13)
    cand2 = {
        "_id": "case_adult_1",
        "chiefComplaints": [{"term": "depression", "hpoId": "HP:0000716"}],
        "patientInfo": {"age": 52.0, "gender": "Male"},
        "medications": [
            {"canonical_molecule": "Vortioxetine", "name": "VORTICA"},
            {"canonical_molecule": "Propranolol", "name": "REPRO-SR 20MG"},
        ],
    }

    mock_kb = MagicMock()
    mock_kb.find_similar_candidates.return_value = [cand1, cand2]

    req = RankMoleculesRequest(
        type="chiefComplaint",
        query="depression",
        patient_age=13.0,
        patient_gender="Male",
        weights=SimilarityWeightsInput(w_lca=0.5, w_vector=0.5, min_similarity=0.5),
    )

    with patch("meldai_v2.services.entity_matrix_service.get_hpo_service") as mock_hpo_svc, \
         patch("meldai_v2.services.entity_matrix_service.get_sapbert_embedder") as mock_embedder, \
         patch("meldai_v2.services.entity_matrix_service.cosine_similarity") as mock_cos, \
         patch("meldai_v2.services.entity_matrix_service.compute_composite_score") as mock_comp:

        mock_hit = MagicMock()
        mock_hit.hpo_id = "HP:0000716"
        mock_hit.label = "Depressive disorder"
        mock_hpo_svc.return_value.search.return_value = [mock_hit]

        import numpy as np
        mock_embedder.return_value.embed_entities.return_value = [np.zeros(768)]
        mock_cos.side_effect = lambda q, e: 0.8
        mock_comp.side_effect = lambda scores, weights: 0.85  # clinical match

        resp = rank_meds(req, mongo_kb=mock_kb)

    # cand2 was filtered out due to age incompatibility (age 52 vs 13)
    assert resp.similar_cases_count == 1
    molecules = {m.canonical_molecule: m for m in resp.molecules}

    # Escitalopram from the adolescent candidate must be present
    assert "Escitalopram" in molecules
    # Adult medications from the incompatible candidate must be filtered out
    assert "Vortioxetine" not in molecules
    assert "Propranolol" not in molecules


def test_rank_meds_for_entity_age_filtering_and_sorting():
    """rank_meds_for_entity filters out age-incompatible candidate cases and sorts remaining meds."""
    cand_young = {
        "_id": "case_y",
        "diagnosis": [{"term": "mdd with anxious distress"}],
        "patientInfo": {"age": 15.0, "gender": "Male"},
        "medications": [
            {"canonical_molecule": "Fluoxetine", "brand_name": "FLUNIL"},
        ],
    }
    cand_old = {
        "_id": "case_o",
        "diagnosis": [{"term": "mdd with anxious distress"}],
        "patientInfo": {"age": 55.0, "gender": "Male"},
        "medications": [
            {"canonical_molecule": "Vortioxetine", "brand_name": "VORTICA"},
            {"canonical_molecule": "Propranolol", "brand_name": "REPRO-SR 20MG"},
        ],
    }

    with patch("meldai_v2.services.entity_matrix_service.compute_composite_score") as mock_comp:
        mock_comp.return_value = 0.9  # clinical match

        row = rank_meds_for_entity(
            query_term="mdd with anxious distress",
            entity_type="diagnosis",
            query_code=None,
            query_curie=None,
            query_vector=None,
            taxonomy=None,
            candidates=[cand_young, cand_old],
            patient_age=13.0,
            patient_gender="Male",
            top_cases_limit=5,
            k=5,
        )

    # Only Fluoxetine from cand_young should be in top_molecules; Vortioxetine/Propranolol excluded
    mol_names = [m["canonical_molecule"] for m in row.top_molecules]
    assert "Fluoxetine" in mol_names
    assert "Vortioxetine" not in mol_names
    assert "Propranolol" not in mol_names


def test_prescribe_medications_v2_request_top_k_alias():
    """PrescribeMedicationsV2Request accepts top_k as alias for top_k_per_entity."""
    data = {
        "case": {
            "caseNo": 169,
            "visitType": "new consultation",
            "patientInfo": {"age": 13.0, "gender": "Male"},
            "symptoms": ["chest pain", "social withdrawal"],
            "diagnosis": ["mdd with anxious distress"],
        },
        "top_k": 7,
    }
    req = PrescribeMedicationsV2Request(**data)
    assert req.top_k == 7
    assert req.top_k_per_entity == 7


def test_prompt_contains_pediatric_guidelines():
    """Verify llm_med_prescribe_prompt.md includes strict pediatric guidelines (<18) and safety contraindications."""
    prompt_path = Path(__file__).resolve().parent.parent / "src" / "meldai_v2" / "services" / "prompts" / "llm_med_prescribe_prompt.md"
    assert prompt_path.is_file(), "Prompt file must exist"
    content = prompt_path.read_text(encoding="utf-8")

    # Guidelines for <18
    assert "<18" in content
    # Monotherapy requirement
    assert "Monotherapy" in content
    # Polypharmacy prohibition
    assert "Polypharmacy" in content
    # Syncope / LOC contraindication
    assert "Syncope" in content or "syncope" in content
    assert "Loss of Consciousness" in content or "LOC" in content
    assert "CONTRAINDICATED" in content
    # Safety overrides pattern
    assert "SUPERSEDES HISTORICAL PATTERNS" in content or "safety concern" in content
