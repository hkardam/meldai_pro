"""Unit tests for BioLORD embedding service and Medicine Master ingestion."""

import numpy as np
import pytest
import torch

from meldai.nlp.biolord import BioLORDEmbedder, _mean_pooling
from meldai.services.medicine_master_service import (
    MedicineMasterService,
    generate_digest_text,
)


def test_generate_digest_text_standard():
    item = {
        "brand_name": "ZOLAVIL",
        "canonical_molecule": "Etizolam",
        "clinical_dosing_indication": "Short-term bridge therapy for generalized anxiety and associated insomnia.",
    }
    result = generate_digest_text(item)
    assert result == "Etizolam, marketed as ZOLAVIL, is used as short-term bridge therapy for generalized anxiety and associated insomnia."


def test_generate_digest_text_preserves_acronym():
    item = {
        "brand_name": "RECTIN",
        "canonical_molecule": "Duloxetine",
        "clinical_dosing_indication": "SNRI therapy for major depression, generalized anxiety, and comorbid neuropathic pain.",
    }
    result = generate_digest_text(item)
    assert result == "Duloxetine, marketed as RECTIN, is used as SNRI therapy for major depression, generalized anxiety, and comorbid neuropathic pain."


def test_generate_digest_text_multi_molecule():
    item = {
        "brand_name": "VILFOL- D3",
        "canonical_molecule": "L-Methylfolate + Methylcobalamin + Pyridoxal-5-Phosphate + Vitamin D3",
        "clinical_dosing_indication": "Adjuvant neurotrophic and metabolic co-factor supplementation in mood and cognitive disorders.",
    }
    result = generate_digest_text(item)
    assert result == "L-Methylfolate + Methylcobalamin + Pyridoxal-5-Phosphate + Vitamin D3, marketed as VILFOL- D3, is used as adjuvant neurotrophic and metabolic co-factor supplementation in mood and cognitive disorders."


def test_generate_digest_text_missing_indication():
    item = {
        "brand_name": "ASPIRIN",
        "canonical_molecule": "Acetylsalicylic acid",
        "clinical_dosing_indication": "",
    }
    result = generate_digest_text(item)
    assert result == "Acetylsalicylic acid, marketed as ASPIRIN."


def test_mean_pooling_calculation():
    # Batch size 2, seq len 3, hidden dim 4
    token_embeddings = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [0.0, 0.0, 0.0, 0.0]],
            [[2.0, 2.0, 2.0, 2.0], [4.0, 4.0, 4.0, 4.0], [6.0, 6.0, 6.0, 6.0]],
        ],
        dtype=torch.float32,
    )
    attention_mask = torch.tensor(
        [[1, 1, 0], [1, 1, 1]],
        dtype=torch.int64,
    )

    model_output = (token_embeddings,)
    pooled = _mean_pooling(model_output, attention_mask)

    # First item: average of tokens 0 and 1 -> [(1+5)/2, (2+6)/2, (3+7)/2, (4+8)/2] = [3, 4, 5, 6]
    expected_first = torch.tensor([3.0, 4.0, 5.0, 6.0])
    # Second item: average of all 3 tokens -> [(2+4+6)/3, (2+4+6)/3, (2+4+6)/3, (2+4+6)/3] = [4, 4, 4, 4]
    expected_second = torch.tensor([4.0, 4.0, 4.0, 4.0])

    assert torch.allclose(pooled[0], expected_first)
    assert torch.allclose(pooled[1], expected_second)


def test_cosine_similarity():
    vec_a = np.array([1.0, 0.0, 0.0])
    vec_b = np.array([1.0, 0.0, 0.0])
    vec_c = np.array([0.0, 1.0, 0.0])

    sim_ab = BioLORDEmbedder.compute_similarity(vec_a, vec_b)
    sim_ac = BioLORDEmbedder.compute_similarity(vec_a, vec_c)

    assert pytest.approx(sim_ab, 1e-5) == 1.0
    assert pytest.approx(sim_ac, 1e-5) == 0.0


def test_medicine_master_service_batching(monkeypatch):
    class MockBioEmbedder:
        def embed_texts(self, texts, normalize=True):
            # Return dummy 768-d unit vector for each text
            return np.ones((len(texts), 768), dtype=np.float32) / np.sqrt(768)

    class MockMongoKB:
        def __init__(self):
            self.batches = []
            self.cleared = False

        def setup_medicine_master_indexes(self):
            pass

        def clear_medicine_master(self):
            self.cleared = True
            return 0

        def upsert_medicine_master_batch(self, docs):
            self.batches.append(docs)
            return {"upserted_count": len(docs), "modified_count": 0, "matched_count": 0}

        def get_medicine_master_count(self):
            return sum(len(b) for b in self.batches)

    svc = MedicineMasterService(
        mongo_kb=MockMongoKB(),
        biolord_embedder=MockBioEmbedder(),
    )

    sample_items = [
        {
            "brand_name": f"BRAND_{i}",
            "canonical_molecule": f"MOL_{i}",
            "clinical_dosing_indication": f"Indication {i}.",
        }
        for i in range(250)
    ]

    import tempfile, json
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
        json.dump(sample_items, tmp)
        tmp_path = tmp.name

    result = svc.load_and_embed_master(
        file_path=tmp_path,
        batch_size=100,
        recreate=True,
    )

    assert result["status"] == "success"
    assert result["total_records"] == 250
    assert result["batches_processed"] == 3
    assert len(svc.mongo_kb.batches) == 3
    assert len(svc.mongo_kb.batches[0]) == 100
    assert len(svc.mongo_kb.batches[1]) == 100
    assert len(svc.mongo_kb.batches[2]) == 50

    # Verify root fields
    first_doc = svc.mongo_kb.batches[0][0]
    assert "embedding" in first_doc
    assert len(first_doc["embedding"]) == 768
    assert "digest_text" in first_doc
    assert first_doc["digest_text"].startswith("MOL_0, marketed as BRAND_0")


def test_search_similar_medicines_vectorized():
    class MockBioEmbedder:
        def embed_texts(self, texts, normalize=True):
            return np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    class MockCollection:
        def find(self, query, projection):
            return [
                {
                    "brand_name": "MED_A",
                    "canonical_molecule": "MOL_A",
                    "embedding": [1.0, 0.0, 0.0],
                },
                {
                    "brand_name": "MED_B",
                    "canonical_molecule": "MOL_B",
                    "embedding": [0.0, 1.0, 0.0],
                },
            ]

    class MockMongoKB:
        @property
        def medicine_master(self):
            return MockCollection()

    svc = MedicineMasterService(
        mongo_kb=MockMongoKB(),
        biolord_embedder=MockBioEmbedder(),
    )

    matches = svc.search_similar_medicines("query phrase", top_k=5, min_score=0.5)
    assert len(matches) == 1
    assert matches[0]["brand_name"] == "MED_A"
    assert matches[0]["similarity_score"] == 1.0
    assert "embedding" not in matches[0]

