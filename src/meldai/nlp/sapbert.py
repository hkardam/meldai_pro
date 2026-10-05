"""Clinical entity embedding engine using SapBERT (Cambridge LTL).

SapBERT (Self-alignment Pretraining for Biomedical Entity Representation) maps
clinical terminology and biomedical surface forms to a dense vector space where
synonyms and closely related clinical concepts cluster tightly.
"""

import logging
from typing import List, Optional, Union
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from meldai.config import Settings, get_settings

logger = logging.getLogger(__name__)


class SapBERTEmbedder:
    """Manages loading and batch inference for the SapBERT clinical embedding model."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.model_name = self.settings.sapbert_model_name
        self.device = torch.device(self.settings.device if torch.cuda.is_available() and self.settings.device == "cuda" else "cpu")
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        """Lazy load tokenizer and model weights on first inference request."""
        if self._model is None or self._tokenizer is None:
            logger.info("Loading SapBERT model '%s' on %s...", self.model_name, self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            logger.info("SapBERT model successfully loaded into memory.")

    def embed_entities(
        self,
        entities: Union[str, List[str]],
        max_length: int = 25,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate dense clinical vector representations for input entity phrases.

        Args:
            entities: Single string or list of clinical phrases / entity terms.
            max_length: Max sequence length (biomedical entities are typically short, <= 25 tokens).
            normalize: If True, vectors are L2-normalized so dot products equal cosine similarities.

        Returns:
            np.ndarray of shape (N, 768) containing 768-dimensional clinical embeddings.
        """
        if isinstance(entities, str):
            entities = [entities]

        if not entities:
            return np.empty((0, 768), dtype=np.float32)

        self._ensure_loaded()

        batch_size = self.settings.embedding_batch_size
        all_embeddings: List[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, len(entities), batch_size):
                batch_text = entities[i : i + batch_size]
                toks = self._tokenizer(
                    batch_text,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                toks = {k: v.to(self.device) for k, v in toks.items()}

                outputs = self._model(**toks)
                # SapBERT uses [CLS] representation (first token)
                cls_rep = outputs.last_hidden_state[:, 0, :]

                if normalize:
                    cls_rep = torch.nn.functional.normalize(cls_rep, p=2, dim=1)

                all_embeddings.append(cls_rep.cpu().numpy())

        return np.vstack(all_embeddings)

    @staticmethod
    def compute_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """
        Compute cosine similarity between two clinical embedding vectors.
        Returns float between -1.0 and 1.0 (typically 0.0 - 1.0 for clinical terms).
        """
        dot = np.dot(vector_a, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))


_embedder_instance: Optional[SapBERTEmbedder] = None


def get_sapbert_embedder(settings: Optional[Settings] = None) -> SapBERTEmbedder:
    """Get or create singleton SapBERTEmbedder instance."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = SapBERTEmbedder(settings)
    return _embedder_instance
