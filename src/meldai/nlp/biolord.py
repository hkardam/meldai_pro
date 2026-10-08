"""Clinical and biomedical concept embedding engine using BioLORD.

BioLORD (Biomedical Learning of Representations for Concepts and Descriptions)
produces dense semantic vector representations for clinical entities, drug
descriptions, and diagnostic indications. It clusters semantically identical
concepts tightly in embedding space even across disparate lexical formulations.
"""

import logging
import os
from typing import Any, List, Optional, Union
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from meldai.config import Settings, get_settings

logger = logging.getLogger(__name__)


def _mean_pooling(model_output: Any, attention_mask: torch.Tensor) -> torch.Tensor:
    """Perform mean pooling on token embeddings accounting for the attention mask."""
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class BioLORDEmbedder:
    """Manages model loading, multi-core CPU optimization, and batch inference for BioLORD."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.model_name = self.settings.biolord_model_name
        self.device = torch.device(
            self.settings.device if torch.cuda.is_available() and self.settings.device == "cuda" else "cpu"
        )
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModel] = None

        # Maximize CPU multi-core utilization if running on CPU
        if self.device.type == "cpu":
            num_cores = os.cpu_count() or 4
            torch.set_num_threads(num_cores)
            try:
                torch.set_num_interop_threads(min(4, max(1, num_cores // 2)))
            except RuntimeError:
                pass  # Inter-op parallelism can only be initialized once
            logger.info("BioLORD initialized with %d PyTorch CPU compute threads.", num_cores)

    def _ensure_loaded(self) -> None:
        """Lazy load tokenizer and BioLORD model weights on first inference request."""
        if self._model is None or self._tokenizer is None:
            logger.info("Loading BioLORD model '%s' on %s...", self.model_name, self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            logger.info("BioLORD model successfully loaded into memory.")

    def embed_texts(
        self,
        texts: Union[str, List[str]],
        max_length: Optional[int] = None,
        batch_size: Optional[int] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate dense clinical vector representations for input phrases or descriptions.

        Args:
            texts: Single string or list of clinical phrases / medication digests.
            max_length: Max sequence length for tokenization (defaults to settings.biolord_max_length).
            batch_size: Inference batch size (defaults to settings.biolord_batch_size).
            normalize: If True, vectors are L2-normalized so dot products equal cosine similarities.

        Returns:
            np.ndarray of shape (N, 768) containing 768-dimensional BioLORD embeddings.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.empty((0, 768), dtype=np.float32)

        self._ensure_loaded()

        b_size = batch_size or self.settings.biolord_batch_size
        m_len = max_length or self.settings.biolord_max_length
        all_embeddings: List[np.ndarray] = []

        with torch.inference_mode():
            for i in range(0, len(texts), b_size):
                batch_slice = texts[i : i + b_size]
                encoded = self._tokenizer(
                    batch_slice,
                    padding=True,
                    truncation=True,
                    max_length=m_len,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                outputs = self._model(**encoded)
                # BioLORD uses mean pooling over token embeddings
                pooled = _mean_pooling(outputs, encoded["attention_mask"])

                if normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

                all_embeddings.append(pooled.cpu().numpy())

        return np.vstack(all_embeddings)

    @staticmethod
    def compute_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        """
        Compute cosine similarity between two BioLORD embedding vectors.
        Returns float between -1.0 and 1.0.
        """
        dot = np.dot(vector_a, vector_b)
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))


_biolord_instance: Optional[BioLORDEmbedder] = None


def get_biolord_embedder(settings: Optional[Settings] = None) -> BioLORDEmbedder:
    """Get or create singleton BioLORDEmbedder instance."""
    global _biolord_instance
    if _biolord_instance is None:
        _biolord_instance = BioLORDEmbedder(settings)
    return _biolord_instance
