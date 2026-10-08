"""Clinical Case Similarity Engine.

Implements term-, set-, demographic-, and encounter-level similarity metrics:
- Term similarity: Hybrid SapBERT cosine + ontology Wu-Palmer with sharpening cutoff (TAU).
- Set similarity: Bidirectional weighted best-match average (BETA, IDF-weighted).
- Patient similarity: Age-band discrete transitions + Log-age Gaussian kernel.
- Visit type similarity: 1.0 if exact match, 0.5 if different.
- Composite score: Normalized across available components.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from meldai.terminology.taxonomy import TaxonomyIndex

# ---------------------------------------------------------------------------
# Config Constants (Triplets tunable)
# ---------------------------------------------------------------------------
ALPHA: float = 0.5         # Weight balancing cosine vector similarity vs taxonomy Wu-Palmer
TAU: float = 0.3           # Weak-match cutoff threshold: raw <= TAU sharpens to 0.0
BETA: float = 0.5          # Query-coverage (1 - BETA) vs candidate-coverage (BETA) weight
GENERIC_DEPTH: int = 2     # Ancestor depth cutoff for deepest common ancestor (LCA)
SIGMA: float = 0.3         # Gaussian scale parameter for log-age difference


# ---------------------------------------------------------------------------
# Term Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: Optional[Union[List[float], np.ndarray]], vec_b: Optional[Union[List[float], np.ndarray]]) -> float:
    """Compute cosine similarity between two vector embeddings."""
    if vec_a is None or vec_b is None:
        return 0.0
    va = np.asarray(vec_a, dtype=np.float32)
    vb = np.asarray(vec_b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    dot = float(np.dot(va, vb))
    return float(dot / (norm_a * norm_b))


def term_similarity(
    term_a: str,
    term_b: str,
    vec_a: Optional[Union[List[float], np.ndarray]] = None,
    vec_b: Optional[Union[List[float], np.ndarray]] = None,
    code_a: Optional[Union[str, int]] = None,
    code_b: Optional[Union[str, int]] = None,
    taxonomy: Optional[TaxonomyIndex] = None,
    alpha: float = ALPHA,
    tau: float = TAU,
) -> float:
    """Compute sharpened term similarity s*(a, b) in [0, 1].

    raw = alpha * max(cosine, 0) + (1 - alpha) * wuPalmer(a, b)
          (if either has no ontology position: raw = max(cosine, 0))
    s*  = max(0, (raw - tau) / (1 - tau))
    Identical term or identical code -> s* = 1.0.
    """
    clean_a = term_a.strip().lower() if term_a else ""
    clean_b = term_b.strip().lower() if term_b else ""

    # Identical term check
    if clean_a and clean_b and clean_a == clean_b:
        return 1.0
    if code_a is not None and code_b is not None and str(code_a).strip() == str(code_b).strip():
        return 1.0

    # Vector cosine similarity
    cos_val = cosine_similarity(vec_a, vec_b) if (vec_a is not None and vec_b is not None) else 0.0
    cos_clipped = max(cos_val, 0.0)

    # Taxonomic Wu-Palmer similarity
    if taxonomy is not None and code_a is not None and code_b is not None:
        wp_val = taxonomy.wu_palmer(code_a, code_b)
        raw = alpha * cos_clipped + (1.0 - alpha) * wp_val
    else:
        # Local concept without ontology position
        raw = cos_clipped

    # Sharpening cutoff
    if raw <= tau:
        return 0.0

    s_star = (raw - tau) / (1.0 - tau)
    return min(max(s_star, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Set Similarity (Weighted Best-Match Average)
# ---------------------------------------------------------------------------

class EntityItem:
    """Standardized entity item for set similarity comparison."""
    def __init__(
        self,
        term: str,
        embedding: Optional[List[float]] = None,
        ontology_code: Optional[Union[str, int]] = None,
    ):
        self.term = term
        self.embedding = embedding
        self.ontology_code = ontology_code

    @property
    def key(self) -> str:
        """Deduplication key: ontology code if available, else lower-cased surface form."""
        if self.ontology_code is not None:
            return f"code:{self.ontology_code}"
        return f"term:{self.term.strip().lower()}"


def deduplicate_entities(entities: List[EntityItem]) -> List[EntityItem]:
    """Deduplicate a list of entities so duplicates do not distort set averages."""
    seen = set()
    deduped = []
    for e in entities:
        k = e.key
        if k not in seen:
            seen.add(k)
            deduped.append(e)
    return deduped


def get_idf_weight(term_key: str, df_counts: Optional[Dict[str, int]] = None, total_docs: int = 1000) -> float:
    """Compute IDF weight: w_t = ln((N + 1) / (df_t + 1)) + 1.0."""
    df = 1
    if df_counts and term_key in df_counts:
        df = df_counts[term_key]
    return math.log((float(total_docs) + 1.0) / (float(df) + 1.0)) + 1.0


def set_similarity(
    query_set: List[EntityItem],
    candidate_set: List[EntityItem],
    taxonomy: Optional[TaxonomyIndex] = None,
    df_counts: Optional[Dict[str, int]] = None,
    total_docs: int = 1000,
    beta: float = BETA,
    alpha: float = ALPHA,
    tau: float = TAU,
) -> Optional[float]:
    """Compute directional weighted best-match set similarity score(Q, C).

    cover(A -> B) = sum_a w_a * max_b s*(a,b) / sum_a w_a
    score(Q, C)   = (1 - BETA) * cover(Q -> C) + BETA * cover(C -> Q)

    Returns None if either set is empty.
    """
    clean_q = deduplicate_entities([e for e in query_set if e.term and e.term.strip()])
    clean_c = deduplicate_entities([e for e in candidate_set if e.term and e.term.strip()])

    if not clean_q or not clean_c:
        return None

    # Precalculate pairwise term similarities matrix: M[i][j] = s*(clean_q[i], clean_c[j])
    n_q = len(clean_q)
    n_c = len(clean_c)
    sim_matrix = np.zeros((n_q, n_c), dtype=np.float32)

    for i, q in enumerate(clean_q):
        for j, c in enumerate(clean_c):
            sim_matrix[i, j] = term_similarity(
                term_a=q.term,
                term_b=c.term,
                vec_a=q.embedding,
                vec_b=c.embedding,
                code_a=q.ontology_code,
                code_b=c.ontology_code,
                taxonomy=taxonomy,
                alpha=alpha,
                tau=tau,
            )

    # Weights for query terms
    weights_q = [get_idf_weight(q.key, df_counts, total_docs) for q in clean_q]
    sum_w_q = sum(weights_q)

    # Weights for candidate terms
    weights_c = [get_idf_weight(c.key, df_counts, total_docs) for c in clean_c]
    sum_w_c = sum(weights_c)

    # Coverage Q -> C
    cover_q_c = 0.0
    for i in range(n_q):
        best_match_for_q = float(np.max(sim_matrix[i, :])) if n_c > 0 else 0.0
        cover_q_c += weights_q[i] * best_match_for_q
    cover_q_c = cover_q_c / sum_w_q if sum_w_q > 0.0 else 0.0

    # Coverage C -> Q
    cover_c_q = 0.0
    for j in range(n_c):
        best_match_for_c = float(np.max(sim_matrix[:, j])) if n_q > 0 else 0.0
        cover_c_q += weights_c[j] * best_match_for_c
    cover_c_q = cover_c_q / sum_w_c if sum_w_c > 0.0 else 0.0

    score = (1.0 - beta) * cover_q_c + beta * cover_c_q
    return min(max(score, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Patient Age Similarity
# ---------------------------------------------------------------------------

def get_age_band(age: float) -> int:
    """Map age in completed years to one of 8 discrete age bands:
    0: 0–5     early childhood
    1: 6–9     childhood
    2: 10–12   pre-adolescent
    3: 13–17   adolescent
    4: 18–25   young adult
    5: 26–40   ad11ult
    6: 41–60   middle age
    7: 61+     older adult
    """
    if age <= 5:
        return 0
    elif age <= 9:
        return 1
    elif age <= 12:
        return 2
    elif age <= 17:
        return 3
    elif age <= 25:
        return 4
    elif age <= 40:
        return 5
    elif age <= 60:
        return 6
    else:
        return 7


def patient_similarity(age_q: Optional[float], age_c: Optional[float], sigma: float = SIGMA) -> Optional[float]:
    """Compute patient demographic similarity: 0.7 * bandScore + 0.3 * gaussScore.

    Missing age on either side -> None.
    """
    if age_q is None or age_c is None:
        return None
    try:
        a_q = float(age_q)
        a_c = float(age_c)
    except (ValueError, TypeError):
        return None

    if a_q < 0 or a_c < 0:
        return None

    # Discrete band score
    band_q = get_age_band(a_q)
    band_c = get_age_band(a_c)
    diff = abs(band_q - band_c)
    if diff == 0:
        band_score = 1.0
    elif diff == 1:
        band_score = 0.7
    elif diff == 2:
        band_score = 0.1
    else:
        band_score = 0.0

    # Gaussian log-age similarity: S = exp(-(ln(1+a1) - ln(1+a2))^2 / (2 * sigma^2))
    x_q = math.log(1.0 + a_q)
    x_c = math.log(1.0 + a_c)
    gauss_score = math.exp(-((x_q - x_c) ** 2) / (2.0 * (sigma ** 2)))

    score = 0.7 * band_score + 0.3 * gauss_score
    return min(max(score, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Visit Type Similarity
# ---------------------------------------------------------------------------

def normalize_visit_type(val: Optional[str]) -> Optional[str]:
    """Normalize raw visitReason or visitType string to 'follow-up' or 'new consultation'."""
    if not val:
        return None
    s = str(val).strip().lower()
    if "follow" in s or "fu" in s:
        return "follow-up"
    if "new" in s or "consult" in s or "first" in s:
        return "new consultation"
    return s


def visit_type_similarity(visit_type_q: Optional[str], visit_type_c: Optional[str]) -> Optional[float]:
    """Compute visit type similarity: 1.0 if exact match, else 0.5."""
    norm_q = normalize_visit_type(visit_type_q)
    norm_c = normalize_visit_type(visit_type_c)

    if not norm_q or not norm_c:
        return 0.5

    return 1.0 if norm_q == norm_c else 0.5


# ---------------------------------------------------------------------------
# Composite Final Score Normalization
# ---------------------------------------------------------------------------

def compute_composite_score(
    component_scores: Dict[str, Optional[float]],
    weights: Dict[str, float],
) -> float:
    """Compute dynamically normalized composite score over available components.

    final = sum(w_i * score_i) / sum(w_i) over available components.
    Guarantees output in [0, 1].
    """
    numerator = 0.0
    denominator = 0.0

    for comp_name, score in component_scores.items():
        if score is not None:
            w = weights.get(comp_name, 0.0)
            if w > 0.0:
                numerator += w * score
                denominator += w

    if denominator <= 0.0:
        return 0.0

    return min(max(numerator / denominator, 0.0), 1.0)
