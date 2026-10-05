"""HPO (Human Phenotype Ontology) offline terminology service.

Loads hp.obo from local disk via pronto and provides term search
by label, synonym, definition, token-level matching, and fuzzy similarity.
No network calls, no external services.
"""

from __future__ import annotations

import difflib
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set

import pronto
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "to", "for", "with",
    "on", "at", "by", "from", "as", "is", "are", "was", "were", "be",
}


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

class HPOResult(BaseModel):
    """A single HPO concept match."""

    hpo_id: str = Field(..., description="HPO term ID, e.g. 'HP:0001250'")
    label: str = Field(..., description="Canonical preferred label")
    synonyms: List[str] = Field(default_factory=list, description="Known alternate labels")
    definition: Optional[str] = Field(None, description="Term definition text, if available")
    match_type: str = Field(
        default="label",
        description=(
            "How the match was found: 'exact_label' | 'synonym' | 'substring_label' | "
            "'substring_synonym' | 'token_label' | 'token_synonym' | 'fuzzy_label' | "
            "'fuzzy_synonym' | 'substring_definition' | 'token_definition' | 'id_lookup'"
        ),
    )
    score: Optional[float] = Field(
        default=None,
        description="Relevance / similarity score (0.0 to 1.0)",
    )
    is_negated: bool = Field(
        default=False,
        description="Whether a clinical negation modifier was detected for this query",
    )
    is_phenotype: bool = Field(
        default=True,
        description="Whether the concept represents an affirmed clinical phenotype",
    )
    assertion_status: str = Field(
        default="affirmed",
        description="Assertion status: 'affirmed' | 'negated' | 'uncertain' | 'historical' | 'family'",
    )


# ---------------------------------------------------------------------------
# Internal index entry
# ---------------------------------------------------------------------------

@dataclass
class _HPOEntry:
    hpo_id: str
    label: str
    label_norm: str
    label_toks: List[str]
    synonyms_raw: List[str]
    synonyms_norm: List[str]
    synonyms_toks: List[List[str]]
    definition: Optional[str] = None
    definition_norm: str = ""
    definition_toks: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class HPOService:
    """Loads hp.obo offline and searches for phenotype terms with multi-tier matching."""

    def __init__(self, obo_path: str) -> None:
        self._obo_path = Path(obo_path)
        self._ontology: Optional[pronto.Ontology] = None
        self._entries: List[_HPOEntry] = []
        self._token_index: Dict[str, Set[int]] = defaultdict(set)
        self._by_id: Dict[str, _HPOEntry] = {}

    # ------------------------------------------------------------------
    # Lazy loader and in-memory indexer
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._ontology is not None:
            return
        if not self._obo_path.exists():
            raise FileNotFoundError(
                f"HPO OBO file not found at '{self._obo_path}'. "
                "Run: python scripts/download_ontologies.py"
            )
        logger.info("Loading HPO ontology from '%s' ...", self._obo_path)
        self._ontology = pronto.Ontology(str(self._obo_path))

        self._entries.clear()
        self._token_index.clear()
        self._by_id.clear()

        for t in self._ontology.terms():  # type: ignore[union-attr]
            if not t.id.startswith("HP:"):
                continue

            label_raw = t.name or ""
            label_norm = _normalize(label_raw)
            label_toks = _tokenize(label_norm)

            syns_raw = [s.description for s in t.synonyms]
            syns_norm = [_normalize(s) for s in syns_raw]
            syns_toks = [_tokenize(sn) for sn in syns_norm]

            definition: Optional[str] = None
            if t.definition:
                definition = str(t.definition).strip().strip('"')
            def_norm = _normalize(definition) if definition else ""
            def_toks = _tokenize(def_norm) if def_norm else []

            idx = len(self._entries)
            entry = _HPOEntry(
                hpo_id=t.id,
                label=label_raw,
                label_norm=label_norm,
                label_toks=label_toks,
                synonyms_raw=syns_raw,
                synonyms_norm=syns_norm,
                synonyms_toks=syns_toks,
                definition=definition,
                definition_norm=def_norm,
                definition_toks=def_toks,
            )
            self._entries.append(entry)
            self._by_id[t.id] = entry

            # Also index alternate IDs if available
            alt_ids = getattr(t, "alternate_ids", None)
            if alt_ids:
                for alt in alt_ids:
                    if alt not in self._by_id:
                        self._by_id[alt] = entry

            # Build inverted token index for fast candidate retrieval
            indexed_tokens = set(label_toks)
            for st in syns_toks:
                indexed_tokens.update(st)

            for tok in indexed_tokens:
                if tok not in STOPWORDS:
                    self._token_index[tok].add(idx)
                    stem = _stem(tok)
                    if stem != tok and len(stem) >= 3:
                        self._token_index[stem].add(idx)
                    if len(tok) >= 4:
                        self._token_index[tok[:4]].add(idx)

        logger.info("HPO ontology indexed: %d terms.", len(self._entries))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, term: str, limit: int = 5) -> List[HPOResult]:
        """Search HPO for concepts matching *term*.

        Strategy (scored and ranked):
          1. Exact label match   (case-insensitive, score: 1.0)
          2. Exact synonym match (case-insensitive, score: 0.98)
          3. Substring label match (score: 0.90 - 0.95)
          4. Substring synonym match (score: 0.85)
          5. Token-level label match (score: 0.60 - 0.85)
          6. Token-level synonym match (score: 0.55 - 0.80)
          7. Fuzzy string match on label / synonym (score: 0.50 - 0.70)
          8. Definition match (substring / token, score: 0.40 - 0.50)

        Returns up to *limit* results sorted by match score.
        """
        self._ensure_loaded()

        query_raw = term.strip()
        query_norm = _normalize(query_raw)
        if not query_norm:
            return []

        # If query looks like an HPO ID, check direct lookup first
        if query_raw.upper().startswith("HP:") and query_raw.upper() in self._by_id:
            entry = self._by_id[query_raw.upper()]
            return [
                HPOResult(
                    hpo_id=entry.hpo_id,
                    label=entry.label,
                    synonyms=entry.synonyms_raw,
                    definition=entry.definition,
                    match_type="id_lookup",
                    score=1.0,
                )
            ]

        q_tokens = _tokenize(query_norm)
        q_effective = [t for t in q_tokens if t not in STOPWORDS] or q_tokens

        # Candidate retrieval via inverted token index
        candidates: Set[int] = set()
        for qt in q_effective:
            candidates.update(self._token_index.get(qt, ()))
            stem = _stem(qt)
            if stem != qt and len(stem) >= 3:
                candidates.update(self._token_index.get(stem, ()))
            if len(qt) >= 4:
                candidates.update(self._token_index.get(qt[:4], ()))

        # Fallback to scanning all if candidates set is very small
        eval_indices = candidates if candidates else range(len(self._entries))

        scored_matches: List[HPOResult] = []

        for idx in eval_indices:
            r = self._entries[idx]
            best_score = 0.0
            best_match_type = ""

            # 1. Exact label match
            if query_norm == r.label_norm:
                best_score = 1.0
                best_match_type = "exact_label"

            # 2. Exact synonym match
            elif query_norm in r.synonyms_norm:
                best_score = 0.98
                best_match_type = "synonym"

            # 3. Substring label match
            elif query_norm in r.label_norm:
                len_ratio = len(query_norm) / max(len(r.label_norm), 1)
                best_score = 0.90 + 0.05 * len_ratio
                best_match_type = "substring_label"

            # 4. Substring synonym match
            elif any(query_norm in sn for sn in r.synonyms_norm):
                best_score = 0.85
                best_match_type = "substring_synonym"

            else:
                # 5. Token-level matching
                l_tok_score = _score_tokens(q_effective, r.label_toks)
                s_tok_scores = [_score_tokens(q_effective, st) for st in r.synonyms_toks]
                s_tok_score = max(s_tok_scores) if s_tok_scores else 0.0

                # 6. Fuzzy string similarity
                l_fuzz = difflib.SequenceMatcher(None, query_norm, r.label_norm).ratio()
                s_fuzz = max(
                    [difflib.SequenceMatcher(None, query_norm, sn).ratio() for sn in r.synonyms_norm]
                    or [0.0]
                )

                if l_tok_score >= 0.6 and l_tok_score >= s_tok_score and l_tok_score >= l_fuzz:
                    best_score = 0.70 * l_tok_score + 0.15
                    best_match_type = "token_label"
                elif s_tok_score >= 0.6 and s_tok_score >= l_fuzz:
                    best_score = 0.65 * s_tok_score + 0.15
                    best_match_type = "token_synonym"
                elif l_fuzz >= 0.72:
                    best_score = 0.60 * l_fuzz + 0.10
                    best_match_type = "fuzzy_label"
                elif s_fuzz >= 0.72:
                    best_score = 0.55 * s_fuzz + 0.10
                    best_match_type = "fuzzy_synonym"
                elif r.definition_norm:
                    if query_norm in r.definition_norm:
                        best_score = 0.50
                        best_match_type = "substring_definition"
                    else:
                        d_tok = _score_tokens(q_effective, r.definition_toks)
                        if d_tok >= 0.75:
                            best_score = 0.40 + 0.10 * d_tok
                            best_match_type = "token_definition"

            if best_score > 0.0:
                scored_matches.append(
                    HPOResult(
                        hpo_id=r.hpo_id,
                        label=r.label,
                        synonyms=r.synonyms_raw,
                        definition=r.definition,
                        match_type=best_match_type,
                        score=round(best_score, 4),
                    )
                )

        scored_matches.sort(key=lambda x: x.score or 0.0, reverse=True)
        return scored_matches[:limit]

    def get_by_id(self, hpo_id: str) -> Optional[HPOResult]:
        """Fetch a specific HPO term by its ID (e.g. 'HP:0001250')."""
        self._ensure_loaded()
        entry = self._by_id.get(hpo_id.strip().upper())
        if not entry:
            return None

        return HPOResult(
            hpo_id=entry.hpo_id,
            label=entry.label,
            synonyms=entry.synonyms_raw,
            definition=entry.definition,
            match_type="id_lookup",
            score=1.0,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation edges."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _tokenize(text: str) -> List[str]:
    """Extract lowercase alphanumeric tokens."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def _stem(token: str) -> str:
    """Basic light suffix stripping for biomedical plural/inflections."""
    return re.sub(r"(ing|ed|es|s)$", "", token)


def _token_similarity(q_tok: str, t_tok: str) -> float:
    """Calculate similarity between two single tokens."""
    if q_tok == t_tok:
        return 1.0
    if q_tok.rstrip("s") == t_tok.rstrip("s"):
        return 0.95

    q_stem = _stem(q_tok)
    t_stem = _stem(t_tok)
    if len(q_stem) >= 3 and q_stem == t_stem:
        return 0.90

    if len(q_tok) >= 4 and len(t_tok) >= 4:
        if t_tok.startswith(q_tok) or q_tok.startswith(t_tok):
            return 0.85
        sim = difflib.SequenceMatcher(None, q_tok, t_tok).ratio()
        if sim >= 0.78:
            return sim

    return 0.0


def _score_tokens(query_tokens: List[str], target_tokens: List[str]) -> float:
    """Calculate token-overlap matching score from query to target."""
    if not query_tokens or not target_tokens:
        return 0.0

    matched_total = 0.0
    for qt in query_tokens:
        best = 0.0
        for tt in target_tokens:
            sim = _token_similarity(qt, tt)
            if sim > best:
                best = sim
                if best == 1.0:
                    break
        matched_total += best

    return matched_total / len(query_tokens)


# ---------------------------------------------------------------------------
# Cached singleton (one instance per obo_path across the process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_hpo_service(obo_path: str) -> HPOService:
    """Return a cached HPOService instance (loaded lazily on first search call)."""
    return HPOService(obo_path)
