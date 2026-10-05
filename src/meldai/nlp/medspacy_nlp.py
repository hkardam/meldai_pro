"""Clinical NLP pipeline using medspaCy + spaCy en_core_web_md for phenotype extraction.

Architecture (no scispacy — incompatible with Python 3.12 due to scipy<1.11 constraint):

  en_core_web_md (spaCy 3.8)
    ├─ tok2vec / tagger / parser / NER   — standard English NLP
    ├─ noun_to_ents (custom)             — promotes noun chunks → SYMPTOM entities
    │                                       so ConText has spans to annotate
    └─ medspacy_context (ConText)        — detects negation, historical context,
                                           uncertainty, family history on entities

Phenotype decision:
    is_phenotype = (has ≥1 entity) AND (no entity is negated/historical/uncertain/family)

Embedding:
    doc.vector  — 300-d L2-normalised word vector from en_core_web_md

HPO code:
    HPOService.search(text, limit=1)  — offline pronto-based OBO lookup (already in codebase)
"""

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional

from spacy.language import Language
from spacy.tokens import Doc, Span
from spacy.util import filter_spans

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SegmentResult:
    """NLP analysis result for a single clinical text segment."""
    text: str
    is_phenotype: bool
    embedding: List[float] = field(default_factory=list)
    hpo_code: Optional[int] = None


# ---------------------------------------------------------------------------
# Custom spaCy component: noun chunks → SYMPTOM entities
# ---------------------------------------------------------------------------

@Language.component("noun_to_ents")
def noun_chunks_to_entities(doc: Doc) -> Doc:
    """Promote non-stop noun chunks to SYMPTOM entities.

    Leverages medspaCy's context graph to detect modifier spans (e.g. negation,
    historical cues) and trims them from candidate noun spans so ConText can
    properly link assertion targets. Overlapping spans are resolved via filter_spans.
    """
    modifier_indices = set()
    context_graph = getattr(doc._, "context_graph", None)
    if context_graph and hasattr(context_graph, "modifiers"):
        for m in context_graph.modifiers:
            span = getattr(m, "modifier_span", None)
            if span:
                modifier_indices.update(range(span[0], span[1]))

    candidates: List[Span] = []
    for chunk in doc.noun_chunks:
        start, end = chunk.start, chunk.end
        while start < end and start in modifier_indices:
            start += 1
        while end > start and (end - 1) in modifier_indices:
            end -= 1
        if start < end:
            sub = doc[start:end]
            if len(sub.text.strip()) >= 2 and not sub.root.is_stop:
                candidates.append(
                    sub.char_span(0, len(sub.text), label="SYMPTOM", alignment_mode="contract")
                    or Span(doc, start, end, label="SYMPTOM")
                )

    # Fallback for isolated single-word or short terms not parsed as noun chunks
    if not candidates and not doc.ents:
        start, end = 0, len(doc)
        while start < end and start in modifier_indices:
            start += 1
        while end > start and (end - 1) in modifier_indices:
            end -= 1
        if start < end:
            sub = doc[start:end]
            if len(sub.text.strip()) >= 2 and not sub.root.is_stop:
                candidates.append(
                    sub.char_span(0, len(sub.text), label="SYMPTOM", alignment_mode="contract")
                    or Span(doc, start, end, label="SYMPTOM")
                )

    all_spans = list(doc.ents) + candidates
    doc.ents = filter_spans(all_spans)
    return doc


# ---------------------------------------------------------------------------
# Singleton NLP service
# ---------------------------------------------------------------------------

class MedspaCyService:
    """Lazy-loaded singleton clinical NLP service.

    Pipeline components (in order):
    - en_core_web_md built-ins (tok2vec, tagger, parser, NER, etc.)
    - noun_to_ents: promotes noun chunks to SYMPTOM entities
    - medspacy_context: ConText negation / historical / uncertainty / family detection

    HPO codes are resolved offline via HPOService (pronto OBO file).
    """

    def __init__(self) -> None:
        self._nlp = None
        self._hpo_service = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the full NLP pipeline on first call."""
        if self._nlp is not None:
            return

        logger.info("Loading en_core_web_md + noun_to_ents + medspaCy ConText …")
        import spacy
        import medspacy  # noqa: F401 — registers medspacy_context component

        nlp = spacy.load("en_core_web_md")

        # Pipeline setup:
        # 1. medspacy_context_modifiers: scans text for assertion modifier cues into doc._.context_graph.modifiers
        # 2. noun_to_ents: extracts noun chunks excluding tokens in modifier spans
        # 3. medspacy_context: links target entities to modifiers and sets ent._.is_negated, etc.
        if "medspacy_context_modifiers" not in nlp.pipe_names:
            nlp.add_pipe("medspacy_context", name="medspacy_context_modifiers", last=True)

        if "noun_to_ents" not in nlp.pipe_names:
            nlp.add_pipe("noun_to_ents", last=True)

        if "medspacy_context" not in nlp.pipe_names:
            nlp.add_pipe("medspacy_context", name="medspacy_context", last=True)

        self._nlp = nlp
        logger.info("MedspaCyService pipeline ready: %s", nlp.pipe_names)

        # Pre-load HPO service (offline pronto OBO lookup)
        try:
            from meldai.config import get_settings
            from meldai.terminology.hpo import get_hpo_service
            settings = get_settings()
            self._hpo_service = get_hpo_service(settings.hpo_obo_path)
            logger.info("HPOService loaded for HPO code lookup")
        except Exception as exc:
            logger.warning("HPOService unavailable (HPO codes will be null): %s", exc)

    def process(self, segments: List[str]) -> List[SegmentResult]:
        """Analyse a list of clinical text segments.

        For each segment:
        - noun_to_ents promotes noun chunks → SYMPTOM entities
        - medspaCy ConText determines assertion status (negated / historical / uncertain)
        - A segment is classified as a phenotype if:
            (a) at least one entity is present, AND
            (b) no entity is negated / historical / uncertain / family
        - Embedding = 300-d L2-normalised doc.vector from en_core_web_md
        - HPO code = offline HPOService.search(text, limit=1)

        Returns a list of SegmentResult objects, one per input segment.
        """
        self._ensure_loaded()

        results: List[SegmentResult] = []

        for doc in self._nlp.pipe(segments, batch_size=32):
            # --- embedding ---
            if getattr(doc, "has_vector", False):
                vec = doc.vector.tolist() if hasattr(doc.vector, "tolist") else list(doc.vector)
            else:
                vec = []

            # --- phenotype detection via ConText-annotated entities ---
            has_entity = len(doc.ents) > 0
            all_affirmed = (
                all(
                    not getattr(ent._, "is_negated", False)
                    and not getattr(ent._, "is_historical", False)
                    and not getattr(ent._, "is_uncertain", False)
                    and not getattr(ent._, "is_family", False)
                    for ent in doc.ents
                )
                if has_entity
                else False
            )
            is_pheno = has_entity and all_affirmed

            # --- HPO code via offline pronto lookup or entity kb_ents fallback ---
            hpo_code: Optional[int] = None
            if is_pheno:
                hpo_service = getattr(self, "_hpo_service", None)
                if hpo_service:
                    try:
                        results_hpo = hpo_service.search(doc.text, limit=1)
                        if results_hpo:
                            m = re.search(r"\d+", results_hpo[0].hpo_id)
                            if m:
                                hpo_code = int(m.group())
                    except Exception as exc:
                        logger.debug("HPO lookup failed for '%s': %s", doc.text, exc)

                # Fallback to ent._.kb_ents if available (e.g. from mock tests or linker)
                if hpo_code is None:
                    for ent in doc.ents:
                        kb_ents = getattr(ent._, "kb_ents", None)
                        if kb_ents:
                            m = re.search(r"\d+", str(kb_ents[0][0]))
                            if m:
                                hpo_code = int(m.group())
                                break

            results.append(
                SegmentResult(
                    text=doc.text,
                    is_phenotype=is_pheno,
                    embedding=vec,
                    hpo_code=hpo_code,
                )
            )

        return results

    def analyze(self, term: str) -> Doc:
        """Parse a clinical search query using medspaCy and return the annotated Doc."""
        self._ensure_loaded()
        return self._nlp(term.strip())

    def analyze_batch(self, terms: List[str], batch_size: int = 64) -> List[Doc]:
        """Parse multiple clinical search queries in batch using medspaCy nlp.pipe."""
        self._ensure_loaded()
        clean_terms = [t.strip() for t in terms]
        return list(self._nlp.pipe(clean_terms, batch_size=batch_size))



@lru_cache(maxsize=1)
def get_medspacy_service() -> MedspaCyService:
    """Return the cached singleton MedspaCyService instance."""
    return MedspaCyService()
