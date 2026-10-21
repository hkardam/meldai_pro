"""Entity Matrix Service — per-entity MongoDB similarity retrieval.

For each symptom and diagnosis in the current encounter we:
  1. Fetch the full candidate pool from MongoDB (same pool as v1, cached per call).
  2. Score each candidate against the SINGLE target entity using the
     existing hybrid LCA + vector scorer from similarity_engine.py.
  3. Select top-k candidates.
  4. Aggregate the doctor's prescription frequency per molecule/FDC across
     those top-k cases.
  5. Return a ranked EntityMoleculeRow for each entity.

One symptom or diagnosis can be linked to 1 or more molecules/FDCs
because doctors may prescribe a primary drug, an adjuvant, and a supplement
all targeting the same complaint.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional

from meldai.api.req_dtos import SimilarityParams
from meldai.api.res_dtos import EntityMoleculeRow
from meldai.db.mongodb import MongoKnowledgeBase
from meldai.services.similarity_engine import (
    EntityItem,
    set_similarity,
)
from meldai.terminology.taxonomy import TaxonomyIndex
from meldai.utils import to_curie

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers to extract medications and normalize to canonical molecules
# ---------------------------------------------------------------------------

def _extract_meds_from_case(case_doc: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract medication objects from a MongoDB case document.

    Returns a list of dicts with keys: name (raw), canonical_molecule, is_fdc.
    """
    raw = case_doc.get("medications") or case_doc.get("medication") or []
    result = []
    for m in raw:
        if isinstance(m, str) and m.strip():
            result.append({"name": m.strip(), "canonical_molecule": m.strip(), "is_fdc": False})
        elif isinstance(m, dict):
            name = (
                m.get("term")
                or m.get("original_term")
                or m.get("brand_name")
                or m.get("name")
                or ""
            ).strip()
            # Prefer the matched canonical_molecule from migration enrichment; fall back to name
            molecule = (
                m.get("canonical_molecule")
                or (", ".join(m["canonical_molecule"]) if isinstance(m.get("canonical_molecule"), list) else None)
                or name
            )
            if isinstance(molecule, list):
                molecule = ", ".join(str(x) for x in molecule)
            is_fdc = bool(m.get("is_fdc", False))
            brand = m.get("brand_name") or name
            if name or molecule:
                result.append({
                    "name": name or molecule,
                    "canonical_molecule": str(molecule).strip() if molecule else name,
                    "brand_name": brand,
                    "is_fdc": is_fdc,
                })
    return result


def _parse_hpo_code(item: Dict[str, Any]) -> Optional[int]:
    code = item.get("hpoCode")
    if isinstance(code, int):
        return code
    if code is not None and str(code).isdigit():
        return int(code)
    hpo_id = item.get("hpoId")
    if hpo_id and isinstance(hpo_id, str):
        digits = "".join(filter(str.isdigit, hpo_id))
        return int(digits) if digits else None
    return None


def _parse_mondo_code(item: Dict[str, Any]) -> Optional[int]:
    code = item.get("mondoCode")
    if isinstance(code, int):
        return code
    if code is not None and str(code).isdigit():
        return int(code)
    mondo_id = item.get("mondoId")
    if mondo_id and isinstance(mondo_id, str):
        digits = "".join(filter(str.isdigit, mondo_id))
        return int(digits) if digits else None
    return None


# ---------------------------------------------------------------------------
# Core retrieval function for a single entity
# ---------------------------------------------------------------------------

def score_candidates_for_entity(
    entity_item: EntityItem,
    candidates: List[Dict[str, Any]],
    entity_type: str,   # "symptom" or "diagnosis"
    taxonomy: Optional[TaxonomyIndex],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Score all MongoDB candidates against a single entity and return the top-k.

    Each returned dict has: {case_doc, score}.
    """
    scored = []
    for cand in candidates:
        # Build the candidate's entity set for the relevant dimension
        if entity_type == "symptom":
            cand_entities: List[EntityItem] = []
            for s in (cand.get("chiefComplaints") or cand.get("symptoms") or []):
                if isinstance(s, dict):
                    cand_entities.append(EntityItem(
                        term=s.get("term") or s.get("text") or "",
                        embedding=s.get("embedding"),
                        ontology_code=_parse_hpo_code(s),
                    ))
                elif isinstance(s, str):
                    cand_entities.append(EntityItem(term=s))
        else:  # diagnosis
            cand_entities = []
            for d in cand.get("diagnosis", []):
                if isinstance(d, dict):
                    cand_entities.append(EntityItem(
                        term=d.get("term") or d.get("text") or "",
                        embedding=d.get("embedding"),
                        ontology_code=_parse_mondo_code(d),
                    ))
                elif isinstance(d, str):
                    cand_entities.append(EntityItem(term=d))

        if not cand_entities:
            continue

        # Use set_similarity with a singleton query set (just this one entity)
        score = set_similarity(
            query_set=[entity_item],
            candidate_set=cand_entities,
            taxonomy=taxonomy,
        )
        if score is not None and score > 0.0:
            scored.append({"case_doc": cand, "score": score})

    # Sort by score descending, take top_k
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Molecule frequency aggregation
# ---------------------------------------------------------------------------

def build_molecule_frequency(
    top_cases: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate molecule/FDC prescription frequency across the top-k cases.

    Returns a list of molecule entries sorted by frequency (descending):
    [
      {
        "canonical_molecule": "Escitalopram",
        "is_fdc": false,
        "frequency": 4,            # how many of the top-k cases prescribed it
        "avg_similarity": 0.87,    # average similarity score of those cases
        "doctor_brands": ["ESCIREN", "CIPRALEX"],
      },
      ...
    ]
    """
    freq: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "canonical_molecule": "",
        "is_fdc": False,
        "frequency": 0,
        "total_similarity": 0.0,
        "doctor_brands": set(),
    })

    for item in top_cases:
        case_doc = item["case_doc"]
        score = item["score"]
        meds = _extract_meds_from_case(case_doc)
        for med in meds:
            mol = med.get("canonical_molecule") or med.get("name") or ""
            if not mol:
                continue
            entry = freq[mol]
            entry["canonical_molecule"] = mol
            entry["is_fdc"] = med.get("is_fdc", False)
            entry["frequency"] += 1
            entry["total_similarity"] += score
            brand = med.get("brand_name") or med.get("name") or ""
            if brand and brand != mol:
                entry["doctor_brands"].add(brand)

    # Convert to sorted list
    rows = []
    for mol, data in freq.items():
        rows.append({
            "canonical_molecule": data["canonical_molecule"],
            "is_fdc": data["is_fdc"],
            "frequency": data["frequency"],
            "avg_similarity": round(data["total_similarity"] / data["frequency"], 4),
            "doctor_brands": list(data["doctor_brands"])[:5],  # cap at 5 brand examples
        })
    rows.sort(key=lambda x: (x["frequency"], x["avg_similarity"]), reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_entity_matrix(
    *,
    # Current encounter entities already decorated by CaseService
    symptoms: List[Any],       # List of DecoratedSymptomItem
    diagnoses: List[Any],      # List of DecoratedDiagnosisItem
    # Only score entities in these lists; others already handled by delta engine
    target_symptom_terms: List[str],
    target_diagnosis_terms: List[str],
    # MongoDB candidate pool (fetched once by the caller)
    candidates: List[Dict[str, Any]],
    top_k: int,
    hpo_taxonomy: Optional[TaxonomyIndex],
    mondo_taxonomy: Optional[TaxonomyIndex],
) -> List[EntityMoleculeRow]:
    """Build the entity-to-molecule matrix.

    For each target symptom and diagnosis:
      - Score the full candidate pool from MongoDB.
      - Collect top-k candidates.
      - Aggregate molecule frequency.
      - Return an EntityMoleculeRow.

    One entity may link to multiple molecules (e.g. Escitalopram + Clonazepam
    both prescribed for anxiety in different cases, or both in the same case).
    """
    target_sym_lower = {t.strip().lower() for t in target_symptom_terms if t.strip()}
    target_dx_lower = {t.strip().lower() for t in target_diagnosis_terms if t.strip()}

    matrix: List[EntityMoleculeRow] = []

    # ------------------------------------------------------------------
    # Process symptoms
    # ------------------------------------------------------------------
    for sym in symptoms:
        term = getattr(sym, "term", "") or ""
        if term.strip().lower() not in target_sym_lower:
            continue  # skip persistent/improving — handled separately

        curie = getattr(sym, "hpoCurie", None) or to_curie("HP", getattr(sym, "hpoCode", None))

        entity_item = EntityItem(
            term=term,
            embedding=getattr(sym, "embedding", None),
            ontology_code=getattr(sym, "hpoCode", None),
        )

        logger.info("Entity matrix — scoring for symptom: %s", term)
        top_cases = score_candidates_for_entity(
            entity_item=entity_item,
            candidates=candidates,
            entity_type="symptom",
            taxonomy=hpo_taxonomy,
            top_k=top_k,
        )

        mol_freq = build_molecule_frequency(top_cases)

        matrix.append(EntityMoleculeRow(
            entity=term,
            entity_type="symptom",
            curie=curie,
            top_molecules=mol_freq,
            attributed_molecules=[],  # filled in by LLM attribution step
        ))

    # ------------------------------------------------------------------
    # Process diagnoses
    # ------------------------------------------------------------------
    for dx in diagnoses:
        term = getattr(dx, "term", "") or ""
        if term.strip().lower() not in target_dx_lower:
            continue

        curie = getattr(dx, "mondoCurie", None) or to_curie("MONDO", getattr(dx, "mondoCode", None))

        entity_item = EntityItem(
            term=term,
            embedding=getattr(dx, "embedding", None),
            ontology_code=getattr(dx, "mondoCode", None),
        )

        logger.info("Entity matrix — scoring for diagnosis: %s", term)
        top_cases = score_candidates_for_entity(
            entity_item=entity_item,
            candidates=candidates,
            entity_type="diagnosis",
            taxonomy=mondo_taxonomy,
            top_k=top_k,
        )

        mol_freq = build_molecule_frequency(top_cases)

        matrix.append(EntityMoleculeRow(
            entity=term,
            entity_type="diagnosis",
            curie=curie,
            top_molecules=mol_freq,
            attributed_molecules=[],
        ))

    return matrix
