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
import numpy as np

from meldai.api.req_dtos import RankMoleculesRequest, SimilarityParams
from meldai.api.res_dtos import EntityMoleculeRow, RankedMoleculeItem, RankMoleculesResponse
from meldai.config import Settings, get_settings
from meldai.db.mongodb import MongoKnowledgeBase
from meldai.nlp.sapbert import get_sapbert_embedder
from meldai.services.similarity_engine import (
    EntityItem,
    compute_composite_score,
    cosine_similarity,
    patient_info_similarity,
    patient_similarity,
    set_similarity,
)
from meldai.terminology.hpo import get_hpo_service
from meldai.terminology.mondo import get_mondo_service
from meldai.terminology.taxonomy import TaxonomyIndex, get_hpo_taxonomy, get_mondo_taxonomy
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


# ---------------------------------------------------------------------------
# Standalone Rank Meds Workflow
# ---------------------------------------------------------------------------

def rank_meds(
    request: RankMoleculesRequest,
    mongo_kb: Optional[MongoKnowledgeBase] = None,
    settings: Optional[Settings] = None,
) -> RankMoleculesResponse:
    """Retrieve top-k molecules ranked by prescription score from similar cases matching LCA + vector (meeting minimum similarity cutoff) and sorted post-match by patient demographic similarity (age group + gender)."""
    cfg = settings or get_settings()
    kb = mongo_kb or MongoKnowledgeBase()

    # 1. Normalize entity type
    raw_type = request.type.strip().lower()
    if raw_type in ("chiefcomplaint", "chiefcomplain", "chielfcomplain", "chielfcomplaint", "symptom", "chief_complaint", "complaint"):
        entity_type = "chiefComplaint"
        clean_type = "symptom"
    else:
        entity_type = "diagnosis"
        clean_type = "diagnosis"

    # 2. Resolve query entity (HPO / MONDO) and generate SapBERT embedding
    hpo_svc = get_hpo_service(cfg.hpo_obo_path)
    mondo_svc = get_mondo_service(cfg.mondo_obo_path)
    embedder = get_sapbert_embedder(cfg)

    query_term = request.query.strip()
    query_vector = embedder.embed_entities([query_term])[0]

    query_code: Optional[int] = None
    query_curie: Optional[str] = None
    matched_term: Optional[str] = None
    taxonomy: Optional[TaxonomyIndex] = None

    if clean_type == "symptom":
        try:
            taxonomy = get_hpo_taxonomy(cfg.hpo_obo_path)
        except Exception:
            taxonomy = None
        hits = hpo_svc.search(query_term, limit=1)
        if hits:
            hit = hits[0]
            matched_term = hit.label
            query_curie = hit.hpo_id
            digits = "".join(filter(str.isdigit, hit.hpo_id))
            query_code = int(digits) if digits else None
    else:
        try:
            taxonomy = get_mondo_taxonomy(cfg.mondo_obo_path)
        except Exception:
            taxonomy = None
        hits = mondo_svc.search(query_term, limit=1)
        if hits:
            hit = hits[0]
            matched_term = hit.label
            query_curie = hit.mondo_id
            digits = "".join(filter(str.isdigit, hit.mondo_id))
            query_code = int(digits) if digits else None

    # 3. Retrieve MongoDB candidate pool
    candidates = kb.find_similar_candidates(limit=request.candidate_pool_limit)

    # 4. Score each candidate case
    # Step 4a: Clinical similarity matching (LCA + vector)
    w_lca = request.weights.w_lca if request.weights else 0.5
    w_vec = request.weights.w_vector if request.weights else 0.5
    min_sim = request.weights.min_similarity if (request.weights and request.weights.min_similarity is not None) else 0.5

    clean_q_low = query_term.lower()
    scored_cases: List[Dict[str, Any]] = []

    for cand in candidates:
        if clean_type == "symptom":
            cand_entities = cand.get("chiefComplaints") or cand.get("symptoms") or []
        else:
            cand_entities = cand.get("diagnosis") or []

        if not cand_entities:
            continue

        best_lca = 0.0
        best_vec = 0.0

        for e in cand_entities:
            if isinstance(e, dict):
                term = (e.get("term") or e.get("text") or "").strip().lower()
                e_code = _parse_hpo_code(e) if clean_type == "symptom" else _parse_mondo_code(e)
                e_vec = e.get("embedding")
            else:
                term = str(e).strip().lower()
                e_code = None
                e_vec = None

            # LCA (Wu-Palmer)
            if term == clean_q_low or (query_code is not None and e_code is not None and query_code == e_code):
                lca_val = 1.0
            elif taxonomy is not None and query_code is not None and e_code is not None:
                lca_val = taxonomy.wu_palmer(query_code, e_code)
            else:
                lca_val = 0.0

            if lca_val > best_lca:
                best_lca = lca_val

            # Vector (Cosine)
            if term == clean_q_low:
                vec_val = 1.0
            elif query_vector is not None and e_vec is not None:
                cos = cosine_similarity(query_vector, e_vec)
                vec_val = max(cos, 0.0)
            else:
                vec_val = 0.0

            if vec_val > best_vec:
                best_vec = vec_val

        # Clinical match similarity (LCA + vector only, demographic excluded from matching)
        component_scores = {
            "lca": best_lca,
            "vector": best_vec,
        }
        weights_dict = {
            "lca": w_lca,
            "vector": w_vec,
        }
        clinical_sim = compute_composite_score(component_scores, weights_dict)

        # Minimum similarity cutoff required to call it a match
        if clinical_sim < min_sim:
            continue

        # Step 4b: Post-match demographic similarity (age group + gender)
        cand_age = None
        cand_gender = None
        p_info = cand.get("patientInfo")
        if isinstance(p_info, dict):
            cand_age = p_info.get("age")
            cand_gender = p_info.get("gender")
        if cand_age is None:
            cand_age = cand.get("age")
            if cand_age is None and isinstance(cand.get("patient"), dict):
                cand_age = (
                    cand["patient"].get("age", {}).get("year")
                    if isinstance(cand["patient"].get("age"), dict)
                    else cand["patient"].get("age")
                )
        if cand_gender is None:
            cand_gender = cand.get("gender")
            if cand_gender is None and isinstance(cand.get("patient"), dict):
                cand_gender = cand["patient"].get("gender")

        # Filter out candidates with incompatible age similarity (age_sim <= 0.0 across 3+ age bands)
        cand_age_sim = None
        if request.patient_age is not None and cand_age is not None:
            cand_age_sim = patient_similarity(request.patient_age, cand_age)
            if cand_age_sim is not None and cand_age_sim <= 0.0:
                continue

        p_sim = patient_info_similarity(
            age_q=request.patient_age,
            age_c=cand_age,
            gender_q=request.patient_gender,
            gender_c=cand_gender,
        )

        # Combined case ranking score (clinical similarity + patient demographic similarity)
        if p_sim is not None:
            case_score = round(0.5 * clinical_sim + 0.5 * p_sim, 4)
        else:
            case_score = round(clinical_sim, 4)

        scored_cases.append({
            "case_doc": cand,
            "score": case_score,
            "clinical_similarity": clinical_sim,
            "lca_score": best_lca,
            "vec_score": best_vec,
            "patient_info_score": p_sim,
            "age_similarity": cand_age_sim,
        })

    # Sort descending by combined case score, secondary tie-break by clinical similarity
    scored_cases.sort(key=lambda x: (x["score"], x["clinical_similarity"]), reverse=True)
    top_cases = scored_cases[:request.top_cases_limit]

    # Calculate applied normalized weights
    tot_w = w_lca + w_vec
    applied_weights = {
        "w_lca": round(w_lca / tot_w, 4) if tot_w > 0 else 0.0,
        "w_vector": round(w_vec / tot_w, 4) if tot_w > 0 else 0.0,
        "min_similarity": min_sim,
    }
    if (request.patient_age is not None or request.patient_gender is not None) and any(
        c.get("patient_info_score") is not None for c in top_cases
    ):
        applied_weights["patient_info_used"] = 1.0

    # 5. Aggregate molecules across top cases
    mol_counts: Dict[str, int] = defaultdict(int)
    mol_scores: Dict[str, List[float]] = defaultdict(list)
    mol_age_sims: Dict[str, List[float]] = defaultdict(list)
    mol_brands: Dict[str, set[str]] = defaultdict(set)
    mol_is_fdc: Dict[str, bool] = {}

    for item in top_cases:
        case_score = item["score"]
        c_age_sim = item.get("age_similarity")
        meds = _extract_meds_from_case(item["case_doc"])
        seen_in_case: set[str] = set()

        for med in meds:
            canon = med.get("canonical_molecule") or med.get("name")
            if not canon:
                continue
            canon = canon.strip()
            if canon.lower() in seen_in_case:
                continue
            seen_in_case.add(canon.lower())

            mol_counts[canon] += 1
            mol_scores[canon].append(case_score)
            if c_age_sim is not None:
                mol_age_sims[canon].append(c_age_sim)
            brand = med.get("brand_name") or med.get("name")
            if brand and brand.strip() and brand.strip() != canon:
                mol_brands[canon].add(brand.strip())
            mol_is_fdc[canon] = med.get("is_fdc", False)

    num_top = len(top_cases)
    ranked_molecules: List[RankedMoleculeItem] = []

    for canon, count in mol_counts.items():
        # Filter out meds whose average age similarity is incompatible
        if request.patient_age is not None and canon in mol_age_sims and mol_age_sims[canon]:
            avg_mol_age_sim = float(np.mean(mol_age_sims[canon]))
            if avg_mol_age_sim <= 0.0:
                continue

        total_score = round(float(np.sum(mol_scores[canon])), 4)
        pct = round((count / num_top) * 100, 1) if num_top > 0 else 0.0
        avg_sim = round(float(np.mean(mol_scores[canon])), 4) if mol_scores[canon] else 0.0
        ranked_molecules.append(RankedMoleculeItem(
            canonical_molecule=canon,
            is_fdc=mol_is_fdc.get(canon, False),
            score=total_score,
            frequency=count,
            percentage=pct,
            avg_similarity=avg_sim,
            doctor_brands=sorted(list(mol_brands[canon]))[:5],
        ))

    # Sort primarily by cumulative score desc, secondarily by frequency desc, then avg_similarity desc
    ranked_molecules.sort(key=lambda x: (x.score, x.frequency, x.avg_similarity), reverse=True)

    return RankMoleculesResponse(
        type=entity_type,
        query=query_term,
        k=request.k,
        curie=query_curie,
        matched_term=matched_term,
        similar_cases_count=num_top,
        weights_applied=applied_weights,
        molecules=ranked_molecules[:request.k],
    )

# ---------------------------------------------------------------------------
# Per-entity rank helper — used by prescribe_medications_v2 (no LLM, no
# MongoDB fetch; caller supplies a pre-filtered candidate pool)
# ---------------------------------------------------------------------------

def rank_meds_for_entity(
    *,
    query_term: str,
    entity_type: str,           # "symptom" | "diagnosis"
    query_code,
    query_curie,
    query_vector,
    taxonomy,
    candidates,
    patient_age=None,
    patient_gender=None,
    w_lca: float = 0.5,
    w_vec: float = 0.5,
    min_sim: float = 0.5,
    top_cases_limit: int = 20,
    k: int = 10,
):
    """Score pre-fetched candidates for a single entity and return an EntityMoleculeRow.

    Uses the same LCA + vector + demographic pipeline as ``rank_meds`` but
    operates on an already-filtered candidate pool so callers can share one
    DB fetch across all entities and avoid redundant MongoDB round-trips.

    Parameters
    ----------
    query_term:       Raw entity string (symptom or diagnosis).
    entity_type:      ``"symptom"`` (chief complaint) or ``"diagnosis"``.
    query_code:       Numeric HPO/MONDO code resolved for the query term.
    query_curie:      Full CURIE string (e.g. ``"HP:0002360"``).
    query_vector:     SapBERT embedding for the query term (numpy array or list).
    taxonomy:         HPO or MONDO TaxonomyIndex for Wu-Palmer LCA scoring.
    candidates:       Pre-fetched (and pre-filtered) MongoDB case documents.
    patient_age:      Current patient age in years (optional).
    patient_gender:   Current patient gender string (optional).
    w_lca / w_vec:    Composite scoring weights.
    min_sim:          Minimum clinical similarity cutoff.
    top_cases_limit:  How many top-scoring cases to aggregate molecules from.
    k:                Cap on molecules to include in top_molecules.
    """
    from meldai.api.res_dtos import EntityMoleculeRow as _EMR

    clean_type = entity_type  # "symptom" or "diagnosis"
    clean_q_low = query_term.strip().lower()

    scored_cases = []

    for cand in candidates:
        if clean_type == "symptom":
            cand_entities = cand.get("chiefComplaints") or cand.get("symptoms") or []
        else:
            cand_entities = cand.get("diagnosis") or []

        if not cand_entities:
            continue

        best_lca = 0.0
        best_vec = 0.0

        for e in cand_entities:
            if isinstance(e, dict):
                term = (e.get("term") or e.get("text") or "").strip().lower()
                e_code = _parse_hpo_code(e) if clean_type == "symptom" else _parse_mondo_code(e)
                e_vec = e.get("embedding")
            else:
                term = str(e).strip().lower()
                e_code = None
                e_vec = None

            # LCA (Wu-Palmer)
            if term == clean_q_low or (query_code is not None and e_code is not None and query_code == e_code):
                lca_val = 1.0
            elif taxonomy is not None and query_code is not None and e_code is not None:
                lca_val = taxonomy.wu_palmer(query_code, e_code)
            else:
                lca_val = 0.0

            if lca_val > best_lca:
                best_lca = lca_val

            # Vector (Cosine)
            if term == clean_q_low:
                vec_val = 1.0
            elif query_vector is not None and e_vec is not None:
                cos = cosine_similarity(query_vector, e_vec)
                vec_val = max(cos, 0.0)
            else:
                vec_val = 0.0

            if vec_val > best_vec:
                best_vec = vec_val

        clinical_sim = compute_composite_score(
            {"lca": best_lca, "vector": best_vec},
            {"lca": w_lca, "vector": w_vec},
        )

        if clinical_sim < min_sim:
            continue

        # Post-match demographic similarity
        cand_age = None
        cand_gender = None
        p_info = cand.get("patientInfo")
        if isinstance(p_info, dict):
            cand_age = p_info.get("age")
            cand_gender = p_info.get("gender")
        if cand_age is None:
            cand_age = cand.get("age")
            if cand_age is None and isinstance(cand.get("patient"), dict):
                cand_age = (
                    cand["patient"].get("age", {}).get("year")
                    if isinstance(cand["patient"].get("age"), dict)
                    else cand["patient"].get("age")
                )
        if cand_gender is None:
            cand_gender = cand.get("gender")
            if cand_gender is None and isinstance(cand.get("patient"), dict):
                cand_gender = cand["patient"].get("gender")

        # Filter out candidates with incompatible age similarity (age_sim <= 0.0 across 3+ age bands)
        cand_age_sim = None
        if patient_age is not None and cand_age is not None:
            cand_age_sim = patient_similarity(patient_age, cand_age)
            if cand_age_sim is not None and cand_age_sim <= 0.0:
                continue

        p_sim = patient_info_similarity(
            age_q=patient_age,
            age_c=cand_age,
            gender_q=patient_gender,
            gender_c=cand_gender,
        )

        case_score = (
            round(0.5 * clinical_sim + 0.5 * p_sim, 4) if p_sim is not None
            else round(clinical_sim, 4)
        )

        scored_cases.append({
            "case_doc": cand,
            "score": case_score,
            "clinical_similarity": clinical_sim,
            "age_similarity": cand_age_sim,
        })

    scored_cases.sort(key=lambda x: (x["score"], x["clinical_similarity"]), reverse=True)
    top_cases = scored_cases[:top_cases_limit]

    # Aggregate molecules from top cases
    mol_counts = defaultdict(int)
    mol_scores_map = defaultdict(list)
    mol_age_sims = defaultdict(list)
    mol_brands = defaultdict(set)
    mol_is_fdc = {}

    for item in top_cases:
        case_score = item["score"]
        c_age_sim = item.get("age_similarity")
        meds = _extract_meds_from_case(item["case_doc"])
        seen_in_case = set()

        for med in meds:
            canon = med.get("canonical_molecule") or med.get("name")
            if not canon:
                continue
            canon = canon.strip()
            if canon.lower() in seen_in_case:
                continue
            seen_in_case.add(canon.lower())

            mol_counts[canon] += 1
            mol_scores_map[canon].append(case_score)
            if c_age_sim is not None:
                mol_age_sims[canon].append(c_age_sim)
            brand = med.get("brand_name") or med.get("name")
            if brand and brand.strip() and brand.strip() != canon:
                mol_brands[canon].add(brand.strip())
            mol_is_fdc[canon] = med.get("is_fdc", False)

    num_top = len(top_cases)
    top_molecules = []
    for canon, count in mol_counts.items():
        # Filter out meds whose average age similarity is incompatible
        if patient_age is not None and canon in mol_age_sims and mol_age_sims[canon]:
            avg_mol_age_sim = float(np.mean(mol_age_sims[canon]))
            if avg_mol_age_sim <= 0.0:
                continue

        scores_list = mol_scores_map[canon]
        avg_sim = round(float(np.mean(scores_list)), 4) if scores_list else 0.0
        total_score = round(float(np.sum(scores_list)), 4)
        pct = round((count / num_top) * 100, 1) if num_top > 0 else 0.0
        top_molecules.append({
            "canonical_molecule": canon,
            "is_fdc": mol_is_fdc.get(canon, False),
            "frequency": count,
            "percentage": pct,
            "avg_similarity": avg_sim,
            "score": total_score,
            "doctor_brands": sorted(list(mol_brands[canon]))[:5],
        })

    top_molecules.sort(key=lambda x: (x["score"], x["frequency"], x["avg_similarity"]), reverse=True)

    entity_type_label = "symptom" if clean_type == "symptom" else "diagnosis"
    return _EMR(
        entity=query_term,
        entity_type=entity_type_label,
        curie=query_curie,
        top_molecules=top_molecules[:k],
        attributed_molecules=[],
    )
