"""Clinical Delta Service — compares past visits against the current encounter.

Classifies symptoms and diagnoses into clinical statuses:
  - stop: was present in prior visits, now completely absent (resolved)
  - maintained: present in prior visits and current visit with unchanged/stable severity
  - decreased: present in prior visits, current visit indicates milder severity (improving)
  - increased: present in prior visits, current visit indicates worsening/increased severity
  - new: not seen before (for new cases, every symptom is new)

These statuses drive the medication review (deprescribing / dose-adjustment) and the
per-entity medication matrix.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple severity heuristics
# ---------------------------------------------------------------------------

_MILD_MARKERS = {"mild", "slight", "minor", "occasional", "intermittent", "improving", "better", "less", "decreased"}
_SEVERE_MARKERS = {"severe", "significant", "marked", "persistent", "constant", "worse", "worsening", "increased", "acute"}


def _is_mild(term: str) -> bool:
    t = term.lower()
    return any(m in t for m in _MILD_MARKERS)


def _is_severe(term: str) -> bool:
    t = term.lower()
    return any(m in t for m in _SEVERE_MARKERS)


def _severity_level(term: str) -> int:
    """Return 0 (mild), 1 (neutral), 2 (severe) based on term text."""
    if _is_mild(term):
        return 0
    if _is_severe(term):
        return 2
    return 1


def _extract_term_set(entities: List[Any]) -> set[str]:
    """Normalize a list of symptom/diagnosis dicts or strings to a set of lower-cased terms."""
    out = set()
    for e in entities:
        if isinstance(e, dict):
            term = e.get("term") or e.get("text") or e.get("note") or ""
        else:
            term = str(e)
        term = term.strip().lower()
        if term:
            out.add(term)
    return out


def _extract_meds(case_doc: Dict[str, Any]) -> List[str]:
    """Extract medication names from a MongoDB case document."""
    meds = case_doc.get("medications") or case_doc.get("medication") or []
    result = []
    for m in meds:
        if isinstance(m, str) and m.strip():
            result.append(m.strip())
        elif isinstance(m, dict):
            val = (
                m.get("term")
                or m.get("original_term")
                or m.get("brand_name")
                or m.get("name")
            )
            if val:
                result.append(str(val).strip())
    return result


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

def compute_delta(
    current_symptoms: List[str],
    current_diagnoses: List[str],
    past_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare current encounter entities against all past visits.

    Returns a dict with:
        new_symptoms         – terms with no match in any past visit
        maintained_symptoms  – terms present in past AND current (unchanged/stable)
        decreased_symptoms   – present in past, current text looks milder (improving)
        increased_symptoms   – present in past, current text looks worse (increasing)
        stop_symptoms        – present in past, completely absent in current (resolved)
        (symmetric fields for diagnoses)
        entity_actions       – {entity_term: "stop" | "maintained" | "decreased" | "increased" | "new"}
        prior_meds_for_entities – {entity_term: {"status": ..., "prior_meds": [...]}}
        summary_text         – human-readable summary for prompt injection
    """
    if not past_cases:
        # New case or patient with no prior history — everything is 'new'
        cur_syms = list(current_symptoms)
        cur_dxs = list(current_diagnoses)
        entity_actions = {s: "new" for s in cur_syms}
        entity_actions.update({d: "new" for d in cur_dxs})

        prior_meds_for_entities = {
            ent: {"status": "new", "prior_meds": []}
            for ent in entity_actions
        }

        return {
            "new_symptoms": cur_syms,
            "maintained_symptoms": [],
            "decreased_symptoms": [],
            "increased_symptoms": [],
            "stop_symptoms": [],
            "resolved_symptoms": [],  # backward compatibility alias
            "persistent_symptoms": [],  # backward compatibility alias
            "improving_symptoms": [],  # backward compatibility alias
            "new_diagnoses": cur_dxs,
            "maintained_diagnoses": [],
            "stop_diagnoses": [],
            "resolved_diagnoses": [],
            "persistent_diagnoses": [],
            "entity_actions": entity_actions,
            "prior_meds_for_entities": prior_meds_for_entities,
            "prior_meds_for_resolved": {},
            "summary_text": "New consultation (no prior visits). All symptoms and diagnoses are NEW.",
        }

    # Aggregate all terms seen across past visits
    past_symptom_terms: set[str] = set()
    past_diagnosis_terms: set[str] = set()
    all_past_meds: List[str] = []

    for visit in past_cases:
        sym_entities = (
            visit.get("chiefComplaints")
            or visit.get("symptoms")
            or []
        )
        past_symptom_terms |= _extract_term_set(sym_entities)
        past_diagnosis_terms |= _extract_term_set(visit.get("diagnosis", []))
        all_past_meds.extend(_extract_meds(visit))

    unique_past_meds = list(dict.fromkeys(all_past_meds))  # deduplicate preserving order
    cur_sym_lower = {s.strip().lower(): s for s in current_symptoms if s.strip()}
    cur_dx_lower = {d.strip().lower(): d for d in current_diagnoses if d.strip()}

    # ------------------------------------------------------------------
    # Classify symptoms: stop | maintained | decreased | increased | new
    # ------------------------------------------------------------------
    new_symptoms: List[str] = []
    maintained_symptoms: List[str] = []
    decreased_symptoms: List[str] = []
    increased_symptoms: List[str] = []
    stop_symptoms: List[str] = []

    for past_term in past_symptom_terms:
        matched_current = _find_match(past_term, cur_sym_lower)
        if matched_current is None:
            stop_symptoms.append(past_term)
        else:
            past_sev = _severity_level(past_term)
            cur_sev = _severity_level(matched_current)
            if cur_sev > past_sev or (_is_severe(matched_current) and not _is_severe(past_term)):
                increased_symptoms.append(matched_current)
            elif cur_sev < past_sev or (_is_mild(matched_current) and not _is_mild(past_term)):
                decreased_symptoms.append(matched_current)
            else:
                maintained_symptoms.append(matched_current)

    for cur_low, cur_orig in cur_sym_lower.items():
        if _find_match(cur_low, {p: p for p in past_symptom_terms}) is None:
            new_symptoms.append(cur_orig)

    # ------------------------------------------------------------------
    # Classify diagnoses: stop | maintained | new (no severity tracking)
    # ------------------------------------------------------------------
    new_diagnoses: List[str] = []
    maintained_diagnoses: List[str] = []
    stop_diagnoses: List[str] = []

    for past_term in past_diagnosis_terms:
        matched_current = _find_match(past_term, cur_dx_lower)
        if matched_current is None:
            stop_diagnoses.append(past_term)
        else:
            maintained_diagnoses.append(matched_current)

    for cur_low, cur_orig in cur_dx_lower.items():
        if _find_match(cur_low, {p: p for p in past_diagnosis_terms}) is None:
            new_diagnoses.append(cur_orig)

    # ------------------------------------------------------------------
    # Map entities to their action status and prior meds
    # ------------------------------------------------------------------
    entity_actions: Dict[str, str] = {}
    prior_meds_for_entities: Dict[str, Dict[str, Any]] = {}

    for s in stop_symptoms:
        entity_actions[s] = "stop"
        prior_meds_for_entities[s] = {"status": "stop", "prior_meds": unique_past_meds}
    for s in maintained_symptoms:
        entity_actions[s] = "maintained"
        prior_meds_for_entities[s] = {"status": "maintained", "prior_meds": unique_past_meds}
    for s in decreased_symptoms:
        entity_actions[s] = "decreased"
        prior_meds_for_entities[s] = {"status": "decreased", "prior_meds": unique_past_meds}
    for s in increased_symptoms:
        entity_actions[s] = "increased"
        prior_meds_for_entities[s] = {"status": "increased", "prior_meds": unique_past_meds}
    for s in new_symptoms:
        entity_actions[s] = "new"
        prior_meds_for_entities[s] = {"status": "new", "prior_meds": []}

    for d in stop_diagnoses:
        entity_actions[d] = "stop"
        prior_meds_for_entities[d] = {"status": "stop", "prior_meds": unique_past_meds}
    for d in maintained_diagnoses:
        entity_actions[d] = "maintained"
        prior_meds_for_entities[d] = {"status": "maintained", "prior_meds": unique_past_meds}
    for d in new_diagnoses:
        entity_actions[d] = "new"
        prior_meds_for_entities[d] = {"status": "new", "prior_meds": []}

    # Backward compatibility: prior_meds_for_resolved contains stop + decreased
    prior_meds_for_resolved = {
        ent: data["prior_meds"]
        for ent, data in prior_meds_for_entities.items()
        if data["status"] in ("stop", "decreased")
    }

    # Summary text for prompt context
    parts = []
    if new_symptoms or new_diagnoses:
        parts.append(f"NEW: {', '.join(new_symptoms + new_diagnoses)}")
    if maintained_symptoms or maintained_diagnoses:
        parts.append(f"MAINTAINED: {', '.join(maintained_symptoms + maintained_diagnoses)}")
    if decreased_symptoms:
        parts.append(f"DECREASED: {', '.join(decreased_symptoms)}")
    if increased_symptoms:
        parts.append(f"INCREASED: {', '.join(increased_symptoms)}")
    if stop_symptoms or stop_diagnoses:
        parts.append(f"STOP (RESOLVED): {', '.join(stop_symptoms + stop_diagnoses)}")

    summary_text = " | ".join(parts) if parts else "No significant changes detected."

    return {
        "new_symptoms": new_symptoms,
        "maintained_symptoms": maintained_symptoms,
        "decreased_symptoms": decreased_symptoms,
        "increased_symptoms": increased_symptoms,
        "stop_symptoms": stop_symptoms,
        # Backward compatibility aliases
        "resolved_symptoms": stop_symptoms,
        "persistent_symptoms": maintained_symptoms,
        "improving_symptoms": decreased_symptoms,
        "new_diagnoses": new_diagnoses,
        "maintained_diagnoses": maintained_diagnoses,
        "stop_diagnoses": stop_diagnoses,
        "resolved_diagnoses": stop_diagnoses,
        "persistent_diagnoses": maintained_diagnoses,
        "entity_actions": entity_actions,
        "prior_meds_for_entities": prior_meds_for_entities,
        "prior_meds_for_resolved": prior_meds_for_resolved,
        "summary_text": summary_text,
    }


def _find_match(target: str, candidates: Dict[str, str]) -> Optional[str]:
    """Find a close match for target in candidates (key = lower-cased term)."""
    if target in candidates:
        return candidates[target]
    for key, original in candidates.items():
        if target in key or key in target:
            return original
    return None


# ---------------------------------------------------------------------------
# Human-Readable Clinical Summary Generator
# ---------------------------------------------------------------------------

_RESOLVED_MARKERS = {
    "no", "none", "nil", "denies", "denied", "zero", "resolved", "free", "stopped", "without",
}

_IMPROVED_MARKERS = {
    "improved", "better", "less", "decreased", "mild", "slight", "occasional", "intermittent",
}

_WORSENED_MARKERS = {
    "worse", "worsening", "increased", "severe", "aggravated", "frequent", "constant", "marked", "persistent", "acute",
}

_STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "for", "to", "of", "and", "or", "last", "days",
    "no", "none", "mild", "severe", "occasional", "worse", "better", "improved", "didnt",
    "didn't", "suit", "once", "twice", "consumption",
}


def _get_core_tokens(text: str) -> set[str]:
    import re
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return {w for w in cleaned.split() if len(w) > 2 and w not in _STOP_WORDS}


def _match_symptom_condition(cur_term: str, past_terms: Dict[str, str]) -> tuple[Optional[str], str]:
    """Match current symptom against past visit symptoms and return (matched_term, status_note)."""
    if not past_terms:
        return None, "new symptom"

    cur_low = cur_term.strip().lower()
    cur_tokens = _get_core_tokens(cur_low)

    best_match: Optional[str] = None
    best_overlap: int = 0

    # 1. Exact or substring match
    for past_low, past_orig in past_terms.items():
        if cur_low == past_low or cur_low in past_low or past_low in cur_low:
            best_match = past_orig
            break
        # 2. Core token overlap match
        past_toks = _get_core_tokens(past_low)
        overlap = len(cur_tokens & past_toks)
        if overlap > best_overlap and overlap >= 1:
            best_overlap = overlap
            best_match = past_orig

    if best_match is None:
        return None, "new symptom"

    import re
    cur_words = set(re.findall(r"\b\w+\b", cur_low))
    past_words = set(re.findall(r"\b\w+\b", best_match.strip().lower()))

    # Condition 1: Symptom resolved or negated (e.g. 'no panic attacks')
    if cur_words & _RESOLVED_MARKERS:
        return best_match, f"condition improved, changed from {best_match}"

    # Condition 2: Improved severity or de-escalated from severe
    if (cur_words & _IMPROVED_MARKERS and not (past_words & _IMPROVED_MARKERS)) or (
        (past_words & _WORSENED_MARKERS) and not (cur_words & _WORSENED_MARKERS)
    ):
        return best_match, f"condition improved, changed from {best_match}"

    # Condition 3: Worsened severity
    if (cur_words & _WORSENED_MARKERS and not (past_words & _WORSENED_MARKERS)) or (
        (past_words & _IMPROVED_MARKERS) and not (cur_words & _IMPROVED_MARKERS)
    ):
        return best_match, f"condition worsened, changed from {best_match}"

    return best_match, "no change"



def _format_relative_time(past_date_str: Optional[str], current_date_str: Optional[str]) -> str:
    """Format date difference into human-readable relative time (e.g. '2 weeks ago')."""
    if not past_date_str:
        return ""
    try:
        from datetime import datetime, timezone
        from dateutil import parser
        p_dt = parser.parse(str(past_date_str))
        c_dt = parser.parse(str(current_date_str)) if current_date_str else datetime.now(timezone.utc)
        diff_days = (c_dt.date() - p_dt.date()).days
        if diff_days < 0:
            return str(past_date_str)
        if diff_days == 0:
            return "today"
        if diff_days == 1:
            return "yesterday"
        if diff_days < 7:
            return f"{diff_days} days ago"
        weeks = diff_days // 7
        if weeks < 5:
            return f"{weeks} week{'s' if weeks > 1 else ''} ago"
        months = diff_days // 30
        if months < 12:
            return f"{months} month{'s' if months > 1 else ''} ago"
        years = diff_days // 365
        return f"{years} year{'s' if years > 1 else ''} ago"
    except Exception:
        return str(past_date_str)


def generate_clinical_summary(
    current_visit_type: str,
    patient_info: Any,
    current_symptoms: List[str],
    current_diagnoses: List[str],
    past_cases: List[Dict[str, Any]],
    current_date: Optional[str] = None,
) -> str:
    """Generate human-readable clinical summary text comparing current encounter with past visits."""
    lines: List[str] = []

    # 1. Header: Case type and demographics
    vt_low = (current_visit_type or "").strip().lower()
    if vt_low in ("follow-up", "follow_up", "followup"):
        lines.append("case type: follow up visit")
    else:
        lines.append("case type: new consultation")

    age = None
    gender = None
    if isinstance(patient_info, dict):
        age = patient_info.get("age")
        gender = patient_info.get("gender")
    elif patient_info:
        age = getattr(patient_info, "age", None)
        gender = getattr(patient_info, "gender", None)

    if age is not None:
        age_str = f"{int(age)} yrs" if isinstance(age, (int, float)) and age == int(age) else f"{age} yrs"
        lines.append(f"patient age: {age_str}")
    if gender:
        lines.append(f"patient gender: {str(gender).lower()}")

    lines.append("")

    # 2. Current visit chief complaints
    lines.append("current visit chief complains:")
    past_symptom_dict: Dict[str, str] = {}
    if past_cases:
        for visit in past_cases:
            syms = visit.get("chiefComplaints") or visit.get("symptoms") or []
            for s in syms:
                term = (s.get("term") or s.get("text") or s.get("note") or "") if isinstance(s, dict) else str(s)
                term = term.strip()
                if term:
                    past_symptom_dict[term.lower()] = term

    if current_symptoms:
        for cur_s in current_symptoms:
            cur_term = cur_s.strip()
            if not cur_term:
                continue
            matched, note = _match_symptom_condition(cur_term, past_symptom_dict)
            if note:
                lines.append(f"{cur_term} ({note})")
            else:
                lines.append(cur_term)
    else:
        lines.append("none recorded")

    lines.append("")

    # 3. Current diagnosis
    lines.append("current diagnosis:")
    if current_diagnoses:
        for d in current_diagnoses:
            d_str = (d.get("term") or d.get("text") or "") if isinstance(d, dict) else str(d)
            if d_str.strip():
                lines.append(d_str.strip().lower())
    else:
        lines.append("none recorded")

    # 4. Past cases
    lines.append("")
    lines.append("---")
    lines.append("past cases:")

    if past_cases:
        for idx, visit in enumerate(past_cases):
            raw_vt = (visit.get("visitType") or visit.get("consultationType") or "visit").lower()
            vt_label = "new consultation" if "new" in raw_vt else "follow up visit"
            v_date = visit.get("caseDate") or visit.get("visitDate") or visit.get("date")
            rel_time = _format_relative_time(v_date, current_date)
            header_time = f" - {rel_time}" if rel_time else ""
            lines.append(f"{vt_label}{header_time}")

            lines.append("chief complaints:")
            syms = visit.get("chiefComplaints") or visit.get("symptoms") or []
            if syms:
                for s in syms:
                    t = (s.get("term") or s.get("text") or s.get("note") or "") if isinstance(s, dict) else str(s)
                    if t.strip():
                        lines.append(t.strip().lower())
            else:
                lines.append("none recorded")

            lines.append("")
            lines.append("diagnosis:")
            dxs = visit.get("diagnosis") or []
            if dxs:
                for d in dxs:
                    t = (d.get("term") or d.get("text") or "") if isinstance(d, dict) else str(d)
                    if t.strip():
                        lines.append(t.strip().lower())
            else:
                lines.append("none recorded")

            if idx < len(past_cases) - 1:
                lines.append("")
                lines.append("---")
    else:
        lines.append("no prior cases recorded")

    lines.append("")
    lines.append("---")

    return "\n".join(lines)

