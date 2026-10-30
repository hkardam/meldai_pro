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

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical Keyword Dictionaries for Clinical Change Detection
# (Single authoritative source replacing duplicate heuristics across services)
# ---------------------------------------------------------------------------

_IMPROVEMENT_KEYWORDS: frozenset[str] = frozenset({
    "improved", "improving", "better", "less", "decreased", "mild", "slight",
    "minor", "occasional", "intermittent", "reduced", "subsided", "controlled",
    "minimal", "fading", "easing", "resolving", "settled", "stable",
    "manageable", "partial", "responding",
})

_WORSENING_KEYWORDS: frozenset[str] = frozenset({
    "worse", "worsening", "increased", "severe", "significant", "marked",
    "persistent", "constant", "acute", "aggravated", "frequent", "escalating",
    "deteriorating", "unbearable", "exacerbated", "flaring", "uncontrolled",
    "intractable", "progressive", "refractory", "spreading", "intensified",
})

_RESOLUTION_KEYWORDS: frozenset[str] = frozenset({
    "no", "none", "nil", "denies", "denied", "zero", "resolved", "free",
    "stopped", "without", "absent", "gone", "cleared", "cured", "remission",
    "asymptomatic", "symptom-free",
})

_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "in", "on", "at", "for", "to", "of", "and", "or", "last", "days",
    "no", "none", "mild", "severe", "occasional", "worse", "better", "improved", "didnt",
    "didn't", "suit", "once", "twice", "consumption",
})

# ---------------------------------------------------------------------------
# Noise Filtering & Heuristic Triage
# ---------------------------------------------------------------------------

_NON_SYMPTOM_PATTERNS: List[re.Pattern] = [
    re.compile(r"^informant\b", re.IGNORECASE),
    re.compile(r"^history\s+(?:to\s+be\s+taken|obtained|given)", re.IGNORECASE),
    re.compile(r"^more\s+history", re.IGNORECASE),
    re.compile(r"to\s+be\s+taken\s+from", re.IGNORECASE),
    re.compile(r"^follow\s*-?up\b", re.IGNORECASE),
    re.compile(r"^review\b", re.IGNORECASE),
    re.compile(r"^referred?\s+to\b", re.IGNORECASE),
    re.compile(r"^investigations?\b", re.IGNORECASE),
    re.compile(r"^labs?\b", re.IGNORECASE),
    re.compile(r"^blood\s+test", re.IGNORECASE),
    re.compile(r"^advised\b", re.IGNORECASE),
    re.compile(r"^plan\b", re.IGNORECASE),
    re.compile(r"^next\s+visit\b", re.IGNORECASE),
    re.compile(r"^attendant\b", re.IGNORECASE),
    re.compile(r"^attending\b", re.IGNORECASE),
    re.compile(r"^accompanied\s+by\b", re.IGNORECASE),
    re.compile(r"^brought\s+by\b", re.IGNORECASE),
    re.compile(r"^reliable\b", re.IGNORECASE),
    re.compile(r"^patient\s+is\s+accompanied\b", re.IGNORECASE),
]

_GLOBAL_STATUS_PATTERNS: List[re.Pattern] = [
    re.compile(r"^(?:feels?|feeling)\s+(?:much\s+|far\s+|a\s+lot\s+)?(?:better|well|good|fine|normal|stable|ok|okay)", re.IGNORECASE),
    re.compile(r"^doing\s+(?:much\s+|far\s+|a\s+lot\s+)?(?:better|well|good|fine|ok|okay)", re.IGNORECASE),
    re.compile(r"^(?:no|nil|zero)\s+(?:active\s+)?(?:symptoms?|complaints?|issues?|problems?)(?:\s+reported|\s+noted)?", re.IGNORECASE),
    re.compile(r"^(?:asymptomatic|symptom\s*-?free|completely\s+normal|in\s+remission|doing\s+fine|no\s+complaints?)", re.IGNORECASE),
    re.compile(r"^patient\s+(?:feels?|is\s+doing)\s+(?:much\s+)?better", re.IGNORECASE),
    re.compile(r"^overall\s+(?:better|improved|stable|good|fine)", re.IGNORECASE),
]

_NEGATED_SYMPTOM_PATTERNS: List[re.Pattern] = [
    re.compile(r"^(?:denies|denied|denying)\s+(.+)", re.IGNORECASE),
    re.compile(r"^(?:no|nil)\s+(.+)", re.IGNORECASE),
    re.compile(r"^without\s+(.+)", re.IGNORECASE),
    re.compile(r"^free\s+of\s+(.+)", re.IGNORECASE),
    re.compile(r"^zero\s+(.+)", re.IGNORECASE),
    re.compile(r"^absence\s+of\s+(.+)", re.IGNORECASE),
    re.compile(r"^(.+?)\s+(?:resolved|stopped|gone|cleared|cured|absent)$", re.IGNORECASE),
    re.compile(r"^(.+?)\s+(?:is\s+)?(?:denied|ruled\s+out)$", re.IGNORECASE),
]


def extract_canonical_from_negation(neg_text: str) -> Optional[str]:
    """Extract the underlying canonical symptom entity from a negated phrase.
    e.g. 'denies paranoia currently' -> 'paranoia'
    """
    cleaned = neg_text.strip()
    for pat in _NEGATED_SYMPTOM_PATTERNS:
        m = pat.match(cleaned)
        if m:
            extracted = m.group(1).strip()
            # Clean trailing time/status markers
            extracted = re.sub(
                r"\b(?:currently|now|today|reported|noted|anymore|at\s+present|any\s+more|since\s+\w+)\b",
                "",
                extracted,
                flags=re.IGNORECASE,
            ).strip()
            extracted = re.sub(r"[^\w\s-]", "", extracted).strip()
            if extracted and len(extracted) > 1:
                return extracted.lower()
    return None


def heuristic_triage_complaint(text: str) -> Tuple[str, Optional[str]]:
    """Classify a single chief complaint into (category, canonical_entity) via heuristics.
    Returns:
      category: 'non_symptom' | 'global_status' | 'negated_symptom' | 'symptom'
      canonical_entity: cleaned entity name or None
    """
    clean = text.strip()
    if not clean:
        return "non_symptom", None

    for pat in _NON_SYMPTOM_PATTERNS:
        if pat.search(clean):
            return "non_symptom", None

    for pat in _GLOBAL_STATUS_PATTERNS:
        if pat.search(clean):
            return "global_status", None

    canonical = extract_canonical_from_negation(clean)
    if canonical:
        return "negated_symptom", canonical

    if clean.lower() in ("nil", "none", "no", "zero", "na", "n/a"):
        return "global_status", None

    return "symptom", clean


def _get_core_tokens(text: str) -> set[str]:
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return {w for w in cleaned.split() if len(w) > 2 and w not in _STOP_WORDS}


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


def _find_match(target: str, candidates: Dict[str, str]) -> Optional[str]:
    """Find a close match for target in candidates (key = lower-cased term)."""
    if target in candidates:
        return candidates[target]
    for key, original in candidates.items():
        if target in key or key in target:
            return original
    return None


def classify_entity_change(term: str, past_terms: Iterable[str]) -> str:
    """Classify a single entity string against previously seen past terms.

    Uses dense keyword matching to determine:
      - 'stop'       — term absent from past AND current term signals resolution/negation
      - 'new'        — term was not present in any past visit
      - 'decreased'  — improvement/mild language detected in current term
      - 'increased'  — worsening language detected in current term
      - 'maintained' — term present before, no directional signal
    """
    words = set(re.findall(r"\b\w+\b", term.strip().lower()))

    term_low = term.strip().lower()
    past_lower = {p.strip().lower() for p in past_terms}

    def _matches_past(query: str) -> bool:
        if query in past_lower:
            return True
        for p in past_lower:
            if query in p or p in query:
                return True
        q_toks = _get_core_tokens(query)
        for p in past_lower:
            p_toks = _get_core_tokens(p)
            if q_toks and p_toks and (q_toks == p_toks):
                return True
            if q_toks and p_toks and (q_toks & p_toks):
                return True
        return False

    seen_before = _matches_past(term_low)

    if not seen_before:
        if words & _RESOLUTION_KEYWORDS:
            return "stop"
        return "new"

    if words & _RESOLUTION_KEYWORDS:
        return "stop"
    if words & _IMPROVEMENT_KEYWORDS:
        return "decreased"
    if words & _WORSENING_KEYWORDS:
        return "increased"
    return "maintained"


def _match_symptom_condition(cur_term: str, past_terms: Dict[str, str]) -> Tuple[Optional[str], str]:
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

    cur_words = set(re.findall(r"\b\w+\b", cur_low))
    past_words = set(re.findall(r"\b\w+\b", best_match.strip().lower()))

    # Condition 1: Symptom resolved or negated (e.g. 'no panic attacks')
    if cur_words & _RESOLUTION_KEYWORDS:
        return best_match, f"condition resolved/stopped, changed from {best_match}"

    # Condition 2: Improved severity or de-escalated from severe
    if (cur_words & _IMPROVEMENT_KEYWORDS and not (past_words & _IMPROVEMENT_KEYWORDS)) or (
        (past_words & _WORSENING_KEYWORDS) and not (cur_words & _WORSENING_KEYWORDS)
    ):
        return best_match, f"condition improved, changed from {best_match}"

    # Condition 3: Worsened severity
    if (cur_words & _WORSENING_KEYWORDS and not (past_words & _WORSENING_KEYWORDS)) or (
        (past_words & _IMPROVEMENT_KEYWORDS) and not (cur_words & _IMPROVEMENT_KEYWORDS)
    ):
        return best_match, f"condition worsened, changed from {best_match}"

    return best_match, "no change"


def _format_relative_time(past_date_str: Optional[str], current_date_str: Optional[str]) -> str:
    """Format date difference into human-readable relative time (e.g. '2 weeks ago')."""
    if not past_date_str:
        return ""
    try:
        from core.utils import parse_clinical_date
        p_dt = parse_clinical_date(past_date_str)
        c_dt = parse_clinical_date(current_date_str) or datetime.now(timezone.utc)
        if not p_dt:
            return str(past_date_str)
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


# ---------------------------------------------------------------------------
# Data Models for Clinical Change Assessment
# ---------------------------------------------------------------------------

@dataclass
class SymptomChangeResult:
    """Detailed change classification for an individual symptom / chief complaint."""
    complaint: str
    keyword_status: str       # Initial heuristic marker (e.g. 'decreased', 'increased', 'maintained', 'new', 'stop')
    confirmed_status: str     # LLM confirmed canonical status ('decreased', 'increased', 'maintained', 'new', 'stop')
    display_status: str       # Human-readable display label ('improved', 'worsened', 'maintained', 'new', 'resolved')
    change_description: str   # Change from past history to current encounter
    notes: str                # Specific clinical observation notes


@dataclass
class TriagedComplaintItem:
    """A single intake complaint item triaged into symptom/noise/status category with delta diff."""
    raw_text: str
    category: str                             # 'symptom' | 'negated_symptom' | 'global_status' | 'non_symptom'
    canonical_entity: Optional[str] = None  # Canonical symptom term if applicable (e.g. 'paranoia')
    status: Optional[str] = "new"           # 'decreased' | 'increased' | 'maintained' | 'stop' | 'new' | None
    display_status: Optional[str] = "new"   # 'improved' | 'worsened' | 'maintained' | 'resolved' | 'new' | None
    change_description: str = ""
    notes: str = ""
    heuristic_mark: Optional[str] = None     # Heuristic keyword mark ('improved', 'worsened', etc.)
    heuristic_category: Optional[str] = None # Heuristic pre-category ('symptom', 'non_symptom', etc.)
    changed: bool = False                    # True if LLM classification differed from heuristic

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict representation for API serialization and delta reporting."""
        return {
            "complaint": self.raw_text,
            "category": self.category,
            "canonical_entity": self.canonical_entity,
            "status": self.status,
            "display_status": self.display_status,
            "llm_status": self.display_status,
            "llm_category": self.category,
            "heuristic_mark": self.heuristic_mark,
            "heuristic_category": self.heuristic_category,
            "changed": self.changed,
            "change_description": self.change_description,
            "notes": self.notes,
        }


@dataclass
class ClinicalChangeAssessment:
    """Overall assessment of clinical changes and patient condition across encounters."""
    symptoms: List[SymptomChangeResult] = field(default_factory=list)
    triaged_items: List[TriagedComplaintItem] = field(default_factory=list)
    active_symptoms: List[str] = field(default_factory=list)
    resolved_symptoms: List[str] = field(default_factory=list)
    global_status_notes: List[str] = field(default_factory=list)
    patient_condition: str = ""
    symptom_notes: str = ""
    raw_llm_response: Optional[str] = None
    used_llm: bool = False
    llm_prompt: Optional[str] = None
    triage_delta: Optional[List[Dict[str, Any]]] = None
    """Per-symptom delta: what the triage LLM classified vs. heuristic pre-mark, and the rationale."""
    deprescribing_recommendations: Optional[List[Dict[str, Any]]] = None
    """Populated by assess_and_deprescribe() — raw deprescribing recommendation dicts from unified LLM call."""
    whats_changed: Optional[str] = None
    """Populated by assess_and_deprescribe() — narrative summary of what changed clinically."""
    historical_prescription: List[str] = field(default_factory=list)
    latest_documented_prescription: List[str] = field(default_factory=list)
    patient_reported_current_use: List[str] = field(default_factory=list)
    actual_discontinuation: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Medication Extraction Helpers (Distinguish Prescriptions from Current Use)
# ---------------------------------------------------------------------------

def _extract_patient_reported_meds(current_symptoms: List[str]) -> List[str]:
    """Heuristically extract patient-reported currently used medications from intake strings."""
    meds: List[str] = []
    use_pattern = re.compile(
        r"(?:currently\s+)?(?:taking|takes|on|uses|using|compliant\s+with|continuing|continued)\s+"
        r"(?:tab(?:let)?\.?\s+|cap(?:sule)?\.?\s+)?([A-Za-z0-9\-\.\+\/]+(?:\s+[A-Za-z0-9\-\.\+\/]+){0,3})",
        re.IGNORECASE,
    )
    for sym in current_symptoms:
        if not isinstance(sym, str):
            continue
        s_clean = sym.strip()
        if re.search(r"\b(?:not|stopped|discontinued|stopped\s+taking|never)\b", s_clean, re.IGNORECASE):
            continue
        m = use_pattern.search(s_clean)
        if m:
            med_text = m.group(1).strip()
            med_text = re.sub(r"[,;:\.\?]+$", "", med_text).strip()
            if med_text and med_text.lower() not in ("medication", "medications", "meds", "pills", "drugs"):
                meds.append(med_text)
    return list(dict.fromkeys(meds))


def _extract_discontinued_meds(
    current_symptoms: List[str],
    change_assessment: Optional[Any] = None,
    historical_prescription: Optional[List[str]] = None,
    latest_documented_prescription: Optional[List[str]] = None,
) -> List[str]:
    """Extract medications confirmed stopped or discontinued by patient, doctor, or assessment."""
    discontinued: List[str] = []
    if change_assessment:
        recs = getattr(change_assessment, "deprescribing_recommendations", None) or []
        for r in recs:
            if isinstance(r, dict):
                act = str(r.get("action") or "").lower()
                status = str(r.get("status_change") or r.get("status") or "").lower()
                med = str(r.get("medicine") or r.get("drug") or "").strip()
                if (act == "stop" or status in ("resolved", "stop")) and med:
                    discontinued.append(med)
            elif hasattr(r, "action"):
                act = str(r.action or "").lower()
                status = str(getattr(r, "status_change", None) or "").lower()
                med = str(getattr(r, "medicine", "")).strip()
                if (act == "stop" or status in ("resolved", "stop")) and med:
                    discontinued.append(med)

        if getattr(change_assessment, "actual_discontinuation", None):
            discontinued.extend(change_assessment.actual_discontinuation)

    disc_pattern = re.compile(
        r"(?:stopped|discontinued|off|omitted|withdrawn|held|not\s+taking|stopped\s+taking)\s+"
        r"(?:tab(?:let)?\.?\s+|cap(?:sule)?\.?\s+)?([A-Za-z0-9\-\.\+\/]+(?:\s+[A-Za-z0-9\-\.\+\/]+){0,3})",
        re.IGNORECASE,
    )
    for sym in current_symptoms:
        if not isinstance(sym, str):
            continue
        m = disc_pattern.search(sym.strip())
        if m:
            med_text = m.group(1).strip()
            med_text = re.sub(r"[,;:\.\?]+$", "", med_text).strip()
            if med_text and med_text.lower() not in ("medication", "medications", "meds", "pills", "drugs"):
                discontinued.append(med_text)

    return list(dict.fromkeys(discontinued))


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

def compute_delta(
    current_symptoms: List[str],
    current_diagnoses: List[str],
    past_cases: List[Dict[str, Any]],
    change_assessment: Optional[ClinicalChangeAssessment] = None,
) -> Dict[str, Any]:
    """Compare the current encounter against all past patient cases.

    Returns a dict with:
      - new_symptoms: symptoms appearing in current visit that were NOT in past visits
      - maintained_symptoms: symptoms present in past visits and current visit with unchanged severity
      - decreased_symptoms: symptoms present before, currently improved/milder
      - increased_symptoms: symptoms present before, currently worsening/more severe
      - stop_symptoms: symptoms present in past visits that are ABSENT or negated in current visit
      - new_diagnoses, maintained_diagnoses, stop_diagnoses: same for diagnoses
      - prior_medications: distinct medications prescribed across prior visits
      - clinical_notes: unified synthesis narrative on the clinical trajectory
    """
    if not past_cases:
        cur_syms: List[str] = []
        if change_assessment and change_assessment.active_symptoms is not None:
            cur_syms = [s.strip() for s in change_assessment.active_symptoms if s.strip()]
        else:
            for s in current_symptoms:
                cat, canon = heuristic_triage_complaint(s)
                if cat == "symptom":
                    cur_syms.append(canon or s.strip())

        from core.utils import clean_diagnosis_term
        cur_dxs = [clean_diagnosis_term(d) for d in current_diagnoses if clean_diagnosis_term(d)]
        notes = ""
        if change_assessment:
            notes = change_assessment.symptom_notes or change_assessment.patient_condition

        patient_reported_use = (
            change_assessment.patient_reported_current_use
            if change_assessment and change_assessment.patient_reported_current_use
            else _extract_patient_reported_meds(current_symptoms)
        )
        actual_disc = (
            change_assessment.actual_discontinuation
            if change_assessment and change_assessment.actual_discontinuation
            else _extract_discontinued_meds(current_symptoms=current_symptoms, change_assessment=change_assessment)
        )
        return {
            "new_symptoms": cur_syms,
            "maintained_symptoms": [],
            "decreased_symptoms": [],
            "increased_symptoms": [],
            "stop_symptoms": [],
            "new_diagnoses": cur_dxs,
            "maintained_diagnoses": [],
            "stop_diagnoses": [],
            "prior_medications": [],
            "historical_prescription": [],
            "latest_documented_prescription": [],
            "patient_reported_current_use": patient_reported_use,
            "actual_discontinuation": actual_disc,
            "clinical_notes": notes or ("New consultation. No prior history." if (cur_syms or cur_dxs) else ""),
        }

    all_past_symptoms: List[str] = []
    all_past_diagnoses: List[str] = []
    all_past_meds: List[str] = []

    for visit in past_cases:
        syms = visit.get("chiefComplaints") or visit.get("symptoms") or []
        for s in syms:
            t = (s.get("term") or s.get("text") or s.get("note") or "") if isinstance(s, dict) else str(s)
            t = t.strip().lower()
            if t:
                all_past_symptoms.append(t)

        dxs = visit.get("diagnosis") or []
        for d in dxs:
            t = (d.get("term") or d.get("text") or "") if isinstance(d, dict) else str(d)
            from core.utils import clean_diagnosis_term
            t = clean_diagnosis_term(t).lower()
            if t:
                all_past_diagnoses.append(t)

        all_past_meds.extend(_extract_meds(visit))

    past_symptom_terms = list(dict.fromkeys(all_past_symptoms))
    past_diagnosis_terms = list(dict.fromkeys(all_past_diagnoses))
    unique_past_meds = list(dict.fromkeys(all_past_meds))

    # ------------------------------------------------------------------
    # Classify symptoms: stop | maintained | decreased | increased | new
    # ------------------------------------------------------------------
    new_symptoms: List[str] = []
    maintained_symptoms: List[str] = []
    decreased_symptoms: List[str] = []
    increased_symptoms: List[str] = []
    stop_symptoms: List[str] = []

    if change_assessment and change_assessment.triaged_items:
        matched_past_terms: Set[str] = set()

        for item in change_assessment.triaged_items:
            # 1. Non-symptoms and global statuses are NEVER entities in delta
            if item.category in ("non_symptom", "global_status"):
                continue

            # 2. Negated symptoms map canonical entity to stop_symptoms
            if item.category == "negated_symptom":
                canonical = (item.canonical_entity or extract_canonical_from_negation(item.raw_text) or item.raw_text).strip()
                p_match = _find_match(canonical.lower(), {p: p for p in past_symptom_terms})
                if not p_match:
                    c_toks = _get_core_tokens(canonical.lower())
                    for p in past_symptom_terms:
                        if c_toks & _get_core_tokens(p):
                            p_match = p
                            break
                resolved_term = p_match or canonical
                stop_symptoms.append(resolved_term)
                if p_match:
                    matched_past_terms.add(p_match)
                continue

            # 3. Genuine active symptoms
            c_name = (item.canonical_entity or item.raw_text).strip()
            st = item.status or "new"
            if st == "decreased":
                decreased_symptoms.append(c_name)
            elif st == "increased":
                increased_symptoms.append(c_name)
            elif st == "maintained":
                maintained_symptoms.append(c_name)
            elif st == "stop":
                stop_symptoms.append(c_name)
            else:
                new_symptoms.append(c_name)

            if st != "new":
                p_match = _find_match(c_name.lower(), {p: p for p in past_symptom_terms})
                if not p_match:
                    c_toks = _get_core_tokens(c_name.lower())
                    for p in past_symptom_terms:
                        if c_toks & _get_core_tokens(p):
                            p_match = p
                            break
                if p_match:
                    matched_past_terms.add(p_match)

        # NOTE: Past symptoms not mentioned in the current visit are NOT automatically resolved.
        # Absence of mention ≠ resolved — chronic/psychiatric conditions persist until explicitly
        # denied or the LLM (in the deprescribe step) determines otherwise from full clinical history.

    elif change_assessment and change_assessment.symptoms:
        # Legacy fallback if triaged_items wasn't populated
        matched_past_terms: Set[str] = set()
        for s_res in change_assessment.symptoms:
            c_name = s_res.complaint
            st = s_res.confirmed_status
            if st == "decreased":
                decreased_symptoms.append(c_name)
            elif st == "increased":
                increased_symptoms.append(c_name)
            elif st == "maintained":
                maintained_symptoms.append(c_name)
            elif st == "stop":
                stop_symptoms.append(c_name)
            else:
                new_symptoms.append(c_name)

            if st != "new":
                p_match = _find_match(c_name.lower(), {p: p for p in past_symptom_terms})
                if not p_match:
                    c_toks = _get_core_tokens(c_name.lower())
                    for p in past_symptom_terms:
                        if c_toks & _get_core_tokens(p):
                            p_match = p
                            break
                if p_match:
                    matched_past_terms.add(p_match)

        # NOTE: Past symptoms not mentioned in the current visit are NOT automatically resolved.
        # Absence of mention ≠ resolved — chronic/psychiatric conditions persist until explicitly
        # denied or the LLM (in the deprescribe step) determines otherwise from full clinical history.
    else:
        # Fallback without change_assessment: apply heuristic triage to current_symptoms
        active_cur_syms: Dict[str, str] = {}
        for s in current_symptoms:
            cat, canon = heuristic_triage_complaint(s)
            if cat == "symptom":
                clean_term = canon or s.strip()
                active_cur_syms[clean_term.lower()] = clean_term
            elif cat == "negated_symptom":
                canonical = canon or s.strip()
                p_match = _find_match(canonical.lower(), {p: p for p in past_symptom_terms})
                stop_symptoms.append(p_match or canonical)

        for past_term in past_symptom_terms:
            matched_current = _find_match(past_term, active_cur_syms)
            if matched_current is not None:
                # Only classify trajectory for symptoms that ARE present in current visit
                status = classify_entity_change(matched_current, {past_term})
                if status == "increased":
                    increased_symptoms.append(matched_current)
                elif status == "decreased":
                    decreased_symptoms.append(matched_current)
                elif status == "stop":
                    stop_symptoms.append(matched_current)
                else:
                    maintained_symptoms.append(matched_current)
            # NOTE: If past_term is NOT in current visit, do NOT auto-classify as stop.
            # Absence of mention ≠ resolved. The deprescribe LLM sees full history and decides.

        for cur_low, cur_orig in active_cur_syms.items():
            if _find_match(cur_low, {p: p for p in past_symptom_terms}) is None:
                new_symptoms.append(cur_orig)

    # Deduplicate lists while preserving order
    new_symptoms = list(dict.fromkeys(new_symptoms))
    maintained_symptoms = list(dict.fromkeys(maintained_symptoms))
    decreased_symptoms = list(dict.fromkeys(decreased_symptoms))
    increased_symptoms = list(dict.fromkeys(increased_symptoms))
    stop_symptoms = list(dict.fromkeys(stop_symptoms))

    # ------------------------------------------------------------------
    # Classify diagnoses: stop | maintained | new (no severity tracking)
    # ------------------------------------------------------------------
    from core.utils import clean_diagnosis_term
    cur_dx_lower = {clean_diagnosis_term(d).lower(): clean_diagnosis_term(d) for d in current_diagnoses if clean_diagnosis_term(d)}
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

    new_diagnoses = list(dict.fromkeys(new_diagnoses))
    maintained_diagnoses = list(dict.fromkeys(maintained_diagnoses))
    stop_diagnoses = list(dict.fromkeys(stop_diagnoses))

    # ------------------------------------------------------------------
    # Canonical non-redundant delta summary
    # ------------------------------------------------------------------
    notes = ""
    if change_assessment:
        notes = change_assessment.symptom_notes or change_assessment.patient_condition

    latest_doc_meds: List[str] = list(dict.fromkeys(_extract_meds(past_cases[-1]))) if past_cases else []
    historical_prescription = (
        change_assessment.historical_prescription
        if change_assessment and change_assessment.historical_prescription
        else unique_past_meds
    )
    latest_documented_prescription = (
        change_assessment.latest_documented_prescription
        if change_assessment and change_assessment.latest_documented_prescription
        else latest_doc_meds
    )
    patient_reported_current_use = (
        change_assessment.patient_reported_current_use
        if change_assessment and change_assessment.patient_reported_current_use
        else _extract_patient_reported_meds(current_symptoms)
    )
    actual_discontinuation = (
        change_assessment.actual_discontinuation
        if change_assessment and change_assessment.actual_discontinuation
        else _extract_discontinued_meds(
            current_symptoms=current_symptoms,
            change_assessment=change_assessment,
            historical_prescription=historical_prescription,
            latest_documented_prescription=latest_documented_prescription,
        )
    )

    return {
        "new_symptoms": new_symptoms,
        "maintained_symptoms": maintained_symptoms,
        "decreased_symptoms": decreased_symptoms,
        "increased_symptoms": increased_symptoms,
        "stop_symptoms": stop_symptoms,
        "new_diagnoses": new_diagnoses,
        "maintained_diagnoses": maintained_diagnoses,
        "stop_diagnoses": stop_diagnoses,
        "prior_medications": unique_past_meds,
        "historical_prescription": historical_prescription,
        "latest_documented_prescription": latest_documented_prescription,
        "patient_reported_current_use": patient_reported_current_use,
        "actual_discontinuation": actual_discontinuation,
        "clinical_notes": notes,
    }


# ---------------------------------------------------------------------------
# Human-Readable Clinical Summary Generator
# ---------------------------------------------------------------------------

def generate_clinical_summary(
    current_visit_type: str,
    patient_info: Any,
    current_symptoms: List[str],
    current_diagnoses: List[str],
    past_cases: Optional[List[Dict[str, Any]]] = None,
    current_date: Optional[str] = None,
    change_assessment: Optional[ClinicalChangeAssessment] = None,
) -> str:
    """Generate concise human-readable clinical summary text for the current consultation."""
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

    # Capture informant metadata in demographics header if present
    informant_text = None
    if change_assessment and change_assessment.triaged_items:
        for it in change_assessment.triaged_items:
            if it.category == "non_symptom" and "informant" in it.raw_text.lower():
                informant_text = it.raw_text.strip()
                break
    if not informant_text:
        for s in current_symptoms:
            if isinstance(s, str) and re.match(r"^informant\b", s.strip(), re.IGNORECASE):
                informant_text = s.strip()
                break
    if informant_text:
        lines.append(informant_text.lower())

    lines.append("")

    # 2. Current visit chief complaints (filtered: only genuine active symptoms)
    lines.append("current visit chief complains:")
    displayed_complaints: List[str] = []

    if change_assessment and change_assessment.triaged_items:
        for item in change_assessment.triaged_items:
            if item.category == "symptom":
                tag = f" ({item.display_status})" if item.display_status else ""
                displayed_complaints.append(f"- {item.raw_text}{tag}")
            elif item.category == "negated_symptom":
                canonical = item.canonical_entity or item.raw_text
                displayed_complaints.append(f"- {canonical} (resolved / denied)")
    elif change_assessment and change_assessment.active_symptoms:
        for s in change_assessment.active_symptoms:
            displayed_complaints.append(f"- {s}")
    else:
        for s in current_symptoms:
            cat, canon = heuristic_triage_complaint(s)
            if cat == "symptom":
                displayed_complaints.append(f"- {s.strip()}")
            elif cat == "negated_symptom":
                canonical = canon or s.strip()
                displayed_complaints.append(f"- {canonical} (resolved / denied)")

    if displayed_complaints:
        lines.extend(displayed_complaints)
    else:
        lines.append("- no active symptoms reported")

    lines.append("")

    # 3. Current diagnosis
    lines.append("current diagnosis:")
    if current_diagnoses:
        from core.utils import clean_diagnosis_term
        for d in current_diagnoses:
            d_str = (d.get("term") or d.get("text") or "") if isinstance(d, dict) else str(d)
            cleaned_d = clean_diagnosis_term(d_str)
            if cleaned_d:
                lines.append(f"- {cleaned_d.lower()}")
    else:
        lines.append("none recorded")

    # 4. Clinical notes (single concise narrative)
    notes = ""
    if change_assessment:
        notes = change_assessment.symptom_notes or change_assessment.patient_condition
    if notes:
        lines.append("")
        lines.append("clinical notes:")
        lines.append(notes)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chief Complaint Change Service (SRP: flow from keywords to LLM confirmation)
# ---------------------------------------------------------------------------

class ChiefComplaintChangeService:
    """Service for classifying changes in chief complaints between patient encounters.

    Workflow:
      1. Start with current encounter chief complaints.
      2. Attach past history (chronological prior visits for follow-up).
      3. Mark current chief complaints with dense keyword search against past encounters.
      4. Pass structured prompt to LLM (confirm current markers, detect subtleties,
         classify directional change from past history to current).
      5. Parse and store changes in complaints, patient condition, and symptom notes.
      6. Graceful fallback to keyword markers when LLM is unavailable or unconfigured.
    """

    def __init__(
        self,
        gemini_service: Optional[Any] = None,
        settings: Optional[Any] = None,
    ) -> None:
        self._gemini = gemini_service
        self._settings = settings

    def _get_gemini(self) -> Optional[Any]:
        if self._gemini is not None:
            return self._gemini
        try:
            from core.services.gemini_service import get_gemini_service
            self._gemini = get_gemini_service(self._settings)
            return self._gemini
        except Exception as exc:
            logger.warning("ChiefComplaintChangeService: Could not initialize GeminiService: %s", exc)
            return None

    def assess_symptom_changes(
        self,
        *,
        current_symptoms: List[str],
        past_cases: List[Dict[str, Any]],
        patient_info: Any = None,
        visit_type: Optional[str] = None,
        current_date: Optional[str] = None,
    ) -> ClinicalChangeAssessment:
        """Classify changes in current chief complaints compared to prior visits (delegates to unified assess_and_deprescribe)."""
        return self.assess_and_deprescribe(
            current_symptoms=current_symptoms,
            past_cases=past_cases,
            prior_medications=[],
            patient_info=patient_info,
            visit_type=visit_type,
            current_date=current_date,
        )

    def assess_and_deprescribe(
        self,
        *,
        current_symptoms: List[str],
        past_cases: List[Dict[str, Any]],
        prior_medications: List[str],
        patient_info: Any = None,
        visit_type: Optional[str] = None,
        current_date: Optional[str] = None,
    ) -> ClinicalChangeAssessment:
        """Single unified LLM call that performs triage + deprescribing in one pass.

        Replaces the previous two-step pattern of:
          1. assess_symptom_changes() -> triage LLM call
          2. build_deprescribe_prompt() -> deprescribe LLM call

        The unified prompt (triage_deprescribe_prompt.md) returns both
        ``triaged_items`` and ``deprescribing_recommendations`` in one JSON response.
        The result is stored in a ClinicalChangeAssessment so the rest of the pipeline
        (SapBERT embedding, compute_delta, generate_clinical_summary) works unchanged.
        """
        if not current_symptoms:
            return ClinicalChangeAssessment(
                patient_condition="No chief complaints recorded for current visit.",
                symptom_notes="No symptoms to assess.",
                used_llm=False,
                deprescribing_recommendations=[],
                whats_changed="No symptoms recorded at this visit.",
            )

        # --- Build keyword pre-marks (same as assess_symptom_changes) ---
        past_symptom_dict: Dict[str, str] = {}
        for visit in past_cases:
            syms = visit.get("chiefComplaints") or visit.get("symptoms") or []
            for s in syms:
                term = (s.get("term") or s.get("text") or s.get("note") or "") if isinstance(s, dict) else str(s)
                t_str = term.strip()
                if t_str:
                    past_symptom_dict[t_str.lower()] = t_str

        keyword_results: List[Dict[str, Any]] = []
        for sym in current_symptoms:
            s_clean = sym.strip()
            if not s_clean:
                continue
            category, canonical = heuristic_triage_complaint(s_clean)
            if category == "non_symptom":
                keyword_results.append({
                    "complaint": s_clean, "category": "non_symptom",
                    "canonical_entity": None, "matched_past": "None",
                    "condition_note": "Administrative metadata or clinical directive (non-symptom)",
                    "kw_status": None, "display_kw": "non_symptom",
                })
            elif category == "global_status":
                keyword_results.append({
                    "complaint": s_clean, "category": "global_status",
                    "canonical_entity": None, "matched_past": "None",
                    "condition_note": "Whole-patient condition statement (global status)",
                    "kw_status": None, "display_kw": "global_status",
                })
            elif category == "negated_symptom":
                canon = canonical or s_clean
                matched_past, _ = _match_symptom_condition(canon, past_symptom_dict)
                keyword_results.append({
                    "complaint": s_clean, "category": "negated_symptom",
                    "canonical_entity": canon, "matched_past": matched_past or "None",
                    "condition_note": f"Negated/denied symptom, canonical entity '{canon}'",
                    "kw_status": "stop", "display_kw": "resolved",
                })
            else:
                matched_past, condition_note = _match_symptom_condition(s_clean, past_symptom_dict)
                kw_status = classify_entity_change(s_clean, set(past_symptom_dict.values()))
                display_kw = {
                    "decreased": "improved", "increased": "worsened",
                    "stop": "resolved", "new": "new",
                }.get(kw_status, "maintained")
                keyword_results.append({
                    "complaint": s_clean, "category": "symptom",
                    "canonical_entity": canonical or s_clean, "matched_past": matched_past or "None",
                    "condition_note": condition_note, "kw_status": kw_status, "display_kw": display_kw,
                })

        # --- Attempt unified LLM call ---
        gemini = self._get_gemini()
        can_call_llm = gemini is not None and getattr(gemini, "is_configured", False)

        if can_call_llm:
            try:
                # Import here to avoid circular import; the function lives in prescription_service
                from meldai_v2.services.prescription_service import build_unified_triage_deprescribe_prompt
                unified_prompt = build_unified_triage_deprescribe_prompt(
                    keyword_results=keyword_results,
                    past_cases=past_cases,
                    prior_medications=prior_medications,
                    patient_info=patient_info,
                    visit_type=visit_type,
                    current_date=current_date,
                )
                raw_text, _metrics = gemini.generate_content_with_metadata(
                    unified_prompt,
                    temperature=0.0,
                    call_name="unified_triage_deprescribe",
                )
                if raw_text and raw_text.strip():
                    assessment = self._parse_unified_response(
                        raw_text=raw_text,
                        keyword_results=keyword_results,
                        unified_prompt=unified_prompt,
                    )
                    if assessment is not None:
                        return assessment
            except Exception as exc:
                logger.warning(
                    "ChiefComplaintChangeService unified LLM call failed, falling back to two-step: %s", exc
                )

        # --- Fallback: keyword heuristic only (no deprescribing recs available) ---
        fallback = self._build_keyword_fallback(keyword_results, past_cases)
        fallback.deprescribing_recommendations = []
        fallback.whats_changed = ""
        return fallback

    def _parse_unified_response(
        self,
        raw_text: str,
        keyword_results: List[Dict[str, Any]],
        unified_prompt: str,
    ) -> Optional[ClinicalChangeAssessment]:
        """Parse the single JSON response from the unified triage+deprescribe LLM call.

        The response contains both ``triaged_items`` and ``deprescribing_recommendations``.
        The triaged_items section is parsed identically to _parse_llm_response (reusing same
        norm_map logic). The deprescribing_recommendations section is stored as raw dicts on
        the ClinicalChangeAssessment so deprescribe_step can use them directly.
        """
        # Reuse the existing triage-only parser to build the ClinicalChangeAssessment
        assessment = self._parse_llm_response(raw_text, keyword_results)
        if assessment is None:
            return None

        # Additionally extract deprescribing_recommendations and whats_changed
        text = raw_text.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        cand = match.group(1).strip() if match else text
        cand = cand.strip().lstrip("```").lstrip("json").strip().rstrip("`").strip()
        data: Dict[str, Any] = {}
        try:
            data = json.loads(cand)
        except Exception:
            m = re.search(r"(\{[\s\S]*\})", cand)
            if m:
                try:
                    data = json.loads(m.group(1))
                except Exception:
                    pass

        assessment.deprescribing_recommendations = data.get("deprescribing_recommendations") or []
        assessment.whats_changed = str(data.get("whats_changed") or "").strip()
        assessment.llm_prompt = unified_prompt

        # Extract medication categorization from unified triage+deprescribe LLM response
        if "historical_prescription" in data and isinstance(data["historical_prescription"], list):
            assessment.historical_prescription = [str(m).strip() for m in data["historical_prescription"] if str(m).strip()]
        if "latest_documented_prescription" in data and isinstance(data["latest_documented_prescription"], list):
            assessment.latest_documented_prescription = [str(m).strip() for m in data["latest_documented_prescription"] if str(m).strip()]
        if "patient_reported_current_use" in data and isinstance(data["patient_reported_current_use"], list):
            assessment.patient_reported_current_use = [str(m).strip() for m in data["patient_reported_current_use"] if str(m).strip()]
        if "actual_discontinuation" in data and isinstance(data["actual_discontinuation"], list):
            assessment.actual_discontinuation = [str(m).strip() for m in data["actual_discontinuation"] if str(m).strip()]

        return assessment



    def _parse_llm_response(
        self,
        raw_text: str,
        keyword_results: List[Dict[str, Any]],
    ) -> Optional[ClinicalChangeAssessment]:
        text = raw_text.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        cand = match.group(1).strip() if match else text
        if cand.startswith("```"):
            cand = cand[3:]
        if cand.startswith("json"):
            cand = cand[4:]
        cand = cand.strip().rstrip("`").strip()

        data: Dict[str, Any] = {}
        try:
            data = json.loads(cand)
        except Exception:
            m = re.search(r"(\{[\s\S]*\})", cand)
            if m:
                try:
                    data = json.loads(m.group(1))
                except Exception:
                    pass

        if not data:
            return None

        triaged_input = data.get("triaged_items") or data.get("symptoms") or []
        parsed_by_name: Dict[str, Dict[str, Any]] = {}
        for item in triaged_input:
            if isinstance(item, dict):
                key = (item.get("raw_text") or item.get("complaint") or "").strip().lower()
                if key:
                    parsed_by_name[key] = item

        norm_map = {
            "improved": ("decreased", "improved"),
            "improving": ("decreased", "improved"),
            "decreased": ("decreased", "improved"),
            "better": ("decreased", "improved"),
            "mild": ("decreased", "improved"),
            "worsened": ("increased", "worsened"),
            "worsening": ("increased", "worsened"),
            "increased": ("increased", "worsened"),
            "worse": ("increased", "worsened"),
            "severe": ("increased", "worsened"),
            "maintained": ("maintained", "maintained"),
            "unchanged": ("maintained", "maintained"),
            "same": ("maintained", "maintained"),
            "stable": ("maintained", "maintained"),
            "resolved": ("stop", "resolved"),
            "stopped": ("stop", "resolved"),
            "stop": ("stop", "resolved"),
            "new": ("new", "new"),
        }

        symptom_change_results: List[SymptomChangeResult] = []
        triaged_items: List[TriagedComplaintItem] = []
        active_symptoms: List[str] = []
        resolved_symptoms: List[str] = []
        global_status_notes: List[str] = []

        for kw_item in keyword_results:
            c_name = kw_item["complaint"]
            c_low = c_name.lower()
            llm_item = parsed_by_name.get(c_low)
            if not llm_item:
                for k, v in parsed_by_name.items():
                    if k in c_low or c_low in k:
                        llm_item = v
                        break

            # Category determination: LLM confirmed or heuristic fallback
            category = kw_item.get("category", "symptom")
            if llm_item and "category" in llm_item:
                cand_cat = str(llm_item["category"]).strip().lower()
                if cand_cat in ("symptom", "negated_symptom", "global_status", "non_symptom"):
                    category = cand_cat

            # Canonical entity determination
            canonical_entity = kw_item.get("canonical_entity")
            if llm_item and llm_item.get("canonical_entity"):
                canonical_entity = str(llm_item["canonical_entity"]).strip().lower()
            elif category == "negated_symptom" and not canonical_entity:
                canonical_entity = extract_canonical_from_negation(c_name)

            # Status determination
            canonical_st = kw_item.get("kw_status")
            display_st = kw_item.get("display_kw")
            if llm_item and llm_item.get("status"):
                raw_st = str(llm_item.get("status", "")).strip().lower()
                if raw_st in norm_map:
                    canonical_st, display_st = norm_map[raw_st]

            if category == "negated_symptom":
                canonical_st = "stop"
                display_st = "resolved"

            change_desc = str(llm_item.get("change_description") if llm_item else kw_item.get("condition_note") or "")
            notes = str(llm_item.get("notes") if llm_item else "")

            # Dispatch based on category
            if category == "symptom":
                target_term = canonical_entity or c_name
                active_symptoms.append(target_term)
                symptom_change_results.append(
                    SymptomChangeResult(
                        complaint=target_term,
                        keyword_status=kw_item.get("kw_status") or "new",
                        confirmed_status=canonical_st or "new",
                        display_status=display_st or "new",
                        change_description=change_desc or f"Status: {display_st}",
                        notes=notes or f"Symptom {display_st}",
                    )
                )
            elif category == "negated_symptom":
                target_term = canonical_entity or c_name
                resolved_symptoms.append(target_term)
                symptom_change_results.append(
                    SymptomChangeResult(
                        complaint=target_term,
                        keyword_status="stop",
                        confirmed_status="stop",
                        display_status="resolved",
                        change_description=change_desc or f"Resolved/negated: {c_name}",
                        notes=notes or f"Negated symptom: {c_name}",
                    )
                )
            elif category == "global_status":
                global_status_notes.append(c_name)

            heuristic_mark = kw_item.get("display_kw")
            heuristic_cat = kw_item.get("category")
            is_changed = (
                (display_st is not None and display_st != heuristic_mark)
                or (category != heuristic_cat)
            )

            triaged_items.append(
                TriagedComplaintItem(
                    raw_text=c_name,
                    category=category,
                    canonical_entity=canonical_entity,
                    status=canonical_st,
                    display_status=display_st,
                    change_description=change_desc,
                    notes=notes,
                    heuristic_mark=heuristic_mark,
                    heuristic_category=heuristic_cat,
                    changed=is_changed,
                )
            )

        # Unified triage_delta is derived directly from triaged_items (single source of truth)
        triage_delta: List[Dict[str, Any]] = [item.to_dict() for item in triaged_items]

        return ClinicalChangeAssessment(
            symptoms=symptom_change_results,
            triaged_items=triaged_items,
            active_symptoms=list(dict.fromkeys(active_symptoms)),
            resolved_symptoms=list(dict.fromkeys(resolved_symptoms)),
            global_status_notes=list(dict.fromkeys(global_status_notes)),
            patient_condition=str(data.get("patient_condition") or "").strip(),
            symptom_notes=str(data.get("symptom_notes") or "").strip(),
            raw_llm_response=raw_text,
            used_llm=True,
            triage_delta=triage_delta,
        )

    def _build_keyword_fallback(
        self,
        keyword_results: List[Dict[str, Any]],
        past_cases: List[Dict[str, Any]],
    ) -> ClinicalChangeAssessment:
        symptom_change_results: List[SymptomChangeResult] = []
        triaged_items: List[TriagedComplaintItem] = []
        active_symptoms: List[str] = []
        resolved_symptoms: List[str] = []
        global_status_notes: List[str] = []
        notes_parts: List[str] = []

        for item in keyword_results:
            c_name = item["complaint"]
            cat = item.get("category", "symptom")
            canonical = item.get("canonical_entity")
            kw_st = item.get("kw_status")
            disp_kw = item.get("display_kw")
            desc = item.get("condition_note", "")

            if cat == "symptom":
                target_term = canonical or c_name
                active_symptoms.append(target_term)
                if item.get("matched_past") != "None":
                    desc = f"{disp_kw}, changed from {item['matched_past']}"
                symptom_change_results.append(
                    SymptomChangeResult(
                        complaint=target_term,
                        keyword_status=kw_st or "new",
                        confirmed_status=kw_st or "new",
                        display_status=disp_kw or "new",
                        change_description=desc,
                        notes=f"Keyword heuristic: {disp_kw}",
                    )
                )
                notes_parts.append(f"{target_term}: {desc}")
            elif cat == "negated_symptom":
                target_term = canonical or c_name
                resolved_symptoms.append(target_term)
                desc = f"Resolved / negated ({c_name})"
                symptom_change_results.append(
                    SymptomChangeResult(
                        complaint=target_term,
                        keyword_status="stop",
                        confirmed_status="stop",
                        display_status="resolved",
                        change_description=desc,
                        notes="Negated symptom",
                    )
                )
                notes_parts.append(f"{target_term}: resolved / denied")
            elif cat == "global_status":
                global_status_notes.append(c_name)
                notes_parts.append(f"Patient reports {c_name.lower()}")

            triaged_items.append(
                TriagedComplaintItem(
                    raw_text=c_name,
                    category=cat,
                    canonical_entity=canonical,
                    status=kw_st,
                    display_status=disp_kw,
                    change_description=desc,
                    notes=f"Heuristic triage: {cat}",
                    heuristic_mark=disp_kw,
                    heuristic_category=cat,
                    changed=False,
                )
            )

        cond = "Follow-up consultation review" if past_cases else "Initial baseline consultation"
        if global_status_notes:
            cond = f"Overall status: {', '.join(global_status_notes)}"

        s_notes = " | ".join(notes_parts) if notes_parts else "No active symptoms reported."

        all_past_meds: List[str] = []
        for v in past_cases:
            all_past_meds.extend(_extract_meds(v))
        unique_past_meds = list(dict.fromkeys(all_past_meds))
        latest_doc_meds = list(dict.fromkeys(_extract_meds(past_cases[-1]))) if past_cases else []
        complaints = [it["complaint"] for it in keyword_results if "complaint" in it]
        patient_use = _extract_patient_reported_meds(complaints)
        disc_meds = _extract_discontinued_meds(current_symptoms=complaints)

        return ClinicalChangeAssessment(
            symptoms=symptom_change_results,
            triaged_items=triaged_items,
            active_symptoms=list(dict.fromkeys(active_symptoms)),
            resolved_symptoms=list(dict.fromkeys(resolved_symptoms)),
            global_status_notes=list(dict.fromkeys(global_status_notes)),
            patient_condition=cond,
            symptom_notes=s_notes,
            raw_llm_response=None,
            used_llm=False,
            triage_delta=[it.to_dict() for it in triaged_items],
            historical_prescription=unique_past_meds,
            latest_documented_prescription=latest_doc_meds,
            patient_reported_current_use=patient_use,
            actual_discontinuation=disc_meds,
        )


# ---------------------------------------------------------------------------
# Entity Ranking Service — SRP: resolves ontology, classifies clinical status,
# runs rank_meds scoring for every entity in a decorated case.
# Injected into PrescriptionServiceV2 via constructor (DIP).
# ---------------------------------------------------------------------------

class EntityRankingService:
    """Resolves ontology metadata, classifies clinical change status, and ranks
    historically prescribed molecules for each entity in a decorated case.

    Single Responsibility: this class owns *only* the per-entity resolution +
    classification + ranking concern.  It knows nothing about MongoDB candidate
    fetching, past-case retrieval, or response assembly.

    Constructor Parameters
    ----------------------
    settings:        App settings (HPO/MONDO paths, SapBERT config).
    hpo_service:     Injected HPO search service (DIP).
    mondo_service:   Injected MONDO search service (DIP).
    hpo_taxonomy:    HPO TaxonomyIndex for Wu-Palmer LCA (optional).
    mondo_taxonomy:  MONDO TaxonomyIndex for Wu-Palmer LCA (optional).
    embedder:        SapBERT embedder (injected, DIP).
    """

    def __init__(
        self,
        *,
        hpo_service: Any,
        mondo_service: Any,
        hpo_taxonomy: Any,
        mondo_taxonomy: Any,
        embedder: Any,
    ) -> None:
        self._hpo_svc = hpo_service
        self._mondo_svc = mondo_service
        self._hpo_taxonomy = hpo_taxonomy
        self._mondo_taxonomy = mondo_taxonomy
        self._embedder = embedder

    def _resolve_hpo(self, term: str, hpo_code: Any, hpo_curie: Any) -> Tuple[Optional[int], Optional[str]]:
        """Return (numeric_code, curie_string) for a symptom term.

        Uses the decorated value if already present; falls back to an HPO
        search otherwise.
        """
        code: Optional[int] = hpo_code if isinstance(hpo_code, int) else None
        if code is None and hpo_code is not None:
            digits = "".join(filter(str.isdigit, str(hpo_code)))
            code = int(digits) if digits else None
        curie: Optional[str] = hpo_curie

        if code is None or curie is None:
            try:
                hits = self._hpo_svc.search(term, limit=1)
                if hits:
                    h = hits[0]
                    curie = curie or h.hpo_id
                    if code is None:
                        digits = "".join(filter(str.isdigit, h.hpo_id))
                        code = int(digits) if digits else None
            except Exception:
                pass
        return code, curie

    def _resolve_mondo(self, term: str, mondo_code: Any, mondo_curie: Any) -> Tuple[Optional[int], Optional[str]]:
        """Return (numeric_code, curie_string) for a diagnosis term.

        Uses the decorated value if already present; falls back to a MONDO
        search otherwise.
        """
        code: Optional[int] = mondo_code if isinstance(mondo_code, int) else None
        if code is None and mondo_code is not None:
            digits = "".join(filter(str.isdigit, str(mondo_code)))
            code = int(digits) if digits else None
        curie: Optional[str] = mondo_curie

        if code is None or curie is None:
            try:
                hits = self._mondo_svc.search(term, limit=1)
                if hits:
                    h = hits[0]
                    curie = curie or h.mondo_id
                    if code is None:
                        digits = "".join(filter(str.isdigit, h.mondo_id))
                        code = int(digits) if digits else None
            except Exception:
                pass
        return code, curie

    def _get_embedding(self, term: str, decorated_embedding: Any) -> Any:
        """Return the SapBERT embedding for a term; generate fresh if absent."""
        if decorated_embedding is not None:
            return decorated_embedding
        try:
            return self._embedder.embed_entities([term])[0]
        except Exception:
            return None

    def _delta_status(
        self,
        term: str,
        entity_type: str,
        delta: Dict[str, Any],
        past_term_set: Set[str],
    ) -> str:
        term_low = term.strip().lower()

        # Check symptoms
        for s in delta.get("decreased_symptoms", []):
            if s.strip().lower() == term_low:
                return "decreased"
        for s in delta.get("increased_symptoms", []):
            if s.strip().lower() == term_low:
                return "increased"
        for s in delta.get("stop_symptoms", []):
            if s.strip().lower() == term_low:
                return "stop"
        for s in delta.get("maintained_symptoms", []):
            if s.strip().lower() == term_low:
                return "maintained"
        for s in delta.get("new_symptoms", []):
            if s.strip().lower() == term_low:
                return "new"

        # Check diagnoses
        for d in delta.get("stop_diagnoses", []):
            if d.strip().lower() == term_low:
                return "stop"
        for d in delta.get("maintained_diagnoses", []):
            if d.strip().lower() == term_low:
                return "maintained"
        for d in delta.get("new_diagnoses", []):
            if d.strip().lower() == term_low:
                return "new"

        return classify_entity_change(term, past_term_set)

    def build_entity_matrix(
        self,
        *,
        decorated_case: Any,
        candidates: List[Dict[str, Any]],
        delta: Dict[str, Any],
        past_cases: List[Dict[str, Any]],
        patient_age: Optional[float],
        patient_gender: Optional[str],
        top_k: int = 5,
    ) -> List[Any]:
        """Build ranked molecule rows for every symptom and diagnosis entity.

        For each entity the method:
          1. Resolves HPO/MONDO code + CURIE.
          2. Resolves the SapBERT embedding.
          3. Classifies clinical change status from delta (with keyword fallback).
          4. Calls rank_meds_for_entity against the shared candidate pool.
          5. Stamps delta_status on the resulting EntityMoleculeRow.
        """
        from meldai_v2.services.entity_matrix_service import rank_meds_for_entity
        from meldai_v2.res_dtos import EntityMoleculeRow

        past_symptom_terms: Set[str] = set()
        past_diagnosis_terms: Set[str] = set()
        for visit in past_cases:
            syms = visit.get("chiefComplaints") or visit.get("symptoms") or []
            for s in syms:
                t = (s.get("term") or s.get("text") or "") if isinstance(s, dict) else str(s)
                if t.strip():
                    past_symptom_terms.add(t.strip())
            for d in visit.get("diagnosis", []):
                t = (d.get("term") or d.get("text") or "") if isinstance(d, dict) else str(d)
                from core.utils import clean_diagnosis_term
                cleaned_d = clean_diagnosis_term(t)
                if cleaned_d:
                    past_diagnosis_terms.add(cleaned_d)

        entity_matrix: List[EntityMoleculeRow] = []

        # --- Chief complaints / symptoms ---
        for sym in decorated_case.symptoms:
            term = sym.term or ""
            if not term.strip():
                continue

            query_code, query_curie = self._resolve_hpo(term, sym.hpoCode, sym.hpoCurie)
            query_vector = self._get_embedding(term, sym.embedding)
            status = self._delta_status(term, "symptom", delta, past_symptom_terms)

            logger.info("EntityRankingService — symptom: %s [%s]", term, status)
            row = rank_meds_for_entity(
                query_term=term,
                entity_type="symptom",
                query_code=query_code,
                query_curie=query_curie,
                query_vector=query_vector,
                taxonomy=self._hpo_taxonomy,
                candidates=candidates,
                patient_age=patient_age,
                patient_gender=patient_gender,
                top_cases_limit=top_k,
                k=top_k,
            )
            row.delta_status = status
            entity_matrix.append(row)

        # --- Diagnoses ---
        for dx in decorated_case.diagnosis:
            term = dx.term or ""
            if not term.strip():
                continue

            query_code, query_curie = self._resolve_mondo(term, dx.mondoCode, dx.mondoCurie)
            query_vector = self._get_embedding(term, dx.embedding)
            status = self._delta_status(term, "diagnosis", delta, past_diagnosis_terms)

            logger.info("EntityRankingService — diagnosis: %s [%s]", term, status)
            row = rank_meds_for_entity(
                query_term=term,
                entity_type="diagnosis",
                query_code=query_code,
                query_curie=query_curie,
                query_vector=query_vector,
                taxonomy=self._mondo_taxonomy,
                candidates=candidates,
                patient_age=patient_age,
                patient_gender=patient_gender,
                top_cases_limit=top_k,
                k=top_k,
            )
            row.delta_status = status
            entity_matrix.append(row)

        logger.info(
            "EntityRankingService — matrix complete: %d rows, %d with molecules",
            len(entity_matrix),
            sum(1 for r in entity_matrix if r.top_molecules),
        )
        return entity_matrix


# Aliases for clean service references
ClinicalDeltaService = ChiefComplaintChangeService


def get_clinical_delta_service() -> ChiefComplaintChangeService:
    """Return ChiefComplaintChangeService instance."""
    return ChiefComplaintChangeService()
