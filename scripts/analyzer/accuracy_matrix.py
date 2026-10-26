"""accuracy_matrix.py

Evaluation and accuracy analysis script for prescription recommendations.
Analyzes results from medication prediction pipeline JSON files (e.g.,
`tmp/prescribe_pipeline_20261012_104613.json` or `tmp/prescribe_pipeline_20261014_102216.json`)
by comparing expected medications (`expectedMeds`) against model-prescribed medications
(`res.medications`).

Evaluates three classification modes:
1. Brand Accuracy: Exact brand name matching (`brand_name`)
2. Molecule Accuracy: Canonical molecule matching (`canonical_molecule`)
3. FDC Molecule Broken (Normalize FDC Evaluation): Breaks Fixed-Dose Combinations
   into constituent active molecules (via `constituent_strengths_mg` or `+` delimiter)
   to evaluate whether the intended active drug components were prescribed.

Supports:
- Variable input file paths via CLI argument (positional or --file / -f) or default fallback
- Breakdowns by Age Group, Gender, Consultation Type, and Expected Med Count
- Multi-label 2x2 Confusion Matrices (Brand, Molecule, Normalized FDC)
- Medication Count Agreement Confusion Matrix
- Publication-quality visual comparison charts using matplotlib & seaborn
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import site
import sys

# Ensure user site-packages are available even when running inside a virtual environment
_user_site = site.getusersitepackages()
if os.path.exists(_user_site) and _user_site not in sys.path:
    sys.path.insert(0, _user_site)

# Import plotting libraries with graceful fallback
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

import numpy as np


# Default pipeline result file paths to check in priority order
DEFAULT_FILE_CANDIDATES = [
    # "tmp/prescribe_pipeline_20261014_102216.json",
    # "tmp/prescribe_pipeline_20261012_104613.json",
    "tmp/prescribe_pipeline_20261015_114006.json",
    # "prescribe_pipeline_20261014_102216.json",
    # "prescribe_pipeline_20261012_104613.json",
    "prescribe_pipeline_20261015_114006.json",
]


# ==============================================================================
# 1. DATA MODELS & INGESTION
# ==============================================================================

def resolve_input_path(filepath: Optional[str] = None) -> str:
    """
    Resolve and validate the dataset file path.
    Searches current directory, project root, and tmp/ subdirectory.

    Args:
        filepath: Explicit file path provided by user, or None for default.

    Returns:
        Absolute or verified relative path to the existing JSON file.
    """
    candidates = []
    if filepath:
        candidates.extend([
            filepath,
            os.path.join("tmp", filepath),
            os.path.join("..", filepath),
            os.path.join("..", "tmp", filepath),
        ])
    else:
        candidates.extend(DEFAULT_FILE_CANDIDATES)

    for cand in candidates:
        if os.path.isfile(cand):
            return os.path.abspath(cand)

    # If not found yet, search project root tmp
    project_root = Path(__file__).resolve().parent.parent.parent
    if filepath:
        cand_root = project_root / filepath
        if cand_root.is_file():
            return str(cand_root)
        cand_tmp = project_root / "tmp" / filepath
        if cand_tmp.is_file():
            return str(cand_tmp)

    # Check tmp/ for latest prescribe_pipeline results (v2 or v1)
    tmp_dir = project_root / "tmp"
    if not filepath and tmp_dir.is_dir():
        found_files = list(tmp_dir.glob("prescribe_pipeline_*.json"))
        if found_files:
            found_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return str(found_files[0])

    for def_cand in DEFAULT_FILE_CANDIDATES:
        cand_root = project_root / def_cand
        if cand_root.is_file():
            return str(cand_root)

    raise FileNotFoundError(
        f"Dataset file not found. Tried: {filepath or DEFAULT_FILE_CANDIDATES}"
    )


def load_data(filepath: str) -> List[Dict[str, Any]]:
    """
    Load raw JSON records from the given filepath.

    Args:
        filepath: Path to the JSON dataset file.

    Returns:
        List of dictionary records.
    """
    resolved = resolve_input_path(filepath)

    with open(resolved, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of case records, got: {type(data)}")

    return data


# ==============================================================================
# 2. BUCKETING & CATEGORIZATION HELPERS
# ==============================================================================

def get_age_group(age: float) -> str:
    """
    Categorize patient age into predefined clinical age cohorts.
    Buckets: 0-5, 6-12, 12-21, 22-45, 46-65, 65+
    """
    if age <= 5:
        return "0-5"
    elif age <= 12:
        return "6-12"
    elif age <= 21:
        return "12-21"
    elif age <= 45:
        return "22-45"
    elif age <= 65:
        return "46-65"
    else:
        return "65+"


def get_gender(gender: str) -> str:
    """Normalize patient gender."""
    return gender.strip().capitalize() if gender else "Unknown"


def get_consultation_type(visit_type: str) -> str:
    """
    Normalize consultation visit type into standard display labels:
    'New Consultation' or 'Follow-up'.
    """
    vt = visit_type.strip().lower() if visit_type else ""
    if "new" in vt:
        return "New Consultation"
    elif "follow" in vt:
        return "Follow-up"
    return visit_type.strip().title() if visit_type else "Unknown"


def get_med_count_bucket(count: int) -> str:
    """
    Bucket expected medication count into:
    '0', '1', '2', '3', '4', '5', '5+'
    """
    if count >= 5:
        return "5+" if count > 5 else "5"
    return str(count)


# ==============================================================================
# 3. MATCHING LOGIC & CASE-LEVEL EVALUATION
# ==============================================================================

@dataclass
class CaseEvaluation:
    """Encapsulates matching outcomes for a single clinical case."""
    case_no: int
    age_group: str
    gender: str
    consultation_type: str
    expected_count: int
    prescribed_count: int
    expected_items: List[str]
    prescribed_items: List[str]
    matched_items: List[str]
    matched_count: int
    unmatched_expected: List[str]
    extra_prescribed: List[str]
    accuracy: float  # matched_count / expected_count if expected_count > 0


def normalize_token(text: str) -> str:
    """Normalize string token for reliable case-insensitive matching."""
    return text.strip().lower() if text else ""


def extract_med_tokens(med: Dict[str, Any], match_by: str = "molecule") -> List[str]:
    """
    Extract normalized tokens from a medication object for matching.

    Args:
        med: Medication dictionary from expectedMeds or res.medications.
        match_by: Evaluation mode:
            - 'brand': Exact brand name (`brand_name`)
            - 'molecule': Exact canonical molecule string (`canonical_molecule`)
            - 'fdc_broken': Normalize FDC evaluation — decomposes Fixed-Dose
              Combinations into their constituent active molecules using
              `constituent_strengths_mg` keys or splitting `canonical_molecule` by '+'.

    Returns:
        List of normalized string tokens representing the drug or its active components.
    """
    if not isinstance(med, dict):
        return []

    if match_by == "brand":
        brand = normalize_token(med.get("brand_name", ""))
        return [brand] if brand else []

    elif match_by == "molecule":
        mol = normalize_token(med.get("canonical_molecule", ""))
        return [mol] if mol else []

    elif match_by in ("fdc_broken", "fdc_molecule_broken", "normalized", "normalized_fdc"):
        # 1. Primary decomposition via constituent_strengths_mg dictionary
        constituents = med.get("constituent_strengths_mg")
        if isinstance(constituents, dict) and len(constituents) > 1:
            tokens = [normalize_token(k) for k in constituents.keys() if normalize_token(k)]
            if tokens:
                return tokens

        # 2. Secondary decomposition via canonical_molecule '+' delimiter
        mol = med.get("canonical_molecule", "")
        if "+" in mol:
            parts = [normalize_token(p) for p in mol.split("+") if normalize_token(p)]
            if parts:
                return parts

        # 3. Single-entry dictionary in constituent_strengths_mg
        if isinstance(constituents, dict) and len(constituents) == 1:
            k = list(constituents.keys())[0]
            norm_k = normalize_token(k)
            if norm_k:
                return [norm_k]

        # 4. Fallback to canonical_molecule or brand_name
        norm_mol = normalize_token(mol)
        if norm_mol:
            return [norm_mol]

        brand = normalize_token(med.get("brand_name", ""))
        return [brand] if brand else []

    else:
        raise ValueError(f"Unknown match_by mode: {match_by}")


def evaluate_single_case(record: Dict[str, Any], match_by: str = "molecule") -> CaseEvaluation:
    """
    Evaluate medication matching for a single case record.

    Matching definition:
        accuracy = no. of matching expected items found in res.medications

    Args:
        record: Single dictionary item from the JSON pipeline output.
        match_by: 'brand', 'molecule', or 'fdc_broken'.

    Returns:
        CaseEvaluation object with detailed counts and lists.
    """
    case_info = record.get("req", {}).get("case", {})
    case_no = case_info.get("caseNo", 0)
    patient_info = case_info.get("patientInfo", {})
    age = float(patient_info.get("age", 0)) if patient_info.get("age") is not None else 0.0
    gender = get_gender(patient_info.get("gender", ""))
    consultation_type = get_consultation_type(case_info.get("visitType", ""))

    expected_meds = record.get("expectedMeds") or []
    res_obj = record.get("res")
    if isinstance(res_obj, dict):
        prescribed_meds = res_obj.get("medications") or []
        if not isinstance(prescribed_meds, list):
            prescribed_meds = []
    else:
        prescribed_meds = []

    # Extract tokens for all expected and prescribed medications
    expected_items: List[str] = []
    for m in expected_meds:
        expected_items.extend(extract_med_tokens(m, match_by=match_by))

    prescribed_items: List[str] = []
    for m in prescribed_meds:
        prescribed_items.extend(extract_med_tokens(m, match_by=match_by))

    prescribed_set = set(prescribed_items)

    # Count how many expected meds were matched in prescribed meds
    matched_items = []
    unmatched_expected = []
    for item in expected_items:
        if item in prescribed_set:
            matched_items.append(item)
        else:
            unmatched_expected.append(item)

    # Extra prescribed meds not in expected
    expected_set = set(expected_items)
    extra_prescribed = [item for item in prescribed_items if item not in expected_set]

    matched_count = len(matched_items)
    expected_count = len(expected_items)
    prescribed_count = len(prescribed_items)

    # Case accuracy: percentage of expected medications found
    if expected_count > 0:
        case_accuracy = (matched_count / expected_count) * 100.0
    else:
        # Edge case: If 0 meds were expected and 0 were prescribed, match is 100%
        case_accuracy = 100.0 if prescribed_count == 0 else 0.0

    return CaseEvaluation(
        case_no=case_no,
        age_group=get_age_group(age),
        gender=gender,
        consultation_type=consultation_type,
        expected_count=expected_count,
        prescribed_count=prescribed_count,
        expected_items=expected_items,
        prescribed_items=prescribed_items,
        matched_items=matched_items,
        matched_count=matched_count,
        unmatched_expected=unmatched_expected,
        extra_prescribed=extra_prescribed,
        accuracy=case_accuracy,
    )


# ==============================================================================
# 4. AGGREGATION & SUB-FLOW BREAKDOWNS
# ==============================================================================

@dataclass
class GroupStats:
    """Summary statistics for an aggregated cohort."""
    cohort_name: str
    case_count: int
    total_expected: int
    total_prescribed: int
    total_matched: int
    overall_accuracy: float     # (total_matched / total_expected) * 100
    avg_case_accuracy: float    # average of case_accuracy across cases
    precision: float            # (total_matched / total_prescribed) * 100


def aggregate_evaluations(evaluations: List[CaseEvaluation], cohort_name: str = "Overall") -> GroupStats:
    """Compute aggregate accuracy statistics from a list of case evaluations."""
    if not evaluations:
        return GroupStats(
            cohort_name=cohort_name,
            case_count=0,
            total_expected=0,
            total_prescribed=0,
            total_matched=0,
            overall_accuracy=0.0,
            avg_case_accuracy=0.0,
            precision=0.0,
        )

    case_count = len(evaluations)
    total_expected = sum(e.expected_count for e in evaluations)
    total_prescribed = sum(e.prescribed_count for e in evaluations)
    total_matched = sum(e.matched_count for e in evaluations)

    # Overall matching accuracy = matching expected meds found / total expected meds
    overall_accuracy = (total_matched / total_expected * 100.0) if total_expected > 0 else 0.0

    # Average case-level accuracy
    avg_case_accuracy = float(np.mean([e.accuracy for e in evaluations]))

    # Model precision = correctly predicted expected meds / total meds prescribed
    precision = (total_matched / total_prescribed * 100.0) if total_prescribed > 0 else 0.0

    return GroupStats(
        cohort_name=cohort_name,
        case_count=case_count,
        total_expected=total_expected,
        total_prescribed=total_prescribed,
        total_matched=total_matched,
        overall_accuracy=overall_accuracy,
        avg_case_accuracy=avg_case_accuracy,
        precision=precision,
    )


def compute_all_breakdowns(
    evaluations: List[CaseEvaluation]
) -> Dict[str, Dict[str, GroupStats]]:
    """
    Computes all requested breakdowns:
    - Overall
    - By Age Group
    - By Gender
    - By Consultation Type
    - By Expected Medicine Count

    Returns:
        Dictionary mapping breakdown categories to dicts of GroupStats.
    """
    breakdowns: Dict[str, Dict[str, GroupStats]] = {}

    # 1. Overall
    breakdowns["Overall"] = {
        "All Cases": aggregate_evaluations(evaluations, "All Cases")
    }

    # 2. By Age Group
    age_cohort_order = ["0-5", "6-12", "12-21", "22-45", "46-65", "65+"]
    age_groups: Dict[str, List[CaseEvaluation]] = defaultdict(list)
    for e in evaluations:
        age_groups[e.age_group].append(e)

    breakdowns["By Age Group"] = {
        cohort: aggregate_evaluations(age_groups[cohort], cohort)
        for cohort in age_cohort_order
    }

    # 3. By Gender
    gender_cohort_order = ["Male", "Female"]
    gender_groups: Dict[str, List[CaseEvaluation]] = defaultdict(list)
    for e in evaluations:
        gender_groups[e.gender].append(e)

    breakdowns["By Gender"] = {
        gender: aggregate_evaluations(gender_groups[gender], gender)
        for gender in gender_cohort_order
    }

    # 4. By Consultation Type
    visit_cohort_order = ["New Consultation", "Follow-up"]
    visit_groups: Dict[str, List[CaseEvaluation]] = defaultdict(list)
    for e in evaluations:
        visit_groups[e.consultation_type].append(e)

    breakdowns["By Consultation Type"] = {
        vtype: aggregate_evaluations(visit_groups[vtype], vtype)
        for vtype in visit_cohort_order
    }

    # 5. By Expected Medicine Count
    count_cohort_order = ["0", "1", "2", "3", "4", "5", "5+"]
    count_groups: Dict[str, List[CaseEvaluation]] = defaultdict(list)
    for e in evaluations:
        bucket = get_med_count_bucket(e.expected_count)
        count_groups[bucket].append(e)

    breakdowns["By Expected Medicine Count"] = {
        bucket: aggregate_evaluations(count_groups[bucket], f"{bucket} Meds")
        for bucket in count_cohort_order
    }

    return breakdowns


# ==============================================================================
# 5. CONFUSION MATRICES
# ==============================================================================

def compute_binary_confusion_matrix(
    evaluations: List[CaseEvaluation]
) -> Tuple[int, int, int, int]:
    """
    Computes aggregated multi-label classification confusion matrix elements:
    - True Positives (TP): Expected medications successfully predicted by the model.
    - False Negatives (FN): Expected medications that the model failed to prescribe (omissions).
    - False Positives (FP): Medications prescribed by model that were not in expected meds.
    - True Negatives (TN): Catalog medications neither expected nor prescribed for the case.

    Returns:
        (TP, FP, FN, TN) counts.
    """
    # Build complete vocabulary catalog of distinct items
    catalog: Set[str] = set()
    for e in evaluations:
        catalog.update(e.expected_items)
        catalog.update(e.prescribed_items)

    total_catalog_size = len(catalog)

    tp_total = 0
    fn_total = 0
    fp_total = 0
    tn_total = 0

    for e in evaluations:
        exp_set = set(e.expected_items)
        res_set = set(e.prescribed_items)

        # TP: in both expected and prescribed
        tp = len(exp_set & res_set)
        # FN: expected but missing in prescribed
        fn = len(exp_set - res_set)
        # FP: prescribed but not expected
        fp = len(res_set - exp_set)
        # TN: all catalog items neither expected nor prescribed
        tn = max(0, total_catalog_size - len(exp_set | res_set))

        tp_total += tp
        fn_total += fn
        fp_total += fp
        tn_total += tn

    return tp_total, fp_total, fn_total, tn_total


def compute_count_confusion_matrix(
    records: List[Dict[str, Any]],
    buckets: Optional[List[str]] = None
) -> Tuple[List[str], np.ndarray]:
    """
    Generates an N x N confusion matrix cross-tabulating:
    Expected Medication Count vs Prescribed Medication Count.

    Args:
        records: List of pipeline JSON records.
        buckets: List of count bucket strings.

    Returns:
        (bucket_labels, 2D numpy array confusion matrix).
    """
    if buckets is None:
        buckets = ["0", "1", "2", "3", "4", "5", "5+"]

    bucket_to_idx = {b: i for i, b in enumerate(buckets)}
    n = len(buckets)
    matrix = np.zeros((n, n), dtype=int)

    for record in records:
        exp_count = len(record.get("expectedMeds") or [])
        res_obj = record.get("res")
        res_meds = res_obj.get("medications") if isinstance(res_obj, dict) else []
        res_count = len(res_meds) if isinstance(res_meds, list) else 0

        exp_bucket = get_med_count_bucket(exp_count)
        res_bucket = get_med_count_bucket(res_count)

        r = bucket_to_idx.get(exp_bucket, n - 1)
        c = bucket_to_idx.get(res_bucket, n - 1)
        matrix[r, c] += 1

    return buckets, matrix


# ==============================================================================
# 6. TEXT / CLI REPORT FORMATTING
# ==============================================================================

def print_breakdown_section(section_name: str, cohort_dict: Dict[str, GroupStats]):
    """Pretty-print a formatted ASCII table for an evaluation breakdown section."""
    print(f"\n--- {section_name} ---")
    header = f"{'Cohort':<20} | {'Cases':<6} | {'Exp Items':<9} | {'Matched':<8} | {'Accuracy %':<12} | {'Avg Case %':<12} | {'Precision %':<12}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for cohort_key, stats in cohort_dict.items():
        print(
            f"{cohort_key:<20} | "
            f"{stats.case_count:<6} | "
            f"{stats.total_expected:<9} | "
            f"{stats.total_matched:<8} | "
            f"{stats.overall_accuracy:>10.2f}% | "
            f"{stats.avg_case_accuracy:>10.2f}% | "
            f"{stats.precision:>10.2f}%"
        )
    print("-" * len(header))


def print_confusion_matrix_ascii(title: str, tp: int, fp: int, fn: int, tn: int):
    """Prints a styled 2x2 confusion matrix with core classification metrics."""
    total = tp + fp + fn + tn
    precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"\n=======================================================")
    print(f" {title.upper()} - 2x2 CONFUSION MATRIX")
    print(f"=======================================================")
    print(f"{'':<22} | {'Predicted Positive':<18} | {'Predicted Negative':<18}")
    print(f"{'-'*22}-|-{'-'*18}-|-{'-'*18}")
    print(f"{'Actual Positive (Exp)':<22} | {f'TP: {tp:,}':<18} | {f'FN: {fn:,}':<18}")
    print(f"{'Actual Negative (Not)':<22} | {f'FP: {fp:,}':<18} | {f'TN: {tn:,}':<18}")
    print(f"-------------------------------------------------------")
    print(f" Metrics:")
    print(f"   • Matching Recall (Sensitivity) : {recall:.2f}%  (Expected items captured)")
    print(f"   • Model Precision              : {precision:.2f}%  (Prescriptions that were expected)")
    print(f"   • F1 Score                     : {f1:.2f}%")
    print(f"   • Specificity                  : {specificity:.2f}%")
    print(f"   • Total Decisions Evaluated    : {total:,}")
    print(f"=======================================================\n")


# ==============================================================================
# 7. CHART VISUALIZATION
# ==============================================================================

def plot_confusion_matrices(
    brand_tp_fp_fn_tn: Tuple[int, int, int, int],
    mol_tp_fp_fn_tn: Tuple[int, int, int, int],
    fdc_tp_fp_fn_tn: Tuple[int, int, int, int],
    count_buckets: List[str],
    count_matrix: np.ndarray,
    output_filepath: str = "confusion_matrices.png"
):
    """
    Generate and save a visual 4-panel figure (2x2 grid) containing:
    1. Brand Detection 2x2 Confusion Matrix Heatmap
    2. Molecule Detection 2x2 Confusion Matrix Heatmap
    3. Normalized FDC Molecule Broken 2x2 Confusion Matrix Heatmap
    4. Medication Count Cross-tabulation Confusion Matrix (Expected vs Prescribed)
    """
    sns.set_theme(style="white")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Color palettes for matrices
    cmaps = ["Blues", "Greens", "Purples"]
    modes = [
        ("Brand Detection (Exact Brand)", brand_tp_fp_fn_tn, axes[0, 0], cmaps[0]),
        ("Molecule Detection (Canonical Molecule)", mol_tp_fp_fn_tn, axes[0, 1], cmaps[1]),
        ("FDC Molecule Broken (Normalized FDC)", fdc_tp_fp_fn_tn, axes[1, 0], cmaps[2]),
    ]

    for title, (tp, fp, fn, tn), ax, cmap in modes:
        mat = np.array([[tp, fn], [fp, tn]])
        labels = np.array([
            [f"TP\n{tp:,}", f"FN\n{fn:,}"],
            [f"FP\n{fp:,}", f"TN\n{tn:,}"]
        ])
        sns.heatmap(
            mat,
            annot=labels,
            fmt="",
            cmap=cmap,
            cbar=False,
            ax=ax,
            xticklabels=["Prescribed", "Not Prescribed"],
            yticklabels=["Expected", "Not Expected"],
            annot_kws={"size": 13, "weight": "bold"}
        )
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
        prec = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
        ax.set_title(
            f"{title}\nRecall: {recall:.1f}% | Precision: {prec:.1f}%",
            fontsize=12,
            weight="bold",
            pad=10
        )
        ax.set_ylabel("Ground Truth (Expected)", fontsize=11, weight="bold")
        ax.set_xlabel("Pipeline Prediction", fontsize=11, weight="bold")

    # Panel 4 (Bottom-Right): Medication Count Agreement Confusion Matrix
    sns.heatmap(
        count_matrix,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        cbar=True,
        ax=axes[1, 1],
        xticklabels=count_buckets,
        yticklabels=count_buckets,
        annot_kws={"size": 11}
    )
    count_diagonal_acc = (np.trace(count_matrix) / np.sum(count_matrix) * 100) if np.sum(count_matrix) > 0 else 0
    axes[1, 1].set_title(
        f"Medication Count Confusion Matrix\nExact Count Agreement: {count_diagonal_acc:.1f}%",
        fontsize=12,
        weight="bold",
        pad=10
    )
    axes[1, 1].set_xlabel("Prescribed Med Count Bucket", fontsize=11, weight="bold")
    axes[1, 1].set_ylabel("Expected Med Count Bucket", fontsize=11, weight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
    plt.savefig(output_filepath, dpi=300)
    plt.close()
    print(f" Saved confusion matrices chart: {output_filepath}")


def plot_accuracy_comparisons(
    brand_breakdowns: Dict[str, Dict[str, GroupStats]],
    mol_breakdowns: Dict[str, Dict[str, GroupStats]],
    fdc_breakdowns: Dict[str, Dict[str, GroupStats]],
    output_filepath: str = "accuracy_breakdown.png"
):
    """
    Generate and save grouped bar charts comparing accuracy across the 3 classification modes:
    - Brand Match
    - Molecule Match
    - FDC Molecule Broken (Normalize FDC Evaluation)

    Panels:
    1. Overall Summary
    2. By Age Group
    3. By Gender
    4. By Consultation Type
    5. By Expected Medicine Count
    """
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(18, 14))

    # Grid layout: Top row for Overall + Gender; Middle row for Age Group & Visit Type; Bottom for Med Count
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.2, 1.2])
    ax_overall = fig.add_subplot(gs[0, 0])
    ax_gender = fig.add_subplot(gs[0, 1])
    ax_age = fig.add_subplot(gs[1, 0])
    ax_consultation = fig.add_subplot(gs[1, 1])
    ax_count = fig.add_subplot(gs[2, :])

    sections = [
        ("Overall", ax_overall, "Overall Accuracy Summary"),
        ("By Gender", ax_gender, "Accuracy by Patient Gender"),
        ("By Age Group", ax_age, "Accuracy by Age Group Cohort"),
        ("By Consultation Type", ax_consultation, "Accuracy by Consultation Type"),
        ("By Expected Medicine Count", ax_count, "Accuracy by Expected Medicine Count"),
    ]

    # Harmonious colors for the 3 modes:
    # Teal/Cyan for Brand, Slate Blue/Navy for Molecule, Coral/Amber for FDC Broken
    color_brand = "#028090"
    color_mol = "#1D3557"
    color_fdc = "#E76F51"

    width = 0.26  # width of each bar in the group

    for section_key, ax, chart_title in sections:
        brand_dict = brand_breakdowns[section_key]
        mol_dict = mol_breakdowns[section_key]
        fdc_dict = fdc_breakdowns[section_key]

        categories = list(mol_dict.keys())
        brand_vals = [brand_dict[c].overall_accuracy for c in categories]
        mol_vals = [mol_dict[c].overall_accuracy for c in categories]
        fdc_vals = [fdc_dict[c].overall_accuracy for c in categories]

        x = np.arange(len(categories))

        rects1 = ax.bar(x - width, brand_vals, width, label="Brand Match", color=color_brand)
        rects2 = ax.bar(x, mol_vals, width, label="Molecule Match", color=color_mol)
        rects3 = ax.bar(x + width, fdc_vals, width, label="FDC Molecule Broken (Norm FDC)", color=color_fdc)

        ax.set_ylabel("Matching Accuracy (%)", fontsize=10, weight="bold")
        ax.set_title(chart_title, fontsize=12, weight="bold", pad=8)
        ax.set_xticks(x)

        rotation = 15 if len(categories) > 4 and section_key != "By Expected Medicine Count" else 0
        ha = "right" if rotation > 0 else "center"
        ax.set_xticklabels(categories, rotation=rotation, ha=ha, fontsize=9)

        max_val = max(brand_vals + mol_vals + fdc_vals + [10])
        ax.set_ylim(0, min(100.0, max_val + 18))
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

        # Value annotations on bars
        for rects, color in [(rects1, color_brand), (rects2, color_mol), (rects3, color_fdc)]:
            for rect in rects:
                h = rect.get_height()
                if h > 0:
                    ax.annotate(
                        f"{h:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        weight="bold"
                    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_filepath)), exist_ok=True)
    plt.savefig(output_filepath, dpi=300)
    plt.close()
    print(f" Saved accuracy breakdown charts: {output_filepath}")


# ==============================================================================
# 8. MAIN ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prescription Recommendation Accuracy & Matrix Evaluator with Normalized FDC Evaluation"
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        default=None,
        help="Path to the pipeline JSON results file (e.g. tmp/prescribe_pipeline_20261014_102216.json)"
    )
    parser.add_argument(
        "--file", "-f",
        dest="flag_filepath",
        default=None,
        help="Alternative flag to specify input JSON results file path"
    )
    parser.add_argument(
        "--output-dir", "-o",
        dest="output_dir",
        default=".",
        help="Output directory to save generated chart images (default: current directory)"
    )

    args = parser.parse_args()

    # Determine targeted input file path
    target_path = args.flag_filepath or args.filepath
    resolved_path = resolve_input_path(target_path)

    print(f"Loading pipeline data from: {resolved_path}")
    raw_records = load_data(resolved_path)
    total_records = len(raw_records)
    print(f"Successfully loaded {total_records} case records.\n")

    # File stem for naming chart images
    stem = Path(resolved_path).stem
    out_dir = Path(args.output_dir)

    # --------------------------------------------------------------------------
    # SUBFLOW 1: BRAND ACCURACY
    # --------------------------------------------------------------------------
    print("=" * 80)
    print(" 1. ACCURACY BY DRUG NAME / BRAND (brand_name)")
    print("=" * 80)

    brand_evaluations = [evaluate_single_case(r, match_by="brand") for r in raw_records]
    brand_breakdowns = compute_all_breakdowns(brand_evaluations)

    for section_title, cohort_dict in brand_breakdowns.items():
        print_breakdown_section(f"Brand Accuracy {section_title}", cohort_dict)

    # --------------------------------------------------------------------------
    # SUBFLOW 2: MOLECULE ACCURACY
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 2. ACCURACY BY MOLECULE (canonical_molecule)")
    print("=" * 80)

    mol_evaluations = [evaluate_single_case(r, match_by="molecule") for r in raw_records]
    mol_breakdowns = compute_all_breakdowns(mol_evaluations)

    for section_title, cohort_dict in mol_breakdowns.items():
        print_breakdown_section(f"Molecule Accuracy {section_title}", cohort_dict)

    # --------------------------------------------------------------------------
    # SUBFLOW 3: FDC MOLECULE BROKEN (NORMALIZE FDC EVALUATION)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 3. ACCURACY BY FDC MOLECULE BROKEN (Normalize FDC Evaluation)")
    print("    (Decomposes FDC drugs into constituent active molecules for component matching)")
    print("=" * 80)

    fdc_evaluations = [evaluate_single_case(r, match_by="fdc_broken") for r in raw_records]
    fdc_breakdowns = compute_all_breakdowns(fdc_evaluations)

    for section_title, cohort_dict in fdc_breakdowns.items():
        print_breakdown_section(f"Normalized FDC Broken Accuracy {section_title}", cohort_dict)

    # --------------------------------------------------------------------------
    # SUBFLOW 4: CONFUSION MATRICES
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 4. CONFUSION MATRICES FOR ALL THREE CLASSIFICATIONS")
    print("=" * 80)

    # 4.1 Brand 2x2 Matrix
    brand_tp_fp_fn_tn = compute_binary_confusion_matrix(brand_evaluations)
    print_confusion_matrix_ascii("Brand Detection", *brand_tp_fp_fn_tn)

    # 4.2 Molecule 2x2 Matrix
    mol_tp_fp_fn_tn = compute_binary_confusion_matrix(mol_evaluations)
    print_confusion_matrix_ascii("Molecule Detection", *mol_tp_fp_fn_tn)

    # 4.3 FDC Broken 2x2 Matrix
    fdc_tp_fp_fn_tn = compute_binary_confusion_matrix(fdc_evaluations)
    print_confusion_matrix_ascii("Normalized FDC Molecule Broken Detection", *fdc_tp_fp_fn_tn)

    # 4.4 Medication Count Agreement Confusion Matrix
    count_buckets, count_matrix = compute_count_confusion_matrix(raw_records)
    print("--- Medication Count Agreement Confusion Matrix ---")
    print(f"{'Exp / Prescribed':<16} | " + " | ".join(f"{b:>5}" for b in count_buckets))
    print("-" * (18 + len(count_buckets) * 8))
    for i, exp_b in enumerate(count_buckets):
        row_str = " | ".join(f"{count_matrix[i, j]:>5}" for j in range(len(count_buckets)))
        print(f"Expected {exp_b:<7} | {row_str}")
    print("-" * (18 + len(count_buckets) * 8))

    # --------------------------------------------------------------------------
    # SUBFLOW 5: GENERATE VISUAL CHARTS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(" 5. GENERATING CHARTS VIA MATPLOTLIB & SEABORN")
    print("=" * 80)

    if PLOTTING_AVAILABLE:
        # Standard filenames
        accuracy_img = str(out_dir / "accuracy_breakdown.png")
        confusion_img = str(out_dir / "confusion_matrices.png")

        # Also file-specific filenames for easy archiving
        accuracy_img_stem = str(out_dir / f"accuracy_breakdown_{stem}.png")
        confusion_img_stem = str(out_dir / f"confusion_matrices_{stem}.png")

        plot_accuracy_comparisons(
            brand_breakdowns=brand_breakdowns,
            mol_breakdowns=mol_breakdowns,
            fdc_breakdowns=fdc_breakdowns,
            output_filepath=accuracy_img
        )
        if accuracy_img != accuracy_img_stem:
            import shutil
            shutil.copyfile(accuracy_img, accuracy_img_stem)
            print(f" Archived copy: {accuracy_img_stem}")

        plot_confusion_matrices(
            brand_tp_fp_fn_tn=brand_tp_fp_fn_tn,
            mol_tp_fp_fn_tn=mol_tp_fp_fn_tn,
            fdc_tp_fp_fn_tn=fdc_tp_fp_fn_tn,
            count_buckets=count_buckets,
            count_matrix=count_matrix,
            output_filepath=confusion_img
        )
        if confusion_img != confusion_img_stem:
            import shutil
            shutil.copyfile(confusion_img, confusion_img_stem)
            print(f" Archived copy: {confusion_img_stem}")

        print("\nAll stats calculated and charts generated successfully.")
    else:
        print("\n[Notice] matplotlib or seaborn not installed in this Python environment.")
        print("All text stats & matrices were computed above. Install matplotlib to generate image charts:")
        print("  ~/.local/bin/pip install matplotlib seaborn --break-system-packages")


if __name__ == "__main__":
    main()
