"""molecule_confusion_matrix.py

Generates per-item One-vs-Rest (OvR) confusion matrices at three granularity levels:

  1. Brand      – each distinct brand name as its own class
  2. Molecule   – each canonical_molecule string (FDCs kept as-is, e.g. "A + B")
  3. FDC-Broken – FDCs decomposed into constituent molecules (e.g. "A + B" → "A", "B")

For every item in the respective vocabulary the script computes:
  TP  – item was expected  AND prescribed (across all cases)
  FP  – item was NOT expected but was still prescribed
  FN  – item was expected  but NOT prescribed
  TN  – item was neither expected nor prescribed
  Precision = TP / (TP + FP)
  Recall    = TP / (TP + FN)
  F1        = 2 * Precision * Recall / (Precision + Recall)
  Support   = TP + FN  (total number of times this item was expected)

Output:
  tmp/molecule_confusion_matrix_brand.csv
  tmp/molecule_confusion_matrix_molecule.csv
  tmp/molecule_confusion_matrix_fdc_broken.csv
  tmp/molecule_confusion_matrix_brand.png       (optional – requires matplotlib)
  tmp/molecule_confusion_matrix_molecule.png    (optional)
  tmp/molecule_confusion_matrix_fdc_broken.png  (optional)

Usage:
  python scripts/analyzer/molecule_confusion_matrix.py
  python scripts/analyzer/molecule_confusion_matrix.py --file tmp/prescribe_pipeline_v2_20261026_073022.json
  python scripts/analyzer/molecule_confusion_matrix.py --top 30  # show only top-30 items by support
  python scripts/analyzer/molecule_confusion_matrix.py --no-plot  # skip chart generation
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import site
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Ensure user site-packages are available even when running inside a virtual environment
_user_site = site.getusersitepackages()
if os.path.exists(_user_site) and _user_site not in sys.path:
    sys.path.insert(0, _user_site)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


# ==============================================================================
# 1. PATH RESOLUTION
# ==============================================================================

def resolve_input_path(filepath: Optional[str] = None) -> str:
    project_root = Path(__file__).resolve().parent.parent.parent
    if filepath:
        for candidate in [Path(filepath), project_root / filepath, project_root / "tmp" / filepath]:
            if candidate.is_file():
                return str(candidate.resolve())
        raise FileNotFoundError(f"File not found: {filepath}")
    tmp_dir = project_root / "tmp"
    if tmp_dir.is_dir():
        files = sorted(tmp_dir.glob("prescribe_pipeline_*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        if files:
            return str(files[0].resolve())
    raise FileNotFoundError("No prescribe_pipeline_*.json found in tmp/. Pass --file.")


def load_data(filepath: Optional[str]) -> List[Dict[str, Any]]:
    resolved = resolve_input_path(filepath)
    print(f"Loading: {resolved}")
    with open(resolved, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of pipeline records.")
    print(f"  {len(data)} records loaded.")
    return data


# ==============================================================================
# 2. TOKEN EXTRACTION
# ==============================================================================

def normalize(text: str) -> str:
    return text.strip().lower() if text else ""


def extract_tokens(med: Dict[str, Any], mode: str) -> List[str]:
    if not isinstance(med, dict):
        return []

    if mode == "brand":
        t = normalize(med.get("brand_name", ""))
        return [t] if t else []

    if mode == "molecule":
        t = normalize(med.get("canonical_molecule", ""))
        return [t] if t else []

    if mode == "fdc_broken":
        constituents = med.get("constituent_strengths_mg")
        if isinstance(constituents, dict) and len(constituents) > 1:
            tokens = [normalize(k) for k in constituents if normalize(k)]
            if tokens:
                return tokens
        mol = med.get("canonical_molecule", "")
        if "+" in mol:
            parts = [normalize(p) for p in mol.split("+") if normalize(p)]
            if parts:
                return parts
        if isinstance(constituents, dict) and len(constituents) == 1:
            k = next(iter(constituents))
            t = normalize(k)
            if t:
                return [t]
        t = normalize(mol) or normalize(med.get("brand_name", ""))
        return [t] if t else []

    raise ValueError(f"Unknown mode: {mode!r}")


def get_meds_from_record(record: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    if key == "expected":
        return record.get("expectedMeds") or []
    if key == "predicted":
        res = record.get("res")
        if isinstance(res, dict):
            meds = res.get("medications") or []
            return meds if isinstance(meds, list) else []
    return []


# ==============================================================================
# 3. PER-ITEM OvR COMPUTATION
# ==============================================================================

def compute_ovr_stats(data: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    """Compute One-vs-Rest TP/FP/FN/TN for every item in the vocabulary."""
    all_expected: List[Set[str]] = []
    all_predicted: List[Set[str]] = []

    for rec in data:
        exp_tokens: List[str] = []
        for m in get_meds_from_record(rec, "expected"):
            exp_tokens.extend(extract_tokens(m, mode))

        pred_tokens: List[str] = []
        for m in get_meds_from_record(rec, "predicted"):
            pred_tokens.extend(extract_tokens(m, mode))

        all_expected.append(set(exp_tokens))
        all_predicted.append(set(pred_tokens))

    vocab: Set[str] = set()
    for s in all_expected:
        vocab |= s
    for s in all_predicted:
        vocab |= s
    vocab = {v for v in vocab if v}

    rows: List[Dict[str, Any]] = []
    for item in vocab:
        tp = fp = fn = tn = 0
        for exp_set, pred_set in zip(all_expected, all_predicted):
            in_exp  = item in exp_set
            in_pred = item in pred_set
            if in_exp and in_pred:
                tp += 1
            elif not in_exp and in_pred:
                fp += 1
            elif in_exp and not in_pred:
                fn += 1
            else:
                tn += 1

        support   = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        rows.append({
            "item":      item,
            "tp":        tp,
            "fp":        fp,
            "fn":        fn,
            "tn":        tn,
            "support":   support,
            "precision": round(precision * 100, 2),
            "recall":    round(recall    * 100, 2),
            "f1":        round(f1        * 100, 2),
        })

    rows.sort(key=lambda r: (-r["support"], r["item"]))
    return rows


# ==============================================================================
# 4. CSV OUTPUT
# ==============================================================================

CSV_FIELDS = ["item", "tp", "fp", "fn", "tn", "support", "precision", "recall", "f1"]


def save_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV saved -> {out_path}")


# ==============================================================================
# 5. CHART VISUALIZATION
# ==============================================================================

def plot_ovr_chart(rows: List[Dict[str, Any]], mode: str, top_n: int, out_path: str) -> None:
    if not PLOTTING_AVAILABLE:
        print("  WARNING: matplotlib not available – skipping chart.")
        return

    display_rows = rows[:top_n]
    if not display_rows:
        return

    # Reverse lists so highest support appears at top of horizontal bar chart
    labels         = [r["item"]      for r in display_rows][::-1]
    precision_vals = [r["precision"] for r in display_rows][::-1]
    recall_vals    = [r["recall"]    for r in display_rows][::-1]
    f1_vals        = [r["f1"]        for r in display_rows][::-1]
    supports       = [r["support"]   for r in display_rows][::-1]

    n     = len(labels)
    bar_h = 0.22
    y     = list(range(n))

    color_prec = "#028090"
    color_rec  = "#E76F51"
    color_f1   = "#1D3557"

    mode_labels = {
        "brand":      "Brand Name",
        "molecule":   "Canonical Molecule (FDC intact)",
        "fdc_broken": "Constituent Molecule (FDC decomposed)",
    }
    title_str = mode_labels.get(mode, mode.title())

    fig_h = max(6, n * 0.45 + 2)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(13, fig_h))

    ax.barh([yi + bar_h for yi in y], precision_vals, bar_h, label="Precision", color=color_prec, alpha=0.9)
    ax.barh([yi         for yi in y], recall_vals,    bar_h, label="Recall",    color=color_rec,  alpha=0.9)
    ax.barh([yi - bar_h for yi in y], f1_vals,        bar_h, label="F1 Score",  color=color_f1,   alpha=0.9)

    for sup, yi in zip(supports, y):
        ax.text(101, yi, f"n={sup}", va="center", ha="left", fontsize=7.5, color="#555")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, 116)
    ax.set_xlabel("Score (%)", fontsize=11, weight="bold")
    ax.set_title(
        f"Per-Item OvR Confusion Metrics – {title_str}\n"
        f"(Top {n} by Expected Frequency  |  n = cases in support column)",
        fontsize=12, weight="bold", pad=10,
    )
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved -> {out_path}")


# ==============================================================================
# 6. CONSOLE REPORT
# ==============================================================================

def print_table(rows: List[Dict[str, Any]], mode: str, top_n: int) -> None:
    mode_labels = {
        "brand":      "BRAND NAME",
        "molecule":   "CANONICAL MOLECULE (FDC intact)",
        "fdc_broken": "CONSTITUENT MOLECULE (FDC broken)",
    }
    print(f"\n{'='*100}")
    print(f"  OvR CONFUSION MATRIX  —  {mode_labels.get(mode, mode.upper())}")
    print(f"{'='*100}")
    header = (
        f"  {'Item':<35} | {'Sup':>4} | "
        f"{'TP':>4} | {'FP':>4} | {'FN':>4} | {'TN':>4} | "
        f"{'Prec%':>7} | {'Rec%':>7} | {'F1%':>7}"
    )
    print(header)
    print(f"  {'-'*35}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

    for r in rows[:top_n]:
        print(
            f"  {r['item'][:35]:<35} | {r['support']:>4} | "
            f"{r['tp']:>4} | {r['fp']:>4} | {r['fn']:>4} | {r['tn']:>4} | "
            f"{r['precision']:>6.1f}% | {r['recall']:>6.1f}% | {r['f1']:>6.1f}%"
        )

    if len(rows) > top_n:
        print(f"  ... and {len(rows) - top_n} more items (see CSV for full list)")
    print(f"{'='*100}")


def print_macro_averages(rows: List[Dict[str, Any]]) -> None:
    valid = [r for r in rows if r["support"] > 0]
    if not valid:
        return
    macro_prec = sum(r["precision"] for r in valid) / len(valid)
    macro_rec  = sum(r["recall"]    for r in valid) / len(valid)
    macro_f1   = sum(r["f1"]        for r in valid) / len(valid)
    print(
        f"\n  Macro-avg (support>0, n={len(valid)} items):  "
        f"Precision={macro_prec:.2f}%  Recall={macro_rec:.2f}%  F1={macro_f1:.2f}%"
    )


# ==============================================================================
# 7. CLI ENTRY POINT
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Per-molecule / per-brand OvR confusion matrix for prescription pipeline results."
    )
    p.add_argument("positional_file", nargs="?", default=None,
                   help="Path to pipeline result JSON (positional).")
    p.add_argument("--file", "-f", default=None, metavar="PATH",
                   help="Path to pipeline result JSON.")
    p.add_argument("--top", "-t", type=int, default=40, metavar="N",
                   help="Top-N items by support to display in console & chart (default: 40).")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip PNG chart generation.")
    p.add_argument("--out-dir", default=None, metavar="DIR",
                   help="Output directory for CSVs and charts (default: <project_root>/tmp/).")
    return p


def main() -> None:
    args = build_parser().parse_args()

    filepath = args.file or args.positional_file
    data     = load_data(filepath)

    project_root = Path(__file__).resolve().parent.parent.parent
    out_dir      = Path(args.out_dir) if args.out_dir else project_root / "tmp"
    out_dir.mkdir(parents=True, exist_ok=True)

    top_n   = args.top
    do_plot = not args.no_plot

    modes = [
        ("brand",      "Brand Name (exact match)"),
        ("molecule",   "Canonical Molecule (FDC kept whole)"),
        ("fdc_broken", "Constituent Molecule (FDC decomposed)"),
    ]

    for mode, description in modes:
        print(f"\n{'─'*60}")
        print(f"  Computing OvR matrix: {description} ...")
        rows = compute_ovr_stats(data, mode)
        print(f"  Vocabulary size: {len(rows)} unique items")

        print_table(rows, mode, top_n)
        print_macro_averages(rows)

        csv_path = str(out_dir / f"molecule_confusion_matrix_{mode}.csv")
        save_csv(rows, csv_path)

        if do_plot:
            chart_path = str(out_dir / f"molecule_confusion_matrix_{mode}.png")
            plot_ovr_chart(rows, mode, top_n, chart_path)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
