"""lighten_result_json.py

Strips bulky prompt fields (top-level `prompts`, `res.llmMedCheck.prompt`,
`res.llmMedPrescribe.prompt`), `entity_matrix`, and optionally
`res.workflow_prompts_file` / `res.llm_log_file` file-path strings out of a
prescription pipeline result JSON into a separate audit log file, producing
a significantly smaller "lightweight" result file that is fast to load, diff,
and commit.

Behaviour
─────────
Given:   tmp/prescribe_pipeline_v2_20261026_073022.json   (e.g. 45 MB)

Produces:
  tmp/prescribe_pipeline_v2_20261026_073022_light.json     ← result without prompts or entity_matrix
  tmp/prescribe_pipeline_v2_20261026_073022_prompts.json   ← prompt audit log

The audit log keeps just enough context to trace every prompt back to its case:
  [
    {
      "runId":   "...",
      "caseNo":  1,
      "prompts": { "step1_llm_med_check_prompt": "...", ... }
    },
    ...
  ]

The lightweight file is identical to the original EXCEPT:
- Top-level "prompts" key is replaced with "prompts_file" (relative path to audit log)
- "llmMedCheck".prompt and "llmMedPrescribe".prompt are excluded
- "entity_matrix" is excluded
- res.workflow_prompts_file and res.llm_log_file are excluded (unless --keep-res-paths)

Usage:
  python scripts/analyzer/lighten_result_json.py
  python scripts/analyzer/lighten_result_json.py tmp/prescribe_pipeline_v2_*.json
  python scripts/analyzer/lighten_result_json.py --file tmp/prescribe_pipeline_v2_20261026_073022.json
  python scripts/analyzer/lighten_result_json.py --keep-res-paths      # keep res.workflow_prompts_file etc.
  python scripts/analyzer/lighten_result_json.py --keep-entity-matrix  # keep entity_matrix in light file
  python scripts/analyzer/lighten_result_json.py --dry-run             # show sizes, do not write
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Res-level fields that are just file paths and add noise to the light file
_RES_PATH_FIELDS = {"workflow_prompts_file", "llm_log_file"}

# Step fields whose prompts should be stripped from the light file
_LLM_STEP_KEYS = {"llmMedCheck", "llmMedPrescribe"}


# ==============================================================================
# PATH RESOLUTION
# ==============================================================================

def resolve_input_path(filepath: Optional[str] = None) -> Path:
    project_root = Path(__file__).resolve().parent.parent.parent

    if filepath:
        for candidate in [Path(filepath),
                          project_root / filepath,
                          project_root / "tmp" / filepath]:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"File not found: {filepath}")

    tmp_dir = project_root / "tmp"
    if tmp_dir.is_dir():
        files = sorted(
            tmp_dir.glob("prescribe_pipeline_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        # Skip files that already end in _light.json or _prompts.json
        files = [f for f in files
                 if not f.stem.endswith("_light")
                 and not f.stem.endswith("_prompts")]
        if files:
            return files[0].resolve()

    raise FileNotFoundError(
        "No prescribe_pipeline_*.json found in tmp/.  Pass an explicit path with --file."
    )


# ==============================================================================
# CORE SPLIT LOGIC
# ==============================================================================

def _clean_res(
    res_val: Dict[str, Any],
    keep_res_paths: bool = False,
    keep_entity_matrix: bool = False,
) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for k, v in res_val.items():
        if k == "entity_matrix" and not keep_entity_matrix:
            continue
        if not keep_res_paths and k in _RES_PATH_FIELDS:
            continue
        if k in _LLM_STEP_KEYS and isinstance(v, dict):
            cleaned[k] = {step_k: step_v for step_k, step_v in v.items() if step_k != "prompt"}
        else:
            cleaned[k] = v
    return cleaned


def split_records(
    data: List[Dict[str, Any]],
    keep_res_paths: bool = False,
    keep_entity_matrix: bool = False,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split pipeline records into (light_records, prompt_audit_records).

    light_records        – original record minus bulky prompts ('prompts',
                           'llmMedCheck.prompt', 'llmMedPrescribe.prompt')
                           and 'entity_matrix'; has 'prompts_file' set to the
                           relative path of the audit file.
    prompt_audit_records – minimal dicts {runId, caseNo, prompts} for each record.
    """
    light_records: List[Dict[str, Any]] = []
    prompt_audits: List[Dict[str, Any]] = []

    for rec in data:
        prompts = rec.get("prompts")

        # --- Build prompt audit entry ---
        audit_prompts: Dict[str, Any] = {}
        if isinstance(prompts, dict):
            audit_prompts.update(prompts)

        # Fallback / augment: grab prompts from res steps if available
        res_dict = rec.get("res")
        if isinstance(res_dict, dict):
            for step_key, prompt_name in [
                ("llmMedCheck", "step1_llm_med_check_prompt"),
                ("llmMedPrescribe", "step2_llm_med_prescribe_prompt"),
            ]:
                step_obj = res_dict.get(step_key)
                if isinstance(step_obj, dict) and step_obj.get("prompt"):
                    if prompt_name not in audit_prompts:
                        audit_prompts[prompt_name] = step_obj["prompt"]

        audit_entry: Dict[str, Any] = {
            "runId":   rec.get("runId"),
            "caseNo":  rec.get("caseNo"),
        }
        if audit_prompts:
            audit_entry["prompts"] = audit_prompts
        prompt_audits.append(audit_entry)

        # --- Build lightweight record ---
        light: Dict[str, Any] = {}
        for key, value in rec.items():
            if key == "prompts":
                # Will be replaced with a back-reference path below
                continue
            elif key == "entity_matrix" and not keep_entity_matrix:
                continue
            elif key in _LLM_STEP_KEYS and isinstance(value, dict):
                light[key] = {step_k: step_v for step_k, step_v in value.items() if step_k != "prompt"}
            elif key == "res" and isinstance(value, dict):
                light[key] = _clean_res(
                    value,
                    keep_res_paths=keep_res_paths,
                    keep_entity_matrix=keep_entity_matrix,
                )
            else:
                light[key] = value

        # Insert prompts_file placeholder (filled in by the caller once paths are known)
        light["prompts_file"] = None
        light_records.append(light)

    return light_records, prompt_audits


# ==============================================================================
# OUTPUT PATHS
# ==============================================================================

def derive_output_paths(input_path: Path) -> tuple[Path, Path]:
    """Return (light_path, prompts_path) sibling of input_path."""
    stem = input_path.stem
    suffix = input_path.suffix
    parent = input_path.parent
    light_path   = parent / f"{stem}_light{suffix}"
    prompts_path = parent / f"{stem}_prompts{suffix}"
    return light_path, prompts_path


# ==============================================================================
# SIZE REPORTING
# ==============================================================================

def human_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024.0:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024.0
    return f"{n_bytes:.1f} TB"


def report_sizes(label: str, original: int, new: int) -> None:
    saved    = original - new
    pct_saved = saved / original * 100 if original > 0 else 0
    arrow    = "↓" if new < original else "↑"
    print(
        f"  {label:<30}  {human_size(original):>10}  →  {human_size(new):>10}  "
        f"({arrow} {pct_saved:.1f}% {'saved' if new < original else 'larger'})"
    )


# ==============================================================================
# MAIN
# ==============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Strip bulky prompts (including res.llmMedCheck.prompt and "
            "res.llmMedPrescribe.prompt) and entity_matrix from a prescription pipeline "
            "JSON into a separate audit file, producing a lightweight result file."
        )
    )
    p.add_argument("positional_file", nargs="?", default=None,
                   help="Path to pipeline result JSON (positional).")
    p.add_argument("--file", "-f", default=None, metavar="PATH",
                   help="Path to pipeline result JSON.")
    p.add_argument("--keep-res-paths", action="store_true",
                   help="Keep res.workflow_prompts_file and res.llm_log_file in light output.")
    p.add_argument("--keep-entity-matrix", action="store_true",
                   help="Keep entity_matrix in light output.")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute sizes and report without writing any files.")
    p.add_argument("--indent", type=int, default=None, metavar="N",
                   help="JSON indent level for output files (default: compact / None).")
    return p


def main() -> None:
    args     = build_parser().parse_args()
    filepath = args.file or args.positional_file

    input_path = resolve_input_path(filepath)
    print(f"\nInput file : {input_path}")
    original_size = input_path.stat().st_size
    print(f"  Original size: {human_size(original_size)}\n")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of pipeline records.")

    n_records = len(data)
    print(f"  Records: {n_records}")

    # Count how many records have prompts
    def _has_prompts(r: Dict[str, Any]) -> bool:
        if r.get("prompts"):
            return True
        res = r.get("res")
        if isinstance(res, dict):
            if isinstance(res.get("llmMedCheck"), dict) and res["llmMedCheck"].get("prompt"):
                return True
            if isinstance(res.get("llmMedPrescribe"), dict) and res["llmMedPrescribe"].get("prompt"):
                return True
        return False

    n_with_prompts = sum(1 for r in data if _has_prompts(r))
    print(f"  Records with prompts: {n_with_prompts}\n")

    # ── Derive output paths ──────────────────────────────────────────────────
    light_path, prompts_path = derive_output_paths(input_path)

    # ── Split ────────────────────────────────────────────────────────────────
    light_records, prompt_audits = split_records(
        data,
        keep_res_paths=args.keep_res_paths,
        keep_entity_matrix=args.keep_entity_matrix,
    )

    # Fill in back-reference path (relative from input_path's directory)
    prompts_rel = os.path.relpath(prompts_path, input_path.parent)
    for rec in light_records:
        rec["prompts_file"] = prompts_rel

    # ── Serialise ────────────────────────────────────────────────────────────
    indent = args.indent  # None = compact (smallest file)

    light_json   = json.dumps(light_records,  ensure_ascii=False, indent=indent)
    prompts_json = json.dumps(prompt_audits, ensure_ascii=False, indent=indent)

    light_size   = len(light_json.encode("utf-8"))
    prompts_size = len(prompts_json.encode("utf-8"))

    print("─" * 70)
    print(f"  {'File':<30}  {'Original':>10}     {'New':>10}  Change")
    print("─" * 70)
    report_sizes(input_path.name,   original_size, light_size)
    report_sizes("(prompts audit)",             0, prompts_size)
    report_sizes("Total (light+audit)", original_size, light_size + prompts_size)
    print("─" * 70)

    if args.dry_run:
        print("\nDry-run mode – no files written.")
        return

    with open(light_path, "w", encoding="utf-8") as f:
        f.write(light_json)
    print(f"\n  Light file written  -> {light_path}")

    with open(prompts_path, "w", encoding="utf-8") as f:
        f.write(prompts_json)
    print(f"  Audit file written  -> {prompts_path}")

    print(
        f"\nDone.  Saved {human_size(original_size - light_size)} "
        f"({(original_size - light_size)/original_size*100:.1f}%) in light file.\n"
    )


if __name__ == "__main__":
    main()
