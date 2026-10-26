from __future__ import annotations

from typing import Any
import unicodedata

KEEP = set("%+/?-().,:;")  # clinically meaningful: 60%, ++, ?GTCS, D/d, etc.


def sanitize(text: str) -> str:
    """Sanitize text by normalizing unicode and stripping leading symbol/bullet glyphs and whitespace."""
    text = unicodedata.normalize("NFKC", text)
    lines = []
    for line in text.splitlines():
        i = 0
        # strip leading symbol/bullet glyphs and whitespace, but keep meaningful ones
        while i < len(line) and (
            line[i].isspace()
            or (line[i] not in KEEP and unicodedata.category(line[i]).startswith(("S", "P")))
        ):
            i += 1
        lines.append(line[i:].rstrip())
    return "\n".join(l for l in lines if l)


def segment(note: str) -> list[str]:
    """Split a string of symptoms or clinical notes into a list of segmented strings.

    Splits only based on newline characters and bullet points (●), and sanitizes each segment.
    Phenotype classification is handled downstream by MedspaCyService.
    """
    note = note.replace("●", "\n")
    results = []
    for line in note.splitlines():
        sanitized_item = sanitize(line).strip(" .")
        if sanitized_item:
            results.append(sanitized_item)
    return results


def to_curie(prefix: str, identifier: str | int | None) -> str | None:
    """Format ontology IDs as 7-digit zero-padded CURIEs (e.g. HP:0002189, MONDO:0008807).

    Normalizes 'HPO:2189' -> 'HP:0002189', 13600 -> 'HP:0013600', 'HP:5200289' -> 'HP:5200289'.
    If prefix is 'HPO' or 'HP', canonical prefix 'HP' is used.
    """
    if identifier is None:
        return None
    raw = str(identifier).strip()
    if not raw or raw.lower() in ("none", "null", "unmatched"):
        return None

    import re
    canonical_prefix = "HP" if prefix.upper() in ("HP", "HPO") else prefix.upper()
    m = re.search(r"(\d+)", raw)
    if not m:
        return None

    digits = m.group(1)
    if len(digits) <= 7:
        formatted_digits = digits.zfill(7)
    else:
        formatted_digits = digits

    return f"{canonical_prefix}:{formatted_digits}"


def generate_run_id(case_no: str | int | None = None) -> str:
    """Generate a unique run ID for prompt/response tracing.

    Format: case_<case_no>_<YYYYMMDD_HHMMSS>_<short_uuid> or run_<YYYYMMDD_HHMMSS>_<short_uuid>.
    """
    import uuid
    from datetime import datetime

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    if case_no is not None:
        return f"case_{case_no}_{ts}_{short_uuid}"
    return f"run_{ts}_{short_uuid}"


def save_prompt_to_tmp(
    prompt: str,
    run_id: str | None = None,
    case_no: str | int | None = None,
    directory: str | None = None,
) -> str:
    """Save the executed prompt to <id>_prompt.txt in a tmp directory.

    Args:
        prompt: Full prompt string.
        run_id: Unique run ID (if None, generated via generate_run_id).
        case_no: Optional case number identifier.
        directory: Optional directory path (defaults to repo_root / 'tmp').

    Returns:
        Absolute string path to the saved prompt file.
    """
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)

    if directory is not None:
        target_dir = Path(directory)
    else:
        repo_root = Path(__file__).resolve().parent.parent.parent
        target_dir = repo_root / "tmp"

    target_dir.mkdir(parents=True, exist_ok=True)

    rid = run_id or generate_run_id(case_no=case_no)
    filename = f"{rid}_prompt.txt"
    file_path = target_dir / filename

    text_to_save = prompt if isinstance(prompt, str) else str(prompt or "")
    try:
        file_path.write_text(text_to_save, encoding="utf-8")
        logger.info("Saved executed prompt (%d chars) to %s", len(text_to_save), file_path)
    except Exception as exc:
        logger.warning("Failed to save prompt to %s: %s", file_path, exc)

    return str(file_path)


def save_response_to_tmp(
    response_text: str | Any,
    run_id: str,
    directory: str | None = None,
) -> str:
    """Save the raw LLM response to <id>_response.txt in a tmp directory.

    Args:
        response_text: Full raw response string or payload returned by LLM.
        run_id: Unique run ID matching the prompt file.
        directory: Optional directory path (defaults to repo_root / 'tmp').

    Returns:
        Absolute string path to the saved response file.
    """
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)

    if directory is not None:
        target_dir = Path(directory)
    else:
        repo_root = Path(__file__).resolve().parent.parent.parent
        target_dir = repo_root / "tmp"

    target_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{run_id}_response.txt"
    file_path = target_dir / filename

    text_to_save = response_text if isinstance(response_text, str) else str(response_text or "")
    try:
        file_path.write_text(text_to_save, encoding="utf-8")
        logger.info("Saved raw LLM response (%d chars) to %s", len(text_to_save), file_path)
    except Exception as exc:
        logger.warning("Failed to save response to %s: %s", file_path, exc)

    return str(file_path)


def parse_clinical_date(date_val: Any) -> Optional[datetime]:
    """Robustly parse clinical encounter dates.

    Handles:
      - 'DD/MM/YYYY' (e.g. '08/10/2025', '12/09/2025')
      - 'YYYY-MM-DD' (e.g. '2025-10-08')
      - datetime / date objects
      - other standard date strings with dateutil.parser (dayfirst=True)
    """
    if not date_val:
        return None
    from datetime import date, datetime
    if isinstance(date_val, datetime):
        return date_val
    if isinstance(date_val, date):
        return datetime.combine(date_val, datetime.min.time())

    s = str(date_val).strip()
    if not s or s.lower() in ("none", "null", "unknown"):
        return None

    try:
        import re
        from dateutil import parser
        # Check standard DD/MM/YYYY or D/M/YYYY
        if re.match(r"^\d{1,2}/\d{1,2}/\d{4}", s):
            return parser.parse(s, dayfirst=True)
        # Check standard YYYY-MM-DD
        if re.match(r"^\d{4}-\d{1,2}-\d{1,2}", s):
            return parser.parse(s, dayfirst=False)
        return parser.parse(s, dayfirst=True)
    except Exception:
        return None


def clean_diagnosis_term(term: str | None) -> str:
    """Clean a diagnosis string by removing any special characters before a letter starts

    and after a letter ends, then trimming whitespace.
    Applied strictly to diagnosis entities only.

    Example:
        "? mdd with anxious distress" -> "mdd with anxious distress"
        "- ? mdd with anxious distress . " -> "mdd with anxious distress"
    """
    if not term or not isinstance(term, str):
        return ""
    import re
    # Match from the first letter (a-zA-Z) to the last letter (a-zA-Z)
    m = re.search(r"[a-zA-Z].*[a-zA-Z]|[a-zA-Z]", term)
    if m:
        return m.group(0).strip()
    return ""

