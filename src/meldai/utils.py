"""Utility methods for MeldAI medical text processing and symptom parsing."""

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

