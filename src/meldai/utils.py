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
