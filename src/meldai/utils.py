"""Utility methods for MeldAI medical text processing and symptom parsing."""

import re


def segment(note: str) -> list[str]:
    """Split a string of symptoms or clinical notes into a list of segmented strings.

    Handles bullet points (●), sentence boundaries, and commas/semicolons that are not
    enclosed within parentheses.
    """
    note = note.replace("●", "\n")
    out = []
    for line in note.splitlines():
        out += re.split(
            r"(?<=\.)\s+"  # sentence end
            r"|[,;](?![^()]*\))",  # comma/semicolon NOT inside parentheses
            line,
        )
    return [s.strip(" .") for s in out if s.strip(" .")]
