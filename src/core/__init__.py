"""Core infrastructure, database drivers, NLP services, and terminology."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.config import Settings, get_settings
    from core.db import MongoKnowledgeBase, PostgresSource
    from core.nlp import BioLORDEmbedder, MedspaCyService, SapBERTEmbedder
    from core.router import core_router
    from core.terminology import HPOService, MONDOService
    from core.migration import MigrationService, PatientMigrationRunner

__all__ = [
    "Settings",
    "get_settings",
    "PostgresSource",
    "MongoKnowledgeBase",
    "SapBERTEmbedder",
    "BioLORDEmbedder",
    "MedspaCyService",
    "HPOService",
    "MONDOService",
    "MigrationService",
    "PatientMigrationRunner",
    "core_router",
]


def __getattr__(name: str):
    if name in ("Settings", "get_settings"):
        import core.config as _cfg
        return getattr(_cfg, name)
    if name in ("PostgresSource", "MongoKnowledgeBase"):
        import core.db as _db
        return getattr(_db, name)
    if name in ("SapBERTEmbedder", "BioLORDEmbedder", "MedspaCyService"):
        import core.nlp as _nlp
        return getattr(_nlp, name)
    if name == "core_router":
        from core.router import core_router
        return core_router
    if name in ("HPOService", "MONDOService"):
        import core.terminology as _term
        return getattr(_term, name)
    if name in ("MigrationService", "PatientMigrationRunner"):
        import core.migration as _mig
        return getattr(_mig, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

