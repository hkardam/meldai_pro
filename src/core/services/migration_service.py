"""Compatibility shim for migration_service — relocated to core.migration."""

from core.migration.migration_service import (
    MigrationService,
    clean_medicine_token,
    search_medicine_molecules,
)

__all__ = [
    "MigrationService",
    "clean_medicine_token",
    "search_medicine_molecules",
]
