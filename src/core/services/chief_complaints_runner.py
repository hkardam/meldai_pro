"""Compatibility shim for chief_complaints_runner — relocated to core.migration."""

from core.migration.chief_complaints_runner import (
    ChiefComplaintsRunner,
    PatientMigrationRunner,
    get_chief_complaints_runner,
    get_patient_migration_runner,
)

__all__ = [
    "PatientMigrationRunner",
    "get_patient_migration_runner",
    "ChiefComplaintsRunner",
    "get_chief_complaints_runner",
]
