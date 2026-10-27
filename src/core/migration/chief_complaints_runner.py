"""Background service runner for unified patient migration with start, stop, and status control."""

import logging
import threading
import time
from typing import Any, Dict, Optional

from core.migration.migration_service import MigrationService

logger = logging.getLogger(__name__)


class PatientMigrationRunner:
    """Manages asynchronous execution, cancellation, and status tracking for patient migration."""

    def __init__(self, migration_service: Optional[MigrationService] = None) -> None:
        self._migration_service = migration_service
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._state: Dict[str, Any] = {
            "status": "idle",  # "idle" | "running" | "stopping" | "completed" | "failed" | "stopped"
            "batch_size": 20,
            "current_batch": 0,
            "total_batches": 0,
            "total_records_processed": 0,
            "documents_saved": 0,
            "current_step": "idle",
            "start_time": None,
            "elapsed_seconds": 0.0,
            "error": None,
        }

    def _get_migration_service(self) -> MigrationService:
        if self._migration_service is not None:
            return self._migration_service
        return MigrationService()

    def get_status(self) -> Dict[str, Any]:
        """Return the current execution status and progress metrics."""
        with self._lock:
            state = dict(self._state)
            if state["status"] in ("running", "stopping") and state["start_time"]:
                state["elapsed_seconds"] = round(time.time() - state["start_time"], 2)
            return state

    def start(self, batch_size: int = 20, limit: Optional[int] = None) -> Dict[str, Any]:
        """Start migration in a background thread. Returns 409 error if already active."""
        with self._lock:
            if self._state["status"] in ("running", "stopping"):
                return {
                    "status": "conflict",
                    "message": f"Patient migration is already in progress with status '{self._state['status']}'.",
                    "current_state": dict(self._state),
                }

            self._stop_event.clear()
            self._state = {
                "status": "running",
                "batch_size": batch_size,
                "current_batch": 0,
                "total_batches": 0,
                "total_records_processed": 0,
                "documents_saved": 0,
                "current_step": "start",
                "start_time": time.time(),
                "elapsed_seconds": 0.0,
                "error": None,
            }

            self._thread = threading.Thread(
                target=self._execute,
                args=(batch_size, limit),
                name="PatientMigrationThread",
                daemon=True,
            )
            self._thread.start()

            return {
                "status": "started",
                "message": "Patient migration started in background.",
                "batch_size": batch_size,
            }

    def stop(self) -> Dict[str, Any]:
        """Request graceful cancellation of the running migration."""
        with self._lock:
            if self._state["status"] not in ("running",):
                return {
                    "status": self._state["status"],
                    "message": f"Cannot stop patient migration: current status is '{self._state['status']}'.",
                }

            self._stop_event.set()
            self._state["status"] = "stopping"
            self._state["current_step"] = "stopping_requested"
            return {
                "status": "stopping",
                "message": "Stop signal sent. Runner will cleanly halt after finishing current batch.",
            }

    def _execute(self, batch_size: int, limit: Optional[int]) -> None:
        """Worker thread loop executing unified patient migration."""
        svc = self._get_migration_service()
        try:
            def _progress_callback(update_dict: Dict[str, Any]) -> bool:
                with self._lock:
                    self._state.update(update_dict)
                    if self._state["start_time"]:
                        self._state["elapsed_seconds"] = round(time.time() - self._state["start_time"], 2)
                return not self._stop_event.is_set()

            result = svc.run_patient_migration(
                batch_size=batch_size,
                limit=limit,
                progress_callback=_progress_callback,
            )

            with self._lock:
                if self._stop_event.is_set():
                    self._state["status"] = "stopped"
                    self._state["current_step"] = "stopped"
                else:
                    self._state["status"] = "completed"
                    self._state["current_step"] = "completed"
                self._state["total_records_processed"] = result.get("total_records_processed", self._state["total_records_processed"])
                self._state["documents_saved"] = result.get("upserted_count", 0) + result.get("modified_count", 0)
                self._state["current_batch"] = result.get("batches_processed", self._state["current_batch"])
                if self._state["start_time"]:
                    self._state["elapsed_seconds"] = round(time.time() - self._state["start_time"], 2)
        except Exception as exc:
            logger.error("Background patient migration failed: %s", exc, exc_info=True)
            with self._lock:
                self._state["status"] = "failed"
                self._state["current_step"] = "failed"
                self._state["error"] = str(exc)
                if self._state["start_time"]:
                    self._state["elapsed_seconds"] = round(time.time() - self._state["start_time"], 2)


_runner_instance: Optional[PatientMigrationRunner] = None
_runner_lock = threading.Lock()


def get_patient_migration_runner() -> PatientMigrationRunner:
    """Return singleton instance of PatientMigrationRunner."""
    global _runner_instance
    if _runner_instance is None:
        with _runner_lock:
            if _runner_instance is None:
                _runner_instance = PatientMigrationRunner()
    return _runner_instance


# Backward-compatibility alias
ChiefComplaintsRunner = PatientMigrationRunner
get_chief_complaints_runner = get_patient_migration_runner
