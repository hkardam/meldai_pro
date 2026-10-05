"""Background service runner for patient chief complaints migration with start, stop, and status control."""

import logging
import threading
import time
from typing import Any, Dict, Optional

from meldai.services.migration_service import MigrationService

logger = logging.getLogger(__name__)


class ChiefComplaintsRunner:
    """Manages asynchronous execution, cancellation, and status tracking for chief complaints migration."""

    def __init__(self, migration_service: Optional[MigrationService] = None) -> None:
        self._migration_service = migration_service
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._state: Dict[str, Any] = {
            "status": "idle",  # "idle" | "running" | "stopping" | "completed" | "failed" | "stopped"
            "batch_size": 100,
            "current_batch": 0,
            "total_batches": 0,
            "total_rows_processed": 0,
            "documents_updated": 0,
            "documents_skipped": 0,
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

    def start(self, batch_size: int = 100, max_workers: Optional[int] = None) -> Dict[str, Any]:
        """Start migration in a background thread. Returns 409 error if already active."""
        with self._lock:
            if self._state["status"] in ("running", "stopping"):
                return {
                    "status": "conflict",
                    "message": f"Migration is already in progress with status '{self._state['status']}'.",
                    "current_state": dict(self._state),
                }

            self._stop_event.clear()
            self._state = {
                "status": "running",
                "batch_size": batch_size,
                "current_batch": 0,
                "total_batches": 0,
                "total_rows_processed": 0,
                "documents_updated": 0,
                "documents_skipped": 0,
                "current_step": "initializing",
                "start_time": time.time(),
                "elapsed_seconds": 0.0,
                "error": None,
            }

            self._thread = threading.Thread(
                target=self._execute,
                args=(batch_size, max_workers),
                name="ChiefComplaintsMigrationThread",
                daemon=True,
            )
            self._thread.start()

            return {
                "status": "started",
                "message": "Chief complaints migration started in background.",
                "batch_size": batch_size,
            }

    def stop(self) -> Dict[str, Any]:
        """Request graceful cancellation of the running migration."""
        with self._lock:
            if self._state["status"] not in ("running",):
                return {
                    "status": self._state["status"],
                    "message": f"Cannot stop migration: current status is '{self._state['status']}'.",
                }

            self._stop_event.set()
            self._state["status"] = "stopping"
            self._state["current_step"] = "stopping_requested"
            return {
                "status": "stopping",
                "message": "Stop signal sent. Runner will cleanly halt after finishing current batch.",
            }

    def _execute(self, batch_size: int, max_workers: Optional[int]) -> None:
        """Worker thread loop executing optimized chief complaints migration."""
        svc = self._get_migration_service()
        try:
            def _progress_callback(update_dict: Dict[str, Any]) -> bool:
                # Returns True to continue, False if stop requested
                with self._lock:
                    self._state.update(update_dict)
                    if self._state["start_time"]:
                        self._state["elapsed_seconds"] = round(time.time() - self._state["start_time"], 2)
                return not self._stop_event.is_set()

            result = svc.load_chief_complaints(
                batch_size=batch_size,
                max_workers=max_workers,
                progress_callback=_progress_callback,
            )

            with self._lock:
                if self._stop_event.is_set():
                    self._state["status"] = "stopped"
                    self._state["current_step"] = "stopped"
                else:
                    self._state["status"] = "completed"
                    self._state["current_step"] = "completed"
                self._state["total_rows_processed"] = result.get("total_rows_processed", self._state["total_rows_processed"])
                self._state["documents_updated"] = result.get("documents_updated", self._state["documents_updated"])
                self._state["documents_skipped"] = result.get("documents_skipped", self._state["documents_skipped"])
                self._state["current_batch"] = result.get("batches_processed", self._state["current_batch"])
                if self._state["start_time"]:
                    self._state["elapsed_seconds"] = round(time.time() - self._state["start_time"], 2)
        except Exception as exc:
            logger.error("Background chief complaints migration failed: %s", exc, exc_info=True)
            with self._lock:
                self._state["status"] = "failed"
                self._state["current_step"] = "failed"
                self._state["error"] = str(exc)
                if self._state["start_time"]:
                    self._state["elapsed_seconds"] = round(time.time() - self._state["start_time"], 2)


_runner_instance: Optional[ChiefComplaintsRunner] = None
_runner_lock = threading.Lock()


def get_chief_complaints_runner() -> ChiefComplaintsRunner:
    """Return the singleton instance of ChiefComplaintsRunner."""
    global _runner_instance
    if _runner_instance is None:
        with _runner_lock:
            if _runner_instance is None:
                _runner_instance = ChiefComplaintsRunner()
    return _runner_instance
