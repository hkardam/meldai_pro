"""Background pipeline runner that iterates over all cases in prescribe-request-all-cases.json,
calls prescribe_medications for each, and streams results into
tmp/prescribe_pipeline_<timestamp>.json one entry at a time.

Only one pipeline may run at a time. Stopping the pipeline does NOT discard the result file —
all previously written entries remain valid and the JSON array is properly closed.
"""

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from meldai.api.req_dtos import PrescribeMedicationsRequest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ALL_CASES_PATH = Path(__file__).resolve().parents[3] / "data" / "cases" / "prescribe-request-all-cases.json"
TMP_DIR = Path(__file__).resolve().parents[3] / "tmp"


def _make_result_path() -> Path:
    """Generate a timestamped result file path: tmp/prescribe_pipeline_<YYYYMMDD_HHMMSS>.json"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return TMP_DIR / f"prescribe_pipeline_{ts}.json"


# ---------------------------------------------------------------------------
# Streaming JSON-array writer
# ---------------------------------------------------------------------------

class _StreamingJsonArrayWriter:
    """
    Streams a JSON array to a file entry-by-entry without buffering all results in memory.

    Usage:
        with _StreamingJsonArrayWriter(path) as writer:
            writer.append(entry)      # called after each case
        # file is properly closed with ] on context exit (even on exception / stop)
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._file = None
        self._count = 0

    def __enter__(self) -> "_StreamingJsonArrayWriter":
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", encoding="utf-8")
        self._file.write("[\n")
        self._file.flush()
        return self

    def append(self, entry: Any) -> None:
        if self._file is None or self._file.closed:
            return
        prefix = "  " if self._count == 0 else ",\n  "
        self._file.write(prefix + json.dumps(entry, indent=None, default=str))
        self._file.flush()
        self._count += 1

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._file and not self._file.closed:
            self._file.write("\n]\n")
            self._file.flush()
            self._file.close()
        return False  # do not suppress exceptions


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class PrescribePipelineRunner:
    """
    Runs prescribe_medications for every case in prescribe-request-all-cases.json
    sequentially in a background thread. Only one run is allowed at a time.

    Result file: tmp/prescribe_pipeline_<timestamp>.json
    Each entry is flushed to disk immediately after the case completes.
    Stopping the pipeline closes the JSON array cleanly — no data is lost.

    Entry schema:
        {
          "runId":        str,
          "req":          dict,          # original request payload
          "expectedMeds": list,          # from the cases file
          "res":          dict | null    # PrescribeMedicationsResponse or {"error": ..., "case_no": ...}
        }
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._state: Dict[str, Any] = {
            "status": "idle",          # idle | running | stopping | completed | failed | stopped
            "run_id": None,
            "total_cases": 0,
            "processed_cases": 0,
            "passed_cases": 0,
            "failed_cases": 0,
            "current_case_no": None,
            "result_file": None,
            "start_time": None,
            "elapsed_seconds": 0.0,
            "error": None,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of the current pipeline state."""
        with self._lock:
            state = dict(self._state)
            if state["status"] in ("running", "stopping") and state["start_time"]:
                state["elapsed_seconds"] = round(time.time() - state["start_time"], 2)
            return state

    def start(self) -> Dict[str, Any]:
        """
        Kick off the pipeline in a background thread.
        Returns a conflict dict if already running or stopping.
        """
        with self._lock:
            if self._state["status"] in ("running", "stopping"):
                return {
                    "status": "conflict",
                    "message": f"Pipeline is already in progress with status '{self._state['status']}'.",
                }

            run_id = uuid.uuid4().hex[:12]
            result_path = _make_result_path()

            self._stop_event.clear()
            self._state = {
                "status": "running",
                "run_id": run_id,
                "total_cases": 0,
                "processed_cases": 0,
                "passed_cases": 0,
                "failed_cases": 0,
                "current_case_no": None,
                "result_file": str(result_path),
                "start_time": time.time(),
                "elapsed_seconds": 0.0,
                "error": None,
            }

        self._thread = threading.Thread(
            target=self._execute,
            args=(run_id, result_path),
            name="PrescribePipelineThread",
            daemon=True,
        )
        self._thread.start()

        return {
            "status": "started",
            "message": "Prescribe pipeline started in background.",
            "run_id": run_id,
            "result_file": str(result_path),
        }

    def stop(self) -> Dict[str, Any]:
        """Signal the running pipeline to stop after the current case finishes."""
        with self._lock:
            if self._state["status"] != "running":
                return {
                    "status": self._state["status"],
                    "message": f"Cannot stop pipeline: current status is '{self._state['status']}'.",
                }
            self._stop_event.set()
            self._state["status"] = "stopping"
            return {
                "status": "stopping",
                "message": "Stop signal sent. Pipeline will halt after the current case. Results written so far are preserved.",
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _execute(self, run_id: str, result_path: Path) -> None:
        """
        Worker thread.
        Opens the result file as a streaming JSON array, then processes cases one by one.
        The array is closed (with `]`) on any exit path — including stop or exception.
        """
        # Late import to avoid circular imports at module load time
        from meldai.services.postgres_prescription_service import PostgresPrescriptionService

        try:
            raw_cases: List[Dict[str, Any]] = json.loads(
                ALL_CASES_PATH.read_text(encoding="utf-8")
            )
        except Exception as exc:
            logger.error("PrescribePipeline: failed to load cases file — %s", exc, exc_info=True)
            with self._lock:
                self._state["status"] = "failed"
                self._state["error"] = f"Failed to load cases file: {exc}"
                self._state["elapsed_seconds"] = round(time.time() - self._state["start_time"], 2)
            return

        with self._lock:
            self._state["total_cases"] = len(raw_cases)

        svc = PostgresPrescriptionService()

        # Open the streaming writer — the `]` is written on __exit__ regardless of how we leave
        with _StreamingJsonArrayWriter(result_path) as writer:
            try:
                for idx, case_entry in enumerate(raw_cases):
                    # Honour stop signal between cases
                    if self._stop_event.is_set():
                        break

                    req_payload = case_entry.get("req", {})
                    expected_meds = case_entry.get("expectedMeds", [])
                    case_no = req_payload.get("case", {}).get("caseNo")

                    with self._lock:
                        self._state["current_case_no"] = case_no

                    res_data: Dict[str, Any]

                    try:
                        request_obj = PrescribeMedicationsRequest(**req_payload)
                        response = svc.prescribe_medications(request_obj)
                        res_data = response.model_dump(exclude_none=True)

                        with self._lock:
                            self._state["passed_cases"] += 1

                    except Exception as exc:
                        logger.warning(
                            "PrescribePipeline: case %s failed — %s", case_no, exc, exc_info=False
                        )
                        res_data = {"error": str(exc), "case_no": case_no}

                        with self._lock:
                            self._state["failed_cases"] += 1

                    # Stream this entry immediately to disk
                    writer.append(
                        {
                            "runId": run_id,
                            "req": req_payload,
                            "expectedMeds": expected_meds,
                            "res": res_data,
                        }
                    )

                    with self._lock:
                        self._state["processed_cases"] = idx + 1
                        self._state["elapsed_seconds"] = round(
                            time.time() - self._state["start_time"], 2
                        )

                # Determine final status
                with self._lock:
                    if self._stop_event.is_set():
                        self._state["status"] = "stopped"
                    else:
                        self._state["status"] = "completed"
                    self._state["current_case_no"] = None
                    self._state["elapsed_seconds"] = round(
                        time.time() - self._state["start_time"], 2
                    )

            except Exception as exc:
                logger.error("PrescribePipeline fatal error: %s", exc, exc_info=True)
                with self._lock:
                    self._state["status"] = "failed"
                    self._state["error"] = str(exc)
                    self._state["current_case_no"] = None
                    self._state["elapsed_seconds"] = round(
                        time.time() - self._state["start_time"], 2
                    )
            # _StreamingJsonArrayWriter.__exit__ closes the JSON array here, always


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_runner_instance: Optional[PrescribePipelineRunner] = None
_runner_lock = threading.Lock()


def get_prescribe_pipeline_runner() -> PrescribePipelineRunner:
    """Return the process-level singleton PrescribePipelineRunner."""
    global _runner_instance
    if _runner_instance is None:
        with _runner_lock:
            if _runner_instance is None:
                _runner_instance = PrescribePipelineRunner()
    return _runner_instance
