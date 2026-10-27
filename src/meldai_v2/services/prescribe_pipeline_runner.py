"""Background pipeline runner V2 that iterates over all 95 cases in prescribe-request-all-cases.json,
executes the full multi-step LLM reasoning flow via PrescriptionServiceV2,
and streams results into tmp/prescribe_pipeline_v2_<timestamp>.json.

Key Features:
  - Exact same 95 unique cases from data/cases/prescribe-request-all-cases.json
  - Multi-step LLM reasoning (Triage -> llmMedCheck -> llmMedPrescribe)
  - Records prompts, req, and res (with LLM explanation & enriched medications)
  - Output is 100% compatible with accuracy_matrix.py
  - Exponential backoff on rate-limits (HTTP 429 / ResourceExhausted)
  - Safe streaming JSON writer: stops cleanly on interruption without data loss
"""

import json
import logging
from pathlib import Path
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from meldai_v2.req_dtos import PrescribeMedicationsV2Request

logger = logging.getLogger(__name__)


def _find_repo_root() -> Path:
    """Dynamically determine repository root by finding project anchors."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "data" / "cases" / "prescribe-request-all-cases.json").is_file():
            return parent
    return Path(__file__).resolve().parents[4]


REPO_ROOT = _find_repo_root()
ALL_CASES_PATH = REPO_ROOT / "data" / "cases" / "prescribe-request-all-cases.json"
TMP_DIR = REPO_ROOT / "tmp"


def _make_result_path() -> Path:
    """Generate a timestamped result file path: tmp/prescribe_pipeline_v2_<YYYYMMDD_HHMMSS>.json"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return TMP_DIR / f"prescribe_pipeline_v2_{ts}.json"


# ---------------------------------------------------------------------------
# Streaming JSON-array writer
# ---------------------------------------------------------------------------

class _StreamingJsonArrayWriter:
    """Streams a JSON array to a file entry-by-entry without buffering all results in memory."""

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
        return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class PrescribePipelineV2Runner:
    """Runs PrescriptionServiceV2 for every case in prescribe-request-all-cases.json
    sequentially in a background thread. Only one run is allowed at a time.

    Result file: tmp/prescribe_pipeline_v2_<timestamp>.json
    Each entry is flushed to disk immediately after the case completes.

    Entry schema:
        {
          "runId":        str,
          "caseNo":       int | null,
          "req":          dict,          # original request payload
          "expectedMeds": list,          # from the cases file
          "prompts":      dict,          # multi-step LLM prompts used
          "res":          dict | null    # PrescribeMedicationsV2Response or {"error": ..., "case_no": ...}
        }
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._state: Dict[str, Any] = {
            "status": "idle",          # idle | running | stopping | completed | failed | stopped
            "run_id": None,
            "version": "v2",
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
        """Kick off the V2 pipeline in a background thread."""
        with self._lock:
            if self._state["status"] in ("running", "stopping"):
                return {
                    "status": "conflict",
                    "message": f"Pipeline V2 is already in progress with status '{self._state['status']}'.",
                }

            run_id = uuid.uuid4().hex[:12]
            result_path = _make_result_path()

            self._stop_event.clear()
            self._state = {
                "status": "running",
                "run_id": run_id,
                "version": "v2",
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
            name="PrescribePipelineV2Thread",
            daemon=True,
        )
        self._thread.start()

        return {
            "status": "started",
            "message": "Prescribe pipeline V2 started in background.",
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
                "message": "Stop signal sent. Pipeline V2 will halt after the current case. Results written so far are preserved.",
            }

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    def _execute(self, run_id: str, result_path: Path) -> None:
        """Worker thread processing 95 cases through PrescriptionServiceV2 sequentially."""
        from meldai_v2.services.prescription_service import PrescriptionServiceV2

        try:
            raw_cases: List[Dict[str, Any]] = json.loads(
                ALL_CASES_PATH.read_text(encoding="utf-8")
            )
        except Exception as exc:
            logger.error("PrescribePipelineV2: failed to load cases file — %s", exc, exc_info=True)
            with self._lock:
                self._state["status"] = "failed"
                self._state["error"] = f"Failed to load cases file: {exc}"
                self._state["elapsed_seconds"] = round(time.time() - self._state["start_time"], 2)
            return

        with self._lock:
            self._state["total_cases"] = len(raw_cases)

        logger.info(
            "PrescribePipelineV2: initializing PrescriptionServiceV2 for %d cases...",
            len(raw_cases),
        )
        try:
            svc = PrescriptionServiceV2()
        except Exception as exc:
            logger.error("PrescribePipelineV2: failed to initialize service — %s", exc, exc_info=True)
            with self._lock:
                self._state["status"] = "failed"
                self._state["error"] = f"Failed to initialize PrescriptionServiceV2: {exc}"
                self._state["elapsed_seconds"] = round(time.time() - self._state["start_time"], 2)
            return

        with _StreamingJsonArrayWriter(result_path) as writer:
            try:
                for idx, case_entry in enumerate(raw_cases):
                    if self._stop_event.is_set():
                        break

                    req_payload = case_entry.get("req", {})
                    expected_meds = case_entry.get("expectedMeds", [])
                    case_no = req_payload.get("case", {}).get("caseNo")

                    with self._lock:
                        self._state["current_case_no"] = case_no

                    res_data: Dict[str, Any]
                    prompts_data: Optional[Dict[str, Any]] = None

                    # Execution with exponential backoff on transient quota/rate limit errors
                    max_retries = 3
                    retry_delay = 5.0
                    for attempt in range(max_retries):
                        try:
                            request_obj = PrescribeMedicationsV2Request(**req_payload)
                            response = svc.prescribe_medications_v2(request_obj)
                            res_data = response.model_dump(exclude_none=True)

                            prompts_data = {
                                "step1_llm_med_check_prompt": (
                                    response.llmMedCheck.prompt if response.llmMedCheck else None
                                ),
                                "step2_llm_med_prescribe_prompt": (
                                    response.llmMedPrescribe.prompt if response.llmMedPrescribe else None
                                ),
                                "workflow_prompts_file": response.workflow_prompts_file,
                                "llm_log_file": response.llm_log_file,
                            }

                            with self._lock:
                                self._state["passed_cases"] += 1
                            break

                        except Exception as exc:
                            err_str = str(exc)
                            is_rate_limit = any(
                                token in err_str.lower()
                                for token in ("429", "resourceexhausted", "quota", "rate limit")
                            )
                            if is_rate_limit and attempt < max_retries - 1:
                                sleep_secs = retry_delay * (2 ** attempt)
                                logger.warning(
                                    "PrescribePipelineV2: rate limit on case %s (attempt %d/%d). Sleeping %.1fs...",
                                    case_no, attempt + 1, max_retries, sleep_secs,
                                )
                                time.sleep(sleep_secs)
                                continue

                            logger.warning(
                                "PrescribePipelineV2: case %s failed — %s", case_no, exc, exc_info=False
                            )
                            res_data = {"error": err_str, "case_no": case_no}
                            with self._lock:
                                self._state["failed_cases"] += 1
                            break

                    # Stream this entry immediately to disk
                    writer.append(
                        {
                            "runId": run_id,
                            "caseNo": case_no,
                            "req": req_payload,
                            "expectedMeds": expected_meds,
                            "prompts": prompts_data,
                            "res": res_data,
                        }
                    )

                    with self._lock:
                        self._state["processed_cases"] = idx + 1
                        self._state["elapsed_seconds"] = round(
                            time.time() - self._state["start_time"], 2
                        )

                    # Pacing interval between cases to avoid API bursts
                    time.sleep(0.2)

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
                logger.error("PrescribePipelineV2 fatal error: %s", exc, exc_info=True)
                with self._lock:
                    self._state["status"] = "failed"
                    self._state["error"] = str(exc)
                    self._state["current_case_no"] = None
                    self._state["elapsed_seconds"] = round(
                        time.time() - self._state["start_time"], 2
                    )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_runner_v2_instance: Optional[PrescribePipelineV2Runner] = None
_runner_v2_lock = threading.Lock()


def get_prescribe_pipeline_v2_runner() -> PrescribePipelineV2Runner:
    """Return the process-level singleton PrescribePipelineV2Runner."""
    global _runner_v2_instance
    if _runner_v2_instance is None:
        with _runner_v2_lock:
            if _runner_v2_instance is None:
                _runner_v2_instance = PrescribePipelineV2Runner()
    return _runner_v2_instance
