"""Gemini LLM Service for medical text and prescription generation."""

from __future__ import annotations

import contextvars
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any
import uuid

from google import genai
from google.genai import types

from core.config import Settings, get_settings

logger = logging.getLogger(__name__)

_current_run_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_run_id_ctx", default=None)
_current_call_name_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("current_call_name_ctx", default=None)


def set_llm_run_context(run_id: str | None, call_name: str | None = None) -> None:
    """Set the active run_id and call_name in contextvars for downstream LLM calls."""
    _current_run_id_ctx.set(run_id)
    _current_call_name_ctx.set(call_name)


def get_llm_run_context() -> tuple[str | None, str | None]:
    """Get active (run_id, call_name) from contextvars."""
    return _current_run_id_ctx.get(), _current_call_name_ctx.get()


def extract_base_run_id(run_id: str) -> str:
    """Normalize run_id to group all calls of a run into the same log file."""
    cleaned = re.sub(
        r"_(?:step\d+(?:_response|_prompt)?|response|prompt|triage|clinical_summary|symptom_notes)$",
        "",
        run_id,
    )
    return cleaned if cleaned else run_id


def get_run_log_file_paths(run_id: str) -> tuple[Path, Path]:
    """Get the log file paths in logs/ and tmp/ for a given run_id."""
    base_id = extract_base_run_id(run_id)
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    logs_dir = repo_root / "logs"
    tmp_dir = repo_root / "tmp"
    logs_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    logs_file = logs_dir / f"{base_id}.log"
    tmp_file = tmp_dir / f"{base_id}_llm_calls.log"
    return logs_file, tmp_file


def log_llm_call(
    *,
    run_id: str,
    call_name: str,
    model_name: str,
    temperature: float | None,
    prompt: str,
    response_text: str | None = None,
    elapsed_seconds: float = 0.0,
    metrics: dict[str, Any] | None = None,
    error: Exception | str | None = None,
) -> tuple[str, str]:
    """Log LLM call (request + response/error) into console and per-run log files."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    base_id = extract_base_run_id(run_id)
    logs_file, tmp_file = get_run_log_file_paths(run_id)

    toks = (metrics or {}).get("tokens_used", {})
    try:
        p_toks = int(toks.get("prompt_tokens", 0) or 0)
    except (TypeError, ValueError):
        p_toks = 0
    try:
        c_toks = int(toks.get("candidates_tokens", 0) or 0)
    except (TypeError, ValueError):
        c_toks = 0
    try:
        t_toks = int(toks.get("total_tokens", 0) or (p_toks + c_toks))
    except (TypeError, ValueError):
        t_toks = p_toks + c_toks

    try:
        cost = float((metrics or {}).get("cost_usd", 0.0) or 0.0)
    except (TypeError, ValueError):
        cost = 0.0

    # 1. Console Logging (Both logger.info and print(..., flush=True) for Docker container log streaming)
    console_req = (
        f"\n{'='*80}\n"
        f"🤖 [LLM REQUEST] [{ts}]\n"
        f"   Run ID      : {run_id} (Base: {base_id})\n"
        f"   Call Name   : {call_name}\n"
        f"   Model       : {model_name}\n"
        f"   Temperature : {temperature}\n"
        f"{'-'*80}\n"
        f"{prompt.strip()}\n"
        f"{'='*80}\n"
    )
    logger.info(console_req)
    print(console_req, flush=True)

    if error is not None:
        console_res = (
            f"\n{'='*80}\n"
            f"❌ [LLM ERROR] [{ts}]\n"
            f"   Run ID      : {run_id}\n"
            f"   Call Name   : {call_name}\n"
            f"   Latency     : {elapsed_seconds}s\n"
            f"   Error       : {error}\n"
            f"{'='*80}\n"
        )
        logger.error(console_res)
        print(console_res, flush=True)
    else:
        res_snippet = (response_text or "").strip()
        console_res = (
            f"\n{'='*80}\n"
            f"✅ [LLM RESPONSE] [{ts}]\n"
            f"   Run ID      : {run_id}\n"
            f"   Call Name   : {call_name}\n"
            f"   Latency     : {elapsed_seconds}s\n"
            f"   Tokens      : {t_toks} (prompt: {p_toks}, candidates: {c_toks})\n"
            f"   Cost        : ${cost:.6f}\n"
            f"{'-'*80}\n"
            f"{res_snippet}\n"
            f"{'='*80}\n"
        )
        logger.info(console_res)
        print(console_res, flush=True)

    # 2. Per-Run Log File
    file_entry_lines = [
        "=" * 80,
        f"[LLM CALL] [{ts}]",
        f"Run ID      : {run_id} (Base: {base_id})",
        f"Call Name   : {call_name}",
        f"Model       : {model_name}",
        f"Temperature : {temperature}",
        "-" * 80,
        ">>> REQUEST PROMPT:",
        prompt.strip(),
        "-" * 80,
    ]
    if error is not None:
        file_entry_lines.extend([
            f"<<< ERROR (Latency: {elapsed_seconds}s):",
            str(error).strip(),
        ])
    else:
        file_entry_lines.extend([
            f"<<< RESPONSE (Latency: {elapsed_seconds}s | Tokens: {t_toks} | Cost: ${cost:.6f}):",
            (response_text or "").strip(),
        ])
    file_entry_lines.extend(["=" * 80, "\n"])
    file_entry_text = "\n".join(file_entry_lines)

    for target_path in (logs_file, tmp_file):
        try:
            if not target_path.exists() or target_path.stat().st_size == 0:
                header = (
                    f"{'='*80}\n"
                    f"MELDAI ENGINE — LLM CALLS AUDIT LOG (RUN ID: {base_id})\n"
                    f"Created at: {ts}\n"
                    f"{'='*80}\n\n"
                )
                target_path.write_text(header + file_entry_text, encoding="utf-8")
            else:
                with open(target_path, "a", encoding="utf-8") as f:
                    f.write(file_entry_text)
        except Exception as exc:
            logger.warning("Failed writing LLM call to %s: %s", target_path, exc)

    return str(logs_file), str(tmp_file)


MEDICATION_SEARCH_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "medication_search" / "active_prompt.md"
)


def get_medication_search_active_prompt() -> str:
    """Read and return active medication search prompt template."""
    if MEDICATION_SEARCH_PROMPT_PATH.is_file():
        return MEDICATION_SEARCH_PROMPT_PATH.read_text(encoding="utf-8").strip()
    return ""


class GeminiService:
    """Client for Google Gemini API powered by google-genai SDK."""

    def __init__(self, settings: Settings | None = None, api_key: str | None = None):
        self._settings = settings or get_settings()
        self._api_key = api_key or self._settings.gemini_api_key
        self._model_name = self._settings.gemini_model_name
        self._temperature = self._settings.gemini_temperature
        self._client: genai.Client | None = None

    @property
    def is_configured(self) -> bool:
        """Check if a non-empty Gemini API key is configured."""
        return bool(self._api_key and self._api_key.strip() and self._api_key != "your_gemini_api_key_here")

    @property
    def client(self) -> genai.Client:
        """Lazy initialized google-genai Client."""
        if self._client is None:
            if not self.is_configured:
                raise ValueError("Cannot initialize genai.Client: Gemini API key is not configured.")
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _infer_call_name(self, prompt: str) -> str:
        """Infer descriptive call_name if not explicitly provided."""
        p_lower = prompt.lower()
        if "step 2: fill clinical gaps" in p_lower:
            return "step2_fill_gap_prescription"
        elif "senior consultant indian psychiatrist" in p_lower:
            return "step1_psychiatrist_prescription"
        elif "evaluating changes in patient symptoms and intake items" in p_lower or "clinical context" in p_lower:
            return "triage_symptom_changes"
        elif "deprescrib" in p_lower:
            return "deprescribing_review"
        elif "active medication search" in p_lower or "patient clinical presentation" in p_lower:
            return "medication_search_queries"
        return "llm_generation"

    def _calculate_cost(self, prompt_tokens: int, candidates_tokens: int) -> float:
        """Calculate estimated cost in USD based on model pricing."""
        m = (self._model_name or "").lower()
        try:
            pt = int(prompt_tokens)
        except (TypeError, ValueError):
            pt = 0
        try:
            ct = int(candidates_tokens)
        except (TypeError, ValueError):
            ct = 0

        m = (self._model_name or "").lower()
        if "pro" in m:
            input_rate = 1.25 / 1_000_000.0
            output_rate = 5.00 / 1_000_000.0
        else:  # flash or default
            input_rate = 0.075 / 1_000_000.0
            output_rate = 0.30 / 1_000_000.0

        cost = (pt * input_rate) + (ct * output_rate)
        return round(cost, 6)

    def generate_content_with_metadata(
        self,
        prompt: str,
        temperature: float | None = None,
        run_id: str | None = None,
        call_name: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Call Google Gemini API using google-genai SDK and return (generated_text, metrics).

        Logs all requests and responses into the console and per-run log file.
        """
        start_time = time.perf_counter()
        ctx_run_id, ctx_call_name = get_llm_run_context()
        active_run_id = (
            run_id
            or ctx_run_id
            or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        )
        active_call_name = call_name or ctx_call_name or self._infer_call_name(prompt)
        effective_temp = temperature if temperature is not None else self._temperature

        if not self.is_configured:
            logger.warning("Gemini API key is not configured. Generating offline mock response.")
            raw_text = self._mock_generation(prompt)
            elapsed = round(time.perf_counter() - start_time, 4)
            p_toks = max(len(prompt.split()) * 4 // 3, 10)
            c_toks = max(len(raw_text.split()) * 4 // 3, 10)
            t_toks = p_toks + c_toks
            cost = self._calculate_cost(p_toks, c_toks)
            metrics = {
                "response_time_seconds": elapsed,
                "tokens_used": {
                    "prompt_tokens": p_toks,
                    "candidates_tokens": c_toks,
                    "total_tokens": t_toks,
                },
                "cost_usd": cost,
            }
            log_llm_call(
                run_id=active_run_id,
                call_name=active_call_name,
                model_name=self._model_name,
                temperature=effective_temp,
                prompt=prompt,
                response_text=raw_text,
                elapsed_seconds=elapsed,
                metrics=metrics,
            )
            return raw_text, metrics

        config = types.GenerateContentConfig(
            temperature=effective_temp,
            response_mime_type="application/json",
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        try:
            response = self.client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=config,
            )
            elapsed = round(time.perf_counter() - start_time, 4)
            raw_text = response.text or ""

            usage = getattr(response, "usage_metadata", None)
            p_raw = getattr(usage, "prompt_token_count", 0) if usage else 0
            c_raw = getattr(usage, "candidates_token_count", 0) if usage else 0
            t_raw = getattr(usage, "total_token_count", 0) if usage else 0

            try:
                p_toks = int(p_raw) if p_raw is not None else 0
            except (TypeError, ValueError):
                p_toks = 0

            try:
                c_toks = int(c_raw) if c_raw is not None else 0
            except (TypeError, ValueError):
                c_toks = 0

            try:
                t_toks = int(t_raw) if t_raw is not None else (p_toks + c_toks)
            except (TypeError, ValueError):
                t_toks = p_toks + c_toks

            if t_toks == 0 and (p_toks + c_toks) > 0:
                t_toks = p_toks + c_toks

            cost = self._calculate_cost(p_toks, c_toks)

            metrics = {
                "response_time_seconds": elapsed,
                "tokens_used": {
                    "prompt_tokens": p_toks,
                    "candidates_tokens": c_toks,
                    "total_tokens": t_toks,
                },
                "cost_usd": cost,
            }
            log_llm_call(
                run_id=active_run_id,
                call_name=active_call_name,
                model_name=self._model_name,
                temperature=effective_temp,
                prompt=prompt,
                response_text=raw_text,
                elapsed_seconds=elapsed,
                metrics=metrics,
            )
            return raw_text, metrics
        except Exception as exc:
            elapsed = round(time.perf_counter() - start_time, 4)
            log_llm_call(
                run_id=active_run_id,
                call_name=active_call_name,
                model_name=self._model_name,
                temperature=effective_temp,
                prompt=prompt,
                elapsed_seconds=elapsed,
                error=exc,
            )
            logger.error("Gemini API request error via google-genai: %s", exc)
            raise

    def generate_content(
        self,
        prompt: str,
        temperature: float | None = None,
        run_id: str | None = None,
        call_name: str | None = None,
    ) -> str:
        """Call Google Gemini API using google-genai SDK and return the generated text."""
        raw_text, _ = self.generate_content_with_metadata(
            prompt=prompt,
            temperature=temperature,
            run_id=run_id,
            call_name=call_name,
        )
        return raw_text

    def generate_prescription(
        self,
        prompt: str,
        temperature: float | None = None,
        run_id: str | None = None,
        call_name: str | None = None,
    ) -> dict[str, Any]:
        """Generate structured prescription recommendations with latency, token usage, and cost."""
        try:
            raw_text, metrics = self.generate_content_with_metadata(
                prompt=prompt,
                temperature=temperature,
                run_id=run_id,
                call_name=call_name or "generate_prescription",
            )
        except Exception as exc:
            if run_id:
                try:
                    from core.utils import save_response_to_tmp
                    save_response_to_tmp(f"ERROR: {exc}", run_id=run_id)
                except Exception:
                    pass
            raise

        if run_id:
            try:
                from core.utils import save_response_to_tmp
                save_response_to_tmp(raw_text, run_id=run_id)
            except Exception as exc:
                logger.warning("Failed to save response to tmp: %s", exc)

        parsed = self._parse_json_response(raw_text)
        parsed["raw_response"] = raw_text
        parsed["response_time_seconds"] = metrics["response_time_seconds"]
        parsed["tokens_used"] = metrics["tokens_used"]
        parsed["cost_usd"] = metrics["cost_usd"]
        return parsed

    def generate_medication_search_queries(
        self,
        symptoms: list[str] | None = None,
        diagnosis: list[str] | None = None,
        duration_context: str | None = None,
        current_regimen: list[str] | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Execute active medication search prompt with Gemini LLM and return vector_semantic_queries & bm25_keywords."""
        system_text = get_medication_search_active_prompt()

        sym_str = ", ".join(symptoms) if symptoms else "None reported"
        diag_str = ", ".join(diagnosis) if diagnosis else "None specified"
        dur_str = duration_context or "Not specified"
        reg_str = ", ".join(current_regimen) if current_regimen else "None"

        patient_presentation = (
            "\n\n### PATIENT CLINICAL PRESENTATION:\n"
            f"- Symptoms: {sym_str}\n"
            f"- Diagnosis: {diag_str}\n"
            f"- Duration/Context: {dur_str}\n"
            f"- Current Regimen: {reg_str}\n"
        )

        full_prompt = system_text + patient_presentation
        raw_text = self.generate_content(
            prompt=full_prompt,
            temperature=temperature,
            call_name="medication_search_queries",
        )

        logger.info("Gemini raw_text: %s", raw_text)

        return self._parse_search_queries_response(raw_text, symptoms=symptoms, diagnosis=diagnosis)

        logger.info("Gemini raw_text: %s", raw_text)

        return self._parse_search_queries_response(raw_text, symptoms=symptoms, diagnosis=diagnosis)

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Safely extract and parse JSON object from Gemini output text."""
        cleaned = text.strip()
        json_codeblock_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        candidate = json_codeblock_match.group(1).strip() if json_codeblock_match else cleaned
        if candidate.startswith("```json"):
            candidate = candidate[7:]
        elif candidate.startswith("```"):
            candidate = candidate[3:]
        if candidate.endswith("```"):
            candidate = candidate[:-3]
        candidate = candidate.strip()

        data = None
        try:
            data = json.loads(candidate)
        except Exception:
            outer_match = re.search(r"(\{[\s\S]*\})", candidate)
            if outer_match:
                try:
                    data = json.loads(outer_match.group(1).strip())
                except Exception:
                    data = None

        if isinstance(data, dict):
            meds = data.get("medications", [])
            if not meds and "recommended_medications" in data:
                raw_recs = data.get("recommended_medications")
                if isinstance(raw_recs, list):
                    meds = [
                        m.get("medication_master_item") or m.get("medication") or m
                        if (isinstance(m, dict) and (m.get("medication_master_item") or m.get("medication") or len(m) > 2))
                        else (m.get("brand_name", "") if isinstance(m, dict) else str(m))
                        for m in raw_recs
                    ]
                    meds = [
                        m if isinstance(m, dict)
                        else m.strip()
                        for m in meds
                        if (isinstance(m, dict) or (isinstance(m, str) and m.strip()))
                    ]
            elif isinstance(meds, str):
                meds = [meds]
            elif not isinstance(meds, list):
                meds = []
            explanation = str(data.get("explanation", "")).strip()
            return {
                "medications": meds,
                "explanation": explanation,
            }

        if isinstance(data, list):
            meds = [
                m if (isinstance(m, dict) and len(m) > 1)
                else (m.get("brand_name", "") if isinstance(m, dict) else str(m))
                for m in data
            ]
            meds = [
                m if isinstance(m, dict)
                else m.strip()
                for m in meds
                if (isinstance(m, dict) or (isinstance(m, str) and m.strip()))
            ]
            return {
                "medications": meds,
                "explanation": "",
            }

        logger.warning("JSON decode failed for Gemini response. Attempting regex extraction.")

        # Fallback regex extraction if raw JSON parsing fails
        meds_list: list[str] = []
        brand_matches = re.findall(r'"brand_name"\s*:\s*"([^"]+)"', text)
        if brand_matches:
            meds_list = [b.strip() for b in brand_matches if b.strip()]
        else:
            meds_match = re.search(r'"medications"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if meds_match:
                meds_list = [m.strip().strip('"').strip("'") for m in meds_match.group(1).split(",") if m.strip()]

        expl_match = re.search(r'"explanation"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', text, re.DOTALL)
        explanation_str = expl_match.group(1) if expl_match else text.strip()

        return {
            "medications": meds_list,
            "explanation": explanation_str,
        }

    def _parse_search_queries_response(
        self,
        text: str,
        symptoms: list[str] | None = None,
        diagnosis: list[str] | None = None,
    ) -> dict[str, Any]:
        """Safely parse JSON response containing vector_semantic_queries and bm25_keywords."""
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                v_queries = data.get("vector_semantic_queries", [])
                if isinstance(v_queries, str):
                    v_queries = [v_queries]
                elif not isinstance(v_queries, list):
                    v_queries = []

                b_keywords = data.get("bm25_keywords", [])
                if isinstance(b_keywords, str):
                    b_keywords = [b_keywords]
                elif not isinstance(b_keywords, list):
                    b_keywords = []

                return {
                    "vector_semantic_queries": [str(q).strip() for q in v_queries if str(q).strip()],
                    "bm25_keywords": [str(k).strip() for k in b_keywords if str(k).strip()],
                }
        except Exception as exc:
            logger.warning("JSON decode failed for search queries (%s).", exc)

        s_list = symptoms or []
        d_list = diagnosis or []
        diag_term = d_list[0] if d_list else (s_list[0] if s_list else "Clinical condition")

        fallback_vec = [
            f"Pharmacotherapy and clinical dosing indication for {diag_term} with {', '.join(s_list[:3])}."
        ] if (s_list or d_list) else ["Pharmacotherapy for general clinical presentation."]

        fallback_bm25 = list(set([t for t in d_list + s_list if t])) or ["Medication"]

        return {
            "vector_semantic_queries": fallback_vec,
            "bm25_keywords": fallback_bm25,
        }

    def _mock_generation(self, prompt: str) -> str:
        """Provide fallback clinical mock response when API key is missing (for testing/offline)."""
        logger.info("Executing mock Gemini response based on prompt heuristics.")

        if "vector_semantic_queries" in prompt or "bm25_keywords" in prompt:
            s_match = re.search(r"- Symptoms:\s*(.*)", prompt)
            d_match = re.search(r"- Diagnosis:\s*(.*)", prompt)
            s_val = s_match.group(1).strip() if s_match else ""
            d_val = d_match.group(1).strip() if d_match else ""

            target_disease = d_val if (d_val and d_val != "None specified") else "clinical condition"
            sym_desc = s_val if (s_val and s_val != "None reported") else ""

            vec_queries = [
                f"Clinical dosing indication for management of {target_disease}" + (f" with {sym_desc}" if sym_desc else "") + "."
            ]
            keywords = []
            if d_val and d_val != "None specified":
                keywords.append(d_val)
            if s_val and s_val != "None reported":
                keywords.extend([kw.strip() for kw in s_val.split(",") if kw.strip()])
            keywords.extend(["SSRI", "Anxiolytic", "Pharmacotherapy"])

            return json.dumps({
                "vector_semantic_queries": vec_queries,
                "bm25_keywords": list(dict.fromkeys(keywords)),
            })

        # Deprescribe prompt mock
        if "deprescrib" in prompt.lower() or "clinical action definitions" in prompt.lower() or "medication action needed" in prompt.lower():
            return json.dumps({
                "whats_changed": "Discontinued resolved acute symptom medications and maintained chronic regimen.",
                "deprescribing_recommendations": [
                    {
                        "medicine": "Clonazepam",
                        "target_symptom": "severe insomnia",
                        "status_change": "decreased",
                        "action": "decreased",
                        "rationale": "Sleep improved; taper acute benzodiazepine to avoid dependence.",
                        "taper_plan": "Reduce Clonazepam to 0.25 mg at night for 1 week then stop.",
                    }
                ],
            })

        # Step 1 llmMedCheck prompt mock
        if "llm_med_check" in prompt.lower() or "check medication" in prompt.lower() or "entities-needing-medications" in prompt.lower() or "directly or indirectly target" in prompt.lower():
            return json.dumps({
                "whats_changed": "Assigned Escitalopram + Clonazepam and Paracetamol from local formulary for entities lacking candidates.",
                "entity_medication_check": [
                    {
                        "entity": "severe anxiety",
                        "entity_type": "symptom",
                        "curie": "HP:0000739",
                        "delta_status": "new",
                        "directly_targeted": True,
                        "indirectly_targeted": False,
                        "attributed_molecules": ["Escitalopram + Clonazepam", "Escitalopram"],
                        "recommended_top_molecules": [
                            {
                                "canonical_molecule": "Escitalopram + Clonazepam",
                                "is_fdc": True,
                                "doctor_brands": ["RTC-LS"],
                                "frequency": 10,
                                "score": 9.5,
                                "percentage": 90.0,
                                "targeting_type": "direct",
                                "rationale": "First-line FDC anxiolytic coverage.",
                            }
                        ],
                    },
                    {
                        "entity": "headache",
                        "entity_type": "symptom",
                        "curie": "HP:0002315",
                        "delta_status": "new",
                        "directly_targeted": True,
                        "indirectly_targeted": False,
                        "attributed_molecules": ["Paracetamol"],
                        "recommended_top_molecules": [
                            {
                                "canonical_molecule": "Paracetamol",
                                "is_fdc": False,
                                "doctor_brands": ["DOLO 650"],
                                "frequency": 5,
                                "score": 8.0,
                                "percentage": 80.0,
                                "targeting_type": "direct",
                                "rationale": "Analgesic for tension headache.",
                            }
                        ],
                    },
                ],
                "summary": "Verified direct and indirect coverage across all clinical entities.",
            })

        # Step 2 V2 llmMedPrescribe prompt mock
        if "llm_med_prescribe" in prompt.lower() or "<updated-entity-matrix>" in prompt or "reproducing their own prescribing pattern" in prompt.lower():
            return json.dumps({
                "medications": [
                    {
                        "brand_name": "RTC-LS",
                        "canonical_molecule": "Escitalopram + Clonazepam",
                        "is_fdc": True,
                        "constituent_strengths_mg": {
                            "Escitalopram": 10.0,
                            "Clonazepam": 0.5,
                        },
                        "modal_strength_mg": 10.0,
                        "release_form": "FDC",
                        "dosage_tier": "STANDARD_MAINTENANCE",
                        "dosage_group_id": "GRP_FDC_ESCITALOPRAM_10_CLONAZEPAM_0.5",
                        "clinical_dosing_indication": "Fixed-dose combination for severe generalized anxiety with panic symptoms and sleep disruption.",
                        "target_entities": ["severe anxiety", "insomnia"],
                        "clinical_rationale": "FDC predicted for immediate anxiolytic coverage while antidepressant takes effect.",
                    },
                    {
                        "brand_name": "ESPRIDE",
                        "canonical_molecule": "Escitalopram",
                        "is_fdc": False,
                        "constituent_strengths_mg": {
                            "Escitalopram": 10.0,
                        },
                        "modal_strength_mg": 10.0,
                        "release_form": "Regular",
                        "dosage_tier": "STANDARD_MAINTENANCE",
                        "dosage_group_id": "GRP_ESCITALOPRAM_STANDARD",
                        "clinical_dosing_indication": "Core SSRI antidepressant therapy for unipolar depression and generalized anxiety disorder.",
                        "target_entities": ["generalized anxiety disorder"],
                        "clinical_rationale": "Non-FDC SSRI monotherapy indicated for maintenance.",
                    },
                    {
                        "brand_name": "DOLO 650",
                        "canonical_molecule": "Paracetamol",
                        "is_fdc": False,
                        "constituent_strengths_mg": {
                            "Paracetamol": 650.0,
                        },
                        "modal_strength_mg": 650.0,
                        "release_form": "Regular",
                        "dosage_tier": "STANDARD_MAINTENANCE",
                        "dosage_group_id": "GRP_PARACETAMOL_650",
                        "clinical_dosing_indication": "Analgesic for episodic tension headache and somatic complaints.",
                        "target_entities": ["headache", "somatic pain"],
                        "clinical_rationale": "Analgesic for tension-type somatic headache relief.",
                    },
                ],
                "deprescribing_recommendations": [
                    {
                        "medicine": "Clonazepam",
                        "target_symptom": "insomnia",
                        "status_change": "stop",
                        "action": "taper",
                        "rationale": "Taper acute benzodiazepine once sleep restores.",
                        "taper_plan": "0.25 mg for 3 days then discontinue.",
                    }
                ],
                "explanation": "Final comprehensive Indian Psychiatrist regimen integrating updated entity matrix.",
                "confidence_score": 0.95,
            })

        # Default mock for generate_prescription
        medications = ["Paracetamol 500mg", "Cetirizine 10mg"]
        if "insomnia" in prompt.lower():
            medications = ["Melatonin 3mg", "Zolpidem 5mg"]
        elif "hypertension" in prompt.lower():
            medications = ["Amlodipine 5mg", "Telmisartan 40mg"]
        elif "asthma" in prompt.lower() or "cough" in prompt.lower():
            medications = ["Salbutamol Inhaler", "Montelukast 10mg"]

        mock_payload = {
            "medications": medications,
            "explanation": (
                "Mock Prescription Recommendation: Medications selected according to patient clinical findings, "
                "addressing confirmed symptoms and diagnoses while adhering to historical prescription patterns."
            ),
        }
        return json.dumps(mock_payload)


_gemini_service_instance: GeminiService | None = None


def get_gemini_service(settings: Settings | None = None) -> GeminiService:
    """Singleton getter for GeminiService."""
    global _gemini_service_instance
    if _gemini_service_instance is None:
        _gemini_service_instance = GeminiService(settings=settings)
    return _gemini_service_instance
