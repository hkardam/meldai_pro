"""Gemini LLM Service for medical text and prescription generation."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from google import genai
from google.genai import types

from meldai.config import Settings, get_settings

logger = logging.getLogger(__name__)


from pathlib import Path

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

    def generate_content(
        self,
        prompt: str,
        temperature: float | None = None,
    ) -> str:
        """Call Google Gemini API using google-genai SDK and return the generated text."""
        if not self.is_configured:
            logger.warning("Gemini API key is not configured. Generating offline mock response.")
            return self._mock_generation(prompt)

        config = types.GenerateContentConfig(
            temperature=temperature if temperature is not None else self._temperature,
            response_mime_type="application/json",
            tools=[{"google_search": {}}]
        )

        try:
            response = self.client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=config,
            )
            return response.text or ""
        except Exception as exc:
            logger.error("Gemini API request error via google-genai: %s", exc)
            raise

    def generate_prescription(
        self,
        prompt: str,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Generate structured prescription recommendations."""
        raw_text = self.generate_content(prompt=prompt, temperature=temperature)
        return self._parse_json_response(raw_text)

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
        raw_text = self.generate_content(prompt=full_prompt, temperature=temperature)

        logger.info("Gemini raw_text: %s", raw_text)

        return self._parse_search_queries_response(raw_text, symptoms=symptoms, diagnosis=diagnosis)

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """Safely extract and parse JSON object from Gemini output text."""
        cleaned = text.strip()
        # Remove markdown codeblocks if present
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
                meds = data.get("medications", [])
                if isinstance(meds, str):
                    meds = [meds]
                elif not isinstance(meds, list):
                    meds = []
                explanation = str(data.get("explanation", "")).strip()
                return {
                    "medications": meds,
                    "explanation": explanation,
                }
        except Exception as exc:
            logger.warning("JSON decode failed for Gemini response (%s). Attempting regex extraction.", exc)

        # Fallback regex extraction if raw JSON parsing fails
        meds_match = re.search(r'"medications"\s*:\s*\[(.*?)\]', cleaned, re.DOTALL)
        meds_list: list[str] = []
        if meds_match:
            meds_list = [m.strip().strip('"').strip("'") for m in meds_match.group(1).split(",") if m.strip()]

        expl_match = re.search(r'"explanation"\s*:\s*"(.*?)"', cleaned, re.DOTALL)
        explanation_str = expl_match.group(1) if expl_match else cleaned

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
