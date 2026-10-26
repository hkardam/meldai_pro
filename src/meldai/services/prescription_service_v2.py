"""Prescription Service V2 — Compatibility Shim.

This module has been relocated to `meldai.v2.services.prescription_service`.
Re-exports are provided here for full backward compatibility.
"""

from meldai.v2.services.prescription_service import (
    PrescriptionService,
    PrescriptionServiceV2,
    _build_entity_ranking_service,
    strip_special_characters,
    format_entity_matrix_for_prompt,
    save_workflow_prompts_to_file,
    build_llm_med_check_prompt,
    build_llm_med_prescribe_prompt,
    PROMPTS_DIR,
    LLM_MED_CHECK_PROMPT_PATH,
    LLM_MED_PRESCRIBE_PROMPT_PATH,
)

__all__ = [
    "PrescriptionService",
    "PrescriptionServiceV2",
    "_build_entity_ranking_service",
    "strip_special_characters",
    "format_entity_matrix_for_prompt",
    "save_workflow_prompts_to_file",
    "build_llm_med_check_prompt",
    "build_llm_med_prescribe_prompt",
    "PROMPTS_DIR",
    "LLM_MED_CHECK_PROMPT_PATH",
    "LLM_MED_PRESCRIBE_PROMPT_PATH",
]
