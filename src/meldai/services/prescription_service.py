import logging
from pathlib import Path
from typing import Any

from meldai.api.req_dtos import (
    FindSimilarCaseRequest,
    PrescribeMedicationsRequest,
    SimilarityParams,
)
from meldai.api.res_dtos import DecoratedCase, PrescribeMedicationsResponse, SimilarCaseItem
from meldai.config import Settings, get_settings
from meldai.db.mongodb import MongoKnowledgeBase
from meldai.services.case_service import CaseService
from meldai.services.gemini_service import GeminiService, get_gemini_service

logger = logging.getLogger(__name__)

ACTIVE_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "medication_prediction" / "active_prompt.md"
FALLBACK_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "active_prompt.md"


def get_active_prompt() -> str:
    """Read and return the active clinical prompt template from active_prompt.md."""
    target_path = ACTIVE_PROMPT_PATH if ACTIVE_PROMPT_PATH.is_file() else FALLBACK_PROMPT_PATH
    if target_path.is_file():
        try:
            return target_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning("Failed to read active prompt from %s: %s", target_path, exc)
    return ""


DUMMY_PROMPT_SYSTEM_TEXT = get_active_prompt()


def _sanitize_clinical_entities(items: list[Any], code_key: str = "code") -> list[dict[str, Any]]:
    """Clean clinical entity dictionaries to exclude any embedding vectors."""
    cleaned: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            clean_item = {k: v for k, v in item.items() if k != "embedding"}
            cleaned.append(clean_item)
        elif hasattr(item, "model_dump"):
            dumped = item.model_dump(exclude={"embedding"})
            cleaned.append(dumped)
        elif isinstance(item, str):
            cleaned.append({"term": item})
    return cleaned


def _extract_medications(case_doc: dict[str, Any]) -> list[str]:
    """Extract list of prescribed medications from case document."""
    meds = case_doc.get("medications") or case_doc.get("medication") or []
    if isinstance(meds, list):
        clean_meds = []
        for m in meds:
            if isinstance(m, str) and m.strip():
                clean_meds.append(m.strip())
            elif isinstance(m, dict) and "name" in m:
                clean_meds.append(str(m["name"]).strip())
        return clean_meds
    elif isinstance(meds, str) and meds.strip():
        return [meds.strip()]
    return []


class PrescriptionService:
    """Service to enrich case details, prepare contextual LLM prompts, and prescribe medications."""

    def __init__(
        self,
        settings: Settings | None = None,
        case_service: CaseService | None = None,
        mongo_kb: MongoKnowledgeBase | None = None,
        gemini_service: GeminiService | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._case_service = case_service or CaseService(settings=self._settings)
        self._mongo_kb = mongo_kb
        self._gemini_service = gemini_service or get_gemini_service(self._settings)

    def prepare_prompt(
        self,
        decorated_case: DecoratedCase,
        past_cases: list[dict[str, Any]],
        similar_cases: list[SimilarCaseItem],
        system_text: str | None = None,
    ) -> str:
        """Construct the prompt combining current case, past visits, and similar cases with medications.

        Explicitly strips out all dense vector embeddings.
        """
        prompt_lines: list[str] = []

        # 1. System & contextual instructions
        clinical_instruction = (system_text if system_text is not None else get_active_prompt()).strip()
        if clinical_instruction:
            prompt_lines.append(clinical_instruction)
            prompt_lines.append("")

        # 2. Input data for current case (decorated with HPO and MONDO codes, no embeddings)
        prompt_lines.append("<data>")
        prompt_lines.append("### CURRENT ENCOUNTER:")
        if decorated_case.visitType:
            prompt_lines.append(f"- Visit Type: {decorated_case.visitType}")

        p_info = decorated_case.patientInfo
        age_str = f"{p_info.age} years" if p_info.age is not None else "Unknown"
        gender_str = p_info.gender if p_info.gender else "Unknown"
        prompt_lines.append(f"- Demographics: Age: {age_str}, Gender: {gender_str}")

        from meldai.utils import to_curie

        # Partition symptoms: presenting symptoms (affirmed phenotypes) vs pertinent negatives (negated findings)
        all_symptoms = decorated_case.symptoms or []
        presenting_symptoms = [s for s in all_symptoms if s.isPheno and not s.isNegation]
        pertinent_negatives = [s for s in all_symptoms if s.isNegation]

        prompt_lines.append("- Presenting Symptoms (with HPO Codes):")
        if presenting_symptoms:
            for s in presenting_symptoms:
                curie = getattr(s, "hpoCurie", None) or to_curie("HP", s.hpoCode)
                if curie and s.hpoTerm:
                    hpo_repr = f"{curie} ({s.hpoTerm})"
                elif curie:
                    hpo_repr = curie
                else:
                    hpo_repr = "null"
                prompt_lines.append(f"  * {s.term} -> {hpo_repr}")
        else:
            prompt_lines.append("  * None reported")

        if pertinent_negatives:
            prompt_lines.append("- Pertinent Negatives:")
            for s in pertinent_negatives:
                curie = getattr(s, "hpoCurie", None) or to_curie("HP", s.hpoCode)
                if curie and s.hpoTerm:
                    neg_repr = f" [NEGATED: {curie} ({s.hpoTerm})]"
                elif curie:
                    neg_repr = f" [NEGATED: {curie}]"
                else:
                    neg_repr = " [NEGATED]"
                prompt_lines.append(f"  * {s.term}{neg_repr}")

        prompt_lines.append("- Clinical Diagnoses (with MONDO Codes):")
        if decorated_case.diagnosis:
            for d in decorated_case.diagnosis:
                curie = getattr(d, "mondoCurie", None) or to_curie("MONDO", d.mondoCode)
                if curie and d.mondoTerm:
                    mondo_repr = f"{curie} ({d.mondoTerm})"
                elif curie:
                    mondo_repr = curie
                else:
                    mondo_repr = "null"
                prompt_lines.append(f"  * {d.term} -> {mondo_repr}")
        else:
            prompt_lines.append("  * None specified")
        prompt_lines.append("")

        # 3. Past visits for this patient (if found)
        prompt_lines.append("### PATIENT PAST ENCOUNTER HISTORY:")
        if past_cases:
            for idx, past in enumerate(past_cases, start=1):
                v_date = past.get("visitDate", "Unknown date")
                v_reason = past.get("visitReason", "")
                meds = _extract_medications(past)
                meds_str = ", ".join(meds) if meds else "None recorded"

                diag_list = []
                for d in past.get("diagnosis", []):
                    if isinstance(d, dict):
                        d_term = d.get("term") or d.get("text")
                        d_code = d.get("mondoCode") or d.get("mondoId")
                        d_curie = to_curie("MONDO", d_code)
                        diag_list.append(f"{d_term} ({d_curie})" if d_curie else f"{d_term} (null)")
                    elif isinstance(d, str):
                        diag_list.append(d)

                sym_list = []
                for s in past.get("symptoms", []):
                    if isinstance(s, dict):
                        s_term = s.get("term") or s.get("text") or s.get("note")
                        s_code = s.get("hpoCode") or s.get("hpoId")
                        s_curie = to_curie("HP", s_code)
                        sym_list.append(f"{s_term} ({s_curie})" if s_curie else f"{s_term} (null)")
                    elif isinstance(s, str):
                        sym_list.append(s)

                entry = f"- Visit #{idx} ({v_date}" + (f", {v_reason}" if v_reason else "") + "):"
                prompt_lines.append(entry)
                if diag_list:
                    prompt_lines.append(f"  * Prior Diagnoses: {', '.join(diag_list)}")
                if sym_list:
                    prompt_lines.append(f"  * Prior Symptoms: {', '.join(sym_list)}")
                prompt_lines.append(f"  * Medications Prescribed: {meds_str}")
        else:
            prompt_lines.append("- No prior encounters recorded for this patient.")
        prompt_lines.append("")

        # 4. Similar cases with medication given (helps LLM adapt to doctor's prescription style)
        prompt_lines.append("### SIMILAR CASES WITH PRESCRIBED MEDICATIONS:")
        if similar_cases:
            for idx, item in enumerate(similar_cases, start=1):
                cand = item.case
                score = item.finalScore
                meds = _extract_medications(cand)
                meds_str = ", ".join(meds) if meds else "None recorded"

                c_age = None
                p_inf = cand.get("patientInfo")
                if isinstance(p_inf, dict) and "age" in p_inf:
                    c_age = p_inf["age"]
                elif isinstance(cand.get("patient"), dict):
                    age_obj = cand["patient"].get("age")
                    if isinstance(age_obj, dict):
                        c_age = age_obj.get("year")

                c_vt = cand.get("visitReason") or cand.get("visitType") or "Encounter"

                diag_list = []
                for d in cand.get("diagnosis", []):
                    if isinstance(d, dict):
                        d_term = d.get("term") or d.get("text")
                        d_code = d.get("mondoCode") or d.get("mondoId")
                        d_curie = to_curie("MONDO", d_code)
                        diag_list.append(f"{d_term} ({d_curie})" if d_curie else f"{d_term} (null)")
                    elif isinstance(d, str):
                        diag_list.append(d)

                sym_list = []
                for s in cand.get("symptoms", []):
                    if isinstance(s, dict):
                        s_term = s.get("term") or s.get("text") or s.get("note")
                        s_code = s.get("hpoCode") or s.get("hpoId")
                        s_curie = to_curie("HP", s_code)
                        sym_list.append(f"{s_term} ({s_curie})" if s_curie else f"{s_term} (null)")
                    elif isinstance(s, str):
                        sym_list.append(s)

                cand_summary = f"- Similar Case #{idx} (Similarity Match: {score:.2f}, {c_vt}, Age: {c_age or 'N/A'}):"
                prompt_lines.append(cand_summary)
                if diag_list:
                    prompt_lines.append(f"  * Diagnoses: {', '.join(diag_list)}")
                if sym_list:
                    prompt_lines.append(f"  * Symptoms: {', '.join(sym_list)}")
                prompt_lines.append(f"  * Prescribed Medications: {meds_str}")
        else:
            prompt_lines.append("- No similar cases identified in the database.")
        prompt_lines.append("</data>")
        prompt_lines.append("")

        # 5. Output format constraint
        prompt_lines.append("### TASK & RESPONSE FORMAT:")
        prompt_lines.append(
            "Based on the current patient encounter, past visits, and doctor's historical prescription patterns in similar cases, "
            "recommend the optimal medications. Respond strictly in valid JSON adhering to this schema:\n"
            "{\n"
            '  "chain_of_thought": {\n'
            '    "hypothesis": "What the picture suggests based on age, sex, diagnoses, and symptoms.",\n'
            '    "history_analysis": "What changed/persisted from past visits and implied actions.",\n'
            '    "similar_cases_analysis": "Pattern of drug classes/brands used by this doctor for this profile.",\n'
            '    "research": "Generic composition and cautions for the considered drugs."\n'
            "  },\n"
            '  "missing_information": ["List of data points the physician should verify"],\n'
            '  "recommended_medications": [\n'
            "    {\n"
            '      "brand_name": "Exact brand / medicine name",\n'
            '      "basis": "patient_history OR similar_cases OR outside_pattern",\n'
            '      "cautions": "Safety concerns or drug conflicts, if any",\n'
            '      "depends_on_missing_info": true OR false\n'
            "    }\n"
            "  ],\n"
            '  "explanation": "Clinical justification explaining the final selection."\n'
            "}"
        )

        return "\n".join(prompt_lines)

    def prescribe_medications(
        self,
        request: PrescribeMedicationsRequest,
    ) -> PrescribeMedicationsResponse:
        """Execute end-to-end prescription workflow: decorate case -> retrieve context -> build prompt -> call Gemini."""
        mongo_kb = self._mongo_kb or MongoKnowledgeBase(self._settings)

        # 1. Enrich case, fetch past cases, and retrieve similar cases with medications
        similar_req = FindSimilarCaseRequest(
            case=request.case,
            top_k=request.top_k,
            params=request.params or SimilarityParams(),
        )
        similar_res = self._case_service.find_similar_cases(similar_req, mongo_kb=mongo_kb)

        decorated_case = similar_res.decoratedCase
        past_cases = similar_res.pastCases
        similar_cases = similar_res.similarCases

        logger.info(
            "Prescription context assembled for caseNo=%s: %d past cases, %d similar cases",
            decorated_case.caseNo, len(past_cases), len(similar_cases)
        )

        # 2. Build prompt (strictly omitting dense vector embeddings)
        prompt = self.prepare_prompt(
            decorated_case=decorated_case,
            past_cases=past_cases,
            similar_cases=similar_cases,
        )

        # 3. Call Gemini LLM
        llm_result = self._gemini_service.generate_prescription(prompt)

        medications = llm_result.get("medications", [])
        explanation = llm_result.get("explanation", "")

        return PrescribeMedicationsResponse(
            medications=medications,
            explanation=explanation,
            pastCasesCount=len(past_cases),
            similarCasesCount=len(similar_cases),
        )
