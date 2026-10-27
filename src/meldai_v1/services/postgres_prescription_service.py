"""PostgreSQL-driven prescription service for case enrichment, similarity retrieval, and prompt generation."""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import HTTPException
from sqlalchemy import text

from meldai_v1.req_dtos import PrescribeMedicationsRequest
from meldai_v1.res_dtos import DecoratedCase, PrescribeMedicationsResponse
from core.config import Settings, get_settings
from core.db.postgres import PostgresSource
from core.services.gemini_service import GeminiService, get_gemini_service, set_llm_run_context
from core.services.case_service import CaseService
from core.utils import generate_run_id, save_prompt_to_tmp, save_response_to_tmp, to_curie

logger = logging.getLogger(__name__)

ACTIVE_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "active_prompt.md"
FALLBACK_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "mp_prompt_v1.md"


def get_active_prompt() -> str:
    """Read the active clinical prompt template, falling back to mp_prompt_v1.md if empty."""
    if ACTIVE_PROMPT_PATH.is_file():
        try:
            content = ACTIVE_PROMPT_PATH.read_text(encoding="utf-8").strip()
            if content:
                return content
        except Exception as exc:
            logger.warning("Failed to read active prompt from %s: %s", ACTIVE_PROMPT_PATH, exc)

    if FALLBACK_PROMPT_PATH.is_file():
        try:
            return FALLBACK_PROMPT_PATH.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning("Failed to read fallback prompt from %s: %s", FALLBACK_PROMPT_PATH, exc)

    return ""


def format_dt_case_no(case_no: Optional[int]) -> Optional[str]:
    """Prefix integer caseNo with 'DT' for PostgreSQL lookup, e.g. 1 -> 'DT1'."""
    if case_no is None:
        return None
    return f"DT{case_no}"


class PostgresPrescriptionService:
    """Service to enrich case details, query similar encounters from PostgreSQL, and construct LLM prompts."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        case_service: Optional[CaseService] = None,
        postgres_source: Optional[PostgresSource] = None,
        gemini_service: Optional[GeminiService] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._case_service = case_service or CaseService(settings=self._settings)
        self._postgres_source = postgres_source or PostgresSource(settings=self._settings)
        self._gemini_service = gemini_service or get_gemini_service(self._settings)

    def fetch_past_encounters(
        self,
        case_no: Optional[int],
        case_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch past visits for a specific caseNo strictly prior to case_date (if specified) from PostgreSQL."""
        if case_no is None:
            return []

        dt_case_no = format_dt_case_no(case_no)
        query_sql = """
        WITH p_info AS (
            SELECT pd."case No", pd."Gender", pd."Age Year" AS "Age"
            FROM temp_migrations.patient_data pd
        ), 
        p_cc AS (
            SELECT pcd."Case No", pcd."Visit Date", pcd."Chief Complaints" 
            FROM temp_migrations.patient_chiefcomplains_data pcd
        ), 
        p_dx AS (
            SELECT pdd."Case No", pdd."Visit Date", string_agg(pdd."Diagnosis Name", ', ') AS "Diagnosis Names"  
            FROM temp_migrations.patient_diagnosis_data pdd
            GROUP BY pdd."Case No", pdd."Visit Date"
        ), 
        p_vr AS (
            SELECT pvd."Case No", pvd."Visit Date", pvd."Visit Reason"  
            FROM temp_migrations.patient_visit_data pvd
        ), 
        p_rx AS (
            SELECT ppd."Case No", ppd."Visit Date", string_agg(ppd."Medicine Name", ', ') AS "Medicine Names"  
            FROM temp_migrations.patient_prescription_data ppd
            GROUP BY ppd."Case No", ppd."Visit Date"
        )
        SELECT 
            p_vr."Case No",
            p_vr."Visit Date",
            to_date(p_vr."Visit Date", 'DD/MM/YYYY') AS parsed_visit_date,
            p_info."Gender",
            p_info."Age",
            p_cc."Chief Complaints",
            p_vr."Visit Reason",
            p_dx."Diagnosis Names",
            p_rx."Medicine Names"
        FROM p_vr
        LEFT JOIN p_info ON p_vr."Case No" = p_info."case No"
        LEFT JOIN p_cc   ON p_vr."Case No" = p_cc."Case No"   AND p_vr."Visit Date" = p_cc."Visit Date"
        LEFT JOIN p_dx   ON p_vr."Case No" = p_dx."Case No"   AND p_vr."Visit Date" = p_dx."Visit Date"
        LEFT JOIN p_rx   ON p_vr."Case No" = p_rx."Case No"   AND p_vr."Visit Date" = p_rx."Visit Date"
        WHERE p_vr."Case No" = :case_no_str
        ORDER BY parsed_visit_date ASC
        """

        params: Dict[str, Any] = {
            "case_no_str": dt_case_no,
        }

        try:
            with self._postgres_source.engine.connect() as conn:
                rows = conn.execute(text(query_sql), params).mappings().all()

                parsed_case_date = None
                if case_date:
                    for fmt in ("%Y-%m-%d", "%d/%m/%Y"):
                        try:
                            parsed_case_date = datetime.strptime(case_date, fmt).date()
                            break
                        except ValueError:
                            continue

                past_visits = []
                for row in rows:
                    p_date = row.get("parsed_visit_date")
                    if parsed_case_date and p_date and p_date >= parsed_case_date:
                        continue
                    past_visits.append({
                        "caseNo": row.get("Case No"),
                        "visitDate": row.get("Visit Date"),
                        "gender": row.get("Gender"),
                        "age": row.get("Age"),
                        "chiefComplaints": row.get("Chief Complaints"),
                        "visitReason": row.get("Visit Reason"),
                        "diagnosisNames": row.get("Diagnosis Names"),
                        "medicineNames": row.get("Medicine Names"),
                    })

                logger.info("Found %d past encounters in PostgreSQL for caseNo=%s", len(past_visits), case_no)
                return past_visits
        except Exception as exc:
            logger.warning("Failed querying past encounters for caseNo=%s: %s", case_no, exc)
            return []

    def find_similar_cases(
        self,
        diagnoses: List[str],
        top_k: int = 5,
        exclude_case_no: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Find top-k similar cases in PostgreSQL matching any input diagnosis, newest first, deduplicated by caseNo."""
        clean_diagnoses = [d.strip() for d in diagnoses if d and d.strip()]
        if not clean_diagnoses:
            return []

        diag_clauses = []
        params: Dict[str, Any] = {"top_k": top_k}
        for idx, diag in enumerate(clean_diagnoses):
            param_key = f"diag_{idx}"
            diag_clauses.append(f'p_dx."Diagnosis Names" ILIKE :{param_key}')
            params[param_key] = f"%{diag}%"

        where_diag_condition = " OR ".join(diag_clauses)

        exclude_clause = ""
        if exclude_case_no is not None:
            dt_exclude = format_dt_case_no(exclude_case_no)
            exclude_clause = 'AND p_vr."Case No" != :curr_case_no'
            params["curr_case_no"] = dt_exclude


        query_sql = f"""
        WITH p_info AS (
            SELECT pd."case No", pd."Gender", pd."Age Year" AS "Age"
            FROM temp_migrations.patient_data pd
        ), 
        p_cc AS (
            SELECT pcd."Case No", pcd."Visit Date", pcd."Chief Complaints" 
            FROM temp_migrations.patient_chiefcomplains_data pcd
        ), 
        p_dx AS (
            SELECT pdd."Case No", pdd."Visit Date", string_agg(pdd."Diagnosis Name", ', ') AS "Diagnosis Names"  
            FROM temp_migrations.patient_diagnosis_data pdd
            GROUP BY pdd."Case No", pdd."Visit Date"
        ), 
        p_vr AS (
            SELECT pvd."Case No", pvd."Visit Date", pvd."Visit Reason"  
            FROM temp_migrations.patient_visit_data pvd
        ), 
        p_rx AS (
            SELECT ppd."Case No", ppd."Visit Date", string_agg(ppd."Medicine Name", ', ') AS "Medicine Names"  
            FROM temp_migrations.patient_prescription_data ppd
            GROUP BY ppd."Case No", ppd."Visit Date"
        ),
        matched_encounters AS (
            SELECT 
                p_vr."Case No",
                p_vr."Visit Date",
                to_date(p_vr."Visit Date", 'DD/MM/YYYY') AS parsed_visit_date,
                p_info."Gender",
                p_info."Age",
                p_cc."Chief Complaints",
                p_vr."Visit Reason",
                p_dx."Diagnosis Names",
                p_rx."Medicine Names",
                ROW_NUMBER() OVER (
                    PARTITION BY p_vr."Case No" 
                    ORDER BY to_date(p_vr."Visit Date", 'DD/MM/YYYY') DESC
                ) AS rn
            FROM p_vr
            LEFT JOIN p_info ON p_vr."Case No" = p_info."case No"
            LEFT JOIN p_cc   ON p_vr."Case No" = p_cc."Case No"   AND p_vr."Visit Date" = p_cc."Visit Date"
            LEFT JOIN p_dx   ON p_vr."Case No" = p_dx."Case No"   AND p_vr."Visit Date" = p_dx."Visit Date"
            LEFT JOIN p_rx   ON p_vr."Case No" = p_rx."Case No"   AND p_vr."Visit Date" = p_rx."Visit Date"
            WHERE ({where_diag_condition})
            {exclude_clause}
        )
        SELECT 
            "Case No",
            "Visit Date",
            "Gender",
            "Age",
            "Chief Complaints",
            "Visit Reason",
            "Diagnosis Names",
            "Medicine Names"
        FROM matched_encounters
        WHERE rn = 1
        ORDER BY parsed_visit_date DESC
        LIMIT :top_k;
        """

        try:
            with self._postgres_source.engine.connect() as conn:
                rows = conn.execute(text(query_sql), params).mappings().all()
                results = []
                for row in rows:
                    results.append({
                        "caseNo": row.get("Case No"),
                        "visitDate": row.get("Visit Date"),
                        "gender": row.get("Gender"),
                        "age": row.get("Age"),
                        "chiefComplaints": row.get("Chief Complaints"),
                        "visitReason": row.get("Visit Reason"),
                        "diagnosisNames": row.get("Diagnosis Names"),
                        "medicineNames": row.get("Medicine Names"),
                    })
                logger.info("Found %d similar cases in PostgreSQL matching diagnoses %s", len(results), clean_diagnoses)
                return results
        except Exception as exc:
            logger.error("Failed querying similar cases in PostgreSQL: %s", exc, exc_info=True)
            return []

    def load_medicine_master(self, file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load and return local medicine master dataset from JSON."""
        target_path = Path(file_path or self._settings.medicine_master_json_path)
        if not target_path.is_file():
            repo_root = Path(__file__).resolve().parent.parent.parent.parent
            candidate = repo_root / target_path
            if candidate.is_file():
                target_path = candidate
        if target_path.is_file():
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception as exc:
                logger.warning("Failed to load local medicine master from %s: %s", target_path, exc)
        return []

    def prepare_prompt(
        self,
        decorated_case: DecoratedCase,
        past_cases: List[Dict[str, Any]],
        similar_cases: List[Dict[str, Any]],
        system_text: Optional[str] = None,
        medicine_master: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Construct the prompt combining current case, past visits, similar cases, and local medicine master."""
        prompt_lines: List[str] = []

        # 1. System & contextual instructions
        clinical_instruction = (system_text if system_text is not None else get_active_prompt()).strip()
        if clinical_instruction:
            prompt_lines.append(clinical_instruction)
            prompt_lines.append("")

        # 2. Input data for current encounter
        prompt_lines.append("<data>")
        prompt_lines.append("### CURRENT ENCOUNTER:")
        if decorated_case.caseNo is not None:
            prompt_lines.append(f"- Case No: {format_dt_case_no(decorated_case.caseNo)}")
        if decorated_case.visitType:
            prompt_lines.append(f"- Visit Type: {decorated_case.visitType}")


        p_info = decorated_case.patientInfo
        age_str = f"{p_info.age} years" if p_info.age is not None else "Unknown"
        gender_str = p_info.gender if p_info.gender else "Unknown"
        prompt_lines.append(f"- Demographics: Age: {age_str}, Gender: {gender_str}")

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

        # 3. Past visits for this patient
        prompt_lines.append("### PATIENT PAST ENCOUNTER HISTORY:")
        if past_cases:
            for idx, past in enumerate(past_cases, start=1):
                v_date = past.get("visitDate") or "Unknown date"
                v_reason = past.get("visitReason") or "Encounter"
                diag_names = past.get("diagnosisNames") or "None recorded"
                cc = past.get("chiefComplaints")
                meds_str = past.get("medicineNames") or "None recorded"

                prompt_lines.append(f"- Visit #{idx} ({v_date}, {v_reason}):")
                if cc:
                    prompt_lines.append(f"  * Chief Complaints: {cc}")
                prompt_lines.append(f"  * Prior Diagnoses: {diag_names}")
                prompt_lines.append(f"  * Medications Prescribed: {meds_str}")
        else:
            prompt_lines.append("- No prior encounters recorded for this patient.")
        prompt_lines.append("")

        # 4. Similar cases from PostgreSQL
        prompt_lines.append("### SIMILAR CASES WITH PRESCRIBED MEDICATIONS:")
        if similar_cases:
            for idx, cand in enumerate(similar_cases, start=1):
                c_no = cand.get("caseNo", "Unknown")
                c_date = cand.get("visitDate", "Unknown")
                c_gen = cand.get("gender", "Unknown")
                c_age = cand.get("age") or "N/A"
                c_reason = cand.get("visitReason", "Encounter")
                cc = cand.get("chiefComplaints")
                diags = cand.get("diagnosisNames") or "None recorded"
                meds = cand.get("medicineNames") or "None recorded"

                prompt_lines.append(f"- Similar Case #{idx} (Case: {c_no}, Date: {c_date}, Gender: {c_gen}, Age: {c_age}, Reason: {c_reason}):")
                if cc:
                    prompt_lines.append(f"  * Chief Complaints: {cc}")
                prompt_lines.append(f"  * Diagnoses: {diags}")
                prompt_lines.append(f"  * Prescribed Medications: {meds}")
        else:
            prompt_lines.append("- No similar cases identified in the database.")
        prompt_lines.append("</data>")
        prompt_lines.append("")

        # 5. Local medicine master (formulary dataset)
        med_master = medicine_master if medicine_master is not None else self.load_medicine_master()
        if med_master:
            prompt_lines.append("<medication-master>")
            prompt_lines.append("```json")
            prompt_lines.append(json.dumps(med_master, indent=2))
            prompt_lines.append("```")
            prompt_lines.append("</medication-master>")
            prompt_lines.append("")

        # 6. Output format constraint
        prompt_lines.append("### RESPONSE FORMAT:")
        prompt_lines.append(
            "Respond strictly in valid JSON adhering to this schema:\n"
            "{\n"
            '  "chain_of_thought": {\n'
            '    "hypothesis": "What the picture suggests based on age, sex, diagnoses, and symptoms.",\n'
            '    "history_analysis": "What changed/persisted from past visits and implied actions.",\n'
            '    "similar_cases_analysis": "Pattern of drug classes/brands used by this doctor for this profile.",\n'
            '    "formulary_match": "Matching patient symptoms/diagnoses with available brands in the local medicine master.",\n'
            '    "research": "Generic composition and cautions for the considered drugs."\n'
            "  },\n"
            '  "missing_information": ["List of data points the physician should verify"],\n'
            '  "recommended_medications": [\n'
            "    {\n"
            '      "brand_name": "Exact brand / medicine name from local medicine master or history",\n'
            '      "basis": "patient_history OR similar_cases OR formulary_match OR outside_pattern",\n'
            '      "cautions": "Safety concerns or drug conflicts, if any",\n'
            '      "depends_on_missing_info": true OR false\n'
            "    }\n"
            "  ],\n"
            '  "medications": [\n'
            "    {\n"
            '      "brand_name": "Exact brand_name from local medicine master",\n'
            '      "canonical_molecule": "Exact canonical_molecule from local medicine master",\n'
            '      "is_fdc": false,\n'
            '      "constituent_strengths_mg": {\n'
            '        "<molecule_name>": 0.0\n'
            "      },\n"
            '      "modal_strength_mg": 0.0,\n'
            '      "release_form": "Exact release_form from local medicine master (e.g., Regular, SR, ER)",\n'
            '      "dosage_tier": "Exact dosage_tier from local medicine master (e.g., STANDARD_MAINTENANCE)",\n'
            '      "dosage_group_id": "Exact dosage_group_id from local medicine master",\n'
            '      "clinical_dosing_indication": "Exact clinical_dosing_indication from local medicine master"\n'
            "    }\n"
            "  ],\n"
            '  "explanation": "Clinical justification explaining the final selection."\n'
            "}\n"
            "CRITICAL: 'medications' must return a list of full objects of each medication master item exactly as present in <medication-master> rather than just the brand or medicine name string."
        )

        return "\n".join(prompt_lines)

    def prescribe_medications(
        self,
        request: PrescribeMedicationsRequest,
    ) -> PrescribeMedicationsResponse:
        """Validate input, decorate case, find similar cases in PostgreSQL, build prompt, log/print prompt, and return."""
        case_input = request.case

        clean_symptoms = [s.strip() for s in case_input.symptoms if s and s.strip()]
        from core.utils import clean_diagnosis_term
        clean_diagnoses = [clean_diagnosis_term(d) for d in case_input.diagnosis if clean_diagnosis_term(d)]

        # Validation requirement: at least 1 symptom and 1 diagnosis required
        if not clean_diagnoses or not clean_symptoms:
            raise HTTPException(
                status_code=400,
                detail="At least 1 symptom and 1 diagnosis is required.",
            )

        # 1. Decorate input case with HPO (symptoms) and MONDO (diagnoses)
        visit_type_str = case_input.visitType.value if hasattr(case_input.visitType, "value") else str(case_input.visitType)
        decorated_case = self._case_service.decorate_case(
            case_no=case_input.caseNo,
            patient_info=case_input.patientInfo,
            symptoms=clean_symptoms,
            diagnosis=clean_diagnoses,
            visit_type=visit_type_str,
        )

        # Fetch past cases from PostgreSQL (if caseNo is provided)
        past_cases = self.fetch_past_encounters(
            case_no=case_input.caseNo,
            case_date=case_input.caseDate,
        )

        # 2. Query similar cases from PostgreSQL
        similar_cases = self.find_similar_cases(
            diagnoses=clean_diagnoses,
            top_k=request.top_k,
            exclude_case_no=case_input.caseNo,
        )

        logger.info(
            "Context assembled for caseNo=%s: %d past encounters, %d similar cases",
            decorated_case.caseNo, len(past_cases), len(similar_cases)
        )

        # 3. Assemble prompt including local medicine master catalog
        medicine_master = self.load_medicine_master()
        prompt = self.prepare_prompt(
            decorated_case=decorated_case,
            past_cases=past_cases,
            similar_cases=similar_cases,
            medicine_master=medicine_master,
        )

        # 4. Generate run ID and save executed prompt to tmp file
        run_id = generate_run_id(case_no=decorated_case.caseNo)
        set_llm_run_context(run_id, call_name="postgres_prescription_synthesis")
        save_prompt_to_tmp(prompt, run_id=run_id)

        # 5. Call Gemini LLM and dump raw response to tmp file
        try:
            llm_result = self._gemini_service.generate_prescription(
                prompt,
                run_id=run_id,
                call_name="postgres_prescription_synthesis",
            )
            save_response_to_tmp(llm_result.get("raw_response", ""), run_id=run_id)
        except Exception as exc:
            save_response_to_tmp(f"ERROR: {exc}", run_id=run_id)
            raise

        medications = llm_result.get("medications", [])
        if not medications and "recommended_medications" in llm_result:
            recs = llm_result.get("recommended_medications")
            if isinstance(recs, list):
                medications = [
                    item.get("medication_master_item") or item.get("medication") or item
                    if (isinstance(item, dict) and (item.get("medication_master_item") or item.get("medication") or len(item) > 2))
                    else (item.get("brand_name") if isinstance(item, dict) else str(item))
                    for item in recs
                    if (isinstance(item, dict) and item.get("brand_name")) or (isinstance(item, str) and item.strip()) or (isinstance(item, dict) and len(item) > 1)
                ]
        explanation = llm_result.get("explanation", "")
        response_time = llm_result.get("response_time_seconds")
        tokens_used = llm_result.get("tokens_used")
        cost_usd = llm_result.get("cost_usd")
        total_tokens = tokens_used.get("total_tokens") if isinstance(tokens_used, dict) else None

        return PrescribeMedicationsResponse(
            medications=medications,
            explanation=explanation,
            pastCasesCount=len(past_cases),
            similarCasesCount=len(similar_cases),
            responseTimeSeconds=response_time,
            tokensUsed=tokens_used,
            totalTokens=total_tokens,
            costUsd=cost_usd,
            runId=run_id,
        )
