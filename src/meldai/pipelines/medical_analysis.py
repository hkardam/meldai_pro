"""End-to-End Medical Science Pipeline:

PostgreSQL (External Source) -> SapBERT Embeddings -> MongoDB (Internal Case Knowledge Base)
"""

from datetime import datetime, timezone
import logging
from typing import List, Optional
import uuid
from rich.console import Console
from rich.table import Table

from meldai.config import Settings, get_settings
from meldai.db.postgres import PostgresSource
from meldai.db.mongodb import (
    MongoKnowledgeBase,
    CaseDocument,
    PatientInfo,
    PatientAge,
    ClinicalEntityItem,
)
from meldai.nlp.sapbert import SapBERTEmbedder

logger = logging.getLogger(__name__)
console = Console()


class MedicalAnalysisPipeline:
    """Orchestrates clinical data extraction, SapBERT embedding, and Case Document ingestion."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        postgres_source: Optional[PostgresSource] = None,
        mongo_sink: Optional[MongoKnowledgeBase] = None,
        sapbert_embedder: Optional[SapBERTEmbedder] = None,
    ):
        self.settings = settings or get_settings()
        self.postgres = postgres_source or PostgresSource(self.settings)
        self.mongo = mongo_sink or MongoKnowledgeBase(self.settings)
        self.embedder = sapbert_embedder or SapBERTEmbedder(self.settings)

    def _annotate_entity(self, text: str) -> ClinicalEntityItem:
        """Embed text with SapBERT and return a ClinicalEntityItem."""
        vec = self.embedder.embed_entities(text)[0].tolist()
        return ClinicalEntityItem(text=text, embedding=vec)

    def process_case(
        self,
        case_no: int,
        visit_date: str,
        patient_name: str,
        patient_gender: str,
        age_years: int,
        age_months: int = 0,
        age_days: int = 0,
        symptoms_text: Optional[List[str]] = None,
        diagnosis_text: Optional[List[str]] = None,
    ) -> CaseDocument:
        """Construct, embed, and return a validated CaseDocument."""
        symptoms_list: List[ClinicalEntityItem] = [
            self._annotate_entity(s) for s in (symptoms_text or [])
        ]
        diagnosis_list: List[ClinicalEntityItem] = [
            self._annotate_entity(d) for d in (diagnosis_text or [])
        ]

        return CaseDocument(
            id=str(uuid.uuid4()),
            caseNo=case_no,
            visitDate=visit_date,
            patient=PatientInfo(
                name=patient_name,
                gender=patient_gender,
                age=PatientAge(year=age_years, month=age_months, day=age_days),
            ),
            symptoms=symptoms_list,
            diagnosis=diagnosis_list,
        )

    def run(self, limit: int = 5) -> List[str]:
        """
        Execute pipeline:
        1. Extract training cohort from external PostgreSQL (or use demo cohort)
        2. Annotate symptoms & diagnosis with SapBERT embeddings
        3. Ingest standardized case documents into MongoDB Knowledge Base
        """
        console.rule("[bold cyan]Executing MeldAI Medical Science Pipeline[/bold cyan]")

        # 1. Attempt extraction from PostgreSQL Source
        console.print(f"📥 [bold yellow]Step 1/2:[/bold yellow] Querying external PostgreSQL ({self.settings.postgres_host}:{self.settings.postgres_port})...")
        cases_to_ingest: List[CaseDocument] = []

        try:
            df = self.postgres.load_training_records(limit=limit)
            if not df.empty:
                console.print(f"   ✓ Extracted [green]{len(df)}[/green] clinical cases from PostgreSQL.")
                for idx, row in df.iterrows():
                    case_doc = self.process_case(
                        case_no=int(row.get("id", idx + 1001)),
                        visit_date=str(row.get("created_at", datetime.now(timezone.utc).strftime("%Y-%m-%d")))[:10],
                        patient_name=str(row.get("patient_id", f"Patient-{idx+1}")),
                        patient_gender="Unknown",
                        age_years=50,
                        symptoms_text=[row.get("clinical_note", "Unspecified symptoms")[:60]],
                        diagnosis_text=[row.get("suspected_condition", "Unspecified diagnosis")],
                    )
                    cases_to_ingest.append(case_doc)
            else:
                console.print("[yellow]PostgreSQL returned 0 records. Creating standardized clinical demonstration cohort.[/yellow]")
        except Exception as exc:
            console.print(f"[yellow]Could not fetch from external PostgreSQL ({exc}). Utilizing local demo clinical cohort.[/yellow]")

        # Demo cohort if PostgreSQL is offline / empty
        if not cases_to_ingest:
            sample_cases_data = [
                {
                    "caseNo": 101,
                    "visitDate": "2026-09-20",
                    "name": "John Doe",
                    "gender": "Male",
                    "age": (58, 6, 14),
                    "symptoms": ["chest pain", "substernal chest pressure"],
                    "diagnosis": ["acute myocardial infarction"],
                },
                {
                    "caseNo": 102,
                    "visitDate": "2026-09-21",
                    "name": "Mary Jane",
                    "gender": "Female",
                    "age": (47, 2, 5),
                    "symptoms": ["polyuria", "polydipsia"],
                    "diagnosis": ["type 2 diabetes mellitus"],
                },
                {
                    "caseNo": 103,
                    "visitDate": "2026-09-22",
                    "name": "Robert Smith",
                    "gender": "Male",
                    "age": (65, 11, 20),
                    "symptoms": ["dyspnea", "wheezing"],
                    "diagnosis": ["chronic obstructive pulmonary disease"],
                },
            ]

            for sc in sample_cases_data[:limit]:
                case_doc = self.process_case(
                    case_no=sc["caseNo"],
                    visit_date=sc["visitDate"],
                    patient_name=sc["name"],
                    patient_gender=sc["gender"],
                    age_years=sc["age"][0],
                    age_months=sc["age"][1],
                    age_days=sc["age"][2],
                    symptoms_text=sc["symptoms"],
                    diagnosis_text=sc["diagnosis"],
                )
                cases_to_ingest.append(case_doc)

        # 2. Sink into MongoDB Knowledge Base
        console.print("💾 [bold yellow]Step 2/2:[/bold yellow] Ingesting validated Case Documents into MongoDB Knowledge Base...")
        inserted_ids = self.mongo.insert_cases(cases_to_ingest)

        # Summary Table
        summary_table = Table(title="Ingested MongoDB Medical Cases", show_lines=True)
        summary_table.add_column("Case No", style="cyan")
        summary_table.add_column("Patient", style="white")
        summary_table.add_column("Age (Y/M/D)", style="dim")
        summary_table.add_column("Symptoms", style="magenta")
        summary_table.add_column("Diagnosis", style="green")

        for c in cases_to_ingest:
            s_str = ", ".join(s.text for s in c.symptoms)
            d_str = ", ".join(d.text for d in c.diagnosis)
            summary_table.add_row(
                str(c.caseNo),
                f"{c.patient.name} ({c.patient.gender})",
                f"{c.patient.age.year}y {c.patient.age.month}m {c.patient.age.day}d",
                s_str,
                d_str,
            )

        console.print(summary_table)
        console.print(f"✨ [bold green]Pipeline finished successfully! Ingested {len(inserted_ids)} cases into MongoDB.[/bold green]")
        return inserted_ids
