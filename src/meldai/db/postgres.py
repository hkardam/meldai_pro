"""PostgreSQL data source driver for clinical training datasets."""

import logging
from typing import Any, Dict, List, Optional
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker

from meldai.config import Settings, get_settings

logger = logging.getLogger(__name__)
Base = declarative_base()


class PostgresSource:
    """Manages connections and data extraction from PostgreSQL clinical data sources."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._engine: Optional[Engine] = None
        self._sessionmaker: Optional[sessionmaker] = None

    @property
    def engine(self) -> Engine:
        """Lazy synchronous SQLAlchemy Engine with connection pooling."""
        if self._engine is None:
            logger.info("Initializing PostgreSQL engine for %s:%s", self.settings.postgres_host, self.settings.postgres_port)
            self._engine = create_engine(
                self.settings.postgres_sync_url,
                pool_pre_ping=True,
                pool_size=10,
                max_overflow=20,
                connect_args={"connect_timeout": 3},
            )
        return self._engine

    def ping(self) -> bool:
        """Check if PostgreSQL source is healthy and accessible."""
        try:
            with self.engine.connect() as conn:
                res = conn.execute(text("SELECT 1")).scalar()
                return res == 1
        except Exception as exc:
            logger.warning("PostgreSQL ping failed: %s", exc)
            return False

    def load_training_records(self, limit: int = 100) -> pd.DataFrame:
        """
        Fetch clinical records from PostgreSQL table 'clinical_training_records'
        returns a pandas DataFrame for statistical/model training analysis.
        """
        query = text("""
            SELECT
                id,
                patient_id,
                encounter_id,
                clinical_note,
                suspected_condition,
                icd10_code,
                concept_id,
                created_at
            FROM clinical_training_records
            ORDER BY id ASC
            LIMIT :limit
        """)

        with self.engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"limit": limit})
            logger.info("Extracted %d clinical records from PostgreSQL", len(df))
            return df

    def get_patient_migration_count(self) -> int:
        """Get total record count for patient encounters to migrate from PostgreSQL."""
        query = text('SELECT COUNT(*) FROM temp_migrations.patient_visit_data')
        with self.engine.connect() as conn:
            return conn.execute(query).scalar() or 0

    def stream_patient_migration_records(self, batch_size: int = 20, limit: Optional[int] = None):
        """
        Extract unified patient records by joining demographic, encounter, diagnosis,
        prescription, chief complaint, and investigation data from PostgreSQL temp_migrations schema.
        Yields batches of dicts in chunks of batch_size (default 20).
        """
        query_str = """
        WITH p_info AS (
            SELECT pd."case No", pd."Gender", pd."Age Day", pd."Age Month", pd."Age Year" 
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
        p_inv AS (
            SELECT pid."CaseNo", pid."Visit Date", string_agg(pid."Investigation Name", ', ') AS "Investigation Names"
            FROM temp_migrations.patient_investigation_data pid
            GROUP BY pid."CaseNo", pid."Visit Date"
        )
        SELECT 
            p_vr."Case No",
            p_vr."Visit Date",
            p_info."Gender",
            p_info."Age Day",
            p_info."Age Month",
            p_info."Age Year",
            p_cc."Chief Complaints",
            p_vr."Visit Reason",
            p_dx."Diagnosis Names",
            p_rx."Medicine Names",
            p_inv."Investigation Names",
            row_number() OVER (PARTITION BY p_vr."Case No" ORDER BY p_vr."Visit Date" DESC) AS "Is Latest"
        FROM p_vr
        LEFT JOIN p_info ON p_vr."Case No" = p_info."case No"
        LEFT JOIN p_cc ON p_vr."Case No" = p_cc."Case No" AND p_vr."Visit Date" = p_cc."Visit Date"
        LEFT JOIN p_dx ON p_vr."Case No" = p_dx."Case No" AND p_vr."Visit Date" = p_dx."Visit Date"
        LEFT JOIN p_rx ON p_vr."Case No" = p_rx."Case No" AND p_vr."Visit Date" = p_rx."Visit Date"
        LEFT JOIN p_inv ON p_vr."Case No" = p_inv."CaseNo" AND p_vr."Visit Date" = p_inv."Visit Date"
        ORDER BY p_vr."Case No"
        """
        if limit is not None and limit > 0:
            query_str += f" LIMIT {int(limit)}"

        def _parse_case_no(val: Optional[str]) -> Any:
            if val is None:
                return 0
            if isinstance(val, int):
                return val
            s_val = str(val).strip()
            if s_val.isdigit():
                return int(s_val)
            digits = "".join(filter(str.isdigit, s_val))
            if digits:
                return int(digits)
            return s_val

        def _parse_age(year: Any, month: Any, day: Any) -> float:
            try:
                y = float(year) if year is not None else 0.0
            except (ValueError, TypeError):
                y = 0.0
            try:
                m = float(month) if month is not None else 0.0
            except (ValueError, TypeError):
                m = 0.0
            try:
                d = float(day) if day is not None else 0.0
            except (ValueError, TypeError):
                d = 0.0
            return round(y + (m / 12.0) + (d / 365.25), 2)

        def _parse_gender(val: Any) -> str:
            if not val:
                return "m"
            s = str(val).strip().lower()
            if s.startswith("f"):
                return "f"
            return "m"

        query = text(query_str)
        with self.engine.connect() as conn:
            result = conn.execution_options(yield_per=batch_size).execute(query)
            while True:
                rows = result.fetchmany(batch_size)
                if not rows:
                    break
                batch = []
                for row in rows:
                    case_no = _parse_case_no(row[0])
                    visit_date = str(row[1]) if row[1] is not None else ""
                    gender = _parse_gender(row[2])
                    age_day = row[3]
                    age_month = row[4]
                    age_year = row[5]
                    age = _parse_age(age_year, age_month, age_day)
                    chief_complaints = row[6]
                    visit_reason = row[7]
                    diagnosis_names = row[8]
                    medicine_names = row[9]
                    investigation_names = row[10]
                    is_latest = int(row[11]) if row[11] is not None else 1

                    batch.append({
                        "caseNo": case_no,
                        "visitDate": visit_date,
                        "gender": gender,
                        "ageDay": age_day,
                        "ageMonth": age_month,
                        "ageYear": age_year,
                        "age": age,
                        "chiefComplaints": chief_complaints,
                        "visitReason": visit_reason,
                        "diagnosisNames": diagnosis_names,
                        "medicineNames": medicine_names,
                        "investigationNames": investigation_names,
                        "isLatest": is_latest,
                    })
                logger.info("Fetched batch of %d unified patient migration records from PostgreSQL", len(batch))
                yield batch

    def close(self) -> None:
        """Dispose of the connection pool."""
        if self._engine:
            self._engine.dispose()
            self._engine = None


