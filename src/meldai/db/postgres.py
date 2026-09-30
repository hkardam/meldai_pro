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

    def stream_patient_visit_data(self, batch_size: int = 1000):
        """
        Extract patient visit data from PostgreSQL table 'temp_migrations.patient_visit_data'
        in batches of batch_size (default 1000).
        Yields list of dicts for each batch.
        """
        query = text("""
            SELECT
                pvd."Case No",
                pvd."Visit Date",
                pvd."Visit Reason"
            FROM temp_migrations.patient_visit_data pvd
        """)

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

        with self.engine.connect() as conn:
            result = conn.execution_options(yield_per=batch_size).execute(query)
            while True:
                rows = result.fetchmany(batch_size)
                if not rows:
                    break
                batch = []
                for row in rows:
                    raw_case_no, visit_date, visit_reason = row[0], row[1], row[2]
                    case_no = _parse_case_no(raw_case_no)
                    batch.append({
                        "caseNo": case_no,
                        "visitDate": visit_date,
                        "visitReason": visit_reason,
                    })
                logger.info("Fetched batch of %d records from PostgreSQL", len(batch))
                yield batch

    def stream_grouped_patient_diagnoses(self, batch_size: int = 1000):
        """
        Extract patient diagnosis data grouped by Case No and Visit Date from 'temp_migrations.patient_diagnosis_data'.
        Aggregates distinct diagnosis names into an array per encounter.
        Yields list of dicts for each batch: [{"caseNo": int/str, "visitDate": str, "diagnosisNames": list[str]}].
        """
        query = text("""
            SELECT
                pdd."Case No",
                pdd."Visit Date",
                ARRAY_AGG(DISTINCT pdd."Diagnosis Name") AS diagnosis_names
            FROM temp_migrations.patient_diagnosis_data pdd
            WHERE pdd."Diagnosis Name" IS NOT NULL AND TRIM(pdd."Diagnosis Name") != ''
            GROUP BY pdd."Case No", pdd."Visit Date"
            ORDER BY pdd."Case No"
        """)

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

        with self.engine.connect() as conn:
            result = conn.execution_options(yield_per=batch_size).execute(query)
            while True:
                rows = result.fetchmany(batch_size)
                if not rows:
                    break
                batch = []
                for row in rows:
                    raw_case_no, visit_date, diagnosis_names = row[0], row[1], row[2]
                    case_no = _parse_case_no(raw_case_no)
                    batch.append({
                        "caseNo": case_no,
                        "visitDate": visit_date,
                        "diagnosisNames": list(diagnosis_names) if diagnosis_names else [],
                    })
                logger.info("Fetched batch of %d grouped diagnosis records from PostgreSQL", len(batch))
                yield batch

    def close(self) -> None:
        """Dispose of the connection pool."""
        if self._engine:
            self._engine.dispose()
            self._engine = None

