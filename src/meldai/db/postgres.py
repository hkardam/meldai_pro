"""PostgreSQL data source driver for clinical training datasets."""

import logging
from typing import Optional
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

    def close(self) -> None:
        """Dispose of the connection pool."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
