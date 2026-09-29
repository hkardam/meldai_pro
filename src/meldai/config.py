"""Application configuration powered by Pydantic Settings."""

from functools import lru_cache
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Environment & Logging
    environment: str = Field(default="development", description="Current runtime environment")
    log_level: str = Field(default="INFO", description="Standard logging level")

    # Debugging
    debug_mode: bool = Field(default=False, description="Whether debugpy is enabled")
    debug_port: int = Field(default=5678, description="Port for remote debugpy attach")
    wait_for_debugger: bool = Field(default=False, description="Pause startup until IDE attaches")

    # PostgreSQL (Training Data Source)
    postgres_user: str = Field(default="meldai_user")
    postgres_password: str = Field(default="meldai_secure_pass")
    postgres_db: str = Field(default="meldai_training_source")
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)

    @property
    def postgres_sync_url(self) -> str:
        """SQLAlchemy sync connection URL for PostgreSQL (psycopg)."""
        return f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def postgres_async_url(self) -> str:
        """SQLAlchemy async connection URL for PostgreSQL (asyncpg)."""
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    # MongoDB (Sink & Knowledge Base)
    mongo_root_user: str = Field(default="admin")
    mongo_root_password: str = Field(default="admin_secure_pass")
    mongo_host: str = Field(default="localhost")
    mongo_port: int = Field(default=27017)
    mongo_db_name: str = Field(default="meldai_knowledge_base")
    mongo_uri: Optional[str] = Field(default=None)

    @property
    def resolved_mongo_uri(self) -> str:
        """Resolved MongoDB connection URI."""
        if self.mongo_uri:
            return self.mongo_uri
        return f"mongodb://{self.mongo_root_user}:{self.mongo_root_password}@{self.mongo_host}:{self.mongo_port}/{self.mongo_db_name}?authSource=admin"

    # Clinical NLP (SapBERT)
    sapbert_model_name: str = Field(
        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        description="SapBERT model for clinical entity representation"
    )
    device: str = Field(default="cpu", description="Inference device: cpu or cuda")
    embedding_batch_size: int = Field(default=16, description="Batch size for embedding generation")

    # Terminology Service (HPO + MONDO — offline OBO files)
    hpo_obo_path: str = Field(
        default="data/ontologies/hp.obo",
        description="Local path to the HPO OBO file (hp.obo)",
    )
    mondo_obo_path: str = Field(
        default="data/ontologies/mondo.obo",
        description="Local path to the MONDO OBO file (mondo.obo)",
    )


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
