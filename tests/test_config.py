"""Unit tests for configuration and settings."""

from meldai.config import Settings


def test_settings_default_values():
    settings = Settings(
        postgres_host="test-postgres",
        mongo_host="test-mongo",
        debug_mode=True,
    )
    assert settings.postgres_host == "test-postgres"
    assert "postgresql+psycopg://" in settings.postgres_sync_url
    assert "test-postgres" in settings.postgres_sync_url
    assert "mongodb://" in settings.resolved_mongo_uri
    assert settings.debug_port == 5678
