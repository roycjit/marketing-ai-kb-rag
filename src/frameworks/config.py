"""Application configuration and settings."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation.

    Values are loaded from environment variables first,
    then from a .env file only in development.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = Field(
        default="postgresql+psycopg2://funnel_rag:funnel_rag_local@localhost:5432/funnel_rag",
        description="PostgreSQL connection string",
    )

    # Ollama
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL",
    )
    ollama_generation_model: str = Field(
        default="mistral:7b",
        description="Model for strategy generation",
    )
    ollama_classification_model: str = Field(
        default="llama3.2:3b",
        description="Model for intent classification and query decomposition",
    )

    # Embeddings
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Sentence-transformers model name",
    )
    embedding_dimension: int = Field(
        default=384,
        ge=1,
        le=4096,
        description="Dimensionality of embedding vectors",
    )

    # Application
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    environment: Literal["development", "staging", "production", "test"] = Field(
        default="development",
        description="Runtime environment",
    )

    @field_validator("ollama_base_url")
    @classmethod
    def _validate_ollama_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_BASE_URL must start with http:// or https://")
        return v.rstrip("/")

    @property
    def project_root(self) -> Path:
        """Return the project root directory."""
        return Path(__file__).resolve().parent.parent.parent

    @property
    def data_dir(self) -> Path:
        """Return the data directory."""
        return self.project_root / "data"

    @property
    def heyflow_export_dir(self) -> Path:
        """Return the Heyflow blog export directory."""
        return self.data_dir / "heyflow_blog_export"

    @property
    def perspective_export_dir(self) -> Path:
        """Return the Perspective blog export directory."""
        return self.data_dir / "perspective_blog_export"

    @property
    def processed_dir(self) -> Path:
        """Return the processed data directory, creating it if needed."""
        path = self.data_dir / "processed"
        path.mkdir(exist_ok=True)
        return path


# Conditional .env loading: only in development to avoid overriding
# container-injected env vars in production.
if os.getenv("ENVIRONMENT", "development").lower() == "development":
    Settings.model_config["env_file"] = ".env"
else:
    Settings.model_config["env_file"] = None

# Singleton instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return cached settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Re-export commonly accessed values for backward compatibility
# New code should prefer `get_settings()` for explicit dependency injection.
_settings_instance = get_settings()
DATABASE_URL = str(_settings_instance.database_url)
OLLAMA_BASE_URL = _settings_instance.ollama_base_url
OLLAMA_GENERATION_MODEL = _settings_instance.ollama_generation_model
OLLAMA_CLASSIFICATION_MODEL = _settings_instance.ollama_classification_model
EMBEDDING_MODEL = _settings_instance.embedding_model
EMBEDDING_DIMENSION = _settings_instance.embedding_dimension
LOG_LEVEL = _settings_instance.log_level
ENVIRONMENT = _settings_instance.environment
DATA_DIR = _settings_instance.data_dir
HEYFLOW_EXPORT_DIR = _settings_instance.heyflow_export_dir
PERSPECTIVE_EXPORT_DIR = _settings_instance.perspective_export_dir
PROCESSED_DIR = _settings_instance.processed_dir
