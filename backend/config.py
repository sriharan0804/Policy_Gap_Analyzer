from functools import lru_cache
from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PGA_",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "AI-Assisted Regulatory Policy Gap Analyzer"
    app_version: str = "0.1.0"

    app_env: Literal["development", "test", "production"] = "development"
    debug: bool = False
    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    data_directory: Path = Path("data")
    regulation_directory: Path = Path("data/regulations")
    policy_directory: Path = Path("data/policies")
    processed_directory: Path = Path("data/processed")
    faiss_directory: Path = Path("data/faiss")

    max_upload_size_mb: int = Field(default=25, ge=1, le=200)
    retrieval_top_k: int = Field(default=5, ge=1, le=50)
    minimum_retrieval_score: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
    )

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    llm_provider: Literal["openai", "groq", "disabled"] = "disabled"
    openai_api_key: str | None = Field(default=None, repr=False)
    groq_api_key: str | None = Field(default=None, repr=False)

    def create_data_directories(self) -> None:
        """Create application data directories."""

        directories = (
            self.data_directory,
            self.regulation_directory,
            self.policy_directory,
            self.processed_directory,
            self.faiss_directory,
        )

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""

    return Settings()
