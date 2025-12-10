"""Application configuration loaded from environment.

This module centralizes all runtime configuration including:
- Chroma host/port and collection name
- Retrieval tuning parameters
- Observability/logging options

These values can be overridden via environment variables so the same
container image can be reused across environments.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # General app metadata
    app_name: str = Field("Utkrusht RAG Assistant", env="APP_NAME")
    app_version: str = Field("1.0.0", env="APP_VERSION")

    # ChromaDB connection
    chroma_host: str = Field("chroma", env="CHROMA_HOST")
    chroma_port: int = Field(8000, env="CHROMA_PORT")
    chroma_collection: str = Field(
        "utkrusht_knowledge", env="CHROMA_COLLECTION"
    )

    # Retrieval tuning
    retrieval_default_top_k: int = Field(8, env="RETRIEVAL_DEFAULT_TOP_K")
    retrieval_max_k: int = Field(12, env="RETRIEVAL_MAX_K")
    retrieval_min_score: float = Field(
        0.3,
        env="RETRIEVAL_MIN_SCORE",
        description=(
            "Minimum similarity score (0-1) required to keep a chunk. "
            "Chunks scoring below this are discarded to avoid noise."
        ),
    )
    max_context_characters: int = Field(
        6000,
        env="MAX_CONTEXT_CHARACTERS",
        description="Upper bound on total context size passed to generation.",
    )

    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Return a cached application settings instance.

    Using `lru_cache` ensures environment variables are only read once
    and the same config object is reused across the app.
    """

    return Settings()  # type: ignore[call-arg]
