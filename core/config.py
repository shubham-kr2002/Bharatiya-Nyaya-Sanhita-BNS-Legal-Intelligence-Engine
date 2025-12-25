"""Configuration module using pydantic-settings for environment validation."""

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _load_streamlit_secrets() -> None:
    """Load Streamlit secrets into environment variables if available."""
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for key in ["GROQ_API_KEY", "LLAMA_CLOUD_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "REDIS_URL"]:
                if key in st.secrets and key not in os.environ:
                    os.environ[key] = st.secrets[key]
    except Exception:
        pass  # Not running in Streamlit or secrets not configured


# Attempt to load Streamlit secrets before Settings initialization
_load_streamlit_secrets()


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All required environment variables must be set for the application to start.
    Use a .env file for local development.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ============================================
    # Application Settings
    # ============================================
    APP_NAME: str = Field(default="Legal Agent RAG", description="Application name")
    APP_VERSION: str = Field(default="0.1.0", description="Application version")
    ENVIRONMENT: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment",
    )
    DEBUG: bool = Field(default=False, description="Debug mode flag")
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )

    # ============================================
    # API Keys (Required)
    # ============================================
    GROQ_API_KEY: str = Field(
        ...,
        description="Groq API key for LLM access",
        min_length=1,
    )
    LLAMA_CLOUD_API_KEY: str = Field(
        ...,
        description="LlamaCloud API key for document parsing",
        min_length=1,
    )

    # ============================================
    # Qdrant Vector Database (Required)
    # ============================================
    QDRANT_URL: str = Field(
        ...,
        description="Qdrant server URL",
        min_length=1,
    )
    QDRANT_API_KEY: str = Field(
        ...,
        description="Qdrant API key for authentication",
        min_length=1,
    )
    QDRANT_COLLECTION_NAME: str = Field(
        default="indian_law_bns",
        description="Qdrant collection name for storing embeddings",
    )

    # ============================================
    # Redis Cache (Required)
    # ============================================
    REDIS_URL: str = Field(
        ...,
        description="Redis connection URL",
        min_length=1,
    )
    REDIS_TTL_SECONDS: int = Field(
        default=3600,
        description="Redis cache TTL in seconds",
        ge=60,
    )

    # ============================================
    # Embedding Model Settings
    # ============================================
    EMBEDDING_MODEL_NAME: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace embedding model name",
    )
    EMBEDDING_DIMENSION: int = Field(
        default=384,
        description="Embedding vector dimension",
    )

    # ============================================
    # Chunking Settings
    # ============================================
    CHUNK_SIZE: int = Field(
        default=1000,
        description="Maximum chunk size in characters",
        ge=100,
        le=10000,
    )
    CHUNK_OVERLAP: int = Field(
        default=200,
        description="Overlap between chunks in characters",
        ge=0,
    )

    # ============================================
    # Retrieval Settings
    # ============================================
    TOP_K_RESULTS: int = Field(
        default=5,
        description="Number of top results to retrieve",
        ge=1,
        le=20,
    )

    @field_validator("CHUNK_OVERLAP")
    @classmethod
    def validate_chunk_overlap(cls, v: int, info) -> int:
        """Ensure chunk overlap is less than chunk size."""
        chunk_size = info.data.get("CHUNK_SIZE", 1000)
        if v >= chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({v}) must be less than CHUNK_SIZE ({chunk_size})"
            )
        return v

    @field_validator("QDRANT_URL", "REDIS_URL")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Basic URL validation."""
        if not v.startswith(("http://", "https://", "redis://")):
            raise ValueError(
                "URL must start with http://, https://, or redis://"
            )
        return v


@lru_cache
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Uses lru_cache to ensure settings are loaded only once.
    
    Returns:
        Settings: Application settings instance.
        
    Raises:
        ValidationError: If required environment variables are missing.
    """
    return Settings()


# Convenience export for direct import
settings = get_settings()
