"""Custom exceptions for the Legal Agent RAG system."""

from typing import Any


class LegalAgentException(Exception):
    """Base exception for Legal Agent application."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(LegalAgentException):
    """Raised when there's a configuration error."""

    pass


class DatabaseConnectionError(LegalAgentException):
    """Raised when database connection fails."""

    pass


class QdrantError(DatabaseConnectionError):
    """Raised when Qdrant operations fail."""

    pass


class RedisError(DatabaseConnectionError):
    """Raised when Redis operations fail."""

    pass


class IngestionError(LegalAgentException):
    """Raised when document ingestion fails."""

    pass


class PDFParsingError(IngestionError):
    """Raised when PDF parsing fails."""

    pass


class ChunkingError(IngestionError):
    """Raised when document chunking fails."""

    pass


class LLMError(LegalAgentException):
    """Raised when LLM operations fail."""

    pass


class EmbeddingError(LegalAgentException):
    """Raised when embedding generation fails."""

    pass


class SearchError(LegalAgentException):
    """Raised when vector search fails."""

    pass
