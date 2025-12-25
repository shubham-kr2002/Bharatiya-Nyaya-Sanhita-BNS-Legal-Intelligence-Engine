"""Embedding service module for vector generation using HuggingFace models."""

from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings

from core.config import get_settings
from core.exceptions import EmbeddingError
from core.logger import get_logger, LoggerMixin
from models.schema import LegalChunk

logger = get_logger(__name__)


class EmbeddingService(LoggerMixin):
    """
    Reusable service for generating embeddings using HuggingFace models.
    
    Uses BAAI/bge-small-en-v1.5 by default, optimized for semantic search
    with normalized embeddings for cosine similarity.
    """

    # Default model configuration
    DEFAULT_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    DEFAULT_DEVICE: str = "cpu"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the EmbeddingService with HuggingFace model.
        
        Args:
            model_name: HuggingFace model name (default: BAAI/bge-small-en-v1.5).
            device: Device to run model on (default: cpu).
        """
        self._settings = get_settings()
        self._model_name = model_name or self._settings.EMBEDDING_MODEL_NAME or self.DEFAULT_MODEL_NAME
        self._device = device or self.DEFAULT_DEVICE
        self._model: Optional[HuggingFaceEmbeddings] = None

        self.logger.info(
            "EmbeddingService initialized",
            model_name=self._model_name,
            device=self._device,
        )

    @property
    def model(self) -> HuggingFaceEmbeddings:
        """
        Lazily initialize and return the embedding model.
        
        Returns:
            HuggingFaceEmbeddings: Configured embedding model.
            
        Raises:
            EmbeddingError: If model loading fails.
        """
        if self._model is None:
            try:
                self._model = HuggingFaceEmbeddings(
                    model_name=self._model_name,
                    model_kwargs={"device": self._device},
                    # Crucial: Normalize embeddings for cosine similarity
                    encode_kwargs={"normalize_embeddings": True},
                )
                self.logger.info(
                    "Embedding model loaded",
                    model_name=self._model_name,
                    device=self._device,
                )
            except Exception as e:
                self.logger.error(
                    "Failed to load embedding model",
                    model_name=self._model_name,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise EmbeddingError(
                    message=f"Failed to load embedding model: {self._model_name}",
                    details={"model_name": self._model_name, "error": str(e)},
                ) from e
        return self._model

    def embed_query(self, text: str) -> list[float]:
        """
        Generate embedding for a single query text.
        
        Args:
            text: User question or query string.
            
        Returns:
            list[float]: Normalized embedding vector.
            
        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty")

        try:
            embedding = self.model.embed_query(text)
            
            self.logger.debug(
                "Query embedded",
                text_length=len(text),
                embedding_dim=len(embedding),
            )
            
            return embedding

        except ValueError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to embed query",
                text_length=len(text),
                error=str(e),
                error_type=type(e).__name__,
            )
            raise EmbeddingError(
                message="Failed to generate query embedding",
                details={"text_length": len(text), "error": str(e)},
            ) from e

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            list[list[float]]: List of normalized embedding vectors.
            
        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not texts:
            return []

        try:
            embeddings = self.model.embed_documents(texts)
            
            self.logger.info(
                "Texts embedded",
                text_count=len(texts),
                embedding_dim=len(embeddings[0]) if embeddings else 0,
            )
            
            return embeddings

        except Exception as e:
            self.logger.error(
                "Failed to embed texts",
                text_count=len(texts),
                error=str(e),
                error_type=type(e).__name__,
            )
            raise EmbeddingError(
                message="Failed to generate text embeddings",
                details={"text_count": len(texts), "error": str(e)},
            ) from e

    def embed_batch(
        self,
        chunks: list[LegalChunk],
        batch_size: int = 32,
    ) -> list[LegalChunk]:
        """
        Generate embeddings for a batch of LegalChunk objects.
        
        Extracts text from chunks, generates embeddings, and assigns
        the vectors back to each chunk's embedding field.
        
        Args:
            chunks: List of LegalChunk objects to embed.
            batch_size: Number of texts to embed at once (for memory management).
            
        Returns:
            list[LegalChunk]: Same chunks with embeddings populated.
            
        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not chunks:
            self.logger.warning("No chunks provided for embedding")
            return []

        try:
            # Extract texts from all chunks
            texts = [chunk.text for chunk in chunks]
            
            self.logger.info(
                "Starting batch embedding",
                chunk_count=len(chunks),
                batch_size=batch_size,
            )

            # Process in batches to manage memory
            all_embeddings: list[list[float]] = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_embeddings = self.model.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                self.logger.debug(
                    "Batch processed",
                    batch_start=i,
                    batch_size=len(batch_texts),
                    total_processed=len(all_embeddings),
                )

            # Assign embeddings back to chunks
            for chunk, embedding in zip(chunks, all_embeddings):
                chunk.embedding = embedding

            self.logger.info(
                "Batch embedding completed",
                chunk_count=len(chunks),
                embedding_dim=len(all_embeddings[0]) if all_embeddings else 0,
            )

            return chunks

        except Exception as e:
            self.logger.error(
                "Failed to embed batch",
                chunk_count=len(chunks),
                error=str(e),
                error_type=type(e).__name__,
            )
            raise EmbeddingError(
                message="Failed to generate batch embeddings",
                details={"chunk_count": len(chunks), "error": str(e)},
            ) from e

    @property
    def embedding_dimension(self) -> int:
        """
        Get the embedding dimension for the current model.
        
        Returns:
            int: Dimension of embedding vectors (384 for BGE-Small).
        """
        return self._settings.EMBEDDING_DIMENSION
