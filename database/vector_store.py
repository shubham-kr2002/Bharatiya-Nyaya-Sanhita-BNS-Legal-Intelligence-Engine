"""Vector database module for Qdrant operations."""

from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
    PayloadSchemaType,
)

from core.config import get_settings
from core.exceptions import QdrantError, SearchError
from core.logger import get_logger, LoggerMixin
from models.schema import LegalChunk, LegalMetadata

logger = get_logger(__name__)


class VectorDB(LoggerMixin):
    """
    Handles all Qdrant vector database operations.
    
    Provides methods for collection management, document upsertion,
    and vector similarity search for the legal RAG system.
    """

    # Collection configuration
    COLLECTION_NAME: str = "indian_law"
    VECTOR_SIZE: int = 384  # BGE-Small embedding dimension
    DISTANCE_METRIC: Distance = Distance.COSINE

    # Payload index fields for efficient filtering
    INDEXED_FIELDS: list[tuple[str, PayloadSchemaType]] = [
        ("metadata.chunk_type", PayloadSchemaType.KEYWORD),
        ("metadata.section_number", PayloadSchemaType.KEYWORD),
        ("metadata.act_name", PayloadSchemaType.KEYWORD),
        ("metadata.source_document", PayloadSchemaType.KEYWORD),
        ("metadata.doc_category", PayloadSchemaType.KEYWORD),
        ("parent_id", PayloadSchemaType.KEYWORD),
    ]

    def __init__(self) -> None:
        """Initialize VectorDB with Qdrant client connection."""
        self._settings = get_settings()
        self._client: Optional[QdrantClient] = None

    @property
    def client(self) -> QdrantClient:
        """
        Lazily initialize and return Qdrant client.
        
        Returns:
            QdrantClient: Connected Qdrant client instance.
            
        Raises:
            QdrantError: If connection fails.
        """
        if self._client is None:
            try:
                self._client = QdrantClient(
                    url=self._settings.QDRANT_URL,
                    api_key=self._settings.QDRANT_API_KEY,
                    timeout=30,
                )
                self.logger.info(
                    "Qdrant client initialized",
                    url=self._settings.QDRANT_URL,
                )
            except Exception as e:
                self.logger.error(
                    "Failed to initialize Qdrant client",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise QdrantError(
                    message="Failed to connect to Qdrant",
                    details={"error": str(e)},
                ) from e
        return self._client

    def _collection_exists(self) -> bool:
        """
        Check if the collection exists.
        
        Returns:
            bool: True if collection exists.
        """
        try:
            collections = self.client.get_collections()
            return any(
                col.name == self.COLLECTION_NAME
                for col in collections.collections
            )
        except Exception as e:
            self.logger.error(
                "Failed to check collection existence",
                collection=self.COLLECTION_NAME,
                error=str(e),
            )
            raise QdrantError(
                message="Failed to check collection existence",
                details={"collection": self.COLLECTION_NAME, "error": str(e)},
            ) from e

    def _create_payload_indexes(self) -> None:
        """
        Create payload indexes for efficient filtering.
        
        Indexes are created for chunk_type, section_number, and act_name
        to enable fast filtered searches.
        """
        for field_name, field_type in self.INDEXED_FIELDS:
            try:
                self.client.create_payload_index(
                    collection_name=self.COLLECTION_NAME,
                    field_name=field_name,
                    field_schema=field_type,
                )
                self.logger.info(
                    "Created payload index",
                    collection=self.COLLECTION_NAME,
                    field=field_name,
                    type=str(field_type),
                )
            except Exception as e:
                # Index might already exist, log and continue
                self.logger.warning(
                    "Failed to create payload index (may already exist)",
                    field=field_name,
                    error=str(e),
                )

    def ensure_collection(self) -> None:
        """
        Ensure the collection exists with proper configuration.
        
        Creates the collection if it doesn't exist and sets up
        payload indexes for efficient filtering.
        
        Raises:
            QdrantError: If collection creation fails.
        """
        try:
            if self._collection_exists():
                self.logger.info(
                    "Collection already exists, ensuring indexes",
                    collection=self.COLLECTION_NAME,
                )
                # Always ensure indexes exist (for existing collections)
                self._create_payload_indexes()
                return

            # Create collection with vector configuration
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.VECTOR_SIZE,
                    distance=self.DISTANCE_METRIC,
                ),
            )

            self.logger.info(
                "Collection created",
                collection=self.COLLECTION_NAME,
                vector_size=self.VECTOR_SIZE,
                distance=str(self.DISTANCE_METRIC),
            )

            # Create payload indexes for efficient filtering
            self._create_payload_indexes()

            self.logger.info(
                "Collection setup completed",
                collection=self.COLLECTION_NAME,
            )

        except QdrantError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to ensure collection",
                collection=self.COLLECTION_NAME,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise QdrantError(
                message="Failed to create collection",
                details={"collection": self.COLLECTION_NAME, "error": str(e)},
            ) from e

    def _chunk_to_point(self, chunk: LegalChunk) -> PointStruct:
        """
        Convert a LegalChunk to a Qdrant PointStruct.
        
        Args:
            chunk: LegalChunk object with embedding.
            
        Returns:
            PointStruct: Qdrant point ready for upsertion.
            
        Raises:
            ValueError: If chunk has no embedding.
        """
        if chunk.embedding is None:
            raise ValueError(
                f"Chunk {chunk.chunk_id} has no embedding. "
                "Embeddings must be generated before upsertion."
            )

        # Build payload with text and all metadata fields
        payload: dict[str, Any] = {
            "text": chunk.text,
            "chunk_id": chunk.chunk_id,
            "metadata": {
                "source_document": chunk.metadata.source_document,
                "act_name": chunk.metadata.act_name,
                "section_number": chunk.metadata.section_number,
                "chapter_title": chunk.metadata.chapter_title,
                "page_number": chunk.metadata.page_number,
                "chunk_type": chunk.metadata.chunk_type,
            },
        }

        # Include parent_id if present (for child chunks)
        parent_id = chunk.__dict__.get("parent_id")
        if parent_id:
            payload["parent_id"] = parent_id

        return PointStruct(
            id=chunk.chunk_id,
            vector=chunk.embedding,
            payload=payload,
        )

    def upsert_chunks(
        self,
        chunks: list[LegalChunk],
        batch_size: int = 100,
    ) -> int:
        """
        Upsert a list of LegalChunk objects to Qdrant.
        
        Args:
            chunks: List of LegalChunk objects with embeddings.
            batch_size: Number of points to upsert per batch.
            
        Returns:
            int: Number of chunks successfully upserted.
            
        Raises:
            QdrantError: If upsertion fails.
            ValueError: If any chunk has no embedding.
        """
        if not chunks:
            self.logger.warning("No chunks provided for upsertion")
            return 0

        # Validate all chunks have embeddings
        chunks_without_embeddings = [
            c.chunk_id for c in chunks if c.embedding is None
        ]
        if chunks_without_embeddings:
            raise ValueError(
                f"The following chunks have no embeddings: "
                f"{chunks_without_embeddings[:5]}... "
                f"({len(chunks_without_embeddings)} total)"
            )

        try:
            # Ensure collection exists
            self.ensure_collection()

            # Convert chunks to points
            points = [self._chunk_to_point(chunk) for chunk in chunks]

            # Upsert in batches
            total_upserted = 0
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                
                self.client.upsert(
                    collection_name=self.COLLECTION_NAME,
                    points=batch,
                    wait=True,
                )
                
                total_upserted += len(batch)
                self.logger.info(
                    "Upserted batch",
                    collection=self.COLLECTION_NAME,
                    batch_size=len(batch),
                    total_upserted=total_upserted,
                    total_chunks=len(chunks),
                )

            self.logger.info(
                "Upsert completed",
                collection=self.COLLECTION_NAME,
                total_chunks=total_upserted,
            )

            return total_upserted

        except ValueError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to upsert chunks",
                collection=self.COLLECTION_NAME,
                chunk_count=len(chunks),
                error=str(e),
                error_type=type(e).__name__,
            )
            raise QdrantError(
                message="Failed to upsert chunks to Qdrant",
                details={
                    "collection": self.COLLECTION_NAME,
                    "chunk_count": len(chunks),
                    "error": str(e),
                },
            ) from e

    def get_collection_info(self) -> dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            dict: Collection statistics and configuration.
        """
        try:
            info = self.client.get_collection(self.COLLECTION_NAME)
            return {
                "name": self.COLLECTION_NAME,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": str(info.status),
                "vector_size": self.VECTOR_SIZE,
                "distance": str(self.DISTANCE_METRIC),
            }
        except Exception as e:
            self.logger.error(
                "Failed to get collection info",
                collection=self.COLLECTION_NAME,
                error=str(e),
            )
            raise QdrantError(
                message="Failed to get collection info",
                details={"collection": self.COLLECTION_NAME, "error": str(e)},
            ) from e

    def delete_collection(self) -> bool:
        """
        Delete the collection (use with caution).
        
        Returns:
            bool: True if deletion was successful.
        """
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
            self.logger.info(
                "Collection deleted",
                collection=self.COLLECTION_NAME,
            )
            return True
        except Exception as e:
            self.logger.error(
                "Failed to delete collection",
                collection=self.COLLECTION_NAME,
                error=str(e),
            )
            raise QdrantError(
                message="Failed to delete collection",
                details={"collection": self.COLLECTION_NAME, "error": str(e)},
            ) from e

    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        offset: int = 0,
        score_threshold: float = 0.35,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[LegalChunk]:
        """
        Search for similar vectors in the collection.
        
        Args:
            query_vector: Query embedding vector.
            limit: Maximum number of results to return (default: 10).
            offset: Number of results to skip (for pagination).
            score_threshold: Minimum similarity score threshold (default: 0.35).
            filters: Optional dictionary of metadata filters.
                     e.g., {'act_name': 'bns.pdf', 'chunk_type': 'parent'}
            
        Returns:
            list[LegalChunk]: Search results as LegalChunk objects.
            
        Raises:
            SearchError: If search operation fails.
        """
        try:
            # Build filter conditions
            query_filter = self._build_filter(filters)
            
            self.logger.info(
                "Executing vector search",
                collection=self.COLLECTION_NAME,
                limit=limit,
                offset=offset,
                score_threshold=score_threshold,
                has_filters=query_filter is not None,
            )

            # Execute search
            results = self.client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                offset=offset,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False,
            )

            self.logger.info(
                "Search completed",
                results_count=len(results),
            )

            # Convert results to LegalChunk objects
            chunks = self._convert_results_to_chunks(results)
            
            return chunks

        except Exception as e:
            self.logger.error(
                "Search failed",
                collection=self.COLLECTION_NAME,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise SearchError(
                message="Vector search failed",
                details={"collection": self.COLLECTION_NAME, "error": str(e)},
            ) from e

    def search_by_metadata(
        self,
        filters: dict[str, Any],
        limit: int = 5,
    ) -> list[LegalChunk]:
        """
        Search for chunks by metadata only (no vector similarity).
        
        Used for GraphRAG citation traversal - fetching specific sections
        by their section_number without needing a query vector.
        
        Args:
            filters: Dictionary of metadata filters (required).
                     e.g., {'section_number': '305', 'source_document': 'bns.pdf'}
            limit: Maximum number of results to return (default: 5).
            
        Returns:
            list[LegalChunk]: Matching chunks.
            
        Raises:
            SearchError: If search operation fails.
        """
        try:
            # Build filter conditions
            query_filter = self._build_metadata_filter(filters)
            
            if query_filter is None:
                self.logger.warning("No valid filters for metadata search")
                return []
            
            self.logger.debug(
                "Executing metadata search",
                collection=self.COLLECTION_NAME,
                filters=filters,
                limit=limit,
            )

            # Use scroll to fetch by filter without vector
            points, _ = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            # Convert to LegalChunk objects
            chunks: list[LegalChunk] = []
            
            for point in points:
                try:
                    payload = point.payload or {}
                    metadata_dict = payload.get("metadata", {})
                    
                    metadata = LegalMetadata(
                        source_document=metadata_dict.get("source_document", "unknown"),
                        act_name=metadata_dict.get("act_name", "unknown"),
                        section_number=metadata_dict.get("section_number"),
                        chapter_title=metadata_dict.get("chapter_title"),
                        page_number=metadata_dict.get("page_number", 1),
                        chunk_type=metadata_dict.get("chunk_type", "parent"),
                        doc_category=metadata_dict.get("doc_category", "act"),
                        case_year=metadata_dict.get("case_year"),
                    )
                    
                    chunk = LegalChunk(
                        chunk_id=payload.get("chunk_id", str(point.id)),
                        text=payload.get("text", ""),
                        metadata=metadata,
                        embedding=None,
                    )
                    chunks.append(chunk)
                    
                except Exception as e:
                    self.logger.warning(
                        "Failed to convert point to LegalChunk",
                        point_id=str(point.id),
                        error=str(e),
                    )
                    continue

            self.logger.debug(
                "Metadata search completed",
                results_count=len(chunks),
            )
            
            return chunks

        except Exception as e:
            self.logger.error(
                "Metadata search failed",
                filters=filters,
                error=str(e),
            )
            raise SearchError(
                message="Metadata search failed",
                details={"filters": filters, "error": str(e)},
            ) from e

    def _build_metadata_filter(
        self,
        filters: dict[str, Any],
    ) -> Optional[Filter]:
        """
        Build Qdrant filter from dictionary for metadata-only search.
        
        Unlike _build_filter, this does NOT add default parent chunk preference.
        
        Args:
            filters: Dictionary of field-value pairs to filter on.
            
        Returns:
            Optional[Filter]: Qdrant filter object or None.
        """
        conditions: list[FieldCondition] = []
        
        if not filters:
            return None
        
        # Map filter keys to Qdrant payload paths
        field_mapping = {
            "act_name": "metadata.act_name",
            "chunk_type": "metadata.chunk_type",
            "section_number": "metadata.section_number",
            "source_document": "metadata.source_document",
            "doc_category": "metadata.doc_category",
            "chapter_title": "metadata.chapter_title",
        }
        
        # Build conditions from filters dict
        for key, value in filters.items():
            if key in field_mapping and value is not None:
                conditions.append(
                    FieldCondition(
                        key=field_mapping[key],
                        match=MatchValue(value=value),
                    )
                )
        
        # Return None if no conditions
        if not conditions:
            return None
        
        # Return filter with all conditions (AND logic)
        return Filter(must=conditions)

    def _build_filter(
        self,
        filters: Optional[dict[str, Any]] = None,
    ) -> Optional[Filter]:
        """
        Build Qdrant filter from dictionary of conditions.
        
        Args:
            filters: Dictionary of field-value pairs to filter on.
                     Supported fields: act_name, chunk_type, section_number,
                     source_document, doc_category
            
        Returns:
            Optional[Filter]: Qdrant filter object or None.
        """
        conditions: list[FieldCondition] = []
        
        # If no filters provided, default to preferring parent chunks
        if filters is None:
            filters = {}
        
        # Add default parent chunk preference if chunk_type not specified
        if "chunk_type" not in filters:
            conditions.append(
                FieldCondition(
                    key="metadata.chunk_type",
                    match=MatchValue(value="parent"),
                )
            )
        
        # Map filter keys to Qdrant payload paths
        field_mapping = {
            "act_name": "metadata.act_name",
            "chunk_type": "metadata.chunk_type",
            "section_number": "metadata.section_number",
            "source_document": "metadata.source_document",
            "doc_category": "metadata.doc_category",
            "chapter_title": "metadata.chapter_title",
        }
        
        # Build conditions from filters dict
        for key, value in filters.items():
            if key in field_mapping and value is not None:
                conditions.append(
                    FieldCondition(
                        key=field_mapping[key],
                        match=MatchValue(value=value),
                    )
                )
        
        # Return None if no conditions
        if not conditions:
            return None
        
        # Return filter with all conditions (AND logic)
        return Filter(must=conditions)

    def _convert_results_to_chunks(
        self,
        results: list,
    ) -> list[LegalChunk]:
        """
        Convert Qdrant ScoredPoint results to LegalChunk objects.
        
        Args:
            results: List of Qdrant ScoredPoint objects.
            
        Returns:
            list[LegalChunk]: Converted chunk objects.
        """
        chunks: list[LegalChunk] = []
        
        for point in results:
            try:
                payload = point.payload
                
                # Extract metadata from payload
                metadata_dict = payload.get("metadata", {})
                
                metadata = LegalMetadata(
                    source_document=metadata_dict.get("source_document", "unknown"),
                    act_name=metadata_dict.get("act_name", "unknown"),
                    section_number=metadata_dict.get("section_number"),
                    chapter_title=metadata_dict.get("chapter_title"),
                    page_number=metadata_dict.get("page_number", 1),
                    chunk_type=metadata_dict.get("chunk_type", "parent"),
                    doc_category=metadata_dict.get("doc_category", "act"),
                    case_year=metadata_dict.get("case_year"),
                )
                
                chunk = LegalChunk(
                    chunk_id=payload.get("chunk_id", str(point.id)),
                    text=payload.get("text", ""),
                    metadata=metadata,
                    embedding=None,  # Don't return embeddings in search results
                )
                
                # Store score as extra attribute for ranking
                chunk.__dict__["score"] = point.score
                
                # Store parent_id if present
                if "parent_id" in payload:
                    chunk.__dict__["parent_id"] = payload["parent_id"]
                
                chunks.append(chunk)
                
            except Exception as e:
                self.logger.warning(
                    "Failed to convert search result to LegalChunk",
                    point_id=str(point.id),
                    error=str(e),
                )
                continue
        
        return chunks

    def search_with_children(
        self,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.35,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Search for parent chunks and retrieve their children for full context.
        
        Args:
            query_vector: Query embedding vector.
            limit: Maximum number of parent results.
            score_threshold: Minimum similarity score.
            filters: Optional metadata filters.
            
        Returns:
            list[dict]: Results with parent and child chunks grouped.
        """
        # Ensure we search for parents
        parent_filters = filters.copy() if filters else {}
        parent_filters["chunk_type"] = "parent"
        
        # Search for parent chunks
        parent_chunks = self.search(
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            filters=parent_filters,
        )
        
        results = []
        for parent in parent_chunks:
            result = {
                "parent": parent,
                "score": parent.__dict__.get("score", 0.0),
                "children": [],
            }
            
            # Fetch children for this parent
            try:
                children = self._get_children_for_parent(parent.chunk_id)
                result["children"] = children
            except Exception as e:
                self.logger.warning(
                    "Failed to fetch children for parent",
                    parent_id=parent.chunk_id,
                    error=str(e),
                )
            
            results.append(result)
        
        return results

    def _get_children_for_parent(self, parent_id: str) -> list[LegalChunk]:
        """
        Retrieve all child chunks for a given parent ID.
        
        Args:
            parent_id: The parent chunk's UUID.
            
        Returns:
            list[LegalChunk]: List of child chunks.
        """
        try:
            # Search by parent_id in payload
            results = self.client.scroll(
                collection_name=self.COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="parent_id",
                            match=MatchValue(value=parent_id),
                        )
                    ]
                ),
                limit=100,
                with_payload=True,
                with_vectors=False,
            )
            
            points = results[0]  # scroll returns (points, next_page_offset)
            return self._convert_results_to_chunks(points)
            
        except Exception as e:
            self.logger.warning(
                "Failed to get children for parent",
                parent_id=parent_id,
                error=str(e),
            )
            return []
