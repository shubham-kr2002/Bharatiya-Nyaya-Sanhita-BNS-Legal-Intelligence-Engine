"""Pydantic schemas for Legal Agent RAG system data flow."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class LegalMetadata(BaseModel):
    """Metadata associated with a legal document chunk."""

    source_document: str = Field(
        ...,
        description="Source PDF filename (e.g., 'BNS_2023.pdf')",
        examples=["BNS_2023.pdf", "IPC_Amendment_2024.pdf"],
    )
    act_name: str = Field(
        default="Bharatiya Nyaya Sanhita",
        description="Name of the legal act or statute",
        examples=["Bharatiya Nyaya Sanhita", "Indian Penal Code"],
    )
    section_number: Optional[str] = Field(
        default=None,
        description="Section number within the act (e.g., '302', '420')",
        examples=["302", "420", "376"],
    )
    chapter_title: Optional[str] = Field(
        default=None,
        description="Title of the chapter containing this section",
        examples=["Offenses against Body", "Offenses against Property"],
    )
    page_number: int = Field(
        ...,
        description="Page number in the source PDF document",
        ge=1,
        examples=[1, 42, 156],
    )
    chunk_type: Literal["parent", "child"] = Field(
        ...,
        description="Type of chunk - 'parent' for main sections, 'child' for sub-sections",
        examples=["parent", "child"],
    )
    doc_category: Literal["act", "judgment"] = Field(
        default="act",
        description="Category of legal document - 'act' for statutes/legislation, 'judgment' for court decisions",
        examples=["act", "judgment"],
    )
    case_year: Optional[int] = Field(
        default=None,
        description="Year of the judgment (only applicable for doc_category='judgment')",
        ge=1900,
        le=2100,
        examples=[2023, 2024],
    )


class LegalChunk(BaseModel):
    """Represents a chunk of legal text with metadata and optional embedding."""

    chunk_id: str = Field(
        ...,
        description="Unique UUID identifier for the chunk",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    text: str = Field(
        ...,
        description="The actual text content of the legal chunk",
        min_length=1,
        examples=[
            "Section 302. Punishment for murder.—Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine."
        ],
    )
    metadata: LegalMetadata = Field(
        ...,
        description="Metadata associated with this legal chunk",
    )
    embedding: Optional[list[float]] = Field(
        default=None,
        description="Vector embedding representation for semantic search",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
                "text": "Section 302. Punishment for murder.—Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.",
                "metadata": {
                    "source_document": "BNS_2023.pdf",
                    "act_name": "Bharatiya Nyaya Sanhita",
                    "section_number": "302",
                    "chapter_title": "Offenses against Body",
                    "page_number": 45,
                    "chunk_type": "parent",
                    "doc_category": "act",
                    "case_year": None,
                },
                "embedding": None,
            }
        }


class IngestionStatus(BaseModel):
    """Tracks the status of async document ingestion process."""

    total_pages_processed: int = Field(
        default=0,
        description="Total number of PDF pages processed so far",
        ge=0,
        examples=[0, 50, 200],
    )
    chunks_created: int = Field(
        default=0,
        description="Total number of chunks created from processed pages",
        ge=0,
        examples=[0, 150, 600],
    )
    status: Literal["processing", "completed", "failed"] = Field(
        default="processing",
        description="Current status of the ingestion process",
        examples=["processing", "completed", "failed"],
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if status is 'failed'",
    )
    document_name: Optional[str] = Field(
        default=None,
        description="Name of the document being ingested",
        examples=["BNS_2023.pdf"],
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "total_pages_processed": 150,
                "chunks_created": 450,
                "status": "completed",
                "error_message": None,
                "document_name": "BNS_2023.pdf",
            }
        }
