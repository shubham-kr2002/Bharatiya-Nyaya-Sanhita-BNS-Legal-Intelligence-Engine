"""Document splitting module implementing Parent-Child Indexing strategy."""

import re
import uuid
from typing import Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter

from core.config import get_settings
from core.logger import get_logger, LoggerMixin
from models.schema import LegalChunk, LegalMetadata

logger = get_logger(__name__)


class LegalSplitter(LoggerMixin):
    """
    Implements Parent-Child Indexing strategy for legal documents.
    
    Parent chunks provide broader context while child chunks enable
    precise retrieval. Each child maintains a reference to its parent
    for context expansion during retrieval.
    
    Supports different chunk sizes based on document category:
    - Acts: Strict 2000/400 parent-child split
    - Judgments: Larger 3000/600 split (verbose case documents)
    """

    # Regex patterns for legal document structure extraction
    # Pattern 1: "Section 302" or "section 420A"
    SECTION_PATTERN = re.compile(
        r"[Ss]ection\s+(\d+[A-Za-z]?)",
        re.IGNORECASE,
    )
    # Pattern 2: Markdown heading like "# 102" or "# Murder"
    HEADING_NUMBER_PATTERN = re.compile(
        r"^#\s*(\d+)\s*$",
        re.MULTILINE,
    )
    # Pattern 3: Section start like "102. Whoever does..." or "1. (1) This Act..."
    NUMBERED_SECTION_PATTERN = re.compile(
        r"^\s*(\d+)\.\s+(?:\([0-9]+\)\s+)?[A-Z]",
        re.MULTILINE,
    )
    CHAPTER_PATTERN = re.compile(
        r"[Cc]hapter\s+([IVXLCDM]+|\d+)\s*[-–:.]?\s*([A-Za-z\s]+)?",
        re.IGNORECASE,
    )
    CASE_YEAR_PATTERN = re.compile(
        r"\b(19|20)\d{2}\b",
        re.IGNORECASE,
    )

    # Chunk size configurations per document category
    CHUNK_CONFIG = {
        "act": {
            "parent_chunk_size": 2000,
            "parent_chunk_overlap": 200,
            "child_chunk_size": 400,
            "child_chunk_overlap": 50,
        },
        "judgment": {
            "parent_chunk_size": 3000,
            "parent_chunk_overlap": 300,
            "child_chunk_size": 600,
            "child_chunk_overlap": 100,
        },
    }

    def __init__(
        self,
        doc_category: str = "act",
    ) -> None:
        """
        Initialize the LegalSplitter with category-specific chunk sizes.
        
        Args:
            doc_category: Document category ('act' or 'judgment').
                         Judgments use larger chunks due to verbose nature.
        """
        self._settings = get_settings()
        self._doc_category = doc_category
        
        # Get chunk configuration based on document category
        config = self.CHUNK_CONFIG.get(doc_category, self.CHUNK_CONFIG["act"])
        
        # Parent splitter - larger chunks for context
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["parent_chunk_size"],
            chunk_overlap=config["parent_chunk_overlap"],
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False,
        )
        
        # Child splitter - smaller chunks for precise retrieval
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["child_chunk_size"],
            chunk_overlap=config["child_chunk_overlap"],
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=False,
        )

        self.logger.info(
            "LegalSplitter initialized",
            doc_category=doc_category,
            parent_chunk_size=config["parent_chunk_size"],
            parent_chunk_overlap=config["parent_chunk_overlap"],
            child_chunk_size=config["child_chunk_size"],
            child_chunk_overlap=config["child_chunk_overlap"],
        )

    def _generate_chunk_id(self) -> str:
        """
        Generate a unique UUID for a chunk.
        
        Returns:
            str: UUID string.
        """
        return str(uuid.uuid4())

    def _extract_section(self, text: str) -> Optional[str]:
        """
        Extract section number from text using multiple regex patterns.
        
        Handles different formats:
        1. "Section 302" or "section 420A" (explicit)
        2. "# 102" (markdown heading)
        3. "102. Whoever does..." (numbered section start)
        
        Args:
            text: Text to search for section patterns.
            
        Returns:
            Optional[str]: Section number if found (e.g., '302', '420A').
        """
        # Try explicit "Section X" pattern first
        match = self.SECTION_PATTERN.search(text)
        if match:
            return match.group(1)
        
        # Try markdown heading pattern "# 102"
        match = self.HEADING_NUMBER_PATTERN.search(text)
        if match:
            return match.group(1)
        
        # Try numbered section start pattern "102. Whoever..."
        match = self.NUMBERED_SECTION_PATTERN.search(text)
        if match:
            return match.group(1)
        
        return None

    def _extract_chapter(self, text: str) -> Optional[str]:
        """
        Extract chapter title from text using regex.
        
        Args:
            text: Text to search for chapter patterns.
            
        Returns:
            Optional[str]: Chapter title if found (e.g., 'Chapter IV - Offenses against Body').
        """
        match = self.CHAPTER_PATTERN.search(text)
        if match:
            chapter_num = match.group(1)
            chapter_title = match.group(2)
            if chapter_title:
                return f"Chapter {chapter_num} - {chapter_title.strip()}"
            return f"Chapter {chapter_num}"
        return None

    def _extract_case_year(self, text: str) -> Optional[int]:
        """
        Extract case year from text (for judgments).
        
        Args:
            text: Text to search for year patterns.
            
        Returns:
            Optional[int]: Year if found (e.g., 2023).
        """
        matches = self.CASE_YEAR_PATTERN.findall(text[:500])  # Check first 500 chars
        if matches:
            # Return the first valid year found
            for match in matches:
                year = int(match + text[text.find(match) + 2:text.find(match) + 4])
            # Try to find a 4-digit year
            full_match = re.search(r"\b(19|20)\d{2}\b", text[:500])
            if full_match:
                return int(full_match.group())
        return None

    def _create_metadata(
        self,
        source_file: str,
        chunk_type: str,
        page_number: int,
        text: str,
        parent_id: Optional[str] = None,
    ) -> LegalMetadata:
        """
        Create metadata for a chunk with extracted section/chapter info.
        
        Args:
            source_file: Source PDF filename.
            chunk_type: Either 'parent' or 'child'.
            page_number: Page number in source document.
            text: Text content for section/chapter extraction.
            parent_id: Parent chunk ID (for child chunks only).
            
        Returns:
            LegalMetadata: Populated metadata object.
        """
        section_number = self._extract_section(text)
        chapter_title = self._extract_chapter(text)
        case_year = self._extract_case_year(text) if self._doc_category == "judgment" else None

        # Create base metadata
        metadata = LegalMetadata(
            source_document=source_file,
            act_name=self._settings.QDRANT_COLLECTION_NAME.replace("_", " ").title()
            if "bns" in self._settings.QDRANT_COLLECTION_NAME.lower()
            else "Bharatiya Nyaya Sanhita",
            section_number=section_number,
            chapter_title=chapter_title,
            page_number=page_number,
            chunk_type=chunk_type,
            doc_category=self._doc_category,
            case_year=case_year,
        )

        return metadata

    def _estimate_page_number(
        self,
        chunk_start_idx: int,
        total_length: int,
        estimated_total_pages: int = 100,
    ) -> int:
        """
        Estimate page number based on chunk position in document.
        
        Args:
            chunk_start_idx: Starting character index of the chunk.
            total_length: Total document length.
            estimated_total_pages: Estimated number of pages in document.
            
        Returns:
            int: Estimated page number (1-indexed).
        """
        if total_length == 0:
            return 1
        page = int((chunk_start_idx / total_length) * estimated_total_pages) + 1
        return max(1, page)

    def split_document(
        self,
        markdown_text: str,
        source_file: str,
        estimated_pages: int = 100,
    ) -> list[LegalChunk]:
        """
        Split a markdown document into parent and child chunks.
        
        Implements Parent-Child Indexing:
        1. Split document into large parent chunks (context)
        2. Split each parent into smaller child chunks (retrieval)
        3. Link children to parents via parent_id in metadata
        
        Args:
            markdown_text: Full markdown text of the legal document.
            source_file: Source PDF filename for metadata.
            estimated_pages: Estimated total pages for page number calculation.
            
        Returns:
            list[LegalChunk]: Flat list containing both parent and child chunks.
        """
        if not markdown_text or not markdown_text.strip():
            self.logger.warning(
                "Empty document provided for splitting",
                source_file=source_file,
            )
            return []

        all_chunks: list[LegalChunk] = []
        total_length = len(markdown_text)

        # Step A: Split into parent chunks
        parent_texts = self._parent_splitter.split_text(markdown_text)
        
        self.logger.info(
            "Created parent chunks",
            source_file=source_file,
            parent_count=len(parent_texts),
        )

        current_position = 0

        # Step B: Process each parent chunk
        for parent_idx, parent_text in enumerate(parent_texts):
            # Generate unique parent ID
            parent_id = self._generate_chunk_id()
            
            # Estimate page number based on position
            page_number = self._estimate_page_number(
                current_position,
                total_length,
                estimated_pages,
            )

            # Create parent chunk
            parent_metadata = self._create_metadata(
                source_file=source_file,
                chunk_type="parent",
                page_number=page_number,
                text=parent_text,
                parent_id=None,
            )

            parent_chunk = LegalChunk(
                chunk_id=parent_id,
                text=parent_text,
                metadata=parent_metadata,
                embedding=None,
            )
            all_chunks.append(parent_chunk)

            # Split parent into child chunks
            child_texts = self._child_splitter.split_text(parent_text)
            
            child_position = current_position
            for child_idx, child_text in enumerate(child_texts):
                child_id = self._generate_chunk_id()
                
                # Estimate child page number
                child_page = self._estimate_page_number(
                    child_position,
                    total_length,
                    estimated_pages,
                )

                # Create child metadata with parent_id reference
                child_metadata = self._create_metadata(
                    source_file=source_file,
                    chunk_type="child",
                    page_number=child_page,
                    text=child_text,
                    parent_id=parent_id,
                )

                # Store parent_id in a way that can be used for retrieval
                # We'll add it as a model field extension via model_extra
                child_chunk = LegalChunk(
                    chunk_id=child_id,
                    text=child_text,
                    metadata=child_metadata,
                    embedding=None,
                )
                
                # Add parent reference (accessible via chunk.model_extra)
                child_chunk.__dict__["parent_id"] = parent_id
                
                all_chunks.append(child_chunk)
                child_position += len(child_text)

            # Update position for next parent
            current_position += len(parent_text)

        # Log summary
        parent_count = sum(1 for c in all_chunks if c.metadata.chunk_type == "parent")
        child_count = sum(1 for c in all_chunks if c.metadata.chunk_type == "child")
        
        self.logger.info(
            "Document splitting completed",
            source_file=source_file,
            total_chunks=len(all_chunks),
            parent_chunks=parent_count,
            child_chunks=child_count,
        )

        return all_chunks

    def get_parent_id(self, chunk: LegalChunk) -> Optional[str]:
        """
        Get the parent ID for a child chunk.
        
        Args:
            chunk: The chunk to get parent ID from.
            
        Returns:
            Optional[str]: Parent chunk ID if this is a child chunk.
        """
        if chunk.metadata.chunk_type == "parent":
            return chunk.chunk_id
        return chunk.__dict__.get("parent_id")
