"""PDF parsing module using LlamaParse for legal document conversion."""

import asyncio
from pathlib import Path
from typing import Optional

import nest_asyncio
from llama_parse import LlamaParse

from core.config import get_settings
from core.exceptions import PDFParsingError
from core.logger import get_logger, LoggerMixin

# Apply nest_asyncio to prevent event loop conflicts
nest_asyncio.apply()

logger = get_logger(__name__)


class LegalParser(LoggerMixin):
    """
    Robust PDF to Markdown parser for legal documents.
    
    Uses LlamaParse API with local caching to minimize API calls
    and improve performance on repeated runs.
    """

    def __init__(self) -> None:
        """Initialize the LegalParser with settings."""
        self._settings = get_settings()

    def _create_parser(self) -> LlamaParse:
        """
        Create a fresh LlamaParse client instance.
        
        Creates a new instance each time to prevent event loop conflicts.
        
        Returns:
            LlamaParse: Configured parser instance.
        """
        return LlamaParse(
            api_key=self._settings.LLAMA_CLOUD_API_KEY,
            result_type="markdown",
            verbose=self._settings.DEBUG,
        )

    def _get_cache_path(self, file_path: str) -> Path:
        """
        Get the cache file path for a given PDF.
        
        Args:
            file_path: Path to the source PDF file.
            
        Returns:
            Path: Path to the cached markdown file.
        """
        return Path(f"{file_path}.md")

    async def _read_cached_markdown(self, cache_path: Path) -> Optional[str]:
        """
        Read cached markdown file if it exists.
        
        Args:
            cache_path: Path to the cached markdown file.
            
        Returns:
            Optional[str]: Cached markdown content or None if not found.
        """
        if not cache_path.exists():
            return None

        self.logger.info(
            "Cache hit - reading from local file",
            cache_path=str(cache_path),
        )

        try:
            # Use asyncio to read file without blocking
            loop = asyncio.get_event_loop()
            content = await loop.run_in_executor(
                None,
                lambda: cache_path.read_text(encoding="utf-8"),
            )
            return content
        except Exception as e:
            self.logger.warning(
                "Failed to read cached file, will re-parse",
                cache_path=str(cache_path),
                error=str(e),
            )
            return None

    async def _save_to_cache(self, cache_path: Path, content: str) -> None:
        """
        Save parsed markdown to cache file.
        
        Args:
            cache_path: Path to save the cached file.
            content: Markdown content to cache.
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: cache_path.write_text(content, encoding="utf-8"),
            )
            self.logger.info(
                "Saved parsed content to cache",
                cache_path=str(cache_path),
            )
        except Exception as e:
            self.logger.warning(
                "Failed to save to cache",
                cache_path=str(cache_path),
                error=str(e),
            )

    async def parse(self, file_path: str) -> str:
        """
        Parse a PDF file to Markdown with caching support.
        
        This method first checks for a cached .md file to save API credits.
        If no cache exists, it calls LlamaParse API and saves the result.
        
        Args:
            file_path: Path to the PDF file to parse.
            
        Returns:
            str: The full markdown text content.
            
        Raises:
            PDFParsingError: If parsing fails.
            FileNotFoundError: If the PDF file doesn't exist.
        """
        pdf_path = Path(file_path)
        cache_path = self._get_cache_path(file_path)

        # Validate input file exists
        if not pdf_path.exists():
            self.logger.error(
                "PDF file not found",
                file_path=file_path,
            )
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        # Check cache first (saves API credits and time)
        cached_content = await self._read_cached_markdown(cache_path)
        if cached_content is not None:
            self.logger.info(
                "Returning cached markdown",
                file_path=file_path,
                content_length=len(cached_content),
            )
            return cached_content

        # Cache miss - call LlamaParse API
        self.logger.info(
            "Cache miss - calling LlamaParse API",
            file_path=file_path,
        )

        try:
            # Create fresh parser instance to avoid event loop issues
            parser = self._create_parser()
            
            # LlamaParse aload_data returns list of Document objects
            loop = asyncio.get_event_loop()
            documents = await loop.run_in_executor(
                None,
                lambda: parser.load_data(str(pdf_path)),
            )

            # Combine all document pages into single markdown string
            markdown_content = "\n\n".join(
                doc.text for doc in documents if doc.text
            )

            if not markdown_content:
                raise PDFParsingError(
                    message="LlamaParse returned empty content",
                    details={"file_path": file_path},
                )

            self.logger.info(
                "Successfully parsed PDF",
                file_path=file_path,
                pages_parsed=len(documents),
                content_length=len(markdown_content),
            )

            # Save to cache for future runs
            await self._save_to_cache(cache_path, markdown_content)

            return markdown_content

        except PDFParsingError:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to parse PDF with LlamaParse",
                file_path=file_path,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise PDFParsingError(
                message=f"Failed to parse PDF: {file_path}",
                details={
                    "file_path": file_path,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            ) from e

    async def parse_multiple(self, file_paths: list[str]) -> dict[str, str]:
        """
        Parse multiple PDF files concurrently.
        
        Args:
            file_paths: List of PDF file paths to parse.
            
        Returns:
            dict[str, str]: Mapping of file paths to markdown content.
        """
        self.logger.info(
            "Starting batch PDF parsing",
            file_count=len(file_paths),
        )

        results: dict[str, str] = {}
        
        # Process files concurrently with controlled concurrency
        tasks = [self.parse(fp) for fp in file_paths]
        parsed_contents = await asyncio.gather(*tasks, return_exceptions=True)

        for file_path, content in zip(file_paths, parsed_contents):
            if isinstance(content, Exception):
                self.logger.error(
                    "Failed to parse file in batch",
                    file_path=file_path,
                    error=str(content),
                )
                continue
            results[file_path] = content

        self.logger.info(
            "Batch parsing completed",
            total_files=len(file_paths),
            successful=len(results),
            failed=len(file_paths) - len(results),
        )

        return results
