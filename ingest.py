"""Main ingestion script for the Legal Agent RAG system.

This script orchestrates the full pipeline for a Data Lake of legal documents:
1. Scan data/ directory for PDF files
2. Classify documents (Act vs Judgment)
3. Parse PDF to Markdown (LlamaParse)
4. Split into Parent-Child chunks (with category-specific chunk sizes)
5. Generate embeddings (BGE-Small)
6. Upsert to Qdrant vector database
"""

import asyncio
import re
import sys
from pathlib import Path
from typing import Literal

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.config import get_settings
from core.embedding import EmbeddingService
from core.logger import get_logger, setup_logging
from database.vector_store import VectorDB
from ingestion.parser import LegalParser
from ingestion.splitter import LegalSplitter
from models.schema import IngestionStatus

# Initialize logging
setup_logging()
logger = get_logger(__name__)

# Document classification patterns
ACT_PATTERNS = ["bns", "bnss", "bsa", "const", "ipc", "crpc", "cpc"]
JUDGMENT_PREFIX = "judgment_"


def classify_document(filename: str) -> Literal["act", "judgment"]:
    """
    Classify a document based on its filename.
    
    Args:
        filename: Name of the PDF file.
        
    Returns:
        Literal["act", "judgment"]: Document category.
    """
    filename_lower = filename.lower()
    
    # Check for judgment prefix
    if filename_lower.startswith(JUDGMENT_PREFIX):
        return "judgment"
    
    # Check for act patterns
    for pattern in ACT_PATTERNS:
        if pattern in filename_lower:
            return "act"
    
    # Default to act if no pattern matches
    logger.warning(
        "Could not classify document, defaulting to 'act'",
        filename=filename,
    )
    return "act"


def get_pdf_files(data_dir: Path) -> list[Path]:
    """
    Get all PDF files from the data directory.
    
    Args:
        data_dir: Path to the data directory.
        
    Returns:
        list[Path]: List of PDF file paths.
    """
    if not data_dir.exists():
        logger.warning(
            "Data directory does not exist, creating it",
            data_dir=str(data_dir),
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        return []
    
    pdf_files = list(data_dir.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(
            "No PDF files found in data directory",
            data_dir=str(data_dir),
        )
    
    return sorted(pdf_files)


async def process_document(
    pdf_path: Path,
    doc_category: Literal["act", "judgment"],
    parser: LegalParser,
    embedding_service: EmbeddingService,
    db: VectorDB,
) -> dict:
    """
    Process a single document through the full pipeline.
    
    Args:
        pdf_path: Path to the PDF file.
        doc_category: Document category ('act' or 'judgment').
        parser: LegalParser instance.
        embedding_service: EmbeddingService instance.
        db: VectorDB instance.
        
    Returns:
        dict: Processing statistics.
    """
    logger.info(
        f"📄 Processing document",
        file=pdf_path.name,
        category=doc_category,
    )
    
    # Step 1: Parse PDF to Markdown
    logger.info(
        "  → Parsing PDF to Markdown...",
        file=pdf_path.name,
    )
    markdown_text = await parser.parse(str(pdf_path))
    
    # Step 2: Split with category-specific splitter
    logger.info(
        "  → Splitting document...",
        file=pdf_path.name,
        category=doc_category,
    )
    splitter = LegalSplitter(doc_category=doc_category)
    chunks = splitter.split_document(
        markdown_text=markdown_text,
        source_file=pdf_path.name,
    )
    
    parent_count = sum(1 for c in chunks if c.metadata.chunk_type == "parent")
    child_count = sum(1 for c in chunks if c.metadata.chunk_type == "child")
    
    # Step 3: Generate embeddings
    logger.info(
        "  → Generating embeddings...",
        file=pdf_path.name,
        chunk_count=len(chunks),
    )
    chunks_with_embeddings = embedding_service.embed_batch(chunks)
    
    # Step 4: Upsert to Qdrant
    logger.info(
        "  → Upserting to Qdrant...",
        file=pdf_path.name,
    )
    upserted_count = db.upsert_chunks(chunks_with_embeddings)
    
    logger.info(
        f"  ✓ Completed: {pdf_path.name}",
        total_chunks=len(chunks),
        parent_chunks=parent_count,
        child_chunks=child_count,
    )
    
    return {
        "filename": pdf_path.name,
        "category": doc_category,
        "total_chunks": len(chunks),
        "parent_chunks": parent_count,
        "child_chunks": child_count,
        "upserted": upserted_count,
    }


async def main() -> None:
    """
    Run the full ingestion pipeline for all documents in data/ directory.
    
    Steps:
        1. Scan data/ for PDF files
        2. Classify each document (Act vs Judgment)
        3. Process each document through the pipeline
        4. Report summary statistics
    """
    settings = get_settings()
    
    # Define data directory
    data_dir = Path(__file__).parent / "data"
    
    logger.info(
        "🚀 Starting Legal Data Lake Ingestion Pipeline",
        data_dir=str(data_dir),
        environment=settings.ENVIRONMENT,
    )

    try:
        # ============================================
        # Initialize Services
        # ============================================
        logger.info("Initializing services...")
        
        parser = LegalParser()
        embedding_service = EmbeddingService()
        db = VectorDB()

        # Ensure Qdrant collection exists
        logger.info("Ensuring Qdrant collection exists...")
        db.ensure_collection()

        # ============================================
        # Scan Data Directory
        # ============================================
        pdf_files = get_pdf_files(data_dir)
        
        if not pdf_files:
            logger.error(
                "❌ No PDF files found in data directory",
                data_dir=str(data_dir),
            )
            print(f"\n❌ No PDF files found in {data_dir}")
            print("   Please add PDF files to the data/ directory.\n")
            sys.exit(1)

        logger.info(
            f"📁 Found {len(pdf_files)} PDF files",
            files=[f.name for f in pdf_files],
        )

        # ============================================
        # Classify and Process Documents
        # ============================================
        results: list[dict] = []
        acts_count = 0
        judgments_count = 0

        for pdf_path in pdf_files:
            # Classify document
            doc_category = classify_document(pdf_path.name)
            
            # Log classification
            category_emoji = "📜" if doc_category == "act" else "⚖️"
            category_label = "Act" if doc_category == "act" else "Judgment"
            
            logger.info(
                f"{category_emoji} Detected [{category_label}]: {pdf_path.name}",
                filename=pdf_path.name,
                category=doc_category,
            )
            print(f"{category_emoji} Detected [{category_label}]: {pdf_path.name}")
            
            if doc_category == "act":
                acts_count += 1
            else:
                judgments_count += 1
            
            # Process document
            result = await process_document(
                pdf_path=pdf_path,
                doc_category=doc_category,
                parser=parser,
                embedding_service=embedding_service,
                db=db,
            )
            results.append(result)

        # ============================================
        # Summary Statistics
        # ============================================
        total_chunks = sum(r["total_chunks"] for r in results)
        total_parents = sum(r["parent_chunks"] for r in results)
        total_children = sum(r["child_chunks"] for r in results)

        # Get collection info
        collection_info = db.get_collection_info()

        logger.info(
            "✅ Ingestion Complete!",
            documents_processed=len(results),
            acts=acts_count,
            judgments=judgments_count,
            total_chunks=total_chunks,
            vectors_in_collection=collection_info.get("vectors_count", 0),
        )

        print("\n" + "=" * 60)
        print("✅ DATA LAKE INGESTION COMPLETE!")
        print("=" * 60)
        print(f"\n📁 Documents Processed: {len(results)}")
        print(f"   - 📜 Acts: {acts_count}")
        print(f"   - ⚖️  Judgments: {judgments_count}")
        print(f"\n📊 Total Chunks: {total_chunks}")
        print(f"   - Parent Chunks: {total_parents}")
        print(f"   - Child Chunks: {total_children}")
        print(f"\n☁️  Collection: {db.COLLECTION_NAME}")
        print(f"🔢 Vectors in DB: {collection_info.get('vectors_count', 0)}")
        print("\n📄 Document Details:")
        print("-" * 60)
        for r in results:
            cat_emoji = "📜" if r["category"] == "act" else "⚖️"
            print(f"  {cat_emoji} {r['filename']}: {r['total_chunks']} chunks")
        print("=" * 60 + "\n")

    except FileNotFoundError as e:
        logger.error(
            "❌ Ingestion failed - File not found",
            error=str(e),
        )
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)

    except Exception as e:
        logger.exception(
            "❌ Ingestion failed",
            error=str(e),
            error_type=type(e).__name__,
        )
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Ingest legal documents into vector database")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Delete existing collection and re-index all documents",
    )
    args = parser.parse_args()
    
    # Create data directory if it doesn't exist
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # If reindex flag is set, delete the collection first
    if args.reindex:
        print("🗑️  Deleting existing collection for re-indexing...")
        db = VectorDB()
        if db.delete_collection():
            print("✅ Collection deleted successfully")
        else:
            print("⚠️  Collection did not exist or failed to delete")
    
    # Run the async main function
    asyncio.run(main())
