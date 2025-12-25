"""Test script to verify the full RAG pipeline via terminal."""

import asyncio
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.generator import LegalGenerator
from core.logger import setup_logging
from core.retriever import HybridRetriever

# Initialize logging
setup_logging()


async def main() -> None:
    """
    Test the full RAG pipeline.
    
    Steps:
        1. Initialize retriever and generator
        2. Run hybrid search (vector + rerank)
        3. Stream LLM response
    """
    print("=" * 60)
    print("🧪 LEGAL RAG PIPELINE TEST")
    print("=" * 60)

    # Initialize components
    print("\n📦 Initializing components...")
    retriever = HybridRetriever()
    generator = LegalGenerator()
    print("✓ Components initialized")

    # Test query
    query = "What is the punishment for mob lynching under BNS?"
    print(f"\n❓ Query: {query}")
    print("-" * 60)

    # ============================================
    # Step 1: Retrieval
    # ============================================
    print("\n🔍 Step 1: Hybrid Search (Vector + Rerank)...")
    
    results = await retriever.search(query, k=5)
    
    print(f"✓ Found {len(results)} documents")
    print("\n📄 Top 3 Results:")
    print("-" * 40)
    
    for i, chunk in enumerate(results[:3], 1):
        section = chunk.metadata.section_number or "N/A"
        source = chunk.metadata.source_document
        score = chunk.__dict__.get("rerank_score", chunk.__dict__.get("score", 0))
        chapter = chunk.metadata.chapter_title or "N/A"
        
        print(f"  {i}. Section: {section}")
        print(f"     Source: {source}")
        print(f"     Chapter: {chapter}")
        print(f"     Score: {score:.4f}" if isinstance(score, float) else f"     Score: {score}")
        print(f"     Preview: {chunk.text[:100]}...")
        print()

    # ============================================
    # Step 2: Generation
    # ============================================
    print("\n🤖 Step 2: Generating Answer (Streaming)...")
    print("-" * 60)
    print()

    try:
        async for token in generator.get_answer_stream(query, results):
            print(token, end="", flush=True)
        
        print("\n")
        print("-" * 60)
        print("✅ RAG Pipeline Test Complete!")
        print("=" * 60)

    except Exception as e:
        print(f"\n\n❌ Generation Error: {e}")
        raise


async def test_multiple_queries() -> None:
    """Test multiple queries to verify different scopes."""
    
    queries = [
        "What is the punishment for murder under BNS?",
        "Explain Section 302 of BNS",
        "What are fundamental rights under Indian Constitution?",
        "What is the procedure for arrest under BNSS?",
    ]

    retriever = HybridRetriever()
    generator = LegalGenerator()

    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print("=" * 60)
        
        results = await retriever.search(query, k=3)
        print(f"Found {len(results)} results")
        
        if results:
            print("\nAnswer:")
            print("-" * 40)
            async for token in generator.get_answer_stream(query, results):
                print(token, end="", flush=True)
            print("\n")


if __name__ == "__main__":
    # Run the main test
    asyncio.run(main())
    
    # Uncomment to run multiple query tests
    # asyncio.run(test_multiple_queries())
