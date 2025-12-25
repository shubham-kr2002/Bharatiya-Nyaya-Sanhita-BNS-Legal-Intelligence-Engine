"""
High-Fidelity Graph Retrieval Engine (GraphRAG) for Indian Law.

This module implements a multi-hop graph traversal system that treats legal
documents as interconnected nodes in a citation network. It bridges the
semantic gap between colloquial user queries and precise statutory language.

Architecture:
    Step 0: Concept Hop   - LLM maps query to specific sections
    Step 1: Vector Hop    - Semantic search for recall
    Step 2: Citation Hop  - Traverse outbound references
    Step 3: Back-Ref Hop  - (Optional) Find sections referencing our results

Author: Legal Agent RAG System
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from flashrank import Ranker, RerankRequest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from core.config import get_settings
from core.embedding import EmbeddingService
from core.exceptions import SearchError
from core.logger import get_logger, LoggerMixin
from database.vector_store import VectorDB
from models.schema import LegalChunk

logger = get_logger(__name__)


# =============================================================================
# Part 1: Schema & Data Structures
# =============================================================================

class LegalNodeType(str, Enum):
    """Types of legal document nodes in the citation graph."""
    BNS_SECTION = "bns_section"
    BNSS_SECTION = "bnss_section"
    BSA_SECTION = "bsa_section"
    CONST_ARTICLE = "const_article"
    CHAPTER = "chapter"
    UNKNOWN = "unknown"


class ChunkSource(str, Enum):
    """Source of how a chunk was retrieved - used for priority ranking."""
    CONCEPT_GRAPH = "concept_graph"      # Direct statutory hit from LLM mapping
    CITATION_GRAPH = "citation_graph"    # Fetched via citation traversal
    VECTOR_SEARCH = "vector_search"      # Standard semantic search
    BACK_REFERENCE = "back_reference"    # Chunks referencing our results


class ReferenceQuery(BaseModel):
    """A structured query to fetch a specific legal section."""
    target_doc: str = Field(..., description="Target document (e.g., 'bns.pdf')")
    target_section: str = Field(..., description="Section number (e.g., '103')")
    target_type: LegalNodeType = Field(default=LegalNodeType.UNKNOWN)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    
    class Config:
        use_enum_values = True


class ConceptBridgeResult(BaseModel):
    """Result from LLM concept bridging."""
    concepts: list[str] = Field(default_factory=list)
    citations: list[ReferenceQuery] = Field(default_factory=list)
    raw_response: str = Field(default="")


@dataclass
class GraphTraceLog:
    """Trace log for debugging graph traversal."""
    concept_mappings: list[str] = field(default_factory=list)
    vector_hits: int = 0
    citation_links: list[str] = field(default_factory=list)
    back_refs: list[str] = field(default_factory=list)
    final_count: int = 0
    concept_count: int = 0
    citation_count: int = 0
    
    def print_trace(self) -> None:
        """Print formatted trace to console - matches requested format."""
        print("\n" + "=" * 60)
        print("📊 RETRIEVAL TRACE")
        print("=" * 60)
        
        # Bridge traces (concept hop results)
        if self.concept_mappings:
            for mapping in self.concept_mappings:
                print(f"[Trace] 🌉 Bridge: {mapping}")
        else:
            print("[Trace] 🌉 Bridge: No direct mappings found")
        
        # Graph traces (citation hop results)
        if self.citation_links:
            for link in self.citation_links[:8]:  # Show first 8
                print(f"[Trace] 🕸️  Graph: Linked to -> {link}")
            if len(self.citation_links) > 8:
                print(f"[Trace] 🕸️  Graph: ... +{len(self.citation_links) - 8} more links")
        
        # Vector traces
        print(f"[Trace] 🔍 Vector: Found {self.vector_hits} semantic matches")
        
        # Back-reference traces
        if self.back_refs:
            for ref in self.back_refs[:3]:
                print(f"[Trace] ↩️  BackRef: {ref}")
        
        # Final summary
        print(f"[Trace] 🚀 Final: Returned {self.final_count} chunks "
              f"({self.concept_count} Bridge, {self.citation_count} Graph, "
              f"{max(0, self.final_count - self.concept_count - self.citation_count)} Vector)")
        print("=" * 60 + "\n")


# =============================================================================
# Part 2: Reference Extraction Patterns
# =============================================================================

class LegalReferenceExtractor:
    """
    Robust regex engine for extracting statutory references from legal text.
    
    Handles normalization (Clause/Section -> unified format) and
    context-aware resolution (inferring target document from context).
    
    Critical Patterns:
        - BNS: Section/Sec./S. + number (e.g., Section 103, Sec. 45)
        - BNSS: Clause/Cl./Section + number (e.g., Clause 479, Cl. 173)
        - Constitution: Article/Art. + number (e.g., Article 21, Art. 23)
    """
    
    # Document-specific patterns - order matters for specificity
    PATTERNS = {
        # BNSS Clause patterns - MUST match before general section
        # Critical: BNSS uses "Clause" terminology (Bill -> Act transition)
        "bnss_clause": re.compile(
            r"(?:Clause|Cl\.)\s*(\d+[A-Z]*)(?:\s*\([^)]*\))?",
            re.IGNORECASE
        ),
        # BNS Section patterns (standard)
        "bns_section": re.compile(
            r"(?:Section|Sec\.|S\.)\s*(\d+[A-Z]*)(?:\s*\([^)]*\))?",
            re.IGNORECASE
        ),
        # Article patterns (Constitution) - includes Art. abbreviation
        "article": re.compile(
            r"(?:Article|Art\.)\s*(\d+[A-Z]*)(?:\s*\([^)]*\))?",
            re.IGNORECASE
        ),
        # Chapter patterns
        "chapter": re.compile(
            r"Chapter\s+([IVXLCDM]+|\d+)",
            re.IGNORECASE
        ),
        # Numeric only (context-dependent)
        "numeric_ref": re.compile(
            r"(?:under|per|in|of)\s+(?:section|sec\.|clause|cl\.)?\s*(\d{2,3})\b",
            re.IGNORECASE
        ),
    }
    
    # Context keywords for document inference
    DOCUMENT_CONTEXT = {
        # BNS indicators (substantive criminal law)
        "bns.pdf": [
            "bharatiya nyaya", "bns", "nyaya sanhita", "penal",
            "offence", "punishment", "crime", "murder", "theft",
            "this sanhita", "of this act", "punishable", "sentenced",
            "death", "imprisonment", "fine", "rigorous"
        ],
        # BNSS indicators (procedural criminal law)
        "bnss.pdf": [
            "bharatiya nagarik", "bnss", "suraksha sanhita",
            "procedure", "bail", "arrest", "trial", "investigation",
            "code of criminal procedure", "crpc", "fir", "cognizable",
            "magistrate", "sessions", "warrant", "summons",
            "detention", "custody", "remand", "anticipatory bail"
        ],
        # BSA indicators (evidence law)
        "bsa.pdf": [
            "bharatiya sakshya", "bsa", "evidence", "witness",
            "testimony", "proof", "admissibility", "relevant",
            "documentary", "oral evidence", "presumption"
        ],
        # Constitution indicators (fundamental law)
        "const.pdf": [
            "constitution", "fundamental rights", "directive",
            "article", "constitutional", "supreme court",
            "high court", "writ", "habeas corpus", "mandamus",
            "right to life", "personal liberty", "equality"
        ],
    }
    
    @classmethod
    def extract_references(
        cls,
        text: str,
        source_document: Optional[str] = None,
    ) -> list[ReferenceQuery]:
        """
        Extract all statutory references from text.
        
        Args:
            text: Text to scan for references.
            source_document: Source document for context-aware resolution.
            
        Returns:
            list[ReferenceQuery]: Structured reference queries.
        """
        references: list[ReferenceQuery] = []
        seen: set[tuple[str, str]] = set()
        text_lower = text.lower()
        
        # CRITICAL: Check for explicit cross-document references first
        is_bnss_context = cls._is_bnss_cross_reference(text_lower)
        is_const_context = cls._is_constitution_cross_reference(text_lower)
        
        # 1. Extract BNSS Clause references (highest priority for procedure)
        for match in cls.PATTERNS["bnss_clause"].finditer(text):
            clause_num = match.group(1)
            key = ("bnss.pdf", clause_num)
            if key not in seen:
                seen.add(key)
                references.append(ReferenceQuery(
                    target_doc="bnss.pdf",
                    target_section=clause_num,
                    target_type=LegalNodeType.BNSS_SECTION,
                    confidence=0.95,  # High confidence for explicit Clause
                ))
        
        # 2. Extract Section references (BNS or BNSS based on context)
        for match in cls.PATTERNS["bns_section"].finditer(text):
            section_num = match.group(1)
            
            # Determine target based on cross-reference context
            if is_bnss_context:
                target_doc = "bnss.pdf"
            else:
                target_doc = cls._infer_target_document(text_lower, source_document, "section")
            
            key = (target_doc, section_num)
            if key not in seen:
                seen.add(key)
                references.append(ReferenceQuery(
                    target_doc=target_doc,
                    target_section=section_num,
                    target_type=cls._get_node_type(target_doc),
                    confidence=0.9 if source_document else 0.7,
                ))
        
        # 3. Extract Article references (Constitution) - includes Art. abbreviation
        for match in cls.PATTERNS["article"].finditer(text):
            article_num = match.group(1)
            key = ("const.pdf", article_num)
            if key not in seen:
                seen.add(key)
                references.append(ReferenceQuery(
                    target_doc="const.pdf",
                    target_section=article_num,
                    target_type=LegalNodeType.CONST_ARTICLE,
                    confidence=0.95,
                ))
        
        # 4. Extract numeric references (context-dependent, lowest priority)
        for match in cls.PATTERNS["numeric_ref"].finditer(text):
            section_num = match.group(1)
            
            # Use cross-reference context if available
            if is_bnss_context:
                target_doc = "bnss.pdf"
            elif is_const_context:
                target_doc = "const.pdf"
            else:
                target_doc = cls._infer_target_document(text_lower, source_document, "numeric")
            
            key = (target_doc, section_num)
            if key not in seen:
                seen.add(key)
                references.append(ReferenceQuery(
                    target_doc=target_doc,
                    target_section=section_num,
                    target_type=cls._get_node_type(target_doc),
                    confidence=0.6,  # Lower confidence for numeric-only
                ))
        
        return references
    
    @classmethod
    def _is_bnss_cross_reference(cls, text_lower: str) -> bool:
        """Check if text contains explicit BNSS cross-reference markers."""
        bnss_markers = [
            "code of criminal procedure",
            "nagarik suraksha",
            "bnss",
            "as per clause",
            "bail under",
            "arrest under",
            "procedure under",
        ]
        return any(marker in text_lower for marker in bnss_markers)
    
    @classmethod
    def _is_constitution_cross_reference(cls, text_lower: str) -> bool:
        """Check if text contains explicit Constitution cross-reference markers."""
        const_markers = [
            "fundamental rights",
            "constitution of india",
            "constitutional",
            "under article",
            "violates article",
        ]
        return any(marker in text_lower for marker in const_markers)
    
    @classmethod
    def _infer_target_document(
        cls,
        text_lower: str,
        source_document: Optional[str],
        ref_type: str,
    ) -> str:
        """
        Infer the target document from context clues in the text.
        
        Uses keyword matching to determine which legal document
        a reference points to.
        """
        # Check for explicit cross-references
        if "code of criminal procedure" in text_lower or "bnss" in text_lower:
            return "bnss.pdf"
        if "evidence" in text_lower and "act" in text_lower:
            return "bsa.pdf"
        if "constitution" in text_lower or "fundamental" in text_lower:
            return "const.pdf"
        
        # Check document context keywords
        scores: dict[str, int] = {doc: 0 for doc in cls.DOCUMENT_CONTEXT}
        for doc, keywords in cls.DOCUMENT_CONTEXT.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[doc] += 1
        
        # If clear winner, use it
        max_score = max(scores.values())
        if max_score > 0:
            for doc, score in scores.items():
                if score == max_score:
                    return doc
        
        # Default to source document if available
        if source_document:
            return source_document
        
        # Final fallback to BNS
        return "bns.pdf"
    
    @classmethod
    def _get_node_type(cls, target_doc: str) -> LegalNodeType:
        """Map document to node type."""
        mapping = {
            "bns.pdf": LegalNodeType.BNS_SECTION,
            "bnss.pdf": LegalNodeType.BNSS_SECTION,
            "bsa.pdf": LegalNodeType.BSA_SECTION,
            "const.pdf": LegalNodeType.CONST_ARTICLE,
        }
        return mapping.get(target_doc, LegalNodeType.UNKNOWN)


# =============================================================================
# Part 3: Main GraphHybridRetriever Class
# =============================================================================

class GraphHybridRetriever(LoggerMixin):
    """
    High-Fidelity Graph Retrieval Engine for Indian Law.
    
    Implements a 3-hop traversal strategy:
        - Step 0: Concept Hop (LLM maps query to sections)
        - Step 1: Vector Hop (Semantic search)
        - Step 2: Citation Hop (Traverse outbound references)
        - Step 3: Back-Reference Hop (Optional - find related sections)
    
    The engine treats legal documents as a citation network and
    assembles legally complete context windows.
    """
    
    # Concept Bridge System Prompt - Enhanced with specific examples
    CONCEPT_BRIDGE_PROMPT = """You are a Senior Indian Legal Researcher. Map user queries to specific statutory citations.

Knowledge Base:
- BNS (bns.pdf): Criminal offences - Murder(101-103), Rape(63-72), Theft(303-305), Assault(115-120)
- BNSS (bnss.pdf): Procedure - Bail(478-484), Arrest(35-44), FIR(173), Trial, Investigation
- BSA (bsa.pdf): Evidence law - Witness, Testimony, Admissibility
- Constitution (const.pdf): Fundamental Rights - Art.14(Equality), Art.19(Speech), Art.21(Life), Art.22(Arrest), Art.23(Exploitation)

Example 1:
Input: "Bail for mob lynching"
Thinking:
- 'Mob Lynching' = Murder by group -> BNS Section 103(2)
- 'Bail' = Procedure -> BNSS Sections 478-484
Output: {{"concepts": ["mob lynching", "bail conditions"], "citations": [{{"target_doc": "bns.pdf", "target_section": "103"}}, {{"target_doc": "bnss.pdf", "target_section": "479"}}]}}

Example 2:
Input: "What is the punishment for rape?"
Output: {{"concepts": ["rape", "sexual offence punishment"], "citations": [{{"target_doc": "bns.pdf", "target_section": "64"}}]}}

Example 3:
Input: "Right to life under constitution"
Output: {{"concepts": ["right to life", "personal liberty"], "citations": [{{"target_doc": "const.pdf", "target_section": "21"}}]}}

Rules:
- Section numbers = digits only ("103" not "Section 103")
- If unsure, return empty citations array
- Return ONLY valid JSON, no markdown

User Query: {query}

JSON:"""

    # Scope keywords for filtering
    SCOPE_KEYWORDS: dict[str, dict[str, str]] = {
        "bns": {"source_document": "bns.pdf"},
        "nyaya": {"source_document": "bns.pdf"},
        "bnss": {"source_document": "bnss.pdf"},
        "nagarik": {"source_document": "bnss.pdf"},
        "procedure": {"source_document": "bnss.pdf"},
        "bsa": {"source_document": "bsa.pdf"},
        "evidence": {"source_document": "bsa.pdf"},
        "constitution": {"source_document": "const.pdf"},
        "fundamental": {"source_document": "const.pdf"},
        "article": {"source_document": "const.pdf"},
    }

    # Token limit for context window
    MAX_CONTEXT_TOKENS = 8000
    AVG_CHARS_PER_TOKEN = 4

    def __init__(
        self,
        ranker_model: str = "ms-marco-MiniLM-L-12-v2",
        cache_dir: Optional[str] = None,
        enable_concept_bridge: bool = True,
        enable_citation_hop: bool = True,
        enable_back_reference: bool = False,  # Expensive, disabled by default
        max_citation_hops: int = 5,
    ) -> None:
        """
        Initialize the GraphHybridRetriever.
        
        Args:
            ranker_model: FlashRank model for re-ranking.
            cache_dir: Cache directory for ranker model.
            enable_concept_bridge: Use LLM for query-to-section mapping.
            enable_citation_hop: Traverse citation links.
            enable_back_reference: Find sections referencing our results.
            max_citation_hops: Maximum sections to fetch via citation hop.
        """
        self._settings = get_settings()
        
        # Core components
        self._vector_db = VectorDB()
        self._embedding_service = EmbeddingService()
        self._ranker: Optional[Ranker] = None
        self._ranker_model = ranker_model
        self._cache_dir = cache_dir or str(Path.home() / ".cache" / "flashrank")
        
        # LLM for concept bridging
        self._llm: Optional[ChatGroq] = None
        
        # Feature flags
        self._enable_concept_bridge = enable_concept_bridge
        self._enable_citation_hop = enable_citation_hop
        self._enable_back_reference = enable_back_reference
        self._max_citation_hops = max_citation_hops
        
        self.logger.info(
            "GraphHybridRetriever initialized",
            ranker_model=ranker_model,
            concept_bridge=enable_concept_bridge,
            citation_hop=enable_citation_hop,
            back_reference=enable_back_reference,
        )

    # =========================================================================
    # Lazy Initialization
    # =========================================================================

    @property
    def ranker(self) -> Ranker:
        """Lazily initialize FlashRank ranker."""
        if self._ranker is None:
            self.logger.info(
                "Initializing FlashRank ranker",
                model=self._ranker_model,
            )
            Path(self._cache_dir).mkdir(parents=True, exist_ok=True)
            self._ranker = Ranker(
                model_name=self._ranker_model,
                cache_dir=self._cache_dir,
            )
        return self._ranker

    @property
    def llm(self) -> ChatGroq:
        """Lazily initialize Groq LLM for concept bridging."""
        if self._llm is None:
            self.logger.info("Initializing Groq LLM for concept bridging")
            self._llm = ChatGroq(
                model="llama-3.1-8b-instant",  # Fast model for bridging
                temperature=0.1,
                max_tokens=500,
                api_key=self._settings.GROQ_API_KEY,
            )
        return self._llm

    # =========================================================================
    # Part 2: Concept Bridge (LLM-Based Expansion)
    # =========================================================================

    async def _bridge_concepts(self, query: str) -> ConceptBridgeResult:
        """
        Use LLM to bridge semantic gap between user query and legal sections.
        
        Maps colloquial queries to specific statutory citations.
        
        Args:
            query: User's natural language query.
            
        Returns:
            ConceptBridgeResult: Mapped concepts and citations.
        """
        if not self._enable_concept_bridge:
            return ConceptBridgeResult()
        
        try:
            prompt = self.CONCEPT_BRIDGE_PROMPT.format(query=query)
            
            messages = [
                SystemMessage(content="You are a legal citation expert. Return only valid JSON."),
                HumanMessage(content=prompt),
            ]
            
            response = await self.llm.ainvoke(messages)
            raw_response = response.content.strip()
            
            # Parse JSON response
            # Handle potential markdown code blocks
            json_str = raw_response
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]
            
            data = json.loads(json_str.strip())
            
            citations = []
            for cit in data.get("citations", []):
                if isinstance(cit, dict) and "target_doc" in cit and "target_section" in cit:
                    citations.append(ReferenceQuery(
                        target_doc=cit["target_doc"],
                        target_section=str(cit["target_section"]).replace("(", "").split(")")[0],
                        confidence=0.95,
                    ))
            
            result = ConceptBridgeResult(
                concepts=data.get("concepts", []),
                citations=citations,
                raw_response=raw_response,
            )
            
            self.logger.info(
                "Concept bridge completed",
                concepts=result.concepts,
                citations_found=len(result.citations),
            )
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.warning(
                "Failed to parse concept bridge response",
                error=str(e),
                raw_response=raw_response[:200] if 'raw_response' in dir() else "N/A",
            )
            return ConceptBridgeResult(raw_response=str(e))
            
        except Exception as e:
            self.logger.warning(
                "Concept bridge failed",
                error=str(e),
            )
            return ConceptBridgeResult()

    # =========================================================================
    # Part 3: Reference Extraction
    # =========================================================================

    def _extract_references(
        self,
        chunks: list[LegalChunk],
    ) -> list[ReferenceQuery]:
        """
        Extract all outbound references from chunk texts.
        
        Args:
            chunks: List of chunks to scan.
            
        Returns:
            list[ReferenceQuery]: Unique references found.
        """
        all_refs: list[ReferenceQuery] = []
        seen: set[tuple[str, str]] = set()
        
        # Track existing sections to avoid fetching what we have
        existing_sections = {
            (chunk.metadata.source_document, chunk.metadata.section_number)
            for chunk in chunks
            if chunk.metadata.section_number
        }
        
        for chunk in chunks:
            refs = LegalReferenceExtractor.extract_references(
                text=chunk.text,
                source_document=chunk.metadata.source_document,
            )
            
            for ref in refs:
                key = (ref.target_doc, ref.target_section)
                # Skip duplicates and existing sections
                if key not in seen and key not in existing_sections:
                    seen.add(key)
                    all_refs.append(ref)
        
        return all_refs

    # =========================================================================
    # Part 4: Graph Traversal Operations
    # =========================================================================

    async def _fetch_by_metadata(
        self,
        references: list[ReferenceQuery],
        source_tag: ChunkSource,
    ) -> list[LegalChunk]:
        """
        Fetch chunks by metadata (section number + document).
        
        Uses Qdrant metadata filtering for O(1) lookups.
        
        Args:
            references: List of reference queries.
            source_tag: Tag for chunk source tracking.
            
        Returns:
            list[LegalChunk]: Fetched chunks with source tags.
        """
        if not references:
            return []
        
        chunks: list[LegalChunk] = []
        fetched_ids: set[str] = set()
        
        # Parallel fetch using asyncio
        loop = asyncio.get_event_loop()
        
        async def fetch_one(ref: ReferenceQuery) -> list[LegalChunk]:
            """Fetch a single reference."""
            try:
                filters = {
                    "section_number": ref.target_section,
                    "source_document": ref.target_doc,
                }
                
                results = await loop.run_in_executor(
                    None,
                    lambda: self._vector_db.search_by_metadata(
                        filters=filters,
                        limit=2,  # Parent + child
                    ),
                )
                
                return results
                
            except Exception as e:
                self.logger.warning(
                    f"Failed to fetch {ref.target_doc} Section {ref.target_section}",
                    error=str(e),
                )
                return []
        
        # Gather all fetches in parallel
        results = await asyncio.gather(
            *[fetch_one(ref) for ref in references[:self._max_citation_hops]],
            return_exceptions=True,
        )
        
        # Process results
        for ref, result in zip(references, results):
            if isinstance(result, Exception):
                continue
            
            for chunk in result:
                if chunk.chunk_id not in fetched_ids:
                    # Tag the source
                    chunk.__dict__["source"] = source_tag.value
                    chunk.__dict__["ref_from"] = f"{ref.target_doc}:{ref.target_section}"
                    chunks.append(chunk)
                    fetched_ids.add(chunk.chunk_id)
        
        return chunks

    async def _vector_search(
        self,
        query: str,
        expanded_query: Optional[str],
        filters: Optional[dict[str, str]],
        candidates: int = 15,
        score_threshold: float = 0.30,
    ) -> list[LegalChunk]:
        """
        Perform vector similarity search.
        
        Args:
            query: Original user query.
            expanded_query: LLM-expanded query (if available).
            filters: Scope filters.
            candidates: Number of candidates to fetch.
            score_threshold: Minimum similarity score.
            
        Returns:
            list[LegalChunk]: Chunks tagged as vector_search.
        """
        # Use expanded query if available
        search_query = expanded_query if expanded_query else query
        
        # Generate embedding
        loop = asyncio.get_event_loop()
        query_vector = await loop.run_in_executor(
            None,
            lambda: self._embedding_service.embed_query(search_query),
        )
        
        # Execute search
        chunks = self._vector_db.search(
            query_vector=query_vector,
            limit=candidates,
            score_threshold=score_threshold,
            filters=filters,
        )
        
        # Tag chunks
        for chunk in chunks:
            chunk.__dict__["source"] = ChunkSource.VECTOR_SEARCH.value
        
        return chunks

    def _detect_scope(self, query: str) -> Optional[dict[str, str]]:
        """
        Detect scope filters from query keywords.
        
        Args:
            query: User's query.
            
        Returns:
            Optional[dict[str, str]]: Filters for scope.
        """
        query_lower = query.lower()
        
        for keyword, filters in self.SCOPE_KEYWORDS.items():
            if keyword in query_lower:
                self.logger.debug(
                    "Scope detected",
                    keyword=keyword,
                    filters=filters,
                )
                return filters
        
        return None

    # =========================================================================
    # Part 5: Context Assembly & Re-Ranking
    # =========================================================================

    def _deduplicate_and_prioritize(
        self,
        concept_chunks: list[LegalChunk],
        vector_chunks: list[LegalChunk],
        citation_chunks: list[LegalChunk],
        backref_chunks: list[LegalChunk],
    ) -> list[LegalChunk]:
        """
        Merge all chunks with deduplication and priority sorting.
        
        Priority Order:
            1. concept_graph (Direct statutory hits)
            2. citation_graph (Legally linked context)
            3. vector_search (Semantic matches)
            4. back_reference (Related sections)
        
        Args:
            concept_chunks: Chunks from concept bridging.
            vector_chunks: Chunks from vector search.
            citation_chunks: Chunks from citation traversal.
            backref_chunks: Chunks from back-reference search.
            
        Returns:
            list[LegalChunk]: Merged, deduplicated, prioritized list.
        """
        seen_ids: set[str] = set()
        merged: list[LegalChunk] = []
        
        # Add in priority order
        for priority, chunks in enumerate([
            concept_chunks,    # Priority 0 (highest)
            citation_chunks,   # Priority 1
            vector_chunks,     # Priority 2
            backref_chunks,    # Priority 3 (lowest)
        ]):
            for chunk in chunks:
                if chunk.chunk_id not in seen_ids:
                    chunk.__dict__["priority"] = priority
                    merged.append(chunk)
                    seen_ids.add(chunk.chunk_id)
        
        return merged

    def _truncate_to_token_limit(
        self,
        chunks: list[LegalChunk],
        max_tokens: int,
    ) -> list[LegalChunk]:
        """
        Truncate chunks to fit within token limit.
        
        Drops lowest priority chunks first.
        
        Args:
            chunks: Prioritized chunks.
            max_tokens: Maximum tokens allowed.
            
        Returns:
            list[LegalChunk]: Truncated list.
        """
        max_chars = max_tokens * self.AVG_CHARS_PER_TOKEN
        
        # Sort by priority (lower = higher priority)
        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.__dict__.get("priority", 999),
        )
        
        result: list[LegalChunk] = []
        total_chars = 0
        
        for chunk in sorted_chunks:
            chunk_chars = len(chunk.text)
            if total_chars + chunk_chars <= max_chars:
                result.append(chunk)
                total_chars += chunk_chars
            else:
                # Stop adding chunks
                break
        
        return result

    async def _rerank_chunks(
        self,
        query: str,
        chunks: list[LegalChunk],
        top_k: int = 10,
    ) -> list[LegalChunk]:
        """
        Re-rank chunks using FlashRank cross-encoder.
        
        Args:
            query: User's query.
            chunks: Chunks to re-rank.
            top_k: Number of top results to return.
            
        Returns:
            list[LegalChunk]: Re-ranked chunks.
        """
        if not chunks:
            return []
        
        if len(chunks) <= top_k:
            return chunks
        
        # Prepare passages for FlashRank
        passages = [
            {"id": chunk.chunk_id, "text": chunk.text[:2000]}
            for chunk in chunks
        ]
        
        # Create chunk lookup
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        
        # Execute re-ranking
        loop = asyncio.get_event_loop()
        
        def do_rerank():
            request = RerankRequest(query=query, passages=passages)
            return self.ranker.rerank(request)
        
        rerank_results = await loop.run_in_executor(None, do_rerank)
        
        # Build reranked list
        reranked: list[LegalChunk] = []
        for result in rerank_results[:top_k]:
            chunk_id = result.get("id") or result.get("passage", {}).get("id")
            if chunk_id and chunk_id in chunk_map:
                chunk = chunk_map[chunk_id]
                chunk.__dict__["rerank_score"] = result.get("score", 0.0)
                reranked.append(chunk)
        
        return reranked

    # =========================================================================
    # Part 6: Main Search Method (3-Hop Strategy)
    # =========================================================================

    async def search(
        self,
        query: str,
        k: int = 10,
        candidates: int = 20,
        score_threshold: float = 0.30,
        filters: Optional[dict[str, str]] = None,
        enable_trace: bool = True,
    ) -> list[LegalChunk]:
        """
        Execute the 3-Hop Graph Retrieval Strategy.
        
        Step 0: Concept Hop - LLM maps query to specific sections
        Step 1: Vector Hop - Semantic search for recall
        Step 2: Citation Hop - Traverse outbound references
        Step 3: Back-Ref Hop - (Optional) Find related sections
        
        Args:
            query: User's natural language query.
            k: Final number of chunks to return.
            candidates: Candidates for vector search.
            score_threshold: Minimum similarity score.
            filters: Optional scope filters (overrides auto-detection).
            enable_trace: Print graph traversal trace.
            
        Returns:
            list[LegalChunk]: Legally complete context window.
            
        Raises:
            SearchError: If search fails.
        """
        trace = GraphTraceLog()
        
        self.logger.info(
            "Starting Graph Retrieval",
            query=query[:100],
            k=k,
        )

        try:
            # Detect scope if not provided
            search_filters = filters or self._detect_scope(query)
            
            # =============================================
            # Step 0: Concept Hop (The Precision Layer)
            # =============================================
            concept_chunks: list[LegalChunk] = []
            concept_result = await self._bridge_concepts(query)
            
            if concept_result.citations:
                # Log concept mappings
                for cit in concept_result.citations:
                    trace.concept_mappings.append(
                        f"'{query[:30]}...' -> {cit.target_doc} Sec {cit.target_section}"
                    )
                
                # Fetch directly by metadata
                concept_chunks = await self._fetch_by_metadata(
                    references=concept_result.citations,
                    source_tag=ChunkSource.CONCEPT_GRAPH,
                )
                
                # If metadata fetch fails (due to chunking issues), fallback to semantic search
                # using the concept names directly - this catches cases where section_number
                # metadata is null but the content exists
                if not concept_chunks and concept_result.concepts:
                    self.logger.info(
                        "Metadata fetch empty, using semantic concept search",
                        concepts=concept_result.concepts,
                    )
                    # Build a concept-enriched query
                    concept_query = " ".join(concept_result.concepts)
                    concept_chunks = await self._vector_search(
                        query=concept_query,
                        expanded_query=None,
                        filters=search_filters,
                        candidates=10,
                        score_threshold=0.35,
                    )
                    # Tag these as concept graph
                    for chunk in concept_chunks:
                        chunk.__dict__["source"] = ChunkSource.CONCEPT_GRAPH.value
                
                trace.concept_count = len(concept_chunks)
                
                self.logger.info(
                    "Concept hop completed",
                    citations_mapped=len(concept_result.citations),
                    chunks_fetched=len(concept_chunks),
                )
            
            # =============================================
            # Step 1: Vector Hop (The Recall Layer)
            # =============================================
            # Build expanded query from concepts
            expanded_query = None
            if concept_result.concepts:
                expanded_query = f"{query} {' '.join(concept_result.concepts)}"
            
            vector_chunks = await self._vector_search(
                query=query,
                expanded_query=expanded_query,
                filters=search_filters,
                candidates=candidates,
                score_threshold=score_threshold,
            )
            trace.vector_hits = len(vector_chunks)
            
            self.logger.info(
                "Vector hop completed",
                chunks_found=len(vector_chunks),
            )
            
            # =============================================
            # Step 2: Citation Hop (The Connection Layer)
            # =============================================
            citation_chunks: list[LegalChunk] = []
            
            if self._enable_citation_hop:
                # Combine concept + vector chunks for reference extraction
                all_chunks = concept_chunks + vector_chunks
                
                # Extract outbound references
                references = self._extract_references(all_chunks)
                
                if references:
                    for ref in references[:10]:
                        trace.citation_links.append(
                            f"{ref.target_doc} Sec {ref.target_section}"
                        )
                    
                    # Fetch cited sections in parallel
                    citation_chunks = await self._fetch_by_metadata(
                        references=references,
                        source_tag=ChunkSource.CITATION_GRAPH,
                    )
                    trace.citation_count = len(citation_chunks)
                    
                    self.logger.info(
                        "Citation hop completed",
                        references_found=len(references),
                        chunks_fetched=len(citation_chunks),
                    )
            
            # =============================================
            # Step 3: Back-Reference Hop (Optional)
            # =============================================
            backref_chunks: list[LegalChunk] = []
            
            if self._enable_back_reference:
                # TODO: Implement back-reference search
                # This would search for chunks that REFERENCE our key sections
                pass
            
            # =============================================
            # Part 5: Context Assembly & Re-Ranking
            # =============================================
            
            # Merge and deduplicate with priority
            merged_chunks = self._deduplicate_and_prioritize(
                concept_chunks=concept_chunks,
                vector_chunks=vector_chunks,
                citation_chunks=citation_chunks,
                backref_chunks=backref_chunks,
            )
            
            # Truncate to token limit
            truncated_chunks = self._truncate_to_token_limit(
                chunks=merged_chunks,
                max_tokens=self.MAX_CONTEXT_TOKENS,
            )
            
            # Re-rank the final set
            final_chunks = await self._rerank_chunks(
                query=query,
                chunks=truncated_chunks,
                top_k=k,
            )
            
            trace.final_count = len(final_chunks)
            
            # Print trace if enabled
            if enable_trace:
                trace.print_trace()
            
            self.logger.info(
                "Graph retrieval completed",
                final_chunks=len(final_chunks),
                concept_chunks=trace.concept_count,
                citation_chunks=trace.citation_count,
                vector_chunks=trace.vector_hits,
            )
            
            return final_chunks

        except Exception as e:
            self.logger.error(
                "Graph retrieval failed",
                query=query[:50],
                error=str(e),
                error_type=type(e).__name__,
            )
            raise SearchError(
                message="Graph retrieval failed",
                details={"query": query[:100], "error": str(e)},
            ) from e

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def search_simple(
        self,
        query: str,
        k: int = 5,
    ) -> list[LegalChunk]:
        """
        Simple search without graph traversal (for testing).
        
        Args:
            query: User query.
            k: Number of results.
            
        Returns:
            list[LegalChunk]: Search results.
        """
        # Temporarily disable features
        original_concept = self._enable_concept_bridge
        original_citation = self._enable_citation_hop
        
        self._enable_concept_bridge = False
        self._enable_citation_hop = False
        
        try:
            results = await self.search(query, k=k, enable_trace=False)
            return results
        finally:
            self._enable_concept_bridge = original_concept
            self._enable_citation_hop = original_citation


# =============================================================================
# Backward Compatibility Alias
# =============================================================================

# Alias for backward compatibility with existing code
HybridRetriever = GraphHybridRetriever
