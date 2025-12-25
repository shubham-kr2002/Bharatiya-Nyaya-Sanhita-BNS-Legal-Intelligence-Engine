"""LLM generation module for legal question answering."""

from typing import AsyncIterator, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_groq import ChatGroq

from core.config import get_settings
from core.exceptions import LLMError
from core.logger import get_logger, LoggerMixin
from models.schema import LegalChunk

logger = get_logger(__name__)


# Legal Expert System Prompt
LEGAL_SYSTEM_PROMPT = """You are an Expert Indian Legal Consultant with deep knowledge of:
- Bharatiya Nyaya Sanhita (BNS) - The new Indian Penal Code
- Bharatiya Nagarik Suraksha Sanhita (BNSS) - The new Criminal Procedure Code  
- Bharatiya Sakshya Adhiniyam (BSA) - The new Indian Evidence Act
- Constitution of India

## Your Responsibilities:
1. Provide accurate, well-researched legal information based ONLY on the provided context
2. Always cite specific Act names and Section numbers in your answers
3. Explain legal concepts in clear, accessible language
4. Highlight important legal provisions and their implications

## Strict Rules:
- Answer ONLY using the provided context chunks
- ALWAYS cite sources using format: "According to Section X of [Act Name]..."
- If information is NOT in the context, clearly state: "I cannot find this information in the provided legal database."
- Never make up legal provisions or section numbers
- Format responses in clean Markdown with appropriate headings

## Response Format:
- Use ## headings for main sections
- Use **bold** for Act names and Section numbers
- Use bullet points for listing provisions
- Include a "📚 Sources" section at the end citing all referenced sections"""


LEGAL_QA_TEMPLATE = """## Context from Legal Database:
{context}

---

## User Question:
{question}

---

## Instructions:
Based on the legal context provided above, answer the user's question comprehensively.
Remember to cite specific sections and acts for every legal point you make.

## Answer:"""


class LegalGenerator(LoggerMixin):
    """
    Handles LLM interactions for legal question answering.
    
    Uses Groq's LLaMA models for fast, high-quality legal responses
    with strict citation and factuality requirements.
    """

    # Available models (in order of capability)
    MODELS = {
        "high_intelligence": "llama-3.3-70b-versatile",
        "balanced": "llama-3.1-8b-instant",
        "fast": "llama-3.1-8b-instant",
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> None:
        """
        Initialize the LegalGenerator with Groq LLM.
        
        Args:
            model_name: Groq model name (default: llama3-70b-8192).
            temperature: Sampling temperature (default: 0.1 for factuality).
            max_tokens: Maximum tokens in response (default: 4096).
        """
        self._settings = get_settings()
        self._model_name = model_name or self.MODELS["high_intelligence"]
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._llm: Optional[ChatGroq] = None

        self.logger.info(
            "LegalGenerator initialized",
            model=self._model_name,
            temperature=temperature,
        )

    @property
    def llm(self) -> ChatGroq:
        """
        Lazily initialize and return the ChatGroq LLM.
        
        Returns:
            ChatGroq: Configured LLM instance.
        """
        if self._llm is None:
            self._llm = ChatGroq(
                model=self._model_name,
                api_key=self._settings.GROQ_API_KEY,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                streaming=True,
            )
            self.logger.info(
                "ChatGroq LLM initialized",
                model=self._model_name,
            )
        return self._llm

    def _get_prompt_template(self) -> ChatPromptTemplate:
        """
        Create the legal QA prompt template.
        
        Returns:
            ChatPromptTemplate: Configured prompt template.
        """
        return ChatPromptTemplate.from_messages([
            ("system", LEGAL_SYSTEM_PROMPT),
            ("human", LEGAL_QA_TEMPLATE),
        ])

    def _format_context(self, chunks: list[LegalChunk]) -> str:
        """
        Format legal chunks into context string.
        
        Args:
            chunks: List of LegalChunk objects.
            
        Returns:
            str: Formatted context string with source citations.
        """
        if not chunks:
            return "No relevant context found in the legal database."

        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            # Build source header
            source_parts = [f"Source {i}"]
            
            if chunk.metadata.source_document:
                source_parts.append(chunk.metadata.source_document)
            
            if chunk.metadata.section_number:
                source_parts.append(f"Section {chunk.metadata.section_number}")
            
            if chunk.metadata.chapter_title:
                source_parts.append(chunk.metadata.chapter_title)
            
            source_header = " | ".join(source_parts)
            
            # Get rerank score if available
            score = chunk.__dict__.get("rerank_score", chunk.__dict__.get("score", "N/A"))
            if isinstance(score, float):
                score = f"{score:.3f}"
            
            # Format chunk
            chunk_text = f"""### [{source_header}]
**Act:** {chunk.metadata.act_name}
**Page:** {chunk.metadata.page_number}
**Relevance Score:** {score}

{chunk.text}
"""
            context_parts.append(chunk_text)
        
        return "\n---\n".join(context_parts)

    def _build_chain(self):
        """
        Build the LangChain processing chain.
        
        Returns:
            Chain: Prompt -> LLM -> Parser chain.
        """
        prompt = self._get_prompt_template()
        parser = StrOutputParser()
        
        return prompt | self.llm | parser

    async def get_answer_stream(
        self,
        query: str,
        context_chunks: list[LegalChunk],
    ) -> AsyncIterator[str]:
        """
        Generate a streaming answer for the legal query.
        
        This method returns an async iterator that yields chunks of text,
        enabling a typing effect in the frontend.
        
        Args:
            query: User's legal question.
            context_chunks: Retrieved legal chunks for context.
            
        Yields:
            str: Chunks of the generated answer.
            
        Raises:
            LLMError: If generation fails.
        """
        self.logger.info(
            "Starting answer generation",
            query=query[:100],
            context_chunks=len(context_chunks),
        )

        try:
            # Format context
            context = self._format_context(context_chunks)
            
            # Build chain
            chain = self._build_chain()
            
            # Stream response
            async for chunk in chain.astream({
                "context": context,
                "question": query,
            }):
                yield chunk

            self.logger.info(
                "Answer generation completed",
                query=query[:50],
            )

        except Exception as e:
            self.logger.error(
                "Answer generation failed",
                query=query[:50],
                error=str(e),
                error_type=type(e).__name__,
            )
            raise LLMError(
                message="Failed to generate answer",
                details={"query": query[:100], "error": str(e)},
            ) from e

    async def get_answer(
        self,
        query: str,
        context_chunks: list[LegalChunk],
    ) -> str:
        """
        Generate a complete answer (non-streaming).
        
        Args:
            query: User's legal question.
            context_chunks: Retrieved legal chunks for context.
            
        Returns:
            str: Complete generated answer.
            
        Raises:
            LLMError: If generation fails.
        """
        self.logger.info(
            "Generating complete answer",
            query=query[:100],
            context_chunks=len(context_chunks),
        )

        try:
            # Format context
            context = self._format_context(context_chunks)
            
            # Build chain
            chain = self._build_chain()
            
            # Invoke chain
            response = await chain.ainvoke({
                "context": context,
                "question": query,
            })

            self.logger.info(
                "Answer generated",
                query=query[:50],
                response_length=len(response),
            )

            return response

        except Exception as e:
            self.logger.error(
                "Answer generation failed",
                query=query[:50],
                error=str(e),
                error_type=type(e).__name__,
            )
            raise LLMError(
                message="Failed to generate answer",
                details={"query": query[:100], "error": str(e)},
            ) from e

    async def summarize_section(
        self,
        section_text: str,
        section_number: str,
        act_name: str,
    ) -> str:
        """
        Generate a summary of a legal section.
        
        Args:
            section_text: Full text of the legal section.
            section_number: Section number.
            act_name: Name of the act.
            
        Returns:
            str: Summarized explanation of the section.
        """
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an Expert Indian Legal Consultant. 
Summarize the following legal section in clear, accessible language.
Include:
1. Main purpose of the section
2. Key provisions
3. Practical implications
4. Any important penalties or consequences"""),
            ("human", """## Section {section_number} of {act_name}

{section_text}

Provide a comprehensive yet concise summary:"""),
        ])

        chain = summary_prompt | self.llm | StrOutputParser()
        
        return await chain.ainvoke({
            "section_number": section_number,
            "act_name": act_name,
            "section_text": section_text,
        })
