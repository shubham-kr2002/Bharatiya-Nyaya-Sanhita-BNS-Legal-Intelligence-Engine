"""
RAG Evaluation Benchmark Pipeline.

Automated testing pipeline to measure RAG system accuracy using:
1. Synthetic test set generation from Qdrant chunks
2. LLM-as-Judge evaluation methodology
3. Faithfulness and correctness scoring
"""

import asyncio
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from core.config import get_settings
from core.generator import LegalGenerator
from core.logger import get_logger, LoggerMixin
from core.retriever import HybridRetriever
from database.vector_store import VectorDB
from models.schema import LegalChunk

logger = get_logger(__name__)

# Evaluation directory
EVAL_DIR = Path(__file__).parent
TEST_SET_PATH = EVAL_DIR / "test_set.json"
REPORT_PATH = EVAL_DIR / "report.md"


# ============================================================
# Pydantic Models for Evaluation
# ============================================================

class TestQuestion(BaseModel):
    """A single test question with ground truth."""
    
    id: int
    question: str
    correct_answer: str
    source_chunk_id: str
    source_text: str = Field(default="", description="Original chunk text")
    metadata: dict[str, Any] = Field(default_factory=dict)


class GradeResult(BaseModel):
    """Grading result from LLM judge."""
    
    faithfulness: float = Field(ge=0.0, le=1.0, description="How faithful is the answer to sources")
    correctness: float = Field(ge=0.0, le=1.0, description="How correct is the answer")
    reasoning: str = Field(default="", description="Judge's reasoning")


class EvaluationResult(BaseModel):
    """Result for a single question evaluation."""
    
    question_id: int
    question: str
    correct_answer: str
    agent_answer: str
    faithfulness: float
    correctness: float
    reasoning: str
    retrieval_count: int = 0


class BenchmarkReport(BaseModel):
    """Complete benchmark report."""
    
    timestamp: str
    total_questions: int
    mean_faithfulness: float
    mean_correctness: float
    mean_accuracy: float
    results: list[EvaluationResult]


# ============================================================
# RAG Judge Class
# ============================================================

class RAGJudge(LoggerMixin):
    """
    Automated evaluation pipeline for RAG system.
    
    Uses LLM-as-Judge methodology to:
    1. Generate synthetic test sets from vector store
    2. Evaluate agent responses against ground truth
    3. Score faithfulness and correctness
    """

    # Prompt for generating test questions
    QUESTION_GEN_PROMPT = """You are a legal exam creator for Indian Law (BNS, BNSS, BSA, Constitution).

Based on the following legal text, generate ONE specific, challenging legal question that can only be answered correctly by someone who has read this exact text.

Legal Text:
---
{chunk_text}
---

Requirements:
1. The question should be specific and require precise knowledge
2. The question should NOT be answerable with general legal knowledge
3. Provide the correct answer based ONLY on the given text

Respond in this exact JSON format:
{{"question": "Your specific legal question here", "answer": "The correct answer based on the text"}}

JSON:"""

    # Prompt for grading agent responses
    GRADING_PROMPT = """You are an impartial legal exam grader. Compare the Agent's Answer to the Correct Answer.

Question: {question}

Correct Answer (Ground Truth):
{correct_answer}

Agent's Answer:
{agent_answer}

Evaluate on two criteria:
1. **Faithfulness** (0-1): Does the agent's answer stay true to legal facts without hallucination?
2. **Correctness** (0-1): Does the agent's answer match the correct answer in meaning and key details?

Scoring Guide:
- 1.0 = Perfect match/completely faithful
- 0.7-0.9 = Mostly correct with minor issues
- 0.4-0.6 = Partially correct
- 0.1-0.3 = Mostly incorrect
- 0.0 = Completely wrong or hallucinated

Respond in this exact JSON format:
{{"faithfulness": 0.8, "correctness": 0.9, "reasoning": "Brief explanation of scores"}}

JSON:"""

    def __init__(self) -> None:
        """Initialize the RAG Judge with all components."""
        self._settings = get_settings()
        self._vector_db = VectorDB()
        self._retriever: Optional[HybridRetriever] = None
        self._generator: Optional[LegalGenerator] = None
        self._llm: Optional[ChatGroq] = None

        self.logger.info("RAGJudge initialized")

    @property
    def llm(self) -> ChatGroq:
        """Lazily initialize LLM for judging."""
        if self._llm is None:
            self._llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=self._settings.GROQ_API_KEY,
                temperature=0.0,  # Deterministic for evaluation
                max_tokens=500,
            )
        return self._llm

    @property
    def retriever(self) -> HybridRetriever:
        """Lazily initialize retriever."""
        if self._retriever is None:
            self._retriever = HybridRetriever()
        return self._retriever

    @property
    def generator(self) -> LegalGenerator:
        """Lazily initialize generator."""
        if self._generator is None:
            self._generator = LegalGenerator()
        return self._generator

    def _sample_chunks_from_qdrant(self, num_samples: int = 10) -> list[LegalChunk]:
        """
        Randomly sample chunks from Qdrant for test generation.
        
        Args:
            num_samples: Number of chunks to sample.
            
        Returns:
            list[LegalChunk]: Sampled chunks.
        """
        self.logger.info(f"Sampling {num_samples} chunks from Qdrant")
        
        # Get collection info to determine total points
        client = self._vector_db.client
        collection_name = self._vector_db.COLLECTION_NAME
        collection_info = client.get_collection(collection_name)
        total_points = collection_info.points_count
        
        if total_points == 0:
            raise ValueError("No points in Qdrant collection")
        
        # Generate random offsets
        offsets = random.sample(range(min(total_points, 1000)), min(num_samples * 2, total_points))
        
        # Scroll through points at random offsets
        sampled_chunks: list[LegalChunk] = []
        
        for offset in offsets:
            if len(sampled_chunks) >= num_samples:
                break
                
            points, _ = client.scroll(
                collection_name=collection_name,
                limit=1,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            
            for point in points:
                if len(sampled_chunks) >= num_samples:
                    break
                    
                payload = point.payload or {}
                text = payload.get("text", "")
                
                # Skip very short chunks
                if len(text) < 200:
                    continue
                
                chunk = LegalChunk(
                    chunk_id=str(point.id),
                    text=text,
                    metadata=payload.get("metadata", {}),
                    embedding=[],
                )
                sampled_chunks.append(chunk)
        
        self.logger.info(f"Sampled {len(sampled_chunks)} valid chunks")
        return sampled_chunks

    async def _generate_question_from_chunk(
        self,
        chunk: LegalChunk,
        question_id: int,
    ) -> Optional[TestQuestion]:
        """
        Generate a test question from a chunk using LLM.
        
        Args:
            chunk: Source chunk for question generation.
            question_id: ID for the question.
            
        Returns:
            TestQuestion or None if generation fails.
        """
        try:
            prompt = self.QUESTION_GEN_PROMPT.format(chunk_text=chunk.text[:2000])
            response = await self.llm.ainvoke(prompt)
            
            # Parse JSON response
            content = response.content.strip()
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            data = json.loads(content)
            
            return TestQuestion(
                id=question_id,
                question=data["question"],
                correct_answer=data["answer"],
                source_chunk_id=chunk.chunk_id,
                source_text=chunk.text[:500],
                metadata=chunk.metadata.model_dump() if hasattr(chunk.metadata, 'model_dump') else dict(chunk.metadata),
            )
            
        except Exception as e:
            self.logger.warning(
                f"Failed to generate question from chunk {chunk.chunk_id}",
                error=str(e),
            )
            return None

    async def generate_test_set(self, num_questions: int = 10) -> list[TestQuestion]:
        """
        Generate a golden test set from Qdrant chunks.
        
        Args:
            num_questions: Number of test questions to generate.
            
        Returns:
            list[TestQuestion]: Generated test set.
        """
        self.logger.info(f"Generating test set with {num_questions} questions")
        
        # Sample chunks from Qdrant
        chunks = self._sample_chunks_from_qdrant(num_questions * 2)  # Sample extra for failures
        
        # Generate questions
        test_questions: list[TestQuestion] = []
        
        for i, chunk in enumerate(chunks):
            if len(test_questions) >= num_questions:
                break
                
            self.logger.info(f"Generating question {len(test_questions) + 1}/{num_questions}")
            
            question = await self._generate_question_from_chunk(chunk, len(test_questions) + 1)
            if question:
                test_questions.append(question)
            
            # Rate limiting
            await asyncio.sleep(0.5)
        
        # Save to JSON
        self._save_test_set(test_questions)
        
        self.logger.info(f"Generated {len(test_questions)} test questions")
        return test_questions

    def _save_test_set(self, questions: list[TestQuestion]) -> None:
        """Save test set to JSON file."""
        data = [q.model_dump() for q in questions]
        TEST_SET_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        self.logger.info(f"Saved test set to {TEST_SET_PATH}")

    def _load_test_set(self) -> list[TestQuestion]:
        """Load test set from JSON file."""
        if not TEST_SET_PATH.exists():
            raise FileNotFoundError(f"Test set not found at {TEST_SET_PATH}. Run generate_test_set() first.")
        
        data = json.loads(TEST_SET_PATH.read_text())
        return [TestQuestion(**q) for q in data]

    async def _get_agent_answer(self, question: str) -> tuple[str, int]:
        """
        Get agent's answer using the full RAG pipeline.
        
        Args:
            question: The question to answer.
            
        Returns:
            tuple[str, int]: (answer, retrieval_count)
        """
        # Retrieve relevant chunks
        chunks = await self.retriever.search(question, k=5)
        
        # Generate answer
        answer_parts: list[str] = []
        async for token in self.generator.get_answer_stream(question, chunks):
            answer_parts.append(token)
        
        return "".join(answer_parts), len(chunks)

    async def _grade_answer(
        self,
        question: str,
        correct_answer: str,
        agent_answer: str,
    ) -> GradeResult:
        """
        Grade agent's answer using LLM-as-Judge.
        
        Args:
            question: The original question.
            correct_answer: Ground truth answer.
            agent_answer: Agent's generated answer.
            
        Returns:
            GradeResult: Faithfulness and correctness scores.
        """
        try:
            prompt = self.GRADING_PROMPT.format(
                question=question,
                correct_answer=correct_answer,
                agent_answer=agent_answer[:2000],  # Truncate long answers
            )
            
            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()
            
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            data = json.loads(content)
            
            return GradeResult(
                faithfulness=float(data.get("faithfulness", 0.5)),
                correctness=float(data.get("correctness", 0.5)),
                reasoning=data.get("reasoning", ""),
            )
            
        except Exception as e:
            self.logger.warning(f"Grading failed: {e}")
            return GradeResult(
                faithfulness=0.5,
                correctness=0.5,
                reasoning=f"Grading error: {str(e)}",
            )

    async def evaluate_agent(self) -> BenchmarkReport:
        """
        Evaluate the RAG agent on the test set.
        
        Returns:
            BenchmarkReport: Complete evaluation results.
        """
        self.logger.info("Starting agent evaluation")
        
        # Load test set
        questions = self._load_test_set()
        self.logger.info(f"Loaded {len(questions)} test questions")
        
        results: list[EvaluationResult] = []
        
        for i, q in enumerate(questions):
            self.logger.info(f"Evaluating question {i + 1}/{len(questions)}: {q.question[:50]}...")
            
            try:
                # Get agent's answer
                agent_answer, retrieval_count = await self._get_agent_answer(q.question)
                
                # Grade the answer
                grade = await self._grade_answer(
                    question=q.question,
                    correct_answer=q.correct_answer,
                    agent_answer=agent_answer,
                )
                
                result = EvaluationResult(
                    question_id=q.id,
                    question=q.question,
                    correct_answer=q.correct_answer,
                    agent_answer=agent_answer[:1000],  # Truncate for report
                    faithfulness=grade.faithfulness,
                    correctness=grade.correctness,
                    reasoning=grade.reasoning,
                    retrieval_count=retrieval_count,
                )
                results.append(result)
                
                self.logger.info(
                    f"Q{q.id}: Faithfulness={grade.faithfulness:.2f}, "
                    f"Correctness={grade.correctness:.2f}"
                )
                
            except Exception as e:
                self.logger.error(f"Evaluation failed for Q{q.id}: {e}")
                results.append(EvaluationResult(
                    question_id=q.id,
                    question=q.question,
                    correct_answer=q.correct_answer,
                    agent_answer=f"ERROR: {str(e)}",
                    faithfulness=0.0,
                    correctness=0.0,
                    reasoning=f"Evaluation error: {str(e)}",
                ))
            
            # Rate limiting
            await asyncio.sleep(1.0)
        
        # Calculate metrics
        mean_faithfulness = sum(r.faithfulness for r in results) / len(results) if results else 0
        mean_correctness = sum(r.correctness for r in results) / len(results) if results else 0
        mean_accuracy = (mean_faithfulness + mean_correctness) / 2
        
        report = BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            total_questions=len(questions),
            mean_faithfulness=round(mean_faithfulness, 3),
            mean_correctness=round(mean_correctness, 3),
            mean_accuracy=round(mean_accuracy, 3),
            results=results,
        )
        
        # Generate markdown report
        self._generate_report(report)
        
        self.logger.info(
            f"Evaluation complete: Accuracy={mean_accuracy:.2%}, "
            f"Faithfulness={mean_faithfulness:.2%}, Correctness={mean_correctness:.2%}"
        )
        
        return report

    def _generate_report(self, report: BenchmarkReport) -> None:
        """Generate markdown evaluation report."""
        md = f"""# 📊 RAG Evaluation Report

**Generated:** {report.timestamp}

---

## 📈 Summary Metrics

| Metric | Score |
|--------|-------|
| **Mean Accuracy** | {report.mean_accuracy:.1%} |
| **Mean Faithfulness** | {report.mean_faithfulness:.1%} |
| **Mean Correctness** | {report.mean_correctness:.1%} |
| **Total Questions** | {report.total_questions} |

---

## 🎯 Score Distribution

| Range | Faithfulness | Correctness |
|-------|--------------|-------------|
| 0.9-1.0 (Excellent) | {sum(1 for r in report.results if r.faithfulness >= 0.9)} | {sum(1 for r in report.results if r.correctness >= 0.9)} |
| 0.7-0.9 (Good) | {sum(1 for r in report.results if 0.7 <= r.faithfulness < 0.9)} | {sum(1 for r in report.results if 0.7 <= r.correctness < 0.9)} |
| 0.5-0.7 (Fair) | {sum(1 for r in report.results if 0.5 <= r.faithfulness < 0.7)} | {sum(1 for r in report.results if 0.5 <= r.correctness < 0.7)} |
| <0.5 (Poor) | {sum(1 for r in report.results if r.faithfulness < 0.5)} | {sum(1 for r in report.results if r.correctness < 0.5)} |

---

## 📝 Detailed Results

"""
        for r in report.results:
            status = "✅" if r.correctness >= 0.7 else "⚠️" if r.correctness >= 0.5 else "❌"
            md += f"""### {status} Question {r.question_id}

**Q:** {r.question}

**Correct Answer:** {r.correct_answer[:300]}{'...' if len(r.correct_answer) > 300 else ''}

**Agent Answer:** {r.agent_answer[:300]}{'...' if len(r.agent_answer) > 300 else ''}

| Metric | Score |
|--------|-------|
| Faithfulness | {r.faithfulness:.2f} |
| Correctness | {r.correctness:.2f} |
| Chunks Retrieved | {r.retrieval_count} |

**Judge Reasoning:** {r.reasoning}

---

"""
        
        REPORT_PATH.write_text(md)
        self.logger.info(f"Report saved to {REPORT_PATH}")


# ============================================================
# CLI Entrypoint
# ============================================================

async def main():
    """Run the evaluation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Evaluation Benchmark")
    parser.add_argument(
        "--generate",
        type=int,
        default=0,
        help="Generate N test questions (default: skip generation)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation on existing test set",
    )
    args = parser.parse_args()
    
    judge = RAGJudge()
    
    if args.generate > 0:
        print(f"\n{'='*60}")
        print(f"📝 GENERATING TEST SET ({args.generate} questions)")
        print(f"{'='*60}\n")
        await judge.generate_test_set(args.generate)
        print(f"\n✅ Test set saved to {TEST_SET_PATH}\n")
    
    if args.evaluate:
        print(f"\n{'='*60}")
        print("🧪 EVALUATING RAG AGENT")
        print(f"{'='*60}\n")
        report = await judge.evaluate_agent()
        print(f"\n{'='*60}")
        print("📊 EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"  Mean Accuracy:     {report.mean_accuracy:.1%}")
        print(f"  Mean Faithfulness: {report.mean_faithfulness:.1%}")
        print(f"  Mean Correctness:  {report.mean_correctness:.1%}")
        print(f"  Total Questions:   {report.total_questions}")
        print(f"\n📄 Full report: {REPORT_PATH}\n")
    
    if not args.generate and not args.evaluate:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
