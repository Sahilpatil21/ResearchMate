"""Generation quality evaluation and human rating framework for ResearchMate.

Provides automated groundedness metrics and 1–5 human evaluation rating scale:
- Automated Metrics:
  - Faithfulness / Groundedness Score
  - Citation Validity Rate
  - Context Term Overlap
  - Insufficient Evidence Accuracy
- Human Expert Rating Scale (1 to 5):
  - Correctness (1-5)
  - Relevance (1-5)
  - Completeness (1-5)
  - Citation Quality (1-5)
  - Groundedness (1-5)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.models.citation import Citation
from src.rag.answer_validator import AnswerValidator, ValidationResult
from src.rag.rag_pipeline import RAGResponse
from src.retrieval.bm25_retriever import tokenize_academic_text


class HumanEvaluationRating(BaseModel):
    """Human expert evaluation feedback for an individual RAG response."""

    evaluator_id: str = Field(default="user", description="Identifier of the evaluator")
    question_id: Optional[str] = Field(default=None, description="Benchmark question ID if applicable")
    query: str = Field(..., description="The evaluated question")
    correctness: int = Field(..., ge=1, le=5, description="Factual correctness against research papers (1-5)")
    relevance: int = Field(..., ge=1, le=5, description="Relevance of the answer to the prompt (1-5)")
    completeness: int = Field(..., ge=1, le=5, description="Thoroughness and completeness (1-5)")
    citation_quality: int = Field(..., ge=1, le=5, description="Accuracy and precision of citations (1-5)")
    groundedness: int = Field(..., ge=1, le=5, description="Freedom from hallucinated facts (1-5)")
    feedback_notes: Optional[str] = Field(default="", description="Optional qualitative reviewer notes")

    @property
    def average_score(self) -> float:
        """Compute composite 1–5 score across all 5 dimensions."""
        scores = [
            self.correctness,
            self.relevance,
            self.completeness,
            self.citation_quality,
            self.groundedness,
        ]
        return round(sum(scores) / len(scores), 2)


class AutomatedGenerationMetrics(BaseModel):
    """Automated evaluation metrics for an individual RAG answer."""

    faithfulness_score: float = Field(..., description="Ratio of verified grounded citations to total citations (0.0 to 1.0)")
    citation_validity_rate: float = Field(..., description="Ratio of valid citation indices to total citation indices (0.0 to 1.0)")
    context_term_overlap: float = Field(..., description="Jaccard word overlap between answer and retrieved context (0.0 to 1.0)")
    has_hallucinated_citations: bool = Field(..., description="True if any citation ID was out-of-range")
    is_insufficient_evidence_declared: bool = Field(..., description="True if model indicated insufficient evidence")
    word_count: int = Field(..., description="Answer length in words")


class GenerationEvaluator:
    """Evaluates generated RAG responses automatically and manages human rating records."""

    @staticmethod
    def evaluate_response(response: RAGResponse) -> AutomatedGenerationMetrics:
        """Compute automated generation metrics for a RAG response.

        Args:
            response: Completed RAGResponse object.

        Returns:
            AutomatedGenerationMetrics instance.
        """
        val: ValidationResult = response.validation

        # 1. Faithfulness Score & Citation Validity
        faithfulness = val.grounding_score
        has_hallucinated = len(val.invalid_citations) > 0
        citation_validity = (
            round(len(val.valid_citations) / max(1, len(val.cited_indices)), 3)
            if val.cited_indices
            else (1.0 if val.is_insufficient_evidence else 0.0)
        )

        # 2. Context Term Overlap (Jaccard similarity between answer and source chunks)
        answer_tokens = set(tokenize_academic_text(response.answer, remove_stopwords=True))
        all_context_text = " ".join(cit.source_text for cit in response.citations)
        context_tokens = set(tokenize_academic_text(all_context_text, remove_stopwords=True))

        if answer_tokens and context_tokens:
            intersection = answer_tokens.intersection(context_tokens)
            union = answer_tokens.union(context_tokens)
            context_overlap = round(len(intersection) / max(1, len(union)), 4)
        else:
            context_overlap = 0.0

        words = len(response.answer.split())

        return AutomatedGenerationMetrics(
            faithfulness_score=faithfulness,
            citation_validity_rate=citation_validity,
            context_term_overlap=context_overlap,
            has_hallucinated_citations=has_hallucinated,
            is_insufficient_evidence_declared=response.is_insufficient_evidence,
            word_count=words,
        )
