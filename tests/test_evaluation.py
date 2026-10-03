"""Unit tests for Stage 7 Evaluation & Benchmarking framework.

Tests EvaluationDataset, RetrievalEvaluator, GenerationEvaluator, and ExperimentLogger.
All tests are 100% mocked and make zero real API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.evaluation.dataset import DEFAULT_EVALUATION_QUESTIONS, EvaluationDataset, EvaluationQuestion
from src.evaluation.experiment_logger import ExperimentLogger
from src.evaluation.generation_evaluator import AutomatedGenerationMetrics, GenerationEvaluator, HumanEvaluationRating
from src.evaluation.retrieval_evaluator import RetrievalEvaluationReport, RetrievalEvaluator
from src.models.citation import Citation
from src.models.retrieval import RetrievalResult
from src.rag.answer_validator import ValidationResult
from src.rag.rag_pipeline import RAGResponse


def test_evaluation_dataset_initialization(tmp_path):
    """Test EvaluationDataset loads 20 default benchmark questions."""
    db_file = tmp_path / "test_eval_dataset.json"
    dataset = EvaluationDataset(dataset_path=db_file)

    assert len(dataset.questions) >= 20
    assert db_file.exists()

    # Test category filtering
    factual_qs = dataset.get_by_category("factual")
    assert len(factual_qs) >= 3
    assert all(q.category == "factual" for q in factual_qs)

    methodology_qs = dataset.get_by_category("methodology")
    assert len(methodology_qs) >= 3

    multi_paper_qs = dataset.get_by_category("multi-paper")
    assert len(multi_paper_qs) >= 3


def test_retrieval_evaluator_metrics(tmp_path):
    """Test RetrievalEvaluator calculates Precision@K, Recall@K, HitRate@K, and MRR."""
    dataset_file = tmp_path / "eval.json"
    dataset = EvaluationDataset(dataset_path=dataset_file)

    q = EvaluationQuestion(
        question_id="test_01",
        question="What is the architecture of ResNet?",
        category="methodology",
        expected_answer="Residual shortcut connections",
        relevant_documents=["resnet.pdf"],
        relevant_pages=[2, 4],
        relevant_keywords=["shortcut connections", "residual mapping"],
    )

    mock_retriever = MagicMock()
    # Return 2 results: Rank 1 matches, Rank 2 does not match
    r1 = RetrievalResult(
        chunk_id="chk_1",
        document_id="doc_res",
        filename="resnet.pdf",
        page_number=4,
        section="Methodology",
        chunk_index=0,
        text="We present residual shortcut connections F(x) + x.",
        score=0.9,
    )
    r2 = RetrievalResult(
        chunk_id="chk_2",
        document_id="doc_other",
        filename="other.pdf",
        page_number=1,
        section="Intro",
        chunk_index=0,
        text="Unrelated text.",
        score=0.5,
    )
    mock_retriever.retrieve.return_value = [r1, r2]

    evaluator = RetrievalEvaluator(retriever=mock_retriever, dataset=dataset)

    # Relevance check
    assert evaluator.is_chunk_relevant(r1, q) is True
    assert evaluator.is_chunk_relevant(r2, q) is False

    # Single query evaluation
    query_eval = evaluator.evaluate_query(q, method="hybrid_rerank", top_k=2)
    assert query_eval.precision == 0.5  # 1 out of 2 is relevant
    assert query_eval.hit_rate == 1.0  # Hit rate is 1.0 since at least 1 match found
    assert query_eval.reciprocal_rank == 1.0  # First match at rank 1


def test_retrieval_benchmark_summary(tmp_path):
    """Test running full benchmark across methods."""
    dataset_file = tmp_path / "eval.json"
    dataset = EvaluationDataset(dataset_path=dataset_file)
    # Use 2 questions for quick test
    dataset.questions = dataset.questions[:2]

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        RetrievalResult(
            chunk_id="chk_1",
            document_id="doc_1",
            filename="IJRTI2304061.pdf",
            page_number=1,
            section="Intro",
            chunk_index=0,
            text="Artificial intelligence simulation of human intelligence.",
            score=0.88,
        )
    ]

    evaluator = RetrievalEvaluator(retriever=mock_retriever, dataset=dataset)
    report = evaluator.run_full_benchmark(
        top_k=5,
        methods=["dense", "sparse", "hybrid", "hybrid_rerank"],
    )

    assert "dense" in report.method_summaries
    assert "hybrid_rerank" in report.method_summaries
    assert report.method_summaries["dense"].total_queries_evaluated == 2


def test_generation_evaluator_metrics():
    """Test GenerationEvaluator computes automated faithfulness and overlap metrics."""
    c1 = Citation(
        citation_id="[1]",
        citation_index=1,
        document_id="doc_1",
        filename="paper.pdf",
        paper_title="Paper Title",
        page_number=1,
        section="Intro",
        chunk_id="chk_1",
        chunk_index=0,
        source_text="Residual learning overcomes the degradation problem in deep networks.",
        retrieval_rank=1,
    )

    resp = RAGResponse(
        query="What does residual learning solve?",
        answer="Residual learning overcomes the degradation problem in deep networks [1].",
        citations=[c1],
        retrieval_results=[],
        validation=ValidationResult(
            is_valid=True,
            has_citations=True,
            cited_indices=[1],
            valid_citations=[1],
            invalid_citations=[],
            grounding_score=1.0,
        ),
        provider="gemini",
        model="gemini-2.5-flash",
        latency_seconds=0.45,
    )

    metrics = GenerationEvaluator.evaluate_response(resp)
    assert metrics.faithfulness_score == 1.0
    assert metrics.citation_validity_rate == 1.0
    assert metrics.has_hallucinated_citations is False
    assert metrics.context_term_overlap > 0.3
    assert metrics.word_count > 5


def test_human_evaluation_rating_model():
    """Test HumanEvaluationRating model validation and average score."""
    rating = HumanEvaluationRating(
        evaluator_id="researcher_1",
        question_id="eval_001",
        query="What is ResNet?",
        correctness=5,
        relevance=5,
        completeness=4,
        citation_quality=5,
        groundedness=5,
        feedback_notes="Accurate with exact page citations.",
    )

    assert rating.average_score == 4.8
    assert rating.evaluator_id == "researcher_1"


def test_experiment_logger_persistence(tmp_path):
    """Test ExperimentLogger persists runs and human ratings without leaking keys."""
    logger = ExperimentLogger(log_dir=tmp_path)

    c1 = Citation(
        citation_id="[1]",
        citation_index=1,
        document_id="doc_1",
        filename="paper.pdf",
        paper_title="Paper Title",
        page_number=1,
        section="Intro",
        chunk_id="chk_1",
        chunk_index=0,
        source_text="Sample text content.",
        retrieval_rank=1,
    )

    resp = RAGResponse(
        query="Sample query",
        answer="Sample answer [1].",
        citations=[c1],
        retrieval_results=[],
        validation=ValidationResult(
            is_valid=True,
            has_citations=True,
            cited_indices=[1],
            valid_citations=[1],
            invalid_citations=[],
            grounding_score=1.0,
        ),
        provider="gemini",
        model="gemini-2.5-flash",
        latency_seconds=0.3,
    )

    # Log experiment
    record = logger.log_experiment_run(response=resp, notes="Test experiment")
    assert record["query"] == "Sample query"
    assert record["faithfulness_score"] == 1.0

    # Verify disk persistence
    recent = logger.load_recent_experiments()
    assert len(recent) == 1
    assert recent[0]["query"] == "Sample query"

    # Log human rating
    rating = HumanEvaluationRating(
        evaluator_id="user_test",
        query="Sample query",
        correctness=5,
        relevance=5,
        completeness=5,
        citation_quality=5,
        groundedness=5,
    )
    logger.log_human_rating(rating)

    ratings_loaded = logger.load_human_ratings()
    assert len(ratings_loaded) == 1
    assert ratings_loaded[0]["evaluator_id"] == "user_test"
