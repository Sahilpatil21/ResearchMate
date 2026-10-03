"""Unit tests for Stage 7 Research Intelligence features.

Tests PaperSummarizer, PaperComparator, ResearchGapAnalyzer, and LiteratureReviewGenerator.
All LLM invocations are mocked to ensure zero real API calls during pytest.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.intelligence.comparator import PaperComparator, PaperComparisonReport
from src.intelligence.gap_analyzer import ResearchGapAnalyzer, ResearchGapReport
from src.intelligence.lit_reviewer import LiteratureReviewGenerator, LiteratureReviewReport
from src.intelligence.summarizer import PaperSummarizer, PaperSummary
from src.llm.provider import BaseLLMProvider, LLMResponse
from src.models.citation import Citation
from src.models.document import Document, DocumentMetadata, Page
from src.models.retrieval import RetrievalResult


class MockLLM(BaseLLMProvider):
    """Deterministic Mock LLM for research intelligence testing."""

    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.call_count = 0

    def is_available(self) -> bool:
        return True

    def get_model_name(self) -> str:
        return "mock-gemini"

    def get_provider_name(self) -> str:
        return "gemini"

    def generate(self, prompt: str, system_instruction: str = None, temperature: float = 0.0) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(
            text=self.response_text,
            model_name="mock-gemini",
            provider_name="gemini",
            latency_seconds=0.01,
        )


@pytest.fixture
def sample_retrieval_results() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="chk_1",
            document_id="doc_1",
            filename="paper1.pdf",
            page_number=2,
            section="Methodology",
            chunk_index=0,
            text="We introduce a residual learning algorithm F(x) + x to prevent vanishing gradients.",
            score=0.95,
            reranker_score=0.95,
        ),
        RetrievalResult(
            chunk_id="chk_2",
            document_id="doc_2",
            filename="paper2.pdf",
            page_number=4,
            section="Architecture",
            chunk_index=1,
            text="The Transformer relies solely on multi-head self-attention mechanisms.",
            score=0.91,
            reranker_score=0.91,
        ),
    ]


def test_paper_summarizer_structured_output(tmp_path, sample_retrieval_results):
    """Test PaperSummarizer generates 9 structured academic fields with citations."""
    mock_json_response = json.dumps({
        "research_problem": "Degradation and vanishing gradient problem in deep networks [1].",
        "objective": "Ease the training of substantially deeper neural networks [1].",
        "methodology": "Residual shortcut connections F(x) + x [1].",
        "dataset": "ImageNet 2012 classification dataset [1].",
        "model_architecture": "152-layer deep ResNet [1].",
        "experimental_setup": "SGD with batch size 256 and weight decay [1].",
        "main_results": "Achieved 3.57% top-5 error rate on ImageNet test set [1].",
        "limitations": "High computational requirements for training 152 layers [1].",
        "conclusion": "Residual learning simplifies optimization for ultra-deep networks [1]."
    })

    mock_llm = MockLLM(mock_json_response)
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = sample_retrieval_results

    summarizer = PaperSummarizer(retriever=mock_retriever, llm_provider=mock_llm)
    summarizer.cache_dir = tmp_path

    summary = summarizer.summarize_paper("doc_1", force_refresh=True)

    assert summary.research_problem.startswith("Degradation and vanishing gradient")
    assert "[1]" in summary.objective
    assert summary.model_architecture == "152-layer deep ResNet [1]."
    assert len(summary.citations) > 0
    assert summary.is_cached is False

    # Test Markdown rendering
    md = summary.to_markdown()
    assert "# 📄 Academic Paper Summary" in md
    assert "## 1. 🎯 Research Problem" in md
    assert "## 8. ⚠️ Limitations & Constraints" in md

    # Test caching
    cached_summary = summarizer.summarize_paper("doc_1", force_refresh=False)
    assert cached_summary.is_cached is True
    assert cached_summary.model_architecture == "152-layer deep ResNet [1]."


def test_paper_comparator_multi_paper(tmp_path, sample_retrieval_results):
    """Test PaperComparator across 2 papers with comparison table and synthesis."""
    mock_synthesis = "Paper 1 focuses on residual convolutional links [1], while Paper 2 uses self-attention [2]."
    mock_llm = MockLLM(mock_synthesis)

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = sample_retrieval_results

    summarizer = PaperSummarizer(retriever=mock_retriever, llm_provider=mock_llm)
    summarizer.cache_dir = tmp_path

    comparator = PaperComparator(retriever=mock_retriever, llm_provider=mock_llm, summarizer=summarizer)
    report = comparator.compare_papers(["doc_1", "doc_2"])

    assert len(report.document_ids) == 2
    assert "Research Problem" in report.comparison_table
    assert "Methodology" in report.comparison_table
    assert "Key Results" in report.comparison_table
    assert "Paper 1 focuses on residual" in report.comparative_synthesis

    # Test dataframe conversion
    df_rows = report.to_dataframe_dict()
    assert len(df_rows) == 10
    assert df_rows[0]["Dimension"] == "Research Problem"


def test_research_gap_analyzer_grounding(sample_retrieval_results):
    """Test ResearchGapAnalyzer extracts evidence-grounded gaps with citation IDs."""
    mock_gaps_json = json.dumps({
        "executive_summary": "Identified major limitations in training cost and dataset diversity.",
        "gaps": [
            {
                "gap_statement": "Potential research gap: Lack of cross-domain evaluation on noisy real-world imagery.",
                "category": "Dataset & Generalization",
                "supporting_evidence": "The model was trained strictly on ImageNet with clean standardized images [1].",
                "source_paper": "paper1.pdf",
                "page_number": 2,
                "section": "Limitations",
                "citation_id": "[1]",
                "future_direction_suggestion": "Evaluate robustness under heavy occlusion and sensor noise."
            }
        ]
    })

    mock_llm = MockLLM(mock_gaps_json)
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = sample_retrieval_results

    analyzer = ResearchGapAnalyzer(retriever=mock_retriever, llm_provider=mock_llm)
    report = analyzer.analyze_gaps(["doc_1"])

    assert len(report.gaps) == 1
    gap = report.gaps[0]
    assert "Potential research gap:" in gap.gap_statement
    assert gap.page_number == 2
    assert gap.citation_id == "[1]"
    assert "Evaluate robustness" in gap.future_direction_suggestion

    md = report.to_markdown()
    assert "# 🔬 Evidence-Grounded Research Gap Analysis" in md
    assert "### Gap #1:" in md


def test_literature_review_generator_10_sections(sample_retrieval_results):
    """Test LiteratureReviewGenerator produces all 10 required academic sections."""
    mock_lit_json = json.dumps({
        "introduction": "This review analyzes deep learning architectures [1].",
        "research_area_overview": "Advances in residual representations and transformers [1, 2].",
        "existing_approaches": "Prior convolutional approaches suffered from optimization degradation [1].",
        "methodology_comparison": "ResNet adds identity shortcuts [1], while Transformer removes recurrence [2].",
        "dataset_trends": "ImageNet and WMT translation corpora serve as standard benchmarks [1, 2].",
        "key_findings": "Substantial accuracy gains achieved over baseline models [1, 2].",
        "limitations": "Computational overhead during multi-GPU pre-training [1].",
        "potential_research_gaps": "Potential research gap: Joint multimodal vision-language pre-training.",
        "future_research_directions": "Exploring parameter-efficient fine-tuning and sparse attention."
    })

    mock_llm = MockLLM(mock_lit_json)
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = sample_retrieval_results

    generator = LiteratureReviewGenerator(retriever=mock_retriever, llm_provider=mock_llm)
    report = generator.generate_review(["doc_1", "doc_2"], topic_focus="Deep Learning Architectures")

    assert report.topic == "Deep Learning Architectures"
    assert "This review analyzes" in report.introduction
    assert "ResNet adds identity shortcuts" in report.methodology_comparison
    assert "Joint multimodal" in report.potential_research_gaps
    assert len(report.citations) > 0

    md = report.to_markdown()
    assert "## 1. Introduction" in md
    assert "## 4. Methodology Comparison" in md
    assert "## 10. References" in md
