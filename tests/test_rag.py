"""Unit tests for Stage 6 RAG Orchestration and Grounded Q&A.

Tests context construction, strict prompt formatting, citation validation,
insufficient evidence handling, multi-paper synthesis, conversational resolution,
and end-to-end RAG workflow with mocked LLM providers.
NO REAL API CALLS are made during test execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.citations.citation_engine import CitationEngine
from src.llm.provider import BaseLLMProvider, LLMResponse
from src.models.citation import Citation
from src.models.retrieval import RetrievalResult
from src.rag.answer_validator import AnswerValidator, ValidationResult
from src.rag.context_builder import ContextBuilder
from src.rag.prompt_templates import (
    ACADEMIC_RAG_SYSTEM_PROMPT,
    build_rag_user_prompt,
    build_reference_resolution_prompt,
)
from src.rag.rag_pipeline import RAGPipeline, RAGResponse


class MockLLMProvider(BaseLLMProvider):
    """Deterministic Mock LLM Provider for unit testing."""

    def __init__(self, mock_answer: str = "Mock answer [1].", is_avail: bool = True) -> None:
        self.mock_answer = mock_answer
        self.is_avail = is_avail
        self.call_count = 0
        self.last_prompt = None

    def is_available(self) -> bool:
        return self.is_avail

    def get_model_name(self) -> str:
        return "mock-model"

    def get_provider_name(self) -> str:
        return "mock-provider"

    def generate(
        self,
        prompt: str,
        system_instruction: str = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.call_count += 1
        self.last_prompt = prompt
        return LLMResponse(
            text=self.mock_answer,
            model_name="mock-model",
            provider_name="mock-provider",
            latency_seconds=0.01,
        )


@pytest.fixture
def sample_citations() -> list[Citation]:
    """Create sample Citation objects from 2 different papers."""
    c1 = Citation(
        citation_id="[1]",
        citation_index=1,
        document_id="doc_resnet",
        filename="resnet.pdf",
        paper_title="Deep Residual Learning for Image Recognition",
        page_number=4,
        section="2. Methodology",
        chunk_id="chk_res_1",
        chunk_index=1,
        source_text="We introduce residual shortcut connections F(x) + x to ease optimization.",
        source_snippet="We introduce residual shortcut connections F(x) + x to ease optimization.",
        retrieval_rank=1,
        reranker_score=0.95,
    )
    c2 = Citation(
        citation_id="[2]",
        citation_index=2,
        document_id="doc_attention",
        filename="attention.pdf",
        paper_title="Attention Is All You Need",
        page_number=3,
        section="3. Architecture",
        chunk_id="chk_att_1",
        chunk_index=2,
        source_text="The Transformer relies entirely on self-attention mechanisms without recurrent layers.",
        source_snippet="The Transformer relies entirely on self-attention mechanisms without recurrent layers.",
        retrieval_rank=2,
        reranker_score=0.88,
    )
    return [c1, c2]


def test_context_construction(sample_citations):
    """Test ContextBuilder output formatting and source attribution tags."""
    builder = ContextBuilder()
    context = builder.build_context_from_citations(sample_citations)

    assert "[Source 1]" in context
    assert "Paper: Deep Residual Learning for Image Recognition" in context
    assert "Filename: resnet.pdf" in context
    assert "Page: 4" in context
    assert "Section: 2. Methodology" in context
    assert "Reranker Score: 0.9500" in context
    assert "We introduce residual shortcut connections" in context

    assert "[Source 2]" in context
    assert "Paper: Attention Is All You Need" in context
    assert "The Transformer relies entirely on self-attention" in context


def test_context_truncation_budget(sample_citations):
    """Test ContextBuilder character budget enforcement."""
    builder = ContextBuilder(max_context_chars=250)
    context = builder.build_context_from_citations(sample_citations)

    # Should include Source 1 and respect safety cap
    assert "[Source 1]" in context
    assert len(context) <= 400


def test_strict_rag_prompt_construction(sample_citations):
    """Test strict prompt generation with context, query, and chat history."""
    builder = ContextBuilder()
    context = builder.build_context_from_citations(sample_citations)
    query = "How do ResNet and Transformer architectures differ?"
    history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "How can I help with your research?"},
    ]

    prompt = build_rag_user_prompt(query=query, context=context, chat_history=history)

    assert "### RETRIEVED RESEARCH PAPER CONTEXT ###" in prompt
    assert "[Source 1]" in prompt
    assert "[Source 2]" in prompt
    assert "### RECENT CONVERSATION CONTEXT" in prompt
    assert "### RESEARCH QUESTION ###" in prompt
    assert query in prompt
    assert "inline citations [1], [2]" in prompt


def test_answer_validator_valid_citations(sample_citations):
    """Test validator when answer has valid, grounded citations."""
    answer = "ResNet introduces residual shortcuts [1], while Transformer uses self-attention [2]."
    result = AnswerValidator.validate_answer(answer, sample_citations)

    assert result.is_valid is True
    assert result.has_citations is True
    assert result.cited_indices == [1, 2]
    assert result.valid_citations == [1, 2]
    assert result.invalid_citations == []
    assert result.grounding_score == 1.0
    assert result.is_insufficient_evidence is False


def test_answer_validator_hallucinated_citations(sample_citations):
    """Test validator detection of hallucinated citation indices."""
    # Context only has sources 1 and 2
    answer = "The model was trained on 1000 GPUs [5] using residual links [1]."
    result = AnswerValidator.validate_answer(answer, sample_citations)

    assert result.is_valid is False
    assert 5 in result.invalid_citations
    assert 1 in result.valid_citations
    assert any("Hallucinated citation IDs detected: [5]" in w for w in result.warnings)


def test_answer_validator_missing_citations(sample_citations):
    """Test validator flagging answers with no inline citations when context was provided."""
    answer = "ResNet is a deep architecture that solves vanishing gradients."
    result = AnswerValidator.validate_answer(answer, sample_citations)

    assert result.is_valid is False
    assert result.has_citations is False
    assert any("contains no inline citations" in w for w in result.warnings)


def test_answer_validator_insufficient_evidence():
    """Test validator recognizing valid insufficient-evidence answers."""
    answer = "The provided papers do not contain enough information to answer this question."
    result = AnswerValidator.validate_answer(answer, citations=[])

    assert result.is_valid is True
    assert result.is_insufficient_evidence is True
    assert result.grounding_score == 1.0


def test_rag_pipeline_empty_query():
    """Test RAG pipeline handling of empty/whitespace query."""
    pipeline = RAGPipeline(llm_provider=MockLLMProvider())
    response = pipeline.answer_question(query="   ")

    assert response.is_insufficient_evidence is True
    assert "Please enter a valid research question." in response.answer
    assert response.latency_seconds == 0.0


def test_rag_pipeline_no_retrieved_chunks(tmp_path):
    """Test RAG pipeline hallucination prevention when retrieval returns 0 chunks."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    mock_llm = MockLLMProvider()
    pipeline = RAGPipeline(
        retriever=mock_retriever,
        llm_provider=mock_llm,
        log_dir=tmp_path,
    )

    response = pipeline.answer_question(query="What is quantum gravity in ResNet?")

    # Must NOT call the LLM when no relevant evidence is found
    assert mock_llm.call_count == 0
    assert response.is_insufficient_evidence is True
    assert "The provided papers do not contain enough information to answer this question." in response.answer
    assert response.citations == []


def test_rag_pipeline_end_to_end_mocked(tmp_path, sample_citations):
    """Test complete end-to-end RAG pipeline execution with mocked retrieval and LLM."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        RetrievalResult(
            chunk_id="chk_res_1",
            document_id="doc_resnet",
            filename="resnet.pdf",
            page_number=4,
            section="2. Methodology",
            chunk_index=1,
            text="We introduce residual shortcut connections F(x) + x to ease optimization.",
            score=0.95,
            reranker_score=0.95,
        ),
        RetrievalResult(
            chunk_id="chk_att_1",
            document_id="doc_attention",
            filename="attention.pdf",
            page_number=3,
            section="3. Architecture",
            chunk_index=2,
            text="The Transformer relies entirely on self-attention mechanisms without recurrent layers.",
            score=0.88,
            reranker_score=0.88,
        ),
    ]

    mock_llm = MockLLMProvider(
        mock_answer="ResNet uses shortcut connections [1], whereas Transformer uses self-attention [2]."
    )

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        llm_provider=mock_llm,
        log_dir=tmp_path,
    )

    response = pipeline.answer_question(
        query="Compare the architectures of ResNet and Transformer.",
        retrieval_mode="hybrid",
        candidate_k=20,
        top_k=5,
        rerank=True,
    )

    # Verifications
    assert mock_llm.call_count == 1
    assert "ResNet uses shortcut connections [1]" in response.answer
    assert len(response.citations) == 2
    assert response.citations[0].filename == "resnet.pdf"
    assert response.citations[1].filename == "attention.pdf"
    assert response.validation.is_valid is True
    assert response.validation.valid_citations == [1, 2]
    assert response.provider == "mock-provider"
    assert response.latency_seconds >= 0.0

    # Verify query logging
    log_file = tmp_path / "rag_queries.jsonl"
    assert log_file.exists()
    log_lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(log_lines) == 1
    log_data = json.loads(log_lines[0])
    assert log_data["query"] == "Compare the architectures of ResNet and Transformer."
    assert log_data["validation_valid"] is True
    assert log_data["reranking_enabled"] is True


def test_conversational_reference_resolution():
    """Test conversational pronoun resolution helper."""
    mock_llm = MockLLMProvider(mock_answer="What is the architecture of Transformer in Attention Is All You Need?")
    pipeline = RAGPipeline(llm_provider=mock_llm)

    history = [
        {"role": "user", "content": "Tell me about ResNet and Attention Is All You Need."},
        {"role": "assistant", "content": "ResNet uses residual links [1] and Attention introduces Transformer [2]."},
    ]

    resolved = pipeline.resolve_conversational_query(
        query="What about the second paper?",
        chat_history=history,
    )

    assert "Transformer" in resolved


def test_rag_pipeline_malformed_llm_error(tmp_path):
    """Test RAG pipeline handling when LLM returns an error."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        RetrievalResult(
            chunk_id="chk_1",
            document_id="doc_1",
            filename="paper.pdf",
            page_number=1,
            section="Intro",
            chunk_index=0,
            text="Some text here.",
            score=0.9,
        )
    ]

    failing_llm = MagicMock(spec=BaseLLMProvider)
    failing_llm.get_provider_name.return_value = "gemini"
    failing_llm.get_model_name.return_value = "gemini-2.5-flash"
    failing_llm.generate.return_value = LLMResponse(
        text="",
        model_name="gemini-2.5-flash",
        provider_name="gemini",
        error="Quota exceeded for Gemini API",
    )

    pipeline = RAGPipeline(
        retriever=mock_retriever,
        llm_provider=failing_llm,
        log_dir=tmp_path,
    )

    response = pipeline.answer_question(query="Summarize findings.")

    assert response.error == "Quota exceeded for Gemini API"
    assert "⚠️ Quota exceeded for Gemini API" in response.answer
    assert response.validation.is_valid is False
