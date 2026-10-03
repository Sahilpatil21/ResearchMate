"""Unit tests for Stage 5 Citation Attribution and Source Snippet Extraction."""

import pytest

from src.citations.citation_engine import CitationEngine
from src.models.citation import Citation
from src.models.document import Document, DocumentMetadata, Page
from src.models.retrieval import RetrievalResult


@pytest.fixture
def sample_retrieval_results() -> list[RetrievalResult]:
    """Create sample retrieval results for citation testing."""
    r1 = RetrievalResult(
        chunk_id="chk_resnet_0001",
        document_id="doc_resnet",
        filename="resnet.pdf",
        page_number=4,
        section="2. Methodology",
        chunk_index=1,
        text=(
            "We present a residual learning framework to ease the training of networks. "
            "Formally, we let stacked nonlinear layers fit a residual mapping F(x) := H(x) - x. "
            "The original mapping is recast into F(x) + x, realized via identity shortcut connections."
        ),
        score=0.92,
        reranker_score=0.92,
        reranker_rank=1,
        dense_score=0.88,
        dense_rank=1,
        sparse_score=14.2,
        sparse_rank=1,
        rrf_score=0.0163,
        retrieval_method="reranked (ms-marco-MiniLM-L-6-v2)",
    )
    r2 = RetrievalResult(
        chunk_id="chk_transformer_0002",
        document_id="doc_transformer",
        filename="attention.pdf",
        page_number=7,
        section="3. Results",
        chunk_index=3,
        text=(
            "On the WMT 2014 English-to-German translation task, the Transformer big model achieves 28.4 BLEU. "
            "This establishes a new state-of-the-art benchmark, improving over previous models by over 2.0 BLEU."
        ),
        score=0.81,
        reranker_score=0.81,
        reranker_rank=2,
        dense_score=0.79,
        dense_rank=2,
        sparse_score=11.5,
        sparse_rank=2,
        rrf_score=0.0159,
        retrieval_method="reranked (ms-marco-MiniLM-L-6-v2)",
    )
    return [r1, r2]


def test_citation_creation(sample_retrieval_results):
    """Test creating structured Citation objects from retrieval results."""
    citations = CitationEngine.create_citations(
        results=sample_retrieval_results,
        query="residual learning shortcut connections",
    )

    assert len(citations) == 2

    c1 = citations[0]
    assert c1.citation_id == "[1]"
    assert c1.citation_index == 1
    assert c1.filename == "resnet.pdf"
    assert c1.page_number == 4
    assert c1.section == "2. Methodology"
    assert c1.chunk_id == "chk_resnet_0001"
    assert c1.retrieval_rank == 1
    assert c1.reranker_score == 0.92

    c2 = citations[1]
    assert c2.citation_id == "[2]"
    assert c2.citation_index == 2
    assert c2.filename == "attention.pdf"
    assert c2.page_number == 7
    assert c2.section == "3. Results"


def test_source_snippet_extraction():
    """Test extracting most relevant sentence snippet while preserving original text."""
    full_text = (
        "Convolutional networks have made great strides in vision. "
        "We introduce residual shortcut connections F(x) + x to ease optimization. "
        "Our 152-layer model achieves 3.57% error on ImageNet."
    )
    query = "shortcut connections F(x) + x"

    snippet = CitationEngine.extract_source_snippet(query=query, text=full_text)

    # Must extract the relevant sentence containing the query terms
    assert "shortcut connections" in snippet
    assert "F(x) + x" in snippet


def test_source_text_integrity(sample_retrieval_results):
    """Verify that original source text is never altered during citation processing."""
    original_text_1 = sample_retrieval_results[0].text
    original_text_2 = sample_retrieval_results[1].text

    citations = CitationEngine.create_citations(
        results=sample_retrieval_results,
        query="WMT 2014 BLEU translation",
    )

    assert citations[0].source_text == original_text_1
    assert citations[1].source_text == original_text_2


def test_formatted_citation_string(sample_retrieval_results):
    """Test human-readable citation formatting."""
    citations = CitationEngine.create_citations(
        results=sample_retrieval_results,
        query="residual mapping",
    )

    formatted_1 = citations[0].to_formatted_citation()
    assert "[1] resnet.pdf — Page 4 — 2. Methodology" in formatted_1

    formatted_2 = citations[1].to_formatted_citation()
    assert "[2] attention.pdf — Page 7 — 3. Results" in formatted_2


def test_citation_document_map_resolution(sample_retrieval_results):
    """Test resolving paper title via document map."""
    doc_map = {
        "doc_resnet": Document(
            metadata=DocumentMetadata(
                document_id="doc_resnet",
                filename="resnet.pdf",
                original_filename="resnet.pdf",
                title="Deep Residual Learning for Image Recognition",
                page_count=4,
                total_pages_in_pdf=4,
            ),
            pages=[],
        )
    }

    citations = CitationEngine.create_citations(
        results=sample_retrieval_results,
        document_map=doc_map,
    )

    assert citations[0].paper_title == "Deep Residual Learning for Image Recognition"


def test_markdown_citation_formatting(sample_retrieval_results):
    """Test Markdown bibliography block output."""
    citations = CitationEngine.create_citations(
        results=sample_retrieval_results,
        query="residual learning",
    )
    md_output = CitationEngine.format_citations_markdown(citations)

    assert "### 📚 Grounded Citations" in md_output
    assert "**[1]**" in md_output
    assert "**[2]**" in md_output
    assert "Page **4**" in md_output
    assert "Page **7**" in md_output


def test_citation_serialization(sample_retrieval_results):
    """Test round-trip dict and JSON serialization of Citation model."""
    citations = CitationEngine.create_citations(results=sample_retrieval_results)
    cit = citations[0]

    json_data = cit.to_json()
    reconstructed = Citation.from_dict(cit.to_dict())

    assert reconstructed.citation_id == cit.citation_id
    assert reconstructed.filename == cit.filename
    assert reconstructed.page_number == cit.page_number
    assert reconstructed.source_text == cit.source_text
