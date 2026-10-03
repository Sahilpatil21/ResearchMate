"""Unit tests for Step 2 Text Chunking and Section Detection."""

import pytest
from src.chunking.text_chunker import (
    TextChunker,
    chunk_document,
    estimate_token_count,
    is_section_header,
    split_into_sentences,
)
from src.models.chunk import Chunk, ChunkMetadata
from src.models.document import Document, DocumentMetadata, Page


@pytest.fixture
def sample_academic_document() -> Document:
    """Create a structured Document with typical research paper sections."""
    pages = [
        Page(
            page_number=1,
            text=(
                "Deep Residual Learning for Vision\n\n"
                "Abstract\n"
                "Deeper neural networks are more difficult to train. We present a residual learning "
                "framework to ease the training of networks that are substantially deeper than those "
                "used previously. We explicitly reformulate the layers as learning residual functions "
                "with reference to the layer inputs, instead of learning unreferenced functions.\n\n"
                "1. Introduction\n"
                "Deep convolutional neural networks have led to a series of breakthroughs for image "
                "classification. Deep networks naturally integrate low/mid/high-level features and "
                "classifiers in an end-to-end multilayer fashion. The depth of representations is of "
                "central importance for many visual recognition tasks."
            ),
        ),
        Page(
            page_number=2,
            text=(
                "2. Proposed Methodology\n"
                "In this section, we formulate the residual building block. Formally, considering H(x) "
                "as an underlying mapping to be fit by a few stacked layers, we let these layers "
                "approximate a residual function F(x) := H(x) - x. The original mapping is recast "
                "into F(x) + x.\n\n"
                "3. Experimental Results\n"
                "We evaluate the residual networks on the ImageNet 2012 classification dataset. "
                "Our 152-layer ResNet achieves a 3.57% top-5 error rate on the ImageNet test set, "
                "winning 1st place in the ILSVRC 2015 competition."
            ),
        ),
    ]

    metadata = DocumentMetadata(
        document_id="doc_resnet_123",
        filename="resnet_2015.pdf",
        original_filename="resnet_2015.pdf",
        title="Deep Residual Learning for Vision",
        authors=["Kaiming He", "Xiangyu Zhang"],
        page_count=2,
        total_pages_in_pdf=2,
        has_extractable_text=True,
    )

    return Document(metadata=metadata, pages=pages)


def test_section_header_detection():
    """Test detection of common academic section headers."""
    assert is_section_header("Abstract") is not None
    assert is_section_header("ABSTRACT") is not None
    assert is_section_header("1. Introduction") is not None
    assert is_section_header("1.1 Background and Related Work") is not None
    assert is_section_header("2. Proposed Methodology:") is not None
    assert is_section_header("III. EXPERIMENTS AND RESULTS") is not None
    assert is_section_header("Conclusion and Future Work") is not None
    assert is_section_header("References") is not None

    # Non-headers
    assert is_section_header("This is just a normal sentence discussing related work.") is None
    assert is_section_header("In table 1 we show the accuracy of our model.") is None


def test_sentence_boundary_splitting():
    """Test splitting sentences while preserving abbreviations."""
    text = "ResNet outperforms VGG (e.g., on ImageNet). See He et al. for further details! Is depth helpful? Yes."
    sentences = split_into_sentences(text)

    assert len(sentences) == 4
    assert "e.g., on ImageNet" in sentences[0]
    assert "He et al." in sentences[1]
    assert sentences[2] == "Is depth helpful?"
    assert sentences[3] == "Yes."


def test_chunk_document_basic(sample_academic_document):
    """Test chunking a document into cohesive chunks with preserved metadata."""
    chunker = TextChunker(chunk_size=400, chunk_overlap=80)
    chunks = chunker.chunk_document(sample_academic_document)

    assert len(chunks) >= 3
    for idx, chk in enumerate(chunks):
        assert isinstance(chk, Chunk)
        assert chk.document_id == "doc_resnet_123"
        assert chk.filename == "resnet_2015.pdf"
        assert chk.chunk_index == idx
        assert chk.chunk_id == f"chk_doc_resnet_123_{idx:04d}"
        assert chk.page_number in [1, 2]
        assert len(chk.text) > 0
        assert chk.metadata.token_count is not None
        assert chk.metadata.token_count > 0


def test_section_tagging_in_chunks(sample_academic_document):
    """Test that chunks correctly inherit academic section tags."""
    chunks = chunk_document(sample_academic_document, chunk_size=300, chunk_overlap=50)

    sections_found = {chk.section for chk in chunks}
    # Should identify Abstract, Introduction, Methodology, or Experiments
    assert any("Abstract" in s for s in sections_found)
    assert any("Introduction" in s or "Methodology" in s or "Results" in s for s in sections_found)


def test_chunk_overlap():
    """Test that consecutive chunks share overlap text."""
    doc = Document(
        metadata=DocumentMetadata(
            document_id="doc_test_overlap",
            filename="overlap.pdf",
            original_filename="overlap.pdf",
            title="Overlap Test",
        ),
        pages=[
            Page(
                page_number=1,
                text=(
                    "Sentence one is the starting statement. "
                    "Sentence two adds supplementary information. "
                    "Sentence three explains the core mechanism. "
                    "Sentence four discusses the practical implications."
                ),
            )
        ],
    )

    chunker = TextChunker(chunk_size=120, chunk_overlap=50, min_chunk_size=30)
    chunks = chunker.chunk_document(doc)

    if len(chunks) >= 2:
        # Check that words from the end of chunk 0 exist in chunk 1
        c0_words = set(chunks[0].text.split())
        c1_words = set(chunks[1].text.split())
        common_words = c0_words.intersection(c1_words)
        assert len(common_words) > 0


def test_token_counter():
    """Test token estimation."""
    text = "Retrieval-Augmented Generation (RAG) combines dense retrieval with generative language models."
    tokens = estimate_token_count(text)
    assert tokens > 5
    assert tokens < 50
    assert estimate_token_count("") == 0


def test_chunk_serialization():
    """Test round-trip JSON serialization of Chunk."""
    metadata = ChunkMetadata(
        chunk_id="chk_abc_0000",
        document_id="doc_abc",
        filename="paper.pdf",
        page_number=3,
        section="3. Methodology",
        chunk_index=0,
        char_count=150,
        token_count=35,
    )
    chunk = Chunk(text="Sample text content for chunk.", metadata=metadata)

    json_str = chunk.to_json()
    reconstructed = Chunk.from_json(json_str)

    assert reconstructed.chunk_id == "chk_abc_0000"
    assert reconstructed.document_id == "doc_abc"
    assert reconstructed.page_number == 3
    assert reconstructed.section == "3. Methodology"
    assert reconstructed.text == "Sample text content for chunk."
    assert reconstructed.metadata.token_count == 35


def test_empty_document_chunking():
    """Test chunking an empty document without crashing."""
    doc = Document(
        metadata=DocumentMetadata(
            document_id="doc_empty",
            filename="empty.pdf",
            original_filename="empty.pdf",
            title="Empty",
        ),
        pages=[],
    )
    chunks = chunk_document(doc)
    assert chunks == []
