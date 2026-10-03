"""Unit tests for MongoStorageService."""

from __future__ import annotations

import pytest

from src.models.chunk import Chunk, ChunkMetadata
from src.models.document import Document, DocumentMetadata, Page
from src.storage.mongo_storage import MongoStorageService


@pytest.fixture
def storage_service(tmp_path):
    """Instantiate MongoStorageService."""
    service = MongoStorageService()
    return service


def test_storage_service_init(storage_service):
    """Test storage service initialization and attributes."""
    assert storage_service is not None
    assert isinstance(storage_service.is_connected, bool)


def test_save_and_load_document(storage_service):
    """Test saving and loading a document for a specific user."""
    user_id = "test_user_storage_01"
    doc = Document(
        metadata=DocumentMetadata(
            document_id="doc_storage_test_01",
            filename="sample_paper.pdf",
            original_filename="sample_paper.pdf",
            title="A Sample Research Paper",
            page_count=2,
            total_pages_in_pdf=2,
        ),
        pages=[
            Page(page_number=1, text="Sample abstract text on page 1."),
            Page(page_number=2, text="Sample conclusion text on page 2."),
        ],
    )

    # Save
    storage_service.save_document(doc, user_id=user_id)

    # Load
    loaded = storage_service.load_document("doc_storage_test_01", user_id=user_id)
    assert loaded is not None
    assert loaded.document_id == "doc_storage_test_01"
    assert loaded.metadata.filename == "sample_paper.pdf"
    assert len(loaded.pages) == 2

    # Load all user documents
    user_docs = storage_service.load_user_documents(user_id=user_id)
    assert len(user_docs) >= 1
    assert any(d.document_id == "doc_storage_test_01" for d in user_docs)

    # Count
    count = storage_service.count_user_documents(user_id=user_id)
    assert count >= 1

    # Cleanup
    storage_service.delete_document("doc_storage_test_01", user_id=user_id)


def test_save_and_load_chunks(storage_service):
    """Test saving and loading chunks."""
    user_id = "test_user_chunks_01"
    doc_id = "doc_chunks_test_01"
    chunks = [
        Chunk(
            text="First chunk text content.",
            metadata=ChunkMetadata(
                chunk_id="chunk_0",
                chunk_index=0,
                document_id=doc_id,
                filename="doc_chunks_test_01.pdf",
                section="Introduction",
                page_number=1,
            ),
        ),
        Chunk(
            text="Second chunk text content.",
            metadata=ChunkMetadata(
                chunk_id="chunk_1",
                chunk_index=1,
                document_id=doc_id,
                filename="doc_chunks_test_01.pdf",
                section="Methods",
                page_number=2,
            ),
        ),
    ]

    # Save
    storage_service.save_chunks(doc_id, chunks, user_id=user_id)

    # Cleanup
    storage_service.delete_document(doc_id, user_id=user_id)


def test_save_and_load_chat_history(storage_service):
    """Test chat turn persistence."""
    user_id = "test_user_chat_01"
    
    # Save turns
    storage_service.save_chat_turn(
        user_id=user_id,
        role="user",
        content="What is the main finding of paper A?",
    )
    storage_service.save_chat_turn(
        user_id=user_id,
        role="assistant",
        content="The main finding is that hybrid search improves MRR by 18% [1].",
        citations=[{"citation_number": 1, "filename": "paperA.pdf", "page_number": 3}],
    )

    # Load
    history = storage_service.load_user_chat_history(user_id=user_id)
    if storage_service.is_connected:
        assert len(history) >= 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    # Clear
    storage_service.clear_user_chat_history(user_id=user_id)


def test_save_and_load_summary(storage_service):
    """Test summary persistence."""
    user_id = "test_user_summary_01"
    doc_id = "doc_sum_01"
    summary_data = {
        "paper_title": "Attention is All You Need",
        "research_problem": "Recurrent models are sequential and slow.",
        "objective": "Propose Transformer architecture.",
    }

    storage_service.save_summary(doc_id, summary_data, user_id=user_id)
    
    if storage_service.is_connected:
        loaded = storage_service.get_summary(doc_id, user_id=user_id)
        assert loaded is not None
        assert loaded.get("paper_title") == "Attention is All You Need"

    # Cleanup
    storage_service.delete_document(doc_id, user_id=user_id)
