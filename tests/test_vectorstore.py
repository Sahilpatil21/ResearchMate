"""Unit tests for Step 2 Embedding Generation and ChromaDB Vector Store."""

import pytest
from pathlib import Path

from src.chunking.text_chunker import TextChunker
from src.embeddings.embedding_model import EmbeddingModel, get_embedding_model
from src.models.document import Document, DocumentMetadata, Page
from src.vectorstore.chroma_store import ChromaStore, SearchResult


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create two sample research documents in different domains."""
    doc1 = Document(
        metadata=DocumentMetadata(
            document_id="doc_resnet_001",
            filename="resnet_paper.pdf",
            original_filename="resnet_paper.pdf",
            title="Deep Residual Learning for Image Recognition",
            authors=["Kaiming He", "Jian Sun"],
            page_count=2,
            total_pages_in_pdf=2,
            has_extractable_text=True,
        ),
        pages=[
            Page(
                page_number=1,
                text=(
                    "Deep Residual Learning for Image Recognition.\n\n"
                    "Abstract\n"
                    "Deeper neural networks are more difficult to train. We present a residual learning "
                    "framework to ease the training of networks that are substantially deeper than those "
                    "used previously. We explicitly reformulate the layers as learning residual functions "
                    "with reference to the layer inputs.\n\n"
                    "1. Introduction\n"
                    "Convolutional networks have significantly advanced image classification benchmarks."
                ),
            ),
            Page(
                page_number=2,
                text=(
                    "2. Methodology\n"
                    "We formulate the residual mapping as F(x) + x, adding skip shortcut connections "
                    "that perform identity mapping without adding extra parameters or computational complexity.\n\n"
                    "3. Results\n"
                    "Our 152-layer ResNet achieves a 3.57% top-5 error on the ImageNet test set."
                ),
            ),
        ],
    )

    doc2 = Document(
        metadata=DocumentMetadata(
            document_id="doc_transformer_002",
            filename="attention_paper.pdf",
            original_filename="attention_paper.pdf",
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            page_count=2,
            total_pages_in_pdf=2,
            has_extractable_text=True,
        ),
        pages=[
            Page(
                page_number=1,
                text=(
                    "Attention Is All You Need.\n\n"
                    "Abstract\n"
                    "The dominant sequence transduction models are based on complex recurrent or "
                    "convolutional neural networks. We propose the Transformer, a model architecture "
                    "eschewing recurrence and relying entirely on an attention mechanism to draw global dependencies.\n\n"
                    "1. Introduction\n"
                    "Recurrent models generate sequences of hidden states aligned to positions."
                ),
            ),
            Page(
                page_number=2,
                text=(
                    "2. Architecture\n"
                    "The Transformer uses multi-head self-attention and positional encodings to process "
                    "all tokens in parallel without sequential recurrent steps.\n\n"
                    "3. Results\n"
                    "On the WMT 2014 English-to-German translation task, the Transformer achieves 28.4 BLEU."
                ),
            ),
        ],
    )

    return [doc1, doc2]


def test_embedding_model_generation():
    """Test generating embeddings with sentence-transformers."""
    embedder = get_embedding_model()
    texts = [
        "Deep residual networks use identity skip connections.",
        "Transformers rely on multi-head self-attention mechanisms.",
    ]
    vectors = embedder.embed_texts(texts)

    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384
    assert embedder.dimension == 384

    # Query embedding
    q_vec = embedder.embed_query("skip connections in neural networks")
    assert len(q_vec) == 384


def test_chroma_store_initialization(tmp_path):
    """Test initializing a persistent Chroma vector store."""
    db_dir = tmp_path / "chroma_test"
    store = ChromaStore(persist_dir=db_dir, collection_name="test_collection")

    stats = store.get_collection_stats()
    assert stats["total_vectors"] == 0
    assert stats["indexed_documents_count"] == 0
    assert stats["collection_name"] == "test_collection"
    assert "Healthy" in stats["status"]


def test_add_and_search_chunks(tmp_path, sample_documents):
    """Test indexing documents and querying via semantic similarity."""
    db_dir = tmp_path / "chroma_test"
    store = ChromaStore(persist_dir=db_dir, collection_name="test_search")

    # Index both documents
    doc1, doc2 = sample_documents
    c1, _ = store.index_document(doc1, chunk_size=300, chunk_overlap=50)
    c2, _ = store.index_document(doc2, chunk_size=300, chunk_overlap=50)

    assert c1 > 0
    assert c2 > 0
    assert store.collection.count() == c1 + c2
    assert store.is_document_indexed("doc_resnet_001") is True
    assert store.is_document_indexed("doc_transformer_002") is True

    # Search query 1: Residual connections -> should rank ResNet top
    results1 = store.search("residual learning skip shortcut connections", n_results=3)
    assert len(results1) > 0
    assert results1[0].document_id == "doc_resnet_001"
    assert results1[0].score > 0.4
    assert results1[0].filename == "resnet_paper.pdf"
    assert isinstance(results1[0].page_number, int)

    # Search query 2: Self-attention and translation -> should rank Transformer top
    results2 = store.search("multi-head self-attention sequence translation BLEU", n_results=3)
    assert len(results2) > 0
    assert results2[0].document_id == "doc_transformer_002"
    assert results2[0].score > 0.4
    assert results2[0].filename == "attention_paper.pdf"


def test_duplicate_prevention_and_reindexing(tmp_path, sample_documents):
    """Test duplicate prevention and reindexing with overwrite=True."""
    db_dir = tmp_path / "chroma_test"
    store = ChromaStore(persist_dir=db_dir, collection_name="test_duplicates")
    doc1 = sample_documents[0]

    # First indexing
    count1, _ = store.index_document(doc1, chunk_size=350, chunk_overlap=50)
    assert count1 > 0
    assert store.collection.count() == count1

    # Attempt second indexing without overwrite -> must raise ValueError
    with pytest.raises(ValueError) as exc:
        store.index_document(doc1, overwrite=False)
    assert "already indexed" in str(exc.value)

    # Re-indexing with overwrite=True -> replaces existing vectors
    count2, _ = store.index_document(doc1, chunk_size=350, chunk_overlap=50, overwrite=True)
    assert count2 == count1
    assert store.collection.count() == count1  # Count does not double


def test_delete_document_vectors(tmp_path, sample_documents):
    """Test deleting document embeddings from Chroma store."""
    db_dir = tmp_path / "chroma_test"
    store = ChromaStore(persist_dir=db_dir, collection_name="test_delete")

    doc1, doc2 = sample_documents
    store.index_document(doc1)
    store.index_document(doc2)

    assert store.is_document_indexed("doc_resnet_001") is True
    assert store.is_document_indexed("doc_transformer_002") is True

    # Delete doc1
    deleted = store.delete_document("doc_resnet_001")
    assert deleted is True
    assert store.is_document_indexed("doc_resnet_001") is False
    assert store.is_document_indexed("doc_transformer_002") is True

    # Deleting non-existent doc returns False
    assert store.delete_document("non_existent_doc") is False


def test_filter_by_document_id(tmp_path, sample_documents):
    """Test scoping semantic search to a specific document ID."""
    db_dir = tmp_path / "chroma_test"
    store = ChromaStore(persist_dir=db_dir, collection_name="test_filter")

    for d in sample_documents:
        store.index_document(d)

    # Search with document filter
    results = store.search(
        query="neural network architecture and results",
        n_results=5,
        filter_doc_id="doc_transformer_002",
    )

    assert len(results) > 0
    for r in results:
        assert r.document_id == "doc_transformer_002"
        assert r.filename == "attention_paper.pdf"


def test_chroma_local_persistence(tmp_path, sample_documents):
    """Test that ChromaDB persists vectors to disk across client restarts."""
    db_dir = tmp_path / "chroma_persist"
    doc1 = sample_documents[0]

    # Instance 1: Add vectors and persist
    store1 = ChromaStore(persist_dir=db_dir, collection_name="persist_col")
    count, _ = store1.index_document(doc1)
    assert count > 0
    del store1

    # Instance 2: Connect to same directory
    store2 = ChromaStore(persist_dir=db_dir, collection_name="persist_col")
    assert store2.collection.count() == count
    assert store2.is_document_indexed("doc_resnet_001") is True

    results = store2.search("residual learning", n_results=1)
    assert len(results) == 1
    assert results[0].document_id == "doc_resnet_001"
