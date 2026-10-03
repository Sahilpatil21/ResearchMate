"""Unit and integration tests for Stage 5 Cross-Encoder Reranking."""

import pytest
from pathlib import Path

from src.models.chunk import Chunk, ChunkMetadata
from src.models.document import Document, DocumentMetadata, Page
from src.models.retrieval import RetrievalResult
from src.reranking.cross_encoder import CrossEncoderReranker, get_reranker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.vectorstore.chroma_store import ChromaStore


@pytest.fixture
def sample_candidate_chunks() -> list[RetrievalResult]:
    """Create a diverse candidate pool of retrieval results."""
    c1 = RetrievalResult(
        chunk_id="chk_resnet_01",
        document_id="doc_resnet",
        filename="resnet_2015.pdf",
        page_number=2,
        section="2. Architecture",
        chunk_index=1,
        text="We reformulate the layers as learning residual functions with reference to layer inputs, F(x) + x, using skip shortcut connections.",
        score=0.85,
        dense_score=0.85,
        dense_rank=1,
        sparse_score=8.5,
        sparse_rank=2,
        rrf_score=0.0163,
        retrieval_method="hybrid",
    )
    c2 = RetrievalResult(
        chunk_id="chk_transformer_01",
        document_id="doc_transformer",
        filename="attention_2017.pdf",
        page_number=3,
        section="3. Attention",
        chunk_index=2,
        text="The Transformer relies entirely on multi-head self-attention mechanisms to compute representations of input sequences.",
        score=0.78,
        dense_score=0.78,
        dense_rank=2,
        sparse_score=12.0,
        sparse_rank=1,
        rrf_score=0.0161,
        retrieval_method="hybrid",
    )
    c3 = RetrievalResult(
        chunk_id="chk_bert_01",
        document_id="doc_bert",
        filename="bert_2018.pdf",
        page_number=4,
        section="4. Experiments",
        chunk_index=3,
        text="BERT pre-trains deep bidirectional representations using Masked Language Model (MLM) and Next Sentence Prediction (NSP).",
        score=0.72,
        dense_score=0.72,
        dense_rank=3,
        sparse_score=6.0,
        sparse_rank=3,
        rrf_score=0.0158,
        retrieval_method="hybrid",
    )
    return [c1, c2, c3]


def test_reranker_lazy_loading():
    """Test that CrossEncoderReranker initializes lazily without immediate heavy loading."""
    reranker = CrossEncoderReranker()
    assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    stats = reranker.get_stats()
    assert "cross-encoder" in stats["model_name"]
    assert "status" in stats


def test_cross_encoder_scoring_and_reranking(sample_candidate_chunks):
    """Test that Cross-Encoder scores query-passage pairs and reorders candidates."""
    reranker = get_reranker()

    # Query specifically targeting Transformer attention
    query = "How does multi-head self-attention mechanism work in transformers?"
    reranked = reranker.rerank(query=query, candidates=sample_candidate_chunks, top_k=3)

    assert len(reranked) == 3

    # Transformer chunk should be ranked #1 by cross-encoder for this specific query
    assert reranked[0].chunk_id == "chk_transformer_01"
    assert reranked[0].reranker_rank == 1
    assert reranked[0].reranker_score is not None

    # Verify score descending order
    scores = [r.score for r in reranked]
    assert scores == sorted(scores, reverse=True)


def test_provenance_and_metadata_preservation(sample_candidate_chunks):
    """Test that reranking preserves all original metadata and previous scores."""
    reranker = get_reranker()
    query = "residual skip shortcut connections"
    reranked = reranker.rerank(query=query, candidates=sample_candidate_chunks, top_k=2)

    assert len(reranked) == 2
    top = reranked[0]

    # Preserves chunk & document metadata
    assert top.document_id in ["doc_resnet", "doc_transformer", "doc_bert"]
    assert top.filename.endswith(".pdf")
    assert isinstance(top.page_number, int)
    assert top.page_number > 0
    assert top.section is not None
    assert top.text is not None

    # Preserves prior retrieval scores
    assert top.dense_score is not None
    assert top.dense_rank is not None
    assert top.sparse_score is not None
    assert top.sparse_rank is not None
    assert top.rrf_score is not None

    # Has new reranker attributes
    assert top.reranker_score is not None
    assert top.reranker_rank == 1
    assert "reranked" in top.retrieval_method


def test_empty_candidates_and_query_handling():
    """Test edge cases: empty candidate list and whitespace queries."""
    reranker = get_reranker()

    assert reranker.rerank(query="", candidates=[]) == []
    assert reranker.rerank(query="test query", candidates=[]) == []

    dummy = RetrievalResult(
        chunk_id="c1",
        document_id="d1",
        filename="f.pdf",
        page_number=1,
        chunk_index=0,
        text="Sample text",
        score=0.5,
    )
    # Empty query should gracefully return original candidate
    res = reranker.rerank(query="", candidates=[dummy])
    assert len(res) == 1
    assert res[0].chunk_id == "c1"


def test_hybrid_retriever_with_reranking(tmp_path):
    """Integration test verifying HybridRetriever execute pipeline with reranking enabled."""
    db_dir = tmp_path / "hybrid_rerank_test"
    chroma_store = ChromaStore(persist_dir=db_dir, collection_name="test_rerank_col")
    bm25 = BM25Retriever()
    retriever = HybridRetriever(chroma_store=chroma_store, bm25_retriever=bm25, auto_sync=False)

    doc = Document(
        metadata=DocumentMetadata(
            document_id="doc_bert_deep",
            filename="bert_paper.pdf",
            original_filename="bert_paper.pdf",
            title="BERT Paper",
            authors=["Devlin et al."],
            page_count=2,
            total_pages_in_pdf=2,
            has_extractable_text=True,
        ),
        pages=[
            Page(
                page_number=1,
                text="1. Introduction\nLanguage representation models pre-train deep neural networks.",
            ),
            Page(
                page_number=2,
                text="2. Training Tasks\nMasked Language Model (MLM) randomly masks tokens to predict them bidirectionally.",
            ),
        ],
    )

    retriever.index_document(doc, chunk_size=200, chunk_overlap=30)

    # Test retrieve with rerank=True
    results = retriever.retrieve(
        query="Masked Language Model token prediction",
        mode="hybrid",
        top_k=2,
        candidate_k=5,
        rerank=True,
    )

    assert len(results) > 0
    assert results[0].document_id == "doc_bert_deep"
    assert results[0].reranker_score is not None
    assert results[0].reranker_rank == 1
    assert "reranked" in results[0].retrieval_method
