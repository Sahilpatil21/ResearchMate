"""Unit and integration tests for Stage 4 Advanced Retrieval (BM25, Hybrid RRF, and Weighted Fusion)."""

import pytest
from pathlib import Path

from src.models.chunk import Chunk, ChunkMetadata
from src.models.document import Document, DocumentMetadata, Page
from src.models.retrieval import RetrievalResult
from src.retrieval.bm25_retriever import BM25Retriever, tokenize_academic_text
from src.retrieval.fusion import reciprocal_rank_fusion, weighted_score_fusion
from src.retrieval.hybrid_retriever import HybridRetriever
from src.vectorstore.chroma_store import ChromaStore, SearchResult


@pytest.fixture
def sample_research_papers() -> list[Document]:
    """Create realistic multi-domain research documents for retrieval testing."""
    doc1 = Document(
        metadata=DocumentMetadata(
            document_id="doc_resnet_001",
            filename="resnet_paper.pdf",
            original_filename="resnet_paper.pdf",
            title="Deep Residual Learning for Image Recognition",
            authors=["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"],
            page_count=2,
            total_pages_in_pdf=2,
            has_extractable_text=True,
        ),
        pages=[
            Page(
                page_number=1,
                text=(
                    "Deep Residual Learning for Image Recognition\n\n"
                    "Abstract\n"
                    "Deeper neural networks are more difficult to train. We present a residual learning "
                    "framework to ease the training of networks that are substantially deeper than those "
                    "used previously. We explicitly reformulate the layers as learning residual functions "
                    "with reference to the layer inputs, instead of learning unreferenced functions.\n\n"
                    "1. Introduction\n"
                    "Deep convolutional networks have led to breakthroughs in computer vision and image classification."
                ),
            ),
            Page(
                page_number=2,
                text=(
                    "2. Deep Residual Architecture\n"
                    "We denote the desired underlying mapping as H(x), and let the stacked nonlinear layers "
                    "fit another mapping of F(x) := H(x) - x. The original mapping is recast into F(x) + x. "
                    "The formulation of F(x) + x can be realized by feedforward neural networks with shortcut connections.\n\n"
                    "3. ImageNet Experiments\n"
                    "Our 152-layer ResNet achieves a 3.57% top-5 error on the ImageNet test set, winning 1st place in ILSVRC 2015."
                ),
            ),
        ],
    )

    doc2 = Document(
        metadata=DocumentMetadata(
            document_id="doc_transformer_002",
            filename="transformer_paper.pdf",
            original_filename="transformer_paper.pdf",
            title="Attention Is All You Need",
            authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
            page_count=2,
            total_pages_in_pdf=2,
            has_extractable_text=True,
        ),
        pages=[
            Page(
                page_number=1,
                text=(
                    "Attention Is All You Need\n\n"
                    "Abstract\n"
                    "The dominant sequence transduction models are based on complex recurrent or "
                    "convolutional neural networks. We propose the Transformer, a model architecture "
                    "eschewing recurrence and relying entirely on an attention mechanism to draw global dependencies.\n\n"
                    "1. Introduction\n"
                    "Recurrent models compute hidden states sequentially aligned to symbol positions."
                ),
            ),
            Page(
                page_number=2,
                text=(
                    "2. Multi-Head Attention Architecture\n"
                    "Multi-head attention allows the model to jointly attend to information from different "
                    "representation subspaces at different positions. We employ scaled dot-product attention with query, key, value matrices.\n\n"
                    "3. Machine Translation Results\n"
                    "On the WMT 2014 English-to-German translation task, the Transformer establishes a new state-of-the-art 28.4 BLEU score."
                ),
            ),
        ],
    )

    doc3 = Document(
        metadata=DocumentMetadata(
            document_id="doc_bert_003",
            filename="bert_paper.pdf",
            original_filename="bert_paper.pdf",
            title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            authors=["Jacob Devlin", "Ming-Wei Chang", "Kenton Lee", "Kristina Toutanova"],
            page_count=2,
            total_pages_in_pdf=2,
            has_extractable_text=True,
        ),
        pages=[
            Page(
                page_number=1,
                text=(
                    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding\n\n"
                    "Abstract\n"
                    "We introduce a new language representation model called BERT, which stands for "
                    "Bidirectional Encoder Representations from Transformers. Unlike recent models, "
                    "BERT pre-trains deep bidirectional representations from unlabeled text.\n\n"
                    "1. Introduction\n"
                    "Language model pre-training has been shown to be effective for improving many natural language processing tasks."
                ),
            ),
            Page(
                page_number=2,
                text=(
                    "2. Pre-training Tasks\n"
                    "BERT is trained using two novel unsupervised tasks: Masked Language Model (MLM) and Next Sentence Prediction (NSP).\n\n"
                    "3. GLUE Benchmark Results\n"
                    "BERT obtains new state-of-the-art results on eleven NLP tasks, including pushing the GLUE score to 80.5%."
                ),
            ),
        ],
    )

    return [doc1, doc2, doc3]


def test_bm25_tokenization():
    """Test academic text tokenization with acronyms and hyphenated phrases."""
    text = "ResNet-152 uses multi-head self-attention and achieves 28.4 BLEU on WMT-2014."
    tokens = tokenize_academic_text(text)

    # Must preserve base tokens and subparts of hyphenated terms
    assert "resnet-152" in tokens
    assert "multi-head" in tokens
    assert "multi" in tokens
    assert "head" in tokens
    assert "bleu" in tokens
    assert "28" in tokens or "28.4" in tokens or "wmt-2014" in tokens


def test_bm25_exact_keyword_retrieval():
    """Test BM25 exact keyword matching for specific acronyms and formula terms."""
    retriever = BM25Retriever()
    
    c1 = Chunk(
        text="The residual building block approximates F(x) := H(x) - x with skip shortcut connections.",
        metadata=ChunkMetadata(
            chunk_id="chk_res_1",
            document_id="doc_res",
            filename="resnet.pdf",
            page_number=2,
            section="Methodology",
            chunk_index=1,
        ),
    )
    c2 = Chunk(
        text="Scaled dot-product attention computes softmax(QK^T / sqrt(d_k))V across multi-head projections.",
        metadata=ChunkMetadata(
            chunk_id="chk_trans_1",
            document_id="doc_trans",
            filename="transformer.pdf",
            page_number=2,
            section="Architecture",
            chunk_index=1,
        ),
    )
    c3 = Chunk(
        text="BERT utilizes Masked Language Model (MLM) and Next Sentence Prediction (NSP) for pre-training.",
        metadata=ChunkMetadata(
            chunk_id="chk_bert_1",
            document_id="doc_bert",
            filename="bert.pdf",
            page_number=2,
            section="Pre-training",
            chunk_index=1,
        ),
    )

    retriever.add_chunks([c1, c2, c3])
    assert retriever.get_stats()["total_indexed_chunks"] == 3

    # Exact query 1: MLM and NSP -> must return BERT chunk as top result
    results_bert = retriever.search("MLM Next Sentence Prediction", n_results=3)
    assert len(results_bert) > 0
    assert results_bert[0].chunk_id == "chk_bert_1"
    assert results_bert[0].document_id == "doc_bert"
    assert results_bert[0].sparse_rank == 1

    # Exact query 2: softmax QK^T -> must return Transformer chunk
    results_trans = retriever.search("softmax QK^T multi-head", n_results=3)
    assert len(results_trans) > 0
    assert results_trans[0].chunk_id == "chk_trans_1"


def test_bm25_metadata_filtering():
    """Test filtering BM25 search by document_id."""
    retriever = BM25Retriever()
    c1 = Chunk(
        text="Neural network optimization with Adam optimizer.",
        metadata=ChunkMetadata(
            chunk_id="chk_1",
            document_id="doc_alpha",
            filename="alpha.pdf",
            page_number=1,
            chunk_index=0,
        ),
    )
    c2 = Chunk(
        text="Neural network regularization with dropout and Adam optimizer.",
        metadata=ChunkMetadata(
            chunk_id="chk_2",
            document_id="doc_beta",
            filename="beta.pdf",
            page_number=1,
            chunk_index=0,
        ),
    )

    retriever.add_chunks([c1, c2])

    # Search with filter
    results = retriever.search("Adam optimizer", filter_doc_id="doc_alpha")
    assert len(results) == 1
    assert results[0].document_id == "doc_alpha"
    assert results[0].chunk_id == "chk_1"


def test_reciprocal_rank_fusion_logic():
    """Test RRF math and ranking behavior with candidate deduplication."""
    # Chunk A is rank 1 in Dense, rank 2 in Sparse
    # Chunk B is rank 2 in Dense, not in Sparse
    # Chunk C is rank 1 in Sparse, not in Dense

    res_a_dense = SearchResult(
        chunk_id="chunk_A",
        document_id="doc_1",
        filename="doc1.pdf",
        page_number=1,
        section="Intro",
        chunk_index=0,
        text="Chunk A text",
        score=0.95,
        distance=0.05,
    )
    res_b_dense = SearchResult(
        chunk_id="chunk_B",
        document_id="doc_1",
        filename="doc1.pdf",
        page_number=2,
        section="Methods",
        chunk_index=1,
        text="Chunk B text",
        score=0.85,
        distance=0.15,
    )

    res_c_sparse = RetrievalResult(
        chunk_id="chunk_C",
        document_id="doc_2",
        filename="doc2.pdf",
        page_number=1,
        section="Abstract",
        chunk_index=0,
        text="Chunk C text",
        score=15.0,
        sparse_score=15.0,
        sparse_rank=1,
        retrieval_method="bm25",
    )
    res_a_sparse = RetrievalResult(
        chunk_id="chunk_A",
        document_id="doc_1",
        filename="doc1.pdf",
        page_number=1,
        section="Intro",
        chunk_index=0,
        text="Chunk A text",
        score=12.0,
        sparse_score=12.0,
        sparse_rank=2,
        retrieval_method="bm25",
    )

    fused = reciprocal_rank_fusion(
        dense_results=[res_a_dense, res_b_dense],
        sparse_results=[res_c_sparse, res_a_sparse],
        k=60,
        dense_weight=0.5,
        sparse_weight=0.5,
        top_k=3,
    )

    assert len(fused) == 3
    # Chunk A was top in dense (rank 1) and #2 in sparse (rank 2) -> its combined RRF score should be highest!
    assert fused[0].chunk_id == "chunk_A"
    assert fused[0].dense_rank == 1
    assert fused[0].sparse_rank == 2
    assert fused[0].dense_score == 0.95
    assert fused[0].sparse_score == 12.0

    # Validate RRF math: 0.5/(60+1) + 0.5/(60+2) = 0.5/61 + 0.5/62
    expected_score = round(0.5 / 61 + 0.5 / 62, 6)
    assert abs(fused[0].score - expected_score) < 1e-5


def test_weighted_score_fusion_logic():
    """Test linear weighted score combination with normalization."""
    dense_res = [
        SearchResult(
            chunk_id="chunk_1",
            document_id="doc_1",
            filename="f1.pdf",
            page_number=1,
            section="Intro",
            chunk_index=0,
            text="Text 1",
            score=0.90,
            distance=0.10,
        )
    ]
    sparse_res = [
        RetrievalResult(
            chunk_id="chunk_1",
            document_id="doc_1",
            filename="f1.pdf",
            page_number=1,
            section="Intro",
            chunk_index=0,
            text="Text 1",
            score=10.0,
            sparse_score=10.0,
            sparse_rank=1,
            retrieval_method="bm25",
        )
    ]

    fused = weighted_score_fusion(
        dense_results=dense_res,
        sparse_results=sparse_res,
        dense_weight=0.7,
        sparse_weight=0.3,
        top_k=1,
    )

    assert len(fused) == 1
    assert fused[0].chunk_id == "chunk_1"
    # dense_norm = 0.9, sparse_norm = 1.0 -> 0.7 * 0.9 + 0.3 * 1.0 = 0.63 + 0.30 = 0.93
    assert abs(fused[0].score - 0.93) < 1e-3


def test_hybrid_retriever_pipeline(tmp_path, sample_research_papers):
    """End-to-end integration test for HybridRetriever with multi-paper indexing and querying."""
    chroma_dir = tmp_path / "chroma_hybrid_test"
    store = ChromaStore(persist_dir=chroma_dir, collection_name="hybrid_test_col")
    bm25 = BM25Retriever()
    retriever = HybridRetriever(chroma_store=store, bm25_retriever=bm25, auto_sync=False)

    # Index all 3 research papers
    for paper in sample_research_papers:
        count, chunks = retriever.index_document(paper, chunk_size=350, chunk_overlap=50)
        assert count > 0

    stats = retriever.get_stats()
    assert stats["chroma_documents"] == 3
    assert stats["bm25_documents"] == 3
    assert stats["chroma_vectors"] == stats["bm25_chunks"]

    # 1. Test Semantic Query (Vector dominates)
    res_semantic = retriever.retrieve(
        query="mitigating the degradation problem in ultra deep neural architectures",
        mode="hybrid",
        top_k=3,
    )
    assert len(res_semantic) > 0
    assert res_semantic[0].document_id == "doc_resnet_001"

    # 2. Test Exact Keyword Query (BM25 keyword matches 'BLEU score' or 'WMT 2014')
    res_keyword = retriever.retrieve(
        query="WMT 2014 translation 28.4 BLEU",
        mode="hybrid",
        top_k=3,
    )
    assert len(res_keyword) > 0
    assert res_keyword[0].document_id == "doc_transformer_002"
    assert res_keyword[0].filename == "transformer_paper.pdf"

    # 3. Test Pure BM25 Mode
    res_sparse_only = retriever.retrieve(
        query="Masked Language Model Next Sentence Prediction GLUE",
        mode="sparse",
        top_k=2,
    )
    assert len(res_sparse_only) > 0
    assert res_sparse_only[0].document_id == "doc_bert_003"
    assert res_sparse_only[0].retrieval_method == "bm25"
    assert res_sparse_only[0].sparse_score is not None

    # 4. Test Pure Dense Mode
    res_dense_only = retriever.retrieve(
        query="attention mechanisms drawing global dependencies across sequences",
        mode="dense",
        top_k=2,
    )
    assert len(res_dense_only) > 0
    assert res_dense_only[0].document_id == "doc_transformer_002"
    assert res_dense_only[0].retrieval_method == "dense"
    assert res_dense_only[0].dense_score is not None

    # 5. Test Filter by Document ID
    res_filtered = retriever.retrieve(
        query="neural network architecture performance",
        mode="hybrid",
        filter_doc_id="doc_bert_003",
        top_k=5,
    )
    assert len(res_filtered) > 0
    for r in res_filtered:
        assert r.document_id == "doc_bert_003"


def test_hybrid_retriever_document_deletion(tmp_path, sample_research_papers):
    """Test deleting documents from both vector store and BM25 index."""
    chroma_dir = tmp_path / "chroma_del_test"
    store = ChromaStore(persist_dir=chroma_dir, collection_name="del_col")
    bm25 = BM25Retriever()
    retriever = HybridRetriever(chroma_store=store, bm25_retriever=bm25, auto_sync=False)

    doc1, doc2 = sample_research_papers[:2]
    retriever.index_document(doc1)
    retriever.index_document(doc2)

    assert retriever.chroma_store.is_document_indexed("doc_resnet_001") is True
    assert retriever.bm25_retriever.is_document_indexed("doc_resnet_001") is True

    # Delete doc1
    deleted = retriever.delete_document("doc_resnet_001")
    assert deleted is True

    assert retriever.chroma_store.is_document_indexed("doc_resnet_001") is False
    assert retriever.bm25_retriever.is_document_indexed("doc_resnet_001") is False
    assert retriever.chroma_store.is_document_indexed("doc_transformer_002") is True
    assert retriever.bm25_retriever.is_document_indexed("doc_transformer_002") is True
