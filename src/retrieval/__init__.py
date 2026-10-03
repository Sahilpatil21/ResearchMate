"""Advanced retrieval package for ResearchMate.

Provides dense vector retrieval, BM25 sparse keyword retrieval, Reciprocal Rank Fusion (RRF),
and unified hybrid search pipelines for research papers.
"""

from src.models.retrieval import RetrievalMetrics, RetrievalResult
from src.retrieval.bm25_retriever import BM25Retriever, tokenize_academic_text
from src.retrieval.fusion import reciprocal_rank_fusion, weighted_score_fusion
from src.retrieval.hybrid_retriever import HybridRetriever, get_hybrid_retriever

__all__ = [
    "BM25Retriever",
    "HybridRetriever",
    "RetrievalMetrics",
    "RetrievalResult",
    "get_hybrid_retriever",
    "reciprocal_rank_fusion",
    "tokenize_academic_text",
    "weighted_score_fusion",
]
