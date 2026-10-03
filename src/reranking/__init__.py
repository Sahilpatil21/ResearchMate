"""Reranking package for ResearchMate.

Provides Cross-Encoder contextual reranking models to score query-passage relevance
and reorder candidate retrieved chunks.
"""

from src.reranking.cross_encoder import CrossEncoderReranker, get_reranker

__all__ = ["CrossEncoderReranker", "get_reranker"]
