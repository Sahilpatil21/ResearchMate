"""Retrieval result models for ResearchMate.

Defines structured representations for chunks retrieved across dense vector search,
sparse BM25 search, and hybrid Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """Structured result from a retrieval operation (dense, sparse, or hybrid)."""

    chunk_id: str = Field(..., description="Unique ID of the matched chunk")
    document_id: str = Field(..., description="Parent document identifier")
    filename: str = Field(..., description="Original or storage filename")
    page_number: int = Field(..., description="1-indexed source PDF page number")
    section: str = Field(default="General", description="Academic section header")
    chunk_index: int = Field(..., description="Sequential chunk index")
    text: str = Field(..., description="Matched text content")
    score: float = Field(..., description="Primary ranking/similarity score (higher is more relevant)")
    
    # Retrieval provenance metadata
    retrieval_method: str = Field(
        default="hybrid",
        description="Method used for retrieval: 'hybrid', 'dense', 'bm25'",
    )
    dense_score: Optional[float] = Field(
        default=None,
        description="Cosine similarity score from dense vector search (0.0 to 1.0)",
    )
    dense_rank: Optional[int] = Field(
        default=None,
        description="1-indexed rank position in dense vector search results",
    )
    sparse_score: Optional[float] = Field(
        default=None,
        description="Raw or normalized BM25 score from sparse keyword search",
    )
    sparse_rank: Optional[int] = Field(
        default=None,
        description="1-indexed rank position in BM25 keyword search results",
    )
    rrf_score: Optional[float] = Field(
        default=None,
        description="Reciprocal Rank Fusion score when hybrid retrieval is used",
    )
    reranker_score: Optional[float] = Field(
        default=None,
        description="Cross-encoder relevance score (higher means higher query-passage relevance)",
    )
    reranker_rank: Optional[int] = Field(
        default=None,
        description="1-indexed rank position after Cross-Encoder reranking",
    )
    source_snippet: Optional[str] = Field(
        default=None,
        description="Key extracted sentence or passage from the chunk relevant to the query",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional chunk or retrieval metadata",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize retrieval result to a dictionary."""
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """Serialize retrieval result to formatted JSON string."""
        return json.dumps(self.model_dump(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RetrievalResult:
        """Instantiate RetrievalResult from a dictionary."""
        return cls.model_validate(data)


class RetrievalMetrics(BaseModel):
    """Execution metrics and stats for a retrieval query."""

    query: str
    retrieval_method: str
    total_candidates_evaluated: int = 0
    dense_candidates_count: int = 0
    sparse_candidates_count: int = 0
    top_k_returned: int = 0
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    rrf_k: int = 60
    filter_doc_id: Optional[str] = None
