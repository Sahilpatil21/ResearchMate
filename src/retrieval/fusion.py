"""Rank and score fusion algorithms for hybrid retrieval in ResearchMate.

Provides Reciprocal Rank Fusion (RRF) and Weighted Score Fusion algorithms
to merge results from dense semantic vector search and sparse BM25 keyword search.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

from src.models.retrieval import RetrievalResult
from src.vectorstore.chroma_store import SearchResult


def reciprocal_rank_fusion(
    dense_results: List[Union[SearchResult, RetrievalResult]],
    sparse_results: List[RetrievalResult],
    k: int = 60,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    top_k: int = 10,
) -> List[RetrievalResult]:
    """Merge and re-rank candidates using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_score(chunk) = (w_dense / (k + rank_dense)) + (w_sparse / (k + rank_sparse))

    Args:
        dense_results: Ordered list of dense vector search results (rank 1 to N).
        sparse_results: Ordered list of sparse BM25 search results (rank 1 to M).
        k: Smoothing constant to avoid dominance by top ranks (default: 60).
        dense_weight: Relative weight multiplier for dense vector results (default: 0.5).
        sparse_weight: Relative weight multiplier for sparse BM25 results (default: 0.5).
        top_k: Number of fused results to return.

    Returns:
        List of fused RetrievalResult instances sorted by RRF score descending.
    """
    if k <= 0:
        raise ValueError("RRF constant k must be positive")
    if dense_weight < 0 or sparse_weight < 0:
        raise ValueError("Weights must be non-negative")
    if dense_weight == 0 and sparse_weight == 0:
        raise ValueError("At least one weight must be positive")

    # Map chunk_id to fused data
    fused_map: Dict[str, Dict] = {}

    # 1. Process dense results
    for rank, item in enumerate(dense_results, start=1):
        chunk_id = item.chunk_id
        dense_score = getattr(item, "score", 0.0)
        
        # Calculate RRF component
        rrf_contrib = dense_weight * (1.0 / (k + rank))

        fused_map[chunk_id] = {
            "chunk_id": chunk_id,
            "document_id": item.document_id,
            "filename": item.filename,
            "page_number": item.page_number,
            "section": getattr(item, "section", "General") or "General",
            "chunk_index": item.chunk_index,
            "text": item.text,
            "dense_rank": rank,
            "dense_score": dense_score,
            "sparse_rank": None,
            "sparse_score": None,
            "rrf_score": rrf_contrib,
            "in_dense": True,
            "in_sparse": False,
        }

    # 2. Process sparse results
    for rank, item in enumerate(sparse_results, start=1):
        chunk_id = item.chunk_id
        sparse_score = item.sparse_score if item.sparse_score is not None else item.score
        rrf_contrib = sparse_weight * (1.0 / (k + rank))

        if chunk_id in fused_map:
            # Chunk was retrieved by both dense and sparse
            entry = fused_map[chunk_id]
            entry["sparse_rank"] = rank
            entry["sparse_score"] = sparse_score
            entry["rrf_score"] += rrf_contrib
            entry["in_sparse"] = True
        else:
            # Chunk was retrieved only by sparse
            fused_map[chunk_id] = {
                "chunk_id": chunk_id,
                "document_id": item.document_id,
                "filename": item.filename,
                "page_number": item.page_number,
                "section": item.section or "General",
                "chunk_index": item.chunk_index,
                "text": item.text,
                "dense_rank": None,
                "dense_score": None,
                "sparse_rank": rank,
                "sparse_score": sparse_score,
                "rrf_score": rrf_contrib,
                "in_dense": False,
                "in_sparse": True,
            }

    if not fused_map:
        return []

    # 3. Sort candidates descending by fused RRF score
    sorted_candidates = sorted(
        fused_map.values(),
        key=lambda x: x["rrf_score"],
        reverse=True,
    )

    # 4. Truncate to top_k
    top_candidates = sorted_candidates[:top_k]

    # Convert to structured RetrievalResult instances
    results: List[RetrievalResult] = []
    for entry in top_candidates:
        method = "hybrid"
        if entry["in_dense"] and not entry["in_sparse"]:
            method = "hybrid (dense-only)"
        elif entry["in_sparse"] and not entry["in_dense"]:
            method = "hybrid (bm25-only)"

        result = RetrievalResult(
            chunk_id=entry["chunk_id"],
            document_id=entry["document_id"],
            filename=entry["filename"],
            page_number=entry["page_number"],
            section=entry["section"],
            chunk_index=entry["chunk_index"],
            text=entry["text"],
            score=round(entry["rrf_score"], 6),
            rrf_score=round(entry["rrf_score"], 6),
            dense_score=entry["dense_score"],
            dense_rank=entry["dense_rank"],
            sparse_score=entry["sparse_score"],
            sparse_rank=entry["sparse_rank"],
            retrieval_method=method,
            metadata={
                "rrf_k": k,
                "dense_weight": dense_weight,
                "sparse_weight": sparse_weight,
                "retrieved_by_dense": entry["in_dense"],
                "retrieved_by_sparse": entry["in_sparse"],
            },
        )
        results.append(result)

    return results


def weighted_score_fusion(
    dense_results: List[Union[SearchResult, RetrievalResult]],
    sparse_results: List[RetrievalResult],
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
    top_k: int = 10,
) -> List[RetrievalResult]:
    """Merge candidates using linear combination of normalized scores.

    Formula:
        Score(chunk) = (w_dense * norm_dense_score) + (w_sparse * norm_sparse_score)
    """
    if dense_weight < 0 or sparse_weight < 0:
        raise ValueError("Weights must be non-negative")
    if dense_weight == 0 and sparse_weight == 0:
        raise ValueError("At least one weight must be positive")

    # Normalize weights to sum to 1.0
    total_w = dense_weight + sparse_weight
    w_d = dense_weight / total_w
    w_s = sparse_weight / total_w

    # Compute max sparse score for min-max normalization
    max_sparse = max((r.sparse_score or r.score for r in sparse_results), default=1.0)
    if max_sparse <= 0:
        max_sparse = 1.0

    fused_map: Dict[str, Dict] = {}

    for rank, item in enumerate(dense_results, start=1):
        chunk_id = item.chunk_id
        dense_score = getattr(item, "score", 0.0)
        norm_dense = max(0.0, min(1.0, dense_score))
        
        fused_map[chunk_id] = {
            "chunk_id": chunk_id,
            "document_id": item.document_id,
            "filename": item.filename,
            "page_number": item.page_number,
            "section": getattr(item, "section", "General") or "General",
            "chunk_index": item.chunk_index,
            "text": item.text,
            "dense_rank": rank,
            "dense_score": dense_score,
            "norm_dense": norm_dense,
            "sparse_rank": None,
            "sparse_score": None,
            "norm_sparse": 0.0,
            "in_dense": True,
            "in_sparse": False,
        }

    for rank, item in enumerate(sparse_results, start=1):
        chunk_id = item.chunk_id
        raw_sparse = item.sparse_score if item.sparse_score is not None else item.score
        norm_sparse = max(0.0, min(1.0, raw_sparse / max_sparse))

        if chunk_id in fused_map:
            entry = fused_map[chunk_id]
            entry["sparse_rank"] = rank
            entry["sparse_score"] = raw_sparse
            entry["norm_sparse"] = norm_sparse
            entry["in_sparse"] = True
        else:
            fused_map[chunk_id] = {
                "chunk_id": chunk_id,
                "document_id": item.document_id,
                "filename": item.filename,
                "page_number": item.page_number,
                "section": item.section or "General",
                "chunk_index": item.chunk_index,
                "text": item.text,
                "dense_rank": None,
                "dense_score": None,
                "norm_dense": 0.0,
                "sparse_rank": rank,
                "sparse_score": raw_sparse,
                "norm_sparse": norm_sparse,
                "in_dense": False,
                "in_sparse": True,
            }

    # Calculate final combined scores
    for entry in fused_map.values():
        combined_score = (w_d * entry["norm_dense"]) + (w_s * entry["norm_sparse"])
        entry["final_score"] = combined_score

    sorted_candidates = sorted(
        fused_map.values(),
        key=lambda x: x["final_score"],
        reverse=True,
    )

    top_candidates = sorted_candidates[:top_k]
    results: List[RetrievalResult] = []

    for entry in top_candidates:
        result = RetrievalResult(
            chunk_id=entry["chunk_id"],
            document_id=entry["document_id"],
            filename=entry["filename"],
            page_number=entry["page_number"],
            section=entry["section"],
            chunk_index=entry["chunk_index"],
            text=entry["text"],
            score=round(entry["final_score"], 4),
            dense_score=entry["dense_score"],
            dense_rank=entry["dense_rank"],
            sparse_score=entry["sparse_score"],
            sparse_rank=entry["sparse_rank"],
            retrieval_method="hybrid (weighted-score)",
            metadata={
                "dense_weight": dense_weight,
                "sparse_weight": sparse_weight,
                "norm_dense_score": round(entry["norm_dense"], 4),
                "norm_sparse_score": round(entry["norm_sparse"], 4),
            },
        )
        results.append(result)

    return results
