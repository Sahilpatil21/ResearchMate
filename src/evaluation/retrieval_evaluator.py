"""Retrieval evaluation engine for ResearchMate.

Benchmarks and compares 4 retrieval strategies across the evaluation dataset:
- A. Dense Vector Retrieval (ChromaDB)
- B. Sparse Keyword Retrieval (BM25)
- C. Hybrid Reciprocal Rank Fusion (RRF)
- D. Hybrid + Cross-Encoder Reranking

Computes objective IR metrics:
- Precision@K
- Recall@K
- Hit Rate@K
- Mean Reciprocal Rank (MRR)
- Latency (ms)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.evaluation.dataset import EvaluationDataset, EvaluationQuestion
from src.models.retrieval import RetrievalResult
from src.retrieval.hybrid_retriever import HybridRetriever, get_hybrid_retriever


class MethodEvaluationSummary(BaseModel):
    """Aggregated evaluation metrics for a specific retrieval strategy."""

    method_name: str = Field(..., description="Retrieval strategy name")
    precision_at_k: float = Field(..., description="Precision@K (0.0 to 1.0)")
    recall_at_k: float = Field(..., description="Recall@K (0.0 to 1.0)")
    hit_rate_at_k: float = Field(..., description="Hit Rate@K (0.0 to 1.0)")
    mrr: float = Field(..., description="Mean Reciprocal Rank (0.0 to 1.0)")
    average_latency_ms: float = Field(..., description="Average query latency in milliseconds")
    total_queries_evaluated: int = Field(..., description="Number of queries evaluated")


class QueryRetrievalResult(BaseModel):
    """Detailed retrieval evaluation metrics for a single query."""

    question_id: str
    question: str
    category: str
    method_name: str
    precision: float
    recall: float
    hit_rate: float
    reciprocal_rank: float
    latency_ms: float
    retrieved_chunk_ids: List[str]
    matched_pages: List[int]


class RetrievalEvaluationReport(BaseModel):
    """Complete multi-method retrieval evaluation report."""

    top_k: int
    candidate_k: int
    method_summaries: Dict[str, MethodEvaluationSummary]
    detailed_query_results: List[QueryRetrievalResult] = Field(default_factory=list)


class RetrievalEvaluator:
    """Evaluates and compares retrieval strategies across ground-truth evaluation datasets."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        dataset: Optional[EvaluationDataset] = None,
    ) -> None:
        """Initialize the RetrievalEvaluator."""
        self.retriever = retriever or get_hybrid_retriever()
        self.dataset = dataset or EvaluationDataset()

    def is_chunk_relevant(
        self,
        result: RetrievalResult,
        question: EvaluationQuestion,
    ) -> bool:
        """Check if a retrieved chunk matches ground-truth document and page criteria."""
        # 1. Check Document Match
        doc_matched = any(
            doc_ref.lower() in result.filename.lower() or doc_ref.lower() in result.document_id.lower()
            for doc_ref in question.relevant_documents
        )
        if not doc_matched:
            return False

        # 2. Check Page Match if specified
        if question.relevant_pages:
            if result.page_number in question.relevant_pages:
                return True

        # 3. Check Keyword Term Overlap
        if question.relevant_keywords:
            text_lower = result.text.lower()
            keyword_matches = sum(1 for kw in question.relevant_keywords if kw.lower() in text_lower)
            if keyword_matches >= 1:
                return True

        # Default fallback to document match if no specific pages/keywords restricted
        return doc_matched

    def evaluate_query(
        self,
        question: EvaluationQuestion,
        method: str,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> QueryRetrievalResult:
        """Evaluate a single query under a given retrieval method."""
        start_time = time.perf_counter()

        rerank = False
        mode = "hybrid"

        if method == "dense":
            mode = "dense"
        elif method == "sparse":
            mode = "sparse"
        elif method == "hybrid":
            mode = "hybrid"
        elif method == "hybrid_rerank":
            mode = "hybrid"
            rerank = True

        results: List[RetrievalResult] = self.retriever.retrieve(
            query=question.question,
            mode=mode,
            top_k=top_k,
            candidate_k=candidate_k,
            rerank=rerank,
        )

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        relevant_count = 0
        first_relevant_rank = 0
        matched_pages = []

        for rank, res in enumerate(results, start=1):
            if self.is_chunk_relevant(res, question):
                relevant_count += 1
                matched_pages.append(res.page_number)
                if first_relevant_rank == 0:
                    first_relevant_rank = rank

        precision = round(relevant_count / max(1, len(results)), 4) if results else 0.0
        # Target relevant pool estimated at min(top_k, len(question.relevant_pages) or 2)
        target_denom = max(1, len(question.relevant_pages) if question.relevant_pages else 2)
        recall = round(min(1.0, relevant_count / target_denom), 4)
        hit_rate = 1.0 if relevant_count > 0 else 0.0
        reciprocal_rank = round(1.0 / first_relevant_rank, 4) if first_relevant_rank > 0 else 0.0

        return QueryRetrievalResult(
            question_id=question.question_id,
            question=question.question,
            category=question.category,
            method_name=method,
            precision=precision,
            recall=recall,
            hit_rate=hit_rate,
            reciprocal_rank=reciprocal_rank,
            latency_ms=latency_ms,
            retrieved_chunk_ids=[r.chunk_id for r in results],
            matched_pages=matched_pages,
        )

    def run_full_benchmark(
        self,
        top_k: int = 5,
        candidate_k: int = 20,
        methods: Optional[List[str]] = None,
    ) -> RetrievalEvaluationReport:
        """Run full evaluation across all dataset questions and comparison methods."""
        eval_methods = methods or ["dense", "sparse", "hybrid", "hybrid_rerank"]
        questions = self.dataset.questions

        method_details: Dict[str, List[QueryRetrievalResult]] = {m: [] for m in eval_methods}

        for q in questions:
            for m in eval_methods:
                res = self.evaluate_query(q, method=m, top_k=top_k, candidate_k=candidate_k)
                method_details[m].append(res)

        method_summaries: Dict[str, MethodEvaluationSummary] = {}

        for m in eval_methods:
            q_results = method_details[m]
            count = len(q_results)
            if count == 0:
                continue

            avg_p = round(sum(r.precision for r in q_results) / count, 4)
            avg_r = round(sum(r.recall for r in q_results) / count, 4)
            avg_hit = round(sum(r.hit_rate for r in q_results) / count, 4)
            avg_mrr = round(sum(r.reciprocal_rank for r in q_results) / count, 4)
            avg_lat = round(sum(r.latency_ms for r in q_results) / count, 2)

            method_summaries[m] = MethodEvaluationSummary(
                method_name=m,
                precision_at_k=avg_p,
                recall_at_k=avg_r,
                hit_rate_at_k=avg_hit,
                mrr=avg_mrr,
                average_latency_ms=avg_lat,
                total_queries_evaluated=count,
            )

        flat_detailed = [item for sublist in method_details.values() for item in sublist]

        return RetrievalEvaluationReport(
            top_k=top_k,
            candidate_k=candidate_k,
            method_summaries=method_summaries,
            detailed_query_results=flat_detailed,
        )

    def evaluate_all_methods(
        self,
        top_k: int = 5,
        candidate_k: int = 20,
        methods: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> RetrievalEvaluationReport:
        """Alias for run_full_benchmark."""
        return self.run_full_benchmark(top_k=top_k, candidate_k=candidate_k, methods=methods)

