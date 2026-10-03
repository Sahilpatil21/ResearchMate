"""Experiment and benchmark persistence logger for ResearchMate.

Saves structured experiment runs, retrieval metrics, generation evaluations,
and human ratings under data/evaluation/ with zero secret leakage.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.evaluation.generation_evaluator import AutomatedGenerationMetrics, GenerationEvaluator, HumanEvaluationRating
from src.rag.rag_pipeline import RAGResponse
from src.utils.file_utils import ensure_directories, get_evaluation_dir


class ExperimentLogger:
    """Logs and retrieves experiment records, benchmark metrics, and human ratings."""

    def __init__(self, log_dir: Optional[Path] = None) -> None:
        """Initialize the ExperimentLogger."""
        ensure_directories()
        self.log_dir = log_dir or get_evaluation_dir()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_file = self.log_dir / "experiments.jsonl"
        self.ratings_file = self.log_dir / "human_ratings.jsonl"

    def log_experiment_run(
        self,
        response: RAGResponse,
        retrieval_method: str = "hybrid_rerank",
        embedding_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        candidate_k: int = 20,
        final_top_k: int = 5,
        selected_papers: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record an end-to-end RAG experiment run."""
        auto_metrics = GenerationEvaluator.evaluate_response(response)

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": response.query,
            "resolved_query": response.resolved_query,
            "retrieval_method": retrieval_method,
            "embedding_model": embedding_model,
            "reranker_model": reranker_model,
            "candidate_k": candidate_k,
            "final_top_k": final_top_k,
            "selected_papers": selected_papers or ["all"],
            "retrieved_chunk_ids": [r.chunk_id for r in response.retrieval_results],
            "citation_ids": [c.citation_id for c in response.citations],
            "model": response.model,
            "provider": response.provider,
            "latency_seconds": response.latency_seconds,
            "faithfulness_score": auto_metrics.faithfulness_score,
            "citation_validity_rate": auto_metrics.citation_validity_rate,
            "context_term_overlap": auto_metrics.context_term_overlap,
            "has_hallucinated_citations": auto_metrics.has_hallucinated_citations,
            "is_insufficient_evidence": response.is_insufficient_evidence,
            "answer_preview": response.answer[:200] + "..." if len(response.answer) > 200 else response.answer,
            "notes": notes or "",
        }

        try:
            with open(self.experiments_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

        return record

    def log_human_rating(self, rating: HumanEvaluationRating) -> None:
        """Persist a human evaluation rating record."""
        record = rating.model_dump()
        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(self.ratings_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def load_recent_experiments(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Load recent experiment logs from disk."""
        if not self.experiments_file.exists():
            return []

        records = []
        try:
            lines = self.experiments_file.read_text(encoding="utf-8").strip().split("\n")
            for line in lines:
                if line.strip():
                    records.append(json.loads(line.strip()))
        except Exception:
            return []

        # Return latest runs first
        return list(reversed(records))[:limit]

    def load_human_ratings(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Load human rating logs from disk."""
        if not self.ratings_file.exists():
            return []

        records = []
        try:
            lines = self.ratings_file.read_text(encoding="utf-8").strip().split("\n")
            for line in lines:
                if line.strip():
                    records.append(json.loads(line.strip()))
        except Exception:
            return []

        return list(reversed(records))[:limit]
