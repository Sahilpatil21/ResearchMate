"""Cross-encoder reranking engine for ResearchMate.

Provides deep contextual query-passage relevance scoring using pre-trained Cross-Encoder models
(default: 'cross-encoder/ms-marco-MiniLM-L-6-v2') with lazy loading, CPU compatibility,
and full provenance preservation.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv

from src.models.retrieval import RetrievalResult

load_dotenv()


class CrossEncoderReranker:
    """Manages Cross-Encoder models for secondary reranking of candidate chunks."""

    _instance: Optional[CrossEncoderReranker] = None

    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialize the cross-encoder reranker.

        Args:
            model_name: HuggingFace cross-encoder model name or local path.
                Defaults to RERANKER_MODEL_NAME environment variable or 'cross-encoder/ms-marco-MiniLM-L-6-v2'.
        """
        self.model_name = (
            model_name
            or os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        )
        self._model = None

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None) -> CrossEncoderReranker:
        """Get or create singleton instance of CrossEncoderReranker."""
        if cls._instance is None or (model_name and cls._instance.model_name != model_name):
            cls._instance = cls(model_name)
        return cls._instance
    def _load_model(self) -> None:
        """Lazy-load the CrossEncoder model on first inference request."""
        if self._model is None:
            try:
                import torch
                torch.set_num_threads(1)
                from sentence_transformers import CrossEncoder

                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model = CrossEncoder(self.model_name, device=device)
                self._model.model.eval()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load CrossEncoder model '{self.model_name}': {e}"
                ) from e

    @property
    def is_loaded(self) -> bool:
        """Return True if model is loaded into memory."""
        return self._model is not None

    def score_pairs(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 32,
    ) -> List[float]:
        """Compute relevance scores for a list of (query, passage) pairs.

        Args:
            pairs: List of (query, text) tuples.
            batch_size: Batch size for model inference.

        Returns:
            List of float scores where higher values indicate greater relevance.
        """
        if not pairs:
            return []

        self._load_model()
        scores = self._model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        if hasattr(scores, "tolist"):
            scores_list = scores.tolist()
        else:
            scores_list = list(scores)

        # Handle scalar or 1D list
        if isinstance(scores_list, float):
            return [scores_list]
        return [float(s) for s in scores_list]

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Rerank a pool of candidate chunks using deep cross-encoder query-passage scoring.

        Preserves all prior retrieval scores and ranking provenance.

        Args:
            query: User natural language search query.
            candidates: Candidate RetrievalResult objects (e.g. top 20 from Hybrid/RRF).
            top_k: Maximum number of top reranked chunks to return. Defaults to len(candidates).

        Returns:
            List of reranked RetrievalResult instances sorted descending by reranker_score.
        """
        if not candidates or not query or not query.strip():
            limit = top_k if top_k is not None else len(candidates)
            return candidates[:limit]

        # Prepare (query, passage) pairs
        clean_query = query.strip()
        pairs = [(clean_query, c.text) for c in candidates]

        # Compute cross-encoder relevance scores
        scores = self.score_pairs(pairs)

        # Build reranked results preserving full provenance
        scored_candidates: List[Tuple[float, RetrievalResult]] = []
        for orig_res, raw_score in zip(candidates, scores):
            float_score = round(float(raw_score), 4)

            # Clone and enrich metadata
            enriched_meta = dict(orig_res.metadata)
            enriched_meta["pre_rerank_rank"] = orig_res.metadata.get(
                "retrieval_rank", getattr(orig_res, "dense_rank", None) or getattr(orig_res, "sparse_rank", None)
            )
            enriched_meta["pre_rerank_score"] = orig_res.score

            reranked_item = RetrievalResult(
                chunk_id=orig_res.chunk_id,
                document_id=orig_res.document_id,
                filename=orig_res.filename,
                page_number=orig_res.page_number,
                section=orig_res.section,
                chunk_index=orig_res.chunk_index,
                text=orig_res.text,
                score=float_score,  # Primary ranking score becomes reranker_score
                reranker_score=float_score,
                dense_score=orig_res.dense_score,
                dense_rank=orig_res.dense_rank,
                sparse_score=orig_res.sparse_score,
                sparse_rank=orig_res.sparse_rank,
                rrf_score=orig_res.rrf_score,
                source_snippet=orig_res.source_snippet,
                retrieval_method=f"reranked ({self.model_name.split('/')[-1]})",
                metadata=enriched_meta,
            )
            scored_candidates.append((float_score, reranked_item))

        # Sort descending by reranker score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        limit = top_k if top_k is not None else len(scored_candidates)
        final_results: List[RetrievalResult] = []

        for rank_idx, (_, item) in enumerate(scored_candidates[:limit], start=1):
            item.reranker_rank = rank_idx
            final_results.append(item)

        return final_results

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic information about the reranker model."""
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "device": "GPU (CUDA)" if self.is_loaded and getattr(self._model, "device", None) and "cuda" in str(self._model.device) else "CPU",
            "status": "Loaded & Ready" if self.is_loaded else "Lazy (Not Loaded Yet)",
        }


def get_reranker(model_name: Optional[str] = None) -> CrossEncoderReranker:
    """Convenience accessor to get singleton CrossEncoderReranker instance."""
    return CrossEncoderReranker.get_instance(model_name)
