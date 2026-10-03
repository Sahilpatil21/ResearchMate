"""Hybrid retrieval orchestration engine for ResearchMate.

Unifies dense semantic vector retrieval (ChromaDB) and sparse keyword retrieval (BM25)
into a modular, multi-strategy retrieval pipeline with Reciprocal Rank Fusion (RRF) and multi-tenant isolation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.chunking.text_chunker import TextChunker
from src.models.chunk import Chunk
from src.models.document import Document
from src.models.retrieval import RetrievalMetrics, RetrievalResult
from src.reranking.cross_encoder import CrossEncoderReranker, get_reranker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.fusion import reciprocal_rank_fusion, weighted_score_fusion
from src.utils.file_utils import ensure_directories, get_chunks_dir, load_all_document_chunks
from src.vectorstore.chroma_store import ChromaStore, SearchResult


class HybridRetriever:
    """Orchestrates multi-strategy retrieval across dense vector search, sparse BM25, and cross-encoder reranking."""

    _instance: Optional[HybridRetriever] = None

    def __init__(
        self,
        chroma_store: Optional[ChromaStore] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        auto_sync: bool = True,
    ) -> None:
        """Initialize the HybridRetriever."""
        ensure_directories()
        self.chroma_store = chroma_store or ChromaStore()
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.reranker = reranker or get_reranker()

        if auto_sync and len(self.bm25_retriever.chunks) == 0:
            self.sync_from_storage()

    @classmethod
    def get_instance(cls, chroma_store: Optional[ChromaStore] = None) -> HybridRetriever:
        """Get or create singleton instance of HybridRetriever."""
        if cls._instance is None:
            cls._instance = cls(chroma_store=chroma_store)
        return cls._instance

    def sync_from_storage(self, chunks_dir: Optional[Path] = None, user_id: Optional[str] = None) -> int:
        """Sync BM25 index from disk-persisted chunk JSON files in data/chunks/."""
        return self.bm25_retriever.sync_from_storage(chunks_dir=chunks_dir, user_id=user_id)

    def index_document(
        self,
        document: Document,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        overwrite: bool = False,
        user_id: Optional[str] = None,
    ) -> Tuple[int, List[Chunk]]:
        """Index a document across both dense vector storage (ChromaDB) and sparse BM25 index."""
        count, chunks = self.chroma_store.index_document(
            document=document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            overwrite=overwrite,
            user_id=user_id,
        )

        if not chunks:
            return 0, []

        self.bm25_retriever.index_document(
            document_id=document.document_id,
            chunks=chunks,
            overwrite=overwrite,
            user_id=user_id,
        )

        return count, chunks

    def delete_document(self, document_id: str, user_id: Optional[str] = None) -> bool:
        """Remove a document from both ChromaDB vector store and BM25 index."""
        chroma_deleted = self.chroma_store.delete_document(document_id, user_id=user_id)
        bm25_deleted = self.bm25_retriever.delete_document(document_id)
        return chroma_deleted or bm25_deleted

    def clear_all(self, user_id: Optional[str] = None) -> None:
        """Clear all indexed data from both ChromaDB vector store and BM25 index."""
        self.chroma_store.clear(user_id=user_id)
        self.bm25_retriever.clear(user_id=user_id)

    def retrieve(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 10,
        candidate_k: int = 20,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60,
        filter_doc_id: Optional[str] = None,
        min_score: float = 0.0,
        rerank: bool = False,
        user_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Execute multi-mode retrieval over research paper chunks."""
        if not query or not query.strip():
            return []

        clean_mode = mode.lower().strip()
        fetch_k = max(top_k, candidate_k) if rerank else top_k

        # 1. Dense-only mode
        if clean_mode in ["dense", "vector"]:
            candidates = self._retrieve_dense(
                query=query,
                top_k=fetch_k,
                filter_doc_id=filter_doc_id,
                min_score=min_score,
                user_id=user_id,
            )
            if rerank:
                return self.reranker.rerank(query=query, candidates=candidates, top_k=top_k)
            return candidates[:top_k]

        # 2. Sparse-only mode (BM25)
        if clean_mode in ["sparse", "bm25", "keyword"]:
            candidates = self.bm25_retriever.search(
                query=query,
                n_results=fetch_k,
                filter_doc_id=filter_doc_id,
                min_score=min_score,
                user_id=user_id,
            )
            if rerank:
                return self.reranker.rerank(query=query, candidates=candidates, top_k=top_k)
            return candidates[:top_k]

        # 3. Dense search candidates
        dense_candidates = self.chroma_store.search(
            query=query,
            n_results=fetch_k,
            filter_doc_id=filter_doc_id,
            min_score=0.0,
            user_id=user_id,
        )

        # 4. Sparse search candidates
        sparse_candidates = self.bm25_retriever.search(
            query=query,
            n_results=fetch_k,
            filter_doc_id=filter_doc_id,
            min_score=0.0,
            user_id=user_id,
        )

        # 5. Weighted Score Fusion mode
        if clean_mode == "weighted":
            candidates = weighted_score_fusion(
                dense_results=dense_candidates,
                sparse_results=sparse_candidates,
                dense_weight=dense_weight,
                sparse_weight=sparse_weight,
                top_k=fetch_k,
            )
            if rerank:
                return self.reranker.rerank(query=query, candidates=candidates, top_k=top_k)
            return candidates[:top_k]

        # 6. Default: Hybrid Reciprocal Rank Fusion (RRF)
        candidates = reciprocal_rank_fusion(
            dense_results=dense_candidates,
            sparse_results=sparse_candidates,
            k=rrf_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            top_k=fetch_k,
        )

        if rerank:
            return self.reranker.rerank(query=query, candidates=candidates, top_k=top_k)
        return candidates[:top_k]

    def _retrieve_dense(
        self,
        query: str,
        top_k: int = 10,
        filter_doc_id: Optional[str] = None,
        min_score: float = 0.0,
        user_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """Internal helper for pure dense retrieval."""
        raw_results = self.chroma_store.search(
            query=query,
            n_results=top_k,
            filter_doc_id=filter_doc_id,
            min_score=min_score,
            user_id=user_id,
        )

        results: List[RetrievalResult] = []
        for rank, item in enumerate(raw_results, start=1):
            res = RetrievalResult(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                filename=item.filename,
                page_number=item.page_number,
                section=item.section,
                chunk_index=item.chunk_index,
                text=item.text,
                score=item.score,
                dense_score=item.score,
                dense_rank=rank,
                retrieval_method="dense",
                metadata={"distance": item.distance},
            )
            results.append(res)
        return results

    def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Return combined status and counts across dense and sparse engines."""
        chroma_stats = self.chroma_store.get_collection_stats(user_id=user_id)
        bm25_chunks_count = len([c for c in self.bm25_retriever.chunks if not user_id or self.bm25_retriever._chunk_to_user.get(c.chunk_id, user_id) == user_id])

        return {
            "chroma_vectors": chroma_stats["total_vectors"],
            "chroma_documents": chroma_stats["indexed_documents_count"],
            "bm25_chunks": bm25_chunks_count,
            "bm25_documents": len(self.bm25_retriever._doc_id_to_chunks),
            "embedding_model": chroma_stats["embedding_model"],
            "vector_dimension": chroma_stats["vector_dimension"],
            "status": "Hybrid Pipeline Ready",
        }



def get_hybrid_retriever(chroma_store: Optional[ChromaStore] = None) -> HybridRetriever:
    """Convenience accessor to get singleton HybridRetriever."""
    return HybridRetriever.get_instance(chroma_store=chroma_store)
