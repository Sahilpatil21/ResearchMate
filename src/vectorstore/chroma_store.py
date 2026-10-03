"""Persistent ChromaDB vector storage and semantic retrieval for ResearchMate.

Stores 384-dimensional dense embeddings with Cosine similarity space,
supporting per-document chunk indexing, filtered retrieval, document removal,
and multi-tenant user isolation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import chromadb
from chromadb.config import Settings
from pydantic import BaseModel, Field

from src.chunking.text_chunker import TextChunker
from src.embeddings.embedding_model import EmbeddingModel, get_embedding_model
from src.models.chunk import Chunk
from src.models.document import Document
from src.utils.file_utils import get_chroma_dir, save_document_chunks


class SearchResult(BaseModel):
    """Encapsulates a single vector search match."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    section: str
    chunk_index: int
    text: str
    score: float
    distance: float


class ChromaStore:
    """Persistent ChromaDB vector database manager with multi-tenant isolation."""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        collection_name: str = "research_papers",
        embedding_model: Optional[EmbeddingModel] = None,
    ) -> None:
        """Initialize persistent ChromaDB client and collection."""
        self.persist_dir = persist_dir or get_chroma_dir()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model = embedding_model or get_embedding_model()

        # Initialize persistent ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False, is_persistent=True),
        )

        # Get or create collection with cosine distance metric
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: List[Chunk],
        overwrite: bool = False,
        user_id: Optional[str] = None,
    ) -> int:
        """Add a list of chunks to the vector database."""
        if not chunks:
            return 0

        doc_id = chunks[0].document_id

        # Check duplicate
        if self.is_document_indexed(doc_id, user_id=user_id):
            if not overwrite:
                raise ValueError(
                    f"Document '{doc_id}' is already indexed in the vector database. "
                    "Set overwrite=True to re-index."
                )
            self.delete_document(doc_id, user_id=user_id)

        # Compute embeddings in batch
        texts = [c.text for c in chunks]
        embeddings = self.embedding_model.embed_texts(texts)

        # Prepare payloads
        ids = [c.chunk_id for c in chunks]
        metadatas = []
        for c in chunks:
            m = c.metadata.to_flat_dict()
            if user_id:
                m["user_id"] = user_id
            metadatas.append(m)

        # Insert into ChromaDB collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        return len(chunks)

    def index_document(
        self,
        document: Document,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        overwrite: bool = False,
        user_id: Optional[str] = None,
    ) -> Tuple[int, List[Chunk]]:
        """Chunk, embed, and index a Document into ChromaDB and disk storage."""
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_document(document)

        if not chunks:
            return 0, []

        count = self.add_chunks(chunks, overwrite=overwrite, user_id=user_id)
        save_document_chunks(document.document_id, chunks, user_id=user_id)

        return count, chunks

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_doc_id: Optional[str] = None,
        min_score: float = 0.0,
        user_id: Optional[str] = None,
    ) -> List[SearchResult]:
        """Perform dense semantic search across indexed research paper chunks."""
        if not query or not query.strip():
            return []

        if self.collection.count() == 0:
            return []

        query_vector = self.embedding_model.embed_query(query)

        # Build filter conditions
        filters = []
        if filter_doc_id:
            filters.append({"document_id": filter_doc_id})
        if user_id:
            filters.append({"user_id": user_id})

        where_filter = None
        if len(filters) > 1:
            where_filter = {"$and": filters}
        elif len(filters) == 1:
            where_filter = filters[0]

        total_vectors = self.collection.count()
        fetch_k = min(max(1, n_results), total_vectors)

        raw_results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=fetch_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        search_results: List[SearchResult] = []

        ids_list = raw_results.get("ids", [[]])[0]
        docs_list = raw_results.get("documents", [[]])[0]
        metas_list = raw_results.get("metadatas", [[]])[0]
        dists_list = raw_results.get("distances", [[]])[0]

        for chunk_id, text, meta, dist in zip(ids_list, docs_list, metas_list, dists_list):
            distance = float(dist)
            similarity = max(0.0, min(1.0, 1.0 - distance))

            if similarity < min_score:
                continue

            result = SearchResult(
                chunk_id=chunk_id,
                document_id=meta.get("document_id", ""),
                filename=meta.get("filename", "unknown.pdf"),
                page_number=int(meta.get("page_number", 1)),
                section=meta.get("section", "General"),
                chunk_index=int(meta.get("chunk_index", 0)),
                text=text,
                score=round(similarity, 4),
                distance=round(distance, 4),
            )
            search_results.append(result)

        search_results.sort(key=lambda r: r.score, reverse=True)
        return search_results

    def delete_document(self, document_id: str, user_id: Optional[str] = None) -> bool:
        """Delete all vectors belonging to a specific document ID."""
        if not self.is_document_indexed(document_id, user_id=user_id):
            return False

        try:
            if user_id:
                self.collection.delete(where={"$and": [{"document_id": document_id}, {"user_id": user_id}]})
            else:
                self.collection.delete(where={"document_id": document_id})
            return True
        except Exception:
            return False

    def clear(self, user_id: Optional[str] = None) -> None:
        """Clear all vectors or all vectors belonging to a specific user."""
        try:
            if user_id:
                self.collection.delete(where={"user_id": user_id})
            else:
                self.client.delete_collection(name=self.collection_name)
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
        except Exception:
            pass

    def get_indexed_document_ids(self, user_id: Optional[str] = None) -> List[str]:
        """Return list of distinct document IDs present in the vector store."""
        if self.collection.count() == 0:
            return []

        where_filter = {"user_id": user_id} if user_id else None
        data = self.collection.get(where=where_filter, include=["metadatas"])
        metadatas = data.get("metadatas", [])
        doc_ids = set()
        for meta in metadatas:
            if meta and "document_id" in meta:
                doc_ids.add(meta["document_id"])
        return sorted(list(doc_ids))

    def is_document_indexed(self, document_id: str, user_id: Optional[str] = None) -> bool:
        """Check whether a specific document has vectors in the collection."""
        if self.collection.count() == 0:
            return False

        if user_id:
            where_filter = {"$and": [{"document_id": document_id}, {"user_id": user_id}]}
        else:
            where_filter = {"document_id": document_id}

        results = self.collection.get(
            where=where_filter,
            limit=1,
            include=["metadatas"],
        )
        return len(results.get("ids", [])) > 0

    def get_document_chunk_count(self, document_id: str, user_id: Optional[str] = None) -> int:
        """Get the count of stored chunks for a given document ID."""
        if self.collection.count() == 0:
            return 0

        if user_id:
            where_filter = {"$and": [{"document_id": document_id}, {"user_id": user_id}]}
        else:
            where_filter = {"document_id": document_id}

        results = self.collection.get(
            where=where_filter,
            include=["metadatas"],
        )
        return len(results.get("ids", []))

    def list_indexed_documents(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return metadata summaries for indexed documents."""
        if self.collection.count() == 0:
            return []

        where_filter = {"user_id": user_id} if user_id else None
        data = self.collection.get(where=where_filter, include=["metadatas"])
        metadatas = data.get("metadatas", [])

        doc_map: Dict[str, Dict[str, Any]] = {}
        for meta in metadatas:
            if not meta:
                continue
            doc_id = meta.get("document_id")
            if not doc_id:
                continue

            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "document_id": doc_id,
                    "filename": meta.get("filename", "unknown.pdf"),
                    "chunk_count": 0,
                    "sections": set(),
                    "pages": set(),
                }

            doc_map[doc_id]["chunk_count"] += 1
            if meta.get("section"):
                doc_map[doc_id]["sections"].add(meta["section"])
            if meta.get("page_number"):
                doc_map[doc_id]["pages"].add(meta["page_number"])

        summaries = []
        for doc_id, info in doc_map.items():
            summaries.append({
                "document_id": doc_id,
                "filename": info["filename"],
                "chunk_count": info["chunk_count"],
                "sections_detected": sorted(list(info["sections"])),
                "pages_indexed": sorted(list(info["pages"])),
            })

        return summaries

    def get_collection_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Return comprehensive status and health metrics for the vector store."""
        if user_id:
            where_filter = {"user_id": user_id}
            data = self.collection.get(where=where_filter, include=["metadatas"])
            total_vectors = len(data.get("ids", []))
            indexed_ids = sorted(list({m.get("document_id") for m in data.get("metadatas", []) if m and m.get("document_id")}))
        else:
            total_vectors = self.collection.count()
            indexed_ids = self.get_indexed_document_ids()

        return {
            "total_vectors": total_vectors,
            "indexed_documents_count": len(indexed_ids),
            "indexed_document_ids": indexed_ids,
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_dir),
            "embedding_model": self.embedding_model.model_name,
            "vector_dimension": self.embedding_model.dimension,
            "status": "Healthy & Connected",
        }

    def reset_collection(self) -> None:
        """Completely clear and reset the vector collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
