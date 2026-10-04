"""MongoDB Atlas Cloud Storage Service for ResearchMate.

Provides seamless cloud persistence and multi-tenant synchronization for:
- User Document metadata & parsed sections ('documents' collection)
- Text chunks ('chunks' collection)
- Structured 9-field paper summaries ('summaries' collection)
- Multi-paper comparisons ('comparisons' collection)
- Grounded conversational Q&A history ('chat_history' collection)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.models.chunk import Chunk
from src.models.document import Document
from src.utils.file_utils import (
    delete_document as delete_document_local,
    get_processed_dir,
    load_all_processed_documents as load_local_processed_docs,
    load_processed_document as load_local_processed_doc,
    save_document_chunks as save_local_chunks,
    save_processed_document as save_local_processed_doc,
)

load_dotenv()


class MongoStorageService:
    """Cloud persistence service communicating directly with MongoDB Atlas."""

    _instance: Optional[MongoStorageService] = None

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        db_name: Optional[str] = None,
    ) -> None:
        """Initialize MongoStorageService.

        Args:
            mongo_uri: MongoDB Atlas connection string.
            db_name: MongoDB database name.
        """
        self.mongo_uri = mongo_uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.db_name = db_name or os.getenv("MONGODB_DB_NAME", "researchmate_db")
        self.is_connected = False
        self._client = None
        self._db = None

        self._init_connection()

    def _init_connection(self) -> None:
        """Establish connection with MongoDB Atlas."""
        try:
            import pymongo

            client = pymongo.MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=2500,
                connectTimeoutMS=2500,
            )
            # Verify live connection
            client.admin.command("ping")
            self._client = client
            self._db = self._client[self.db_name]

            # Ensure indexes for rapid multi-tenant lookups
            self._db["documents"].create_index([("user_id", 1), ("document_id", 1)], unique=True)
            self._db["papers"].create_index([("user_id", 1), ("document_id", 1)], unique=True)
            self._db["chunks"].create_index([("user_id", 1), ("document_id", 1), ("chunk_id", 1)])
            self._db["summaries"].create_index([("user_id", 1), ("document_id", 1)], unique=True)
            self._db["comparisons"].create_index([("user_id", 1), ("created_at", -1)])
            self._db["chat_history"].create_index([("user_id", 1), ("timestamp", 1)])

            self.is_connected = True
        except Exception:
            self.is_connected = False
            self._client = None
            self._db = None

    @classmethod
    def get_instance(cls) -> MongoStorageService:
        """Singleton accessor."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # -----------------------------------------------------------------
    # PAPERS & DOCUMENTS PERSISTENCE
    # -----------------------------------------------------------------

    def save_paper_record(
        self,
        user_id: str,
        document_id: str,
        filename: str,
        title: str,
        authors: Optional[List[str]] = None,
        cloudinary_public_id: Optional[str] = None,
        cloudinary_url: Optional[str] = None,
        page_count: int = 0,
        file_size_bytes: int = 0,
        status: str = "indexed",
    ) -> bool:
        """Save or update paper asset and Cloudinary reference in 'papers' collection."""
        if self.is_connected and self._db is not None:
            try:
                record = {
                    "document_id": document_id,
                    "user_id": user_id,
                    "filename": filename,
                    "title": title,
                    "authors": authors or [],
                    "cloudinary_public_id": cloudinary_public_id,
                    "cloudinary_url": cloudinary_url,
                    "cloudinary_resource_type": "raw",
                    "cloudinary_format": "pdf",
                    "page_count": page_count,
                    "file_size_bytes": file_size_bytes,
                    "status": status,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                self._db["papers"].replace_one(
                    {"user_id": user_id, "document_id": document_id},
                    record,
                    upsert=True,
                )
                return True
            except Exception:
                return False
        return False

    def get_paper_record(self, user_id: str, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve paper record by user_id and document_id."""
        if self.is_connected and self._db is not None:
            try:
                record = self._db["papers"].find_one({"user_id": user_id, "document_id": document_id})
                if record:
                    record.pop("_id", None)
                    return record
            except Exception:
                pass
        return None

    def get_user_paper_records(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve all paper records for a specific user."""
        if self.is_connected and self._db is not None:
            try:
                cursor = self._db["papers"].find({"user_id": user_id}).sort("updated_at", -1)
                records = []
                for r in cursor:
                    r.pop("_id", None)
                    records.append(r)
                return records
            except Exception:
                pass
        return []

    def update_paper_status(self, user_id: str, document_id: str, status: str) -> bool:
        """Update lifecycle status of a paper (uploaded, processed, indexed, failed)."""
        if self.is_connected and self._db is not None:
            try:
                self._db["papers"].update_one(
                    {"user_id": user_id, "document_id": document_id},
                    {"$set": {"status": status, "updated_at": datetime.now(timezone.utc).isoformat()}},
                )
                return True
            except Exception:
                return False
        return False

    def save_document(self, document: Document, user_id: Optional[str] = None) -> bool:
        """Save document metadata and structure to MongoDB Atlas and local cache."""
        uid = user_id or "default"
        # 1. Always save local cache for offline/instant access
        save_local_processed_doc(document, user_id=user_id)

        # 2. Persist to MongoDB Atlas documents and papers collections
        if self.is_connected and self._db is not None:
            try:
                doc_dict = document.to_dict()
                doc_dict["user_id"] = uid
                doc_dict["document_id"] = document.document_id
                doc_dict["updated_at"] = datetime.now(timezone.utc).isoformat()
                
                self._db["documents"].replace_one(
                    {"user_id": uid, "document_id": document.document_id},
                    doc_dict,
                    upsert=True,
                )

                meta = document.metadata
                self.save_paper_record(
                    user_id=uid,
                    document_id=document.document_id,
                    filename=meta.filename,
                    title=meta.title,
                    authors=meta.authors,
                    cloudinary_public_id=meta.cloudinary_public_id,
                    cloudinary_url=meta.cloudinary_url,
                    page_count=meta.page_count,
                    file_size_bytes=meta.file_size_bytes,
                    status=meta.status or "indexed",
                )
                return True
            except Exception:
                return False
        return False

    def load_document(self, document_id: str, user_id: Optional[str] = None) -> Optional[Document]:
        """Load document by document_id from MongoDB Atlas (or local cache fallback)."""
        if self.is_connected and self._db is not None:
            try:
                query: Dict[str, Any] = {"$or": [{"document_id": document_id}, {"metadata.document_id": document_id}]}
                if user_id:
                    query = {"user_id": user_id, "$or": [{"document_id": document_id}, {"metadata.document_id": document_id}]}
                raw = self._db["documents"].find_one(query)
                if raw:
                    raw.pop("_id", None)
                    raw.pop("user_id", None)
                    raw.pop("updated_at", None)
                    raw.pop("document_id", None)
                    return Document.from_dict(raw)
            except Exception:
                pass

        # Fallback to local cache
        return load_local_processed_doc(document_id, user_id=user_id)

    def load_user_documents(self, user_id: Optional[str] = None) -> List[Document]:
        """Load all documents belonging to a user, seamlessly merging MongoDB Atlas and local disk."""
        doc_map: Dict[str, Document] = {}

        # 1. Load from MongoDB Atlas
        if self.is_connected and self._db is not None:
            try:
                query: Dict[str, Any] = {}
                if user_id:
                    query["user_id"] = user_id
                cursor = self._db["documents"].find(query).sort("metadata.upload_time", -1)
                for item in cursor:
                    item.pop("_id", None)
                    item.pop("user_id", None)
                    item.pop("updated_at", None)
                    item.pop("document_id", None)
                    try:
                        d = Document.from_dict(item)
                        doc_map[d.document_id] = d
                    except Exception:
                        continue
            except Exception:
                pass

        # 2. Merge with local cache / disk
        local_docs = load_local_processed_docs(user_id=user_id)
        for ld in local_docs:
            if ld.document_id not in doc_map:
                doc_map[ld.document_id] = ld
                # Sync missing local doc up to MongoDB Atlas in background
                if self.is_connected and self._db is not None:
                    try:
                        self.save_document(ld, user_id=user_id)
                    except Exception:
                        pass

        # Sort newest first
        all_docs = list(doc_map.values())
        all_docs.sort(key=lambda x: getattr(x.metadata, "upload_time", "") or "", reverse=True)
        return all_docs

    def delete_document(self, document_id: str, user_id: Optional[str] = None) -> bool:
        """Delete document, chunks, and summaries from MongoDB Atlas and local disk."""
        deleted_cloud = False
        if self.is_connected and self._db is not None:
            try:
                query = {"$or": [{"document_id": document_id}, {"metadata.document_id": document_id}]}
                if user_id:
                    query = {"user_id": user_id, "$or": [{"document_id": document_id}, {"metadata.document_id": document_id}]}
                self._db["documents"].delete_many(query)
                self._db["papers"].delete_many(query)
                self._db["chunks"].delete_many(query)
                self._db["summaries"].delete_many(query)
                deleted_cloud = True
            except Exception:
                pass

        deleted_local = delete_document_local(document_id, user_id=user_id)
        return deleted_cloud or deleted_local

    def count_user_documents(self, user_id: Optional[str] = None) -> int:
        """Count total documents for user in MongoDB Atlas."""
        if self.is_connected and self._db is not None:
            try:
                query: Dict[str, Any] = {}
                if user_id:
                    query["user_id"] = user_id
                return self._db["documents"].count_documents(query)
            except Exception:
                pass
        return len(load_local_processed_docs(user_id=user_id))

    # -----------------------------------------------------------------
    # CHUNKS PERSISTENCE
    # -----------------------------------------------------------------

    def save_chunks(self, document_id: str, chunks: List[Chunk], user_id: Optional[str] = None) -> bool:
        """Save chunk batch to MongoDB Atlas and local cache."""
        save_local_chunks(document_id, chunks, user_id=user_id)

        if self.is_connected and self._db is not None and chunks:
            try:
                # Remove existing chunks for this document
                uid = user_id or "default"
                self._db["chunks"].delete_many({"user_id": uid, "document_id": document_id})
                chunk_docs = []
                for c in chunks:
                    cd = c.to_dict()
                    cd["user_id"] = uid
                    cd["document_id"] = document_id
                    chunk_docs.append(cd)
                if chunk_docs:
                    self._db["chunks"].insert_many(chunk_docs)
                return True
            except Exception:
                return False
        return False

    def load_chunks(self, document_id: str, user_id: Optional[str] = None) -> List[Chunk]:
        """Load chunks from MongoDB Atlas, falling back to local storage."""
        if self.is_connected and self._db is not None:
            try:
                uid = user_id or "default"
                cursor = self._db["chunks"].find({"user_id": uid, "document_id": document_id}).sort("chunk_index", 1)
                chunks = []
                for doc in cursor:
                    doc.pop("_id", None)
                    doc.pop("user_id", None)
                    chunks.append(Chunk.from_dict(doc))
                if chunks:
                    return chunks
            except Exception:
                pass
        return load_local_chunks(document_id, user_id=user_id)

    def load_document_chunks(self, document_id: str, user_id: Optional[str] = None) -> List[Chunk]:
        """Alias for load_chunks."""
        return self.load_chunks(document_id=document_id, user_id=user_id)

    # -----------------------------------------------------------------
    # RESEARCH INTELLIGENCE (SUMMARIES & COMPARISONS)
    # -----------------------------------------------------------------

    def save_summary(
        self,
        document_id: str,
        summary_dict: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> bool:
        """Save a structured paper summary to MongoDB Atlas."""
        if self.is_connected and self._db is not None:
            try:
                uid = user_id or "default"
                data = dict(summary_dict)
                data["user_id"] = uid
                data["document_id"] = document_id
                data["saved_at"] = datetime.now(timezone.utc).isoformat()
                self._db["summaries"].replace_one(
                    {"user_id": uid, "document_id": document_id},
                    data,
                    upsert=True,
                )
                return True
            except Exception:
                return False
        return False

    def get_summary(
        self,
        document_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a stored summary from MongoDB Atlas."""
        if self.is_connected and self._db is not None:
            try:
                uid = user_id or "default"
                doc = self._db["summaries"].find_one({"user_id": uid, "document_id": document_id})
                if doc:
                    doc.pop("_id", None)
                    return doc
            except Exception:
                pass
        return None

    def load_summary(self, document_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Alias for get_summary."""
        return self.get_summary(document_id=document_id, user_id=user_id)

    def save_comparison(
        self,
        document_ids: List[str],
        comparison_dict: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> bool:
        """Save a multi-paper comparison matrix to MongoDB Atlas."""
        if self.is_connected and self._db is not None:
            try:
                uid = user_id or "default"
                data = dict(comparison_dict)
                data["user_id"] = uid
                data["document_ids"] = document_ids
                data["created_at"] = datetime.now(timezone.utc).isoformat()
                self._db["comparisons"].insert_one(data)
                return True
            except Exception:
                return False
        return False

    def get_user_comparisons(
        self,
        user_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Retrieve past multi-paper comparisons for a user from MongoDB Atlas."""
        if self.is_connected and self._db is not None:
            try:
                uid = user_id or "default"
                cursor = self._db["comparisons"].find({"user_id": uid}).sort("created_at", -1).limit(limit)
                items = []
                for item in cursor:
                    item.pop("_id", None)
                    items.append(item)
                return items
            except Exception:
                pass
        return []

    # -----------------------------------------------------------------
    # CHAT HISTORY PERSISTENCE
    # -----------------------------------------------------------------

    def save_chat_turn(
        self,
        user_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        filter_document_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> bool:
        """Save a single user/assistant chat turn to MongoDB Atlas."""
        doc_id = document_id or filter_document_id
        if self.is_connected and self._db is not None:
            try:
                entry = {
                    "user_id": user_id,
                    "role": role,
                    "content": content,
                    "citations": citations or [],
                    "document_id": doc_id,
                    "filter_document_id": doc_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self._db["chat_history"].insert_one(entry)
                return True
            except Exception:
                return False
        return False

    def load_user_chat_history(
        self,
        user_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Load conversation turns for a user from MongoDB Atlas."""
        if self.is_connected and self._db is not None:
            try:
                cursor = self._db["chat_history"].find({"user_id": user_id}).sort("timestamp", 1).limit(limit)
                history = []
                for item in cursor:
                    item.pop("_id", None)
                    history.append(item)
                return history
            except Exception:
                pass
        return []

    def clear_user_chat_history(self, user_id: str) -> bool:
        """Clear all chat history turns for a user in MongoDB Atlas."""
        if self.is_connected and self._db is not None:
            try:
                self._db["chat_history"].delete_many({"user_id": user_id})
                return True
            except Exception:
                pass
        return False


def get_mongo_storage() -> MongoStorageService:
    """Singleton getter for MongoStorageService."""
    return MongoStorageService.get_instance()
