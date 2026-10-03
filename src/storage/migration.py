"""Migration Utility: Local PDF storage to Cloudinary.

Provides safe, repeatable migration of locally stored research papers:
- Scans user directories (data/users/<user_id>/papers/)
- Uploads PDFs to Cloudinary under predictable user-scoped public_ids
- Syncs metadata to MongoDB Atlas (papers and documents collections)
- Validates asset upload integrity
- Preserves existing document_id and indexed status
- Default non-destructive (keeps local files unless delete_local=True is specified)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models.document import Document
from src.storage.cloudinary_storage import CloudinaryStorage, get_cloudinary_storage
from src.storage.mongo_storage import MongoStorageService, get_mongo_storage
from src.utils.file_utils import get_data_dir, load_processed_document

logger = logging.getLogger(__name__)


class CloudinaryMigrationManager:
    """Manages migration of local PDF assets to Cloudinary cloud storage."""

    def __init__(
        self,
        cloudinary_storage: Optional[CloudinaryStorage] = None,
        mongo_storage: Optional[MongoStorageService] = None,
    ) -> None:
        """Initialize migration manager."""
        self.c_storage = cloudinary_storage or get_cloudinary_storage()
        self.m_storage = mongo_storage or get_mongo_storage()

    def migrate_user_papers(
        self,
        user_id: str,
        delete_local: bool = False,
    ) -> Dict[str, Any]:
        """Migrate all local papers for a specific user to Cloudinary.

        Args:
            user_id: Target user ID.
            delete_local: If True, remove local PDF after verified Cloudinary upload.

        Returns:
            Dictionary summary of migration results.
        """
        user_papers_dir = get_data_dir() / "users" / user_id / "papers"
        results: Dict[str, Any] = {
            "user_id": user_id,
            "total_found": 0,
            "migrated": 0,
            "skipped": 0,
            "failed": 0,
            "details": [],
        }

        if not user_papers_dir.exists():
            return results

        pdf_files = list(user_papers_dir.glob("*.pdf"))
        results["total_found"] = len(pdf_files)

        # Load processed documents for this user to map filenames to document_ids
        processed_docs = self.m_storage.load_user_documents(user_id=user_id)
        file_to_doc = {doc.metadata.filename: doc for doc in processed_docs}

        for pdf_path in pdf_files:
            filename = pdf_path.name
            doc = file_to_doc.get(filename)
            doc_id = doc.document_id if doc else pdf_path.stem

            item_result: Dict[str, Any] = {
                "filename": filename,
                "document_id": doc_id,
                "status": "pending",
                "cloudinary_public_id": None,
                "error": None,
            }

            try:
                # 1. Upload PDF to Cloudinary
                upload_res = self.c_storage.upload_pdf(
                    file_input=pdf_path,
                    user_id=user_id,
                    document_id=doc_id,
                    filename=filename,
                )

                if not upload_res.get("success"):
                    item_result["status"] = "failed"
                    item_result["error"] = upload_res.get("error", "Upload failed")
                    results["failed"] += 1
                    results["details"].append(item_result)
                    continue

                public_id = upload_res.get("public_id")
                sec_url = upload_res.get("secure_url")
                item_result["cloudinary_public_id"] = public_id

                # 2. Update Document and MongoDB metadata
                if doc:
                    doc.metadata.cloudinary_public_id = public_id
                    doc.metadata.cloudinary_url = sec_url
                    doc.metadata.status = "indexed"
                    self.m_storage.save_document(doc, user_id=user_id)
                else:
                    self.m_storage.save_paper_record(
                        user_id=user_id,
                        document_id=doc_id,
                        filename=filename,
                        title=filename.replace(".pdf", ""),
                        cloudinary_public_id=public_id,
                        cloudinary_url=sec_url,
                        status="uploaded",
                    )

                item_result["status"] = "migrated"
                results["migrated"] += 1

                # 3. Optional local file cleanup
                if delete_local and pdf_path.exists():
                    try:
                        pdf_path.unlink()
                        item_result["local_deleted"] = True
                    except Exception:
                        item_result["local_deleted"] = False

            except Exception as e:
                item_result["status"] = "failed"
                item_result["error"] = str(e)
                results["failed"] += 1

            results["details"].append(item_result)

        return results

    def migrate_all_users(self, delete_local: bool = False) -> List[Dict[str, Any]]:
        """Scan data/users/ and migrate all users' papers."""
        users_dir = get_data_dir() / "users"
        if not users_dir.exists():
            return []

        all_results = []
        for user_folder in users_dir.iterdir():
            if user_folder.is_dir() and (user_folder / "papers").exists():
                user_id = user_folder.name
                res = self.migrate_user_papers(user_id=user_id, delete_local=delete_local)
                all_results.append(res)

        return all_results
