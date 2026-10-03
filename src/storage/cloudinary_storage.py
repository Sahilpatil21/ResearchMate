"""Cloudinary Cloud Storage Service for ResearchMate.

Provides secure, user-isolated cloud storage for research paper PDFs:
- Predictable multi-tenant public_id structure: researchmate/users/<user_id>/papers/<document_id>
- Upload, download, secure URL generation, deletion, and asset verification
- Safe credential management (never leaks secrets in logs or UI)
- Graceful fallbacks and comprehensive error handling
"""

from __future__ import annotations

import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Union

from dotenv import load_dotenv

load_dotenv()


class CloudinaryStorage:
    """Enterprise Cloud Storage Client using Cloudinary Python SDK."""

    _instance: Optional[CloudinaryStorage] = None

    def __init__(
        self,
        cloud_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> None:
        """Initialize CloudinaryStorage.

        Args:
            cloud_name: Cloudinary cloud name (env: CLOUDINARY_CLOUD_NAME)
            api_key: Cloudinary API key (env: CLOUDINARY_API_KEY)
            api_secret: Cloudinary API secret (env: CLOUDINARY_API_SECRET)
        """
        self.cloud_name = (cloud_name or os.getenv("CLOUDINARY_CLOUD_NAME", "")).strip()
        self.api_key = (api_key or os.getenv("CLOUDINARY_API_KEY", "")).strip()
        self.api_secret = (api_secret or os.getenv("CLOUDINARY_API_SECRET", "")).strip()
        
        self._is_configured = bool(self.cloud_name and self.api_key and self.api_secret)
        self._cloudinary = None
        self._uploader = None
        self._api = None
        self._utils = None

        if self._is_configured:
            self._configure_sdk()

    def _configure_sdk(self) -> None:
        """Configure official Cloudinary SDK."""
        try:
            import cloudinary
            import cloudinary.api
            import cloudinary.uploader
            import cloudinary.utils

            cloudinary.config(
                cloud_name=self.cloud_name,
                api_key=self.api_key,
                api_secret=self.api_secret,
                secure=True,
            )
            self._cloudinary = cloudinary
            self._uploader = cloudinary.uploader
            self._api = cloudinary.api
            self._utils = cloudinary.utils
            self._is_configured = True
        except Exception:
            self._is_configured = False

    @property
    def is_configured(self) -> bool:
        """Return True if Cloudinary credentials are fully configured."""
        return self._is_configured

    @classmethod
    def get_instance(cls) -> CloudinaryStorage:
        """Singleton accessor."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def sanitize_identifier(identifier: str) -> str:
        """Sanitize user_id or document_id for Cloudinary public_id safety."""
        if not identifier:
            return "unknown"
        cleaned = identifier.replace("..", "_")
        cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", cleaned.strip())
        return cleaned or "unknown"

    def build_public_id(self, user_id: str, document_id: str) -> str:
        """Build predictable user-scoped public_id path.

        Format: researchmate/users/<user_id>/papers/<document_id>
        """
        clean_user = self.sanitize_identifier(user_id)
        clean_doc = self.sanitize_identifier(document_id)
        if not clean_doc.endswith(".pdf"):
            clean_doc = f"{clean_doc}.pdf"
        return f"researchmate/users/{clean_user}/papers/{clean_doc}"

    def upload_pdf(
        self,
        file_input: Union[bytes, BinaryIO, str, Path],
        user_id: str,
        document_id: str,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload a PDF paper to Cloudinary under the user's isolated folder.

        Args:
            file_input: Raw bytes, file-like object, or local Path to the PDF.
            user_id: Current authenticated user ID.
            document_id: Unique ResearchMate document identifier.
            filename: Original PDF filename.

        Returns:
            Dictionary with upload metadata and success flag.
        """
        if not self.is_configured or self._uploader is None:
            return {
                "success": False,
                "error": "Cloudinary is not configured. Set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, and CLOUDINARY_API_SECRET in .env.",
                "public_id": None,
                "url": None,
                "secure_url": None,
            }

        public_id = self.build_public_id(user_id, document_id)

        try:
            upload_target = file_input
            if isinstance(file_input, Path):
                upload_target = str(file_input)

            response = self._uploader.upload(
                upload_target,
                public_id=public_id,
                resource_type="raw",
                overwrite=True,
                tags=["researchmate", f"user_{self.sanitize_identifier(user_id)}"],
                context={
                    "original_filename": filename or "document.pdf",
                    "document_id": document_id,
                    "user_id": user_id,
                },
            )

            return {
                "success": True,
                "public_id": response.get("public_id", public_id),
                "secure_url": response.get("secure_url"),
                "url": response.get("url"),
                "bytes": response.get("bytes", 0),
                "format": response.get("format", "pdf"),
                "resource_type": response.get("resource_type", "raw"),
                "created_at": response.get("created_at"),
                "error": None,
            }

        except Exception as e:
            error_msg = str(e)
            if self.api_secret and self.api_secret in error_msg:
                error_msg = error_msg.replace(self.api_secret, "[REDACTED]")
            return {
                "success": False,
                "error": f"Cloudinary upload failed: {error_msg}",
                "public_id": public_id,
                "url": None,
                "secure_url": None,
            }

    def download_pdf(self, user_id: str, document_id: str) -> Optional[bytes]:
        """Download raw PDF bytes from Cloudinary.

        Args:
            user_id: Owner user ID (for authorization and path lookup).
            document_id: Document ID.

        Returns:
            PDF bytes or None if download fails.
        """
        if not self.is_configured:
            return None

        public_id = self.build_public_id(user_id, document_id)

        try:
            url = self.generate_secure_url(user_id, document_id)
            if not url:
                return None

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ResearchMate-Storage/1.0"},
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                return response.read()

        except Exception:
            return None

    def download_pdf_to_temp_file(self, user_id: str, document_id: str) -> Optional[Path]:
        """Download PDF from Cloudinary to a temporary local file."""
        pdf_bytes = self.download_pdf(user_id, document_id)
        if not pdf_bytes:
            return None

        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                return Path(tmp.name)
        except Exception:
            return None

    def generate_secure_url(self, user_id: str, document_id: str, expires_in: int = 3600) -> Optional[str]:
        """Generate a secure delivery or download URL for a user's PDF.

        Args:
            user_id: Owner user ID.
            document_id: Document ID.
            expires_in: Expiration in seconds for signed links.

        Returns:
            Secure HTTPS URL string or None.
        """
        if not self.is_configured or self._utils is None:
            return None

        public_id = self.build_public_id(user_id, document_id)

        try:
            url, _ = self._utils.cloudinary_url(
                public_id,
                resource_type="raw",
                secure=True,
                sign_url=True,
            )
            return url
        except Exception:
            return f"https://res.cloudinary.com/{self.cloud_name}/raw/upload/{public_id}"

    def delete_pdf(self, user_id: str, document_id: str) -> bool:
        """Delete PDF asset from Cloudinary.

        Args:
            user_id: Owner user ID.
            document_id: Document ID.

        Returns:
            True if deleted successfully, False otherwise.
        """
        if not self.is_configured or self._uploader is None:
            return False

        public_id = self.build_public_id(user_id, document_id)

        try:
            resp = self._uploader.destroy(public_id, resource_type="raw")
            return resp.get("result") in ["ok", "not found"]
        except Exception:
            return False

    def asset_exists(self, user_id: str, document_id: str) -> bool:
        """Verify whether the PDF asset exists in Cloudinary."""
        if not self.is_configured or self._api is None:
            return False

        public_id = self.build_public_id(user_id, document_id)

        try:
            self._api.resource(public_id, resource_type="raw")
            return True
        except Exception:
            return False

    def get_asset_metadata(self, user_id: str, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve Cloudinary asset details and metadata."""
        if not self.is_configured or self._api is None:
            return None

        public_id = self.build_public_id(user_id, document_id)

        try:
            res = self._api.resource(public_id, resource_type="raw")
            return {
                "public_id": res.get("public_id"),
                "bytes": res.get("bytes"),
                "format": res.get("format"),
                "secure_url": res.get("secure_url"),
                "created_at": res.get("created_at"),
            }
        except Exception:
            return None

    def __repr__(self) -> str:
        """Safe representation without leaking api_secret."""
        status = "configured" if self.is_configured else "unconfigured"
        return f"<CloudinaryStorage cloud_name='{self.cloud_name}' status='{status}'>"


def get_cloudinary_storage() -> CloudinaryStorage:
    """Singleton getter for CloudinaryStorage."""
    return CloudinaryStorage.get_instance()
