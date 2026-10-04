"""Document data models for ResearchMate.

Provides structured representations for processed research papers,
preserving document metadata and page-level extracted content.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field


class Page(BaseModel):
    """Represents a single extracted page from a research paper."""

    page_number: int = Field(
        ...,
        description="1-indexed page number from the original document",
        ge=1,
    )
    text: str = Field(
        ...,
        description="Extracted plain text content of the page",
    )
    char_count: int = Field(
        default=0,
        description="Total character count in the page text",
        ge=0,
    )
    word_count: int = Field(
        default=0,
        description="Total word count in the page text",
        ge=0,
    )

    def model_post_init(self, __context: object) -> None:
        """Calculate counts automatically if not provided or 0."""
        if not self.char_count:
            self.char_count = len(self.text)
        if not self.word_count:
            self.word_count = len(self.text.split()) if self.text.strip() else 0


class DocumentMetadata(BaseModel):
    """Metadata associated with a research paper."""

    document_id: str = Field(
        ...,
        description="Unique identifier for the document",
    )
    filename: str = Field(
        ...,
        description="Storage filename on disk",
    )
    original_filename: str = Field(
        ...,
        description="Original uploaded filename",
    )
    title: str = Field(
        default="Untitled Document",
        description="Detected paper title or fallback filename",
    )
    authors: List[str] = Field(
        default_factory=list,
        description="List of detected author names",
    )
    page_count: int = Field(
        default=0,
        description="Number of non-empty pages with extracted text",
        ge=0,
    )
    total_pages_in_pdf: int = Field(
        default=0,
        description="Total number of physical pages in the source PDF",
        ge=0,
    )
    upload_time: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of when the paper was uploaded/processed",
    )
    file_size_bytes: int = Field(
        default=0,
        description="Size of the raw PDF file in bytes",
        ge=0,
    )
    has_extractable_text: bool = Field(
        default=True,
        description="Indicates whether the PDF contains extractable text",
    )
    sha256_hash: str = Field(
        default="",
        description="SHA-256 hash of the PDF file content for integrity and deduplication",
    )
    cloudinary_public_id: Optional[str] = Field(
        default=None,
        description="Cloudinary asset public ID, e.g. researchmate/users/<user_id>/papers/<doc_id>",
    )
    cloudinary_url: Optional[str] = Field(
        default=None,
        description="Secure delivery or storage URL on Cloudinary",
    )
    cloudinary_resource_type: str = Field(
        default="raw",
        description="Cloudinary resource type ('raw' for PDF documents)",
    )
    cloudinary_format: str = Field(
        default="pdf",
        description="Asset format on Cloudinary",
    )
    status: str = Field(
        default="indexed",
        description="Paper processing and indexing lifecycle state (uploaded, processed, indexed, failed)",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Owner user_id for multi-tenant workspace isolation",
    )


class Document(BaseModel):
    """Structured representation of a complete processed research paper."""

    metadata: DocumentMetadata
    pages: List[Page] = Field(default_factory=list)

    @property
    def document_id(self) -> str:
        """Shortcut to document ID."""
        return self.metadata.document_id

    @property
    def title(self) -> str:
        """Shortcut to title."""
        return self.metadata.title

    @property
    def filename(self) -> str:
        """Shortcut to filename."""
        return self.metadata.filename

    @property
    def page_count(self) -> int:
        """Shortcut to number of extracted pages."""
        return len(self.pages)

    @property
    def pdf_url(self) -> Optional[str]:
        """Shortcut to Cloudinary / storage PDF URL."""
        return self.metadata.cloudinary_url

    @pdf_url.setter
    def pdf_url(self, url: Optional[str]) -> None:
        """Setter for Cloudinary / storage PDF URL."""
        self.metadata.cloudinary_url = url

    @property
    def total_words(self) -> int:
        """Calculate total words across all extracted pages."""
        return sum(page.word_count for page in self.pages)

    @property
    def total_chars(self) -> int:
        """Calculate total characters across all extracted pages."""
        return sum(page.char_count for page in self.pages)

    def get_full_text(self, separator: str = "\n\n") -> str:
        """Combine text across all pages with a separator."""
        return separator.join(page.text for page in self.pages)

    def get_page(self, page_number: int) -> Optional[Page]:
        """Retrieve a specific page by its 1-indexed page number."""
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def to_dict(self) -> dict:
        """Serialize document to a dictionary."""
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """Serialize document to formatted JSON string."""
        return json.dumps(self.model_dump(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> Document:
        """Instantiate Document from a dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> Document:
        """Instantiate Document from a JSON string."""
        return cls.model_validate_json(json_str)
