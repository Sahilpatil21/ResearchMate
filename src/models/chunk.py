"""Chunk data models for ResearchMate.

Defines representations for text chunks extracted from research papers,
preserving source attribution (document_id, original filename, page number, section, chunk index).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata attributes associated with a single text chunk."""

    chunk_id: str = Field(
        ...,
        description="Unique identifier for the chunk, e.g. chk_doc123_0001",
    )
    document_id: str = Field(
        ...,
        description="ID of the parent document",
    )
    filename: str = Field(
        ...,
        description="Storage or original filename of the document",
    )
    page_number: int = Field(
        ...,
        description="1-indexed source PDF page number where this chunk originates",
        ge=1,
    )
    page_numbers: List[int] = Field(
        default_factory=list,
        description="All 1-indexed page numbers spanned by this chunk",
    )
    section: Optional[str] = Field(
        default=None,
        description="Detected academic section title (e.g. '1. Introduction', 'Methodology')",
    )
    chunk_index: int = Field(
        ...,
        description="0-indexed sequential position of the chunk in the document",
        ge=0,
    )
    char_count: int = Field(
        default=0,
        description="Total character count in the chunk text",
        ge=0,
    )
    token_count: Optional[int] = Field(
        default=None,
        description="Estimated or exact token count of the chunk text",
    )
    strategy: str = Field(
        default="section_aware",
        description="Chunking strategy used to generate this chunk",
    )

    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert metadata to flat dictionary suitable for ChromaDB metadata storage.

        Chroma requires metadata values to be primitive types: str, int, float, bool.
        """
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "section": self.section or "General",
            "chunk_index": self.chunk_index,
            "char_count": self.char_count,
            "token_count": self.token_count if self.token_count is not None else -1,
            "strategy": self.strategy,
        }


class Chunk(BaseModel):
    """Represents a discrete, searchable chunk of text with metadata."""

    text: str = Field(
        ...,
        description="Cleaned text content of the chunk",
    )
    metadata: ChunkMetadata = Field(
        ...,
        description="Source and indexing metadata",
    )

    def model_post_init(self, __context: object) -> None:
        """Ensure character count and page numbers are populated."""
        if not self.metadata.char_count:
            self.metadata.char_count = len(self.text)
        if not self.metadata.page_numbers:
            self.metadata.page_numbers = [self.metadata.page_number]

    @property
    def chunk_id(self) -> str:
        """Shortcut to chunk_id."""
        return self.metadata.chunk_id

    @property
    def document_id(self) -> str:
        """Shortcut to parent document ID."""
        return self.metadata.document_id

    @property
    def filename(self) -> str:
        """Shortcut to filename."""
        return self.metadata.filename

    @property
    def page_number(self) -> int:
        """Shortcut to primary page number."""
        return self.metadata.page_number

    @property
    def section(self) -> Optional[str]:
        """Shortcut to detected section header."""
        return self.metadata.section

    @property
    def chunk_index(self) -> int:
        """Shortcut to sequential chunk index."""
        return self.metadata.chunk_index

    def to_dict(self) -> Dict[str, Any]:
        """Serialize chunk to a dictionary."""
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """Serialize chunk to formatted JSON string."""
        return json.dumps(self.model_dump(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Chunk:
        """Instantiate Chunk from a dictionary."""
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, json_str: str) -> Chunk:
        """Instantiate Chunk from a JSON string."""
        return cls.model_validate_json(json_str)
