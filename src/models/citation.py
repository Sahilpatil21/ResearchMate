"""Citation attribution models for ResearchMate.

Provides verifiable, grounded citation representations tying retrieved information
directly to source research papers, exact page numbers, academic sections, and relevance scores.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Represents a grounded, verifiable citation linked to an extracted research paper passage."""

    citation_id: str = Field(
        ...,
        description="Human-readable citation label, e.g. '[1]', '[2]'",
    )
    citation_index: int = Field(
        ...,
        description="1-indexed numeric order of the citation",
        ge=1,
    )
    document_id: str = Field(
        ...,
        description="Unique identifier of the cited source document",
    )
    filename: str = Field(
        ...,
        description="Original or storage PDF filename",
    )
    paper_title: str = Field(
        default="Untitled Document",
        description="Title of the cited research paper",
    )
    page_number: int = Field(
        ...,
        description="Exact 1-indexed page number in the source PDF",
        ge=1,
    )
    section: str = Field(
        default="General",
        description="Academic section where the passage originates (e.g. 'Methodology', 'Results')",
    )
    chunk_id: str = Field(
        ...,
        description="Unique chunk identifier",
    )
    chunk_index: int = Field(
        ...,
        description="Sequential index of the chunk within the document",
        ge=0,
    )
    source_text: str = Field(
        ...,
        description="Complete original chunk text content without alterations",
    )
    source_snippet: str = Field(
        default="",
        description="Most relevant key sentence or extracted sub-passage for the query",
    )
    retrieval_rank: int = Field(
        default=1,
        description="Final ranking position among retrieved results",
        ge=1,
    )
    reranker_score: Optional[float] = Field(
        default=None,
        description="Cross-encoder relevance score if reranked",
    )
    dense_score: Optional[float] = Field(
        default=None,
        description="Dense vector cosine similarity score",
    )
    sparse_score: Optional[float] = Field(
        default=None,
        description="BM25 keyword match score",
    )
    rrf_score: Optional[float] = Field(
        default=None,
        description="Reciprocal Rank Fusion score",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional citation metadata or provenance attributes",
    )

    def to_formatted_citation(self, include_snippet: bool = False) -> str:
        """Generate standard human-readable citation string.

        Format: '[1] filename.pdf — Page 4 — Methodology'
        """
        sec = self.section if self.section else "General"
        base = f"{self.citation_id} {self.filename} — Page {self.page_number} — {sec}"
        if include_snippet and self.source_snippet:
            clean_snip = self.source_snippet.strip().replace("\n", " ")
            if len(clean_snip) > 140:
                clean_snip = clean_snip[:137] + "..."
            return f'{base}\n    "{clean_snip}"'
        return base

    def to_dict(self) -> Dict[str, Any]:
        """Serialize citation to a dictionary."""
        return self.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """Serialize citation to formatted JSON string."""
        return json.dumps(self.model_dump(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Citation:
        """Instantiate Citation from a dictionary."""
        return cls.model_validate(data)
