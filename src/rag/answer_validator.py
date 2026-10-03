"""Answer validation and hallucination protection module for ResearchMate RAG.

Validates that LLM generated answers adhere to grounding constraints:
- Verifies inline citation tags exist and map to retrieved sources
- Detects non-existent or hallucinated citation IDs
- Flags answers missing citations or claiming unsupported facts
- Detects proper insufficient-evidence fallback phrasing
"""

from __future__ import annotations

import re
from typing import List, Optional, Set
from pydantic import BaseModel, Field

from src.models.citation import Citation


class ValidationResult(BaseModel):
    """Structured validation report for a generated RAG answer."""

    is_valid: bool = Field(..., description="True if answer satisfies grounding rules")
    has_citations: bool = Field(..., description="True if answer includes inline citations")
    cited_indices: List[int] = Field(default_factory=list, description="All citation indices found in text")
    valid_citations: List[int] = Field(default_factory=list, description="Citation indices that exist in retrieved sources")
    invalid_citations: List[int] = Field(default_factory=list, description="Citation indices that do not exist in retrieved sources")
    is_insufficient_evidence: bool = Field(default=False, description="True if model expressed lack of sufficient evidence")
    grounding_score: float = Field(default=1.0, description="Ratio of valid citations to total citations (0.0 to 1.0)")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings or grounding discrepancies")


class AnswerValidator:
    """Validates citations, grounding, and hallucination protection in LLM responses."""

    INSUFFICIENT_EVIDENCE_PHRASES = [
        "do not contain enough information",
        "does not contain enough information",
        "not enough information",
        "insufficient information",
        "no information provided",
        "not mentioned in the provided",
        "cannot answer this question based on the provided",
        "provided papers do not contain",
        "provided context does not contain",
        "no evidence found in the provided",
    ]

    CITATION_PATTERN = re.compile(r"\[(?:Source\s*)?(\d+)(?:,\s*(?:Source\s*)?(\d+))*\]", re.IGNORECASE)

    @classmethod
    def extract_citation_indices(cls, text: str) -> List[int]:
        """Extract all numeric citation indices from text (e.g. '[1]', '[1, 2]', '[Source 3]').

        Args:
            text: Generated response text.

        Returns:
            Sorted list of unique integer citation indices.
        """
        if not text:
            return []

        indices: Set[int] = set()

        # Find all bracketed tokens like [1], [2], [1, 2], [Source 1]
        matches = re.findall(r"\[([0-9,\s\w]+)\]", text)
        for m in matches:
            # Check if this match is numeric or source-prefixed
            parts = m.replace("Source", "").replace("source", "").split(",")
            for p in parts:
                p_clean = p.strip()
                if p_clean.isdigit():
                    indices.add(int(p_clean))

        return sorted(list(indices))

    @classmethod
    def check_insufficient_evidence(cls, text: str) -> bool:
        """Check if the response indicates insufficient evidence in the provided papers.

        Args:
            text: LLM response text.

        Returns:
            True if text contains standard insufficient-evidence phrasing.
        """
        if not text:
            return False

        lower_text = text.lower()
        return any(phrase in lower_text for phrase in cls.INSUFFICIENT_EVIDENCE_PHRASES)

    @classmethod
    def validate_answer(
        cls,
        answer_text: str,
        citations: List[Citation],
        allow_uncited_insufficient: bool = True,
    ) -> ValidationResult:
        """Validate LLM generated answer against the retrieved citations.

        Args:
            answer_text: Generated response string.
            citations: List of retrieved Citation objects available to the LLM.
            allow_uncited_insufficient: If True, allows un-cited answers if the model declares insufficient evidence.

        Returns:
            ValidationResult with detailed breakdown.
        """
        warnings: List[str] = []
        is_insufficient = cls.check_insufficient_evidence(answer_text)
        cited_indices = cls.extract_citation_indices(answer_text)

        num_sources = len(citations)
        valid_indices: List[int] = []
        invalid_indices: List[int] = []

        for idx in cited_indices:
            if 1 <= idx <= num_sources:
                valid_indices.append(idx)
            else:
                invalid_indices.append(idx)

        has_citations = len(cited_indices) > 0

        # Compute grounding score
        if cited_indices:
            grounding_score = round(len(valid_indices) / len(cited_indices), 3)
        elif is_insufficient:
            grounding_score = 1.0
        else:
            grounding_score = 0.0

        # Check for invalid/hallucinated citation IDs
        if invalid_indices:
            warnings.append(
                f"Hallucinated citation IDs detected: {invalid_indices}. "
                f"Only sources [1] through [{num_sources}] were provided."
            )

        # Check if citations are missing when evidence was provided
        if not has_citations and not is_insufficient and num_sources > 0:
            warnings.append("Answer contains no inline citations to support factual claims.")

        # Overall validity check
        is_valid = True
        if invalid_indices:
            is_valid = False
        elif not has_citations and not (is_insufficient and allow_uncited_insufficient) and num_sources > 0:
            is_valid = False

        return ValidationResult(
            is_valid=is_valid,
            has_citations=has_citations,
            cited_indices=cited_indices,
            valid_citations=valid_indices,
            invalid_citations=invalid_indices,
            is_insufficient_evidence=is_insufficient,
            grounding_score=grounding_score,
            warnings=warnings,
        )
