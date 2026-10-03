"""Context builder for ResearchMate RAG pipeline.

Constructs grounded, structured academic context from retrieved and reranked chunks.
Ensures every context block contains citation identifiers, paper titles, filenames,
exact page numbers, sections, and relevance scores.
"""

from __future__ import annotations

from typing import List, Optional

from src.models.citation import Citation
from src.models.retrieval import RetrievalResult


class ContextBuilder:
    """Builds structured context prompts strictly from retrieved research paper passages."""

    def __init__(self, max_context_chars: int = 16000) -> None:
        """Initialize the ContextBuilder.

        Args:
            max_context_chars: Safety cap on total context characters to prevent token overflow.
        """
        self.max_context_chars = max_context_chars

    def build_context_from_citations(
        self,
        citations: List[Citation],
        max_chars: Optional[int] = None,
    ) -> str:
        """Format a list of grounded Citation objects into structured LLM context blocks.

        Args:
            citations: List of Citation items.
            max_chars: Optional override for character limit.

        Returns:
            Formatted context string with [Source 1], [Source 2], etc.
        """
        if not citations:
            return ""

        limit = max_chars or self.max_context_chars
        context_blocks: List[str] = []
        current_len = 0

        for cit in citations:
            # Build individual source block
            score_info = []
            if cit.reranker_score is not None:
                score_info.append(f"Reranker Score: {cit.reranker_score:.4f}")
            elif cit.rrf_score is not None:
                score_info.append(f"RRF Score: {cit.rrf_score:.4f}")
            elif cit.dense_score is not None:
                score_info.append(f"Dense Similarity: {cit.dense_score:.4f}")

            score_line = f" ({', '.join(score_info)})" if score_info else ""

            block = (
                f"[Source {cit.citation_index}]\n"
                f"Paper: {cit.paper_title}\n"
                f"Filename: {cit.filename}\n"
                f"Page: {cit.page_number}\n"
                f"Section: {cit.section}{score_line}\n\n"
                f"Source text:\n{cit.source_text.strip()}\n"
            )

            block_len = len(block)
            if current_len + block_len > limit and context_blocks:
                # If adding this block exceeds limit, break or trim
                remaining_budget = limit - current_len - 150
                if remaining_budget > 200:
                    truncated_text = cit.source_text.strip()[:remaining_budget] + "... [truncated]"
                    trimmed_block = (
                        f"[Source {cit.citation_index}]\n"
                        f"Paper: {cit.paper_title}\n"
                        f"Filename: {cit.filename}\n"
                        f"Page: {cit.page_number}\n"
                        f"Section: {cit.section}{score_line}\n\n"
                        f"Source text:\n{truncated_text}\n"
                    )
                    context_blocks.append(trimmed_block)
                break

            context_blocks.append(block)
            current_len += block_len

        return "\n---\n\n".join(context_blocks)

    def build_context_from_results(
        self,
        results: List[RetrievalResult],
        max_chars: Optional[int] = None,
    ) -> str:
        """Format raw RetrievalResult items into context blocks."""
        from src.citations.citation_engine import CitationEngine
        citations = CitationEngine.create_citations(results)
        return self.build_context_from_citations(citations, max_chars=max_chars)
