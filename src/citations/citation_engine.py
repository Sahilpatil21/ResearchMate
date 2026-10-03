"""Citation attribution and source snippet extraction engine for ResearchMate.

Ties retrieved research paper passages directly to grounded, verifiable citations
with exact PDF page numbers, section headers, and relevant sentence snippet extraction.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from src.chunking.text_chunker import split_into_sentences
from src.models.citation import Citation
from src.models.document import Document
from src.models.retrieval import RetrievalResult
from src.retrieval.bm25_retriever import tokenize_academic_text


class CitationEngine:
    """Extracts grounded citations and relevant sentence snippets from retrieved chunks."""

    @staticmethod
    def extract_source_snippet(
        query: str,
        text: str,
        max_sentences: int = 2,
    ) -> str:
        """Identify the most relevant sentence or sub-passage within a chunk for a given query.

        Never alters the original chunk source text.

        Args:
            query: User search query.
            text: Full original chunk text.
            max_sentences: Maximum number of key sentences to return.

        Returns:
            Extracted source sentence snippet.
        """
        if not text or not text.strip():
            return ""

        sentences = split_into_sentences(text)
        if len(sentences) <= 1 or not query or not query.strip():
            # Return first sentence or up to 250 characters
            first = sentences[0] if sentences else text.strip()
            return first if len(first) <= 250 else first[:247] + "..."

        query_tokens = set(tokenize_academic_text(query, remove_stopwords=True))
        if not query_tokens:
            query_tokens = set(tokenize_academic_text(query, remove_stopwords=False))

        scored_sentences: List[tuple[float, int, str]] = []

        for idx, sentence in enumerate(sentences):
            sent_tokens = set(tokenize_academic_text(sentence, remove_stopwords=False))
            overlap = len(query_tokens.intersection(sent_tokens))
            
            # Phrase bonus if exact query substrings appear
            phrase_bonus = 1.5 if query.lower() in sentence.lower() else 0.0
            
            # Density score (overlap / sqrt(length))
            length_norm = max(1.0, len(sent_tokens) ** 0.5)
            score = (overlap + phrase_bonus) / length_norm

            scored_sentences.append((score, idx, sentence))

        # Sort descending by relevance score
        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        if not scored_sentences or scored_sentences[0][0] == 0:
            # Fallback to first sentence if no term overlap found
            return sentences[0]

        # Select top scored sentences and sort by original document order
        top_selected = scored_sentences[:max_sentences]
        top_selected.sort(key=lambda x: x[1])  # Preserve chronological reading order

        snippet = " ".join(s[2] for s in top_selected).strip()
        return snippet

    @classmethod
    def create_citations(
        cls,
        results: List[RetrievalResult],
        query: Optional[str] = None,
        document_map: Optional[Dict[str, Document]] = None,
    ) -> List[Citation]:
        """Convert a list of ranked retrieval results into grounded Citation objects.

        Args:
            results: Ordered list of retrieved or reranked results.
            query: Optional query used to extract relevant key sentence snippets.
            document_map: Optional mapping of document_id -> Document for title resolution.

        Returns:
            List of Citation objects with exact page numbers and section attribution.
        """
        citations: List[Citation] = []

        for idx, res in enumerate(results, start=1):
            citation_id = f"[{idx}]"

            # Determine paper title
            paper_title = res.filename
            if document_map and res.document_id in document_map:
                paper_title = document_map[res.document_id].title
            elif res.metadata.get("title"):
                paper_title = res.metadata["title"]

            # Extract focused sentence snippet
            snippet = ""
            if query:
                snippet = cls.extract_source_snippet(query=query, text=res.text)
            elif res.source_snippet:
                snippet = res.source_snippet
            else:
                sents = split_into_sentences(res.text)
                snippet = sents[0] if sents else res.text[:200]

            citation = Citation(
                citation_id=citation_id,
                citation_index=idx,
                document_id=res.document_id,
                filename=res.filename,
                paper_title=paper_title,
                page_number=res.page_number,
                section=res.section if res.section else "General",
                chunk_id=res.chunk_id,
                chunk_index=res.chunk_index,
                source_text=res.text,
                source_snippet=snippet,
                retrieval_rank=idx,
                reranker_score=res.reranker_score,
                dense_score=res.dense_score,
                sparse_score=res.sparse_score,
                rrf_score=res.rrf_score,
                metadata=dict(res.metadata),
            )
            citations.append(citation)

        return citations

    @staticmethod
    def format_citations_markdown(
        citations: List[Citation],
        include_snippets: bool = True,
    ) -> str:
        """Format citations into a Markdown bibliography block.

        Example:
        - **[1]** `resnet_paper.pdf` (Page 2, Section: *2. Deep Residual Architecture*)
          > "We formulate the residual mapping as F(x) + x..."
        """
        if not citations:
            return "_No citations available._"

        lines: List[str] = ["### 📚 Grounded Citations & Sources", ""]
        for cit in citations:
            sec_text = f", Section: *{cit.section}*" if cit.section else ""
            lines.append(f"- **{cit.citation_id}** **{cit.paper_title}** — `{cit.filename}` (Page **{cit.page_number}**{sec_text})")
            if include_snippets and cit.source_snippet:
                clean_snippet = cit.source_snippet.strip().replace("\n", " ")
                lines.append(f"  > *\"{clean_snippet}\"*")
            lines.append("")

        return "\n".join(lines)
