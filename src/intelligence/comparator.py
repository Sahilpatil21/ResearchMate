"""Multi-paper comparison engine for ResearchMate.

Allows comparative analysis across 2 to 5 research papers across 10 academic dimensions:
1. Research Problem
2. Objective
3. Methodology
4. Architecture/Model
5. Dataset
6. Experimental Setup
7. Evaluation Metrics
8. Results
9. Limitations
10. Conclusions
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.citations.citation_engine import CitationEngine
from src.intelligence.summarizer import PaperSummarizer, PaperSummary
from src.llm.factory import get_llm_provider
from src.llm.provider import BaseLLMProvider
from src.models.citation import Citation
from src.rag.context_builder import ContextBuilder
from src.retrieval.hybrid_retriever import HybridRetriever, get_hybrid_retriever
from src.utils.file_utils import load_processed_document


class PaperComparisonReport(BaseModel):
    """Structured comparative analysis across 2 to 5 research papers."""

    document_ids: List[str] = Field(..., description="Document identifiers compared")
    paper_titles: List[str] = Field(..., description="Paper titles")
    comparison_table: Dict[str, Dict[str, str]] = Field(
        ...,
        description="Dimension -> {Paper Title -> Value with Citations}",
    )
    comparative_synthesis: str = Field(..., description="High-level narrative synthesis contrasting approaches")
    citations: List[Citation] = Field(default_factory=list, description="Source citations backing the comparisons")
    topic: Optional[str] = Field(default=None, description="Optional comparison focus topic")

    @property
    def comparison_matrix(self) -> List[Dict[str, Any]]:
        """Return matrix format for UI table rendering: [{dimension, values: {paper: text}}]."""
        matrix = []
        for dimension, paper_map in self.comparison_table.items():
            matrix.append({
                "dimension": dimension,
                "values": paper_map,
            })
        return matrix

    @property
    def papers_compared(self) -> List[str]:
        """Alias for paper_titles."""
        return self.paper_titles

    @property
    def synthesis(self) -> str:
        """Alias for comparative_synthesis."""
        return self.comparative_synthesis

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        """Custom dump providing both original and frontend-friendly alias keys."""
        data = super().model_dump(**kwargs)
        data["comparison_matrix"] = self.comparison_matrix
        data["papers_compared"] = self.papers_compared
        data["synthesis"] = self.synthesis
        return data

    def to_dataframe_dict(self) -> List[Dict[str, Any]]:
        """Convert comparative table into a list of row dicts for tabular rendering."""
        rows = []
        for dimension, paper_map in self.comparison_table.items():
            row = {"Dimension": dimension}
            row.update(paper_map)
            rows.append(row)
        return rows


COMPARISON_SYNTHESIS_PROMPT = """You are ResearchMate's Comparative Research Analyst.
Your task is to analyze the provided research summaries for multiple papers and generate a concise, grounded comparative synthesis narrative.

CRITICAL RULES:
1. Objectively compare and contrast the goals, methods, and results across the papers.
2. Clearly identify where methods diverge, where trade-offs exist, and where findings agree or conflict.
3. Every factual statement must cite the source papers using inline citations [1], [2], etc.
4. If an aspect is missing for a paper, note that it was "Not reported in the paper".
"""


class PaperComparator:
    """Orchestrates structured comparative analysis across multiple research papers."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        summarizer: Optional[PaperSummarizer] = None,
    ) -> None:
        """Initialize the PaperComparator."""
        self.retriever = retriever or get_hybrid_retriever()
        self.llm_provider = llm_provider or get_llm_provider()
        self.summarizer = summarizer or PaperSummarizer(
            retriever=self.retriever,
            llm_provider=self.llm_provider,
        )

    def compare_papers(
        self,
        document_ids: List[str],
        topic: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> PaperComparisonReport:
        """Execute multi-paper comparison for 2 to 5 documents.

        Args:
            document_ids: List of document IDs (between 2 and 5).
            topic: Optional comparison focus / angle.
            user_id: Optional user identifier for multi-tenant isolation.

        Returns:
            PaperComparisonReport instance.
        """
        if not document_ids:
            return PaperComparisonReport(
                document_ids=[],
                paper_titles=[],
                comparison_table={},
                comparative_synthesis="No papers selected for comparison.",
                citations=[],
                topic=topic,
            )

        # 1. Fetch structured summaries for each paper
        summaries: List[PaperSummary] = []
        all_citations: List[Citation] = []
        paper_titles: List[str] = []

        for doc_id in document_ids:
            summary = self.summarizer.summarize_paper(doc_id, user_id=user_id)
            summaries.append(summary)
            paper_titles.append(summary.paper_title)
            all_citations.extend(summary.citations)

        # 2. Build 10-dimension comparison table
        dimensions = [
            ("Research Problem", "research_problem"),
            ("Objective", "objective"),
            ("Methodology", "methodology"),
            ("Architecture / Model", "model_architecture"),
            ("Dataset", "dataset"),
            ("Experimental Setup", "experimental_setup"),
            ("Evaluation Metrics", "main_results"),
            ("Key Results", "main_results"),
            ("Limitations", "limitations"),
            ("Conclusions", "conclusion"),
        ]

        table: Dict[str, Dict[str, str]] = {}
        for dim_label, attr in dimensions:
            table[dim_label] = {}
            for s in summaries:
                val = getattr(s, attr, "Not reported in the paper.")
                table[dim_label][s.paper_title] = val

        # 3. Generate Comparative Narrative Synthesis
        context_parts = []
        for s in summaries:
            context_parts.append(
                f"Paper: {s.paper_title} ({s.filename})\n"
                f"- Problem: {s.research_problem}\n"
                f"- Objective: {s.objective}\n"
                f"- Methodology: {s.methodology}\n"
                f"- Dataset: {s.dataset}\n"
                f"- Results: {s.main_results}\n"
                f"- Limitations: {s.limitations}\n"
            )

        joint_context = "\n---\n".join(context_parts)

        topic_clause = f"\nSpecific Comparison Focus / Theme: {topic}\n" if topic else ""

        prompt = (
            f"Compare and contrast the following {len(summaries)} research papers:\n\n"
            f"{joint_context}\n\n"
            f"{topic_clause}"
            f"Provide a structured 3-paragraph comparative synthesis highlighting key differences in methodology, empirical results, and practical trade-offs."
        )

        synthesis_text = "Comparative synthesis generated from structured paper metrics."
        if self.llm_provider.is_available():
            resp = self.llm_provider.generate(
                prompt=prompt,
                system_instruction=COMPARISON_SYNTHESIS_PROMPT,
                temperature=0.0,
            )
            if resp.is_success:
                synthesis_text = resp.text.strip()

        # Deduplicate citations
        seen_chunks = set()
        dedup_citations = []
        for cit in all_citations:
            if cit.chunk_id not in seen_chunks:
                seen_chunks.add(cit.chunk_id)
                dedup_citations.append(cit)

        return PaperComparisonReport(
            document_ids=document_ids,
            paper_titles=paper_titles,
            comparison_table=table,
            comparative_synthesis=synthesis_text,
            citations=dedup_citations,
            topic=topic,
        )

