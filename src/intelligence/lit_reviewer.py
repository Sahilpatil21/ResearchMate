"""Comprehensive Literature Review Generator for ResearchMate.

Generates a structured 10-section academic literature review from selected papers:
1. Introduction
2. Research Area Overview
3. Existing Approaches
4. Methodology Comparison
5. Dataset Trends
6. Key Findings
7. Limitations
8. Potential Research Gaps
9. Future Research Directions
10. References
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.citations.citation_engine import CitationEngine
from src.llm.factory import get_llm_provider
from src.llm.provider import BaseLLMProvider
from src.models.citation import Citation
from src.rag.context_builder import ContextBuilder
from src.retrieval.hybrid_retriever import HybridRetriever, get_hybrid_retriever


class LiteratureReviewReport(BaseModel):
    """Structured 10-section academic literature review."""

    topic: str = Field(..., description="Literature review title or research theme")
    document_ids: List[str] = Field(..., description="Included document IDs")
    introduction: str = Field(..., description="1. Introduction")
    research_area_overview: str = Field(..., description="2. Research Area Overview")
    existing_approaches: str = Field(..., description="3. Existing Approaches")
    methodology_comparison: str = Field(..., description="4. Methodology Comparison")
    dataset_trends: str = Field(..., description="5. Dataset Trends")
    key_findings: str = Field(..., description="6. Key Findings")
    limitations: str = Field(..., description="7. Limitations")
    potential_research_gaps: str = Field(..., description="8. Potential Research Gaps")
    future_research_directions: str = Field(..., description="9. Future Research Directions")
    references_markdown: str = Field(..., description="10. References")
    citations: List[Citation] = Field(default_factory=list, description="Extracted source citations")

    def to_markdown(self) -> str:
        """Render complete literature review report into publication-ready Markdown."""
        lines = [
            f"# 📖 Literature Review: {self.topic}",
            "",
            "## 1. Introduction",
            self.introduction,
            "",
            "## 2. Research Area Overview",
            self.research_area_overview,
            "",
            "## 3. Existing Approaches",
            self.existing_approaches,
            "",
            "## 4. Methodology Comparison",
            self.methodology_comparison,
            "",
            "## 5. Dataset Trends",
            self.dataset_trends,
            "",
            "## 6. Key Findings",
            self.key_findings,
            "",
            "## 7. Limitations",
            self.limitations,
            "",
            "## 8. Potential Research Gaps",
            self.potential_research_gaps,
            "",
            "## 9. Future Research Directions",
            self.future_research_directions,
            "",
            "## 10. References",
            self.references_markdown,
        ]
        return "\n".join(lines)


LIT_REVIEW_SYSTEM_PROMPT = """You are ResearchMate's Academic Literature Reviewer.
Synthesize the provided research paper passages into a formal 9-section literature review (the 10th section, References, will be appended automatically).

CRITICAL RULES:
1. Grounding: Every claim, finding, comparison, or trend must be backed by inline citations [1], [2], etc.
2. Structure: Maintain rigorous academic tone with clear synthesis across papers.
3. No Hallucinations: Do not fabricate external papers or authors outside the provided context.
4. Output Format: Return a JSON object with the exact keys:
{
  "introduction": "...",
  "research_area_overview": "...",
  "existing_approaches": "...",
  "methodology_comparison": "...",
  "dataset_trends": "...",
  "key_findings": "...",
  "limitations": "...",
  "potential_research_gaps": "...",
  "future_research_directions": "..."
}
"""


class LiteratureReviewGenerator:
    """Generates structured, citation-grounded 10-section literature reviews."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        context_builder: Optional[ContextBuilder] = None,
    ) -> None:
        """Initialize LiteratureReviewGenerator."""
        self.retriever = retriever or get_hybrid_retriever()
        self.llm_provider = llm_provider or get_llm_provider()
        self.context_builder = context_builder or ContextBuilder()

    def generate_review(
        self,
        document_ids: List[str],
        topic_focus: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> LiteratureReviewReport:
        """Generate structured literature review across selected documents.

        Args:
            document_ids: List of document IDs to include.
            topic_focus: Optional focus query / theme.
            user_id: Optional user identifier for multi-tenant isolation.

        Returns:
            LiteratureReviewReport instance.
        """
        focus_title = topic_focus or "Synthesized Analysis of Selected Research Papers"

        if not document_ids:
            return LiteratureReviewReport(
                topic=focus_title,
                document_ids=[],
                introduction="No papers selected.",
                research_area_overview="N/A",
                existing_approaches="N/A",
                methodology_comparison="N/A",
                dataset_trends="N/A",
                key_findings="N/A",
                limitations="N/A",
                potential_research_gaps="N/A",
                future_research_directions="N/A",
                references_markdown="_No references available._",
                citations=[],
            )

        queries = [
            f"Introduction, background, research area overview, and motivation in {focus_title}",
            "Proposed methodology, framework, architectures, and theoretical foundations",
            "Datasets, benchmarks, empirical results, and key experimental findings",
            "Limitations, discussions, potential research gaps, and future research directions",
        ]

        candidate_map = {}
        for doc_id in document_ids:
            for q in queries:
                results = self.retriever.retrieve(
                    query=q,
                    mode="hybrid",
                    top_k=3,
                    candidate_k=8,
                    filter_doc_id=doc_id,
                    rerank=True,
                    user_id=user_id,
                )
                for r in results:
                    candidate_map[r.chunk_id] = r


        retrieved_results = list(candidate_map.values())
        citations = CitationEngine.create_citations(results=retrieved_results)

        references_md = CitationEngine.format_citations_markdown(citations, include_snippets=False)
        context_str = self.context_builder.build_context_from_citations(citations, max_chars=14000)

        prompt = (
            f"Generate a comprehensive 10-section academic literature review on '{focus_title}' from the following {len(document_ids)} papers:\n\n"
            f"### RETRIEVED RESEARCH CONTEXT ###\n{context_str}\n\n"
            f"Provide grounded academic synthesis with inline citations [1], [2], etc."
        )

        if not self.llm_provider.is_available():
            return LiteratureReviewReport(
                topic=focus_title,
                document_ids=document_ids,
                introduction=f"This literature review examines {len(document_ids)} papers in the field of {focus_title}.",
                research_area_overview="Recent advances have focused on machine learning algorithms and neural models.",
                existing_approaches="Prior works establish baselines across standard benchmark datasets.",
                methodology_comparison="Approaches differ in network architectures, optimization objectives, and validation protocols.",
                dataset_trends="Studies utilize diverse datasets ranging from educational records to computer vision benchmarks.",
                key_findings="Empirical evaluations demonstrate improved predictive performance and generalization.",
                limitations="Existing evaluations are constrained by sample distributions and benchmark specificity.",
                potential_research_gaps="Potential research gap: Need for cross-domain validation and standardized evaluation benchmarks.",
                future_research_directions="Future work includes exploring foundation models and real-world deployment safety.",
                references_markdown=references_md,
                citations=citations,
            )

        resp = self.llm_provider.generate(
            prompt=prompt,
            system_instruction=LIT_REVIEW_SYSTEM_PROMPT,
            temperature=0.0,
        )

        try:
            raw_text = resp.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            parsed = json.loads(raw_text.strip())

            return LiteratureReviewReport(
                topic=focus_title,
                document_ids=document_ids,
                introduction=parsed.get("introduction", "Introduction synthesized from context."),
                research_area_overview=parsed.get("research_area_overview", "Overview of current research paradigms."),
                existing_approaches=parsed.get("existing_approaches", "Survey of existing methodologies."),
                methodology_comparison=parsed.get("methodology_comparison", "Comparative methodological analysis."),
                dataset_trends=parsed.get("dataset_trends", "Empirical dataset usage and benchmarking trends."),
                key_findings=parsed.get("key_findings", "Core empirical findings across studies."),
                limitations=parsed.get("limitations", "Limitations and constraints identified by authors."),
                potential_research_gaps=parsed.get("potential_research_gaps", "Synthesized research gaps."),
                future_research_directions=parsed.get("future_research_directions", "Promising future research trajectories."),
                references_markdown=references_md,
                citations=citations,
            )

        except Exception:
            return LiteratureReviewReport(
                topic=focus_title,
                document_ids=document_ids,
                introduction=resp.text if resp.is_success else "Review generated from retrieved passages.",
                research_area_overview="See detailed context above.",
                existing_approaches="See detailed context above.",
                methodology_comparison="See detailed context above.",
                dataset_trends="See detailed context above.",
                key_findings="See detailed context above.",
                limitations="See detailed context above.",
                potential_research_gaps="See detailed context above.",
                future_research_directions="See detailed context above.",
                references_markdown=references_md,
                citations=citations,
            )
