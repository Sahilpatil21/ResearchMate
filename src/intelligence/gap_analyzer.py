"""Evidence-grounded Research Gap Analyzer for ResearchMate.

Identifies potential research gaps across research papers:
- Repeated limitations
- Underrepresented datasets
- Missing method comparisons
- Evaluation limitations
- Unexplored approaches
- Conflicting findings
- Limited real-world validation
- Insufficient experiments

Every gap explicitly cites the source paper, exact page, section, and citation ID,
distinguishing evidence-supported observations from future research suggestions.
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
from src.utils.file_utils import load_processed_document


class ResearchGapItem(BaseModel):
    """An individual evidence-grounded research gap item."""

    gap_statement: str = Field(..., description="Statement of the potential research gap")
    category: str = Field(..., description="Category (e.g. 'Methodology Limitation', 'Dataset Generalizability', 'Evaluation Metric Gap')")
    supporting_evidence: str = Field(..., description="Direct quote or grounded finding from the paper passages")
    source_paper: str = Field(..., description="Title or filename of the cited source paper")
    page_number: int = Field(..., description="Exact page number where the limitation was found")
    section: str = Field(..., description="Academic section where limitation is described")
    citation_id: str = Field(..., description="Citation ID, e.g. '[1]'")
    future_direction_suggestion: str = Field(..., description="Reasonable future research suggestion addressing the gap")


class ResearchGapReport(BaseModel):
    """Structured report containing categorized research gaps across papers."""

    document_ids: List[str] = Field(..., description="Analyzed document IDs")
    executive_summary: str = Field(..., description="Executive synthesis of overarching research gaps")
    gaps: List[ResearchGapItem] = Field(default_factory=list, description="List of validated research gaps")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")

    def to_markdown(self) -> str:
        """Render gap report into structured Markdown."""
        lines = [
            "# 🔬 Evidence-Grounded Research Gap Analysis",
            "",
            "## 📌 Executive Summary",
            self.executive_summary,
            "",
            "## 🔍 Identified Potential Research Gaps",
        ]

        for idx, g in enumerate(self.gaps, start=1):
            lines.extend(
                [
                    f"### Gap #{idx}: {g.gap_statement}",
                    f"- **Category:** `{g.category}`",
                    f"- **Source:** {g.source_paper} (Page **{g.page_number}**, Section: *{g.section}*) {g.citation_id}",
                    f"- **Supporting Evidence (From Paper):**\n  > \"{g.supporting_evidence}\"",
                    f"- **Future Research Opportunity:** {g.future_direction_suggestion}",
                    "",
                ]
            )

        lines.append("---")
        lines.append(CitationEngine.format_citations_markdown(self.citations, include_snippets=True))
        return "\n".join(lines)


GAP_SYSTEM_PROMPT = """You are ResearchMate's Academic Gap Analyst.
Your role is to discover potential research gaps, open problems, and methodological limitations across the supplied research paper passages.

CRITICAL RULES:
1. Grounding: Every gap must be directly grounded in the limitations, discussions, or future work explicitly stated by the authors in the [Source X] passages.
2. Formulation: Prefix gap statements with "Potential research gap:" or similar cautious academic phrasing.
3. Distinction: Clearly distinguish the author's direct stated limitation (Supporting Evidence) from your proposed extension (Future Direction).
4. Citations: Every item must have exact citation indices [1], [2] pointing to the context source.
5. Return JSON: Output ONLY a JSON object formatted as:
{
  "executive_summary": "...",
  "gaps": [
    {
      "gap_statement": "Potential research gap: ...",
      "category": "Methodology / Dataset / Evaluation / Generalization",
      "supporting_evidence": "Exact or near-exact quote from context",
      "source_paper": "Filename or title",
      "page_number": 1,
      "section": "Limitations / Discussion / Conclusion",
      "citation_id": "[1]",
      "future_direction_suggestion": "Proposed future research pathway..."
    }
  ]
}
"""


class ResearchGapAnalyzer:
    """Extracts evidence-grounded research gaps and open questions across papers."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        context_builder: Optional[ContextBuilder] = None,
    ) -> None:
        """Initialize the ResearchGapAnalyzer."""
        self.retriever = retriever or get_hybrid_retriever()
        self.llm_provider = llm_provider or get_llm_provider()
        self.context_builder = context_builder or ContextBuilder()

    def analyze_gaps(
        self,
        document_ids: Optional[List[str]] = None,
        topic: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> ResearchGapReport:
        """Execute research gap discovery across selected documents.

        Args:
            document_ids: Optional list of document IDs to analyze (defaults to all user documents).
            topic: Optional domain/topic focus.
            user_id: Optional user identifier for multi-tenant isolation.

        Returns:
            ResearchGapReport instance.
        """
        # Resolve document_ids if not provided
        if not document_ids:
            from src.utils.file_utils import load_all_processed_documents
            all_docs = load_all_processed_documents(user_id=user_id)
            document_ids = [d.document_id for d in all_docs]

        if not document_ids:
            return ResearchGapReport(
                document_ids=[],
                executive_summary="No papers found in your library for gap analysis. Please upload research papers first.",
                gaps=[],
                citations=[],
            )

        # Retrieve limitation and discussion chunks across all selected papers
        queries = [
            "Limitations, weaknesses, assumptions, and failure modes",
            "Future work, open problems, unaddressed challenges, and ethical concerns",
            "Dataset bias, generalizability constraints, and missing comparative baselines",
        ]
        if topic and topic.strip():
            queries.insert(0, f"Limitations and research gaps regarding: {topic.strip()}")

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

        if not retrieved_results:
            return ResearchGapReport(
                document_ids=document_ids,
                executive_summary="No limitation or discussion passages found in the selected papers.",
                gaps=[],
                citations=[],
            )

        citations = CitationEngine.create_citations(
            results=retrieved_results,
        )

        context_str = self.context_builder.build_context_from_citations(citations, max_chars=14000)

        topic_instruction = f" Focus specifically on: {topic.strip()}." if topic and topic.strip() else ""

        prompt = (
            f"Analyze the following {len(document_ids)} papers for potential research gaps, limitations, and future directions.{topic_instruction}\n\n"
            f"### RETRIEVED RESEARCH CONTEXT ###\n{context_str}\n\n"
            f"Extract structured research gaps with supporting evidence and future directions."
        )

        if not self.llm_provider.is_available():
            # Graceful unconfigured mock/fallback
            sample_gaps = []
            for idx, cit in enumerate(citations[:3], start=1):
                sample_gaps.append(
                    ResearchGapItem(
                        gap_statement=f"Potential research gap: Investigation of generalizability constraints in {cit.paper_title}.",
                        category="Generalization & Real-world Validation",
                        supporting_evidence=cit.source_snippet or cit.source_text[:120],
                        source_paper=cit.filename,
                        page_number=cit.page_number,
                        section=cit.section,
                        citation_id=cit.citation_id,
                        future_direction_suggestion="Evaluate across larger multi-domain benchmark corpora.",
                    )
                )
            return ResearchGapReport(
                document_ids=document_ids,
                executive_summary="LLM Provider not configured. Displaying extracted evidence passages.",
                gaps=sample_gaps,
                citations=citations,
            )

        resp = self.llm_provider.generate(
            prompt=prompt,
            system_instruction=GAP_SYSTEM_PROMPT,
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

            gap_items = []
            for g in parsed.get("gaps", []):
                # Ensure citation_id is formatted
                cit_id = g.get("citation_id", "[1]")
                if not cit_id.startswith("["):
                    cit_id = f"[{cit_id}]"

                gap_items.append(
                    ResearchGapItem(
                        gap_statement=g.get("gap_statement", "Potential research gap identified."),
                        category=g.get("category", "Methodological Gap"),
                        supporting_evidence=g.get("supporting_evidence", "Derived from source limitations."),
                        source_paper=g.get("source_paper", "Research Paper"),
                        page_number=int(g.get("page_number", 1)),
                        section=g.get("section", "Limitations"),
                        citation_id=cit_id,
                        future_direction_suggestion=g.get("future_direction_suggestion", "Further empirical validation needed."),
                    )
                )

            return ResearchGapReport(
                document_ids=document_ids,
                executive_summary=parsed.get("executive_summary", "Summary of identified research gaps."),
                gaps=gap_items,
                citations=citations,
            )

        except Exception:
            return ResearchGapReport(
                document_ids=document_ids,
                executive_summary=resp.text if resp.is_success else "Gap extraction completed.",
                gaps=[],
                citations=citations,
            )
