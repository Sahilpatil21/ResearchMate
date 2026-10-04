"""Structured academic paper summarizer for ResearchMate.

Generates comprehensive, citation-grounded 9-field academic summaries:
- Research Problem
- Objective
- Methodology
- Dataset
- Model/Architecture
- Experimental Setup
- Main Results
- Limitations
- Conclusion

Uses targeted retrieval + cross-encoder reranking to assemble rich context
without blindly sending entire PDF dumps to the LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.citations.citation_engine import CitationEngine
from src.llm.factory import get_llm_provider
from src.llm.provider import BaseLLMProvider
from src.models.citation import Citation
from src.models.document import Document
from src.rag.context_builder import ContextBuilder
from src.rag.prompt_templates import ACADEMIC_RAG_SYSTEM_PROMPT
from src.retrieval.hybrid_retriever import HybridRetriever, get_hybrid_retriever
from src.utils.file_utils import ensure_directories, get_data_dir, get_processed_dir, load_processed_document


class PaperSummary(BaseModel):
    """Structured 9-field academic summary for a single research paper."""

    document_id: str = Field(..., description="Unique document identifier")
    paper_title: str = Field(..., description="Title of the research paper")
    filename: str = Field(..., description="PDF filename")
    research_problem: str = Field(..., description="The core research problem addressed by the authors")
    objective: str = Field(..., description="Primary research objectives and proposed contributions")
    methodology: str = Field(..., description="Underlying methods, algorithms, and theoretical techniques")
    dataset: str = Field(..., description="Datasets, corpora, or data sources used for experiments")
    model_architecture: str = Field(..., description="Specific models, neural network architectures, or baselines")
    experimental_setup: str = Field(..., description="Hardware, hyperparameters, baselines, and evaluation protocols")
    main_results: str = Field(..., description="Quantitative and qualitative findings, benchmarks, and key metrics")
    limitations: str = Field(..., description="Explicitly mentioned limitations, constraints, or assumptions")
    conclusion: str = Field(..., description="Concluding remarks and proposed future research directions")
    citations: List[Citation] = Field(default_factory=list, description="Source citations backing the summary claims")
    is_cached: bool = Field(default=False, description="True if retrieved from persistent summary cache")

    def to_markdown(self) -> str:
        """Format summary into a clean academic Markdown report."""
        lines = [
            f"# 📄 Academic Paper Summary: {self.paper_title}",
            f"**File:** `{self.filename}` | **Document ID:** `{self.document_id}`",
            "",
            "## 1. 🎯 Research Problem",
            self.research_problem,
            "",
            "## 2. 💡 Objective & Contributions",
            self.objective,
            "",
            "## 3. ⚙️ Methodology",
            self.methodology,
            "",
            "## 4. 📊 Dataset & Corpus",
            self.dataset,
            "",
            "## 5. 🧠 Model / Architecture",
            self.model_architecture,
            "",
            "## 6. 🔬 Experimental Setup",
            self.experimental_setup,
            "",
            "## 7. 📈 Main Results & Metrics",
            self.main_results,
            "",
            "## 8. ⚠️ Limitations & Constraints",
            self.limitations,
            "",
            "## 9. 🏁 Conclusion & Future Work",
            self.conclusion,
            "",
            "---",
            CitationEngine.format_citations_markdown(self.citations, include_snippets=True),
        ]
        return "\n".join(lines)


SUMMARY_SYSTEM_PROMPT = """You are ResearchMate's Academic Paper Summarizer.
Your goal is to produce a rigorous, structured 9-field academic summary based ONLY on the provided research context passages.

CRITICAL RULES:
1. Grounding Only: Answer strictly based on the provided [Source X] passages. Never invent details.
2. Missing Info: If any field is not mentioned in the provided passages, explicitly output "Not reported in the paper" for that field.
3. Citations: Every statement must include inline citations such as [1], [2] pointing to the source evidence.
4. Output JSON: You must return ONLY a valid JSON object with the exact keys:
{
  "research_problem": "...",
  "objective": "...",
  "methodology": "...",
  "dataset": "...",
  "model_architecture": "...",
  "experimental_setup": "...",
  "main_results": "...",
  "limitations": "...",
  "conclusion": "..."
}
"""


import re

def _extract_json_dict(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract dictionary from LLM output across codeblocks, braces, and key patterns."""
    if not text:
        return None
    raw = text.strip()

    # 1. Direct JSON parse
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 2. Markdown fenced json: ```json ... ``` or ``` ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1).strip())
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # 3. Outermost curly braces { ... }
    m = re.search(r"(\{.*\})", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(1).strip())
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # 4. Key-by-key regex fallback for 9 standard fields
    keys = [
        "research_problem", "objective", "methodology", "dataset",
        "model_architecture", "experimental_setup", "main_results",
        "limitations", "conclusion"
    ]
    extracted = {}
    for key in keys:
        pattern = rf'"{key}"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"'
        km = re.search(pattern, raw, re.IGNORECASE)
        if km:
            try:
                val = km.group(1).encode("utf-8").decode("unicode_escape", errors="replace")
                extracted[key] = val.strip()
            except Exception:
                extracted[key] = km.group(1).strip()

    if len(extracted) >= 3:
        return extracted

    return None


class PaperSummarizer:
    """Extracts structured 9-field academic summaries from indexed research papers."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        context_builder: Optional[ContextBuilder] = None,
    ) -> None:
        """Initialize the PaperSummarizer."""
        ensure_directories()
        self.retriever = retriever or get_hybrid_retriever()
        self.llm_provider = llm_provider or get_llm_provider()
        self.context_builder = context_builder or ContextBuilder()
        
        self.cache_dir = get_data_dir() / "processed" / "summaries"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def summarize_paper(
        self,
        document_id: str,
        force_refresh: bool = False,
        user_id: Optional[str] = None,
    ) -> PaperSummary:
        """Generate or retrieve cached structured summary for a document.

        Args:
            document_id: Unique document identifier.
            force_refresh: If True, regenerates summary ignoring cache.
            user_id: Optional user identifier for multi-tenant isolation.

        Returns:
            PaperSummary instance.
        """
        user_cache_dir = get_processed_dir(user_id) / "summaries" if user_id else self.cache_dir
        user_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = user_cache_dir / f"{document_id}_summary.json"

        # Check persistent cache (reject corrupted/failed placeholder summaries)
        if not force_refresh and cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                prob = data.get("research_problem", "")
                if prob and not prob.startswith("Extraction failed") and prob != "Extraction failed.":
                    summary = PaperSummary.model_validate(data)
                    summary.is_cached = True
                    return summary
            except Exception:
                pass

        doc = load_processed_document(document_id, user_id=user_id)
        paper_title = doc.title if doc else document_id
        filename = doc.metadata.original_filename if doc else f"{document_id}.pdf"

        # Targeted multi-query retrieval to gather comprehensive paper coverage
        queries = [
            "Research problem, introduction, motivation, objective, and contributions",
            "Methodology, algorithms, model architecture, and framework design",
            "Dataset, data collection, experimental setup, hyperparameters, and baselines",
            "Results, performance metrics, evaluation, limitations, and conclusion",
        ]

        candidate_map = {}
        for q in queries:
            results = self.retriever.retrieve(
                query=q,
                mode="hybrid",
                top_k=4,
                candidate_k=10,
                filter_doc_id=document_id,
                rerank=True,
                user_id=user_id,
            )
            for r in results:
                candidate_map[r.chunk_id] = r

        retrieved_results = list(candidate_map.values())

        # Fallback to direct document chunks if retrieval yielded few or no results
        if len(retrieved_results) < 2:
            from src.utils.file_utils import load_document_chunks
            direct_chunks = load_document_chunks(document_id, user_id=user_id)
            if direct_chunks:
                from src.models.retrieval import RetrievalResult
                for rank, c in enumerate(direct_chunks[:12], start=1):
                    if c.chunk_id not in candidate_map:
                        res = RetrievalResult(
                            chunk_id=c.chunk_id,
                            document_id=c.document_id,
                            filename=doc.metadata.filename if doc else f"{document_id}.pdf",
                            page_number=c.page_number,
                            section=c.section,
                            chunk_index=c.chunk_index,
                            text=c.text,
                            score=1.0,
                            dense_score=1.0,
                            dense_rank=rank,
                            retrieval_method="dense",
                        )
                        retrieved_results.append(res)
                        candidate_map[c.chunk_id] = res

        if not retrieved_results:
            return PaperSummary(
                document_id=document_id,
                paper_title=paper_title,
                filename=filename,
                research_problem="No indexed text available for this paper. Please ensure the PDF was processed.",
                objective="Not reported in the paper.",
                methodology="Not reported in the paper.",
                dataset="Not reported in the paper.",
                model_architecture="Not reported in the paper.",
                experimental_setup="Not reported in the paper.",
                main_results="Not reported in the paper.",
                limitations="Not reported in the paper.",
                conclusion="Not reported in the paper.",
                citations=[],
                is_cached=False,
            )

        citations = CitationEngine.create_citations(
            results=retrieved_results,
            document_map={document_id: doc} if doc else None,
        )

        context_str = self.context_builder.build_context_from_citations(citations, max_chars=14000)

        user_prompt = (
            f"Please generate a comprehensive 9-field academic summary for the paper '{paper_title}' ({filename}).\n\n"
            f"### RETRIEVED RESEARCH PAPER CONTEXT ###\n{context_str}\n\n"
            f"Return ONLY valid JSON matching the 9 required fields with inline citations [1], [2], etc."
        )

        if not self.llm_provider.is_available():
            return PaperSummary(
                document_id=document_id,
                paper_title=paper_title,
                filename=filename,
                research_problem="LLM provider is not configured. Please add GEMINI_API_KEY in .env.",
                objective="Not reported in the paper.",
                methodology="Not reported in the paper.",
                dataset="Not reported in the paper.",
                model_architecture="Not reported in the paper.",
                experimental_setup="Not reported in the paper.",
                main_results="Not reported in the paper.",
                limitations="Not reported in the paper.",
                conclusion="Not reported in the paper.",
                citations=citations,
                is_cached=False,
            )

        resp = self.llm_provider.generate(
            prompt=user_prompt,
            system_instruction=SUMMARY_SYSTEM_PROMPT,
            temperature=0.0,
        )

        if not resp.is_success:
            return PaperSummary(
                document_id=document_id,
                paper_title=paper_title,
                filename=filename,
                research_problem=f"Failed to generate summary: {resp.error or 'LLM request failed.'}",
                objective="Please click 'Regenerate' to retry summary generation.",
                methodology="Not reported in the paper.",
                dataset="Not reported in the paper.",
                model_architecture="Not reported in the paper.",
                experimental_setup="Not reported in the paper.",
                main_results="Not reported in the paper.",
                limitations="Not reported in the paper.",
                conclusion="Not reported in the paper.",
                citations=citations,
                is_cached=False,
            )

        parsed = _extract_json_dict(resp.text)
        if parsed and isinstance(parsed, dict):
            summary = PaperSummary(
                document_id=document_id,
                paper_title=paper_title,
                filename=filename,
                research_problem=parsed.get("research_problem") or parsed.get("core_objective") or "Not reported in the paper.",
                objective=parsed.get("objective") or parsed.get("contributions") or "Not reported in the paper.",
                methodology=parsed.get("methodology") or "Not reported in the paper.",
                dataset=parsed.get("dataset") or parsed.get("datasets_benchmarks") or "Not reported in the paper.",
                model_architecture=parsed.get("model_architecture") or parsed.get("mathematical_formulations") or "Not reported in the paper.",
                experimental_setup=parsed.get("experimental_setup") or parsed.get("practical_implications") or "Not reported in the paper.",
                main_results=parsed.get("main_results") or parsed.get("key_findings") or "Not reported in the paper.",
                limitations=parsed.get("limitations") or parsed.get("critical_limitations") or "Not reported in the paper.",
                conclusion=parsed.get("conclusion") or parsed.get("future_directions") or "Not reported in the paper.",
                citations=citations,
                is_cached=False,
            )

            # Save valid summary to persistent cache
            try:
                cache_file.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
            except Exception:
                pass
            return summary

        # Fallback parsing for non-JSON formatted text
        summary = PaperSummary(
            document_id=document_id,
            paper_title=paper_title,
            filename=filename,
            research_problem=resp.text.strip(),
            objective="Included in research synthesis above.",
            methodology="Included in research synthesis above.",
            dataset="Included in research synthesis above.",
            model_architecture="Included in research synthesis above.",
            experimental_setup="Included in research synthesis above.",
            main_results="Included in research synthesis above.",
            limitations="Included in research synthesis above.",
            conclusion="Included in research synthesis above.",
            citations=citations,
            is_cached=False,
        )
        return summary
