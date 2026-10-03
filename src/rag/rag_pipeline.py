"""Comprehensive RAG pipeline orchestration for ResearchMate.

Implements the end-to-end grounded research Q&A workflow:
Query -> Hybrid Retrieval -> Cross-Encoder Reranking -> Citations -> Strict Context -> LLM -> Validation.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.citations.citation_engine import CitationEngine
from src.llm.factory import get_llm_provider
from src.llm.provider import BaseLLMProvider, LLMResponse
from src.models.citation import Citation
from src.models.retrieval import RetrievalResult
from src.rag.answer_validator import AnswerValidator, ValidationResult
from src.rag.context_builder import ContextBuilder
from src.rag.prompt_templates import (
    ACADEMIC_RAG_SYSTEM_PROMPT,
    REFERENCE_RESOLUTION_SYSTEM_PROMPT,
    build_rag_user_prompt,
    build_reference_resolution_prompt,
)
from src.retrieval.hybrid_retriever import HybridRetriever, get_hybrid_retriever
from src.utils.file_utils import ensure_directories, get_data_dir


class RAGResponse(BaseModel):
    """Complete structured response from the ResearchMate RAG pipeline."""

    query: str = Field(..., description="Original user question")
    resolved_query: Optional[str] = Field(default=None, description="Disambiguated search query if conversational")
    answer: str = Field(..., description="LLM generated answer text with inline citations")
    citations: List[Citation] = Field(default_factory=list, description="Retrieved and cited research paper sources")
    retrieval_results: List[RetrievalResult] = Field(default_factory=list, description="Ranked retrieval results")
    validation: ValidationResult = Field(..., description="Groundedness and citation validation report")
    provider: str = Field(..., description="LLM provider name (e.g. gemini)")
    model: str = Field(..., description="Model identifier used")
    latency_seconds: float = Field(default=0.0, description="Total end-to-end execution time in seconds")
    error: Optional[str] = Field(default=None, description="Error message if pipeline encountered an issue")
    is_insufficient_evidence: bool = Field(default=False, description="True if no evidence found or model indicated lack of info")

    @property
    def formatted_sources_markdown(self) -> str:
        """Formatted markdown bibliography block."""
        return CitationEngine.format_citations_markdown(self.citations, include_snippets=True)


class RAGPipeline:
    """Orchestrates hybrid retrieval, cross-encoder reranking, strict prompt building, LLM inference, and citation validation."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        context_builder: Optional[ContextBuilder] = None,
        log_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the RAG pipeline.

        Args:
            retriever: HybridRetriever instance (ChromaDB + BM25 + CrossEncoder).
            llm_provider: BaseLLMProvider instance (defaults to GeminiProvider).
            context_builder: ContextBuilder instance for structured formatting.
            log_dir: Directory for persisting query logs.
        """
        ensure_directories()
        self.retriever = retriever or get_hybrid_retriever()
        self.llm_provider = llm_provider or get_llm_provider()
        self.context_builder = context_builder or ContextBuilder()
        
        self.log_dir = log_dir or (get_data_dir() / "logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "rag_queries.jsonl"

    def resolve_conversational_query(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Resolve pronouns or referential follow-up questions using conversation history.

        Args:
            query: Follow-up question from user.
            chat_history: Past conversation messages.

        Returns:
            Resolved query string for retrieval.
        """
        if not chat_history or len(chat_history) < 2:
            return query

        # Check if query contains referential pronouns or short follow-ups
        referential_triggers = ["it", "they", "this paper", "that paper", "the second", "the first", "former", "latter", "compare both", "what about"]
        query_lower = query.lower()
        if not any(trigger in query_lower for trigger in referential_triggers) and len(query.split()) > 5:
            return query

        if not self.llm_provider.is_available():
            return query

        try:
            prompt = build_reference_resolution_prompt(query, chat_history)
            response = self.llm_provider.generate(
                prompt=prompt,
                system_instruction=REFERENCE_RESOLUTION_SYSTEM_PROMPT,
                temperature=0.0,
            )
            if response.is_success and len(response.text.strip()) > 3:
                return response.text.strip().replace('"', '')
        except Exception:
            pass

        return query

    def answer_question(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        retrieval_mode: str = "hybrid",
        candidate_k: int = 20,
        top_k: int = 5,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        filter_doc_id: Optional[str] = None,
        rerank: bool = True,
        temperature: float = 0.0,
        user_id: Optional[str] = None,
    ) -> RAGResponse:
        """Execute complete RAG pipeline for a research question.

        Args:
            query: User's research question.
            chat_history: Optional conversation history.
            retrieval_mode: Retrieval strategy ('hybrid', 'dense', 'sparse', 'weighted').
            candidate_k: Initial candidate pool size retrieved before reranking.
            top_k: Final number of top chunks sent to the LLM context.
            dense_weight: Dense semantic score weight.
            sparse_weight: Sparse BM25 score weight.
            filter_doc_id: Optional document ID to filter search scope.
            rerank: If True, applies Cross-Encoder reranking to candidate chunks.
            temperature: LLM sampling temperature.
            user_id: Optional user identifier for multi-tenant isolation.

        Returns:
            Structured RAGResponse object with grounded answer, citations, and validation report.
        """
        start_time = time.perf_counter()

        # 0. Handle empty query
        if not query or not query.strip():
            return RAGResponse(
                query=query,
                answer="Please enter a valid research question.",
                citations=[],
                retrieval_results=[],
                validation=ValidationResult(
                    is_valid=True,
                    has_citations=False,
                    is_insufficient_evidence=True,
                ),
                provider=self.llm_provider.get_provider_name(),
                model=self.llm_provider.get_model_name(),
                latency_seconds=0.0,
                is_insufficient_evidence=True,
            )

        # 1. Conversational Query Disambiguation (if applicable)
        resolved_query = query
        if chat_history:
            resolved_query = self.resolve_conversational_query(query, chat_history)

        # 2. Hybrid / Dense / Sparse Retrieval with Reranking
        retrieval_results = self.retriever.retrieve(
            query=resolved_query,
            mode=retrieval_mode,
            top_k=top_k,
            candidate_k=candidate_k,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            filter_doc_id=filter_doc_id,
            rerank=rerank,
            user_id=user_id,
        )


        # 3. Handle Zero Retrieved Chunks (Hallucination Protection)
        if not retrieval_results:
            elapsed = round(time.perf_counter() - start_time, 3)
            insufficient_msg = "The provided papers do not contain enough information to answer this question."
            response = RAGResponse(
                query=query,
                resolved_query=resolved_query if resolved_query != query else None,
                answer=insufficient_msg,
                citations=[],
                retrieval_results=[],
                validation=ValidationResult(
                    is_valid=True,
                    has_citations=False,
                    is_insufficient_evidence=True,
                    grounding_score=1.0,
                ),
                provider=self.llm_provider.get_provider_name(),
                model=self.llm_provider.get_model_name(),
                latency_seconds=elapsed,
                is_insufficient_evidence=True,
            )
            self._log_query(response, retrieval_mode, candidate_k, top_k, rerank, filter_doc_id)
            return response

        # 4. Generate Grounded Citations
        citations = CitationEngine.create_citations(
            results=retrieval_results,
            query=resolved_query,
        )

        # 5. Build Structured Academic Context
        context_str = self.context_builder.build_context_from_citations(citations)

        # 6. Build Strict Academic RAG Prompt
        prompt = build_rag_user_prompt(
            query=query,
            context=context_str,
            chat_history=chat_history,
        )

        # 7. Execute LLM Generation
        llm_resp = self.llm_provider.generate(
            prompt=prompt,
            system_instruction=ACADEMIC_RAG_SYSTEM_PROMPT,
            temperature=temperature,
        )

        elapsed = round(time.perf_counter() - start_time, 3)

        if not llm_resp.is_success:
            # LLM returned an error or configuration message
            val_result = ValidationResult(
                is_valid=False,
                has_citations=False,
                warnings=[llm_resp.error or "LLM generation failed"],
            )
            response = RAGResponse(
                query=query,
                resolved_query=resolved_query if resolved_query != query else None,
                answer=f"⚠️ {llm_resp.error or 'Failed to generate answer.'}",
                citations=citations,
                retrieval_results=retrieval_results,
                validation=val_result,
                provider=self.llm_provider.get_provider_name(),
                model=self.llm_provider.get_model_name(),
                latency_seconds=elapsed,
                error=llm_resp.error,
            )
            self._log_query(response, retrieval_mode, candidate_k, top_k, rerank, filter_doc_id)
            return response

        # 8. Validate Groundedness & Citations
        validation = AnswerValidator.validate_answer(
            answer_text=llm_resp.text,
            citations=citations,
        )

        response = RAGResponse(
            query=query,
            resolved_query=resolved_query if resolved_query != query else None,
            answer=llm_resp.text.strip(),
            citations=citations,
            retrieval_results=retrieval_results,
            validation=validation,
            provider=self.llm_provider.get_provider_name(),
            model=self.llm_provider.get_model_name(),
            latency_seconds=elapsed,
            is_insufficient_evidence=validation.is_insufficient_evidence,
        )

        # 9. Log RAG Query
        self._log_query(response, retrieval_mode, candidate_k, top_k, rerank, filter_doc_id)

        return response

    def _log_query(
        self,
        resp: RAGResponse,
        retrieval_mode: str,
        candidate_k: int,
        top_k: int,
        rerank: bool,
        filter_doc_id: Optional[str],
    ) -> None:
        """Persist structured RAG query log entry without storing API keys."""
        try:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": resp.query,
                "resolved_query": resp.resolved_query,
                "retrieval_strategy": retrieval_mode,
                "candidate_k": candidate_k,
                "final_top_k": top_k,
                "reranking_enabled": rerank,
                "filter_doc_id": filter_doc_id,
                "retrieved_chunk_ids": [r.chunk_id for r in resp.retrieval_results],
                "citation_ids": [c.citation_id for c in resp.citations],
                "model": resp.model,
                "provider": resp.provider,
                "latency_seconds": resp.latency_seconds,
                "validation_valid": resp.validation.is_valid,
                "has_citations": resp.validation.has_citations,
                "is_insufficient_evidence": resp.is_insufficient_evidence,
                "answer_length": len(resp.answer),
            }
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # Logging should never block user execution
