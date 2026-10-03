"""RAG orchestration package for ResearchMate.

Provides end-to-end grounded question answering, context building, prompt engineering,
and citation validation.
"""

from src.rag.answer_validator import AnswerValidator, ValidationResult
from src.rag.context_builder import ContextBuilder
from src.rag.prompt_templates import (
    ACADEMIC_RAG_SYSTEM_PROMPT,
    REFERENCE_RESOLUTION_SYSTEM_PROMPT,
    build_rag_user_prompt,
    build_reference_resolution_prompt,
)
from src.rag.rag_pipeline import RAGPipeline, RAGResponse

__all__ = [
    "RAGPipeline",
    "RAGResponse",
    "ContextBuilder",
    "AnswerValidator",
    "ValidationResult",
    "ACADEMIC_RAG_SYSTEM_PROMPT",
    "REFERENCE_RESOLUTION_SYSTEM_PROMPT",
    "build_rag_user_prompt",
    "build_reference_resolution_prompt",
]
