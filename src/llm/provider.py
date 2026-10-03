"""Abstract base LLM provider interface and response data models for ResearchMate.

Provides a unified, pluggable contract for LLM providers (Gemini, OpenAI, Ollama)
supporting system instructions, temperature, token usage, and structured error reporting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class LLMResponse(BaseModel):
    """Structured response from an LLM inference call."""

    text: str = Field(..., description="Generated answer text")
    model_name: str = Field(..., description="Model identifier used for generation")
    provider_name: str = Field(..., description="Provider name (e.g. 'gemini', 'openai', 'ollama')")
    latency_seconds: float = Field(default=0.0, description="Inference execution duration in seconds")
    prompt_tokens: Optional[int] = Field(default=None, description="Input prompt token count")
    completion_tokens: Optional[int] = Field(default=None, description="Generated completion token count")
    error: Optional[str] = Field(default=None, description="Error message if inference failed")
    raw_response: Optional[Dict[str, Any]] = Field(default=None, description="Optional raw provider metadata")

    @property
    def is_success(self) -> bool:
        """Return True if generation succeeded without fatal errors."""
        return self.error is None and len(self.text.strip()) > 0


class BaseLLMProvider(ABC):
    """Abstract Base Class for all LLM providers in ResearchMate."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate text from the LLM given a prompt and optional system instructions.

        Args:
            prompt: User prompt containing query and contextual citations.
            system_instruction: Optional high-priority system instructions.
            temperature: Sampling temperature (0.0 for deterministic factual Q&A).

        Returns:
            LLMResponse object containing generated text, metadata, and error details.
        """
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the provider is configured and available (e.g. API key present)."""
        raise NotImplementedError

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the active model name."""
        raise NotImplementedError

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider identifier."""
        raise NotImplementedError
