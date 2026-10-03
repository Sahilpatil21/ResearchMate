"""OpenAI-compatible LLM provider for ResearchMate.

Supports OpenAI, Azure OpenAI, Groq, Together, DeepSeek, or any OpenAI-compatible endpoint.
Reads API key securely from .env and protects against key leaks.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from src.llm.provider import BaseLLMProvider, LLMResponse

load_dotenv()


class OpenAICompatibleProvider(BaseLLMProvider):
    """Provider for OpenAI and OpenAI-compatible API endpoints."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """Initialize the OpenAI-compatible provider.

        Args:
            api_key: Optional API key (defaults to OPENAI_API_KEY or LLM_API_KEY).
            model_name: Optional model name (defaults to OPENAI_MODEL or LLM_MODEL or 'gpt-4o-mini').
            base_url: Optional base URL (defaults to OPENAI_BASE_URL or LLM_BASE_URL).
        """
        self.api_key = (
            api_key
            if api_key is not None
            else (os.getenv("OPENAI_API_KEY", "") or os.getenv("LLM_API_KEY", ""))
        ).strip()
        self.model_name = (
            model_name
            or os.getenv("OPENAI_MODEL", "")
            or os.getenv("LLM_MODEL", "")
            or "gpt-4o-mini"
        ).strip()
        self.base_url = (
            base_url
            or os.getenv("OPENAI_BASE_URL", "")
            or os.getenv("LLM_BASE_URL", "")
            or None
        )
        self._client = None

    def is_available(self) -> bool:
        """Return True if an API key is configured."""
        return bool(self.api_key and len(self.api_key) > 3)

    def get_model_name(self) -> str:
        """Return active model name."""
        return self.model_name

    def get_provider_name(self) -> str:
        """Return provider identifier."""
        return "openai"

    def _get_client(self):
        """Initialize or return cached OpenAI client."""
        if not self.is_available():
            raise ValueError("OPENAI_API_KEY is not set.")

        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                raise RuntimeError(
                    "The 'openai' Python package is not installed. Install it via 'pip install openai'."
                )
        return self._client

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Execute text generation via OpenAI-compatible endpoint.

        Args:
            prompt: User prompt containing research context and query.
            system_instruction: High-priority system instructions.
            temperature: Sampling temperature.

        Returns:
            LLMResponse object with generated text or error details.
        """
        if not self.is_available():
            return LLMResponse(
                text="",
                model_name=self.model_name,
                provider_name=self.get_provider_name(),
                error="OpenAI API key is missing. Set OPENAI_API_KEY or LLM_API_KEY in .env.",
            )

        start_time = time.perf_counter()

        try:
            client = self._get_client()

            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

            completion = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
            )

            elapsed = round(time.perf_counter() - start_time, 3)

            choice = completion.choices[0] if completion.choices else None
            generated_text = choice.message.content if choice and choice.message else ""

            prompt_tokens = completion.usage.prompt_tokens if completion.usage else None
            comp_tokens = completion.usage.completion_tokens if completion.usage else None

            if not generated_text:
                return LLMResponse(
                    text="",
                    model_name=self.model_name,
                    provider_name=self.get_provider_name(),
                    latency_seconds=elapsed,
                    error="OpenAI returned an empty completion response.",
                )

            return LLMResponse(
                text=generated_text,
                model_name=self.model_name,
                provider_name=self.get_provider_name(),
                latency_seconds=elapsed,
                prompt_tokens=prompt_tokens,
                completion_tokens=comp_tokens,
            )

        except Exception as e:
            elapsed = round(time.perf_counter() - start_time, 3)
            err_msg = str(e)
            if self.api_key and self.api_key in err_msg:
                err_msg = err_msg.replace(self.api_key, "[REDACTED_API_KEY]")

            return LLMResponse(
                text="",
                model_name=self.model_name,
                provider_name=self.get_provider_name(),
                latency_seconds=elapsed,
                error=f"OpenAI API Error: {err_msg}",
            )
