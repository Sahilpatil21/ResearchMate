"""Google Gemini LLM provider for ResearchMate.

Implements the BaseLLMProvider interface using the official Google Gemini API.
Reads API key securely from .env and protects against leaks or crashes on missing configuration.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from src.llm.provider import BaseLLMProvider, LLMResponse

load_dotenv()


class GeminiProvider(BaseLLMProvider):
    """Provider for Google Gemini models (e.g. gemini-2.5-flash, gemini-1.5-flash)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize the GeminiProvider.

        Args:
            api_key: Optional Gemini API key (defaults to GEMINI_API_KEY env variable).
            model_name: Optional Gemini model identifier (defaults to GEMINI_MODEL or 'gemini-2.5-flash').
        """
        self.api_key = (
            api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        ).strip()
        self.model_name = (
            model_name
            if model_name is not None
            else os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        ).strip() or "gemini-2.5-flash"
        self._client = None

    def is_available(self) -> bool:
        """Return True if an API key is configured."""
        return bool(self.api_key and len(self.api_key) > 5)

    def get_model_name(self) -> str:
        """Return active model name."""
        return self.model_name

    def get_provider_name(self) -> str:
        """Return provider identifier."""
        return "gemini"

    def _get_client(self):
        """Initialize or return the cached Gemini client."""
        if not self.is_available():
            raise ValueError(
                "GEMINI_API_KEY is not set. Please add GEMINI_API_KEY=your_key in the .env file."
            )

        if self._client is None:
            # Try new google.genai Client first, fallback to google.generativeai
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception:
                try:
                    import google.generativeai as gai
                    gai.configure(api_key=self.api_key)
                    self._client = gai
                except Exception as e:
                    raise RuntimeError(f"Failed to initialize Gemini SDK: {e}") from e

        return self._client

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Execute text generation via Gemini API.

        Args:
            prompt: User prompt containing research context and question.
            system_instruction: High-priority system instructions.
            temperature: Sampling temperature (0.0 for deterministic factual Q&A).

        Returns:
            LLMResponse object with generated text or descriptive error details.
        """
        if not self.is_available():
            return LLMResponse(
                text="",
                model_name=self.model_name,
                provider_name=self.get_provider_name(),
                error="GEMINI_API_KEY is missing. Please add 'GEMINI_API_KEY=your_key_here' to your .env file to enable LLM Q&A.",
            )

        start_time = time.perf_counter()

        try:
            client = self._get_client()

            # 1. New google.genai Client invocation
            if hasattr(client, "models") and hasattr(client.models, "generate_content"):
                config = {}
                if system_instruction:
                    config["system_instruction"] = system_instruction
                if temperature is not None:
                    config["temperature"] = temperature

                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config if config else None,
                )
                generated_text = response.text if hasattr(response, "text") and response.text else ""

            # 2. Legacy google.generativeai fallback
            else:
                model = client.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=system_instruction,
                    generation_config={"temperature": temperature},
                )
                response = model.generate_content(prompt)
                generated_text = response.text if hasattr(response, "text") and response.text else ""

            elapsed = round(time.perf_counter() - start_time, 3)

            if not generated_text:
                return LLMResponse(
                    text="",
                    model_name=self.model_name,
                    provider_name=self.get_provider_name(),
                    latency_seconds=elapsed,
                    error="Gemini returned an empty response. The content may have been filtered or blocked by safety guidelines.",
                )

            return LLMResponse(
                text=generated_text,
                model_name=self.model_name,
                provider_name=self.get_provider_name(),
                latency_seconds=elapsed,
            )

        except Exception as e:
            elapsed = round(time.perf_counter() - start_time, 3)
            err_msg = str(e)
            
            # Mask any accidental key reflection
            if self.api_key and self.api_key in err_msg:
                err_msg = err_msg.replace(self.api_key, "[REDACTED_API_KEY]")

            return LLMResponse(
                text="",
                model_name=self.model_name,
                provider_name=self.get_provider_name(),
                latency_seconds=elapsed,
                error=f"Gemini API Error: {err_msg}",
            )
