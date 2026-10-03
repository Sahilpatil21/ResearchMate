"""Factory for creating and resolving LLM providers in ResearchMate.

Defaults to Google Gemini, but supports OpenAI-compatible endpoints and local Ollama.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

from src.llm.gemini_provider import GeminiProvider
from src.llm.ollama_provider import OllamaProvider
from src.llm.openai_provider import OpenAICompatibleProvider
from src.llm.provider import BaseLLMProvider

load_dotenv()


def get_llm_provider(
    provider_name: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
) -> BaseLLMProvider:
    """Instantiate and return the configured LLM provider.

    Args:
        provider_name: LLM provider name ('gemini', 'openai', 'ollama'). Defaults to env LLM_PROVIDER or 'gemini'.
        api_key: Optional API key override.
        model_name: Optional model name override.
        base_url: Optional base URL override.

    Returns:
        Instance of BaseLLMProvider.
    """
    provider = (
        provider_name
        or os.getenv("LLM_PROVIDER", "gemini").strip().lower()
        or "gemini"
    )

    if provider == "gemini":
        return GeminiProvider(api_key=api_key, model_name=model_name)
    elif provider in ("openai", "openai-compatible"):
        return OpenAICompatibleProvider(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
        )
    elif provider == "ollama":
        return OllamaProvider(
            model_name=model_name,
            base_url=base_url,
        )
    else:
        # Fallback to Gemini with warning
        return GeminiProvider(api_key=api_key, model_name=model_name)
