"""LLM provider package for ResearchMate.

Provides modular providers for Google Gemini, OpenAI-compatible APIs, and local Ollama.
"""

from src.llm.factory import get_llm_provider
from src.llm.gemini_provider import GeminiProvider
from src.llm.ollama_provider import OllamaProvider
from src.llm.openai_provider import OpenAICompatibleProvider
from src.llm.provider import BaseLLMProvider, LLMResponse

__all__ = [
    "BaseLLMProvider",
    "LLMResponse",
    "GeminiProvider",
    "OpenAICompatibleProvider",
    "OllamaProvider",
    "get_llm_provider",
]
