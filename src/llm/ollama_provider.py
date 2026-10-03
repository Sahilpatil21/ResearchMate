"""Ollama / Local LLM provider for ResearchMate.

Provides local model inference via Ollama's HTTP API (e.g. llama3.2, mistral, qwen2.5).
No cloud API key required.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
from typing import Any, Dict, Optional

from dotenv import load_dotenv

from src.llm.provider import BaseLLMProvider, LLMResponse

load_dotenv()


class OllamaProvider(BaseLLMProvider):
    """Provider for locally running Ollama models."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """Initialize the Ollama provider.

        Args:
            model_name: Model identifier in Ollama (defaults to OLLAMA_MODEL or 'llama3.2').
            base_url: Ollama server endpoint (defaults to OLLAMA_BASE_URL or 'http://localhost:11434').
        """
        self.model_name = (
            model_name
            or os.getenv("OLLAMA_MODEL", "")
            or os.getenv("LLM_MODEL", "")
            or "llama3.2"
        ).strip()
        self.base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL", "")
            or os.getenv("LLM_BASE_URL", "")
            or "http://localhost:11434"
        ).rstrip("/")

    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/version", method="GET")
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_model_name(self) -> str:
        """Return active model name."""
        return self.model_name

    def get_provider_name(self) -> str:
        """Return provider identifier."""
        return "ollama"

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Generate answer via Ollama generate / chat endpoint.

        Args:
            prompt: User prompt containing research context and query.
            system_instruction: High-priority system instructions.
            temperature: Sampling temperature.

        Returns:
            LLMResponse object with generated text or error details.
        """
        start_time = time.perf_counter()

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": [],
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if system_instruction:
            payload["messages"].append({"role": "system", "content": system_instruction})
        payload["messages"].append({"role": "user", "content": prompt})

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            elapsed = round(time.perf_counter() - start_time, 3)
            generated_text = ""
            if "message" in result and "content" in result["message"]:
                generated_text = result["message"]["content"]
            elif "response" in result:
                generated_text = result["response"]

            prompt_tokens = result.get("prompt_eval_count")
            comp_tokens = result.get("eval_count")

            if not generated_text:
                return LLMResponse(
                    text="",
                    model_name=self.model_name,
                    provider_name=self.get_provider_name(),
                    latency_seconds=elapsed,
                    error="Ollama returned an empty response.",
                )

            return LLMResponse(
                text=generated_text,
                model_name=self.model_name,
                provider_name=self.get_provider_name(),
                latency_seconds=elapsed,
                prompt_tokens=prompt_tokens,
                completion_tokens=comp_tokens,
            )

        except urllib.error.URLError as e:
            elapsed = round(time.perf_counter() - start_time, 3)
            return LLMResponse(
                text="",
                model_name=self.model_name,
                provider_name=self.get_provider_name(),
                latency_seconds=elapsed,
                error=f"Ollama connection error at '{self.base_url}': {e}. Ensure Ollama is running.",
            )
        except Exception as e:
            elapsed = round(time.perf_counter() - start_time, 3)
            return LLMResponse(
                text="",
                model_name=self.model_name,
                provider_name=self.get_provider_name(),
                latency_seconds=elapsed,
                error=f"Ollama error: {e}",
            )
