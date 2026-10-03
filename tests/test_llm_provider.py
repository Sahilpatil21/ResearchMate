"""Unit tests for Stage 6 LLM Provider Abstraction.

Verifies provider abstraction, configuration handling, error resilience, key redaction,
and mocked inference for Google Gemini, OpenAI-compatible, and Ollama providers.
NO REAL API CALLS are made during test execution.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from src.llm.factory import get_llm_provider
from src.llm.gemini_provider import GeminiProvider
from src.llm.ollama_provider import OllamaProvider
from src.llm.openai_provider import OpenAICompatibleProvider
from src.llm.provider import BaseLLMProvider, LLMResponse


def test_llm_response_model():
    """Test LLMResponse properties and success determination."""
    success_resp = LLMResponse(
        text="This is a generated answer [1].",
        model_name="gemini-2.5-flash",
        provider_name="gemini",
        latency_seconds=0.45,
    )
    assert success_resp.is_success is True
    assert success_resp.error is None
    assert success_resp.text == "This is a generated answer [1]."

    fail_resp = LLMResponse(
        text="",
        model_name="gemini-2.5-flash",
        provider_name="gemini",
        error="API Key missing",
    )
    assert fail_resp.is_success is False
    assert fail_resp.error == "API Key missing"


def test_gemini_unconfigured_availability():
    """Test GeminiProvider behavior when API key is missing."""
    provider = GeminiProvider(api_key="", model_name="gemini-2.5-flash")
    assert provider.is_available() is False

    response = provider.generate(prompt="What is ResNet?")
    assert response.is_success is False
    assert "GEMINI_API_KEY is missing" in response.error
    assert response.provider_name == "gemini"


def test_gemini_mocked_generation():
    """Test GeminiProvider generation with mocked Google GenAI client."""
    provider = GeminiProvider(api_key="mocked_fake_gemini_key_12345", model_name="gemini-2.5-flash")
    assert provider.is_available() is True

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "ResNet utilizes identity shortcut connections [1]."
    mock_client.models.generate_content.return_value = mock_resp

    with patch.object(provider, "_get_client", return_value=mock_client):
        response = provider.generate(
            prompt="Explain ResNet architecture.",
            system_instruction="You are an academic assistant.",
        )

        assert response.is_success is True
        assert "ResNet utilizes identity shortcut connections [1]." in response.text
        assert response.model_name == "gemini-2.5-flash"
        assert response.provider_name == "gemini"
        assert response.latency_seconds >= 0.0


def test_gemini_key_redaction_in_errors():
    """Ensure GeminiProvider never leaks the API key in exception messages."""
    fake_key = "secret_gemini_key_9999"
    provider = GeminiProvider(api_key=fake_key)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception(
        f"Unauthorized access with key {fake_key}: quota exceeded"
    )

    with patch.object(provider, "_get_client", return_value=mock_client):
        response = provider.generate(prompt="Test query")

        assert response.is_success is False
        assert fake_key not in response.error
        assert "[REDACTED_API_KEY]" in response.error


def test_openai_unconfigured_and_mocked():
    """Test OpenAICompatibleProvider unconfigured and mocked execution."""
    unconfigured = OpenAICompatibleProvider(api_key="")
    assert unconfigured.is_available() is False
    resp_unconf = unconfigured.generate(prompt="test")
    assert resp_unconf.is_success is False
    assert "API key is missing" in resp_unconf.error

    # Mocked generation
    provider = OpenAICompatibleProvider(api_key="sk-mocked-key")
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "OpenAI answer [1]."
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    mock_completion.usage.prompt_tokens = 100
    mock_completion.usage.completion_tokens = 20
    mock_client.chat.completions.create.return_value = mock_completion

    with patch.object(provider, "_get_client", return_value=mock_client):
        response = provider.generate(prompt="test")
        assert response.is_success is True
        assert response.text == "OpenAI answer [1]."
        assert response.prompt_tokens == 100
        assert response.completion_tokens == 20


def test_ollama_mocked_generation():
    """Test OllamaProvider local mocked execution."""
    provider = OllamaProvider(model_name="llama3.2", base_url="http://localhost:11434")
    assert provider.get_provider_name() == "ollama"

    mock_http_response = MagicMock()
    mock_http_response.read.return_value = b'{"message": {"content": "Ollama answer [1]."}, "prompt_eval_count": 50, "eval_count": 15}'
    mock_http_response.__enter__.return_value = mock_http_response
    mock_http_response.__exit__.return_value = None

    with patch("urllib.request.urlopen", return_value=mock_http_response):
        response = provider.generate(prompt="test")
        assert response.is_success is True
        assert response.text == "Ollama answer [1]."


def test_llm_factory():
    """Test get_llm_provider factory resolution."""
    p_gemini = get_llm_provider("gemini")
    assert isinstance(p_gemini, GeminiProvider)
    assert p_gemini.get_provider_name() == "gemini"

    p_openai = get_llm_provider("openai")
    assert isinstance(p_openai, OpenAICompatibleProvider)
    assert p_openai.get_provider_name() == "openai"

    p_ollama = get_llm_provider("ollama")
    assert isinstance(p_ollama, OllamaProvider)
    assert p_ollama.get_provider_name() == "ollama"
