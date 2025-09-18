"""Utility module that provides pluggable LLM client implementations."""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping

import requests
from openai import OpenAI

logger = logging.getLogger(__name__)


Message = Mapping[str, str]


class BaseLLMClient(ABC):
    """Abstract base class for chat-completion style LLM clients."""

    provider_name: str = "base"

    @abstractmethod
    def generate(self, messages: Iterable[Message]) -> str:
        """Generate a chat completion response."""


class OpenAIClient(BaseLLMClient):
    """LLM client that proxies requests to the OpenAI Chat Completions API."""

    provider_name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 500,
        timeout: int = 10,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI API 金鑰未設置")
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout

    def generate(self, messages: Iterable[Message]) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=list(messages),
            max_tokens=self._max_tokens,
            timeout=self._timeout,
        )
        return response.choices[0].message.content


class OllamaClient(BaseLLMClient):
    """LLM client that calls a local Ollama server."""

    provider_name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: int = 30) -> None:
        self._base_url = base_url.rstrip("/") or "http://localhost:11434"
        self._model = model
        self._timeout = timeout

    def generate(self, messages: Iterable[Message]) -> str:
        payload = {
            "model": self._model,
            "messages": list(messages),
            "stream": False,
        }
        response = requests.post(
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, Mapping):
            message = data.get("message")
            if isinstance(message, Mapping):
                content = message.get("content")
                if isinstance(content, str):
                    return content
            error = data.get("error")
            if error:
                raise RuntimeError(f"Ollama error: {error}")
        raise RuntimeError("Unexpected response from Ollama 伺服器")


def get_llm_client() -> BaseLLMClient:
    """Factory that builds an LLM client based on environment configuration."""

    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider == "openai":
        return OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "500")),
            timeout=int(os.getenv("OPENAI_TIMEOUT", "10")),
        )
    if provider == "ollama":
        return OllamaClient(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            timeout=int(os.getenv("OLLAMA_TIMEOUT", "30")),
        )

    logger.error("不支援的 LLM_PROVIDER: %s", provider)
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")
