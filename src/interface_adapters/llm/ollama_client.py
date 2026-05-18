"""Ollama LLM client for local inference."""

from __future__ import annotations

import json
from typing import Any

import requests
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from frameworks.config import OLLAMA_BASE_URL

logger = structlog.get_logger(__name__)


class OllamaClient:
    """Thin wrapper around Ollama's REST API.

    Supports both generation and structured JSON output.
    Uses a persistent requests.Session for connection pooling.
    """

    def __init__(self, base_url: str | None = None, timeout: int = 120) -> None:
        """Initialize client with optional base URL and timeout."""
        self._base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError)),
        reraise=True,
    )
    def generate(
        self,
        model: str,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str:
        """Generate text from a prompt.

        Args:
            model: Ollama model name (e.g., 'mistral:7b').
            prompt: User prompt.
            system: Optional system message.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            json_mode: If True, request JSON output and parse it.

        Returns:
            Generated text string.
        """
        url = f"{self._base_url}/api/generate"
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        try:
            response = self._session.post(url, json=payload, timeout=self._timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.error("ollama_generate_request_failed", error=str(exc), model=model)
            raise

        data: dict[str, Any] = response.json()
        text = str(data.get("response", "")).strip()

        if json_mode and text:
            # Validate it's parseable JSON
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                logger.warning("ollama_generate_invalid_json", text=text[:200])
                raise ValueError(f"Ollama returned invalid JSON: {exc}") from exc

        return text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError)),
        reraise=True,
    )
    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Chat-style generation with message history.

        Args:
            model: Ollama model name.
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Assistant's response text.
        """
        url = f"{self._base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            response = self._session.post(url, json=payload, timeout=self._timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.error("ollama_chat_request_failed", error=str(exc), model=model)
            raise

        data: dict[str, Any] = response.json()
        message: dict[str, Any] = data.get("message", {})
        return str(message.get("content", "")).strip()
