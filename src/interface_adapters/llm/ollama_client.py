"""Ollama LLM client for local inference."""

import json
from typing import Any, Dict, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from frameworks.config import OLLAMA_BASE_URL


class OllamaClient:
    """Thin wrapper around Ollama's REST API.

    Supports both generation and structured JSON output.
    """

    def __init__(self, base_url: str | None = None, timeout: int = 120) -> None:
        self._base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
        self._timeout = timeout

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
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
        payload: Dict[str, Any] = {
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

        response = requests.post(url, json=payload, timeout=self._timeout)
        response.raise_for_status()
        data = response.json()

        text = data.get("response", "").strip()
        if json_mode and text:
            # Validate it's parseable JSON
            json.loads(text)
        return text

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
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
        response = requests.post(url, json=payload, timeout=self._timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "").strip()
