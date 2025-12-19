from __future__ import annotations
from typing import Optional, Union, Dict, Any, List
import json
import httpx


class OpenAICompatClient:
    """
    Minimal OpenAI-compatible chat client (works with Ollama / vLLM / llama.cpp servers that expose
    the /chat/completions route).

    Features:
    - Accepts Ollama "options" (e.g., num_thread, num_batch, num_ctx, temperature, repeat_penalty, keep_alive, etc.)
    - Optional stream flag passthrough (defaults to non-streaming unless you set it)
    - Backward-compatible defaults
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        options: Union[str, Dict[str, Any], None] = None,  # Ollama runtime options
        stream: Optional[bool] = None,                     # top-level "stream" param passthrough
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.options = options
        self.stream = stream
        self.extra_headers = extra_headers or {}
        self._client = httpx.Client(timeout=timeout)

    def set_options(self, options: Union[str, Dict[str, Any], None]) -> None:
        """Update Ollama/OpenAI runtime options at runtime."""
        self.options = options

    def set_stream(self, stream: Optional[bool]) -> None:
        """Enable/disable top-level streaming (if your backend supports it)."""
        self.stream = stream

    def _parse_options(self) -> Optional[Dict[str, Any]]:
        """Coerce `self.options` into a dict if it's a JSON string; otherwise return as-is."""
        if self.options is None:
            return None
        if isinstance(self.options, dict):
            return self.options
        if isinstance(self.options, str):
            try:
                return json.loads(self.options)
            except Exception:
                # If the user passed a non-JSON string, ignore silently to remain robust.
                return None
        return None

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.2,
        max_tokens: int = 256,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Send a chat completion request to an OpenAI-compatible endpoint.

        Notes:
        - `temperature` and `max_tokens` are still sent at the top level for OpenAI compatibility.
        - Ollama-specific runtime config goes in `payload["options"]`.
        - If `self.stream` is set, it will be forwarded as `payload["stream"]`.
          (This client returns only the final text; streaming responses are *not* iterated here.)
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        if stop:
            payload["stop"] = stop

        # Top-level stream flag (if you want to request streaming from the server)
        if self.stream is not None:
            payload["stream"] = bool(self.stream)

        # Ollama "options" block (num_thread, num_batch, num_ctx, repeat_penalty, keep_alive, etc.)
        opts = self._parse_options()
        if opts:
            payload["options"] = opts

        resp = self._client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

        # Non-streaming path: return the first choice content (standard OpenAI format)
        return data["choices"][0]["message"]["content"]
