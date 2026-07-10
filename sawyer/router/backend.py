"""Sawyer Inference Backend — proxies to an OpenAI-compatible server.

Supports any backend that implements the OpenAI chat completions API:
- llama.cpp server (local GPU)
- vLLM (local or remote)
- OpenAI API (direct proxy)
- RunPod serverless endpoints

This is the bridge between the Sawyer pipeline (auth, token accounting,
routing) and the actual model serving infrastructure.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Default backend URL — override with SAWYER_BACKEND_URL env var
DEFAULT_BACKEND_URL = "http://localhost:8080/v1"
DEFAULT_BACKEND_API_KEY = ""


@dataclass
class BackendConfig:
    """Configuration for the inference backend."""

    url: str = DEFAULT_BACKEND_URL
    api_key: str = DEFAULT_BACKEND_API_KEY
    timeout_seconds: float = 120.0
    max_retries: int = 3
    # Model name mapping: Sawyer model name -> backend model name
    model_map: dict[str, str] | None = None


class InferenceBackend:
    """OpenAI-compatible inference backend.

    Sends chat completion requests to a backend server and returns
    the response text along with token counts.
    """

    def __init__(self, config: BackendConfig | None = None) -> None:
        self.config = config or BackendConfig(
            url=os.environ.get("SAWYER_BACKEND_URL", DEFAULT_BACKEND_URL),
            api_key=os.environ.get("SAWYER_BACKEND_API_KEY", DEFAULT_BACKEND_API_KEY),
        )
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.config.url,
                headers=headers,
                timeout=httpx.Timeout(self.config.timeout_seconds),
            )
        return self._client

    def _map_model(self, model: str) -> str:
        """Map Sawyer model name to backend model name."""
        if self.config.model_map and model in self.config.model_map:
            return self.config.model_map[model]
        # Pass through the model name as-is if no mapping
        return model

    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Send a chat completion request to the backend.

        Returns:
            dict with keys: text, prompt_tokens, completion_tokens, total_tokens,
            latency_ms, model, finish_reason
        """
        backend_model = self._map_model(model)
        client = await self._get_client()

        payload = {
            "model": backend_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        start_time = time.time()
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = await client.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()

                latency_ms = (time.time() - start_time) * 1000

                # Parse OpenAI-format response
                choice = data.get("choices", [{}])[0]
                message = choice.get("message", {})
                text = message.get("content", "")
                finish_reason = choice.get("finish_reason", "stop")

                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)

                # Estimate tokens if backend doesn't report them
                if total_tokens == 0:
                    # Rough estimate: ~4 chars per token
                    prompt_tokens = prompt_tokens or len(str(messages)) // 4
                    completion_tokens = completion_tokens or len(text) // 4
                    total_tokens = prompt_tokens + completion_tokens

                return {
                    "text": text,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "latency_ms": latency_ms,
                    "model": data.get("model", backend_model),
                    "finish_reason": finish_reason,
                    "node_id": "backend",
                }

            except httpx.HTTPStatusError as e:
                logger.warning(
                    "Backend returned %d (attempt %d/%d): %s",
                    e.response.status_code,
                    attempt + 1,
                    self.config.max_retries,
                    e.response.text[:200],
                )
                last_error = e
                if e.response.status_code == 429:
                    # Rate limited — wait and retry
                    import asyncio
                    await asyncio.sleep(2 ** attempt)
                elif e.response.status_code >= 500:
                    # Server error — retry
                    import asyncio
                    await asyncio.sleep(1)
                else:
                    # Client error — don't retry
                    break

            except httpx.RequestError as e:
                logger.warning(
                    "Backend request failed (attempt %d/%d): %s",
                    attempt + 1,
                    self.config.max_retries,
                    e,
                )
                last_error = e
                import asyncio
                await asyncio.sleep(1)

        # All retries exhausted
        raise RuntimeError(
            f"Backend inference failed after {self.config.max_retries} attempts: {last_error}"
        ) from last_error

    async def health_check(self) -> dict[str, Any]:
        """Check if the backend is healthy and which models are available."""
        client = await self._get_client()
        try:
            response = await client.get("/models")
            response.raise_for_status()
            data = response.json()
            models = [m["id"] for m in data.get("data", [])]
            return {"status": "healthy", "models": models}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "models": []}

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()