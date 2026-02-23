"""
helix/models/providers/ollama.py

Ollama provider — run open-source models locally, zero API cost.

Install:  Download Ollama from https://ollama.com
          Then: ollama pull llama3.2 / ollama pull mistral / etc.
Models:   Any model pulled via `ollama pull <model>`
Env var:  OLLAMA_HOST (default: http://localhost:11434)
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from helix.config import ModelResponse, TokenUsage, ToolCallRecord
from helix.errors import HelixProviderError
from helix.interfaces import LLMProvider

_DEFAULT_HOST = "http://localhost:11434"


class OllamaProvider(LLMProvider):
    """
    Ollama local model provider. No API key needed.
    Start Ollama: `ollama serve`
    Pull a model: `ollama pull llama3.2`
    """

    def __init__(self, host: str | None = None) -> None:
        self._host = host or os.environ.get("OLLAMA_HOST", _DEFAULT_HOST)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str = "llama3.2",
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> ModelResponse:
        try:
            import httpx

            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
            if tools:
                payload["tools"] = [{"type": "function", "function": t} for t in tools]

            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(f"{self._host}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
            return self._normalize(data, model)
        except Exception as e:
            retryable = "timeout" in str(e).lower() or "connect" in str(e).lower()
            raise HelixProviderError(
                model=model, provider="ollama", original=e, retryable=retryable
            ) from e

    async def stream(self, messages, model="llama3.2", **kwargs) -> AsyncIterator[str]:
        try:
            import json as _json

            import httpx

            payload = {"model": model, "messages": messages, "stream": True}
            async with (
                httpx.AsyncClient(timeout=120.0) as client,
                client.stream("POST", f"{self._host}/api/chat", json=payload) as resp,
            ):
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = _json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                    except Exception:
                        continue
        except Exception as e:
            raise HelixProviderError(model=model, provider="ollama", original=e) from e

    async def count_tokens(self, messages: list[dict], model: str) -> int:
        text = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        return len(text) // 4

    async def list_local_models(self) -> list[str]:
        """Return models currently pulled in Ollama."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self._host}/api/tags")
                data = resp.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    def supported_models(self) -> list[str]:
        return [
            "llama3.2",
            "llama3.2:1b",
            "llama3.1",
            "llama3.1:8b",
            "llama3.1:70b",
            "mistral",
            "mistral-nemo",
            "mixtral",
            "qwen2.5",
            "qwen2.5-coder",
            "gemma2",
            "gemma2:2b",
            "phi3",
            "phi3.5",
            "deepseek-r1",
            "deepseek-coder-v2",
            "codellama",
            "starcoder2",
        ]

    async def health(self) -> bool:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._host}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    def _normalize(self, data: dict, model: str) -> ModelResponse:
        import json as _json

        msg = data.get("message", {})
        content = msg.get("content", "")
        tool_calls = []

        # Ollama tool_calls support (newer versions)
        raw_tools = msg.get("tool_calls", [])
        for tc in raw_tools:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = _json.loads(args)
                except Exception:
                    args = {}
            tool_calls.append(
                ToolCallRecord(id="", tool_name=fn.get("name", ""), arguments=args, step=0)
            )

        usage = TokenUsage(
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )
        finish = "tool_calls" if tool_calls else "stop"
        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=model,
            provider="ollama",
            finish_reason=finish,
        )
