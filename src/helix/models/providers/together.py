"""
helix/models/providers/together.py

Together AI provider — hosted open-source models via OpenAI-compatible API.

Install:  pip install together
Models:   meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo,
          mistralai/Mixtral-8x7B-Instruct-v0.1,
          Qwen/Qwen2.5-72B-Instruct-Turbo, and 200+ more
Env var:  TOGETHER_API_KEY
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from helix.config import ModelResponse, TokenUsage
from helix.errors import HelixProviderError
from helix.interfaces import LLMProvider


class TogetherProvider(LLMProvider):
    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from together import AsyncTogether

                self._client = AsyncTogether(api_key=self._api_key)
            except ImportError as err:
                raise ImportError("pip install together") from err
        return self._client

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> ModelResponse:
        try:
            client = self._get_client()
            kwargs_ = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            response = await client.chat.completions.create(**kwargs_)
            return self._normalize(response, model)
        except Exception as e:
            retryable = any(k in str(e).lower() for k in ("rate", "timeout", "503"))
            raise HelixProviderError(
                model=model, provider="together", original=e, retryable=retryable
            ) from e

    async def stream(
        self, messages, model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", **kwargs
    ) -> AsyncIterator[str]:
        try:
            client = self._get_client()
            stream = await client.chat.completions.create(
                model=model, messages=messages, stream=True
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise HelixProviderError(model=model, provider="together", original=e) from e

    async def count_tokens(self, messages: list[dict], model: str) -> int:
        text = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        return len(text) // 4

    def supported_models(self) -> list[str]:
        return [
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-R1",
            "google/gemma-2-27b-it",
        ]

    async def health(self) -> bool:
        try:
            client = self._get_client()
            await client.models.list()
            return True
        except Exception:
            return False

    def _normalize(self, raw, model: str) -> ModelResponse:
        choice = raw.choices[0]
        content = choice.message.content or ""
        usage = TokenUsage(
            prompt_tokens=raw.usage.prompt_tokens if raw.usage else 0,
            completion_tokens=raw.usage.completion_tokens if raw.usage else 0,
        )
        return ModelResponse(
            content=content, usage=usage, model=model, provider="together", finish_reason="stop"
        )
