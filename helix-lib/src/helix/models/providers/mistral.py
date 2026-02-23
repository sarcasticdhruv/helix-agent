"""
helix/models/providers/mistral.py

Mistral AI provider.

Install:  pip install mistralai
Models:   mistral-large-latest, mistral-small-latest, open-mistral-nemo,
          codestral-latest, pixtral-large-latest
Env var:  MISTRAL_API_KEY
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from helix.config import ModelResponse, TokenUsage, ToolCallRecord
from helix.errors import HelixProviderError
from helix.interfaces import LLMProvider


class MistralProvider(LLMProvider):
    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from mistralai import Mistral

                self._client = Mistral(api_key=self._api_key)
            except ImportError as err:
                raise ImportError("pip install mistralai") from err
        return self._client

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str = "mistral-large-latest",
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> ModelResponse:
        try:
            client = self._get_client()
            kwargs_ = {
                "messages": messages, "model": model, "temperature": temperature, "max_tokens": max_tokens
            }
            if tools:
                kwargs_["tools"] = [{"type": "function", "function": t} for t in tools]
            response = await client.chat.complete_async(**kwargs_)
            return self._normalize(response, model)
        except Exception as e:
            retryable = any(k in str(e).lower() for k in ("rate", "timeout", "503"))
            raise HelixProviderError(
                model=model, provider="mistral", original=e, retryable=retryable
            ) from e

    async def stream(self, messages, model="mistral-large-latest", **kwargs) -> AsyncIterator[str]:
        try:
            client = self._get_client()
            async for chunk in await client.chat.stream_async(messages=messages, model=model):
                delta = chunk.data.choices[0].delta.content if chunk.data.choices else ""
                if delta:
                    yield delta
        except Exception as e:
            raise HelixProviderError(model=model, provider="mistral", original=e) from e

    async def count_tokens(self, messages: list[dict], model: str) -> int:
        text = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        return len(text) // 4

    def supported_models(self) -> list[str]:
        return [
            "mistral-large-latest",
            "mistral-small-latest",
            "open-mistral-nemo",
            "codestral-latest",
            "pixtral-large-latest",
            "open-mixtral-8x22b",
        ]

    async def health(self) -> bool:
        try:
            client = self._get_client()
            await client.models.list_async()
            return True
        except Exception:
            return False

    def _normalize(self, raw, model: str) -> ModelResponse:
        choice = raw.choices[0]
        msg = choice.message
        content = msg.content or ""
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = (
                        json.loads(tc.function.arguments)
                        if isinstance(tc.function.arguments, str)
                        else tc.function.arguments
                    )
                except Exception:
                    args = {}
                tool_calls.append(
                    ToolCallRecord(
                        id=getattr(tc, "id", ""), tool_name=tc.function.name, arguments=args, step=0
                    )
                )
        usage = TokenUsage(
            prompt_tokens=raw.usage.prompt_tokens if raw.usage else 0,
            completion_tokens=raw.usage.completion_tokens if raw.usage else 0,
        )
        finish = "tool_calls" if tool_calls else "stop"
        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=model,
            provider="mistral",
            finish_reason=finish,
        )
