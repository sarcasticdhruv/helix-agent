"""
helix/models/providers/openai.py

OpenAI LLMProvider implementation.

Wraps the openai SDK and normalizes all responses into ModelResponse.
All SDK exceptions are caught and re-raised as HelixProviderError.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from helix.config import ModelResponse, TokenUsage, ToolCallRecord
from helix.errors import HelixProviderError
from helix.interfaces import LLMProvider


class OpenAIProvider(LLMProvider):
    """
    Wraps openai.AsyncOpenAI.
    Requires OPENAI_API_KEY environment variable.
    """

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        try:
            import openai

            self._client = openai.AsyncOpenAI(
                api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=base_url,
            )
        except ImportError as err:
            raise ImportError("openai package required. Install with: pip install openai") from err

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> ModelResponse:
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
                kwargs["tool_choice"] = "auto"
            if response_format:
                kwargs["response_format"] = response_format

            raw = await self._client.chat.completions.create(**kwargs)

            return self._normalize(raw, model)

        except Exception as e:
            err_str = str(e).lower()
            retryable = "rate" in err_str or "timeout" in err_str or "529" in err_str
            raise HelixProviderError(
                provider="openai",
                model=model,
                reason=str(e),
                retryable=retryable,
            ) from e

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        try:
            async with self._client.chat.completions.stream(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ) as stream:
                async for event in stream:
                    if (
                        hasattr(event, "delta")
                        and hasattr(event.delta, "content")
                        and event.delta.content
                    ):
                        yield event.delta.content
        except Exception as e:
            raise HelixProviderError(
                provider="openai", model=model, reason=str(e), retryable=False
            ) from e

    def count_tokens(self, messages: list[dict[str, Any]], model: str) -> int:
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(model)
            total = 0
            for m in messages:
                content = m.get("content", "")
                if isinstance(content, str):
                    total += len(enc.encode(content))
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            total += len(enc.encode(block.get("text", "")))
            return total
        except Exception:
            # Fallback: characters / 4
            return sum(len(str(m.get("content", ""))) // 4 for m in messages)

    def supported_models(self) -> list[str]:
        return ["gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o3-mini"]

    async def health(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False

    def _normalize(self, raw: Any, model: str) -> ModelResponse:
        """Convert raw OpenAI response to ModelResponse."""
        choice = raw.choices[0]
        message = choice.message

        tool_calls: list[ToolCallRecord] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                import json

                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(
                    ToolCallRecord(
                        tool_name=tc.function.name,
                        arguments=args,
                    )
                )

        usage = TokenUsage(
            prompt_tokens=raw.usage.prompt_tokens if raw.usage else 0,
            completion_tokens=raw.usage.completion_tokens if raw.usage else 0,
            cached_tokens=getattr(
                getattr(raw.usage, "prompt_tokens_details", None), "cached_tokens", 0
            )
            if raw.usage
            else 0,
        )

        return ModelResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            usage=usage,
            model=model,
            provider="openai",
            finish_reason=choice.finish_reason or "stop",
        )
