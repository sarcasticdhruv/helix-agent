"""
helix/models/providers/groq.py

Groq provider — ultra-fast inference for open-source models.

Install:  pip install groq
Models:   llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768,
          gemma2-9b-it, llama3-groq-70b-8192-tool-use-preview
Env var:  GROQ_API_KEY
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, List, Optional

from helix.config import ModelResponse, TokenUsage, ToolCallRecord
from helix.errors import HelixProviderError
from helix.interfaces import LLMProvider


class GroqProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.environ.get("GROQ_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from groq import AsyncGroq
                self._client = AsyncGroq(api_key=self._api_key)
            except ImportError:
                raise ImportError("pip install groq")
        return self._client

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        model: str = "llama-3.3-70b-versatile",
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> ModelResponse:
        try:
            client = self._get_client()
            kwargs_ = dict(messages=messages, model=model, temperature=temperature, max_tokens=max_tokens)
            if tools:
                kwargs_["tools"] = [{"type": "function", "function": t} for t in tools]
            response = await client.chat.completions.create(**kwargs_)
            return self._normalize(response, model)
        except Exception as e:
            retryable = any(k in str(e).lower() for k in ("rate", "timeout", "503", "529"))
            raise HelixProviderError(model=model, provider="groq", original=e, retryable=retryable)

    async def stream(self, messages, model="llama-3.3-70b-versatile", **kwargs) -> AsyncIterator[str]:
        try:
            client = self._get_client()
            async with client.chat.completions.stream(messages=messages, model=model) as stream:
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
        except Exception as e:
            raise HelixProviderError(model=model, provider="groq", original=e)

    async def count_tokens(self, messages: List[Dict], model: str) -> int:
        text = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        return len(text) // 4

    def supported_models(self) -> List[str]:
        return [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-groq-70b-8192-tool-use-preview",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ]

    async def health(self) -> bool:
        try:
            client = self._get_client()
            await client.models.list()
            return True
        except Exception:
            return False

    def _normalize(self, raw, model: str) -> ModelResponse:
        import json
        choice = raw.choices[0]
        msg = choice.message
        content = msg.content or ""
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                tool_calls.append(ToolCallRecord(
                    id=tc.id, tool_name=tc.function.name, arguments=args, step=0
                ))
        usage = TokenUsage(
            prompt_tokens=raw.usage.prompt_tokens if raw.usage else 0,
            completion_tokens=raw.usage.completion_tokens if raw.usage else 0,
        )
        finish = "tool_calls" if tool_calls else "stop"
        return ModelResponse(content=content, tool_calls=tool_calls, usage=usage,
                             model=model, provider="groq", finish_reason=finish)
