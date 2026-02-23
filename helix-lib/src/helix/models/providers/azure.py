"""
helix/models/providers/azure.py

Azure OpenAI provider.

Install:  pip install openai  (same SDK, different base URL)
Env vars: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION
          AZURE_OPENAI_DEPLOYMENT (optional, defaults to model name)
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, List, Optional

from helix.config import ModelResponse, TokenUsage, ToolCallRecord
from helix.errors import HelixProviderError
from helix.interfaces import LLMProvider


class AzureOpenAIProvider(LLMProvider):
    """
    Azure OpenAI — uses the same openai SDK but with Azure authentication.

    Configuration::

        agent = Agent(
            model=ModelConfig(primary="gpt-4o"),
            ...
        )
        # Set env vars:
        # AZURE_OPENAI_API_KEY=...
        # AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
        # AZURE_OPENAI_API_VERSION=2024-02-01
        # AZURE_OPENAI_DEPLOYMENT=your-deployment-name
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        deployment: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self._endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        self._api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self._deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncAzureOpenAI
                self._client = AsyncAzureOpenAI(
                    api_key=self._api_key,
                    azure_endpoint=self._endpoint,
                    api_version=self._api_version,
                )
            except ImportError:
                raise ImportError("pip install openai")
        return self._client

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        model: str = "gpt-4o",
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
        **kwargs,
    ) -> ModelResponse:
        try:
            client = self._get_client()
            deployment = self._deployment or model
            kwargs_ = dict(
                model=deployment, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            if tools:
                kwargs_["tools"] = [{"type": "function", "function": t} for t in tools]
            if response_format:
                kwargs_["response_format"] = response_format
            response = await client.chat.completions.create(**kwargs_)
            return self._normalize(response, model)
        except Exception as e:
            retryable = any(k in str(e).lower() for k in ("rate", "timeout", "503", "529"))
            raise HelixProviderError(model=model, provider="azure", original=e, retryable=retryable)

    async def stream(self, messages, model="gpt-4o", **kwargs) -> AsyncIterator[str]:
        try:
            client = self._get_client()
            deployment = self._deployment or model
            async with client.chat.completions.stream(model=deployment, messages=messages) as stream:
                async for event in stream:
                    if event.choices and event.choices[0].delta.content:
                        yield event.choices[0].delta.content
        except Exception as e:
            raise HelixProviderError(model=model, provider="azure", original=e)

    async def count_tokens(self, messages: List[Dict], model: str) -> int:
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model("gpt-4")
            return sum(len(enc.encode(m.get("content", "") or "")) for m in messages)
        except Exception:
            return sum(len((m.get("content") or "")) // 4 for m in messages)

    def supported_models(self) -> List[str]:
        return ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-35-turbo"]

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
            cached_tokens=getattr(getattr(raw.usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0,
        )
        finish = "tool_calls" if tool_calls else choice.finish_reason or "stop"
        return ModelResponse(content=content, tool_calls=tool_calls, usage=usage,
                             model=model, provider="azure", finish_reason=finish)
