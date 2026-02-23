"""
helix/models/providers/openai_compat.py

Generic OpenAI-compatible provider.

Works with any server that implements the OpenAI chat completions API:
  - LM Studio         (http://localhost:1234/v1)
  - vLLM              (http://localhost:8000/v1)
  - LocalAI           (http://localhost:8080/v1)
  - OpenRouter        (https://openrouter.ai/api/v1)
  - Perplexity AI     (https://api.perplexity.ai)
  - Fireworks AI      (https://api.fireworks.ai/inference/v1)
  - DeepSeek          (https://api.deepseek.com/v1)
  - xAI Grok          (https://api.x.ai/v1)
  - Moonshot / Kimi   (https://api.moonshot.cn/v1)
  - Any other OpenAI-compatible endpoint

Usage::

    from helix.models.providers.openai_compat import OpenAICompatProvider

    provider = OpenAICompatProvider(
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-...",
    )

Env vars:
    OPENAI_COMPAT_BASE_URL   — endpoint URL
    OPENAI_COMPAT_API_KEY    — API key (optional for local servers)
    OPENAI_COMPAT_MODEL      — default model name
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from helix.config import ModelResponse, TokenUsage, ToolCallRecord
from helix.errors import HelixProviderError
from helix.interfaces import LLMProvider

# Well-known OpenAI-compatible endpoints
KNOWN_ENDPOINTS: dict[str, dict[str, str]] = {
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "env": "OPENROUTER_API_KEY"},
    "perplexity": {"base_url": "https://api.perplexity.ai", "env": "PERPLEXITY_API_KEY"},
    "fireworks": {"base_url": "https://api.fireworks.ai/inference/v1", "env": "FIREWORKS_API_KEY"},
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "env": "DEEPSEEK_API_KEY"},
    "xai": {"base_url": "https://api.x.ai/v1", "env": "XAI_API_KEY"},
    "moonshot": {"base_url": "https://api.moonshot.cn/v1", "env": "MOONSHOT_API_KEY"},
    "lmstudio": {"base_url": "http://localhost:1234/v1", "env": ""},
    "vllm": {"base_url": "http://localhost:8000/v1", "env": ""},
    "localai": {"base_url": "http://localhost:8080/v1", "env": ""},
}


class OpenAICompatProvider(LLMProvider):
    """
    Drop-in provider for any OpenAI-compatible API endpoint.

    Examples::

        # OpenRouter (access 100+ models with one key)
        provider = OpenAICompatProvider.from_preset("openrouter")

        # DeepSeek
        provider = OpenAICompatProvider.from_preset("deepseek")

        # Local LM Studio
        provider = OpenAICompatProvider.from_preset("lmstudio")

        # Custom endpoint
        provider = OpenAICompatProvider(
            base_url="https://my-llm-server.com/v1",
            api_key="my-key",
        )
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        default_model: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or os.environ.get("OPENAI_COMPAT_API_KEY", "none")
        self._default_model = default_model or os.environ.get("OPENAI_COMPAT_MODEL", "default")
        self._extra_headers = extra_headers or {}
        self._client = None

    @classmethod
    def from_preset(cls, name: str, api_key: str | None = None) -> OpenAICompatProvider:
        """
        Create a provider from a well-known preset name.

        Presets: openrouter, perplexity, fireworks, deepseek, xai,
                 moonshot, lmstudio, vllm, localai
        """
        if name not in KNOWN_ENDPOINTS:
            raise ValueError(f"Unknown preset '{name}'. Known: {list(KNOWN_ENDPOINTS)}")
        info = KNOWN_ENDPOINTS[name]
        key = api_key or (os.environ.get(info["env"]) if info["env"] else "none")
        return cls(base_url=info["base_url"], api_key=key)

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(
                    api_key=self._api_key,
                    base_url=self._base_url,
                    default_headers=self._extra_headers,
                )
            except ImportError as err:
                raise ImportError("pip install openai") from err
        return self._client

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict | None = None,
        **kwargs,
    ) -> ModelResponse:
        target = model or self._default_model
        try:
            client = self._get_client()
            kwargs_ = {
                "model": target,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                kwargs_["tools"] = [{"type": "function", "function": t} for t in tools]
            if response_format:
                kwargs_["response_format"] = response_format
            response = await client.chat.completions.create(**kwargs_)
            return self._normalize(response, target)
        except Exception as e:
            retryable = any(
                k in str(e).lower() for k in ("rate", "timeout", "503", "529", "overload")
            )
            raise HelixProviderError(
                model=target, provider=self._base_url, original=e, retryable=retryable
            ) from e

    async def stream(self, messages, model=None, **kwargs) -> AsyncIterator[str]:
        target = model or self._default_model
        try:
            client = self._get_client()
            async with client.chat.completions.stream(model=target, messages=messages) as stream:
                async for event in stream:
                    if event.choices and event.choices[0].delta.content:
                        yield event.choices[0].delta.content
        except Exception as e:
            raise HelixProviderError(model=target, provider=self._base_url, original=e) from e

    async def count_tokens(self, messages: list[dict], model: str) -> int:
        text = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        return len(text) // 4

    def supported_models(self) -> list[str]:
        return [self._default_model]

    async def health(self) -> bool:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{self._base_url}/models",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                return resp.status_code < 500
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
                    args = json.loads(tc.function.arguments)
                except Exception:
                    args = {}
                tool_calls.append(
                    ToolCallRecord(
                        id=getattr(tc, "id", ""),
                        tool_name=tc.function.name,
                        arguments=args,
                        step=0,
                    )
                )
        usage = TokenUsage(
            prompt_tokens=raw.usage.prompt_tokens if raw.usage else 0,
            completion_tokens=raw.usage.completion_tokens if raw.usage else 0,
        )
        finish = "tool_calls" if tool_calls else choice.finish_reason or "stop"
        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=model,
            provider=self._base_url,
            finish_reason=finish,
        )
