"""
helix/models/providers/cohere.py

Cohere provider.

Install:  pip install cohere
Models:   command-r-plus-08-2024, command-r-08-2024, command-light
Env var:  COHERE_API_KEY
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, List, Optional

from helix.config import ModelResponse, TokenUsage, ToolCallRecord
from helix.errors import HelixProviderError
from helix.interfaces import LLMProvider


class CohereProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or os.environ.get("COHERE_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import cohere
                self._client = cohere.AsyncClientV2(api_key=self._api_key)
            except ImportError:
                raise ImportError("pip install cohere")
        return self._client

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        model: str = "command-r-plus-08-2024",
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> ModelResponse:
        try:
            client = self._get_client()
            # Convert to Cohere message format
            cohere_messages = self._convert_messages(messages)
            cohere_tools = self._convert_tools(tools) if tools else None

            kwargs_ = dict(model=model, messages=cohere_messages, temperature=temperature, max_tokens=max_tokens)
            if cohere_tools:
                kwargs_["tools"] = cohere_tools
            response = await client.chat(**kwargs_)
            return self._normalize(response, model)
        except Exception as e:
            retryable = any(k in str(e).lower() for k in ("rate", "timeout", "503"))
            raise HelixProviderError(model=model, provider="cohere", original=e, retryable=retryable)

    async def stream(self, messages, model="command-r-plus-08-2024", **kwargs) -> AsyncIterator[str]:
        try:
            client = self._get_client()
            cohere_messages = self._convert_messages(messages)
            async for event in client.chat_stream(model=model, messages=cohere_messages):
                if hasattr(event, "text") and event.text:
                    yield event.text
        except Exception as e:
            raise HelixProviderError(model=model, provider="cohere", original=e)

    async def count_tokens(self, messages: List[Dict], model: str) -> int:
        text = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        return len(text) // 4

    def supported_models(self) -> List[str]:
        return ["command-r-plus-08-2024", "command-r-08-2024", "command-r", "command-light", "command"]

    async def health(self) -> bool:
        try:
            import cohere
            cohere.AsyncClientV2(api_key=self._api_key)
            return True
        except Exception:
            return False

    def _convert_messages(self, messages: List[Dict]) -> List[Dict]:
        result = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                result.append({"role": "system", "content": content})
            elif role == "assistant":
                result.append({"role": "assistant", "content": content})
            else:
                result.append({"role": "user", "content": content})
        return result

    def _convert_tools(self, tools: List[Dict]) -> List[Dict]:
        result = []
        for t in tools:
            result.append({
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameter_definitions": {
                    k: {"description": v.get("description", ""), "type": v.get("type", "str"), "required": k in t.get("required", [])}
                    for k, v in t.get("parameters", {}).get("properties", {}).items()
                }
            })
        return result

    def _normalize(self, raw, model: str) -> ModelResponse:
        content = ""
        tool_calls = []
        if hasattr(raw, "message"):
            msg = raw.message
            for block in getattr(msg, "content", []) or []:
                if hasattr(block, "text"):
                    content += block.text
                elif hasattr(block, "tool_use"):
                    tu = block.tool_use
                    tool_calls.append(ToolCallRecord(
                        id=getattr(tu, "id", ""), tool_name=tu.name,
                        arguments=tu.input or {}, step=0
                    ))
        usage_obj = getattr(raw, "usage", None) or getattr(raw, "meta", None)
        usage = TokenUsage(
            prompt_tokens=getattr(usage_obj, "input_tokens", 0) if usage_obj else 0,
            completion_tokens=getattr(usage_obj, "output_tokens", 0) if usage_obj else 0,
        )
        return ModelResponse(content=content, tool_calls=tool_calls, usage=usage,
                             model=model, provider="cohere", finish_reason="stop")
