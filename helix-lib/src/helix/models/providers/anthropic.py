"""
helix/models/providers/anthropic.py

Anthropic LLMProvider implementation.

Wraps the anthropic SDK and normalizes all responses into ModelResponse.
Applies prefix cache breakpoints automatically for long system prompts.
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, List, Optional

from helix.config import ModelResponse, TokenUsage, ToolCallRecord
from helix.errors import HelixProviderError
from helix.interfaces import LLMProvider

# Minimum tokens in system prompt to apply prefix caching
_PREFIX_CACHE_MIN_TOKENS = 1024


class AnthropicProvider(LLMProvider):
    """
    Wraps anthropic.AsyncAnthropic.
    Requires ANTHROPIC_API_KEY environment variable.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic(
                api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            )
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> ModelResponse:
        try:
            system_text, user_messages = self._split_messages(messages)
            prepared_messages = self._prepare_messages(user_messages)

            kwargs: Dict[str, Any] = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": prepared_messages,
            }
            if system_text:
                kwargs["system"] = self._maybe_cache_system(system_text)
            if tools:
                kwargs["tools"] = [self._to_anthropic_tool(t) for t in tools]

            raw = await self._client.messages.create(**kwargs)
            return self._normalize(raw, model)

        except Exception as e:
            err_str = str(e).lower()
            retryable = "rate" in err_str or "timeout" in err_str or "overloaded" in err_str
            raise HelixProviderError(
                provider="anthropic",
                model=model,
                reason=str(e),
                retryable=retryable,
            ) from e

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        system_text, user_messages = self._split_messages(messages)
        prepared = self._prepare_messages(user_messages)

        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": prepared,
        }
        if system_text:
            kwargs["system"] = system_text

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            raise HelixProviderError(
                provider="anthropic", model=model, reason=str(e), retryable=False
            ) from e

    def count_tokens(self, messages: List[Dict[str, Any]], model: str) -> int:
        """Approximate: chars / 3.5 (Anthropic uses ~3.5 chars per token)."""
        total = 0
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                total += int(len(content) / 3.5)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        total += int(len(str(block.get("text", ""))) / 3.5)
        return total

    def supported_models(self) -> List[str]:
        return ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]

    async def health(self) -> bool:
        try:
            # Minimal API call to check connectivity
            await self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Anthropic-specific helpers
    # ------------------------------------------------------------------

    def _split_messages(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Split Helix message list into system string + user/assistant messages."""
        system_parts = []
        others = []
        for m in messages:
            if m.get("role") == "system":
                system_parts.append(m.get("content", ""))
            else:
                others.append(m)
        return "\n".join(system_parts), others

    def _prepare_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert Helix tool messages to Anthropic's tool_result format.
        Anthropic requires tool results inside user messages.
        """
        prepared = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")

            if role == "tool":
                # Wrap as user message with tool_result block
                prepared.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_call_id", "unknown"),
                        "content": content,
                    }]
                })
            else:
                prepared.append({"role": role, "content": content})

        return prepared

    def _maybe_cache_system(self, system_text: str) -> Any:
        """
        Add cache_control breakpoint if system prompt is large enough
        to benefit from Anthropic prefix caching.
        """
        token_estimate = len(system_text) // 4
        if token_estimate >= _PREFIX_CACHE_MIN_TOKENS:
            return [
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        return system_text

    def _to_anthropic_tool(self, tool_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Helix tool schema to Anthropic format."""
        return {
            "name": tool_schema["name"],
            "description": tool_schema["description"],
            "input_schema": tool_schema.get("parameters", {"type": "object", "properties": {}}),
        }

    def _normalize(self, raw: Any, model: str) -> ModelResponse:
        """Convert raw Anthropic response to ModelResponse."""
        import json as _json

        content_text = ""
        tool_calls: List[ToolCallRecord] = []

        for block in raw.content:
            if block.type == "text":
                content_text += block.text
            elif block.type == "tool_use":
                args = block.input if isinstance(block.input, dict) else {}
                tool_calls.append(ToolCallRecord(
                    tool_name=block.name,
                    arguments=args,
                ))

        usage = TokenUsage(
            prompt_tokens=raw.usage.input_tokens,
            completion_tokens=raw.usage.output_tokens,
            cached_tokens=getattr(raw.usage, "cache_read_input_tokens", 0) or 0,
        )

        finish_reason = "stop"
        if raw.stop_reason == "tool_use":
            finish_reason = "tool_calls"
        elif raw.stop_reason == "max_tokens":
            finish_reason = "length"

        return ModelResponse(
            content=content_text,
            tool_calls=tool_calls,
            usage=usage,
            model=model,
            provider="anthropic",
            finish_reason=finish_reason,
        )
