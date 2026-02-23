"""
helix/cache/prefix.py

PrefixCacheHook — marks long system prompts for provider-side caching.

Anthropic: cache_control breakpoints deliver 90% cost reduction on cached tokens.
OpenAI: prefix caching is automatic, no client-side changes needed.
"""

from __future__ import annotations

from typing import Any

_ANTHROPIC_MIN_TOKENS = 1024  # Minimum tokens to bother with cache_control


class PrefixCacheHook:
    """
    Prepares message lists for provider-native prefix caching.

    Called by the ModelRouter before each LLM call.
    Modifies messages in-place to add cache_control markers where beneficial.
    """

    def prepare(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> list[dict[str, Any]]:
        """
        Return a (possibly modified) message list with cache markers applied.

        For Anthropic models: adds cache_control to long system messages.
        For OpenAI models: returns unchanged (caching is automatic).
        """
        if "claude" in model.lower():
            return self._mark_anthropic(messages)
        return messages

    def _mark_anthropic(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Add Anthropic cache_control breakpoints to system messages
        that are large enough to benefit from prefix caching.
        """
        result = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    token_estimate = len(content) // 4
                    if token_estimate >= _ANTHROPIC_MIN_TOKENS:
                        msg = {
                            **msg,
                            "content": [
                                {
                                    "type": "text",
                                    "text": content,
                                    "cache_control": {"type": "ephemeral"},
                                }
                            ],
                        }
            result.append(msg)
        return result
