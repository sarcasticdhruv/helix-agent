"""
helix/core/hooks.py

Lightweight event hook system for real-time agent observability.

Unlike full ObservabilityConfig which writes traces to disk,
hooks fire synchronously (or asynchronously) on every meaningful
agent event, giving developers live introspection without any config.

Usage::

    import helix
    from helix.core.hooks import HookEvent

    async def my_hook(event: HookEvent) -> None:
        if event.type == "tool_call":
            print(f"  → {event.data['tool_name']}({event.data['args']})")
        elif event.type == "step_end":
            print(f"  ✓ step {event.data['step']} — ${event.cost_so_far:.4f} spent")
        elif event.type == "llm_call":
            print(f"  [LLM] {event.data['model']}")

    agent = helix.Agent(
        name="Researcher",
        role="Researcher",
        goal="Find info.",
        on_event=my_hook,
    )
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HookEvent:
    """
    A single event emitted by the agent runtime.

    Attributes:
        type:         Event identifier string. One of:
                      ``"step_start"``   — beginning of a reasoning loop iteration
                      ``"step_end"``     — end of a reasoning loop iteration
                      ``"llm_call"``     — about to call the LLM
                      ``"llm_response"`` — LLM replied
                      ``"tool_call"``    — about to execute a tool
                      ``"tool_result"``  — tool returned a result
                      ``"tool_error"``   — tool raised an error
                      ``"cache_hit"``    — semantic cache returned a response
                      ``"done"``         — agent run completed
                      ``"error"``        — agent run failed
        data:         Event-specific payload (varies by type, see below).
        cost_so_far:  Total USD spent in this run up to this event.
        step:         Current reasoning loop step number (0-based).

    ``data`` payloads by event type:
    - ``step_start``   → ``{"step": int}``
    - ``step_end``     → ``{"step": int, "output_preview": str}``
    - ``llm_call``     → ``{"model": str, "messages": int}``
    - ``llm_response`` → ``{"model": str, "tokens": int, "finish_reason": str}``
    - ``tool_call``    → ``{"tool_name": str, "args": dict}``
    - ``tool_result``  → ``{"tool_name": str, "result_preview": str}``
    - ``tool_error``   → ``{"tool_name": str, "error": str}``
    - ``cache_hit``    → ``{"similarity": float, "saved_usd": float}``
    - ``done``         → ``{"output_preview": str, "steps": int, "cost_usd": float}``
    - ``error``        → ``{"error": str}``
    """

    type: str
    data: dict[str, Any] = field(default_factory=dict)
    cost_so_far: float = 0.0
    step: int = 0


# Type alias
HookFn = Callable[[HookEvent], Awaitable[None] | None]


async def fire(hook: HookFn | None, event: HookEvent) -> None:
    """
    Fire a hook function, supporting both sync and async callables.
    Errors in hooks are silently swallowed so they never crash the agent.
    """
    if hook is None:
        return
    try:
        result = hook(event)
        if asyncio.iscoroutine(result):
            await result
    except Exception:
        pass  # Hooks must never affect agent execution
