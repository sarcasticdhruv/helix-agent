"""
helix/core/handoff.py

First-class handoff primitive: an Agent can transfer a conversation to
another Agent mid-run. The handoff is visible to the model as a normal
tool call — matching the OpenAI Agents SDK's handoff pattern — rather
than being an invisible orchestration decision buried inside Team or
GroupChat's string-forwarding.

Usage::

    triage = helix.Agent(name="Triage", role="...", goal="...",
                          handoffs=[billing_agent, tech_support_agent])
    result = await triage.run("My invoice looks wrong")
    # If the model calls transfer_to_billing, `result` is billing_agent's
    # AgentResult, and result.handoff_chain == ["Triage"].
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from helix.core.tool import RegisteredTool

if TYPE_CHECKING:
    from helix.core.agent import Agent

HANDOFF_TOOL_PREFIX = "transfer_to_"


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return slug or "agent"


def handoff_tool_name(target: Agent) -> str:
    """The tool name the model sees for handing off to `target`."""
    return f"{HANDOFF_TOOL_PREFIX}{_slugify(target.name)}"


def make_handoff_tool(target: Agent) -> RegisteredTool:
    """
    Build the tool that represents handing off to `target`.

    The tool itself does no real work — it just returns a marker dict.
    Agent._reasoning_loop recognizes calls to a transfer_to_* tool by
    name and ends the current agent's turn, delegating to `target`.
    """
    name = handoff_tool_name(target)

    async def _transfer(reason: str = "") -> dict[str, Any]:
        return {"handoff": True, "target": target.name, "reason": reason}

    return RegisteredTool(
        fn=_transfer,
        name=name,
        description=(
            f"Transfer this conversation to {target.name} ({target.config.role}) "
            f"when the task is better handled by them than by you. "
            f"{target.config.goal}"
        ),
        timeout_s=5.0,
        retries=0,
        on_error="raise",
        fallback_fn=None,
        parameters_schema={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why this task is being handed off.",
                }
            },
            "required": [],
        },
    )
