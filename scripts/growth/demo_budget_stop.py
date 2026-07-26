"""Deterministic, offline demo: Helix stopping a runaway agent at its budget cap.

No API key or network access required — a fake in-process LLMProvider stands
in for a real one. Run with:

    PYTHONPATH=src python3.11 scripts/growth/demo_budget_stop.py
"""
import asyncio

from helix.core.agent import Agent
from helix.core.tool import tool
from helix.config import ModelConfig, BudgetConfig, ModelResponse, TokenUsage, ToolCallRecord
from helix.interfaces import LLMProvider


@tool(description="Keep researching.")
async def keep_going(note: str) -> str:
    return f"noted: {note}"


class FakeProvider(LLMProvider):
    """Always asks for another tool call, at a fixed simulated cost per step."""

    async def complete(self, messages, model, tools=None, temperature=0.7, max_tokens=4096, response_format=None):
        return ModelResponse(
            content="Still working...",
            tool_calls=[ToolCallRecord(tool_name="keep_going", arguments={"note": "x"})],
            usage=TokenUsage(prompt_tokens=1000, completion_tokens=1000),
            model=model,
            provider="fake",
            finish_reason="tool_calls",
        )

    async def stream(self, messages, model, **kw):
        yield "fake"

    def count_tokens(self, messages, model):
        return 100

    def supported_models(self):
        return ["gpt-4o"]

    async def health(self):
        return True


async def main():
    agent = Agent(
        name="Demo",
        role="tester",
        goal="loop until budget stops me",
        model=ModelConfig(primary="gpt-4o", auto_route=False, max_tokens=1000),
        budget=BudgetConfig(budget_usd=0.03),
        tools=[keep_going],
    )
    await agent._ensure_initialized()
    agent._llm_router._providers["openai:gpt-4o"] = FakeProvider()

    print("Agent budget: $0.03 — asking it to run forever...\n")
    result = await agent.run("Keep going forever")
    print(f"steps={result.steps} cost=${result.cost_usd:.4f}")
    print("OUTPUT:", result.output)


asyncio.run(main())
