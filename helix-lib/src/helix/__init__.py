"""
helix — Production-grade AI agent framework.

Quickstart
----------

    import helix

    @helix.tool(description="Search the web for information.")
    async def search(query: str) -> str:
        return f"Results for: {query}"

    agent = helix.Agent(
        name="Researcher",
        role="Research analyst",
        goal="Find accurate, cited answers.",
        tools=[search],
    )

    # Synchronous — works anywhere, including plain scripts
    result = helix.run(agent, "What is quantum entanglement?")
    print(result.output)

    # Asynchronous — use inside async functions
    async def main():
        result = await agent.run("What is quantum entanglement?")
        print(result.output)

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
from typing import Any

from helix.config import (
    AgentConfig,
    AgentMode,
    BudgetConfig,
    BudgetStrategy,
    CacheConfig,
    EpisodeOutcome,
    EvalCase,
    FailureClass,
    HITLConfig,
    MemoryConfig,
    ModelConfig,
    ObservabilityConfig,
    PermissionConfig,
    RuntimeConfig,
    SessionConfig,
    StructuredOutputConfig,
    TeamConfig,
    WorkflowConfig,
    WorkflowMode,
)

# ── Auto-load keys from ~/.helix/config.json and .env on import ─────────────
from helix.config_store import apply_saved_config as _apply_saved_config
from helix.context import ExecutionContext
from helix.core.agent import Agent, AgentResult
from helix.core.session import Session
from helix.core.team import Team
from helix.core.tool import ToolRegistry, registry, tool
from helix.core.workflow import Workflow, step
from helix.errors import (
    BudgetExceededError,
    HelixError,
    LoopDetectedError,
    ToolError,
    ToolPermissionError,
    ToolTimeoutError,
)

_apply_saved_config()


def run(
    agent: Agent,
    task: str,
    session_id: str | None = None,
    output_schema: Any | None = None,
) -> AgentResult:
    """
    Run an agent synchronously. Works in plain scripts, notebooks, and
    anywhere there is no running event loop.

    This is the recommended entry point for non-async code. It manages
    the event loop so you never need to write ``asyncio.run(...)`` yourself.

    Args:
        agent:         A configured :class:`helix.Agent` instance.
        task:          The task string to execute.
        session_id:    Optional session ID for multi-turn memory.
        output_schema: Pydantic model or JSON Schema for structured output.

    Returns:
        :class:`helix.AgentResult` with output, cost_usd, steps, cache_hits,
        run_id, and optional trace.

    Example::

        import helix

        agent = helix.Agent(name="Bot", role="Helper", goal="Answer questions.")
        result = helix.run(agent, "What is 2 + 2?")
        print(result.output)
        print(f"Cost: ${result.cost_usd:.6f}")
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    agent.run(task, session_id=session_id, output_schema=output_schema),
                )
                return future.result()
        coro = agent.run(task, session_id=session_id, output_schema=output_schema)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(agent.run(task, session_id=session_id, output_schema=output_schema))


async def run_async(
    agent: Agent,
    task: str,
    session_id: str | None = None,
    output_schema: Any | None = None,
) -> AgentResult:
    """
    Async version of :func:`helix.run`. Use this inside ``async def`` functions.

    Example::

        import asyncio, helix

        async def main():
            agent = helix.Agent(name="Bot", role="Helper", goal="Answer questions.")
            result = await helix.run_async(agent, "Explain transformers.")
            print(result.output)

        asyncio.run(main())
    """
    return await agent.run(task, session_id=session_id, output_schema=output_schema)


def create_agent(name: str, role: str, goal: str, **kwargs: Any) -> Agent:
    """Shorthand factory for creating an Agent."""
    return Agent(name=name, role=role, goal=goal, **kwargs)


def wrap_llm(llm_client: Any, budget_usd: float = 10.0, **kwargs: Any) -> Any:
    """Wrap any LangChain/CrewAI/AutoGen LLM client with Helix cost governance."""
    from helix.adapters.universal import wrap_llm as _wrap_llm

    return _wrap_llm(llm_client, budget_usd=budget_usd, **kwargs)


def from_crewai(crew: Any, budget_usd: float, **kwargs: Any) -> Any:
    """Run a CrewAI crew under Helix governance."""
    from helix.adapters.universal import from_crewai as _from_crewai

    return _from_crewai(crew, budget_usd=budget_usd, **kwargs)


def from_langchain(chain: Any, budget_usd: float, **kwargs: Any) -> Any:
    """Run a LangChain chain under Helix governance."""
    from helix.adapters.universal import from_langchain as _from_langchain

    return _from_langchain(chain, budget_usd=budget_usd, **kwargs)


def from_autogen(agent_obj: Any, budget_usd: float, **kwargs: Any) -> Any:
    """Run an AutoGen agent under Helix governance."""
    from helix.adapters.universal import from_autogen as _from_autogen

    return _from_autogen(agent_obj, budget_usd=budget_usd, **kwargs)


def eval_suite(name: str, **kwargs: Any) -> Any:
    """Create an EvalSuite."""
    from helix.eval.suite import EvalSuite

    return EvalSuite(name=name, **kwargs)


try:
    from importlib.metadata import PackageNotFoundError as _PNFE
    from importlib.metadata import version as _pkg_version

    __version__: str = _pkg_version("helix-agent")
except _PNFE:  # editable / source install without metadata
    __version__ = "0.3.0"

__all__ = [
    "run",
    "run_async",
    "create_agent",
    "wrap_llm",
    "from_crewai",
    "from_langchain",
    "from_autogen",
    "eval_suite",
    "Agent",
    "AgentResult",
    "Workflow",
    "Team",
    "Session",
    "step",
    "tool",
    "ToolRegistry",
    "registry",
    "ExecutionContext",
    "AgentConfig",
    "AgentMode",
    "BudgetConfig",
    "BudgetStrategy",
    "CacheConfig",
    "EvalCase",
    "HITLConfig",
    "MemoryConfig",
    "ModelConfig",
    "ObservabilityConfig",
    "PermissionConfig",
    "RuntimeConfig",
    "SessionConfig",
    "StructuredOutputConfig",
    "TeamConfig",
    "WorkflowConfig",
    "WorkflowMode",
    "EpisodeOutcome",
    "FailureClass",
    "HelixError",
    "BudgetExceededError",
    "LoopDetectedError",
    "ToolError",
    "ToolTimeoutError",
    "ToolPermissionError",
    "__version__",
]
