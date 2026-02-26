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
import contextlib
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
from helix.core.agent_decorator import agent
from helix.core.graph import END, START, CompiledGraph, StateGraph
from helix.core.group_chat import (
    ChatMessage,
    ConversableAgent,
    GroupChat,
    GroupChatResult,
    HumanAgent,
)
from helix.core.hooks import HookEvent
from helix.core.pipeline import AgentPipeline
from helix.core.session import Session
from helix.core.task import Pipeline, PipelineResult, Task, TaskOutput
from helix.core.team import Team
from helix.core.tool import ToolRegistry, discover_tools, registry, tool
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

# ── Register built-in tools in sys.modules so 'from helix.tools.builtin import …' works ──
with contextlib.suppress(Exception):
    import helix.tools.builtin as _tools_builtin  # noqa: F401 — side-effect import

# ── Preset agents: helix.presets.web_researcher(), helix.presets.coder()…
with contextlib.suppress(Exception):
    from helix import presets  # noqa: F401


def quick(
    system_prompt: str,
    *,
    name: str = "Agent",
    model: str | None = None,
    tools: list[Any] | None = None,
    budget_usd: float = 0.10,
    on_event: Any | None = None,
) -> Agent:
    """
    Create a minimal agent from a single system prompt string.

    This is the fastest way to get started — no role, goal, or config
    objects required.  Perfect for experimentation and notebooks.

    Args:
        system_prompt: The agent's purpose, written as plain instructions.
        name:          Agent name (shown in traces).  Default ``"Agent"``.
        model:         LLM model string, e.g. ``"gpt-4o"`` or ``"claude-sonnet-4-6"``.
                       Omit to auto-detect from available API keys.
        tools:         List of ``@helix.tool``-decorated functions.
        budget_usd:    Hard spend cap per run.  Default ``0.10``.
        on_event:      Optional async or sync callback for live events.

    Returns:
        A :class:`helix.Agent` ready to run.

    Example::

        import helix

        agent = helix.quick("You are a concise Python tutor.")
        result = helix.run(agent, "Explain list comprehensions.")
        print(result.output)
    """
    return Agent(
        name=name,
        role=name,
        goal=system_prompt,
        system_prompt=system_prompt,
        tools=tools or [],
        model=ModelConfig(primary=model) if model else ModelConfig(),
        budget=BudgetConfig(budget_usd=budget_usd),
        on_event=on_event,
    )


def chain(*agents_or_pipelines: Any) -> AgentPipeline:
    """
    Build an :class:`AgentPipeline` from two or more agents.

    Equivalent to ``agent_a | agent_b | agent_c`` but more readable
    when composing many agents.

    Example::

        from helix.presets import web_researcher, summariser, writer

        pipeline = helix.chain(web_researcher(), summariser(), writer())
        result = pipeline.run_sync("Quantum computing advances 2026")
    """
    if not agents_or_pipelines:
        raise ValueError("chain() requires at least one agent")
    result: Any = agents_or_pipelines[0]
    for nxt in agents_or_pipelines[1:]:
        result = result | nxt
    return result


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


def from_yaml(
    agents_yaml: str,
    tasks_yaml: str,
    inputs: dict | None = None,
    **kwargs: Any,
) -> Pipeline:
    """
    Load agents + tasks from YAML files and return a ready-to-run Pipeline.

    Example::

        pipeline = helix.from_yaml("agents.yaml", "tasks.yaml", inputs={"topic": "AI"})
        result = pipeline.kickoff()
        print(result.final_output)
    """
    from helix.core.yaml_config import from_yaml as _from_yaml

    return _from_yaml(agents_yaml, tasks_yaml, inputs=inputs, **kwargs)


try:
    from importlib.metadata import PackageNotFoundError as _PNFE
    from importlib.metadata import version as _pkg_version

    __version__: str = _pkg_version("helix-framework")
except _PNFE:  # editable / source install without metadata
    __version__ = "0.3.3"

__all__ = [
    "run",
    "run_async",
    "quick",
    "chain",
    "create_agent",
    "wrap_llm",
    "from_crewai",
    "from_langchain",
    "from_autogen",
    "from_yaml",
    "eval_suite",
    "discover_tools",
    # Decorators
    "agent",
    "tool",
    "step",
    # Core classes
    "Agent",
    "AgentResult",
    "AgentPipeline",
    "ConversableAgent",
    "HumanAgent",
    "GroupChat",
    "GroupChatResult",
    "ChatMessage",
    "Task",
    "TaskOutput",
    "Pipeline",
    "PipelineResult",
    "Workflow",
    "Team",
    "Session",
    "ToolRegistry",
    "registry",
    "ExecutionContext",
    "HookEvent",
    # Config
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
    # Graph
    "StateGraph",
    "CompiledGraph",
    "END",
    "START",
    # Errors
    "HelixError",
    "BudgetExceededError",
    "LoopDetectedError",
    "ToolError",
    "ToolTimeoutError",
    "ToolPermissionError",
    "__version__",
]
