"""
helix/core/pipeline.py

AgentPipeline — compose agents with the | operator so output of
one agent becomes the task for the next.

Usage::

    from helix.presets import web_researcher, coder
    from helix.core.pipeline import AgentPipeline

    # Functional
    result = (web_researcher() | coder()).run_sync("Build a Python client for Stripe")

    # Explicit
    pipe = AgentPipeline([researcher, analyst, writer])
    result = await pipe.run("Quantum computing trends")
    print(result.output)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from helix.core.agent import Agent, AgentResult


class AgentPipeline:
    """
    A linear chain of agents where the output of each agent becomes
    the input task for the next.

    Created automatically when you use the ``|`` operator on an Agent::

        pipeline = researcher | analyst | writer
        result = helix.run(pipeline, "Summarise AI trends 2026")
    """

    def __init__(self, agents: list[Agent]) -> None:
        self._agents = list(agents)

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def __or__(self, other: Agent | AgentPipeline) -> AgentPipeline:
        if isinstance(other, AgentPipeline):
            return AgentPipeline(self._agents + other._agents)
        return AgentPipeline(self._agents + [other])

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def run(
        self,
        task: str,
        session_id: str | None = None,
    ) -> AgentResult:
        """
        Pass the task through the agent chain.  The output of each agent
        becomes the task string of the next.  Returns the last AgentResult.
        """
        current: str = task
        result: Any = None
        for agent in self._agents:
            result = await agent.run(current, session_id=session_id)
            current = str(result.output)
        return result  # type: ignore[return-value]

    def run_sync(self, task: str, session_id: str | None = None) -> AgentResult:
        """Synchronous wrapper — safe to call from plain scripts."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, self.run(task, session_id=session_id))
                    return future.result()
            return loop.run_until_complete(self.run(task, session_id=session_id))
        except RuntimeError:
            return asyncio.run(self.run(task, session_id=session_id))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        names = " | ".join(a.name for a in self._agents)
        return f"AgentPipeline({names})"

    @property
    def agents(self) -> list[Agent]:
        return list(self._agents)

    def add(self, agent: Agent) -> AgentPipeline:
        """Append another agent to the end of the pipeline.  Returns self."""
        self._agents.append(agent)
        return self
