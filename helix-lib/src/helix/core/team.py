"""
helix/core/team.py

Team — coordinates multiple agents working on a shared goal.

Strategies:
  sequential   — agents run one after another, output passed forward
  parallel     — all agents run on the same input concurrently
  hierarchical — lead agent delegates subtasks to specialist agents
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from helix.config import TeamConfig
from helix.core.agent import Agent, AgentResult


@dataclass
class TeamResult:
    team_name: str
    final_output: Any
    agent_results: List[AgentResult] = field(default_factory=list)
    total_cost_usd: float = 0.0
    duration_s: float = 0.0
    error: Optional[str] = None


class Team:
    """
    A coordinated group of agents.

    Usage::

        team = Team(
            name="research-team",
            agents=[searcher, analyst, writer],
            strategy="sequential",
            budget_usd=5.00,
        )
        result = await team.run("Write a report on quantum computing trends")
    """

    def __init__(
        self,
        name: str,
        agents: List[Agent],
        strategy: str = "sequential",
        lead: Optional[Agent] = None,
        shared_memory: bool = True,
        budget_usd: Optional[float] = None,
        config: Optional[TeamConfig] = None,
    ) -> None:
        self._name = name
        self._agents = agents
        self._strategy = strategy
        self._lead = lead
        self._shared_memory = shared_memory
        self._budget_usd = budget_usd
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, task: str) -> TeamResult:
        start = time.time()
        try:
            if self._strategy == "sequential":
                return await self._run_sequential(task, start)
            if self._strategy == "parallel":
                return await self._run_parallel(task, start)
            if self._strategy == "hierarchical":
                return await self._run_hierarchical(task, start)
            raise ValueError(f"Unknown team strategy: {self._strategy}")
        except Exception as e:
            return TeamResult(
                team_name=self._name,
                final_output=None,
                duration_s=time.time() - start,
                error=str(e),
            )

    def run_sync(self, task: str) -> TeamResult:
        return asyncio.run(self.run(task))

    def add_agent(self, agent: Agent) -> "Team":
        self._agents.append(agent)
        return self

    def remove_agent(self, name: str) -> "Team":
        self._agents = [a for a in self._agents if a.name != name]
        return self

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    async def _run_sequential(self, task: str, start: float) -> TeamResult:
        """Each agent receives the previous agent's output as its input."""
        results: List[AgentResult] = []
        current_input = task

        for agent in self._agents:
            result = await agent.run(current_input)
            results.append(result)
            current_input = str(result.output)

        total_cost = sum(r.cost_usd for r in results)
        return TeamResult(
            team_name=self._name,
            final_output=current_input,
            agent_results=results,
            total_cost_usd=total_cost,
            duration_s=time.time() - start,
        )

    async def _run_parallel(self, task: str, start: float) -> TeamResult:
        """All agents run on the same input concurrently. Outputs returned as list."""
        coroutines = [agent.run(task) for agent in self._agents]
        results: List[AgentResult] = await asyncio.gather(*coroutines, return_exceptions=False)

        total_cost = sum(r.cost_usd for r in results)
        outputs = [str(r.output) for r in results]
        return TeamResult(
            team_name=self._name,
            final_output=outputs,
            agent_results=results,
            total_cost_usd=total_cost,
            duration_s=time.time() - start,
        )

    async def _run_hierarchical(self, task: str, start: float) -> TeamResult:
        """
        Lead agent decomposes the task and delegates subtasks to specialists.
        Falls back to sequential if no lead is set.
        """
        if self._lead is None:
            return await self._run_sequential(task, start)

        # Lead decomposes task
        specialist_names = ", ".join(f"{a.name} ({a.config.role})" for a in self._agents)
        decompose_prompt = (
            f"Break this task into subtasks, one per specialist. "
            f"Available specialists: {specialist_names}\n\n"
            f"Task: {task}\n\n"
            f"Output a JSON array: [{{'specialist': name, 'subtask': description}}, ...]"
        )
        lead_result = await self._lead.run(decompose_prompt)

        # Parse subtasks
        import json, re
        subtasks = []
        try:
            raw = str(lead_result.output)
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                subtasks = json.loads(match.group())
        except Exception:
            return await self._run_sequential(task, start)

        # Dispatch subtasks to specialists
        agent_map = {a.name: a for a in self._agents}
        results: List[AgentResult] = [lead_result]
        outputs: List[str] = []

        dispatch_coros = []
        for st in subtasks:
            specialist_name = st.get("specialist", "")
            subtask = st.get("subtask", task)
            agent = agent_map.get(specialist_name)
            if agent:
                dispatch_coros.append(agent.run(subtask))

        if dispatch_coros:
            specialist_results = await asyncio.gather(*dispatch_coros, return_exceptions=True)
            for r in specialist_results:
                if isinstance(r, AgentResult):
                    results.append(r)
                    outputs.append(str(r.output))

        total_cost = sum(r.cost_usd for r in results)
        final_output = "\n\n".join(outputs) if outputs else str(lead_result.output)

        return TeamResult(
            team_name=self._name,
            final_output=final_output,
            agent_results=results,
            total_cost_usd=total_cost,
            duration_s=time.time() - start,
        )
