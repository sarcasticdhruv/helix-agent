"""
helix/core/task.py

Task — a first-class, declarative unit of work assigned to an Agent.

Inspired by CrewAI's Task model; extended with:
  - output_schema    (Pydantic structured output)
  - guardrails       (validation chain with auto-retry — both callable and string)
  - context          (task dependencies: wait for outputs of other tasks)
  - callback         (post-completion hook)
  - output_file      (persist output to disk)
  - async_execution  (run concurrently without blocking next task)
  - expected_output  (description used in the agent system prompt)
  - template var substitution via format_inputs()
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# TaskOutput
# ---------------------------------------------------------------------------


@dataclass
class TaskOutput:
    """Structured result returned by Task.run()."""

    description: str
    raw: str
    agent_name: str
    duration_s: float = 0.0
    cost_usd: float = 0.0
    pydantic: BaseModel | None = None
    json_dict: dict[str, Any] | None = None
    error: str | None = None

    @property
    def summary(self) -> str:
        """Auto-summary: first 10 words of description."""
        words = self.description.split()
        return " ".join(words[:10]) + ("..." if len(words) > 10 else "")

    def to_dict(self) -> dict[str, Any]:
        if self.json_dict:
            return self.json_dict
        if self.pydantic:
            return self.pydantic.model_dump()
        return {"raw": self.raw}

    def __str__(self) -> str:
        if self.pydantic:
            return str(self.pydantic)
        if self.json_dict:
            import json

            return json.dumps(self.json_dict, indent=2)
        return self.raw


# ---------------------------------------------------------------------------
# Guardrail helpers
# ---------------------------------------------------------------------------


async def _run_guardrail(
    guardrail: Callable | str,
    output: TaskOutput,
    agent: Any,
) -> tuple[bool, Any]:
    """
    Run one guardrail. Supports:
      - callable(TaskOutput) → (bool, Any)
      - string description   → LLM validates against the description
    """
    if callable(guardrail):
        if asyncio.iscoroutinefunction(guardrail):
            return await guardrail(output)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, guardrail, output)

    # String guardrail — use the agent's LLM to validate
    if agent is None:
        return True, output.raw

    validation_prompt = (
        f"Validate the following output against this criterion:\n\n"
        f"CRITERION: {guardrail}\n\n"
        f"OUTPUT:\n{output.raw}\n\n"
        f"Reply with EXACTLY one of:\n"
        f"PASS: <optional brief note>\n"
        f"FAIL: <specific reason why it fails>"
    )
    from helix.core.agent import AgentResult

    result: AgentResult = await agent.run(validation_prompt)
    raw_verdict = str(result.output).strip()
    if raw_verdict.upper().startswith("PASS"):
        return True, output.raw
    reason = re.sub(r"^FAIL:\s*", "", raw_verdict, flags=re.IGNORECASE).strip()
    return False, reason or "Guardrail failed"


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class Task:
    """
    A declarative unit of work executed by an Agent.

    Usage::

        from helix import Task, Agent

        researcher = Agent(name="Researcher", role="Researcher", goal="Find facts.")
        writer     = Agent(name="Writer",     role="Writer",     goal="Write prose.")

        research = Task(
            description="Research the history of {topic}.",
            expected_output="A list of 5 key facts about {topic}.",
            agent=researcher,
        )
        article = Task(
            description="Write a 3-paragraph article based on the research.",
            expected_output="A well-structured article.",
            agent=writer,
            context=[research],          # waits for research to finish
            output_file="article.md",
        )

        # Run tasks in sequence
        from helix import Pipeline
        pipeline = Pipeline(tasks=[research, article])
        result = await pipeline.run(inputs={"topic": "quantum computing"})
    """

    def __init__(
        self,
        description: str,
        expected_output: str = "",
        agent: Any | None = None,
        tools: list[Any] | None = None,
        context: list[Task] | None = None,
        async_execution: bool = False,
        guardrail: Callable | str | None = None,
        guardrails: list[Callable | str] | None = None,
        guardrail_max_retries: int = 3,
        output_schema: type[BaseModel] | None = None,
        output_file: str | None = None,
        callback: Callable[[TaskOutput], Any] | None = None,
        name: str | None = None,
        markdown: bool = False,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.description = description
        self.expected_output = expected_output
        self.agent = agent
        self.tools = tools or []
        self.context: list[Task] = context or []
        self.async_execution = async_execution
        self.guardrail_max_retries = guardrail_max_retries
        self.output_schema = output_schema
        self.output_file = output_file
        self.callback = callback
        self.name = name or description[:40].strip()
        self.markdown = markdown
        self.config = config or {}

        # Normalise guardrails into a single list
        if guardrails:
            self._guardrails: list[Callable | str] = list(guardrails)
        elif guardrail:
            self._guardrails = [guardrail]
        else:
            self._guardrails = []

        # Set after execution
        self.output: TaskOutput | None = None

    # ------------------------------------------------------------------
    # Template variable substitution
    # ------------------------------------------------------------------

    def format_inputs(self, inputs: dict[str, Any]) -> Task:
        """Return a copy of this Task with {variables} substituted."""

        def _sub(s: str) -> str:
            for k, v in inputs.items():
                s = s.replace(f"{{{k}}}", str(v))
            return s

        clone = Task(
            description=_sub(self.description),
            expected_output=_sub(self.expected_output),
            agent=self.agent,
            tools=self.tools,
            context=self.context,
            async_execution=self.async_execution,
            guardrails=self._guardrails or None,
            guardrail_max_retries=self.guardrail_max_retries,
            output_schema=self.output_schema,
            output_file=self.output_file,
            callback=self.callback,
            name=self.name,
            markdown=self.markdown,
            config=self.config,
        )
        return clone

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def run(
        self,
        context_text: str = "",
        inputs: dict[str, Any] | None = None,
    ) -> TaskOutput:
        """Execute this task. Called by Pipeline; rarely called directly."""
        if self.agent is None:
            raise ValueError(f"Task '{self.name}' has no agent assigned.")

        # Apply template substitution
        task = self.format_inputs(inputs or {})

        # Build prompt
        parts = [task.description]
        if context_text:
            parts.append(f"\n\nContext from previous tasks:\n{context_text}")
        if task.expected_output:
            parts.append(f"\n\nExpected output format: {task.expected_output}")
        if task.markdown:
            parts.append(
                "\n\nFormat your final answer using proper Markdown "
                "(headers with #, bold with **, lists with -)."
            )

        prompt = "\n".join(parts)

        # Override agent tools if task specifies them
        agent = task.agent
        if task.tools:
            from helix.core.agent import Agent as HelixAgent

            if isinstance(agent, HelixAgent):
                # Temp-inject tools for this task run
                for t in task.tools:
                    agent._registry.register(t)

        # Guardrail retry loop
        last_output: TaskOutput | None = None
        for attempt in range(self.guardrail_max_retries + 1):
            start = time.time()
            agent_result = await agent.run(
                prompt if attempt == 0 else _retry_prompt(prompt, last_output),
                output_schema=task.output_schema,
            )
            duration = time.time() - start

            # Build TaskOutput
            raw = str(agent_result.output)
            pydantic_obj = (
                agent_result.output if isinstance(agent_result.output, BaseModel) else None
            )
            json_d: dict[str, Any] | None = None
            if pydantic_obj:
                json_d = pydantic_obj.model_dump()

            task_output = TaskOutput(
                description=task.description,
                raw=raw,
                agent_name=agent.name if hasattr(agent, "name") else str(agent),
                duration_s=duration,
                cost_usd=agent_result.cost_usd,
                pydantic=pydantic_obj,
                json_dict=json_d,
            )

            # Run guardrail chain
            if not self._guardrails:
                break

            all_passed = True
            current_raw = raw
            for g in self._guardrails:
                # Pass a TaskOutput with possibly-updated raw
                current_output = TaskOutput(
                    description=task.description,
                    raw=current_raw,
                    agent_name=task_output.agent_name,
                    duration_s=task_output.duration_s,
                    cost_usd=task_output.cost_usd,
                    pydantic=pydantic_obj,
                    json_dict=json_d,
                )
                passed, result = await _run_guardrail(g, current_output, agent)
                if passed:
                    current_raw = str(result)
                else:
                    all_passed = False
                    task_output = TaskOutput(
                        description=task.description,
                        raw=current_raw,
                        agent_name=task_output.agent_name,
                        duration_s=task_output.duration_s,
                        cost_usd=task_output.cost_usd,
                        error=str(result),
                    )
                    last_output = task_output
                    prompt = prompt  # keep original, retry with feedback
                    break
                task_output = TaskOutput(
                    description=task.description,
                    raw=current_raw,
                    agent_name=task_output.agent_name,
                    duration_s=task_output.duration_s,
                    cost_usd=task_output.cost_usd,
                    pydantic=pydantic_obj,
                    json_dict=json_d,
                )

            if all_passed:
                break
        else:
            # Max retries reached — return last output with error flag
            pass

        # Persist to file
        if self.output_file and task_output:
            _write_output_file(self.output_file, task_output.raw)

        # Fire callback
        if self.callback and task_output:
            if asyncio.iscoroutinefunction(self.callback):
                await self.callback(task_output)
            else:
                self.callback(task_output)

        self.output = task_output
        return task_output


def _retry_prompt(original: str, failed_output: TaskOutput | None) -> str:
    if failed_output and failed_output.error:
        return (
            f"{original}\n\n"
            f"Your previous answer did not meet requirements. "
            f"Reason: {failed_output.error}\n"
            f"Please revise your answer."
        )
    return original


def _write_output_file(path: str, content: str) -> None:
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Pipeline — ordered list of Tasks (CrewAI-style kickoff)
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Aggregated result from running a Pipeline."""

    task_outputs: list[TaskOutput] = field(default_factory=list)
    final_output: str = ""
    total_cost_usd: float = 0.0
    duration_s: float = 0.0
    error: str | None = None

    def __str__(self) -> str:
        return self.final_output


class Pipeline:
    """
    Runs a list of Tasks in sequence (or concurrently for async tasks),
    automatically passing context between steps.

    This is the Helix equivalent of CrewAI's ``crew.kickoff()``.

    Usage::

        pipeline = Pipeline(tasks=[task1, task2, task3])
        result = await pipeline.run(inputs={"topic": "AI"})
        print(result.final_output)

        # Sync convenience
        result = pipeline.kickoff(inputs={"topic": "AI"})
    """

    def __init__(self, tasks: list[Task]) -> None:
        self._tasks = tasks

    async def run(self, inputs: dict[str, Any] | None = None) -> PipelineResult:
        """Execute all tasks, passing outputs forward as context."""
        _inputs = inputs or {}
        start = time.time()
        outputs: list[TaskOutput] = []
        context_parts: list[str] = []

        # Identify which tasks can run async
        pending_async: list[asyncio.Task] = []
        async_task_map: dict[str, asyncio.Task] = {}

        for task in self._tasks:
            # Wait for context-dependency tasks first
            for dep in task.context:
                if dep.output is None:
                    # If dependency is still pending, wait
                    dep_entry = async_task_map.get(id(dep))  # type: ignore[arg-type]
                    if dep_entry:
                        await dep_entry

            # Build context string from dependency outputs + sequential outputs
            ctx_parts = list(context_parts)
            for dep in task.context:
                if dep.output:
                    ctx_parts.append(f"[{dep.name}]: {dep.output.raw}")
            context_text = "\n\n".join(ctx_parts)

            if task.async_execution:
                # Schedule concurrently, will be awaited by dependant tasks
                coro = task.run(context_text=context_text, inputs=_inputs)
                atask = asyncio.create_task(coro)
                async_task_map[id(task)] = atask  # type: ignore[assignment]
                pending_async.append(atask)
            else:
                # Sequential: wait for all pending async first if they are context deps
                try:
                    task_output = await task.run(context_text=context_text, inputs=_inputs)
                    outputs.append(task_output)
                    context_parts.append(f"[{task.name}]: {task_output.raw}")
                except Exception as exc:
                    err_output = TaskOutput(
                        description=task.description,
                        raw="",
                        agent_name=str(task.agent),
                        error=str(exc),
                    )
                    outputs.append(err_output)

        # Collect any remaining async tasks
        if pending_async:
            async_results = await asyncio.gather(*pending_async, return_exceptions=True)
            for r in async_results:
                if isinstance(r, TaskOutput):
                    outputs.append(r)

        total_cost = sum(o.cost_usd for o in outputs)
        final = outputs[-1].raw if outputs else ""

        return PipelineResult(
            task_outputs=outputs,
            final_output=final,
            total_cost_usd=total_cost,
            duration_s=time.time() - start,
        )

    def kickoff(self, inputs: dict[str, Any] | None = None) -> PipelineResult:
        """Synchronous entry point — equivalent to CrewAI's crew.kickoff()."""
        return asyncio.run(self.run(inputs=inputs))
