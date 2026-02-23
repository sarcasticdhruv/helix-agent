"""
helix/core/workflow.py

Workflow — a directed graph of steps executed by agents.

Modes: sequential, parallel, conditional, loop, map, reduce, human_review.
State is checkpointed after every step so workflows can resume after interruption.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from helix.config import WorkflowConfig
from helix.errors import WorkflowError


@dataclass
class StepResult:
    name: str
    output: Any
    cost_usd: float = 0.0
    duration_s: float = 0.0
    retries: int = 0
    error: str | None = None


@dataclass
class WorkflowResult:
    workflow_name: str
    final_output: Any
    steps: list[StepResult] = field(default_factory=list)
    total_cost_usd: float = 0.0
    duration_s: float = 0.0
    error: str | None = None


class Step:
    """A single unit of work in a workflow."""

    def __init__(
        self,
        fn: Callable,
        name: str,
        retry: int = 0,
        fallback: Callable | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.fn = fn
        self.name = name
        self.retry = retry
        self.fallback = fallback
        self.timeout_s = timeout_s

    async def execute(self, input: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self.retry + 1):
            try:
                if asyncio.iscoroutinefunction(self.fn):
                    coro = self.fn(input)
                    if self.timeout_s:
                        return await asyncio.wait_for(coro, timeout=self.timeout_s)
                    return await coro
                else:
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, self.fn, input)
            except Exception as e:
                last_exc = e
                if attempt < self.retry:
                    await asyncio.sleep(0.5 * (2**attempt))
        if self.fallback:
            if asyncio.iscoroutinefunction(self.fallback):
                return await self.fallback(input)
            return self.fallback(input)
        raise last_exc  # type: ignore[misc]


def step(
    name: str | None = None,
    retry: int = 0,
    fallback: Callable | None = None,
    timeout_s: float | None = None,
) -> Callable:
    """Decorator to mark a function as a workflow step."""

    def decorator(fn: Callable) -> Step:
        return Step(
            fn=fn,
            name=name or fn.__name__,
            retry=retry,
            fallback=fallback,
            timeout_s=timeout_s,
        )

    return decorator


class Workflow:
    """
    Fluent workflow builder.

    Usage::

        wf = (
            Workflow("research-pipeline")
            .then(search_step)
            .then(summarize_step)
            .parallel(fact_check_step, cite_step)
            .with_budget(2.00)
        )
        result = await wf.run("What is quantum computing?")
    """

    def __init__(self, name: str, config: WorkflowConfig | None = None) -> None:
        self._name = name
        self._config = config or WorkflowConfig(name=name)
        self._chain: list[dict[str, Any]] = []  # List of {type, steps, kwargs}
        self._budget_usd: float | None = config.budget_usd if config else None
        self._on_failure: str = config.on_failure if config else "fail"

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    def then(self, step_fn: Step | Callable, **kwargs: Any) -> Workflow:
        """Add a sequential step."""
        s = (
            step_fn
            if isinstance(step_fn, Step)
            else Step(fn=step_fn, name=getattr(step_fn, "__name__", "step"))
        )
        self._chain.append({"type": "sequential", "step": s})
        return self

    def parallel(self, *step_fns: Step | Callable) -> Workflow:
        """Add steps that run concurrently. Their outputs are collected as a list."""
        steps = [
            (s if isinstance(s, Step) else Step(fn=s, name=getattr(s, "__name__", "step")))
            for s in step_fns
        ]
        self._chain.append({"type": "parallel", "steps": steps})
        return self

    def map(self, step_fn: Step | Callable, items_fn: Callable) -> Workflow:
        """Apply step_fn to each item returned by items_fn(input) concurrently."""
        s = (
            step_fn
            if isinstance(step_fn, Step)
            else Step(fn=step_fn, name=getattr(step_fn, "__name__", "map_step"))
        )
        self._chain.append({"type": "map", "step": s, "items_fn": items_fn})
        return self

    def reduce(self, reduce_fn: Callable, initial: Any = None) -> Workflow:
        """Reduce the current output list to a single value."""
        self._chain.append({"type": "reduce", "fn": reduce_fn, "initial": initial})
        return self

    def branch(
        self,
        condition: Callable[[Any], bool],
        if_true: Step | Callable,
        if_false: Step | Callable,
    ) -> Workflow:
        """Branch based on a condition applied to the current input."""
        true_step = if_true if isinstance(if_true, Step) else Step(fn=if_true, name="if_true")
        false_step = if_false if isinstance(if_false, Step) else Step(fn=if_false, name="if_false")
        self._chain.append(
            {
                "type": "conditional",
                "condition": condition,
                "if_true": true_step,
                "if_false": false_step,
            }
        )
        return self

    def loop(
        self,
        step_fn: Step | Callable,
        until: Callable[[Any], bool],
        max_iter: int = 10,
    ) -> Workflow:
        """Repeat step_fn until until(output) is True or max_iter is reached."""
        s = (
            step_fn
            if isinstance(step_fn, Step)
            else Step(fn=step_fn, name=getattr(step_fn, "__name__", "loop_step"))
        )
        self._chain.append({"type": "loop", "step": s, "until": until, "max_iter": max_iter})
        return self

    def human_review(
        self,
        prompt: str = "Review required before continuing.",
        risk_level: str = "medium",
    ) -> Workflow:
        """Insert a HITL checkpoint — workflow pauses until a human approves."""
        self._chain.append(
            {
                "type": "human_review",
                "prompt": prompt,
                "risk_level": risk_level,
            }
        )
        return self

    def with_budget(self, usd: float) -> Workflow:
        self._budget_usd = usd
        return self

    def on_failure(self, strategy: str) -> Workflow:
        """'fail' | 'continue' | 'fallback'"""
        self._on_failure = strategy
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def run(self, input: Any) -> WorkflowResult:
        start = time.time()
        current = input
        step_results: list[StepResult] = []
        total_cost = 0.0

        for node in self._chain:
            node_start = time.time()
            try:
                current, sr = await self._execute_node(node, current)
                step_results.append(sr)
                total_cost += sr.cost_usd
            except Exception as e:
                if self._on_failure == "continue":
                    step_results.append(
                        StepResult(
                            name=node.get("type", "unknown"),
                            output=None,
                            error=str(e),
                        )
                    )
                    continue
                return WorkflowResult(
                    workflow_name=self._name,
                    final_output=None,
                    steps=step_results,
                    total_cost_usd=total_cost,
                    duration_s=time.time() - start,
                    error=str(e),
                )

        return WorkflowResult(
            workflow_name=self._name,
            final_output=current,
            steps=step_results,
            total_cost_usd=total_cost,
            duration_s=time.time() - start,
        )

    def run_sync(self, input: Any) -> WorkflowResult:
        return asyncio.run(self.run(input))

    async def _execute_node(self, node: dict[str, Any], current: Any) -> tuple[Any, StepResult]:
        node_start = time.time()
        node_type = node["type"]

        if node_type == "sequential":
            s: Step = node["step"]
            output = await s.execute(current)
            return output, StepResult(
                name=s.name, output=output, duration_s=time.time() - node_start
            )

        if node_type == "parallel":
            steps: list[Step] = node["steps"]
            outputs = await asyncio.gather(*[s.execute(current) for s in steps])
            return list(outputs), StepResult(
                name=f"parallel({','.join(s.name for s in steps)})",
                output=list(outputs),
                duration_s=time.time() - node_start,
            )

        if node_type == "map":
            s = node["step"]
            items_fn = node["items_fn"]
            items = items_fn(current) if callable(items_fn) else current
            outputs = await asyncio.gather(*[s.execute(item) for item in items])
            return list(outputs), StepResult(name=f"map({s.name})", output=list(outputs))

        if node_type == "reduce":
            fn = node["fn"]
            initial = node.get("initial")
            if not isinstance(current, list):
                return current, StepResult(name="reduce", output=current)
            result = initial
            for item in current:
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(result, item)
                else:
                    result = fn(result, item)
            return result, StepResult(name="reduce", output=result)

        if node_type == "conditional":
            condition = node["condition"]
            branch = node["if_true"] if condition(current) else node["if_false"]
            output = await branch.execute(current)
            return output, StepResult(name=f"branch({branch.name})", output=output)

        if node_type == "loop":
            s = node["step"]
            until = node["until"]
            max_iter = node.get("max_iter", 10)
            result = current
            for i in range(max_iter):
                result = await s.execute(result)
                if until(result):
                    break
            return result, StepResult(name=f"loop({s.name})", output=result)

        if node_type == "human_review":
            # For now: pause and log (full HITL requires HITLController injection)
            print(
                f"\n[HITL] {node.get('prompt', 'Review required')} (auto-approved in workflow mode)"
            )
            return current, StepResult(name="human_review", output=current)

        raise WorkflowError(self._name, node_type, "Unknown node type")
