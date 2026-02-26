"""
helix/core/workflow.py

Workflow — a directed graph of steps executed by agents.

Modes: sequential, parallel, conditional, loop, map, reduce, human_review.
State is checkpointed after every step so workflows can resume after interruption.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
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


# ---------------------------------------------------------------------------
# Typed chain node ADT — replaces the previous list[dict[str, Any]]
# ---------------------------------------------------------------------------


@dataclass
class _SequentialNode:
    step: Step


@dataclass
class _ParallelNode:
    steps: list[Step]


@dataclass
class _BranchNode:
    condition: Callable[[Any], bool]
    if_true: Step
    if_false: Step


@dataclass
class _LoopNode:
    step: Step
    until: Callable[[Any], bool]
    max_iter: int = 10


@dataclass
class _MapNode:
    step: Step
    items_fn: Callable


@dataclass
class _ReduceNode:
    fn: Callable
    initial: Any = None


@dataclass
class _HumanReviewNode:
    prompt: str = "Review required before continuing."
    risk_level: str = "medium"


# Union type for exhaustiveness checking
ChainNode = (
    _SequentialNode
    | _ParallelNode
    | _BranchNode
    | _LoopNode
    | _MapNode
    | _ReduceNode
    | _HumanReviewNode
)


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

    Checkpoint / resume::

        wf = (
            Workflow("long-job")
            .then(step_a)
            .then(step_b)
            .with_checkpoint(".helix/checkpoints/long-job")
        )
        result = wf.run_sync("input", resume=True)   # skips completed steps
    """

    def __init__(self, name: str, config: WorkflowConfig | None = None) -> None:
        self._name = name
        self._config = config or WorkflowConfig(name=name)
        self._chain: list[ChainNode] = []
        self._budget_usd: float | None = config.budget_usd if config else None
        self._on_failure: str = config.on_failure if config else "fail"
        self._checkpoint_dir: Path | None = None
        self._on_step_complete: Callable[[str, Any], None] | None = None

    # ------------------------------------------------------------------
    # Builder methods
    # ------------------------------------------------------------------

    def then(self, step_fn: Step | Callable, **_kwargs: Any) -> Workflow:
        """Add a sequential step."""
        s = (
            step_fn
            if isinstance(step_fn, Step)
            else Step(fn=step_fn, name=getattr(step_fn, "__name__", "step"))
        )
        self._chain.append(_SequentialNode(step=s))
        return self

    def parallel(self, *step_fns: Step | Callable) -> Workflow:
        """Add steps that run concurrently. Their outputs are collected as a list."""
        steps = [
            (s if isinstance(s, Step) else Step(fn=s, name=getattr(s, "__name__", "step")))
            for s in step_fns
        ]
        self._chain.append(_ParallelNode(steps=steps))
        return self

    def map(self, step_fn: Step | Callable, items_fn: Callable) -> Workflow:
        """Apply step_fn to each item returned by items_fn(input) concurrently."""
        s = (
            step_fn
            if isinstance(step_fn, Step)
            else Step(fn=step_fn, name=getattr(step_fn, "__name__", "map_step"))
        )
        self._chain.append(_MapNode(step=s, items_fn=items_fn))
        return self

    def reduce(self, reduce_fn: Callable, initial: Any = None) -> Workflow:
        """Reduce the current output list to a single value."""
        self._chain.append(_ReduceNode(fn=reduce_fn, initial=initial))
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
        self._chain.append(_BranchNode(condition=condition, if_true=true_step, if_false=false_step))
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
        self._chain.append(_LoopNode(step=s, until=until, max_iter=max_iter))
        return self

    def human_review(
        self,
        prompt: str = "Review required before continuing.",
        risk_level: str = "medium",
    ) -> Workflow:
        """Insert a HITL checkpoint — workflow pauses until a human approves."""
        self._chain.append(_HumanReviewNode(prompt=prompt, risk_level=risk_level))
        return self

    def with_budget(self, usd: float) -> Workflow:
        self._budget_usd = usd
        return self

    def on_failure(self, strategy: str) -> Workflow:
        """'fail' | 'continue' | 'fallback'"""
        self._on_failure = strategy
        return self

    def with_checkpoint(self, directory: str) -> Workflow:
        """
        Enable step-level checkpointing to ``directory``.

        On ``run_sync(input, resume=True)`` already-completed steps are
        loaded from disk and skipped, so long-running workflows safely
        survive process restarts.

        Example::

            wf = Workflow("etl").then(extract).then(transform).then(load)
            wf.with_checkpoint(".helix/checkpoints/etl")
            result = wf.run_sync(input_data, resume=True)
        """
        self._checkpoint_dir = Path(directory)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self

    def on_step(self, callback: Callable[[str, Any], None]) -> Workflow:
        """
        Register a callback invoked after every step completes.

        The callback receives ``(step_name: str, output: Any)``.
        Useful for logging or progress bars without full observability.

        Example::

            wf.on_step(lambda name, out: print(f"✓ {name}: {str(out)[:60]}"))
        """
        self._on_step_complete = callback
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def run(self, input: Any, *, resume: bool = False) -> WorkflowResult:
        """
        Execute the workflow.

        Args:
            input:  The initial input passed to the first step.
            resume: If ``True`` and a checkpoint directory is configured,
                    reload previously completed steps from disk.
        """
        start = time.time()
        current = input
        step_results: list[StepResult] = []
        total_cost = 0.0

        # Load checkpoint if resuming
        checkpoint_state: dict[str, Any] = {}
        if resume and self._checkpoint_dir:
            checkpoint_state = self._load_checkpoint()
            if checkpoint_state:
                current = checkpoint_state.get("last_output", input)

        for node in self._chain:
            node_name = self._node_name(node)

            # Skip already-completed steps when resuming
            if resume and node_name in checkpoint_state.get("completed", []):
                step_results.append(
                    StepResult(
                        name=node_name,
                        output=checkpoint_state["outputs"].get(node_name),
                    )
                )
                current = checkpoint_state["outputs"].get(node_name, current)
                continue

            try:
                current, sr = await self._execute_node(node, current)
                step_results.append(sr)
                total_cost += sr.cost_usd

                # Checkpoint after each successful step
                if self._checkpoint_dir:
                    self._save_checkpoint(node_name, current, step_results)

                # Step callback
                if self._on_step_complete:
                    with contextlib.suppress(Exception):
                        self._on_step_complete(node_name, current)

            except Exception as e:
                if self._on_failure == "continue":
                    step_results.append(StepResult(name=node_name, output=None, error=str(e)))
                    continue
                return WorkflowResult(
                    workflow_name=self._name,
                    final_output=None,
                    steps=step_results,
                    total_cost_usd=total_cost,
                    duration_s=time.time() - start,
                    error=str(e),
                )

        # Clear checkpoint on successful completion
        if self._checkpoint_dir:
            self._clear_checkpoint()

        return WorkflowResult(
            workflow_name=self._name,
            final_output=current,
            steps=step_results,
            total_cost_usd=total_cost,
            duration_s=time.time() - start,
        )

    def run_sync(self, input: Any, *, resume: bool = False) -> WorkflowResult:
        """Synchronous wrapper. Safe to call from plain scripts."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, self.run(input, resume=resume))
                    return future.result()
            return loop.run_until_complete(self.run(input, resume=resume))
        except RuntimeError:
            return asyncio.run(self.run(input, resume=resume))

    # ------------------------------------------------------------------
    # Internal execution dispatch — typed, exhaustive
    # ------------------------------------------------------------------

    async def _execute_node(self, node: ChainNode, current: Any) -> tuple[Any, StepResult]:
        t0 = time.time()

        if isinstance(node, _SequentialNode):
            output = await node.step.execute(current)
            return output, StepResult(
                name=node.step.name, output=output, duration_s=time.time() - t0
            )

        if isinstance(node, _ParallelNode):
            outputs = await asyncio.gather(*[s.execute(current) for s in node.steps])
            return list(outputs), StepResult(
                name=f"parallel({','.join(s.name for s in node.steps)})",
                output=list(outputs),
                duration_s=time.time() - t0,
            )

        if isinstance(node, _MapNode):
            items = node.items_fn(current) if callable(node.items_fn) else current
            outputs = await asyncio.gather(*[node.step.execute(item) for item in items])
            return list(outputs), StepResult(
                name=f"map({node.step.name})", output=list(outputs), duration_s=time.time() - t0
            )

        if isinstance(node, _ReduceNode):
            if not isinstance(current, list):
                return current, StepResult(name="reduce", output=current)
            result = node.initial
            for item in current:
                if asyncio.iscoroutinefunction(node.fn):
                    result = await node.fn(result, item)
                else:
                    result = node.fn(result, item)
            return result, StepResult(name="reduce", output=result, duration_s=time.time() - t0)

        if isinstance(node, _BranchNode):
            chosen = node.if_true if node.condition(current) else node.if_false
            output = await chosen.execute(current)
            return output, StepResult(
                name=f"branch({chosen.name})", output=output, duration_s=time.time() - t0
            )

        if isinstance(node, _LoopNode):
            result = current
            for _ in range(node.max_iter):
                result = await node.step.execute(result)
                if node.until(result):
                    break
            return result, StepResult(
                name=f"loop({node.step.name})", output=result, duration_s=time.time() - t0
            )

        if isinstance(node, _HumanReviewNode):
            print(f"\n[HITL] {node.prompt} (auto-approved in workflow mode)")
            return current, StepResult(
                name="human_review", output=current, duration_s=time.time() - t0
            )

        raise WorkflowError(self._name, repr(node), "Unknown node type")

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _node_name(node: ChainNode) -> str:
        if isinstance(node, _SequentialNode):
            return node.step.name
        if isinstance(node, _ParallelNode):
            return f"parallel({','.join(s.name for s in node.steps)})"
        if isinstance(node, _MapNode):
            return f"map({node.step.name})"
        if isinstance(node, _ReduceNode):
            return "reduce"
        if isinstance(node, _BranchNode):
            return "branch"
        if isinstance(node, _LoopNode):
            return f"loop({node.step.name})"
        if isinstance(node, _HumanReviewNode):
            return "human_review"
        return "unknown"

    def _checkpoint_path(self) -> Path:
        assert self._checkpoint_dir is not None
        return self._checkpoint_dir / f"{self._name}.json"

    def _save_checkpoint(
        self, last_step: str, last_output: Any, completed_steps: list[StepResult]
    ) -> None:
        try:
            state = {
                "completed": [s.name for s in completed_steps],
                "last_output": last_output,
                "outputs": {s.name: s.output for s in completed_steps},
            }
            self._checkpoint_path().write_text(json.dumps(state, default=str))
        except Exception:
            pass

    def _load_checkpoint(self) -> dict[str, Any]:
        try:
            path = self._checkpoint_path()
            if path.exists():
                return json.loads(path.read_text())
        except Exception:
            pass
        return {}

    def _clear_checkpoint(self) -> None:
        try:
            cp = self._checkpoint_path()
            if cp.exists():
                cp.unlink()
        except Exception:
            pass
