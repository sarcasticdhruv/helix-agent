"""
helix/observability/replay.py

FailureReplay — interactive step-through of a failed agent run.

Allows inspecting any step's input/output, overriding it, and
re-running the agent from that point forward with the override applied.

Powers: helix replay <run_id>
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class StepSnapshot:
    step: int
    name: str
    input: Any | None
    output: Any | None
    duration_ms: float | None
    error: str | None
    cost_usd: float | None = None


class FailureReplay:
    """
    Interactive replay of a failed agent run.

    Load a trace, inspect steps, override outputs, and re-run
    from any point to diagnose and fix failures without re-running
    the entire expensive pipeline.

    Usage::

        replay = FailureReplay.from_run_id("run_abc123")
        snapshot = replay.inspect_step(3)
        print(snapshot.output)

        replay.override_step(3, new_output="corrected output")
        result = await replay.resume_from(3, agent)
    """

    def __init__(self, trace: dict[str, Any]) -> None:
        self._trace = trace
        self._spans: list[dict[str, Any]] = trace.get("spans", [])
        self._overrides: dict[int, Any] = {}

    @classmethod
    def from_run_id(cls, run_id: str, trace_dir: str = ".helix/traces") -> FailureReplay:
        path = Path(trace_dir) / f"{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Trace not found: {path}")
        trace = json.loads(path.read_text())
        return cls(trace)

    @classmethod
    def from_trace(cls, trace: dict[str, Any]) -> FailureReplay:
        return cls(trace)

    def inspect_step(self, step_n: int) -> StepSnapshot:
        """Return a snapshot of a specific step's state."""
        if step_n >= len(self._spans):
            raise IndexError(f"Step {step_n} out of range. Run has {len(self._spans)} spans.")
        span = self._spans[step_n]
        meta = span.get("meta", {})
        return StepSnapshot(
            step=step_n,
            name=span.get("name", "unknown"),
            input=meta.get("input"),
            output=self._overrides.get(step_n, meta.get("output")),
            duration_ms=span.get("duration_ms"),
            error=span.get("error"),
            cost_usd=meta.get("cost_usd"),
        )

    def inspect_all(self) -> list[StepSnapshot]:
        """Return snapshots for all steps."""
        return [self.inspect_step(i) for i in range(len(self._spans))]

    def override_step(self, step_n: int, new_output: Any) -> None:
        """
        Replace a step's output. All subsequent steps are invalidated.
        The override is applied when resume_from() is called.
        """
        self._overrides[step_n] = new_output
        # Invalidate all later overrides
        to_remove = [k for k in self._overrides if k > step_n]
        for k in to_remove:
            del self._overrides[k]

    async def resume_from(self, step_n: int, agent: Any) -> Any:
        """
        Re-run the agent from step_n, injecting all overridden step outputs
        as pre-existing context.

        The agent must have been initialized before calling this.
        """
        from helix.context import ExecutionContext

        # Build a fresh context restoring state up to step_n
        ctx = ExecutionContext(config=agent.config)

        # Replay messages from trace into context window
        for i in range(step_n):
            span = self._spans[i]
            meta = span.get("meta", {})

            # Inject override or original output
            output = self._overrides.get(i, meta.get("output"))
            if output is not None:
                await ctx.window.inject_step_output(i, output)

            ctx.window.tick()

        # Inject the override for step_n itself
        if step_n in self._overrides:
            await ctx.window.inject_step_output(step_n, self._overrides[step_n])
            ctx.window.tick()

        # Resume agent from this reconstructed context
        task = self._trace.get("task", "")
        return await agent._reasoning_loop(ctx, task)

    def summary(self) -> str:
        """Human-readable summary of the run for interactive debugging."""
        lines = [
            f"Run ID:   {self._trace.get('run_id', 'unknown')}",
            f"Agent:    {self._trace.get('agent_name', 'unknown')}",
            f"Duration: {self._trace.get('duration_s', 0):.2f}s",
            f"Spans:    {len(self._spans)}",
            "",
            "Steps:",
        ]
        for i, span in enumerate(self._spans):
            override_marker = " [OVERRIDDEN]" if i in self._overrides else ""
            error_marker = " [ERROR]" if span.get("error") else ""
            lines.append(
                f"  {i:2d}. {span.get('name', 'unknown')}"
                f" ({span.get('duration_ms', 0):.0f}ms)"
                f"{override_marker}{error_marker}"
            )
        return "\n".join(lines)
