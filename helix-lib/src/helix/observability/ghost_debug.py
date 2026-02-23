"""
helix/observability/ghost_debug.py

GhostDebugResolver — compares two traces of the same agent task
and identifies exactly where and why they diverged.

Powers: helix trace <run_id> --diff <run_id_2>
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DivergencePoint:
    step: int
    span_name: str
    state_a: Any
    state_b: Any
    tool_name: Optional[str] = None


@dataclass
class DivergenceReport:
    identical: bool
    diverged_at_step: Optional[int] = None
    diverged_at_span: Optional[str] = None
    run_a_state: Any = None
    run_b_state: Any = None
    likely_cause: str = ""
    recommendation: str = ""


class GhostDebugResolver:
    """
    Compares two run traces to find divergence.
    Loaded from local trace files or passed as dicts.
    """

    def __init__(self, trace_dir: str = ".helix/traces") -> None:
        self._trace_dir = Path(trace_dir)

    def load_trace(self, run_id: str) -> Optional[Dict[str, Any]]:
        path = self._trace_dir / f"{run_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    async def compare(
        self,
        run_id_a: str,
        run_id_b: str,
    ) -> DivergenceReport:
        trace_a = self.load_trace(run_id_a)
        trace_b = self.load_trace(run_id_b)

        if trace_a is None:
            return DivergenceReport(
                identical=False,
                likely_cause=f"Trace for run '{run_id_a}' not found.",
            )
        if trace_b is None:
            return DivergenceReport(
                identical=False,
                likely_cause=f"Trace for run '{run_id_b}' not found.",
            )

        return self.compare_traces(trace_a, trace_b)

    def compare_traces(
        self,
        trace_a: Dict[str, Any],
        trace_b: Dict[str, Any],
    ) -> DivergenceReport:
        spans_a = trace_a.get("spans", [])
        spans_b = trace_b.get("spans", [])

        dp = self._find_divergence(spans_a, spans_b)
        if dp is None:
            return DivergenceReport(identical=True)

        return DivergenceReport(
            identical=False,
            diverged_at_step=dp.step,
            diverged_at_span=dp.span_name,
            run_a_state=dp.state_a,
            run_b_state=dp.state_b,
            likely_cause=self._diagnose(dp),
            recommendation=self._recommend(dp),
        )

    def _find_divergence(
        self,
        spans_a: List[Dict],
        spans_b: List[Dict],
    ) -> Optional[DivergencePoint]:
        """Walk span lists in parallel and return first point of difference."""
        for i, (sa, sb) in enumerate(zip(spans_a, spans_b)):
            name_a = sa.get("name", "")
            name_b = sb.get("name", "")
            if name_a != name_b:
                return DivergencePoint(
                    step=i,
                    span_name=name_a,
                    state_a=sa.get("meta", {}),
                    state_b=sb.get("meta", {}),
                )
            # Compare outputs
            out_a = sa.get("meta", {}).get("output", sa.get("meta", {}).get("content_preview"))
            out_b = sb.get("meta", {}).get("output", sb.get("meta", {}).get("content_preview"))
            if out_a != out_b:
                return DivergencePoint(
                    step=i,
                    span_name=name_a,
                    state_a={"output": out_a},
                    state_b={"output": out_b},
                    tool_name=sa.get("meta", {}).get("tool"),
                )

        # Different number of spans
        if len(spans_a) != len(spans_b):
            step = min(len(spans_a), len(spans_b))
            return DivergencePoint(
                step=step,
                span_name="(end)",
                state_a={"span_count": len(spans_a)},
                state_b={"span_count": len(spans_b)},
            )

        return None

    def _diagnose(self, dp: DivergencePoint) -> str:
        span = dp.span_name
        if "llm" in span:
            return (
                "Non-deterministic LLM output. "
                "The model produced different text on identical inputs. "
                "Consider lowering temperature or using seed (OpenAI)."
            )
        if "tool" in span:
            tool = dp.tool_name or "unknown tool"
            return (
                f"Tool '{tool}' returned different results. "
                "Check if the tool depends on real-time data, randomness, or external state."
            )
        if "context" in span:
            return (
                "Context state differed before the LLM call. "
                "Memory retrieval or context compaction may be non-deterministic."
            )
        if "(end)" in span:
            return (
                "Runs completed different numbers of steps. "
                "One run may have hit a loop limit, budget cap, or tool error the other didn't."
            )
        return "Divergence source unclear. Compare span metadata for clues."

    def _recommend(self, dp: DivergencePoint) -> str:
        span = dp.span_name
        if "llm" in span:
            return "Set temperature=0.0 for deterministic runs, or use the same seed."
        if "tool" in span:
            return "Add a mock/stub for this tool in tests to ensure deterministic results."
        if "context" in span:
            return "Pin the system prompt and disable memory retrieval for reproducibility."
        return "Use helix replay <run_id> to step through the execution interactively."
