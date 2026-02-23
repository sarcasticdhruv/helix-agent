"""
helix/eval/suite.py

EvalSuite — define, run, and track agent evaluations.

The integrated eval engine is Helix's strongest differentiator.
Zero external tools needed — works because Helix already owns
the agent structure, tool registry, and trace format.

Usage::

    suite = EvalSuite("my-agent")

    @suite.case
    def test_basic():
        return EvalCase(
            input="What is the capital of France?",
            expected_facts=["Paris"],
            max_cost_usd=0.05,
        )

    results = await suite.run(agent)
    suite.assert_pass_rate(0.90)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from helix.config import EvalCase, EvalCaseResult, EvalRunResult
from helix.eval.scoring import DEFAULT_SCORERS
from helix.eval.trajectory import TrajectoryScorer
from helix.interfaces import EvalScorer


class EvalSuite:
    """
    A named collection of EvalCases run against an agent.
    Results are persisted to disk for comparison and regression gating.
    """

    def __init__(
        self,
        name: str,
        scorers: Optional[List[EvalScorer]] = None,
        results_dir: str = ".helix/eval_results",
    ) -> None:
        self.name = name
        self._cases: List[EvalCase] = []
        self._scorers = scorers or (DEFAULT_SCORERS + [TrajectoryScorer()])
        self._results_dir = Path(results_dir)
        self._history: List[EvalRunResult] = []

    # ------------------------------------------------------------------
    # Case registration
    # ------------------------------------------------------------------

    def case(self, fn: Callable) -> "EvalSuite":
        """Decorator to register an EvalCase factory function."""
        result = fn()
        if isinstance(result, EvalCase):
            # Use function name as case name if not set
            if result.name == result.name:  # always true — just ensure name is set
                pass
            self._cases.append(result.model_copy(update={"name": fn.__name__}))
        return self

    def add_case(self, case: EvalCase) -> "EvalSuite":
        """Register a case directly."""
        self._cases.append(case)
        return self

    def add_cases(self, cases: List[EvalCase]) -> "EvalSuite":
        for case in cases:
            self._cases.append(case)
        return self

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    async def run(
        self,
        agent: Any,
        subset: Optional[List[str]] = None,
        verbose: bool = False,
    ) -> EvalRunResult:
        """
        Run all (or a subset of) cases against the agent.
        Returns EvalRunResult with per-case scores.
        """
        start = time.time()
        cases = (
            [c for c in self._cases if c.name in subset]
            if subset else self._cases
        )

        results: List[EvalCaseResult] = []
        for case in cases:
            if verbose:
                print(f"  Running: {case.name}...")
            result = await self._run_case(agent, case)
            results.append(result)
            if verbose:
                status = "✓" if result.passed else "✗"
                print(f"  {status} {case.name}: {result.overall:.3f}")

        run_result = EvalRunResult(
            suite_name=self.name,
            pass_count=sum(1 for r in results if r.passed),
            fail_count=sum(1 for r in results if not r.passed),
            results=results,
            total_cost_usd=sum(r.cost_usd for r in results),
            duration_s=time.time() - start,
        )
        self._history.append(run_result)
        self._save_result(run_result)
        return run_result

    async def _run_case(self, agent: Any, case: EvalCase) -> EvalCaseResult:
        """Run a single case and score the result."""
        case_start = time.time()
        output = ""
        tool_calls = []
        cost_usd = 0.0
        steps = 0
        failure_reason = None

        try:
            result = await agent.run(case.input)
            output = str(result.output)
            tool_calls = []  # AgentResult doesn't carry full ToolCallRecords — use trace
            cost_usd = result.cost_usd
            steps = result.steps
        except Exception as e:
            failure_reason = str(e)

        # Score across all scorers
        scores: Dict[str, float] = {}
        total_weight = sum(s.weight for s in self._scorers)

        for scorer in self._scorers:
            try:
                raw_score = await scorer.score(
                    case=case,
                    result_output=output,
                    tool_calls=tool_calls,
                    cost_usd=cost_usd,
                    steps=steps,
                )
                scores[scorer.name] = raw_score
            except Exception:
                scores[scorer.name] = 0.0

        # Weighted overall
        overall = sum(
            scores.get(s.name, 0.0) * s.weight
            for s in self._scorers
        )
        if total_weight > 0:
            overall /= total_weight
        overall = max(0.0, min(1.0, overall))

        return EvalCaseResult(
            case_name=case.name,
            input=case.input,
            output=output,
            passed=overall >= case.pass_threshold,
            scores=scores,
            overall=overall,
            cost_usd=cost_usd,
            steps=steps,
            duration_s=time.time() - case_start,
            failure_reason=failure_reason,
        )

    async def _run_case_from_result(
        self,
        agent_result: Any,
        case: EvalCase,
    ) -> EvalCaseResult:
        """Score an already-completed agent result (for production monitoring)."""
        output = str(getattr(agent_result, "output", ""))
        cost_usd = getattr(agent_result, "cost_usd", 0.0)
        steps = getattr(agent_result, "steps", 0)

        scores: Dict[str, float] = {}
        total_weight = sum(s.weight for s in self._scorers)

        for scorer in self._scorers:
            try:
                raw = await scorer.score(
                    case=case, result_output=output,
                    tool_calls=[], cost_usd=cost_usd, steps=steps,
                )
                scores[scorer.name] = raw
            except Exception:
                scores[scorer.name] = 0.0

        overall = sum(scores.get(s.name, 0.0) * s.weight for s in self._scorers)
        if total_weight > 0:
            overall /= total_weight
        overall = max(0.0, min(1.0, overall))

        return EvalCaseResult(
            case_name=case.name,
            input=case.input,
            output=output,
            passed=overall >= case.pass_threshold,
            scores=scores,
            overall=overall,
            cost_usd=cost_usd,
            steps=steps,
        )

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    def assert_pass_rate(
        self,
        threshold: float,
        run: Optional[EvalRunResult] = None,
    ) -> None:
        """
        Raise AssertionError if pass rate is below threshold.
        Useful in CI pipelines.
        """
        result = run or (self._history[-1] if self._history else None)
        if result is None:
            raise AssertionError("No eval results yet. Run the suite first.")
        if result.pass_rate < threshold:
            raise AssertionError(
                f"EvalSuite '{self.name}' pass rate {result.pass_rate:.2%} "
                f"is below threshold {threshold:.2%}. "
                f"Failed: {result.fail_count}/{result.pass_count + result.fail_count}"
            )

    # ------------------------------------------------------------------
    # History and persistence
    # ------------------------------------------------------------------

    def compare(self, run_a: EvalRunResult, run_b: EvalRunResult) -> Dict[str, Any]:
        """Compare two eval runs and return diff."""
        diffs = {}
        for case_name in set(run_a.scores_by_case) | set(run_b.scores_by_case):
            a = run_a.scores_by_case.get(case_name, 0.0)
            b = run_b.scores_by_case.get(case_name, 0.0)
            diffs[case_name] = {"run_a": a, "run_b": b, "delta": b - a}
        return {
            "run_a_pass_rate": run_a.pass_rate,
            "run_b_pass_rate": run_b.pass_rate,
            "delta_pass_rate": run_b.pass_rate - run_a.pass_rate,
            "case_diffs": diffs,
        }

    def load_result(self, run_id: str) -> Optional[EvalRunResult]:
        path = self._results_dir / f"{run_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return EvalRunResult(**data)
        except Exception:
            return None

    def _save_result(self, result: EvalRunResult) -> None:
        self._results_dir.mkdir(parents=True, exist_ok=True)
        path = self._results_dir / f"{result.id}.json"
        try:
            path.write_text(json.dumps(result.model_dump(mode="json"), default=str))
        except Exception:
            pass
