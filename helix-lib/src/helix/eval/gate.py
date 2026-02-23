"""
helix/eval/gate.py

RegressionGate — compares eval scores against a baseline run
and blocks deployment if scores drop beyond the tolerance.

CI/CD integration: exits with code 1 on regression.
"""

from __future__ import annotations

import contextlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from helix.eval.suite import EvalSuite


@dataclass
class Regression:
    case: str
    baseline_score: float
    current_score: float
    delta: float


@dataclass
class GateResult:
    passed: bool
    regressions: list[Regression] = field(default_factory=list)
    current_run_id: str | None = None
    baseline_run_id: str | None = None

    def summary(self) -> str:
        if self.passed:
            return "✓ Regression gate passed. No regressions detected."
        lines = [f"✗ Regression gate FAILED. {len(self.regressions)} regression(s):"]
        for r in self.regressions:
            lines.append(
                f"  {r.case}: {r.baseline_score:.3f} → {r.current_score:.3f} (Δ {r.delta:+.3f})"
            )
        return "\n".join(lines)

    def exit_if_failed(self) -> None:
        """Call this in CI scripts to fail the pipeline on regression."""
        if not self.passed:
            print(self.summary(), file=sys.stderr)
            sys.exit(1)


class RegressionGate:
    """
    Compares a current eval run against a saved baseline.

    Usage in CI::

        gate = RegressionGate(suite=suite, baseline_run_id="run_abc123")
        result = await gate.check(agent)
        result.exit_if_failed()   # exits with code 1 if regressions found
    """

    def __init__(
        self,
        suite: EvalSuite,
        baseline_run_id: str,
        tolerance: float = 0.10,  # 10% drop is acceptable
        results_dir: str = ".helix/eval_results",
    ) -> None:
        self._suite = suite
        self._baseline_run_id = baseline_run_id
        self._tolerance = tolerance
        self._results_dir = Path(results_dir)

    async def check(self, agent: Any) -> GateResult:
        """Run the eval suite and compare against the baseline."""

        baseline = self._load_baseline()
        if baseline is None:
            # No baseline yet — save current as baseline
            current = await self._suite.run(agent)
            self._save_result(current)
            return GateResult(
                passed=True,
                current_run_id=current.id,
                baseline_run_id=None,
            )

        current = await self._suite.run(agent)
        self._save_result(current)

        regressions = []
        for case_name, baseline_score in baseline.scores_by_case.items():
            current_score = current.scores_by_case.get(case_name, 0.0)
            delta = current_score - baseline_score
            if delta < -self._tolerance:
                regressions.append(
                    Regression(
                        case=case_name,
                        baseline_score=baseline_score,
                        current_score=current_score,
                        delta=delta,
                    )
                )

        return GateResult(
            passed=len(regressions) == 0,
            regressions=regressions,
            current_run_id=current.id,
            baseline_run_id=self._baseline_run_id,
        )

    def _load_baseline(self) -> Any | None:
        path = self._results_dir / f"{self._baseline_run_id}.json"
        if not path.exists():
            return None
        try:
            from helix.config import EvalRunResult

            data = json.loads(path.read_text())
            return EvalRunResult(**data)
        except Exception:
            return None

    def _save_result(self, result: Any) -> None:
        self._results_dir.mkdir(parents=True, exist_ok=True)
        path = self._results_dir / f"{result.id}.json"
        with contextlib.suppress(Exception):
            path.write_text(json.dumps(result.model_dump(mode="json"), default=str))
