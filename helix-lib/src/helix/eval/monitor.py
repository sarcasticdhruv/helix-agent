"""
helix/eval/monitor.py

ProductionEvalMonitor — samples live agent runs and evaluates them
asynchronously in the background.

Never on the critical path. Alerts when quality degrades in production.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from helix.config import EvalCase
    from helix.eval.suite import EvalSuite


class ProductionEvalMonitor:
    """
    Samples production agent runs and evaluates them in the background.

    Usage::

        monitor = ProductionEvalMonitor(suite=suite, sample_rate=0.05)

        # In your agent result handler:
        await monitor.maybe_eval(result=agent_result, case=eval_case)
    """

    def __init__(
        self,
        suite: EvalSuite,
        sample_rate: float = 0.05,
        alert_fn: Callable | None = None,
    ) -> None:
        self._suite = suite
        self._sample_rate = sample_rate
        self._alert_fn = alert_fn or self._default_alert
        self._results: list[Any] = []
        self._lock = asyncio.Lock()

    async def maybe_eval(
        self,
        result: Any,  # AgentResult
        case: EvalCase,
    ) -> None:
        """
        Probabilistically evaluate a production result.
        Always returns immediately — evaluation is fire-and-forget.
        """
        if random.random() > self._sample_rate:
            return
        asyncio.create_task(self._evaluate(result, case))

    async def _evaluate(self, result: Any, case: EvalCase) -> None:
        try:
            eval_result = await self._suite._run_case_from_result(result, case)
            async with self._lock:
                self._results.append(eval_result)
            if not eval_result.passed:
                await self._alert_fn(eval_result)
        except Exception:
            pass

    async def _default_alert(self, result: Any) -> None:
        print(
            f"[HELIX MONITOR] Quality degradation detected in production:\n"
            f"  Case: {result.case_name}\n"
            f"  Score: {result.overall:.3f}\n"
            f"  Input: {result.input[:80]}...\n"
        )

    def recent_pass_rate(self, n: int = 100) -> float:
        """Return pass rate for the last n evaluated results."""
        recent = self._results[-n:]
        if not recent:
            return 1.0
        return sum(1 for r in recent if r.passed) / len(recent)

    def report(self) -> dict[str, Any]:
        return {
            "total_sampled": len(self._results),
            "pass_rate": self.recent_pass_rate(),
            "sample_rate": self._sample_rate,
        }
