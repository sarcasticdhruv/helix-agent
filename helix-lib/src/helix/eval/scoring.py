"""
helix/eval/scoring.py

Eval scoring implementations.

Deterministic scorers: tool selection, cost, step count.
LLM-as-judge scorers: factual accuracy, quality.
"""

from __future__ import annotations

from helix.config import EvalCase, ToolCallRecord
from helix.interfaces import EvalScorer


class ToolSelectionScorer(EvalScorer):
    """Scores whether the agent called the expected tools."""

    @property
    def name(self) -> str:
        return "tool_selection"

    @property
    def weight(self) -> float:
        return 0.25

    async def score(
        self,
        case: EvalCase,
        result_output: str,
        tool_calls: list[ToolCallRecord],
        cost_usd: float,
        steps: int,
    ) -> float:
        if not case.expected_tools:
            return 1.0  # No expectation — full score
        called = {tc.tool_name for tc in tool_calls}
        expected = set(case.expected_tools)
        if not expected:
            return 1.0
        intersection = called & expected
        return len(intersection) / len(expected)


class CostScorer(EvalScorer):
    """Scores whether the agent stayed within the cost budget for the case."""

    @property
    def name(self) -> str:
        return "cost"

    @property
    def weight(self) -> float:
        return 0.10

    async def score(
        self,
        case: EvalCase,
        result_output: str,
        tool_calls: list[ToolCallRecord],
        cost_usd: float,
        steps: int,
    ) -> float:
        if cost_usd <= case.max_cost_usd:
            return 1.0
        # Partial score: 0.5 if within 2x, 0.0 beyond
        if cost_usd <= case.max_cost_usd * 2:
            return 0.5
        return 0.0


class StepScorer(EvalScorer):
    """Scores efficiency — did the agent complete the task in few steps?"""

    @property
    def name(self) -> str:
        return "steps"

    @property
    def weight(self) -> float:
        return 0.05

    async def score(
        self,
        case: EvalCase,
        result_output: str,
        tool_calls: list[ToolCallRecord],
        cost_usd: float,
        steps: int,
    ) -> float:
        if steps <= case.max_steps:
            return 1.0
        if steps <= case.max_steps * 1.5:
            return 0.5
        return 0.0


class FactScorer(EvalScorer):
    """
    LLM-as-judge: checks whether expected facts appear in the output.
    Uses a cheap model to avoid eval becoming expensive.
    """

    def __init__(self, judge_model: str = "gpt-4o-mini") -> None:
        self._judge_model = judge_model

    @property
    def name(self) -> str:
        return "facts"

    @property
    def weight(self) -> float:
        return 0.35

    async def score(
        self,
        case: EvalCase,
        result_output: str,
        tool_calls: list[ToolCallRecord],
        cost_usd: float,
        steps: int,
    ) -> float:
        if not case.expected_facts:
            return 1.0

        # Simple containment check first (fast path)
        output_lower = result_output.lower()
        hits = sum(1 for fact in case.expected_facts if fact.lower() in output_lower)
        simple_score = hits / len(case.expected_facts)

        # If score is already 1.0, skip LLM judge
        if simple_score == 1.0:
            return 1.0

        # LLM judge for semantic fact verification
        try:
            from helix.models.router import ModelRouter

            router = ModelRouter(primary_model=self._judge_model)
            facts_str = "\n".join(f"- {f}" for f in case.expected_facts)
            prompt = (
                f"Does this response contain the following facts? "
                f"Answer only with a number between 0.0 and 1.0 representing "
                f"the fraction of facts present.\n\n"
                f"Expected facts:\n{facts_str}\n\n"
                f"Response:\n{result_output[:2000]}\n\n"
                f"Score (0.0 to 1.0):"
            )
            response = await router.complete(
                messages=[{"role": "user", "content": prompt}],
                model=self._judge_model,
                max_tokens=10,
                temperature=0.0,
            )
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except Exception:
            return simple_score


class QualityScorer(EvalScorer):
    """
    LLM-as-judge: scores overall response quality (coherence, helpfulness, accuracy).
    """

    def __init__(self, judge_model: str = "gpt-4o-mini") -> None:
        self._judge_model = judge_model

    @property
    def name(self) -> str:
        return "quality"

    @property
    def weight(self) -> float:
        return 0.25

    async def score(
        self,
        case: EvalCase,
        result_output: str,
        tool_calls: list[ToolCallRecord],
        cost_usd: float,
        steps: int,
    ) -> float:
        try:
            from helix.models.router import ModelRouter

            router = ModelRouter(primary_model=self._judge_model)
            prompt = (
                f"Rate the quality of this response to the question on a scale of 0.0 to 1.0. "
                f"Consider: accuracy, completeness, clarity, and helpfulness. "
                f"Output only a number.\n\n"
                f"Question: {case.input}\n\n"
                f"Response: {result_output[:2000]}\n\n"
                f"Quality score (0.0 to 1.0):"
            )
            response = await router.complete(
                messages=[{"role": "user", "content": prompt}],
                model=self._judge_model,
                max_tokens=10,
                temperature=0.0,
            )
            score = float(response.content.strip())
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5  # Neutral fallback


# Default scorer set used by EvalSuite
DEFAULT_SCORERS: list[EvalScorer] = [
    ToolSelectionScorer(),
    CostScorer(),
    StepScorer(),
    FactScorer(),
    QualityScorer(),
]
