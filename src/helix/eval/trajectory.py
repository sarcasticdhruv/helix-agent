"""
helix/eval/trajectory.py

Trajectory evaluation — scores the path the agent took, not just the output.

Evaluates: tool sequence accuracy, efficiency, and loop detection.
"""

from __future__ import annotations

from helix.config import EvalCase, ToolCallRecord
from helix.interfaces import EvalScorer


class TrajectoryScorer(EvalScorer):
    """
    Scores the agent's execution path:
      - Did it call tools in the expected sequence?
      - Did it avoid unnecessary steps?
      - Did it avoid calling forbidden tools?
      - Did it avoid loops?
    """

    @property
    def name(self) -> str:
        return "trajectory"

    @property
    def weight(self) -> float:
        return 0.20

    async def score(
        self,
        case: EvalCase,
        result_output: str,
        tool_calls: list[ToolCallRecord],
        cost_usd: float,
        steps: int,
    ) -> float:
        if not case.expected_trajectory:
            return 1.0  # No trajectory expectation — full score

        traj = case.expected_trajectory
        actual_tools = [tc.tool_name for tc in tool_calls]

        # Forbidden tools check — hard penalty
        for forbidden in traj.must_not_call:
            if forbidden in actual_tools:
                return 0.0

        score = 0.0
        components = 0

        # Sequence match (if expected sequence defined)
        if traj.tool_sequence:
            seq_score = self._sequence_match(actual_tools, traj.tool_sequence)
            score += seq_score
            components += 1

        # Efficiency: fewer steps is better
        if traj.max_steps:
            efficiency = min(1.0, traj.max_steps / max(len(actual_tools), 1))
            score += efficiency
            components += 1

        # Loop detection penalty
        has_loop = len(actual_tools) != len(set(actual_tools))
        loop_penalty = 0.3 if has_loop else 0.0

        if components == 0:
            return 1.0 - loop_penalty

        return max(0.0, (score / components) - loop_penalty)

    def _sequence_match(self, actual: list[str], expected: list[str]) -> float:
        """
        Longest common subsequence ratio.
        Allows extra steps but rewards hitting expected steps in order.
        """
        if not expected:
            return 1.0
        if not actual:
            return 0.0

        # LCS dynamic programming
        m, n = len(actual), len(expected)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if actual[i - 1] == expected[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_length = dp[m][n]
        return lcs_length / len(expected)
