"""
helix/models/complexity.py

ComplexityEstimator — scores a task to determine which model tier to use.

No external calls. Pure computation from message content and tool list.
"""

from __future__ import annotations

from typing import Any

from helix.config import ComplexityTier


class ComplexityEstimator:
    """
    Stateless estimator. All methods are class-level.
    """

    # Keywords that indicate higher complexity
    _COMPLEX_KEYWORDS = frozenset(
        [
            "analyze",
            "analyse",
            "compare",
            "synthesize",
            "evaluate",
            "calculate",
            "compute",
            "model",
            "optimize",
            "predict",
            "research",
            "investigate",
            "design",
            "architect",
            "plan",
            "multi-step",
            "step by step",
            "first",
            "then",
            "finally",
            "if",
            "when",
            "unless",
            "otherwise",
        ]
    )

    _SIMPLE_KEYWORDS = frozenset(
        [
            "what is",
            "define",
            "summarize",
            "list",
            "translate",
            "format",
            "convert",
            "extract",
            "find",
        ]
    )

    @classmethod
    def estimate(
        cls,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        hint: str | None = None,
    ) -> ComplexityTier:
        """
        Score the task and return a ComplexityTier.

        Score breakdown:
          - Token volume:   0–2 points
          - Tool count:     0–2 points
          - Complex words:  0–2 points
          - Simple words:  -1 point
          - Multi-step:     1 point
          - Math/analysis:  1 point

        Total → tier mapping:
          0–2  → LOW
          3–4  → MEDIUM
          5–6  → HIGH
          7+   → MAX
        """
        if hint == "max":
            return ComplexityTier.MAX
        if hint == "low":
            return ComplexityTier.LOW

        text = " ".join(
            m.get("content", "") for m in messages if isinstance(m.get("content"), str)
        ).lower()

        score = 0.0

        # Token volume (rough: chars / 4)
        token_estimate = len(text) / 4
        score += min(token_estimate / 1000, 2.0) * 0.5

        # Tool count
        score += min(len(tools) / 5, 1.0) * 1.5

        # Complex keywords
        complex_hits = sum(1 for kw in cls._COMPLEX_KEYWORDS if kw in text)
        score += min(complex_hits * 0.3, 2.0)

        # Simple keywords penalize
        simple_hits = sum(1 for kw in cls._SIMPLE_KEYWORDS if kw in text)
        score -= min(simple_hits * 0.5, 1.0)

        # Multi-step indicators
        multi_step = any(kw in text for kw in ("step 1", "first,", "then ", "finally,", "next,"))
        if multi_step:
            score += 1.0

        # Math / analysis
        math_hit = any(kw in text for kw in ("calculate", "compute", "analyse", "analyze", "model"))
        if math_hit:
            score += 1.0

        score = max(0.0, score)

        if score < 2.0:
            return ComplexityTier.LOW
        if score < 4.0:
            return ComplexityTier.MEDIUM
        if score < 6.0:
            return ComplexityTier.HIGH
        return ComplexityTier.MAX
