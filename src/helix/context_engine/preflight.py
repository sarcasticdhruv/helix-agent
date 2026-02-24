"""
helix/context_engine/preflight.py

PreflightEstimator — estimates cost before a run starts.
Powers --dry-run in the CLI and the pre-flight budget check in Agent.
"""

from __future__ import annotations

from typing import Any

from helix.config import AgentConfig


class PreflightEstimate:
    """Result of a pre-flight cost estimation."""

    def __init__(
        self,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
        estimated_steps: int,
        estimated_cost_usd: float,
        confidence: str,
        warning: str | None = None,
    ) -> None:
        self.estimated_input_tokens = estimated_input_tokens
        self.estimated_output_tokens = estimated_output_tokens
        self.estimated_steps = estimated_steps
        self.estimated_cost_usd = estimated_cost_usd
        self.confidence = confidence  # "low" | "medium" | "high"
        self.warning = warning

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimated_input_tokens": self.estimated_input_tokens,
            "estimated_output_tokens": self.estimated_output_tokens,
            "estimated_steps": self.estimated_steps,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "confidence": self.confidence,
            "warning": self.warning,
        }

    def __repr__(self) -> str:
        return (
            f"PreflightEstimate(cost=${self.estimated_cost_usd:.4f}, "
            f"steps≈{self.estimated_steps}, confidence={self.confidence})"
        )


class PreflightEstimator:
    """
    Estimates the cost of an agent run before it starts.

    Uses:
      1. Task token count (static)
      2. System prompt token count (static)
      3. Historical episode data for similar tasks (dynamic)
      4. Heuristic step/token multipliers per model

    This is intentionally simple — a rough estimate, not a guarantee.
    """

    # Average tokens per step for common models (from benchmarks)
    _TOKENS_PER_STEP: dict[str, dict[str, int]] = {
        "gpt-4o": {"input": 1200, "output": 400},
        "gpt-4o-mini": {"input": 800, "output": 300},
        "claude-sonnet-4-6": {"input": 1500, "output": 500},
        "claude-haiku-4-5-20251001": {"input": 600, "output": 200},
        "claude-opus-4-6": {"input": 2000, "output": 600},
    }

    _DEFAULT_TOKENS: dict[str, int] = {"input": 1000, "output": 350}

    def estimate(
        self,
        task: str,
        config: AgentConfig,
        system_prompt: str = "",
        similar_episodes: list | None = None,
    ) -> PreflightEstimate:
        model = config.model.primary
        token_rates = self._TOKENS_PER_STEP.get(model, self._DEFAULT_TOKENS)

        # Token counts
        task_tokens = len(task.split()) * 1.3
        system_tokens = len(system_prompt.split()) * 1.3
        base_input = int(task_tokens + system_tokens)

        # Step estimation
        if similar_episodes:
            avg_steps = sum(ep.steps for ep in similar_episodes) / len(similar_episodes)
            estimated_steps = max(1, int(avg_steps))
            confidence = "high" if len(similar_episodes) >= 3 else "medium"
        else:
            # Heuristic: simple tasks = 2-3 steps, complex = 5-8
            words = len(task.split())
            estimated_steps = min(3 + words // 20, config.loop_limit // 2)
            confidence = "low"

        # Cost calculation
        total_input = base_input + estimated_steps * token_rates["input"]
        total_output = estimated_steps * token_rates["output"]

        from helix.models.router import MODEL_PRICING

        pricing = MODEL_PRICING.get(model, {"prompt": 0.005, "completion": 0.015})
        cost = total_input / 1000 * pricing.get(
            "prompt", 0.005
        ) + total_output / 1000 * pricing.get("completion", 0.015)

        warning = None
        if config.budget and cost > config.budget.budget_usd * 0.8:
            warning = (
                f"Estimated cost ${cost:.4f} is close to budget ${config.budget.budget_usd:.4f}. "
                "Consider decomposing the task or increasing the budget."
            )

        return PreflightEstimate(
            estimated_input_tokens=total_input,
            estimated_output_tokens=total_output,
            estimated_steps=estimated_steps,
            estimated_cost_usd=cost,
            confidence=confidence,
            warning=warning,
        )
