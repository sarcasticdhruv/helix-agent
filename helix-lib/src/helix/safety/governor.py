"""
helix/safety/governor.py

CostGovernor — enforces budget limits before every LLM call.

Law 3 of Helix architecture: Cost is a gate, not a report.
The governor runs BEFORE the call. It does not record what happened —
it prevents it from happening.
"""

from __future__ import annotations

import asyncio
from typing import Any

from helix.config import BudgetConfig, ModelPricing, TokenUsage
from helix.errors import BudgetExceededError

# Default pricing table (USD per 1K tokens)
DEFAULT_PRICING: dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(
        model="gpt-4o",
        prompt_cost_per_1k=0.0025,
        completion_cost_per_1k=0.010,
        cached_cost_per_1k=0.00125,
    ),
    "gpt-4o-mini": ModelPricing(
        model="gpt-4o-mini",
        prompt_cost_per_1k=0.00015,
        completion_cost_per_1k=0.0006,
        cached_cost_per_1k=0.000075,
    ),
    "claude-sonnet-4-6": ModelPricing(
        model="claude-sonnet-4-6",
        prompt_cost_per_1k=0.003,
        completion_cost_per_1k=0.015,
        cached_cost_per_1k=0.0003,
    ),
    "claude-haiku-4-5-20251001": ModelPricing(
        model="claude-haiku-4-5-20251001",
        prompt_cost_per_1k=0.0008,
        completion_cost_per_1k=0.004,
        cached_cost_per_1k=0.00008,
    ),
    "claude-opus-4-6": ModelPricing(
        model="claude-opus-4-6",
        prompt_cost_per_1k=0.015,
        completion_cost_per_1k=0.075,
        cached_cost_per_1k=0.0015,
    ),
}


class CostGovernor:
    """
    Budget enforcement for agent runs.

    Usage::

        gov = CostGovernor(BudgetConfig(budget_usd=1.00))
        await gov.check_gate("gpt-4o", estimated_tokens=1000)
        # ... call LLM ...
        gov.record(usage, model="gpt-4o")
        print(gov.report())
    """

    def __init__(
        self,
        config: BudgetConfig,
        agent_id: str = "",
        pricing: dict[str, ModelPricing] | None = None,
    ) -> None:
        self._config = config
        self._agent_id = agent_id
        self._pricing = pricing or DEFAULT_PRICING
        self._spent_usd: float = 0.0
        self._calls: int = 0
        self._lock = asyncio.Lock()
        self._warned: bool = False

    async def check_gate(
        self,
        model: str,
        estimated_tokens: int = 1000,
    ) -> None:
        """
        Raises BudgetExceededError if the next call would exceed budget.
        Must be called BEFORE every LLM call.
        """
        estimated_cost = self._estimate_cost(model, estimated_tokens)
        async with self._lock:
            if self._spent_usd + estimated_cost > self._config.budget_usd:
                raise BudgetExceededError(
                    agent_id=self._agent_id,
                    budget_usd=self._config.budget_usd,
                    spent_usd=self._spent_usd,
                    attempted_usd=estimated_cost,
                )

    async def record(self, usage: TokenUsage, model: str) -> None:
        """Record actual token usage after a successful LLM call."""
        cost = self._calculate_cost(usage, model)
        async with self._lock:
            self._spent_usd += cost
            self._calls += 1

    async def record_from_response(self, response: Any, model: str) -> None:
        """
        Extract token usage from a raw provider response.
        Falls back to tiktoken estimation when provider doesn't report usage.
        """
        usage = self._extract_usage(response, model)
        await self.record(usage, model)

    def should_warn(self) -> bool:
        """True once when budget crosses the warn threshold."""
        if self._warned:
            return False
        pct = self._spent_usd / self._config.budget_usd if self._config.budget_usd else 0
        if pct >= self._config.warn_at_pct:
            self._warned = True
            return True
        return False

    def pct_used(self) -> float:
        if not self._config.budget_usd:
            return 0.0
        return self._spent_usd / self._config.budget_usd

    def report(self) -> dict[str, Any]:
        return {
            "spent_usd": round(self._spent_usd, 6),
            "budget_usd": self._config.budget_usd,
            "pct_used": round(self.pct_used(), 4),
            "calls": self._calls,
            "remaining_usd": round(max(0, self._config.budget_usd - self._spent_usd), 6),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_cost(self, model: str, estimated_tokens: int) -> float:
        pricing = self._pricing.get(model)
        if not pricing:
            return estimated_tokens / 1000 * 0.005  # Conservative default
        return estimated_tokens / 1000 * pricing.prompt_cost_per_1k

    def _calculate_cost(self, usage: TokenUsage, model: str) -> float:
        pricing = self._pricing.get(model)
        if not pricing:
            return usage.total_tokens / 1000 * 0.005
        return pricing.calculate_cost(usage)

    def _extract_usage(self, response: Any, model: str) -> TokenUsage:
        """Try to extract token usage from any provider response format."""
        # OpenAI
        if hasattr(response, "usage") and response.usage:
            u = response.usage
            return TokenUsage(
                prompt_tokens=getattr(u, "prompt_tokens", 0),
                completion_tokens=getattr(u, "completion_tokens", 0),
            )
        # Anthropic
        if hasattr(response, "usage"):
            u = response.usage
            return TokenUsage(
                prompt_tokens=getattr(u, "input_tokens", 0),
                completion_tokens=getattr(u, "output_tokens", 0),
            )
        # LangChain
        if hasattr(response, "llm_output") and response.llm_output:
            tu = response.llm_output.get("token_usage", {})
            if tu:
                return TokenUsage(
                    prompt_tokens=tu.get("prompt_tokens", 0),
                    completion_tokens=tu.get("completion_tokens", 0),
                )
        # Tiktoken fallback
        return self._tiktoken_estimate(response, model)

    def _tiktoken_estimate(self, response: Any, model: str) -> TokenUsage:
        try:
            import tiktoken

            safe_model = model if "gpt" in model else "gpt-4"
            enc = tiktoken.encoding_for_model(safe_model)
            content = ""
            if hasattr(response, "content"):
                content = str(response.content)
            return TokenUsage(
                prompt_tokens=0,
                completion_tokens=len(enc.encode(content)),
            )
        except Exception:
            return TokenUsage(completion_tokens=len(str(response)) // 4)
