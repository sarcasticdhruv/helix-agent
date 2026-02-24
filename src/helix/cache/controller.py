"""
helix/cache/controller.py

CacheController — unified interface for all three cache tiers:
  Tier 1: SemanticCache (similar query → cached response)
  Tier 2: PlanCache (similar task → cached plan structure)
  Tier 3: PrefixCacheHook (long system prompt → provider-side cache)

The controller is the only cache object Agent needs to know about.
"""

from __future__ import annotations

from typing import Any

from helix.cache.plan import PlanCache
from helix.cache.prefix import PrefixCacheHook
from helix.cache.semantic import SemanticCache
from helix.config import CacheConfig, CacheHit


class CacheController:
    """
    Orchestrates lookup across semantic and plan caches.
    Exposes a single .lookup() method to the agent runtime.
    """

    def __init__(
        self,
        config: CacheConfig,
        semantic: SemanticCache | None = None,
        plan: PlanCache | None = None,
    ) -> None:
        self._config = config
        self.semantic = semantic or SemanticCache(config=config)
        self.plan = plan or PlanCache(config=config)
        self.prefix = PrefixCacheHook()
        self._total_requests = 0
        self._total_hits = 0

    async def initialize(self) -> None:
        """Set up embedders for semantic and plan caches."""
        from helix.models.embedder import OpenAIEmbedder

        embedder = OpenAIEmbedder()
        await self.semantic.initialize(embedder=embedder)
        await self.plan.initialize(embedder=embedder)

    async def lookup(self, query: str, context_hash: str) -> CacheHit | None:
        """
        Check semantic cache first.
        If miss, plan cache match is handled separately by the agent
        (plan templates guide execution, not replace it).
        """
        self._total_requests += 1
        hit = await self.semantic.get(query=query, context_hash=context_hash)
        if hit:
            self._total_hits += 1
            return hit
        return None

    def stats(self) -> dict[str, Any]:
        total = self._total_requests
        return {
            "total_requests": total,
            "total_hits": self._total_hits,
            "overall_hit_rate": self._total_hits / total if total > 0 else 0.0,
            "semantic": self.semantic.stats(),
            "plan": self.plan.stats(),
        }
