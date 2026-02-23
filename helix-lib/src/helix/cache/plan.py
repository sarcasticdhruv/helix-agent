"""
helix/cache/plan.py

Agentic Plan Cache (APC) — caches the structure of successful agent runs
and adapts them for similar new tasks.

Based on research showing ~50% cost reduction after initial runs.
Stores WHAT tools were called in WHAT order, not the specific content.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

from helix.config import CacheConfig, PlanTemplate
from helix.interfaces import EmbeddingProvider


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    import math
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _extract_keywords(text: str) -> list[str]:
    """Simple keyword extraction — no NLTK dependency."""
    import re
    stop = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "with", "by", "from", "is", "are", "was",
        "what", "how", "why", "when", "where", "who", "which",
        "i", "you", "we", "they", "it", "this", "that", "my",
    }
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return list({w for w in words if w not in stop})[:20]


class PlanStore:
    """File-backed plan template store. No external dependencies."""

    def __init__(self, store_path: Path) -> None:
        self._path = store_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._templates: dict[str, PlanTemplate] = self._load()
        self._lock = asyncio.Lock()

    def _load(self) -> dict[str, PlanTemplate]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text())
            return {k: PlanTemplate(**v) for k, v in data.items()}
        except Exception:
            return {}

    def _save(self) -> None:
        try:
            self._path.write_text(
                json.dumps(
                    {k: v.model_dump() for k, v in self._templates.items()},
                    default=str,
                )
            )
        except Exception:
            pass

    async def upsert(self, template: PlanTemplate) -> None:
        async with self._lock:
            existing = self._templates.get(template.id)
            if existing:
                # Update success rate
                total = existing.run_count + 1
                new_rate = (existing.success_rate * existing.run_count + 1.0) / total
                updated = template.model_copy(update={
                    "run_count": total,
                    "success_rate": new_rate,
                    "avg_cost_usd": (
                        (existing.avg_cost_usd * existing.run_count + template.avg_cost_usd) / total
                    ),
                })
                self._templates[template.id] = updated
            else:
                self._templates[template.id] = template
            self._save()

    async def all(self) -> list[PlanTemplate]:
        async with self._lock:
            return list(self._templates.values())


class PlanCache:
    """
    Caches successful agent plan structures for future reuse.

    When a new task is similar to a past successful task:
      1. Retrieve the cached plan template
      2. Adapt it to the new task using a cheap model
      3. Use the adapted plan to guide the agent

    This skips the expensive initial planning phase.
    """

    def __init__(
        self,
        config: CacheConfig,
        embedder: Optional[EmbeddingProvider] = None,
        store_path: str = ".helix/plan_cache.json",
    ) -> None:
        self._config = config
        self._embedder = embedder
        self._store = PlanStore(Path(store_path))
        self._hits = 0
        self._total_saved_usd = 0.0

    async def initialize(self, embedder: Optional[EmbeddingProvider] = None) -> None:
        if embedder:
            self._embedder = embedder
        if self._embedder is None:
            from helix.models.embedder import OpenAIEmbedder
            self._embedder = OpenAIEmbedder()

    async def match(self, task: str) -> Optional[PlanTemplate]:
        """
        Return the best matching plan template if similarity >= threshold.
        Returns None if no match or plan cache is disabled.
        """
        if not self._config.plan_cache_enabled:
            return None
        if self._embedder is None:
            return None

        try:
            embedding = await self._embedder.embed_one(task)
            keywords = _extract_keywords(task)
            templates = await self._store.all()

            best: Optional[PlanTemplate] = None
            best_score = 0.0

            for template in templates:
                # Keyword overlap score
                template_kw = set(template.keywords)
                task_kw = set(keywords)
                kw_overlap = len(template_kw & task_kw) / max(len(task_kw), 1)

                # Embedding similarity (if template has embedding)
                emb_score = 0.0
                if template.task_embedding:
                    emb_score = _cosine_similarity(embedding, template.task_embedding)

                # Combined score: weight embedding more, use keywords as tiebreaker
                combined = (emb_score * 0.7) + (kw_overlap * 0.3)

                if combined > best_score:
                    best_score = combined
                    best = template

            if best and best_score >= self._config.plan_match_threshold:
                self._hits += 1
                self._total_saved_usd += best.avg_cost_usd * 0.5  # Estimate 50% savings
                return best.model_copy(update={"score": best_score})

            return None
        except Exception:
            return None

    async def store(self, task: str, ctx: Any) -> None:
        """
        Store a successful run's plan structure.
        ctx is an ExecutionContext.
        """
        if not self._config.plan_cache_enabled:
            return
        try:
            tool_sequence = [tc.tool_name for tc in ctx.tool_calls]
            steps_desc = self._describe_steps(ctx)

            embedding: list[float] = []
            if self._embedder:
                embedding = await self._embedder.embed_one(task)

            template = PlanTemplate(
                task_description=task[:200],
                task_embedding=embedding if embedding else None,
                keywords=_extract_keywords(task),
                steps_description=steps_desc,
                tool_sequence=tool_sequence,
                avg_cost_usd=ctx.cost.spent_usd,
            )
            await self._store.upsert(template)
        except Exception:
            pass

    def _describe_steps(self, ctx: Any) -> str:
        """Convert tool call sequence into a reusable plan description."""
        if not ctx.tool_calls:
            return "Direct response without tool calls."
        lines = ["Plan steps:"]
        for i, tc in enumerate(ctx.tool_calls, 1):
            lines.append(f"  {i}. Call {tc.tool_name} to gather information")
        return "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        return {
            "plan_hits": self._hits,
            "estimated_saved_usd": round(self._total_saved_usd, 6),
        }
