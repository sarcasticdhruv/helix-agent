"""
helix/cache/semantic.py

SemanticCache — caches complete LLM responses keyed by
embedding similarity of the query + context hash.

Research basis: 40-70% cost reduction on repeated similar queries,
response time from ~850ms to <120ms on cache hits.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from helix.config import CacheConfig, CacheEntry, CacheHit
from helix.interfaces import CacheBackend, EmbeddingProvider


class InMemoryCacheBackend(CacheBackend):
    """Default in-process cache backend. No external dependencies."""

    def __init__(self) -> None:
        self._store: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, entry: CacheEntry, ttl_seconds: int) -> None:
        async with self._lock:
            entry_with_expiry = entry.model_copy(update={"age_s": 0.0})
            self._store[entry.id] = entry_with_expiry
            self._expiry: dict[str, float] = getattr(self, "_expiry", {})
            self._expiry[entry.id] = time.time() + ttl_seconds

    async def search(
        self,
        query_embedding: list[float],
        threshold: float,
    ) -> CacheEntry | None:
        async with self._lock:
            self._evict_expired()
            best: CacheEntry | None = None
            best_score = -1.0  # Must be < any valid score so 0.0 >= threshold wins
            for entry in self._store.values():
                score = _cosine_similarity(query_embedding, entry.query_embedding)
                if score >= threshold and score > best_score:
                    best_score = score
                    best = entry
            if best:
                age = time.time() - best.created_at
                return best.model_copy(update={"similarity": best_score, "age_s": age})
            return None

    async def delete(self, entry_id: str) -> None:
        async with self._lock:
            self._store.pop(entry_id, None)

    async def flush(self) -> int:
        async with self._lock:
            count = len(self._store)
            self._store.clear()
            self._expiry = {}
            return count

    async def stats(self) -> dict[str, Any]:
        return {"size": len(self._store), "backend": "inmemory"}

    async def health(self) -> bool:
        return True

    def _evict_expired(self) -> None:
        now = time.time()
        expiry = getattr(self, "_expiry", {})
        expired = [k for k, t in expiry.items() if now > t]
        for k in expired:
            self._store.pop(k, None)
            expiry.pop(k, None)


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


class SemanticCache:
    """
    Caches LLM responses for semantically similar queries.

    Two queries with embedding similarity >= threshold and matching
    context_hash return the same cached response without an LLM call.

    The context_hash ensures cached responses aren't returned when
    the agent's context has substantially changed.
    """

    def __init__(
        self,
        config: CacheConfig,
        backend: CacheBackend | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self._config = config
        self._backend = backend or InMemoryCacheBackend()
        self._embedder = embedder
        self._hits = 0
        self._misses = 0
        self._total_saved_usd = 0.0

    async def initialize(self, embedder: EmbeddingProvider | None = None) -> None:
        if embedder:
            self._embedder = embedder
        if self._embedder is None:
            from helix.models.embedder import OpenAIEmbedder

            self._embedder = OpenAIEmbedder()

    async def get(self, query: str, context_hash: str) -> CacheHit | None:
        """
        Look up a cached response.
        Returns CacheHit if a semantically similar query exists with
        the same context_hash, otherwise None.
        """
        if not self._should_cache(query):
            return None
        if self._embedder is None:
            return None

        try:
            embedding = await self._embedder.embed_one(query)
            entry = await self._backend.search(
                query_embedding=embedding,
                threshold=self._config.semantic_threshold,
            )
            if entry and entry.context_hash == context_hash:
                self._hits += 1
                hit = CacheHit(
                    response=entry.response,
                    similarity=entry.similarity,
                    age_s=entry.age_s,
                    tier="semantic",
                    saved_usd=entry.cost_usd,
                )
                self._total_saved_usd += entry.cost_usd
                return hit
            self._misses += 1
            return None
        except Exception:
            self._misses += 1
            return None

    async def set(
        self,
        query: str,
        context_hash: str,
        response: str,
        cost_usd: float,
    ) -> None:
        """Store a response in the semantic cache."""
        if not self._should_cache(query):
            return
        if self._embedder is None:
            return

        try:
            embedding = await self._embedder.embed_one(query)
            entry = CacheEntry(
                query=query,
                query_embedding=embedding,
                context_hash=context_hash,
                response=response,
                cost_usd=cost_usd,
            )
            await self._backend.upsert(entry, ttl_seconds=self._config.ttl_seconds)
        except Exception:
            pass  # Cache write failure is non-fatal

    async def clear(self) -> int:
        return await self._backend.flush()

    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "total_saved_usd": round(self._total_saved_usd, 6),
        }

    def _should_cache(self, query: str) -> bool:
        """Skip cache for queries containing excluded patterns."""
        q_lower = query.lower()
        return all(pattern.lower() not in q_lower for pattern in self._config.exclude_patterns)
