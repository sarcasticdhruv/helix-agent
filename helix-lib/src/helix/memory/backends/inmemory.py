"""
helix/memory/backends/inmemory.py

In-memory MemoryBackend implementation.

Used for: local development, testing, single-process deployments.
Not for: multi-process, multi-replica, or persistent storage needs.
"""

from __future__ import annotations

import asyncio
import math
from typing import Dict, List, Optional

from helix.config import Episode, MemoryEntry
from helix.interfaces import MemoryBackend


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class InMemoryBackend(MemoryBackend):
    """
    In-process vector store using cosine similarity over Python lists.
    Zero dependencies. Suitable for dev and test.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, MemoryEntry] = {}
        self._episodes: Dict[str, Episode] = {}
        self._shared: Dict[str, MemoryEntry] = {}  # key → entry (for team memory)
        self._lock = asyncio.Lock()

    async def upsert(self, entry: MemoryEntry) -> None:
        async with self._lock:
            self._entries[entry.id] = entry

    async def delete(self, entry_id: str) -> None:
        async with self._lock:
            self._entries.pop(entry_id, None)

    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        kind_filter: Optional[str] = None,
    ) -> List[MemoryEntry]:
        async with self._lock:
            candidates = list(self._entries.values())

        if kind_filter:
            candidates = [e for e in candidates if e.kind.value == kind_filter]

        if not query_embedding:
            return candidates[:top_k]

        scored = [
            (e, _cosine_similarity(query_embedding, e.embedding or []))
            for e in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:top_k]]

    async def upsert_episode(self, episode: Episode) -> None:
        async with self._lock:
            self._episodes[episode.id] = episode

    async def search_episodes(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        outcome_filter: Optional[str] = None,
    ) -> List[Episode]:
        async with self._lock:
            candidates = list(self._episodes.values())

        if outcome_filter:
            candidates = [e for e in candidates if e.outcome.value == outcome_filter]

        if not query_embedding:
            return candidates[:top_k]

        scored = [
            (e, _cosine_similarity(query_embedding, e.task_embedding or []))
            for e in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:top_k]]

    async def compare_and_swap(
        self,
        key: str,
        expected_version: int,
        new_entry: MemoryEntry,
    ) -> bool:
        async with self._lock:
            current = self._shared.get(key)
            current_version = current.version if current else 0
            if current_version != expected_version:
                return False
            new_entry.version = current_version + 1
            self._shared[key] = new_entry
            return True

    async def health(self) -> bool:
        return True
