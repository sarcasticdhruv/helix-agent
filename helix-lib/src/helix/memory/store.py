"""
helix/memory/store.py

MemoryStore — the agent's memory system.

Design:
  - Short-term: in-process rolling buffer (fast, no I/O)
  - Long-term: vector store backend (persistent, searchable)
  - Promotion: WAL-backed to guarantee at-least-once delivery
  - Episodic: records completed runs; consulted at next run start
  - Shared team memory: optimistic locking via backend CAS
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from helix.config import (
    Episode,
    EpisodeOutcome,
    MemoryConfig,
    MemoryEntry,
    MemoryKind,
)
from helix.errors import MemoryBackendError, MemoryConflictError
from helix.interfaces import MemoryBackend


class ShortTermBuffer:
    """
    In-process rolling buffer for recent memory entries.
    Thread-safe via asyncio.Lock.
    """

    def __init__(self, limit: int) -> None:
        self._entries: List[MemoryEntry] = []
        self._limit = limit
        self._lock = asyncio.Lock()

    async def add(self, entry: MemoryEntry) -> None:
        async with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._limit:
                # Evict least important non-pinned entries
                self._entries.sort(key=lambda e: e.importance, reverse=True)
                self._entries = self._entries[: self._limit]

    async def recent(self, n: int) -> List[MemoryEntry]:
        async with self._lock:
            return list(reversed(self._entries[-n:]))

    async def all(self) -> List[MemoryEntry]:
        async with self._lock:
            return list(self._entries)

    def recent_str(self, n: int) -> str:
        """Synchronous string representation for system prompt injection."""
        entries = self._entries[-n:]
        if not entries:
            return ""
        lines = []
        for e in reversed(entries):
            lines.append(f"[{e.kind.value}] {e.content[:200]}")
        return "\n".join(lines)


class WriteAheadLog:
    """
    File-backed WAL for memory promotion.
    Ensures entries survive process restart and are retried on next startup.
    """

    def __init__(self, wal_path: Path) -> None:
        self._path = wal_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._pending: List[MemoryEntry] = self._load()
        self._lock = asyncio.Lock()

    def _load(self) -> List[MemoryEntry]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text())
            return [MemoryEntry(**e) for e in data]
        except Exception:
            return []

    def _save(self) -> None:
        try:
            self._path.write_text(
                json.dumps([e.model_dump() for e in self._pending], default=str)
            )
        except Exception:
            pass  # WAL save failure is not fatal; entry stays in-memory pending

    async def enqueue(self, entry: MemoryEntry) -> None:
        async with self._lock:
            self._pending.append(entry)
            self._save()

    async def dequeue(self, entry_id: str) -> None:
        async with self._lock:
            self._pending = [e for e in self._pending if e.id != entry_id]
            self._save()

    async def pending(self) -> List[MemoryEntry]:
        async with self._lock:
            return list(self._pending)


class MemoryStore:
    """
    Public interface to the Helix memory system.

    Usage::

        store = MemoryStore(config=MemoryConfig())
        await store.initialize()

        await store.add(MemoryEntry(content="User prefers bullet lists"))
        similar = await store.recall("formatting preferences")
    """

    def __init__(
        self,
        config: MemoryConfig,
        backend: Optional[MemoryBackend] = None,
        wal_dir: str = ".helix/wal",
    ) -> None:
        self._config = config
        self._buffer = ShortTermBuffer(limit=config.short_term_limit)
        self._backend: Optional[MemoryBackend] = backend
        self._wal = WriteAheadLog(Path(wal_dir) / "memory_wal.json")
        self._embedder: Optional[Any] = None

    async def initialize(self) -> None:
        """Set up backend and embedder. Flush any pending WAL entries."""
        if self._backend is None:
            self._backend = self._create_backend()

        self._embedder = self._create_embedder()

        # Retry any promotions that failed in a previous run
        await self._flush_wal()

    def _create_backend(self) -> MemoryBackend:
        backend_name = self._config.backend
        if backend_name == "inmemory":
            from helix.memory.backends.inmemory import InMemoryBackend
            return InMemoryBackend()
        if backend_name == "qdrant":
            from helix.memory.backends.qdrant import QdrantBackend
            return QdrantBackend()
        if backend_name == "pinecone":
            from helix.memory.backends.pinecone import PineconeBackend
            return PineconeBackend()
        if backend_name == "chroma":
            from helix.memory.backends.chroma import ChromaBackend
            return ChromaBackend()
        raise ValueError(f"Unknown memory backend: {backend_name}")

    def _create_embedder(self) -> Any:
        """Return a lightweight embedder. Defaults to OpenAI embeddings."""
        from helix.models.embedder import OpenAIEmbedder
        return OpenAIEmbedder(model=self._config.embedding_model)

    # ------------------------------------------------------------------
    # Short-term operations
    # ------------------------------------------------------------------

    async def add(self, entry: MemoryEntry) -> MemoryEntry:
        """
        Add to short-term buffer.
        If importance >= threshold and auto-promote is on,
        also promote to long-term via WAL.
        """
        await self._buffer.add(entry)
        if (
            self._config.auto_promote
            and entry.importance >= self._config.importance_threshold
        ):
            await self._promote(entry)
        return entry

    async def recent(self, n: int = 5) -> List[MemoryEntry]:
        return await self._buffer.recent(n)

    def recent_str(self, n: int = 5) -> str:
        return self._buffer.recent_str(n)

    # ------------------------------------------------------------------
    # Long-term operations (vector store)
    # ------------------------------------------------------------------

    async def recall(
        self,
        query: str,
        top_k: int = 5,
        kind: Optional[MemoryKind] = None,
    ) -> List[MemoryEntry]:
        """Semantic search over long-term memory."""
        if self._backend is None or self._embedder is None:
            return []
        try:
            embedding = await self._embedder.embed_one(query)
            return await self._backend.search(
                query_embedding=embedding,
                top_k=top_k,
                kind_filter=kind.value if kind else None,
            )
        except Exception as e:
            raise MemoryBackendError(
                backend=self._config.backend,
                operation="search",
                reason=str(e),
            ) from e

    async def forget(self, entry_id: str) -> None:
        if self._backend:
            await self._backend.delete(entry_id)

    async def embed(self, text: str) -> List[float]:
        """Expose embedder for external use (e.g. episode similarity)."""
        if self._embedder is None:
            return []
        return await self._embedder.embed_one(text)

    # ------------------------------------------------------------------
    # Episodic memory
    # ------------------------------------------------------------------

    async def record_episode(self, episode: Episode) -> None:
        """Record a completed agent run for future episodic recall."""
        if self._backend is None or self._embedder is None:
            return
        try:
            if not episode.task_embedding:
                embedding = await self._embedder.embed_one(episode.task)
                episode = episode.model_copy(update={"task_embedding": embedding})
            await self._backend.upsert_episode(episode)
        except Exception:
            pass  # Episodic recording failure is non-fatal

    async def recall_similar_episodes(
        self,
        task: str,
        top_k: int = 3,
        outcome: Optional[EpisodeOutcome] = None,
    ) -> List[Episode]:
        """Return episodes similar to the given task."""
        if self._backend is None or self._embedder is None:
            return []
        try:
            embedding = await self._embedder.embed_one(task)
            return await self._backend.search_episodes(
                query_embedding=embedding,
                top_k=top_k,
                outcome_filter=outcome.value if outcome else None,
            )
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Shared team memory
    # ------------------------------------------------------------------

    async def share(self, content: str, key: str, agent_id: str) -> None:
        """Write to shared team memory with optimistic locking."""
        if self._backend is None:
            return
        entry = MemoryEntry(
            content=content,
            kind=MemoryKind.FACT,
            agent_id=agent_id,
            metadata={"shared": True, "key": key},
        )
        success = await self._backend.compare_and_swap(
            key=key,
            expected_version=0,
            new_entry=entry,
        )
        if not success:
            raise MemoryConflictError(key=key, agent_id=agent_id)

    async def recall_shared(self, query: str, top_k: int = 3) -> List[MemoryEntry]:
        """Search shared team memory."""
        return await self.recall(query=query, top_k=top_k)

    # ------------------------------------------------------------------
    # WAL-backed promotion
    # ------------------------------------------------------------------

    async def _promote(self, entry: MemoryEntry) -> None:
        """
        Promote entry to long-term storage via WAL.
        WAL ensures promotion survives process restarts.
        """
        await self._wal.enqueue(entry)
        asyncio.create_task(self._do_promote(entry))

    async def _do_promote(self, entry: MemoryEntry) -> None:
        """Actual promotion attempt. Retried on failure via WAL flush."""
        if self._backend is None or self._embedder is None:
            return
        try:
            if not entry.embedding:
                embedding = await self._embedder.embed_one(entry.content)
                entry.embedding = embedding
            await self._backend.upsert(entry)
            await self._wal.dequeue(entry.id)
        except Exception:
            pass  # WAL preserves it; will retry on next startup

    async def _flush_wal(self) -> None:
        """Called on initialize() to retry any failed promotions."""
        pending = await self._wal.pending()
        for entry in pending:
            await self._do_promote(entry)
