"""
helix/memory/backends/sqlite.py

SQLite-backed MemoryBackend — persists across process restarts using the
stdlib `sqlite3` module (no extra dependency). Search is brute-force
cosine similarity over embeddings loaded from disk, same strategy as
InMemoryBackend, just durable.

Not a scalable vector index — for large corpora, use a real vector
database backend once one is implemented. This exists so the default
"local, zero-dependency" path actually survives a restart.
"""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
from pathlib import Path

from helix.config import Episode, EpisodeOutcome, MemoryEntry, MemoryKind
from helix.interfaces import MemoryBackend

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    kind TEXT NOT NULL,
    importance REAL NOT NULL,
    embedding TEXT,
    metadata TEXT NOT NULL,
    created_at REAL NOT NULL,
    agent_id TEXT,
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    task TEXT NOT NULL,
    task_embedding TEXT,
    outcome TEXT NOT NULL,
    summary TEXT NOT NULL,
    steps INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    tools_used TEXT NOT NULL,
    failure_reason TEXT,
    learned_strategy TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS shared (
    key TEXT PRIMARY KEY,
    entry TEXT NOT NULL,
    version INTEGER NOT NULL
);
"""


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class SQLiteBackend(MemoryBackend):
    """Persistent MemoryBackend backed by a local SQLite file."""

    def __init__(self, db_path: str = ".helix/memory.db") -> None:
        self._db_path = db_path
        self._lock = asyncio.Lock()
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        return self._conn

    async def upsert(self, entry: MemoryEntry) -> None:
        async with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO entries "
                "(id, content, kind, importance, embedding, metadata, created_at, agent_id, version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.id,
                    entry.content,
                    entry.kind.value,
                    entry.importance,
                    json.dumps(entry.embedding) if entry.embedding else None,
                    json.dumps(entry.metadata, default=str),
                    entry.created_at,
                    entry.agent_id,
                    entry.version,
                ),
            )
            conn.commit()

    async def delete(self, entry_id: str) -> None:
        async with self._lock:
            conn = self._get_conn()
            conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
            conn.commit()

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        kind_filter: str | None = None,
    ) -> list[MemoryEntry]:
        async with self._lock:
            conn = self._get_conn()
            if kind_filter:
                rows = conn.execute(
                    "SELECT id, content, kind, importance, embedding, metadata, created_at, "
                    "agent_id, version FROM entries WHERE kind = ?",
                    (kind_filter,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT id, content, kind, importance, embedding, metadata, created_at, "
                    "agent_id, version FROM entries"
                ).fetchall()

        entries = [self._row_to_entry(r) for r in rows]
        if not query_embedding:
            return entries[:top_k]
        scored = [(e, _cosine_similarity(query_embedding, e.embedding or [])) for e in entries]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:top_k]]

    async def upsert_episode(self, episode: Episode) -> None:
        async with self._lock:
            conn = self._get_conn()
            conn.execute(
                "INSERT OR REPLACE INTO episodes "
                "(id, agent_id, task, task_embedding, outcome, summary, steps, cost_usd, "
                "tools_used, failure_reason, learned_strategy, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    episode.id,
                    episode.agent_id,
                    episode.task,
                    json.dumps(episode.task_embedding) if episode.task_embedding else None,
                    episode.outcome.value,
                    episode.summary,
                    episode.steps,
                    episode.cost_usd,
                    json.dumps(episode.tools_used),
                    episode.failure_reason,
                    episode.learned_strategy,
                    episode.created_at,
                ),
            )
            conn.commit()

    async def search_episodes(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        outcome_filter: str | None = None,
    ) -> list[Episode]:
        async with self._lock:
            conn = self._get_conn()
            cols = (
                "id, agent_id, task, task_embedding, outcome, summary, steps, "
                "cost_usd, tools_used, failure_reason, learned_strategy, created_at"
            )
            if outcome_filter:
                rows = conn.execute(
                    f"SELECT {cols} FROM episodes WHERE outcome = ?", (outcome_filter,)
                ).fetchall()
            else:
                rows = conn.execute(f"SELECT {cols} FROM episodes").fetchall()

        episodes = [self._row_to_episode(r) for r in rows]
        if not query_embedding:
            return episodes[:top_k]
        scored = [
            (e, _cosine_similarity(query_embedding, e.task_embedding or [])) for e in episodes
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
            conn = self._get_conn()
            row = conn.execute("SELECT version FROM shared WHERE key = ?", (key,)).fetchone()
            current_version = row[0] if row else 0
            if current_version != expected_version:
                return False
            new_entry.version = current_version + 1
            conn.execute(
                "INSERT OR REPLACE INTO shared (key, entry, version) VALUES (?, ?, ?)",
                (key, new_entry.model_dump_json(), new_entry.version),
            )
            conn.commit()
            return True

    async def health(self) -> bool:
        try:
            conn = self._get_conn()
            conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def _row_to_entry(self, row: tuple) -> MemoryEntry:
        (id_, content, kind, importance, embedding, metadata, created_at, agent_id, version) = row
        return MemoryEntry(
            id=id_,
            content=content,
            kind=MemoryKind(kind),
            importance=importance,
            embedding=json.loads(embedding) if embedding else None,
            metadata=json.loads(metadata),
            created_at=created_at,
            agent_id=agent_id,
            version=version,
        )

    def _row_to_episode(self, row: tuple) -> Episode:
        (
            id_,
            agent_id,
            task,
            task_embedding,
            outcome,
            summary,
            steps,
            cost_usd,
            tools_used,
            failure_reason,
            learned_strategy,
            created_at,
        ) = row
        return Episode(
            id=id_,
            agent_id=agent_id,
            task=task,
            task_embedding=json.loads(task_embedding) if task_embedding else None,
            outcome=EpisodeOutcome(outcome),
            summary=summary,
            steps=steps,
            cost_usd=cost_usd,
            tools_used=json.loads(tools_used),
            failure_reason=failure_reason,
            learned_strategy=learned_strategy,
            created_at=created_at,
        )
