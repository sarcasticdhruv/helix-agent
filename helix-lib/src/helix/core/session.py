"""
helix/core/session.py

Session — persists multi-turn conversation state across agent.run() calls.

Without sessions, every agent.run() is stateless.
With sessions, context, memory, and history persist across calls
so agents behave coherently across a conversation.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from helix.config import SessionConfig
from helix.errors import SessionExpiredError, SessionNotFoundError
from helix.interfaces import SessionStore


class InMemorySessionStore(SessionStore):
    """Default session store for single-process deployments."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def save(self, session_id: str, state: Dict[str, Any]) -> None:
        async with self._lock:
            self._sessions[session_id] = state

    async def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def list_sessions(self, agent_id: Optional[str] = None) -> List[str]:
        async with self._lock:
            sessions = list(self._sessions.keys())
            if agent_id:
                sessions = [
                    sid for sid in sessions
                    if self._sessions[sid].get("agent_id") == agent_id
                ]
            return sessions

    async def health(self) -> bool:
        return True


class Session:
    """
    A persistent multi-turn conversation session.

    Usage::

        session = Session(agent=my_agent)
        await session.start()

        result1 = await session.send("What is Python?")
        result2 = await session.send("Give me an example of decorators")
        # Agent remembers the first message in the second call

        await session.end()
    """

    def __init__(
        self,
        agent: Any,
        config: Optional[SessionConfig] = None,
        store: Optional[SessionStore] = None,
    ) -> None:
        self._agent = agent
        self._config = config or SessionConfig(agent_id=agent.agent_id)
        self._store = store or InMemorySessionStore()
        self._session_id = self._config.session_id
        self._created_at = time.time()
        self._last_active = time.time()
        self._turn_count = 0
        self._history: List[Dict[str, str]] = []  # [{role, content}]
        self._active = False

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def turn_count(self) -> int:
        return self._turn_count

    async def start(self) -> None:
        """Initialize or restore the session."""
        existing = await self._store.load(self._session_id)
        if existing:
            self._history = existing.get("history", [])
            self._turn_count = existing.get("turn_count", 0)
            self._created_at = existing.get("created_at", time.time())
        self._active = True
        self._last_active = time.time()
        await self._persist()

    async def send(self, message: str) -> Any:
        """
        Send a message in this session and get a response.
        The agent retains context from previous turns.
        """
        if not self._active:
            raise RuntimeError("Session not started. Call session.start() first.")

        self._check_expiry()
        self._last_active = time.time()
        self._turn_count += 1

        # Build context-aware task with history
        contextualized_task = self._build_task(message)

        result = await self._agent.run(
            task=contextualized_task,
            session_id=self._session_id,
        )

        # Record turn in history
        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": str(result.output)})

        # Keep history bounded (last 20 turns = 40 messages)
        if len(self._history) > 40:
            self._history = self._history[-40:]

        await self._persist()
        return result

    async def end(self) -> None:
        """Close the session and clean up stored state."""
        self._active = False
        await self._store.delete(self._session_id)

    def _build_task(self, message: str) -> str:
        """Prepend recent history to the task for context continuity."""
        if not self._history:
            return message

        history_lines = []
        for turn in self._history[-10:]:  # Last 5 exchanges
            role = turn["role"].capitalize()
            history_lines.append(f"{role}: {turn['content'][:300]}")

        history_str = "\n".join(history_lines)
        return (
            f"[Conversation history]\n{history_str}\n\n"
            f"[Current message]\n{message}"
        )

    def _check_expiry(self) -> None:
        now = time.time()
        idle = now - self._last_active
        total = now - self._created_at
        if idle > self._config.idle_timeout_s:
            raise SessionExpiredError(self._session_id)
        if total > self._config.max_duration_s:
            raise SessionExpiredError(self._session_id)

    async def _persist(self) -> None:
        state = {
            "session_id": self._session_id,
            "agent_id": self._config.agent_id,
            "history": self._history,
            "turn_count": self._turn_count,
            "created_at": self._created_at,
            "last_active": self._last_active,
        }
        await self._store.save(self._session_id, state)
