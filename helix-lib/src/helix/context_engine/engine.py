"""
helix/context_engine/engine.py

ContextEngine — manages the full context lifecycle for an agent run.

Responsibilities:
  - Multi-factor relevance decay (time, reference, semantic, role)
  - Reference tracking (which prior messages did the LLM cite?)
  - Compaction trigger and orchestration
  - Pre-flight estimation
"""

from __future__ import annotations

import math
from typing import Any

from helix.config import AgentConfig, ContextMessage, ContextMessageRole
from helix.context import ExecutionContext
from helix.context_engine.compactor import IntelligentCompactor
from helix.context_engine.preflight import PreflightEstimate, PreflightEstimator

# Default decay weights (configurable per-agent in future)
_ALPHA = 0.3  # Time decay weight
_BETA = 0.4  # Reference score weight
_GAMMA = 0.2  # Semantic similarity weight
_DELTA = 0.1  # Role weight

# Role base weights
_ROLE_WEIGHTS = {
    ContextMessageRole.SYSTEM: 1.0,
    ContextMessageRole.TOOL: 0.8,
    ContextMessageRole.USER: 0.7,
    ContextMessageRole.ASSISTANT: 0.5,
}

# Lambda for exponential decay (larger = faster decay)
_LAMBDA_BY_ROLE = {
    ContextMessageRole.SYSTEM: 0.01,  # Slow decay — system context stays relevant
    ContextMessageRole.TOOL: 0.05,  # Medium — tool results decay faster
    ContextMessageRole.USER: 0.03,
    ContextMessageRole.ASSISTANT: 0.08,  # Fast — intermediate reasoning decays quickest
}


def _time_decay(step_added: int, current_step: int, role: ContextMessageRole) -> float:
    """Exponential decay based on steps elapsed since message was added."""
    steps_elapsed = max(0, current_step - step_added)
    lam = _LAMBDA_BY_ROLE.get(role, 0.05)
    return math.exp(-lam * steps_elapsed)


def _role_weight(role: ContextMessageRole) -> float:
    return _ROLE_WEIGHTS.get(role, 0.5)


def _compute_relevance(
    message: ContextMessage,
    current_step: int,
    semantic_score: float = 0.0,
) -> float:
    """
    Multi-factor relevance score.

    relevance = α × time_decay + β × reference_score + γ × semantic + δ × role_weight
    """
    td = _time_decay(message.step_added, current_step, message.role)
    ref = message.reference_score
    sem = semantic_score
    role = _role_weight(message.role)

    return _ALPHA * td + _BETA * ref + _GAMMA * sem + _DELTA * role


class ContextEngine:
    """
    Manages the context window for an agent run.

    Called by Agent at:
      - Run start: build_initial_context()
      - Each step: update_relevance()
      - When approaching limit: compact()
    """

    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._compactor = IntelligentCompactor(
            cheap_model="gpt-4o-mini",
            relevance_threshold=0.4,
        )
        self._preflight = PreflightEstimator()

    async def update_relevance(
        self,
        ctx: ExecutionContext,
        last_response: str,
        embedder: Any | None = None,
    ) -> None:
        """
        Update relevance scores for all messages in the context window.

        1. Apply multi-factor decay to each message
        2. Boost reference scores for messages cited in last_response
        3. Optional: compute semantic similarity to current task
        """
        messages = ctx.window.messages()
        current_step = ctx.window.step

        # Reference detection: find which messages are referenced in the last response
        referenced_ids = self._detect_references(messages, last_response)

        for msg in messages:
            if msg.pinned:
                msg.relevance = 1.0
                continue

            # Boost reference score if this message was cited
            if msg.id in referenced_ids:
                msg.reference_score = 1.0

            # Compute new relevance (semantic score = 0 unless embedder provided)
            semantic_score = 0.0
            msg.relevance = _compute_relevance(
                message=msg,
                current_step=current_step,
                semantic_score=semantic_score,
            )

    def _detect_references(
        self,
        messages: list[ContextMessage],
        response: str,
    ) -> set:
        """
        Detect which prior messages are implicitly referenced in the response.
        Simple heuristic: check if significant noun phrases from the message
        appear in the response.
        """
        referenced = set()
        response_lower = response.lower()

        for msg in messages:
            if msg.role == ContextMessageRole.SYSTEM:
                continue
            # Extract meaningful words (length > 5, not common words)
            words = [w for w in msg.content.lower().split() if len(w) > 5 and w.isalpha()]
            # If 2+ significant words appear in the response, consider it referenced
            hits = sum(1 for w in words[:20] if w in response_lower)
            if hits >= 2:
                referenced.add(msg.id)

        return referenced

    async def compact(
        self,
        ctx: ExecutionContext,
        embedder: Any | None = None,
        llm_router: Any | None = None,
    ) -> None:
        """
        Compact the context window when approaching the token limit.
        Replaces low-relevance message groups with summaries.
        """
        messages = ctx.window.messages()
        compacted = await self._compactor.compact(
            messages=messages,
            embedder=embedder,
            llm_router=llm_router,
        )
        # Rebuild the window with compacted messages
        ctx.window._messages = compacted  # type: ignore[attr-defined]

    def preflight_estimate(
        self,
        task: str,
        system_prompt: str = "",
        similar_episodes: list | None = None,
    ) -> PreflightEstimate:
        """Estimate cost before the run starts."""
        return self._preflight.estimate(
            task=task,
            config=self._config,
            system_prompt=system_prompt,
            similar_episodes=similar_episodes,
        )
