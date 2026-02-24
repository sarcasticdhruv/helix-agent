"""
helix/context_engine/compactor.py

IntelligentCompactor — summarizes groups of low-relevance messages
before evicting them from the context window.

Does not simply delete — groups related messages by topic embedding
and replaces each group with a single summary message, preserving
information while freeing token budget.
"""

from __future__ import annotations

import math
from typing import Any

from helix.config import ContextMessage, ContextMessageRole


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _cluster_by_similarity(
    messages: list[ContextMessage],
    embeddings: list[list[float]],
    threshold: float = 0.75,
) -> list[list[int]]:
    """
    Simple greedy clustering: each message joins the first cluster
    whose centroid is within threshold similarity.
    Returns list of index groups.
    """
    clusters: list[list[int]] = []
    centroids: list[list[float]] = []

    for i, emb in enumerate(embeddings):
        best_cluster = -1
        best_score = threshold

        for j, centroid in enumerate(centroids):
            score = _cosine_similarity(emb, centroid)
            if score > best_score:
                best_score = score
                best_cluster = j

        if best_cluster == -1:
            clusters.append([i])
            centroids.append(emb)
        else:
            clusters[best_cluster].append(i)
            # Update centroid as mean
            cluster_embs = [embeddings[k] for k in clusters[best_cluster]]
            n = len(cluster_embs)
            centroids[best_cluster] = [
                sum(cluster_embs[k][d] for k in range(n)) / n for d in range(len(emb))
            ]

    return clusters


class IntelligentCompactor:
    """
    Compacts a list of context messages into a shorter list
    by summarizing low-relevance groups.

    The compaction uses a cheap model (not the primary agent model)
    to produce summaries, keeping cost minimal.
    """

    def __init__(
        self,
        cheap_model: str = "gpt-4o-mini",
        relevance_threshold: float = 0.4,
    ) -> None:
        self._cheap_model = cheap_model
        self._relevance_threshold = relevance_threshold

    async def compact(
        self,
        messages: list[ContextMessage],
        embedder: Any | None = None,
        llm_router: Any | None = None,
    ) -> list[ContextMessage]:
        """
        Compact the message list.

        Steps:
          1. Separate pinned and compressible messages
          2. Cluster compressible messages by topic
          3. Summarize each cluster into a single message
          4. Return: pinned + summaries + recent messages
        """
        pinned = [m for m in messages if m.pinned]
        compressible = [
            m for m in messages if not m.pinned and m.relevance < self._relevance_threshold
        ]
        recent = [m for m in messages if not m.pinned and m.relevance >= self._relevance_threshold]

        if len(compressible) <= 2:
            # Not enough to compact
            return messages

        # Get embeddings for clustering
        summaries: list[ContextMessage] = []
        if embedder and len(compressible) > 1:
            try:
                texts = [m.content[:500] for m in compressible]
                embeddings = await embedder.embed_batch(texts)
                clusters = _cluster_by_similarity(compressible, embeddings)
                for cluster_indices in clusters:
                    group = [compressible[i] for i in cluster_indices]
                    summary = await self._summarize_group(group, llm_router)
                    summaries.append(summary)
            except Exception:
                # Fallback: simple truncation without embedding
                summaries = self._truncate_fallback(compressible)
        else:
            summaries = self._truncate_fallback(compressible)

        return pinned + summaries + recent

    async def _summarize_group(
        self,
        group: list[ContextMessage],
        llm_router: Any | None,
    ) -> ContextMessage:
        """Summarize a group of messages into one summary message."""
        if len(group) == 1:
            return group[0]

        combined = "\n".join(f"[{m.role.value}]: {m.content[:300]}" for m in group)
        summary_text = f"[SUMMARY of {len(group)} messages] {combined[:200]}..."

        if llm_router:
            try:
                response = await llm_router.complete(
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Summarize these agent conversation messages in 1-2 sentences, "
                                "preserving key facts and decisions:\n\n" + combined
                            ),
                        }
                    ],
                    model=self._cheap_model,
                    max_tokens=200,
                    temperature=0.3,
                )
                summary_text = f"[CONTEXT SUMMARY] {response.content}"
            except Exception:
                pass

        return ContextMessage(
            role=ContextMessageRole.SYSTEM,
            content=summary_text,
            pinned=False,
            relevance=0.6,
        )

    def _truncate_fallback(self, messages: list[ContextMessage]) -> list[ContextMessage]:
        """When embedding/LLM unavailable: merge into a single summary."""
        if not messages:
            return []
        content = "; ".join(m.content[:100] for m in messages[:5])
        return [
            ContextMessage(
                role=ContextMessageRole.SYSTEM,
                content=f"[COMPACTED: {len(messages)} messages] {content}",
                pinned=False,
                relevance=0.5,
            )
        ]
