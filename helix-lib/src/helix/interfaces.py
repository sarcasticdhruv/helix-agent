"""
helix/interfaces.py

All abstract base classes and protocols for Helix.
These are the extension contracts. Every backend, provider, guardrail,
and transport must implement one of these interfaces.

Rules:
  - No business logic here. Contracts only.
  - Every interface is async-first.
  - Every interface is fully typed.
  - These are the stable public extension API.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    # Forward references to avoid circular imports.
    # These types are defined in config.py and context.py.
    from helix.config import (
        AuditEntry,
        CacheEntry,
        Episode,
        EvalCase,
        GuardrailResult,
        HITLRequest,
        HITLResponse,
        MemoryEntry,
        ModelResponse,
        ToolCallRecord,
    )
    from helix.context import ExecutionContext


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class MemoryBackend(abc.ABC):
    """
    Persistent storage for agent memory entries and episodes.

    Implementors: InMemoryBackend, PineconeBackend, QdrantBackend, ChromaBackend.

    Contract:
      - upsert / delete are fire-and-forget safe to retry (idempotent).
      - search returns results ordered by relevance descending.
      - All methods are async.
    """

    @abc.abstractmethod
    async def upsert(self, entry: MemoryEntry) -> None:
        """Insert or update a memory entry by its id."""
        ...

    @abc.abstractmethod
    async def delete(self, entry_id: str) -> None:
        """Remove a memory entry permanently."""
        ...

    @abc.abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        kind_filter: str | None = None,
    ) -> list[MemoryEntry]:
        """
        Return top_k entries most similar to query_embedding.
        Optional kind_filter restricts results to a single memory kind.
        """
        ...

    @abc.abstractmethod
    async def upsert_episode(self, episode: Episode) -> None:
        """Store a completed agent run episode."""
        ...

    @abc.abstractmethod
    async def search_episodes(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        outcome_filter: str | None = None,
    ) -> list[Episode]:
        """Return episodes similar to the given embedding."""
        ...

    @abc.abstractmethod
    async def compare_and_swap(
        self,
        key: str,
        expected_version: int,
        new_entry: MemoryEntry,
    ) -> bool:
        """
        Atomic update for shared team memory.
        Returns True if the swap succeeded, False if the version has changed.
        """
        ...

    @abc.abstractmethod
    async def health(self) -> bool:
        """Return True if the backend is reachable and healthy."""
        ...


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class CacheBackend(abc.ABC):
    """
    Storage layer for semantic response cache and plan cache.

    Implementors: InMemoryCacheBackend, RedisCacheBackend.

    Contract:
      - Entries expire after their TTL. Backends must enforce TTL.
      - search uses embedding similarity, not exact key lookup.
    """

    @abc.abstractmethod
    async def upsert(self, entry: CacheEntry, ttl_seconds: int) -> None:
        """Store a cache entry with an expiry TTL."""
        ...

    @abc.abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        threshold: float,
    ) -> CacheEntry | None:
        """Return the best matching entry above threshold, or None."""
        ...

    @abc.abstractmethod
    async def delete(self, entry_id: str) -> None:
        """Evict a specific cache entry."""
        ...

    @abc.abstractmethod
    async def flush(self) -> int:
        """Evict all entries. Returns count of entries removed."""
        ...

    @abc.abstractmethod
    async def stats(self) -> dict[str, Any]:
        """Return backend-level stats: size, hit_rate, memory_mb, etc."""
        ...

    @abc.abstractmethod
    async def health(self) -> bool: ...


# ---------------------------------------------------------------------------
# LLM Provider
# ---------------------------------------------------------------------------


class LLMProvider(abc.ABC):
    """
    Uniform interface for all LLM backends.

    Implementors: OpenAIProvider, AnthropicProvider, LocalProvider.

    Contract:
      - complete() returns a fully resolved response.
      - stream() yields text chunks as they arrive.
      - Both methods accept the same parameters.
      - Providers raise HelixProviderError on all failures —
        never raw SDK exceptions.
    """

    @abc.abstractmethod
    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict[str, Any] | None = None,
    ) -> ModelResponse:
        """Send messages and return a complete response."""
        ...

    @abc.abstractmethod
    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """Stream response text chunks."""
        ...

    @abc.abstractmethod
    def count_tokens(self, messages: list[dict[str, Any]], model: str) -> int:
        """
        Count tokens for the given messages without making an API call.
        Used by the cost governor when provider token counts are absent.
        """
        ...

    @abc.abstractmethod
    def supported_models(self) -> list[str]:
        """List model identifiers this provider supports."""
        ...

    @abc.abstractmethod
    async def health(self) -> bool: ...


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class EmbeddingProvider(abc.ABC):
    """
    Interface for converting text to embedding vectors.

    Used by: MemoryBackend, SemanticCache, PlanCache, ContextEngine.

    Contract:
      - embed_one is a convenience wrapper over embed_batch.
      - Implementors must guarantee consistent dimensionality.
    """

    @abc.abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single string. Default delegates to embed_batch."""
        results = await self.embed_batch([text])
        return results[0]

    @property
    @abc.abstractmethod
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors produced."""
        ...


# ---------------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------------


class Guardrail(abc.ABC):
    """
    A single, composable content filter.

    Guardrails are applied in sequence. Each guardrail either passes
    the content through (possibly modified) or blocks it.

    Implementors: PIIRedactor, LengthGuard, SchemaGuard, ToxicityGuard.

    Contract:
      - check() never raises; it returns a GuardrailResult.
      - If passed=False, the pipeline stops at this guardrail.
      - modified_content may differ from input (e.g. PII redacted).
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable identifier for this guardrail (used in audit logs)."""
        ...

    @abc.abstractmethod
    async def check(
        self,
        content: str,
        context: ExecutionContext,
    ) -> GuardrailResult:
        """
        Evaluate content. Returns a GuardrailResult with:
          passed: bool
          modified_content: Optional[str]  — if content was cleaned
          reason: Optional[str]            — why it failed, if it did
        """
        ...


# ---------------------------------------------------------------------------
# HITL Transport
# ---------------------------------------------------------------------------


class HITLTransport(abc.ABC):
    """
    Delivery mechanism for human-in-the-loop approval requests.

    Implementors: CLITransport, WebhookTransport, QueueTransport (Redis/SQS).

    Contract:
      - send_request() blocks until a decision is received or timeout.
      - On timeout, implementations must return ESCALATE, never raise.
    """

    @abc.abstractmethod
    async def send_request(self, request: HITLRequest) -> HITLResponse:
        """
        Deliver the request and await a human decision.
        Must handle its own timeout and return HITLResponse on expiry.
        """
        ...

    @abc.abstractmethod
    async def health(self) -> bool: ...


# ---------------------------------------------------------------------------
# Audit Log
# ---------------------------------------------------------------------------


class AuditLogBackend(abc.ABC):
    """
    Append-only, tamper-evident audit trail.

    Implementors: LocalFileAuditBackend, S3AuditBackend, BigQueryAuditBackend.

    Contract:
      - append() is idempotent by entry id.
      - No implementation may modify or delete existing entries.
      - export() streams entries in insertion order.
    """

    @abc.abstractmethod
    async def append(self, entry: AuditEntry) -> None:
        """Append an immutable audit entry."""
        ...

    @abc.abstractmethod
    async def export(
        self,
        since_timestamp: float | None = None,
        agent_id: str | None = None,
    ) -> AsyncIterator[AuditEntry]:
        """Stream audit entries, optionally filtered."""
        ...

    @abc.abstractmethod
    async def verify_chain(self) -> tuple[bool, str | None]:
        """
        Verify the hash chain is unbroken.
        Returns (ok, first_broken_entry_id).
        """
        ...

    @abc.abstractmethod
    async def health(self) -> bool: ...


# ---------------------------------------------------------------------------
# Trace / Observability
# ---------------------------------------------------------------------------


class TraceExporter(abc.ABC):
    """
    Exports trace spans to an observability backend.

    Implementors: LocalJSONExporter, OTelExporter, DatadogExporter.
    """

    @abc.abstractmethod
    async def export_spans(self, spans: list[dict[str, Any]]) -> None:
        """Ship spans to the configured backend."""
        ...


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class ToolProtocol(abc.ABC):
    """
    The minimal contract a callable must satisfy to be registered
    as a Helix tool.

    Normally you use the @tool decorator instead of subclassing this directly.
    This interface exists for advanced cases (e.g. dynamic tool generation).
    """

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def description(self) -> str: ...

    @property
    @abc.abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema describing the tool's input parameters."""
        ...

    @abc.abstractmethod
    async def __call__(self, **kwargs: Any) -> Any:
        """Execute the tool. Must be async."""
        ...


# ---------------------------------------------------------------------------
# Eval Scorer
# ---------------------------------------------------------------------------


class EvalScorer(abc.ABC):
    """
    A single scoring dimension for the evaluation engine.

    Implementors: ToolSelectionScorer, FactScorer, QualityScorer,
                  CostScorer, TrajectoryScorer.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @property
    @abc.abstractmethod
    def weight(self) -> float:
        """Relative weight of this scorer in the overall score (0.0–1.0)."""
        ...

    @abc.abstractmethod
    async def score(
        self,
        case: EvalCase,
        result_output: str,
        tool_calls: list[ToolCallRecord],
        cost_usd: float,
        steps: int,
    ) -> float:
        """Return a score in [0.0, 1.0]."""
        ...


# ---------------------------------------------------------------------------
# Framework Adapter
# ---------------------------------------------------------------------------


class FrameworkAdapter(abc.ABC):
    """
    Bridge between an external agent framework and Helix.

    Implementors: CrewAIAdapter, LangChainAdapter, AutoGenAdapter.

    Contract:
      - run() executes the wrapped framework's agent/workflow and
        returns a result that Helix can observe (cost, trace, output).
      - Adapters must surrender LLM call interception to the HelixLLMShim.
      - Adapters must NOT modify the external framework's internal state
        beyond patching LLM clients.
    """

    @abc.abstractmethod
    async def run(self, task: str, context: ExecutionContext) -> dict[str, Any]:
        """
        Run the wrapped framework under Helix observation.
        Returns dict with keys: output, cost_usd, steps, raw_result.
        """
        ...

    @abc.abstractmethod
    def patch_llm(self, shim: Any) -> None:
        """
        Replace the framework's LLM client(s) with the HelixLLMShim.
        Called once during adapter construction.
        """
        ...

    @abc.abstractmethod
    def extract_tool_calls(self, raw_result: Any) -> list[ToolCallRecord]:
        """
        Parse framework-specific result format into Helix ToolCallRecords.
        """
        ...


# ---------------------------------------------------------------------------
# Session Store
# ---------------------------------------------------------------------------


class SessionStore(abc.ABC):
    """
    Persistence layer for multi-turn conversation sessions.

    Implementors: InMemorySessionStore, RedisSessionStore.
    """

    @abc.abstractmethod
    async def save(self, session_id: str, state: dict[str, Any]) -> None: ...

    @abc.abstractmethod
    async def load(self, session_id: str) -> dict[str, Any] | None:
        """Return None if session does not exist."""
        ...

    @abc.abstractmethod
    async def delete(self, session_id: str) -> None: ...

    @abc.abstractmethod
    async def list_sessions(self, agent_id: str | None = None) -> list[str]:
        """List session IDs, optionally filtered by agent."""
        ...

    @abc.abstractmethod
    async def health(self) -> bool: ...


# ---------------------------------------------------------------------------
# Prompt Registry
# ---------------------------------------------------------------------------


class PromptRegistry(abc.ABC):
    """
    Versioned storage and retrieval for agent prompts.

    Enables non-engineers to edit prompts without touching code,
    and enables A/B testing between prompt variants.
    """

    @abc.abstractmethod
    async def get(self, prompt_id: str, version: str | None = None) -> str:
        """
        Retrieve a prompt by id.
        If version is None, return the active version.
        Raises HelixPromptNotFoundError if not found.
        """
        ...

    @abc.abstractmethod
    async def set(
        self,
        prompt_id: str,
        content: str,
        version: str,
        activate: bool = False,
    ) -> None:
        """Store a prompt version. Optionally make it the active version."""
        ...

    @abc.abstractmethod
    async def activate(self, prompt_id: str, version: str) -> None:
        """Switch the active version for a prompt id."""
        ...

    @abc.abstractmethod
    async def list_versions(self, prompt_id: str) -> list[str]:
        """Return all version strings for a prompt id, oldest first."""
        ...

    @abc.abstractmethod
    async def health(self) -> bool: ...
