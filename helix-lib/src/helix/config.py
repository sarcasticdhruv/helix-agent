"""
helix/config.py

All configuration and data models for Helix.

Design rules:
  - Pure Pydantic. No business logic. No imports from helix internals.
  - Every model is fully typed and validated on construction.
  - Defaults are production-safe, not demo-friendly.
  - Models are immutable where state should not change after construction.
  - Enums over raw strings for all categorical fields.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AgentMode(str, Enum):
    EXPLORE = "explore"
    PRODUCTION = "production"


class MemoryKind(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    TOOL_RESULT = "tool_result"
    REASONING = "reasoning"
    EPISODE_REF = "episode_ref"


class EpisodeOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"


class FailureClass(str, Enum):
    TIMEOUT = "timeout"
    AUTH_ERROR = "auth_error"
    SCHEMA_MISMATCH = "schema_mismatch"
    RATE_LIMIT = "rate_limit"
    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"
    HALLUCINATED_CALL = "hallucinated"
    NETWORK_ERROR = "network"
    VALIDATION_ERROR = "validation"
    UNKNOWN = "unknown"


class HITLDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"
    MODIFY = "modify"


class ContextMessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class WorkflowMode(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    MAP = "map"
    REDUCE = "reduce"
    HUMAN_REVIEW = "human_review"


class BudgetStrategy(str, Enum):
    STOP = "stop"  # Hard stop when budget exceeded (default)
    DEGRADE = "degrade"  # Switch to cheaper models as budget depletes


class ComplexityTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"


class AuditEventType(str, Enum):
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TOOL_FAILURE = "tool_failure"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    LLM_CALL = "llm_call"
    LLM_RESPONSE = "llm_response"
    COST_CHECK = "cost_check"
    HITL_REQUEST = "hitl_request"
    HITL_RESPONSE = "hitl_response"
    GUARDRAIL_PASS = "guardrail_pass"
    GUARDRAIL_BLOCK = "guardrail_block"
    LOOP_DETECTED = "loop_detected"
    SAFETY_VIOLATION = "safety_violation"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXCEEDED = "budget_exceeded"


# ---------------------------------------------------------------------------
# Memory models
# ---------------------------------------------------------------------------


class MemoryEntry(BaseModel):
    """A single entry in short-term or long-term agent memory."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    kind: MemoryKind = MemoryKind.FACT
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    agent_id: str | None = None
    version: int = 1  # For optimistic locking in shared memory

    model_config = ConfigDict(frozen=False)


class Episode(BaseModel):
    """
    A completed agent run stored for episodic learning.
    Failed episodes include a post-mortem analysis.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    task: str
    task_embedding: list[float] | None = None
    outcome: EpisodeOutcome
    summary: str = ""
    steps: int = 0
    cost_usd: float = 0.0
    tools_used: list[str] = Field(default_factory=list)
    failure_reason: str | None = None
    learned_strategy: str | None = None  # Post-mortem recommendation
    created_at: float = Field(default_factory=time.time)

    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# Context models
# ---------------------------------------------------------------------------


class ContextMessage(BaseModel):
    """A single message in an agent's active context window."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: ContextMessageRole
    content: str
    step_added: int = 0
    relevance: float = 1.0
    pinned: bool = False  # Pinned messages are never evicted
    reference_score: float = 0.0  # Boosted when cited in subsequent LLM response
    tool_name: str | None = None  # For role=tool messages
    token_count: int | None = None

    model_config = ConfigDict(frozen=False)


# ---------------------------------------------------------------------------
# Token usage / cost
# ---------------------------------------------------------------------------


class TokenUsage(BaseModel):
    """Token counts for a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0  # Provider-reported prefix cache hits

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    model_config = ConfigDict(frozen=True)


class ModelPricing(BaseModel):
    """Per-model pricing in USD per 1K tokens."""

    model: str
    prompt_cost_per_1k: float
    completion_cost_per_1k: float
    cached_cost_per_1k: float = 0.0

    def calculate_cost(self, usage: TokenUsage) -> float:
        return (
            (usage.prompt_tokens - usage.cached_tokens) / 1000 * self.prompt_cost_per_1k
            + usage.cached_tokens / 1000 * self.cached_cost_per_1k
            + usage.completion_tokens / 1000 * self.completion_cost_per_1k
        )

    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# Model response
# ---------------------------------------------------------------------------


class ToolCallRecord(BaseModel):
    """A single tool call made by the model during an agent run."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_name: str
    arguments: dict[str, Any]
    result: Any | None = None
    failure_class: FailureClass | None = None
    retries: int = 0
    duration_ms: float | None = None
    step: int = 0


class ModelResponse(BaseModel):
    """Structured response from any LLMProvider."""

    content: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    model: str
    provider: str
    finish_reason: str = "stop"  # stop | tool_calls | length | content_filter
    cached: bool = False


# ---------------------------------------------------------------------------
# Cache models
# ---------------------------------------------------------------------------


class CacheEntry(BaseModel):
    """A single entry in the semantic response cache."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    query_embedding: list[float]
    context_hash: str
    response: str
    cost_usd: float
    similarity: float = 1.0  # Populated on retrieval
    age_s: float = 0.0  # Populated on retrieval
    created_at: float = Field(default_factory=time.time)


class CacheHit(BaseModel):
    """Returned by CacheController when a cache hit occurs."""

    response: str
    similarity: float
    age_s: float
    tier: str  # "semantic" | "plan"
    saved_usd: float


class PlanTemplate(BaseModel):
    """
    Cached plan structure extracted from a successful agent run.
    Used by Agentic Plan Caching (APC).
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_description: str
    task_embedding: list[float] | None = None
    keywords: list[str] = Field(default_factory=list)
    steps_description: str  # Structured step sequence
    tool_sequence: list[str] = Field(default_factory=list)
    success_rate: float = 1.0
    run_count: int = 1
    avg_cost_usd: float = 0.0
    score: float = 0.0  # Populated during match
    created_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# HITL models
# ---------------------------------------------------------------------------


class HITLRequest(BaseModel):
    """An approval request sent to a human reviewer."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    prompt: str
    context_summary: str = ""
    risk_level: str = "medium"  # low | medium | high | critical
    timeout_seconds: float = 300.0
    proposed_action: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class HITLResponse(BaseModel):
    """A human reviewer's decision on a HITL request."""

    request_id: str
    decision: HITLDecision
    reviewer_id: str | None = None
    note: str | None = None
    modified_action: str | None = None  # Used when decision=MODIFY
    responded_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


class AuditEntry(BaseModel):
    """A single, immutable entry in the tamper-evident audit log."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AuditEventType
    agent_id: str
    session_id: str | None = None
    tenant_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
    prev_hash: str = ""  # Hash of preceding entry
    entry_hash: str = ""  # Hash of this entry's canonical content

    def verify(self, prev_entry: AuditEntry) -> bool:
        return self.prev_hash == prev_entry.entry_hash

    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# Guardrail result
# ---------------------------------------------------------------------------


class GuardrailResult(BaseModel):
    """Result from a single Guardrail.check() call."""

    passed: bool
    guardrail_name: str
    modified_content: str | None = None
    reason: str | None = None

    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# Eval models
# ---------------------------------------------------------------------------


class ExpectedTrajectory(BaseModel):
    """Expected tool call sequence for trajectory evaluation."""

    tool_sequence: list[str] = Field(default_factory=list)
    max_steps: int | None = None
    must_not_call: list[str] = Field(default_factory=list)


class EvalCase(BaseModel):
    """A single evaluation test case."""

    name: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    input: str
    expected_tools: list[str] = Field(default_factory=list)
    expected_facts: list[str] = Field(default_factory=list)
    expected_trajectory: ExpectedTrajectory | None = None
    max_steps: int = 10
    max_cost_usd: float = 1.0
    pass_threshold: float = 0.7
    tags: list[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)


class EvalCaseResult(BaseModel):
    """Result of running a single EvalCase against an agent."""

    case_name: str
    input: str
    output: str
    passed: bool
    scores: dict[str, float]  # {scorer_name: score}
    overall: float
    cost_usd: float
    steps: int
    duration_s: float = 0.0
    failure_reason: str | None = None

    model_config = ConfigDict(frozen=True)


class EvalRunResult(BaseModel):
    """Result of running an entire EvalSuite."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    suite_name: str
    timestamp: float = Field(default_factory=time.time)
    pass_count: int
    fail_count: int
    results: list[EvalCaseResult]
    total_cost_usd: float
    duration_s: float = 0.0

    @property
    def pass_rate(self) -> float:
        total = self.pass_count + self.fail_count
        return self.pass_count / total if total > 0 else 0.0

    @property
    def scores_by_case(self) -> dict[str, float]:
        return {r.case_name: r.overall for r in self.results}

    model_config = ConfigDict(frozen=True)


# ---------------------------------------------------------------------------
# Sub-configs (used inside AgentConfig)
# ---------------------------------------------------------------------------


class MemoryConfig(BaseModel):
    backend: str = "inmemory"  # "inmemory" | "pinecone" | "qdrant" | "chroma"
    short_term_limit: int = 20  # Max messages in rolling buffer
    auto_promote: bool = True  # Promote important memories to long-term
    importance_threshold: float = 0.7
    embedding_model: str = "text-embedding-3-small"


class CacheConfig(BaseModel):
    enabled: bool = True
    semantic_threshold: float = 0.92
    ttl_seconds: int = 3600
    max_entries: int = 10_000
    exclude_patterns: list[str] = Field(
        default_factory=lambda: ["today", "now", "current", "latest", "price"]
    )
    plan_cache_enabled: bool = True
    plan_match_threshold: float = 0.85


class HITLConfig(BaseModel):
    enabled: bool = False
    on_confidence_below: float | None = None  # e.g. 0.5
    on_tool_risk: list[str] = Field(default_factory=list)  # tool names that require approval
    on_cost_above_usd: float | None = None
    transport: str = "cli"  # "cli" | "webhook" | "queue"
    transport_config: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = 300.0


class PermissionConfig(BaseModel):
    allowed_tools: list[str] | None = None  # None = all allowed
    denied_tools: list[str] = Field(default_factory=list)
    allowed_domains: list[str] | None = None  # For web tools
    max_file_size_mb: float = 10.0


class ObservabilityConfig(BaseModel):
    trace_enabled: bool = True
    trace_backend: str = "local"  # "local" | "s3" | "otel"
    trace_config: dict[str, Any] = Field(default_factory=dict)
    audit_enabled: bool = True
    audit_backend: str = "local"
    audit_config: dict[str, Any] = Field(default_factory=dict)
    metrics_enabled: bool = True


class BudgetDegradationStep(BaseModel):
    """One step in a budget degradation strategy."""

    at_pct: float  # e.g. 0.7 = when 70% of budget spent
    action: str  # "switch_model" | "skip_optional" | "summarize_conclude"
    switch_to_model: str | None = None


class BudgetConfig(BaseModel):
    budget_usd: float
    warn_at_pct: float = 0.8
    strategy: BudgetStrategy = BudgetStrategy.STOP
    degradation_steps: list[BudgetDegradationStep] = Field(default_factory=list)

    @field_validator("budget_usd")
    @classmethod
    def budget_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("budget_usd must be positive")
        return v

    @field_validator("warn_at_pct")
    @classmethod
    def warn_pct_range(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("warn_at_pct must be between 0 and 1")
        return v


class StructuredOutputConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    """Configuration for typed/structured output from agent runs."""

    enabled: bool = False
    json_schema: dict[str, Any] | None = (
        None  # JSON Schema (renamed from schema to avoid Pydantic v2 conflict)
    )
    pydantic_model: Any | None = None  # Pydantic model class
    max_retries: int = 2
    use_native: bool = True  # Use provider native structured output when available


class ModelConfig(BaseModel):
    """LLM selection and routing configuration."""

    primary: str = ""  # Empty = auto-detect from available keys at runtime
    fallback_chain: list[str] = Field(default_factory=list)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    auto_route: bool = True  # Use complexity estimator to pick model
    provider_overrides: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# AgentConfig — the top-level developer-facing config
# ---------------------------------------------------------------------------


class AgentConfig(BaseModel):
    """
    Complete configuration for a Helix agent.

    This is the single source of truth for everything an agent
    needs to run. It contains no logic — only validated data.

    Production mode enforces budget_usd. Explore mode is permissive.
    """

    # Identity
    name: str
    role: str
    goal: str
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str | None = None

    # Mode
    mode: AgentMode = AgentMode.EXPLORE

    # Model
    model: ModelConfig = Field(default_factory=ModelConfig)

    # Budget (required in production)
    budget: BudgetConfig | None = None

    # Memory
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    # Cache
    cache: CacheConfig = Field(default_factory=CacheConfig)

    # Safety
    permissions: PermissionConfig = Field(default_factory=PermissionConfig)
    hitl: HITLConfig = Field(default_factory=HITLConfig)
    guardrails: list[str] = Field(default_factory=list)  # Guardrail names

    # Loops and context
    loop_limit: int = 50
    context_limit_tokens: int = 128_000

    # Output
    structured_output: StructuredOutputConfig = Field(default_factory=StructuredOutputConfig)

    # Observability
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    # Prompt registry
    system_prompt_id: str | None = None  # Lookup from PromptRegistry
    system_prompt_override: str | None = None  # Inline override

    @model_validator(mode="after")
    def production_requires_budget(self) -> AgentConfig:
        if self.mode == AgentMode.PRODUCTION and self.budget is None:
            raise ValueError(
                "AgentConfig: 'budget' is required when mode='production'. "
                "Set budget=BudgetConfig(budget_usd=X.XX)."
            )
        return self

    @model_validator(mode="after")
    def production_tightens_loop_limit(self) -> AgentConfig:
        if self.mode == AgentMode.PRODUCTION and self.loop_limit > 20:
            # Silently enforce the production cap
            object.__setattr__(self, "loop_limit", 20)
        return self

    model_config = ConfigDict(frozen=False)


# ---------------------------------------------------------------------------
# WorkflowConfig
# ---------------------------------------------------------------------------


class StepConfig(BaseModel):
    name: str
    retry: int = 0
    fallback: str | None = None  # Name of fallback step
    timeout_s: float | None = None


class WorkflowConfig(BaseModel):
    name: str
    mode: WorkflowMode = WorkflowMode.SEQUENTIAL
    budget_usd: float | None = None
    max_parallel: int = 10
    loop_limit: int = 50
    on_failure: str = "fail"  # "fail" | "continue" | "fallback"


# ---------------------------------------------------------------------------
# TeamConfig
# ---------------------------------------------------------------------------


class TeamConfig(BaseModel):
    name: str
    strategy: str = "sequential"  # "sequential" | "parallel" | "hierarchical"
    lead_agent_id: str | None = None
    shared_memory: bool = True
    budget_usd: float | None = None

    @field_validator("strategy")
    @classmethod
    def valid_strategy(cls, v: str) -> str:
        if v not in ("sequential", "parallel", "hierarchical"):
            raise ValueError(f"Invalid team strategy: {v}")
        return v


# ---------------------------------------------------------------------------
# SessionConfig
# ---------------------------------------------------------------------------


class SessionConfig(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    tenant_id: str | None = None
    idle_timeout_s: float = 1800.0  # 30 minutes
    max_duration_s: float = 86400.0  # 24 hours
    store: str = "inmemory"  # "inmemory" | "redis"
    store_config: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# RuntimeConfig
# ---------------------------------------------------------------------------


class RuntimeConfig(BaseModel):
    """Top-level Helix runtime configuration."""

    workers: int = Field(default=4, gt=0)
    queue_max_size: int = 1000
    default_mode: AgentMode = AgentMode.EXPLORE
    trace_dir: str = ".helix/traces"
    wal_dir: str = ".helix/wal"
    health_check_interval_s: float = 30.0
    shutdown_timeout_s: float = 30.0

    model_config = ConfigDict(frozen=True)
