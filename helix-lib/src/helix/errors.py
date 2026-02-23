"""
helix/errors.py

All Helix exceptions in one place.

Design rules:
  - Every error carries enough context to be actionable without a stack trace.
  - Errors form a hierarchy so callers can catch at the right level.
  - No raw strings — structured fields on every exception class.
  - HTTP-style status codes on errors that map to API responses.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


class HelixError(Exception):
    """
    Base for all Helix exceptions.

    All subclasses must pass a human-readable message and may
    attach structured context via the `details` dict.
    """

    status_code: int = 500

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: Dict[str, Any] = details or {}

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}({self.message!r}, details={self.details})"


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------


class HelixConfigError(HelixError):
    """Raised when AgentConfig or any subsystem config is invalid."""

    status_code = 422

    def __init__(self, field: str, reason: str, value: Any = None) -> None:
        super().__init__(
            message=f"Configuration error on field '{field}': {reason}",
            details={"field": field, "reason": reason, "value": value},
        )
        self.field = field
        self.reason = reason


# ---------------------------------------------------------------------------
# Budget / Cost
# ---------------------------------------------------------------------------


class BudgetExceededError(HelixError):
    """
    Raised by CostGovernor when an agent would exceed its budget.

    The agent is stopped immediately. The error carries the
    final cost breakdown so the caller can report it.
    """

    status_code = 402

    def __init__(
        self,
        agent_id: str,
        budget_usd: float,
        spent_usd: float,
        attempted_usd: float,
    ) -> None:
        super().__init__(
            message=(
                f"Agent '{agent_id}' budget ${budget_usd:.4f} exceeded. "
                f"Spent ${spent_usd:.4f}, attempted ${attempted_usd:.4f}."
            ),
            details={
                "agent_id": agent_id,
                "budget_usd": budget_usd,
                "spent_usd": spent_usd,
                "attempted_usd": attempted_usd,
            },
        )
        self.agent_id = agent_id
        self.budget_usd = budget_usd
        self.spent_usd = spent_usd
        self.attempted_usd = attempted_usd


# ---------------------------------------------------------------------------
# Tool errors
# ---------------------------------------------------------------------------


class ToolError(HelixError):
    """Base for all tool-related failures."""

    status_code = 502

    def __init__(
        self,
        tool_name: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message=message, details={"tool_name": tool_name, **(details or {})})
        self.tool_name = tool_name


class ToolTimeoutError(ToolError):
    """Tool exceeded its execution time limit."""

    def __init__(self, tool_name: str, timeout_s: float) -> None:
        super().__init__(
            tool_name=tool_name,
            message=f"Tool '{tool_name}' timed out after {timeout_s}s.",
            details={"timeout_s": timeout_s},
        )
        self.timeout_s = timeout_s


class ToolAuthError(ToolError):
    """Tool credentials are missing or invalid."""

    status_code = 401

    def __init__(self, tool_name: str, reason: str = "Invalid or missing credentials") -> None:
        super().__init__(
            tool_name=tool_name,
            message=f"Tool '{tool_name}' auth failed: {reason}",
            details={"reason": reason},
        )


class ToolSchemaMismatchError(ToolError):
    """Arguments do not match the tool's declared schema."""

    status_code = 422

    def __init__(self, tool_name: str, validation_errors: Any) -> None:
        super().__init__(
            tool_name=tool_name,
            message=f"Tool '{tool_name}' received invalid arguments.",
            details={"validation_errors": str(validation_errors)},
        )
        self.validation_errors = validation_errors


class ToolRateLimitError(ToolError):
    """Upstream API quota hit. Retry after delay."""

    status_code = 429

    def __init__(self, tool_name: str, retry_after_s: Optional[float] = None) -> None:
        super().__init__(
            tool_name=tool_name,
            message=f"Tool '{tool_name}' rate limited.",
            details={"retry_after_s": retry_after_s},
        )
        self.retry_after_s = retry_after_s


class ToolNotFoundError(ToolError):
    """Target resource does not exist."""

    status_code = 404

    def __init__(self, tool_name: str, resource: str) -> None:
        super().__init__(
            tool_name=tool_name,
            message=f"Tool '{tool_name}' could not find resource: {resource}",
            details={"resource": resource},
        )


class ToolPermissionError(ToolError):
    """Agent does not have permission to call this tool."""

    status_code = 403

    def __init__(self, tool_name: str, agent_id: str) -> None:
        super().__init__(
            tool_name=tool_name,
            message=f"Agent '{agent_id}' does not have permission to call '{tool_name}'.",
            details={"agent_id": agent_id},
        )


class ToolHallucinatedError(ToolError):
    """
    The model called a tool that does not exist in the registry.
    This is a model error, not a tool error — but it's caught here.
    """

    status_code = 422

    def __init__(self, tool_name: str, available_tools: list[str]) -> None:
        super().__init__(
            tool_name=tool_name,
            message=(
                f"Model called nonexistent tool '{tool_name}'. "
                f"Available: {available_tools}"
            ),
            details={"available_tools": available_tools},
        )


class ToolNetworkError(ToolError):
    """Transient connectivity failure."""

    def __init__(self, tool_name: str, reason: str) -> None:
        super().__init__(
            tool_name=tool_name,
            message=f"Tool '{tool_name}' network error: {reason}",
            details={"reason": reason},
        )


class ToolValidationError(ToolError):
    """Tool returned output that fails its declared output schema."""

    status_code = 502

    def __init__(self, tool_name: str, validation_errors: Any) -> None:
        super().__init__(
            tool_name=tool_name,
            message=f"Tool '{tool_name}' returned invalid output.",
            details={"validation_errors": str(validation_errors)},
        )


# ---------------------------------------------------------------------------
# Context / Memory
# ---------------------------------------------------------------------------


class ContextLimitError(HelixError):
    """Context window exceeded and cannot be compacted further."""

    status_code = 413

    def __init__(self, agent_id: str, tokens: int, limit: int) -> None:
        super().__init__(
            message=(
                f"Agent '{agent_id}' context window full: "
                f"{tokens} tokens exceeds limit {limit}."
            ),
            details={"agent_id": agent_id, "tokens": tokens, "limit": limit},
        )


class MemoryBackendError(HelixError):
    """Memory backend is unavailable or returned an error."""

    def __init__(self, backend: str, operation: str, reason: str) -> None:
        super().__init__(
            message=f"Memory backend '{backend}' failed on '{operation}': {reason}",
            details={"backend": backend, "operation": operation, "reason": reason},
        )


class MemoryConflictError(HelixError):
    """
    Optimistic locking failed on shared team memory.
    Another agent wrote to the same key between read and write.
    """

    status_code = 409

    def __init__(self, key: str, agent_id: str) -> None:
        super().__init__(
            message=(
                f"Agent '{agent_id}' lost write conflict on memory key '{key}'. "
                "Retry the operation."
            ),
            details={"key": key, "agent_id": agent_id},
        )


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------


class LoopDetectedError(HelixError):
    """
    The orchestration engine detected an agent in an infinite loop.
    Execution is stopped; full state is preserved in details.
    """

    status_code = 508

    def __init__(
        self,
        agent_id: str,
        signal: str,
        step_count: int,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=(
                f"Agent '{agent_id}' loop detected (signal: {signal}) "
                f"after {step_count} steps."
            ),
            details={
                "agent_id": agent_id,
                "signal": signal,
                "step_count": step_count,
                **(details or {}),
            },
        )
        self.signal = signal
        self.step_count = step_count


# ---------------------------------------------------------------------------
# Provider / Model
# ---------------------------------------------------------------------------


class HelixProviderError(HelixError):
    """
    Raised by LLMProvider implementations.
    Wraps raw SDK exceptions so callers never see provider-specific errors.

    Accepts either:
      HelixProviderError(provider=..., model=..., reason="message")
      HelixProviderError(provider=..., model=..., original=Exception())
    """

    def __init__(
        self,
        provider: str,
        model: str,
        reason: Optional[str] = None,
        original: Optional[Exception] = None,
        status_code: int = 502,
        retryable: bool = False,
    ) -> None:
        if reason is None and original is not None:
            reason = f"{type(original).__name__}: {original}"
        elif reason is None:
            reason = "Unknown provider error"
        super().__init__(
            message=f"Provider '{provider}' model '{model}' error: {reason}",
            details={
                "provider": provider,
                "model": model,
                "reason": reason,
                "retryable": retryable,
            },
        )
        self.status_code = status_code
        self.retryable = retryable
        self.original = original


class ModelNotFoundError(HelixProviderError):
    """The requested model is not available on this provider."""

    def __init__(self, provider: str, model: str) -> None:
        super().__init__(
            provider=provider,
            model=model,
            reason=f"Model '{model}' not available on provider '{provider}'.",
            status_code=404,
            retryable=False,
        )


class AllModelsExhaustedError(HelixError):
    """All models in the fallback chain failed."""

    def __init__(self, attempted: list[str], errors: Optional[Dict[str, str]] = None) -> None:
        if errors:
            error_lines = "; ".join(f"{m}: {e}" for m, e in errors.items())
            message = f"All models exhausted. Attempted: {attempted}. Errors: {error_lines}"
        else:
            message = f"All models exhausted. Attempted: {attempted}"
        super().__init__(
            message=message,
            details={"attempted": attempted, "errors": errors or {}},
        )


# ---------------------------------------------------------------------------
# Safety / Guardrails / Permissions
# ---------------------------------------------------------------------------


class GuardrailViolationError(HelixError):
    """Content blocked by a guardrail."""

    status_code = 451

    def __init__(self, guardrail_name: str, reason: str, content_preview: str = "") -> None:
        super().__init__(
            message=f"Guardrail '{guardrail_name}' blocked content: {reason}",
            details={
                "guardrail": guardrail_name,
                "reason": reason,
                "content_preview": content_preview[:100],
            },
        )
        self.guardrail_name = guardrail_name


class PermissionDeniedError(HelixError):
    """Agent attempted an action outside its declared permission scope."""

    status_code = 403

    def __init__(self, agent_id: str, action: str, resource: str) -> None:
        super().__init__(
            message=(
                f"Agent '{agent_id}' denied: action '{action}' on '{resource}' "
                "is outside declared permissions."
            ),
            details={"agent_id": agent_id, "action": action, "resource": resource},
        )


class SafetyViolationError(HelixError):
    """
    Hard safety boundary crossed. Execution stops immediately.
    This is always logged to the audit trail.
    """

    status_code = 451

    def __init__(self, agent_id: str, violation_type: str, details: Dict[str, Any]) -> None:
        super().__init__(
            message=f"Safety violation by agent '{agent_id}': {violation_type}",
            details={"agent_id": agent_id, "violation_type": violation_type, **details},
        )


# ---------------------------------------------------------------------------
# HITL
# ---------------------------------------------------------------------------


class HITLTimeoutError(HelixError):
    """Human approval was not received within the configured timeout."""

    status_code = 408

    def __init__(self, request_id: str, timeout_s: float) -> None:
        super().__init__(
            message=f"HITL request '{request_id}' timed out after {timeout_s}s.",
            details={"request_id": request_id, "timeout_s": timeout_s},
        )


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class WorkflowError(HelixError):
    """Base for workflow execution failures."""

    def __init__(self, workflow_name: str, step: str, reason: str) -> None:
        super().__init__(
            message=f"Workflow '{workflow_name}' failed at step '{step}': {reason}",
            details={"workflow_name": workflow_name, "step": step, "reason": reason},
        )


class StepMaxRetriesError(WorkflowError):
    """A workflow step exhausted its retry budget."""

    def __init__(self, workflow_name: str, step: str, max_retries: int) -> None:
        super().__init__(
            workflow_name=workflow_name,
            step=step,
            reason=f"Exceeded max retries ({max_retries}).",
        )
        self.details["max_retries"] = max_retries


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


class EvalRegressionError(HelixError):
    """Raised by RegressionGate when scores drop below baseline."""

    status_code = 424

    def __init__(self, regressions: list[dict]) -> None:
        super().__init__(
            message=(
                f"{len(regressions)} eval regression(s) detected. "
                "Deployment blocked."
            ),
            details={"regressions": regressions},
        )
        self.regressions = regressions


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class SessionNotFoundError(HelixError):
    """The requested session id does not exist."""

    status_code = 404

    def __init__(self, session_id: str) -> None:
        super().__init__(
            message=f"Session '{session_id}' not found.",
            details={"session_id": session_id},
        )


class SessionExpiredError(HelixError):
    """Session exceeded its idle or absolute timeout."""

    status_code = 410

    def __init__(self, session_id: str) -> None:
        super().__init__(
            message=f"Session '{session_id}' has expired.",
            details={"session_id": session_id},
        )


# ---------------------------------------------------------------------------
# Prompt Registry
# ---------------------------------------------------------------------------


class HelixPromptNotFoundError(HelixError):
    """The requested prompt id or version does not exist."""

    status_code = 404

    def __init__(self, prompt_id: str, version: Optional[str] = None) -> None:
        super().__init__(
            message=(
                f"Prompt '{prompt_id}'"
                + (f" version '{version}'" if version else "")
                + " not found."
            ),
            details={"prompt_id": prompt_id, "version": version},
        )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class AdapterError(HelixError):
    """Failure during framework adapter execution."""

    def __init__(self, framework: str, reason: str) -> None:
        super().__init__(
            message=f"Adapter for '{framework}' failed: {reason}",
            details={"framework": framework, "reason": reason},
        )


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class CacheBackendError(HelixError):
    """Cache backend is unavailable."""

    def __init__(self, backend: str, operation: str, reason: str) -> None:
        super().__init__(
            message=f"Cache backend '{backend}' failed on '{operation}': {reason}",
            details={"backend": backend, "operation": operation, "reason": reason},
        )
