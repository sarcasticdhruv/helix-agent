"""
helix/tools/taxonomy.py

Failure taxonomy engine and recovery strategy definitions.

Every tool failure is classified into a FailureClass.
Each class maps to a recovery strategy.
Strategies are objects, not strings — they carry parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from helix.config import FailureClass

# ---------------------------------------------------------------------------
# Recovery strategy types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryStrategy:
    max: int = 3
    backoff: str = "exponential"  # "exponential" | "fixed" | "linear"
    delay_s: float = 1.0
    jitter: bool = True
    hint: str | None = None  # Injected into retry prompt for schema errors


@dataclass(frozen=True)
class EscalateStrategy:
    notify: bool = True
    message: str | None = None
    check_permissions: bool = False
    include_context: bool = False


@dataclass(frozen=True)
class FallbackStrategy:
    try_alternative: bool = True
    fallback_tool: str | None = None


@dataclass(frozen=True)
class SkipStrategy:
    log: bool = True
    warn_model: bool = True  # Include real tool list in next prompt


@dataclass(frozen=True)
class AbortStrategy:
    reason: str = ""


RecoveryStrategy = (
    RetryStrategy | EscalateStrategy | FallbackStrategy | SkipStrategy | AbortStrategy
)


# ---------------------------------------------------------------------------
# Strategy mapping
# ---------------------------------------------------------------------------

RECOVERY_STRATEGIES: dict[FailureClass, RecoveryStrategy] = {
    FailureClass.TIMEOUT: RetryStrategy(max=3, backoff="exponential", jitter=True),
    FailureClass.AUTH_ERROR: EscalateStrategy(
        notify=True, message="Tool authentication failed. Provide updated credentials."
    ),
    FailureClass.SCHEMA_MISMATCH: RetryStrategy(
        max=2,
        backoff="fixed",
        delay_s=0.5,
        hint="The arguments did not match the tool's expected schema. Correct them.",
    ),
    FailureClass.RATE_LIMIT: RetryStrategy(max=5, backoff="fixed", delay_s=60.0, jitter=False),
    FailureClass.NOT_FOUND: FallbackStrategy(try_alternative=True),
    FailureClass.PERMISSION_DENIED: EscalateStrategy(
        notify=True,
        check_permissions=True,
        message="Agent attempted to access a resource it does not have permission to use.",
    ),
    FailureClass.HALLUCINATED_CALL: SkipStrategy(log=True, warn_model=True),
    FailureClass.NETWORK_ERROR: RetryStrategy(max=3, backoff="exponential", jitter=True),
    FailureClass.VALIDATION_ERROR: RetryStrategy(
        max=2,
        backoff="fixed",
        delay_s=0.5,
        hint="The tool output did not match the expected format. Try again.",
    ),
    FailureClass.UNKNOWN: EscalateStrategy(notify=True, include_context=True),
}


# ---------------------------------------------------------------------------
# Taxonomy engine
# ---------------------------------------------------------------------------


class FailureTaxonomyEngine:
    """
    Classifies a tool execution failure into a FailureClass.

    Classification is deterministic and based on:
      1. Whether the tool exists in the registry (hallucination check)
      2. HTTP status code if available
      3. Exception type name
      4. Exception message string patterns
    """

    def __init__(self, registry_names: list[str]) -> None:
        self._registry_names = set(registry_names)

    def classify(
        self,
        tool_name: str,
        exception: Exception,
        call_args: dict[str, Any] | None = None,
    ) -> FailureClass:
        # Hallucination: tool not registered
        if tool_name not in self._registry_names:
            return FailureClass.HALLUCINATED_CALL

        # Already classified by tool.py executor
        from helix.errors import (
            ToolAuthError,
            ToolHallucinatedError,
            ToolNetworkError,
            ToolNotFoundError,
            ToolPermissionError,
            ToolRateLimitError,
            ToolSchemaMismatchError,
            ToolTimeoutError,
            ToolValidationError,
        )

        exc_map = {
            ToolTimeoutError: FailureClass.TIMEOUT,
            ToolAuthError: FailureClass.AUTH_ERROR,
            ToolSchemaMismatchError: FailureClass.SCHEMA_MISMATCH,
            ToolRateLimitError: FailureClass.RATE_LIMIT,
            ToolNotFoundError: FailureClass.NOT_FOUND,
            ToolPermissionError: FailureClass.PERMISSION_DENIED,
            ToolHallucinatedError: FailureClass.HALLUCINATED_CALL,
            ToolNetworkError: FailureClass.NETWORK_ERROR,
            ToolValidationError: FailureClass.VALIDATION_ERROR,
        }
        for exc_type, failure_class in exc_map.items():
            if isinstance(exception, exc_type):
                return failure_class

        # Fallback: inspect exception attributes and message
        exc_str = str(exception).lower()
        exc_type_name = type(exception).__name__

        if hasattr(exception, "status_code"):
            code = exception.status_code
            if code in (401, 403):
                return FailureClass.AUTH_ERROR
            if code == 404:
                return FailureClass.NOT_FOUND
            if code == 429:
                return FailureClass.RATE_LIMIT
            if code == 403:
                return FailureClass.PERMISSION_DENIED

        if "timeout" in exc_str or "timed out" in exc_str:
            return FailureClass.TIMEOUT
        if "rate limit" in exc_str or "too many requests" in exc_str:
            return FailureClass.RATE_LIMIT
        if "permission" in exc_str or "access denied" in exc_str or "forbidden" in exc_str:
            return FailureClass.PERMISSION_DENIED
        if "not found" in exc_str or "404" in exc_str:
            return FailureClass.NOT_FOUND
        if "auth" in exc_str or "unauthorized" in exc_str or "401" in exc_str:
            return FailureClass.AUTH_ERROR
        if exc_type_name in ("ConnectionError", "NetworkError", "ConnectTimeout"):
            return FailureClass.NETWORK_ERROR
        if exc_type_name in ("ValidationError", "SchemaError", "ValueError"):
            return FailureClass.SCHEMA_MISMATCH

        return FailureClass.UNKNOWN

    def recovery_strategy(self, failure_class: FailureClass) -> RecoveryStrategy:
        return RECOVERY_STRATEGIES.get(failure_class, EscalateStrategy())
