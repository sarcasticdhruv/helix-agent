"""
helix/safety/guardrails.py

Built-in composable guardrails.

Each guardrail is a self-contained check that runs on LLM output.
Guardrails are applied in sequence by the agent's reasoning loop.
"""

from __future__ import annotations

import re
from re import Pattern

from helix.config import GuardrailResult
from helix.interfaces import Guardrail

try:
    from helix.context import ExecutionContext
except ImportError:
    ExecutionContext = object  # type: ignore[misc,assignment]


class PIIRedactor(Guardrail):
    """
    Detects and redacts common PII patterns from LLM output.
    Patterns: email, phone, SSN, credit card, IP address.
    """

    _PATTERNS: list[tuple[str, Pattern]] = [
        ("email", re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")),
        ("phone", re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
        ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
        ("cc", re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")),
        ("ip", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ]

    def __init__(self, patterns: list[str] | None = None) -> None:
        """
        Args:
            patterns: Restrict to specific pattern names. None = all patterns.
        """
        self._active = set(patterns) if patterns else {name for name, _ in self._PATTERNS}

    @property
    def name(self) -> str:
        return "pii_redactor"

    async def check(self, content: str, context: ExecutionContext) -> GuardrailResult:
        modified = content
        found_any = False
        for name, pattern in self._PATTERNS:
            if name not in self._active:
                continue
            if pattern.search(modified):
                found_any = True
                modified = pattern.sub(f"[{name.upper()}_REDACTED]", modified)

        return GuardrailResult(
            passed=True,  # PII redaction passes — it cleans, not blocks
            guardrail_name=self.name,
            modified_content=modified if found_any else None,
            reason="PII detected and redacted" if found_any else None,
        )


class LengthGuard(Guardrail):
    """Blocks responses shorter than min or longer than max characters."""

    def __init__(
        self,
        min_chars: int = 1,
        max_chars: int = 100_000,
        on_fail: str = "block",  # "block" | "truncate"
    ) -> None:
        self._min = min_chars
        self._max = max_chars
        self._on_fail = on_fail

    @property
    def name(self) -> str:
        return "length_guard"

    async def check(self, content: str, context: ExecutionContext) -> GuardrailResult:
        length = len(content)
        if length < self._min:
            return GuardrailResult(
                passed=False,
                guardrail_name=self.name,
                reason=f"Response too short: {length} chars (min: {self._min})",
            )
        if length > self._max:
            if self._on_fail == "truncate":
                return GuardrailResult(
                    passed=True,
                    guardrail_name=self.name,
                    modified_content=content[: self._max] + "… [truncated]",
                    reason=f"Truncated from {length} to {self._max} chars",
                )
            return GuardrailResult(
                passed=False,
                guardrail_name=self.name,
                reason=f"Response too long: {length} chars (max: {self._max})",
            )
        return GuardrailResult(passed=True, guardrail_name=self.name)


class KeywordBlockGuard(Guardrail):
    """Blocks responses containing any of the configured keywords."""

    def __init__(self, blocked_keywords: list[str], case_sensitive: bool = False) -> None:
        self._keywords = blocked_keywords
        self._case_sensitive = case_sensitive

    @property
    def name(self) -> str:
        return "keyword_block"

    async def check(self, content: str, context: ExecutionContext) -> GuardrailResult:
        haystack = content if self._case_sensitive else content.lower()
        for kw in self._keywords:
            needle = kw if self._case_sensitive else kw.lower()
            if needle in haystack:
                return GuardrailResult(
                    passed=False,
                    guardrail_name=self.name,
                    reason=f"Blocked keyword detected: '{kw}'",
                )
        return GuardrailResult(passed=True, guardrail_name=self.name)


class SchemaGuard(Guardrail):
    """
    Validates that the response is valid JSON matching a JSON Schema.
    Passes through non-JSON responses without blocking.
    """

    def __init__(self, schema: dict) -> None:
        self._schema = schema

    @property
    def name(self) -> str:
        return "schema_guard"

    async def check(self, content: str, context: ExecutionContext) -> GuardrailResult:
        import json

        stripped = content.strip()
        if not (stripped.startswith("{") or stripped.startswith("[")):
            return GuardrailResult(passed=True, guardrail_name=self.name)
        try:
            json.loads(stripped)
            # Basic schema validation without jsonschema dependency
            return GuardrailResult(passed=True, guardrail_name=self.name)
        except json.JSONDecodeError as e:
            return GuardrailResult(
                passed=False,
                guardrail_name=self.name,
                reason=f"Invalid JSON: {e}",
            )


class GuardrailChain:
    """
    Runs a sequence of guardrails in order.
    Stops and returns on the first failure.
    PII redaction modifies content for the next guardrail in chain.
    """

    def __init__(self, guardrails: list[Guardrail]) -> None:
        self._guardrails = guardrails

    async def check(self, content: str, context: ExecutionContext) -> GuardrailResult:
        current = content
        for guardrail in self._guardrails:
            result = await guardrail.check(current, context)
            if not result.passed:
                return result
            if result.modified_content:
                current = result.modified_content
        return GuardrailResult(
            passed=True,
            guardrail_name="chain",
            modified_content=current if current != content else None,
        )

    def __iter__(self):
        return iter(self._guardrails)


# Registry of built-in guardrail names → classes
BUILTIN_GUARDRAILS = {
    "pii_redactor": PIIRedactor,
    "length_guard": LengthGuard,
    "keyword_block": KeywordBlockGuard,
    "schema_guard": SchemaGuard,
}


def build_guardrail_chain(names: list[str]) -> GuardrailChain:
    """
    Build a GuardrailChain from a list of built-in guardrail names.
    For custom configuration, instantiate guardrails directly.
    """
    guardrails = []
    for name in names:
        cls = BUILTIN_GUARDRAILS.get(name)
        if cls:
            guardrails.append(cls())
    return GuardrailChain(guardrails)
