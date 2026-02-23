"""
helix/core/tool.py

Tool registration, decoration, and the ToolRegistry.

Design:
  - @tool decorator works on both sync and async functions.
  - Registry is the single source of truth for tool schemas
    sent to LLM providers.
  - Tool execution is isolated: failures raise typed HelixErrors,
    never raw exceptions.
  - Permissions are checked at call time against ExecutionContext.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from helix.config import FailureClass, ToolCallRecord
from helix.errors import (
    ToolError,
    ToolPermissionError,
    ToolSchemaMismatchError,
    ToolTimeoutError,
)
from helix.interfaces import ToolProtocol

# ---------------------------------------------------------------------------
# Internal registered tool wrapper
# ---------------------------------------------------------------------------


class RegisteredTool(ToolProtocol):
    """
    Wraps a developer-provided function with metadata and execution logic.
    Created by the @tool decorator — not instantiated directly.
    """

    def __init__(
        self,
        fn: Callable,
        name: str,
        description: str,
        timeout_s: float,
        retries: int,
        on_error: str,  # "raise" | "return_none" | "fallback"
        fallback_fn: Callable | None,
        parameters_schema: dict[str, Any],
    ) -> None:
        self._fn = fn
        self._name = name
        self._description = description
        self._timeout_s = timeout_s
        self._retries = retries
        self._on_error = on_error
        self._fallback_fn = fallback_fn
        self._schema = parameters_schema
        self._is_async = asyncio.iscoroutinefunction(fn)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._schema

    async def __call__(self, **kwargs: Any) -> Any:
        last_exc: Exception | None = None

        for attempt in range(self._retries + 1):
            try:
                return await self._execute(**kwargs)
            except ToolTimeoutError:
                raise  # Timeout is not retried
            except ToolError as e:
                last_exc = e
                if attempt < self._retries:
                    await asyncio.sleep(0.5 * (2**attempt))  # Exponential backoff
                    continue
                break

        # All retries exhausted
        if self._on_error == "return_none":
            return None
        if self._on_error == "fallback" and self._fallback_fn is not None:
            if asyncio.iscoroutinefunction(self._fallback_fn):
                return await self._fallback_fn(**kwargs)
            return self._fallback_fn(**kwargs)
        raise last_exc  # type: ignore[misc]

    async def _execute(self, **kwargs: Any) -> Any:
        if self._is_async:
            coro = self._fn(**kwargs)
            try:
                return await asyncio.wait_for(coro, timeout=self._timeout_s)
            except TimeoutError:
                raise ToolTimeoutError(tool_name=self._name, timeout_s=self._timeout_s)
        else:
            try:
                loop = asyncio.get_event_loop()
                return await asyncio.wait_for(
                    loop.run_in_executor(None, functools.partial(self._fn, **kwargs)),
                    timeout=self._timeout_s,
                )
            except TimeoutError:
                raise ToolTimeoutError(tool_name=self._name, timeout_s=self._timeout_s)

    def to_llm_schema(self) -> dict[str, Any]:
        """
        OpenAI / Anthropic compatible tool schema for inclusion in LLM calls.
        """
        return {
            "name": self._name,
            "description": self._description,
            "parameters": self._schema,
        }


# ---------------------------------------------------------------------------
# Schema extraction from function signature
# ---------------------------------------------------------------------------


def _extract_schema(fn: Callable) -> dict[str, Any]:
    """
    Build a JSON Schema from function type annotations.
    Supports: str, int, float, bool, list, dict, Optional[X].
    Pydantic BaseModel parameters are expanded inline.
    """
    sig = inspect.signature(fn)
    hints = fn.__annotations__ if hasattr(fn, "__annotations__") else {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        list: "array",
        dict: "object",
    }

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        annotation = hints.get(param_name, Any)
        origin = getattr(annotation, "__origin__", None)

        # Handle Optional[X] -> nullable X
        if origin is type(None) or str(annotation) in ("typing.Optional",):
            # Simplify: treat Optional as optional field
            properties[param_name] = {"type": "string", "nullable": True}
            continue

        # Check for Pydantic model
        try:
            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                properties[param_name] = annotation.model_json_schema()
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)
                continue
        except Exception:
            pass

        # Primitive types
        json_type = type_map.get(annotation, "string")
        prop: dict[str, Any] = {"type": json_type}

        # Add description from docstring if available (rough extraction)
        prop_desc = _extract_param_doc(fn, param_name)
        if prop_desc:
            prop["description"] = prop_desc

        properties[param_name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def _extract_param_doc(fn: Callable, param_name: str) -> str | None:
    """Cheap extraction of :param name: from docstring."""
    doc = inspect.getdoc(fn) or ""
    for line in doc.splitlines():
        line = line.strip()
        if line.startswith(f":param {param_name}:"):
            return line.split(":", 2)[-1].strip()
        if line.startswith(f"{param_name}:"):
            return line.split(":", 1)[-1].strip()
    return None


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------


def tool(
    description: str,
    name: str | None = None,
    timeout: float = 30.0,
    retries: int = 0,
    on_error: str = "raise",
    fallback: Callable | None = None,
    parameters_schema: dict[str, Any] | None = None,
) -> Callable:
    """
    Decorator that registers a function as a Helix tool.

    Args:
        description: Human-readable description sent to the LLM.
        name: Override tool name. Defaults to function name.
        timeout: Execution timeout in seconds.
        retries: Number of retry attempts on ToolError.
        on_error: "raise" | "return_none" | "fallback"
        fallback: Callable used when on_error="fallback".
        parameters_schema: Override auto-generated JSON Schema.

    Example::

        @tool(description="Search the web for current information.")
        async def web_search(query: str, max_results: int = 5) -> list:
            ...
    """

    def decorator(fn: Callable) -> RegisteredTool:
        tool_name = name or fn.__name__
        schema = parameters_schema or _extract_schema(fn)
        registered = RegisteredTool(
            fn=fn,
            name=tool_name,
            description=description,
            timeout_s=timeout,
            retries=retries,
            on_error=on_error,
            fallback_fn=fallback,
            parameters_schema=schema,
        )
        # Preserve original function attributes for introspection
        functools.update_wrapper(registered, fn)
        return registered

    return decorator


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """
    Central registry for all tools available to Helix agents.

    - A single global registry is available at `helix.core.tool.registry`.
    - Agents receive a filtered view based on their PermissionConfig.
    - The registry is the single source of truth for LLM tool schemas.
    """

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, tool_or_fn: Any, name: str | None = None) -> ToolRegistry:
        """
        Register a tool. Accepts either a RegisteredTool (from @tool decorator)
        or a plain callable (auto-wrapped with minimal config).
        """
        if isinstance(tool_or_fn, RegisteredTool):
            self._tools[tool_or_fn.name] = tool_or_fn
        elif callable(tool_or_fn):
            tool_name = name or getattr(tool_or_fn, "__name__", "unknown")
            doc = inspect.getdoc(tool_or_fn) or "No description provided."
            schema = _extract_schema(tool_or_fn)
            wrapped = RegisteredTool(
                fn=tool_or_fn,
                name=tool_name,
                description=doc,
                timeout_s=30.0,
                retries=0,
                on_error="raise",
                fallback_fn=None,
                parameters_schema=schema,
            )
            self._tools[tool_name] = wrapped
        else:
            raise ValueError(f"Cannot register {type(tool_or_fn)} as a tool.")
        return self

    def get(self, name: str) -> RegisteredTool:
        if name not in self._tools:
            from helix.errors import ToolHallucinatedError

            raise ToolHallucinatedError(
                tool_name=name,
                available_tools=list(self._tools.keys()),
            )
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def all(self) -> list[RegisteredTool]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def filtered(
        self,
        allowed: list[str] | None,
        denied: list[str],
    ) -> ToolRegistryView:
        """Return a filtered view of the registry for an agent's permission scope."""
        return ToolRegistryView(self, allowed=allowed, denied=denied)

    def schemas(self, allowed: list[str] | None = None) -> list[dict[str, Any]]:
        """Return LLM-compatible tool schemas."""
        tools = self.all()
        if allowed is not None:
            tools = [t for t in tools if t.name in allowed]
        return [t.to_llm_schema() for t in tools]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return self.has(name)


class ToolRegistryView:
    """
    A filtered, read-only view of a ToolRegistry scoped to an agent's permissions.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        allowed: list[str] | None,
        denied: list[str],
    ) -> None:
        self._registry = registry
        self._allowed = allowed
        self._denied = set(denied)

    def _is_permitted(self, name: str) -> bool:
        if name in self._denied:
            return False
        if self._allowed is not None and name not in self._allowed:
            return False
        return True

    def get(self, name: str, agent_id: str = "") -> RegisteredTool:
        if not self._is_permitted(name):
            raise ToolPermissionError(tool_name=name, agent_id=agent_id)
        return self._registry.get(name)

    def all(self) -> list[RegisteredTool]:
        return [t for t in self._registry.all() if self._is_permitted(t.name)]

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_llm_schema() for t in self.all()]

    def has(self, name: str) -> bool:
        return self._is_permitted(name) and self._registry.has(name)


# ---------------------------------------------------------------------------
# Tool execution with timing and record production
# ---------------------------------------------------------------------------


async def execute_tool(
    registry_view: ToolRegistryView,
    tool_name: str,
    arguments: dict[str, Any],
    step: int,
    agent_id: str = "",
) -> ToolCallRecord:
    """
    Execute a tool and return a ToolCallRecord.
    Never raises — failures are captured in the record's failure_class.
    Callers inspect the record to decide recovery strategy.
    """
    start = time.monotonic()
    record = ToolCallRecord(
        tool_name=tool_name,
        arguments=arguments,
        step=step,
    )

    try:
        fn = registry_view.get(tool_name, agent_id=agent_id)
        result = await fn(**arguments)
        record.result = result
        record.duration_ms = (time.monotonic() - start) * 1000
    except ToolPermissionError:
        record.failure_class = FailureClass.PERMISSION_DENIED
        record.duration_ms = (time.monotonic() - start) * 1000
    except ToolTimeoutError:
        record.failure_class = FailureClass.TIMEOUT
        record.duration_ms = (time.monotonic() - start) * 1000
    except ToolSchemaMismatchError:
        record.failure_class = FailureClass.SCHEMA_MISMATCH
        record.duration_ms = (time.monotonic() - start) * 1000
    except ToolError as e:
        # Let taxonomy engine classify the rest
        record.failure_class = FailureClass.UNKNOWN
        record.result = str(e)
        record.duration_ms = (time.monotonic() - start) * 1000
    except Exception as e:
        record.failure_class = FailureClass.UNKNOWN
        record.result = str(e)
        record.duration_ms = (time.monotonic() - start) * 1000

    return record


# ---------------------------------------------------------------------------
# Module-level default registry
# ---------------------------------------------------------------------------

registry = ToolRegistry()
