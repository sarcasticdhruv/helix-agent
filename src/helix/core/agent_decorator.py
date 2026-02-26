"""
helix/core/agent_decorator.py

@helix.agent — class-based agent definition decorator.

Lets developers define agents as annotated Python classes.
The class docstring becomes the system prompt / goal.
Methods decorated with @helix.tool become the agent's tools.

Usage::

    import helix

    @helix.agent(model="claude-sonnet-4-6", budget_usd=2.00)
    class WebResearcher:
        \"\"\"
        You are an expert web researcher.
        Find accurate, up-to-date information and always cite sources.
        \"\"\"

        @helix.tool(description="Search the web for recent information.")
        async def search(self, query: str) -> list[dict]:
            from helix.tools.builtin import web_search
            return await web_search(query)

        @helix.tool(description="Fetch and read a URL.")
        async def fetch(self, url: str) -> str:
            from helix.tools.builtin import fetch_url
            result = await fetch_url(url)
            return result.get("content", "")

    # The decorator returns a factory; call it to get an Agent instance
    researcher = WebResearcher()
    result = helix.run(researcher, "Latest AI safety research 2026")
"""

from __future__ import annotations

import functools
import inspect
from typing import Any


def agent(
    model: str | None = None,
    budget_usd: float = 0.50,
    mode: str = "explore",
    name: str | None = None,
    backstory: str = "",
    **agent_kwargs: Any,
):
    """
    Class decorator that turns a plain class into an Agent factory.

    Args:
        model:       LLM model string (e.g. ``"gpt-4o"``, ``"claude-sonnet-4-6"``).
                     Defaults to auto-detected best available model.
        budget_usd:  Spending cap in USD per run.  Default 0.50.
        mode:        ``"explore"`` (default) or ``"production"``.
        name:        Override the agent name.  Defaults to the class name.
        backstory:   Rich background context injected into the system prompt.
        **agent_kwargs: Any additional kwargs forwarded to :class:`helix.Agent`.

    Returns:
        A callable that, when called (optionally) with extra ``Agent`` kwargs,
        returns a configured :class:`helix.Agent` instance with all
        ``@helix.tool``-decorated methods registered as tools.
    """

    def decorator(cls):
        from helix.config import AgentMode, BudgetConfig, ModelConfig
        from helix.core.agent import Agent
        from helix.core.tool import RegisteredTool
        from helix.core.tool import tool as tool_decorator

        goal = inspect.getdoc(cls) or f"You are a {cls.__name__} agent."
        agent_name = name or cls.__name__

        @functools.wraps(cls)
        def factory(*args, **kwargs) -> Agent:
            # Instantiate the class so methods are bound
            instance = cls(*args, **kwargs)

            # Collect @tool-decorated methods from the instance
            tools: list[Any] = []
            for attr_name in dir(cls):
                # Skip dunder and private attributes
                if attr_name.startswith("_"):
                    continue
                raw = getattr(cls, attr_name, None)
                if raw is None:
                    continue

                # Accept either a RegisteredTool or a function flagged with _helix_tool
                is_registered = isinstance(raw, RegisteredTool)
                is_flagged = callable(raw) and getattr(raw, "_helix_tool", False)

                if is_registered or is_flagged:
                    bound_fn: Any
                    if is_registered:
                        # Bind the underlying function to the instance and re-decorate
                        underlying = raw._fn
                        bound_fn = functools.partial(underlying, instance)
                        bound_fn.__name__ = attr_name
                        bound_fn.__annotations__ = _drop_self(
                            getattr(underlying, "__annotations__", {})
                        )
                        wrapped = tool_decorator(
                            description=raw.description,
                            timeout=raw._timeout_s,
                            retries=raw._retries,
                            on_error=raw._on_error,
                        )(bound_fn)
                    else:
                        bound_fn = functools.partial(raw, instance)
                        bound_fn.__name__ = attr_name
                        bound_fn.__annotations__ = _drop_self(getattr(raw, "__annotations__", {}))
                        desc = getattr(raw, "__doc__", "") or attr_name
                        wrapped = tool_decorator(description=desc)(bound_fn)

                    tools.append(wrapped)

            return Agent(
                name=agent_name,
                role=agent_name,
                goal=goal,
                backstory=backstory,
                tools=tools,
                model=ModelConfig(primary=model) if model else ModelConfig(),
                budget=BudgetConfig(budget_usd=budget_usd),
                mode=AgentMode(mode),
                **agent_kwargs,
            )

        # Preserve the original class so isinstance checks still work
        factory.__wrapped__ = cls  # type: ignore[attr-defined]
        factory.__helix_agent__ = True  # type: ignore[attr-defined]
        return factory

    return decorator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drop_self(annotations: dict) -> dict:
    """Remove 'self' from an annotations dict."""
    return {k: v for k, v in annotations.items() if k != "self"}
