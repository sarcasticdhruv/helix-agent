"""
helix/presets.py

Ready-to-use Agent archetypes.

Each factory returns a fully configured :class:`helix.Agent` with
appropriate role, goal, tools, and defaults so you can start working
in a single line of code.

Usage::

    from helix.presets import web_researcher, coder, data_analyst

    # Web research
    result = helix.run(web_researcher(), "Top AI papers this week")

    # Code generation
    result = helix.run(coder("TypeScript"), "Write a UUID v4 generator")

    # Pipeline via | operator
    research_then_write = web_researcher() | writer()
    result = research_then_write.run_sync("Write a report on quantum computing")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from helix.core.agent import Agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(
    name: str,
    role: str,
    goal: str,
    backstory: str = "",
    tools: list[Any] | None = None,
    model: str | None = None,
    budget_usd: float = 0.50,
    **kwargs: Any,
) -> Agent:
    """Internal factory used by all presets."""
    from helix.config import BudgetConfig, ModelConfig
    from helix.core.agent import Agent

    return Agent(
        name=name,
        role=role,
        goal=goal,
        backstory=backstory,
        tools=tools or [],
        model=ModelConfig(primary=model) if model else ModelConfig(),
        budget=BudgetConfig(budget_usd=budget_usd),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Web / Research
# ---------------------------------------------------------------------------


def web_researcher(
    name: str = "WebResearcher",
    budget_usd: float = 0.50,
    model: str | None = None,
    max_results: int = 5,
    **kwargs: Any,
) -> Agent:
    """
    A web research agent with ``web_search`` and ``fetch_url`` built-in.

    Example::

        researcher = web_researcher(budget_usd=0.25)
        result = helix.run(researcher, "Latest breakthroughs in LLM alignment")
    """
    import helix.tools.builtin as _bt  # registers tools globally

    return _make_agent(
        name=name,
        role="Senior Web Research Analyst",
        goal=(
            "Find accurate, up-to-date information from the web. "
            "Always cite your sources with URLs. "
            f"Aim for depth — use up to {max_results} search results and follow relevant links."
        ),
        backstory=(
            "You have years of experience as an investigative researcher. "
            "You are thorough, sceptical of unsupported claims, and always verify "
            "information across multiple sources before drawing conclusions."
        ),
        tools=[_bt.web_search, _bt.fetch_url],
        model=model,
        budget_usd=budget_usd,
        **kwargs,
    )


def writer(
    name: str = "Writer",
    style: str = "clear, engaging prose",
    budget_usd: float = 0.30,
    model: str | None = None,
    **kwargs: Any,
) -> Agent:
    """
    A writing agent that turns research or bullet points into polished text.

    Example::

        result = helix.run(writer(style="technical blog post"), research_output)
    """
    return _make_agent(
        name=name,
        role="Professional Content Writer",
        goal=(
            f"Transform the provided information into well-structured, {style}. "
            "Organise content with clear headings, accurate facts, and a compelling narrative."
        ),
        backstory=(
            "You are an award-winning content writer with expertise in translating "
            "complex topics into accessible, engaging articles."
        ),
        model=model,
        budget_usd=budget_usd,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Code
# ---------------------------------------------------------------------------


def coder(
    name: str = "Coder",
    language: str = "Python",
    budget_usd: float = 1.00,
    model: str | None = None,
    allow_file_io: bool = True,
    **kwargs: Any,
) -> Agent:
    """
    A code-generation and debugging agent.

    Example::

        result = helix.run(coder("TypeScript"), "Write a rate-limiter middleware")
    """
    import helix.tools.builtin as _bt

    tools = [_bt.read_file, _bt.write_file] if allow_file_io else []

    return _make_agent(
        name=name,
        role=f"Expert {language} Engineer",
        goal=(
            f"Write clean, tested, production-ready {language} code. "
            "Follow established conventions, add docstrings, and handle edge cases. "
            "When appropriate, provide usage examples."
        ),
        backstory=(
            f"You are a senior {language} engineer with 10+ years of experience. "
            "You write elegant, well-documented code, prefer simple solutions over clever ones, "
            "and always consider performance, security, and maintainability."
        ),
        tools=tools,
        model=model,
        budget_usd=budget_usd,
        **kwargs,
    )


def code_reviewer(
    name: str = "CodeReviewer",
    language: str = "Python",
    budget_usd: float = 0.50,
    model: str | None = None,
    **kwargs: Any,
) -> Agent:
    """
    A code review agent that identifies bugs, style issues, and security risks.

    Example::

        result = helix.run(code_reviewer("Python"), open("my_module.py").read())
    """
    import helix.tools.builtin as _bt

    return _make_agent(
        name=name,
        role=f"Principal {language} Code Reviewer",
        goal=(
            f"Thoroughly review the provided {language} code. "
            "Identify bugs, security vulnerabilities, performance issues, and style problems. "
            "Provide actionable, prioritised feedback with specific line references."
        ),
        backstory=(
            "You have reviewed thousands of pull requests across multiple codebases. "
            "You are constructive, precise, and focus on issues that actually matter."
        ),
        tools=[_bt.read_file],
        model=model,
        budget_usd=budget_usd,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Data / Analysis
# ---------------------------------------------------------------------------


def data_analyst(
    name: str = "DataAnalyst",
    budget_usd: float = 1.00,
    model: str | None = None,
    **kwargs: Any,
) -> Agent:
    """
    A data analysis agent with calculator and file-reading tools.

    Example::

        result = helix.run(data_analyst(), "Analyse sales data in data/q1.csv")
    """
    import helix.tools.builtin as _bt

    return _make_agent(
        name=name,
        role="Senior Data Analyst",
        goal=(
            "Analyse data to extract actionable insights. "
            "Use statistical reasoning, identify trends, anomalies, and patterns. "
            "Present findings clearly with numbers and concrete recommendations."
        ),
        backstory=(
            "You are a data scientist with expertise in statistics and business intelligence. "
            "You prefer evidence-based conclusions and always quantify uncertainty."
        ),
        tools=[_bt.calculator, _bt.read_file],
        model=model,
        budget_usd=budget_usd,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# API / Integration
# ---------------------------------------------------------------------------


def api_agent(
    name: str = "APIAgent",
    base_url: str = "",
    auth_token: str = "",
    budget_usd: float = 0.50,
    model: str | None = None,
    **kwargs: Any,
) -> Agent:
    """
    An agent configured for REST API interaction via ``fetch_url``.

    Example::

        agent = api_agent(base_url="https://api.github.com", auth_token=TOKEN)
        result = helix.run(agent, "List my open pull requests")
    """
    import helix.tools.builtin as _bt

    goal = (
        "Make HTTP requests to the target API and process the results. "
        "Parse JSON responses carefully, handle errors gracefully, "
        "and present the information in a human-readable format."
    )
    if base_url:
        goal = f"Work with the REST API at {base_url}. " + goal

    return _make_agent(
        name=name,
        role="REST API Orchestrator",
        goal=goal,
        backstory=(
            "You are an expert at working with REST APIs. "
            "You understand HTTP methods, status codes, authentication patterns, "
            "pagination, and rate limiting."
        ),
        tools=[_bt.fetch_url],
        model=model,
        budget_usd=budget_usd,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# General purpose
# ---------------------------------------------------------------------------


def assistant(
    name: str = "Assistant",
    domain: str = "general knowledge",
    budget_usd: float = 0.25,
    model: str | None = None,
    **kwargs: Any,
) -> Agent:
    """
    A general-purpose assistant for the given domain.

    Example::

        agent = assistant(domain="Python programming")
        result = helix.run(agent, "What is the GIL?")
    """
    return _make_agent(
        name=name,
        role=f"{domain.title()} Expert",
        goal=(
            f"Provide accurate, helpful answers about {domain}. "
            "Be concise when the question is simple; go deep when it requires it. "
            "Acknowledge uncertainty rather than guessing."
        ),
        model=model,
        budget_usd=budget_usd,
        **kwargs,
    )


def summariser(
    name: str = "Summariser",
    style: str = "concise bullet points",
    budget_usd: float = 0.20,
    model: str | None = None,
    **kwargs: Any,
) -> Agent:
    """
    Compress long text into a summary.

    Example::

        pipeline = web_researcher() | summariser()
        result = pipeline.run_sync("Summarise current state of fusion energy")
    """
    return _make_agent(
        name=name,
        role="Expert Summariser",
        goal=(
            f"Compress the provided text into {style}. "
            "Preserve key facts, numbers, and conclusions. "
            "Omit filler without losing meaning."
        ),
        model=model,
        budget_usd=budget_usd,
        **kwargs,
    )


def fact_checker(
    name: str = "FactChecker",
    budget_usd: float = 0.50,
    model: str | None = None,
    **kwargs: Any,
) -> Agent:
    """
    Checks claims against web sources.

    Example::

        pipeline = writer() | fact_checker()
    """
    import helix.tools.builtin as _bt

    return _make_agent(
        name=name,
        role="Investigative Fact Checker",
        goal=(
            "Verify each factual claim in the provided text using web searches. "
            "Label each claim as VERIFIED, UNVERIFIED, or FALSE. "
            "Provide source URLs for each verdict."
        ),
        tools=[_bt.web_search],
        model=model,
        budget_usd=budget_usd,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Convenience alias
# ---------------------------------------------------------------------------

researcher = web_researcher  # alias: helix.presets.researcher(...)

__all__ = [
    "web_researcher",
    "researcher",
    "writer",
    "summariser",
    "fact_checker",
    "coder",
    "code_reviewer",
    "data_analyst",
    "api_agent",
    "assistant",
]
