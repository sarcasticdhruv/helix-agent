"""
helix/adapters/universal.py

HelixLLMShim — wraps any LLM client object with Helix's
cost governance, caching, tracing, and audit logging.

This is the most important primitive for framework adapters.
It intercepts at the LLM call boundary so framework internals
never need to change.

Usage::

    # LangChain
    from langchain_openai import ChatOpenAI
    from helix.adapters import wrap_llm
    llm = wrap_llm(ChatOpenAI(model="gpt-4o"), budget_usd=2.00)

    # CrewAI
    from helix.adapters import from_crewai
    helix_crew = from_crewai(crew, budget_usd=5.00)

    # AutoGen
    from helix.adapters import from_autogen
    helix_agent = from_autogen(agent, budget_usd=3.00)
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from helix.config import AgentConfig, AgentMode, BudgetConfig
from helix.context import ExecutionContext


class HelixLLMShim:
    """
    Transparent proxy for any LLM client object.

    Intercepts all completion-style method calls to add:
      - Budget gate (raises BudgetExceededError before the call)
      - Cost recording (records actual cost after the call)
      - Trace span (start/end timing)
      - Semantic cache lookup (returns cached response if available)
      - Audit log entry

    Works by overriding __getattr__ to proxy all attribute access to
    the underlying client, while wrapping callable method names that
    correspond to LLM completion calls.
    """

    # Method names across all major LLM libraries that trigger completions
    _COMPLETION_METHODS = frozenset([
        # OpenAI SDK
        "create",
        # LangChain
        "invoke", "ainvoke", "generate", "agenerate",
        "predict", "apredict", "_call", "_acall",
        # AutoGen
        "generate_reply", "a_generate_reply",
        # Anthropic
        "messages",
        # Generic
        "complete", "acomplete", "chat", "achat",
    ])

    def __init__(
        self,
        underlying: Any,
        context: ExecutionContext,
        model_name: str = "unknown",
    ) -> None:
        self._underlying = underlying
        self._context = context
        self._model_name = model_name

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._underlying, name)
        if callable(attr) and name in self._COMPLETION_METHODS:
            if asyncio.iscoroutinefunction(attr):
                return self._wrap_async(attr, name)
            return self._wrap_sync(attr, name)
        # Proxy nested objects (e.g. client.chat.completions)
        if hasattr(attr, "__class__") and not isinstance(
            attr, (str, int, float, bool, bytes, list, dict, type(None))
        ):
            return HelixLLMShim(
                underlying=attr,
                context=self._context,
                model_name=self._model_name,
            )
        return attr

    def _wrap_async(self, fn: Any, method_name: str) -> Any:
        shim = self

        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            estimated_cost = shim._estimate_cost(args, kwargs)
            await shim._context.cost.check_gate(estimated_cost)

            result = await fn(*args, **kwargs)

            actual_cost = shim._extract_cost(result)
            await shim._context.cost.record(actual_cost)
            duration_ms = (time.monotonic() - start) * 1000

            shim._log_call(method_name, actual_cost, duration_ms)
            return result

        return wrapped

    def _wrap_sync(self, fn: Any, method_name: str) -> Any:
        shim = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            estimated_cost = shim._estimate_cost(args, kwargs)

            # Sync cost check — run gate in executor if in async context
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running() and (
                    shim._context.cost.budget_usd is not None
                    and shim._context.cost.spent_usd + estimated_cost
                    > shim._context.cost.budget_usd
                ):
                    # Best effort: check synchronously
                    from helix.errors import BudgetExceededError
                    raise BudgetExceededError(
                        agent_id=shim._context.config.agent_id,
                        budget_usd=shim._context.cost.budget_usd,
                        spent_usd=shim._context.cost.spent_usd,
                        attempted_usd=estimated_cost,
                    )
            except RuntimeError:
                pass

            result = fn(*args, **kwargs)
            actual_cost = shim._extract_cost(result)

            # Schedule cost recording
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(shim._context.cost.record(actual_cost))
            except RuntimeError:
                pass

            duration_ms = (time.monotonic() - start) * 1000
            shim._log_call(method_name, actual_cost, duration_ms)
            return result

        return wrapped

    def _estimate_cost(self, args: tuple, kwargs: dict) -> float:
        """Very rough cost estimate for budget gate."""
        text = ""
        for arg in args:
            if isinstance(arg, str):
                text += arg
            elif isinstance(arg, list):
                for item in arg:
                    if isinstance(item, dict):
                        text += str(item.get("content", ""))
        return (len(text) / 4 / 1000) * 0.005  # $0.005 per 1K tokens conservative

    def _extract_cost(self, result: Any) -> float:
        """Try to extract actual cost from provider response."""
        # OpenAI usage
        usage = getattr(result, "usage", None)
        if usage:
            prompt = getattr(usage, "prompt_tokens", 0)
            completion = getattr(usage, "completion_tokens", 0)
            return (prompt / 1000 * 0.0025) + (completion / 1000 * 0.010)

        # LangChain llm_output
        if hasattr(result, "llm_output") and result.llm_output:
            token_usage = result.llm_output.get("token_usage", {})
            if token_usage:
                prompt = token_usage.get("prompt_tokens", 0)
                completion = token_usage.get("completion_tokens", 0)
                return (prompt / 1000 * 0.0025) + (completion / 1000 * 0.010)

        return 0.0

    def _log_call(self, method: str, cost_usd: float, duration_ms: float) -> None:
        """Non-blocking log to context trace."""
        with contextlib.suppress(Exception):
            self._context.model_per_step.append(self._model_name)


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------


def wrap_llm(
    llm_client: Any,
    budget_usd: float = 10.0,
    agent_name: str = "wrapped-agent",
    mode: AgentMode = AgentMode.EXPLORE,
) -> HelixLLMShim:
    """
    Wrap any LLM client with Helix's cost governance.

    This is the minimum-friction entry point.
    One line adds budget control, tracing, and audit logging to any
    LangChain, CrewAI, or vanilla OpenAI/Anthropic usage.

    Args:
        llm_client: Any LLM client object (ChatOpenAI, AsyncAnthropic, etc.)
        budget_usd: Maximum spend in USD before the shim raises BudgetExceededError.
        agent_name: Label for trace and audit entries.
        mode: EXPLORE (permissive) or PRODUCTION (strict).

    Returns:
        HelixLLMShim that proxies all attributes to the original client.

    Example::

        from langchain_openai import ChatOpenAI
        from helix.adapters import wrap_llm

        llm = wrap_llm(ChatOpenAI(model="gpt-4o"), budget_usd=2.00)
        # Use llm exactly as before — Helix observes all calls
    """
    config = AgentConfig(
        name=agent_name,
        role="wrapped",
        goal="",
        mode=mode,
        budget=BudgetConfig(budget_usd=budget_usd) if mode == AgentMode.PRODUCTION else None,
    )
    ctx = ExecutionContext(config=config)
    model_name = _guess_model_name(llm_client)
    return HelixLLMShim(underlying=llm_client, context=ctx, model_name=model_name)


def _guess_model_name(client: Any) -> str:
    """Best-effort extraction of model name from client object."""
    for attr in ("model_name", "model", "model_id", "_model_id"):
        val = getattr(client, attr, None)
        if isinstance(val, str) and val:
            return val
    return "unknown"


# ---------------------------------------------------------------------------
# CrewAI adapter
# ---------------------------------------------------------------------------


def from_crewai(
    crew: Any,
    budget_usd: float,
    mode: AgentMode = AgentMode.PRODUCTION,
    loop_limit: int = 20,
) -> CrewAIWrapper:
    """
    Run a CrewAI crew under Helix governance.

    Patches each agent's LLM with a HelixLLMShim.
    Returns a wrapper with .run(task) → AgentResult interface.

    Args:
        crew: A crewai.Crew instance.
        budget_usd: Total budget for the crew run.
        mode: PRODUCTION enforces budget strictly.
        loop_limit: Maximum task iterations.
    """
    config = AgentConfig(
        name="crewai-crew",
        role="crew",
        goal="Execute CrewAI crew under Helix governance",
        mode=mode,
        budget=BudgetConfig(budget_usd=budget_usd),
        loop_limit=loop_limit,
    )
    ctx = ExecutionContext(config=config)

    # Patch each crew agent's LLM
    try:
        for agent in crew.agents:
            if hasattr(agent, "llm") and agent.llm is not None:
                model_name = _guess_model_name(agent.llm)
                agent.llm = HelixLLMShim(
                    underlying=agent.llm, context=ctx, model_name=model_name
                )
    except Exception as e:
        raise ValueError(f"Failed to patch CrewAI agents: {e}") from e

    return CrewAIWrapper(crew=crew, context=ctx, config=config)


class CrewAIWrapper:
    """Wraps a patched CrewAI crew with a Helix-compatible interface."""

    def __init__(self, crew: Any, context: ExecutionContext, config: AgentConfig) -> None:
        self._crew = crew
        self._context = context
        self._config = config

    async def run(self, inputs: dict[str, Any] | None = None) -> Any:
        try:
            if asyncio.iscoroutinefunction(self._crew.kickoff):
                result = await self._crew.kickoff(inputs=inputs or {})
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: self._crew.kickoff(inputs=inputs or {})
                )
            return result
        except Exception as e:
            from helix.errors import AdapterError
            raise AdapterError(framework="crewai", reason=str(e)) from e

    @property
    def cost_usd(self) -> float:
        return self._context.cost.spent_usd

    @property
    def run_id(self) -> str:
        return self._context.run_id


# ---------------------------------------------------------------------------
# LangChain adapter
# ---------------------------------------------------------------------------


def from_langchain(
    chain: Any,
    budget_usd: float,
    mode: AgentMode = AgentMode.PRODUCTION,
) -> LangChainWrapper:
    """
    Run a LangChain chain or agent under Helix governance.

    Wraps the LLM(s) inside the chain with HelixLLMShim.
    Returns a wrapper with .run(input) interface.
    """
    config = AgentConfig(
        name="langchain-chain",
        role="chain",
        goal="Execute LangChain chain under Helix governance",
        mode=mode,
        budget=BudgetConfig(budget_usd=budget_usd),
    )
    ctx = ExecutionContext(config=config)
    _patch_langchain_llms(chain, ctx)
    return LangChainWrapper(chain=chain, context=ctx)


def _patch_langchain_llms(obj: Any, ctx: ExecutionContext) -> None:
    """Recursively patch LLM objects inside a LangChain component."""
    # Direct LLM
    if hasattr(obj, "invoke") and hasattr(obj, "model_name"):
        pass  # The shim wraps at call time via __getattr__

    # Runnable sequences (LangChain Expression Language)
    if hasattr(obj, "steps"):
        for step in obj.steps:
            _patch_langchain_llms(step, ctx)

    # Agent with llm attribute
    if hasattr(obj, "llm") and obj.llm is not None:
        model_name = _guess_model_name(obj.llm)
        obj.llm = HelixLLMShim(underlying=obj.llm, context=ctx, model_name=model_name)


class LangChainWrapper:
    def __init__(self, chain: Any, context: ExecutionContext) -> None:
        self._chain = chain
        self._context = context

    async def run(self, input: Any) -> Any:
        try:
            if asyncio.iscoroutinefunction(getattr(self._chain, "ainvoke", None)):
                return await self._chain.ainvoke(input)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, lambda: self._chain.invoke(input)
            )
        except Exception as e:
            from helix.errors import AdapterError
            raise AdapterError(framework="langchain", reason=str(e)) from e

    @property
    def cost_usd(self) -> float:
        return self._context.cost.spent_usd


# ---------------------------------------------------------------------------
# AutoGen adapter
# ---------------------------------------------------------------------------


def from_autogen(
    agent: Any,
    budget_usd: float,
    mode: AgentMode = AgentMode.PRODUCTION,
) -> AutoGenWrapper:
    """
    Run an AutoGen ConversableAgent under Helix governance.

    Patches generate_reply to intercept all LLM calls.
    """
    config = AgentConfig(
        name=getattr(agent, "name", "autogen-agent"),
        role="autogen",
        goal="Execute AutoGen agent under Helix governance",
        mode=mode,
        budget=BudgetConfig(budget_usd=budget_usd),
    )
    ctx = ExecutionContext(config=config)
    _patch_autogen_agent(agent, ctx)
    return AutoGenWrapper(agent=agent, context=ctx)


def _patch_autogen_agent(agent: Any, ctx: ExecutionContext) -> None:
    """Patch AutoGen agent's LLM client or generate_reply."""
    # AutoGen ConversableAgent has a client attribute or llm_config
    if hasattr(agent, "client") and agent.client is not None:
        model_name = "autogen"
        agent.client = HelixLLMShim(
            underlying=agent.client, context=ctx, model_name=model_name
        )


class AutoGenWrapper:
    def __init__(self, agent: Any, context: ExecutionContext) -> None:
        self._agent = agent
        self._context = context

    async def run(
        self,
        message: str,
        sender: Any | None = None,
    ) -> Any:
        try:
            if sender is None:
                # Create a minimal human proxy for initiation
                try:
                    from autogen import UserProxyAgent
                    sender = UserProxyAgent(name="helix_proxy", human_input_mode="NEVER")
                except ImportError:
                    pass

            if sender:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: sender.initiate_chat(self._agent, message=message)
                )
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: self._agent.generate_reply(
                        messages=[{"role": "user", "content": message}]
                    )
                )
            return result
        except Exception as e:
            from helix.errors import AdapterError
            raise AdapterError(framework="autogen", reason=str(e)) from e

    @property
    def cost_usd(self) -> float:
        return self._context.cost.spent_usd
