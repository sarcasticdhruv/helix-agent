"""
helix/core/agent.py

The Agent class. The developer-facing runtime object.

Design:
  - Agent is a thin orchestrator. It holds config and delegates
    every concern to the appropriate subsystem.
  - AgentConfig is validated at construction. No silent defaults.
  - run() is the primary entry point. Async-first; run_sync() wraps it.
  - The reasoning loop is minimal: system prompt → LLM call →
    tool dispatch → record → repeat until done or limit hit.
  - Every concern (cost, safety, memory, cache) is delegated, never inline.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from helix.config import (
    AgentConfig,
    AgentMode,
    BudgetConfig,
    CacheConfig,
    EpisodeOutcome,
    MemoryConfig,
    ModelConfig,
    ObservabilityConfig,
    PermissionConfig,
    StructuredOutputConfig,
    TokenUsage,
    ToolCallRecord,
)
from helix.context import ExecutionContext
from helix.core.hooks import HookEvent, HookFn
from helix.core.hooks import fire as _fire_hook
from helix.core.tool import ToolRegistry, execute_tool
from helix.core.tool import registry as _global_registry
from helix.errors import (
    BudgetExceededError,
    HelixError,
    LoopDetectedError,
)

# ---------------------------------------------------------------------------
# AgentResult
# ---------------------------------------------------------------------------


class AgentResult(BaseModel):
    """The return value of Agent.run()."""

    output: Any  # str or typed model if structured_output enabled
    steps: int
    cost_usd: float
    run_id: str
    agent_id: str
    agent_name: str
    duration_s: float
    tool_calls: int
    cache_hits: int
    cache_savings_usd: float
    episodes_used: int = 0
    model_used: str | None = None  # Which model was actually called
    error: str | None = None
    trace: dict[str, Any] | None = None  # Populated if observability enabled
    tool_call_records: list[ToolCallRecord] = Field(
        default_factory=list
    )  # Full records; `tool_calls` above stays an int for backward compat
    handoff_chain: list[str] = Field(
        default_factory=list
    )  # Agent names this run passed through, in order

    model_config = ConfigDict(frozen=True)

    @property
    def success(self) -> bool:
        """True if the run completed without error. Prefer this over string-matching `.output`."""
        return self.error is None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    """
    A Helix agent.

    Instantiation validates config. Calling run(task) executes the
    full reasoning loop under cost governance, context management,
    memory recall, caching, and safety.

    Example::

        agent = Agent(
            name="Researcher",
            role="Information gatherer",
            goal="Find accurate, cited answers.",
            model=ModelConfig(),
            budget=BudgetConfig(budget_usd=0.50),
            mode=AgentMode.PRODUCTION,
        )
        result = await agent.run("What is the capital of France?")
    """

    def __init__(
        self,
        name: str,
        role: str,
        goal: str,
        backstory: str = "",
        model: ModelConfig | None = None,
        budget: BudgetConfig | None = None,
        mode: AgentMode = AgentMode.EXPLORE,
        tools: list[Any] | None = None,
        memory: MemoryConfig | None = None,
        cache: CacheConfig | None = None,
        permissions: PermissionConfig | None = None,
        structured_output: StructuredOutputConfig | None = None,
        observability: ObservabilityConfig | None = None,
        system_prompt: str | None = None,
        on_event: HookFn | None = None,
        inherit_global_tools: bool = False,
        handoffs: list[Agent] | None = None,
        **extra_config: Any,
    ) -> None:
        self._config = AgentConfig(
            name=name,
            role=role,
            goal=goal,
            backstory=backstory,
            mode=mode,
            model=model or ModelConfig(),
            budget=budget,
            memory=memory or MemoryConfig(),
            cache=cache or CacheConfig(),
            permissions=permissions or PermissionConfig(),
            structured_output=structured_output or StructuredOutputConfig(),
            observability=observability or ObservabilityConfig(),
            system_prompt_override=system_prompt,
            # Forwards fields like guardrails=[...], hitl=HITLConfig(...),
            # tenant_id=..., loop_limit=... — previously accepted here and
            # silently discarded without ever reaching AgentConfig.
            **extra_config,
        )

        # Tool registry: per-agent by default (secure by default — a bare
        # `import helix` registers 13 builtins, including execute_python and
        # write_file, into the global registry; agents must opt in to inherit
        # them explicitly rather than getting them for free).
        self._registry = ToolRegistry()
        if inherit_global_tools:
            for t in _global_registry.all():
                self._registry.register(t)
        # Register agent-specific tools
        if tools:
            for t in tools:
                self._registry.register(t)

        # Handoffs — each target gets a transfer_to_<name> tool the model
        # can call; Agent._reasoning_loop recognizes it and Agent._execute
        # delegates to the target instead of finalizing this agent's result.
        self._handoffs: dict[str, Agent] = {}
        if handoffs:
            from helix.core.handoff import handoff_tool_name, make_handoff_tool

            for target in handoffs:
                self._registry.register(make_handoff_tool(target))
                self._handoffs[handoff_tool_name(target)] = target

        # Subsystems — lazily initialized on first run
        self._memory_store: Any | None = None
        self._cache_controller: Any | None = None
        self._llm_router: Any | None = None
        self._tracer: Any | None = None
        self._cost_governor: Any | None = None
        self._guardrail_chain: Any | None = None
        self._hitl_controller: Any | None = None
        self._context_engine: Any | None = None
        self._audit_log: Any | None = None
        self._embedder: Any | None = None

        self._initialized: bool = False
        self._on_event: HookFn | None = on_event

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        task: str,
        session_id: str | None = None,
        parent_run_id: str | None = None,
        output_schema: type[BaseModel] | dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Execute the agent on the given task.

        Args:
            task: The task string to execute.
            session_id: Tie this run to a session for multi-turn memory.
            parent_run_id: For nested agents / workflows.
            output_schema: Pydantic model or JSON Schema for structured output.

        Returns:
            AgentResult with output, cost, trace, and metadata.
        """
        await self._ensure_initialized()

        ctx = ExecutionContext(
            config=self._config,
            session_id=session_id,
            parent_run_id=parent_run_id,
        )

        # A fresh Tracer per run — Agent instances are reused across many
        # run() calls (Session, Team, GroupChat, evals), so a tracer built
        # once at lazy-init time would leak spans across runs and never
        # carry the right run_id.
        if self._config.observability.trace_enabled:
            from helix.observability.tracer import Tracer

            self._tracer = Tracer(
                run_id=ctx.run_id,
                agent_id=self._config.agent_id,
                agent_name=self._config.name,
            )

        try:
            result = await self._execute(ctx, task, output_schema=output_schema)
        except BudgetExceededError as e:
            result = self._error_result(ctx, str(e))
            await _fire_hook(
                self._on_event,
                HookEvent(type="error", data={"error": str(e)}, cost_so_far=ctx.cost.spent_usd),
            )
        except LoopDetectedError as e:
            result = self._error_result(ctx, str(e))
            await _fire_hook(
                self._on_event,
                HookEvent(type="error", data={"error": str(e)}, cost_so_far=ctx.cost.spent_usd),
            )
        except HelixError as e:
            result = self._error_result(ctx, str(e))
            await _fire_hook(
                self._on_event,
                HookEvent(type="error", data={"error": str(e)}, cost_so_far=ctx.cost.spent_usd),
            )
        except Exception as e:
            msg = f"Unexpected error: {e}"
            result = self._error_result(ctx, msg)
            await _fire_hook(
                self._on_event,
                HookEvent(type="error", data={"error": msg}, cost_so_far=ctx.cost.spent_usd),
            )
        finally:
            await self._finalize(ctx)

        return result

    def run_sync(self, task: str, **kwargs: Any) -> AgentResult:
        """Synchronous wrapper for environments without an event loop."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # In Jupyter or nested async context
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, self.run(task, **kwargs))
                    return future.result()
            return loop.run_until_complete(self.run(task, **kwargs))
        except RuntimeError:
            return asyncio.run(self.run(task, **kwargs))

    # ------------------------------------------------------------------
    # LangChain-compatible aliases
    # ------------------------------------------------------------------

    async def ainvoke(
        self,
        task: str,
        session_id: str | None = None,
        output_schema: Any | None = None,
    ) -> AgentResult:
        """
        Async alias for :meth:`run`.  Matches the LangChain ``ainvoke`` API.

        Example::

            result = await agent.ainvoke("Summarise quarterly earnings.")
        """
        return await self.run(task, session_id=session_id, output_schema=output_schema)

    def invoke(
        self,
        task: str,
        session_id: str | None = None,
        output_schema: Any | None = None,
    ) -> AgentResult:
        """
        Sync alias for :meth:`run_sync`.  Matches the LangChain ``invoke`` API.

        Example::

            result = agent.invoke("Summarise quarterly earnings.")
            print(result.output)
        """
        return self.run_sync(task, session_id=session_id, output_schema=output_schema)

    async def stream(
        self,
        task: str,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream response tokens as they arrive from the LLM.
        Does not support tool calls in streaming mode.
        """
        await self._ensure_initialized()
        ctx = ExecutionContext(config=self._config, session_id=session_id)
        await self._build_context(ctx, task)

        messages = ctx.window.as_llm_messages()
        model = ctx.effective_model()

        async for chunk in self._llm_router.stream(messages=messages, model=model):
            yield chunk

    def __or__(self, other: Agent | Any) -> Any:
        """
        Pipe two agents together.  Output of ``self`` becomes the task for ``other``.

        Example::

            pipeline = researcher | analyst | writer
            result = helix.run(pipeline, "Quantum AI 2026")
        """
        from helix.core.pipeline import AgentPipeline

        if isinstance(other, AgentPipeline):
            return AgentPipeline([self] + other.agents)
        return AgentPipeline([self, other])

    def add_tool(self, tool_or_fn: Any) -> Agent:
        """Register an additional tool. Returns self for chaining."""
        self._registry.register(tool_or_fn)
        return self

    def clone(self, **overrides: Any) -> Agent:
        """
        Create a copy of this agent with config overrides.
        Useful for A/B testing prompt variants or model differences.
        """
        config_data = self._config.model_dump()
        config_data.update(overrides)
        new_agent = Agent.__new__(Agent)
        new_agent._config = AgentConfig(**config_data)
        new_agent._registry = self._registry  # Shared tool registry
        new_agent._handoffs = self._handoffs
        new_agent._initialized = False
        # Subsystems will re-init on first run
        for attr in (
            "_memory_store",
            "_cache_controller",
            "_llm_router",
            "_tracer",
            "_cost_governor",
            "_guardrail_chain",
            "_hitl_controller",
            "_context_engine",
            "_audit_log",
            "_embedder",
        ):
            setattr(new_agent, attr, None)
        return new_agent

    @property
    def config(self) -> AgentConfig:
        return self._config

    @property
    def agent_id(self) -> str:
        return self._config.agent_id

    @property
    def name(self) -> str:
        return self._config.name

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    async def _execute(
        self,
        ctx: ExecutionContext,
        task: str,
        output_schema: Any | None = None,
    ) -> AgentResult:
        # 0. Guardrails previously only ever checked the model's output —
        # nothing screened the incoming task itself (e.g. for prompt
        # injection) before it entered context, memory, and every
        # downstream LLM call.
        task = await self._run_guardrails(ctx, task)

        # 1. Build initial context (system prompt + episodic memory + task)
        await self._build_context(ctx, task)

        # 2. Check semantic cache before any LLM call
        cache_hit = await self._check_cache(ctx, task)
        if cache_hit:
            output = cache_hit.response
            ctx.record_cache_hit(cache_hit.saved_usd)
            await _fire_hook(
                self._on_event,
                HookEvent(
                    type="cache_hit",
                    data={"similarity": cache_hit.similarity, "saved_usd": cache_hit.saved_usd},
                    cost_so_far=0.0,
                    step=0,
                ),
            )
            result = self._build_result(ctx, output, episodes_used=0)
            await _fire_hook(
                self._on_event,
                HookEvent(
                    type="done",
                    data={
                        "output_preview": str(output)[:120],
                        "steps": result.steps,
                        "cost_usd": result.cost_usd,
                    },
                    cost_so_far=result.cost_usd,
                ),
            )
            return result

        # 3. Check plan cache — adapt plan template if available
        plan = await self._check_plan_cache(ctx, task)

        # 4. Reasoning loop
        output = await self._reasoning_loop(ctx, task, plan=plan)

        # A handoff tool was called — this agent is deferring to another
        # agent rather than finalizing its own answer. Delegate and return
        # the target's result (with this agent's cost folded in) instead of
        # building a result from this agent's own (incomplete) output.
        if ctx.handoff_target is not None:
            target = ctx.handoff_target
            handoff_task = self._build_handoff_task(task, output, ctx.handoff_reason)
            await _fire_hook(
                self._on_event,
                HookEvent(
                    type="handoff",
                    data={"target": target.name, "reason": ctx.handoff_reason},
                    cost_so_far=ctx.cost.spent_usd,
                ),
            )
            target_result = await target.run(
                handoff_task, session_id=ctx.session_id, parent_run_id=ctx.run_id
            )
            return target_result.model_copy(
                update={
                    "cost_usd": round(target_result.cost_usd + ctx.cost.spent_usd, 6),
                    "handoff_chain": [self._config.name, *target_result.handoff_chain],
                }
            )

        # 5. Apply structured output if configured
        if output_schema or self._config.structured_output.enabled:
            output = await self._apply_structured_output(ctx, output, output_schema)

        # Record the real savings this plan-cache hit produced, if any.
        if plan is not None and self._cache_controller:
            with contextlib.suppress(Exception):
                self._cache_controller.plan.record_actual_savings(plan, ctx.cost.spent_usd)

        # 6. Store successful plan to plan cache
        await self._store_plan(ctx, task)

        result = self._build_result(ctx, output)
        await _fire_hook(
            self._on_event,
            HookEvent(
                type="done",
                data={
                    "output_preview": str(output)[:120],
                    "steps": result.steps,
                    "cost_usd": result.cost_usd,
                },
                cost_so_far=result.cost_usd,
            ),
        )
        return result

    async def _reasoning_loop(
        self,
        ctx: ExecutionContext,
        task: str,
        plan: Any | None = None,
    ) -> str:
        """
        Core agentic loop:
          1. Prepare messages from context window
          2. Gate cost
          3. Call LLM
          4. Check guardrails on response
          5. Execute any tool calls
          6. Record in context, memory, trace
          7. Repeat until finish_reason == "stop" or loop limit
        """
        registry_view = self._registry.filtered(
            allowed=self._config.permissions.allowed_tools,
            denied=self._config.permissions.denied_tools,
        )
        tool_schemas = registry_view.schemas()
        final_output = ""

        if plan is not None:
            # Give the model a real shortcut: the plan cache's whole point is
            # to skip re-deriving a known-good approach. Without this, a
            # cache "hit" changed nothing about how the agent behaved.
            await ctx.window.add_system(
                "A similar task was solved successfully before. Reuse this "
                f"approach where it applies, adapting details to the current task:\n"
                f"{plan.steps_description}",
                pinned=False,
            )

        while True:
            ctx.window.tick()
            ctx.check_loop()

            # Compact context if approaching limit
            if ctx.window.needs_compaction():
                await self._context_engine.compact(
                    ctx, embedder=self._embedder, llm_router=self._llm_router
                )

            messages = ctx.window.as_llm_messages()
            model = ctx.effective_model()

            await _fire_hook(
                self._on_event,
                HookEvent(
                    type="step_start",
                    data={"step": ctx.window.step},
                    cost_so_far=ctx.cost.spent_usd,
                    step=ctx.window.step,
                ),
            )

            # Cost gate
            estimated_cost = self._estimate_call_cost(messages, model)
            await ctx.cost.check_gate(estimated_cost)
            if self._config.budget:
                await self._warn_budget(ctx)

            await _fire_hook(
                self._on_event,
                HookEvent(
                    type="llm_call",
                    data={"model": model, "messages": len(messages)},
                    cost_so_far=ctx.cost.spent_usd,
                    step=ctx.window.step,
                ),
            )

            # LLM call
            span_cm = (
                self._tracer.span("llm.call", model=model, step=ctx.window.step)
                if self._tracer
                else contextlib.nullcontext()
            )
            with span_cm as llm_span:
                response = await self._llm_router.complete(
                    messages=messages,
                    model=model,
                    tools=tool_schemas if tool_schemas else None,
                    temperature=self._config.model.temperature,
                    max_tokens=self._config.model.max_tokens,
                )
            ctx.model_per_step.append(model)

            # Record actual cost
            call_cost = self._calculate_actual_cost(response.usage, model)
            await ctx.cost.record(call_cost)
            if self._tracer and llm_span is not None:
                llm_span.meta["tokens"] = response.usage.model_dump()
                llm_span.meta["cost_usd"] = call_cost

            await _fire_hook(
                self._on_event,
                HookEvent(
                    type="llm_response",
                    data={
                        "model": model,
                        "tokens": response.usage.total_tokens,
                        "finish_reason": response.finish_reason,
                    },
                    cost_so_far=ctx.cost.spent_usd,
                    step=ctx.window.step,
                ),
            )

            # Audit
            await self._audit(
                "llm_response",
                ctx,
                {
                    "model": model,
                    "tokens": response.usage.model_dump(),
                    "finish_reason": response.finish_reason,
                },
            )

            # Guardrails on response
            cleaned = await self._run_guardrails(ctx, response.content)

            # Store assistant response in context, including the tool calls it
            # requested — providers need this turn replayed verbatim on the
            # next call, or the follow-up request is malformed.
            wire_tool_calls = (
                [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ]
                if response.tool_calls
                else None
            )
            await ctx.window.add_assistant(cleaned, tool_calls=wire_tool_calls)
            ctx.record_step_output(ctx.window.step, cleaned)
            final_output = cleaned

            if self._tracer:
                self._tracer.log_step(step=ctx.window.step, response_content=cleaned, model=model)

            # Multi-factor relevance decay — feeds compaction's compressible-
            # message gate above. Runs every step, per the context engine's design.
            await self._context_engine.update_relevance(ctx, cleaned, embedder=self._embedder)

            # Cache the response for future semantic lookups
            await self._store_semantic_cache(ctx, task, cleaned, ctx.cost.spent_usd)

            await _fire_hook(
                self._on_event,
                HookEvent(
                    type="step_end",
                    data={
                        "step": ctx.window.step,
                        "output_preview": cleaned[:120],
                    },
                    cost_so_far=ctx.cost.spent_usd,
                    step=ctx.window.step,
                ),
            )

            # No tool calls → we're done
            if response.finish_reason == "stop" or not response.tool_calls:
                break

            # Execute tool calls
            handoff_triggered = False
            for tc in response.tool_calls:
                await _fire_hook(
                    self._on_event,
                    HookEvent(
                        type="tool_call",
                        data={"tool_name": tc.tool_name, "args": tc.arguments},
                        cost_so_far=ctx.cost.spent_usd,
                        step=ctx.window.step,
                    ),
                )
                tool_span_cm = (
                    self._tracer.span("tool.call", tool_name=tc.tool_name, step=ctx.window.step)
                    if self._tracer
                    else contextlib.nullcontext()
                )
                with tool_span_cm:
                    record = await execute_tool(
                        registry_view=registry_view,
                        tool_name=tc.tool_name,
                        arguments=tc.arguments,
                        step=ctx.window.step,
                        agent_id=self._config.agent_id,
                    )
                ctx.record_tool_call(record)

                # Fire tool hook
                if record.failure_class is None:
                    await _fire_hook(
                        self._on_event,
                        HookEvent(
                            type="tool_result",
                            data={
                                "tool_name": tc.tool_name,
                                "result_preview": str(record.result)[:120]
                                if record.result is not None
                                else "",
                            },
                            cost_so_far=ctx.cost.spent_usd,
                            step=ctx.window.step,
                        ),
                    )
                else:
                    await _fire_hook(
                        self._on_event,
                        HookEvent(
                            type="tool_error",
                            data={
                                "tool_name": tc.tool_name,
                                "error": str(record.failure_class),
                            },
                            cost_so_far=ctx.cost.spent_usd,
                            step=ctx.window.step,
                        ),
                    )

                # Classify failure and decide recovery
                if record.failure_class is not None:
                    recovery = await self._handle_tool_failure(ctx, record)
                    if recovery == "abort":
                        break

                result_content = (
                    str(record.result)
                    if record.result is not None
                    else f"[Tool {tc.tool_name} failed: {record.failure_class}]"
                )
                await ctx.window.add_tool_result(tc.tool_name, result_content, tool_call_id=tc.id)

                # Memory: auto-store important tool results
                await self._maybe_store_memory(ctx, tc.tool_name, result_content)

                if tc.tool_name in self._handoffs and record.failure_class is None:
                    ctx.handoff_target = self._handoffs[tc.tool_name]
                    ctx.handoff_reason = (
                        record.result.get("reason", "") if isinstance(record.result, dict) else ""
                    )
                    handoff_triggered = True

            if handoff_triggered:
                break

        return final_output

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------

    async def _build_context(self, ctx: ExecutionContext, task: str) -> None:
        """Build the initial context window for a run."""
        system_prompt = await self._build_system_prompt(ctx, task)
        await ctx.window.add_system(system_prompt, pinned=True)
        await ctx.window.add_user(task)

    async def _build_system_prompt(self, ctx: ExecutionContext, task: str) -> str:
        """
        Compose the system prompt from:
          1. Role/goal base
          2. Episodic memory (past similar tasks)
          3. Recent short-term memory
          4. Mode-specific instructions
        """
        cfg = self._config

        # Base
        lines = [
            f"You are {cfg.name}, a {cfg.role}.",
            f"Goal: {cfg.goal}",
        ]

        # Backstory (adds rich character context like CrewAI)
        if cfg.backstory:
            lines.append(f"Background: {cfg.backstory}")

        # System prompt override or registry lookup
        if cfg.system_prompt_override:
            lines.append(cfg.system_prompt_override)

        # Episodic memory injection
        episodes_used = 0
        if self._memory_store:
            try:
                task_embedding = await self._memory_store.embed(task)
                episodes = await self._memory_store.backend.search_episodes(
                    query_embedding=task_embedding,
                    top_k=3,
                )
                if episodes:
                    episodes_used = len(episodes)
                    lines.append("\n[Past Experience — use to inform your approach]")
                    for ep in episodes:
                        icon = "✓" if ep.outcome.value == "success" else "✗"
                        lines.append(
                            f"{icon} Task: '{ep.task[:80]}'\n"
                            f"   Steps: {ep.steps}, Cost: ${ep.cost_usd:.4f}"
                        )
                        if ep.failure_reason:
                            lines.append(f"   Failed because: {ep.failure_reason}")
                        if ep.learned_strategy:
                            lines.append(f"   Better approach: {ep.learned_strategy}")
                ctx.episodes_used = episodes_used  # type: ignore[attr-defined]
            except Exception:
                pass  # Memory failure is non-fatal

        # Recent memory
        if self._memory_store:
            try:
                recent = self._memory_store.recent_str(n=5)
                if recent:
                    lines.append(f"\n[Recent Context]\n{recent}")
            except Exception:
                pass

        # Mode instructions
        if cfg.mode == AgentMode.PRODUCTION:
            lines.append(
                "\nOperate carefully. Prefer tool calls you are certain about. "
                "State uncertainty rather than guessing."
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    async def _check_cache(self, ctx: ExecutionContext, task: str) -> Any | None:
        if self._cache_controller is None:
            return None
        try:
            return await self._cache_controller.lookup(
                query=task,
                context_hash=ctx.window.context_hash(),
            )
        except Exception:
            return None

    async def _check_plan_cache(self, ctx: ExecutionContext, task: str) -> Any | None:
        if self._cache_controller is None:
            return None
        try:
            return await self._cache_controller.plan.match(task)
        except Exception:
            return None

    async def _store_semantic_cache(
        self, ctx: ExecutionContext, task: str, response: str, cost_usd: float
    ) -> None:
        if self._cache_controller is None:
            return
        with contextlib.suppress(Exception):
            await self._cache_controller.semantic.set(
                query=task,
                context_hash=ctx.window.context_hash(),
                response=response,
                cost_usd=cost_usd,
            )

    async def _store_plan(self, ctx: ExecutionContext, task: str) -> None:
        if self._cache_controller is None:
            return
        with contextlib.suppress(Exception):
            await self._cache_controller.plan.store(task, ctx)

    def _build_handoff_task(self, original_task: str, last_output: str, reason: str) -> str:
        """Compose the task string the target agent receives on a handoff."""
        parts = [f"[Handed off from {self._config.name}]"]
        if reason:
            parts.append(f"Reason: {reason}")
        parts.append(f"Original task: {original_task}")
        if last_output:
            parts.append(f"Context from {self._config.name}: {last_output}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Safety helpers
    # ------------------------------------------------------------------

    async def _run_guardrails(self, ctx: ExecutionContext, content: str) -> str:
        if self._guardrail_chain is None:
            return content
        result_content = content
        for guardrail in self._guardrail_chain:
            result = await guardrail.check(result_content, ctx)
            if not result.passed:
                from helix.errors import GuardrailViolationError

                await self._audit(
                    "guardrail_block",
                    ctx,
                    {
                        "guardrail": guardrail.name,
                        "reason": result.reason,
                    },
                )
                raise GuardrailViolationError(
                    guardrail_name=guardrail.name,
                    reason=result.reason or "Content blocked",
                    content_preview=content[:100],
                )
            if result.modified_content:
                result_content = result.modified_content
        return result_content

    async def _handle_tool_failure(self, ctx: ExecutionContext, record: Any) -> str:
        """
        Classify failure and apply recovery strategy.
        Returns "continue" or "abort".
        """
        from helix.tools.taxonomy import RECOVERY_STRATEGIES, EscalateStrategy

        strategy = RECOVERY_STRATEGIES.get(record.failure_class)
        if strategy is None:
            return "continue"

        if isinstance(strategy, EscalateStrategy) and self._hitl_controller:
            from helix.config import HITLRequest

            req = HITLRequest(
                agent_id=self._config.agent_id,
                prompt=f"Tool '{record.tool_name}' failed ({record.failure_class}). Continue?",
                risk_level="high",
            )
            response = await self._hitl_controller.send_request(req)
            from helix.config import HITLDecision

            if response.decision == HITLDecision.REJECT:
                return "abort"

        return "continue"

    async def _warn_budget(self, ctx: ExecutionContext) -> None:
        """Emit a warning when budget threshold is crossed."""
        if not self._config.budget:
            return
        pct = ctx.cost.budget_pct
        if pct and pct >= self._config.budget.warn_at_pct:
            await self._audit(
                "budget_warning",
                ctx,
                {
                    "spent_usd": ctx.cost.spent_usd,
                    "budget_usd": ctx.cost.budget_usd,
                    "pct": pct,
                },
            )

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    async def _maybe_store_memory(
        self, ctx: ExecutionContext, tool_name: str, content: str
    ) -> None:
        """Heuristically determine if a tool result should be stored in memory."""
        if self._memory_store is None:
            return
        if len(content) < 20:
            return
        importance = 0.6
        try:
            from helix.config import MemoryEntry, MemoryKind

            entry = MemoryEntry(
                content=f"[{tool_name}] {content[:500]}",
                kind=MemoryKind.TOOL_RESULT,
                importance=importance,
                agent_id=self._config.agent_id,
            )
            await self._memory_store.add(entry)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Cost helpers
    # ------------------------------------------------------------------

    def _estimate_call_cost(self, messages: list[dict], model: str) -> float:
        """
        Pre-call cost estimate for the budget gate, using the same
        per-model pricing table as the post-call actual-cost calculation
        (not a flat blended rate). Completion tokens are estimated at
        the configured max_tokens — the worst case for this call — since
        this is a hard budget *gate*: it should never wave through a call
        that could exceed budget, even if most calls finish well short of
        their max_tokens ceiling.
        """
        prompt_tokens = sum(len(str(m.get("content", "") or "")) // 4 for m in messages)
        estimated_usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=self._config.model.max_tokens,
        )
        if self._llm_router:
            return self._llm_router.calculate_cost(estimated_usage, model)
        return (prompt_tokens / 1000) * 0.005

    def _calculate_actual_cost(self, usage: Any, model: str) -> float:
        """Calculate actual cost from token usage."""
        if self._llm_router:
            return self._llm_router.calculate_cost(usage, model)
        return 0.0

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    async def _audit(self, event: str, ctx: ExecutionContext, details: dict) -> None:
        if self._audit_log is None:
            return
        try:
            from helix.config import AuditEntry, AuditEventType

            entry = AuditEntry(
                event_type=AuditEventType(event),
                agent_id=self._config.agent_id,
                session_id=ctx.session_id,
                tenant_id=self._config.tenant_id,
                details=details,
            )
            await self._audit_log.append(entry)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------

    async def _apply_structured_output(
        self,
        ctx: ExecutionContext,
        raw_output: str,
        schema: Any | None,
    ) -> Any:
        """
        Parse raw LLM output into a typed model.
        Retries with correction hint on validation failure.
        """
        import json as _json

        target_schema = schema or self._config.structured_output.pydantic_model
        max_retries = self._config.structured_output.max_retries
        is_pydantic_model = isinstance(target_schema, type) and issubclass(target_schema, BaseModel)
        # A Pydantic *class* stringifies to "<class '...'>" — useless as a
        # correction hint. Use its real JSON schema instead when available.
        schema_for_prompt = (
            target_schema.model_json_schema() if is_pydantic_model else target_schema
        )
        # Native structured-output mode: OpenAI/Azure/openai-compatible
        # providers use response_format to constrain decoding to valid JSON;
        # other providers accept and ignore it (no behavior change for them).
        response_format = (
            {"type": "json_object"} if self._config.structured_output.use_native else None
        )

        for attempt in range(max_retries + 1):
            try:
                # Strip markdown code fences
                cleaned = raw_output.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]

                parsed = _json.loads(cleaned)

                if is_pydantic_model:
                    return target_schema(**parsed)
                return parsed
            except Exception as e:
                if attempt < max_retries:
                    # Ask LLM to fix the output
                    correction_prompt = (
                        f"Your previous output could not be parsed as JSON: {e}. "
                        f"Output ONLY valid JSON matching this schema: {schema_for_prompt}."
                    )
                    await ctx.window.add_user(correction_prompt)
                    messages = ctx.window.as_llm_messages()
                    model = ctx.effective_model()
                    response = await self._llm_router.complete(
                        messages=messages, model=model, response_format=response_format
                    )
                    raw_output = response.content
                else:
                    # Return raw string on final failure
                    return raw_output
        return raw_output

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    async def _finalize(self, ctx: ExecutionContext) -> None:
        """Post-run cleanup: record episode, flush trace."""
        outcome = EpisodeOutcome.FAILURE if ctx.error else EpisodeOutcome.SUCCESS
        if self._memory_store:
            try:
                from helix.config import Episode

                ep = Episode(
                    agent_id=self._config.agent_id,
                    task=ctx.step_outputs.get(0, "")[:200],
                    outcome=outcome,
                    steps=ctx.window.step,
                    cost_usd=ctx.cost.spent_usd,
                    tools_used=list({tc.tool_name for tc in ctx.tool_calls}),
                )
                await self._memory_store.record_episode(ep)
            except Exception:
                pass

        if self._tracer:
            with contextlib.suppress(Exception):
                self._tracer.finalize(ctx)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        await self._initialize_subsystems()
        self._initialized = True

    async def _initialize_subsystems(self) -> None:
        """Lazily initialize all subsystems on first run."""
        cfg = self._config

        # LLM Router — auto-detect model from available keys if none specified
        from helix.config_store import best_available_model
        from helix.models.router import FALLBACK_CHAINS, ModelRouter

        primary_was_specified = bool(cfg.model.primary.strip())
        primary = cfg.model.primary.strip() or best_available_model()
        # When auto-detecting, disable complexity-based routing — use the
        # detected model directly so we don't accidentally route to gpt-4o-mini.
        effective_auto_route = cfg.model.auto_route and primary_was_specified
        # Always build the fallback from the selected primary's chain.
        fallback = cfg.model.fallback_chain or FALLBACK_CHAINS.get(primary, [])
        self._llm_router = ModelRouter(
            primary_model=primary,
            fallback_chain=fallback,
            auto_route=effective_auto_route,
        )
        # Store resolved model so result.model_used is correct
        self._resolved_primary = primary
        # Write back to config so ctx.effective_model() returns the real model,
        # not the empty string that gets appended to ctx.model_per_step.
        self._config.model.primary = primary

        # Memory
        from helix.memory.store import MemoryStore

        self._memory_store = MemoryStore(config=cfg.memory)
        await self._memory_store.initialize()

        # Guardrails — cfg.guardrails is a list of built-in names (e.g.
        # "pii_redactor", "length_guard"); _run_guardrails() no-ops if this
        # stays None, so without this, AgentConfig.guardrails was unreachable.
        if cfg.guardrails:
            from helix.safety.guardrails import build_guardrail_chain

            self._guardrail_chain = build_guardrail_chain(cfg.guardrails)

        # Cache
        if cfg.cache.enabled:
            from helix.cache.controller import CacheController

            self._cache_controller = CacheController(config=cfg.cache)
            await self._cache_controller.initialize()

        # Context engine
        from helix.context_engine.engine import ContextEngine

        self._context_engine = ContextEngine(config=cfg)

        # Embedder — powers relevance decay's semantic term and compaction
        # clustering. Same convention as cache/memory: always construct it;
        # it degrades to zero vectors on its own if no embedding key is set.
        from helix.models.embedder import OpenAIEmbedder

        self._embedder = OpenAIEmbedder()

        # Audit log
        if cfg.observability.audit_enabled:
            from helix.safety.audit import LocalFileAuditLog

            self._audit_log = LocalFileAuditLog(
                agent_id=cfg.agent_id,
                log_dir=".helix/audit",
            )

        # Tracer is (re)created per run() call, not here — see Agent.run().

    # ------------------------------------------------------------------
    # Result builders
    # ------------------------------------------------------------------

    def _build_result(
        self,
        ctx: ExecutionContext,
        output: Any,
        episodes_used: int = 0,
    ) -> AgentResult:
        trace = None
        if self._tracer:
            trace = self._tracer.export()

        model_used = (
            ctx.model_per_step[-1]
            if ctx.model_per_step
            else getattr(self, "_resolved_primary", self._config.model.primary)
        )

        return AgentResult(
            output=output,
            steps=ctx.window.step,
            cost_usd=round(ctx.cost.spent_usd, 6),
            run_id=ctx.run_id,
            agent_id=self._config.agent_id,
            agent_name=self._config.name,
            duration_s=round(time.time() - ctx.started_at, 3),
            tool_calls=len(ctx.tool_calls),
            tool_call_records=list(ctx.tool_calls),
            cache_hits=ctx.cache_hits,
            cache_savings_usd=round(ctx.cache_savings_usd, 6),
            episodes_used=episodes_used,
            model_used=model_used,
            trace=trace,
        )

    def _error_result(self, ctx: ExecutionContext, error_msg: str) -> AgentResult:
        ctx.error = Exception(error_msg)
        # Surface the error in output so callers who only read result.output see it
        output_msg = f"[ERROR] {error_msg}"
        return AgentResult(
            output=output_msg,
            steps=ctx.window.step,
            cost_usd=round(ctx.cost.spent_usd, 6),
            run_id=ctx.run_id,
            agent_id=self._config.agent_id,
            agent_name=self._config.name,
            duration_s=round(time.time() - ctx.started_at, 3),
            tool_calls=len(ctx.tool_calls),
            tool_call_records=list(ctx.tool_calls),
            cache_hits=ctx.cache_hits,
            cache_savings_usd=0.0,
            error=error_msg,
        )
