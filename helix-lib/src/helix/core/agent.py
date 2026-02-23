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
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Type, Union

from pydantic import BaseModel, ConfigDict

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
)
from helix.context import ExecutionContext
from helix.core.tool import ToolRegistry, ToolRegistryView, execute_tool, registry as _global_registry
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

    output: Any                  # str or typed model if structured_output enabled
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
    model_used: Optional[str] = None   # Which model was actually called
    error: Optional[str] = None
    trace: Optional[Dict[str, Any]] = None  # Populated if observability enabled

    model_config = ConfigDict(frozen=True)


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
        model: Optional[ModelConfig] = None,
        budget: Optional[BudgetConfig] = None,
        mode: AgentMode = AgentMode.EXPLORE,
        tools: Optional[List[Any]] = None,
        memory: Optional[MemoryConfig] = None,
        cache: Optional[CacheConfig] = None,
        permissions: Optional[PermissionConfig] = None,
        structured_output: Optional[StructuredOutputConfig] = None,
        observability: Optional[ObservabilityConfig] = None,
        system_prompt: Optional[str] = None,
        **extra_config: Any,
    ) -> None:
        self._config = AgentConfig(
            name=name,
            role=role,
            goal=goal,
            mode=mode,
            model=model or ModelConfig(),
            budget=budget,
            memory=memory or MemoryConfig(),
            cache=cache or CacheConfig(),
            permissions=permissions or PermissionConfig(),
            structured_output=structured_output or StructuredOutputConfig(),
            observability=observability or ObservabilityConfig(),
            system_prompt_override=system_prompt,
        )

        # Tool registry: start with global, add agent-specific tools
        self._registry = ToolRegistry()
        # Inherit global registered tools
        for t in _global_registry.all():
            self._registry.register(t)
        # Register agent-specific tools
        if tools:
            for t in tools:
                self._registry.register(t)

        # Subsystems — lazily initialized on first run
        self._memory_store: Optional[Any] = None
        self._cache_controller: Optional[Any] = None
        self._llm_router: Optional[Any] = None
        self._tracer: Optional[Any] = None
        self._cost_governor: Optional[Any] = None
        self._guardrail_chain: Optional[Any] = None
        self._hitl_controller: Optional[Any] = None
        self._context_engine: Optional[Any] = None
        self._audit_log: Optional[Any] = None

        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        task: str,
        session_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
        output_schema: Optional[Union[Type[BaseModel], Dict[str, Any]]] = None,
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

        try:
            result = await self._execute(ctx, task, output_schema=output_schema)
        except BudgetExceededError as e:
            result = self._error_result(ctx, str(e))
        except LoopDetectedError as e:
            result = self._error_result(ctx, str(e))
        except HelixError as e:
            result = self._error_result(ctx, str(e))
        except Exception as e:
            result = self._error_result(ctx, f"Unexpected error: {e}")
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

    async def stream(
        self,
        task: str,
        session_id: Optional[str] = None,
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

    def add_tool(self, tool_or_fn: Any) -> "Agent":
        """Register an additional tool. Returns self for chaining."""
        self._registry.register(tool_or_fn)
        return self

    def clone(self, **overrides: Any) -> "Agent":
        """
        Create a copy of this agent with config overrides.
        Useful for A/B testing prompt variants or model differences.
        """
        config_data = self._config.model_dump()
        config_data.update(overrides)
        new_agent = Agent.__new__(Agent)
        new_agent._config = AgentConfig(**config_data)
        new_agent._registry = self._registry  # Shared tool registry
        new_agent._initialized = False
        # Subsystems will re-init on first run
        for attr in (
            "_memory_store", "_cache_controller", "_llm_router", "_tracer",
            "_cost_governor", "_guardrail_chain", "_hitl_controller",
            "_context_engine", "_audit_log",
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
        output_schema: Optional[Any] = None,
    ) -> AgentResult:
        # 1. Build initial context (system prompt + episodic memory + task)
        await self._build_context(ctx, task)

        # 2. Check semantic cache before any LLM call
        cache_hit = await self._check_cache(ctx, task)
        if cache_hit:
            output = cache_hit.response
            ctx.record_cache_hit(cache_hit.saved_usd)
            return self._build_result(ctx, output, episodes_used=0)

        # 3. Check plan cache — adapt plan template if available
        plan = await self._check_plan_cache(ctx, task)

        # 4. Reasoning loop
        output = await self._reasoning_loop(ctx, task, plan=plan)

        # 5. Apply structured output if configured
        if output_schema or self._config.structured_output.enabled:
            output = await self._apply_structured_output(ctx, output, output_schema)

        # 6. Store successful plan to plan cache
        await self._store_plan(ctx, task)

        return self._build_result(ctx, output)

    async def _reasoning_loop(
        self,
        ctx: ExecutionContext,
        task: str,
        plan: Optional[Any] = None,
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

        while True:
            ctx.window.tick()
            ctx.check_loop()

            # Compact context if approaching limit
            if ctx.window.needs_compaction():
                await self._context_engine.compact(ctx)

            messages = ctx.window.as_llm_messages()
            model = ctx.effective_model()

            # Cost gate
            estimated_cost = self._estimate_call_cost(messages, model)
            await ctx.cost.check_gate(estimated_cost)
            if self._config.budget:
                await self._warn_budget(ctx)

            # LLM call
            response = await self._llm_router.complete(
                messages=messages,
                model=model,
                tools=tool_schemas if tool_schemas else None,
                temperature=self._config.model.temperature,
                max_tokens=self._config.model.max_tokens,
            )
            ctx.model_per_step.append(model)

            # Record actual cost
            await ctx.cost.record(
                self._calculate_actual_cost(response.usage, model)
            )

            # Audit
            await self._audit("llm_response", ctx, {
                "model": model,
                "tokens": response.usage.model_dump(),
                "finish_reason": response.finish_reason,
            })

            # Guardrails on response
            cleaned = await self._run_guardrails(ctx, response.content)

            # Store assistant response in context
            await ctx.window.add_assistant(cleaned)
            ctx.record_step_output(ctx.window.step, cleaned)
            final_output = cleaned

            # Cache the response for future semantic lookups
            await self._store_semantic_cache(ctx, task, cleaned, ctx.cost.spent_usd)

            # No tool calls → we're done
            if response.finish_reason == "stop" or not response.tool_calls:
                break

            # Execute tool calls
            for tc in response.tool_calls:
                record = await execute_tool(
                    registry_view=registry_view,
                    tool_name=tc.tool_name,
                    arguments=tc.arguments,
                    step=ctx.window.step,
                    agent_id=self._config.agent_id,
                )
                ctx.record_tool_call(record)

                # Classify failure and decide recovery
                if record.failure_class is not None:
                    recovery = await self._handle_tool_failure(ctx, record)
                    if recovery == "abort":
                        break

                result_content = (
                    str(record.result) if record.result is not None
                    else f"[Tool {tc.tool_name} failed: {record.failure_class}]"
                )
                await ctx.window.add_tool_result(tc.tool_name, result_content)

                # Memory: auto-store important tool results
                await self._maybe_store_memory(ctx, tc.tool_name, result_content)

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

    async def _check_cache(self, ctx: ExecutionContext, task: str) -> Optional[Any]:
        if self._cache_controller is None:
            return None
        try:
            return await self._cache_controller.lookup(
                query=task,
                context_hash=ctx.window.context_hash(),
            )
        except Exception:
            return None

    async def _check_plan_cache(self, ctx: ExecutionContext, task: str) -> Optional[Any]:
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
        try:
            await self._cache_controller.semantic.set(
                query=task,
                context_hash=ctx.window.context_hash(),
                response=response,
                cost_usd=cost_usd,
            )
        except Exception:
            pass

    async def _store_plan(self, ctx: ExecutionContext, task: str) -> None:
        if self._cache_controller is None:
            return
        try:
            await self._cache_controller.plan.store(task, ctx)
        except Exception:
            pass

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
                await self._audit("guardrail_block", ctx, {
                    "guardrail": guardrail.name,
                    "reason": result.reason,
                })
                raise GuardrailViolationError(
                    guardrail_name=guardrail.name,
                    reason=result.reason or "Content blocked",
                    content_preview=content[:100],
                )
            if result.modified_content:
                result_content = result.modified_content
        return result_content

    async def _handle_tool_failure(
        self, ctx: ExecutionContext, record: Any
    ) -> str:
        """
        Classify failure and apply recovery strategy.
        Returns "continue" or "abort".
        """
        from helix.tools.taxonomy import RECOVERY_STRATEGIES, RetryStrategy, EscalateStrategy

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
            await self._audit("budget_warning", ctx, {
                "spent_usd": ctx.cost.spent_usd,
                "budget_usd": ctx.cost.budget_usd,
                "pct": pct,
            })

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

    def _estimate_call_cost(self, messages: List[Dict], model: str) -> float:
        """Rough pre-call cost estimate for budget gate."""
        token_count = sum(len(m.get("content", "")) // 4 for m in messages)
        cost_per_1k = 0.005  # Conservative default
        return (token_count / 1000) * cost_per_1k

    def _calculate_actual_cost(self, usage: Any, model: str) -> float:
        """Calculate actual cost from token usage."""
        if self._llm_router:
            return self._llm_router.calculate_cost(usage, model)
        return 0.0

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    async def _audit(self, event: str, ctx: ExecutionContext, details: Dict) -> None:
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
        schema: Optional[Any],
    ) -> Any:
        """
        Parse raw LLM output into a typed model.
        Retries with correction hint on validation failure.
        """
        import json as _json

        target_schema = schema or self._config.structured_output.pydantic_model
        max_retries = self._config.structured_output.max_retries

        for attempt in range(max_retries + 1):
            try:
                # Strip markdown code fences
                cleaned = raw_output.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]

                parsed = _json.loads(cleaned)

                if isinstance(target_schema, type) and issubclass(target_schema, BaseModel):
                    return target_schema(**parsed)
                return parsed
            except Exception as e:
                if attempt < max_retries:
                    # Ask LLM to fix the output
                    correction_prompt = (
                        f"Your previous output could not be parsed as JSON: {e}. "
                        f"Output ONLY valid JSON matching this schema: {target_schema}."
                    )
                    await ctx.window.add_user(correction_prompt)
                    messages = ctx.window.as_llm_messages()
                    model = ctx.effective_model()
                    response = await self._llm_router.complete(
                        messages=messages, model=model
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
        outcome = (
            EpisodeOutcome.FAILURE if ctx.error else EpisodeOutcome.SUCCESS
        )
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
            try:
                self._tracer.finalize(ctx)
            except Exception:
                pass

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
        from helix.models.router import ModelRouter, FALLBACK_CHAINS
        from helix.config_store import best_available_model
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

        # Cache
        if cfg.cache.enabled:
            from helix.cache.controller import CacheController
            self._cache_controller = CacheController(config=cfg.cache)
            await self._cache_controller.initialize()

        # Context engine
        from helix.context_engine.engine import ContextEngine
        self._context_engine = ContextEngine(config=cfg)

        # Audit log
        if cfg.observability.audit_enabled:
            from helix.safety.audit import LocalFileAuditLog
            self._audit_log = LocalFileAuditLog(
                agent_id=cfg.agent_id,
                log_dir=".helix/audit",
            )

        # Tracer
        if cfg.observability.trace_enabled:
            from helix.observability.tracer import Tracer
            self._tracer = Tracer(
                run_id="",  # Updated per-run
                agent_id=cfg.agent_id,
                agent_name=cfg.name,
            )

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

        model_used = ctx.model_per_step[-1] if ctx.model_per_step else getattr(self, "_resolved_primary", self._config.model.primary)

        return AgentResult(
            output=output,
            steps=ctx.window.step,
            cost_usd=round(ctx.cost.spent_usd, 6),
            run_id=ctx.run_id,
            agent_id=self._config.agent_id,
            agent_name=self._config.name,
            duration_s=round(time.time() - ctx.started_at, 3),
            tool_calls=len(ctx.tool_calls),
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
            cache_hits=ctx.cache_hits,
            cache_savings_usd=0.0,
            error=error_msg,
        )
