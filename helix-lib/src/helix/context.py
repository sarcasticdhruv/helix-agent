"""
helix/context.py

ExecutionContext — the single object that carries all mutable state
through a single agent run.

Design rules:
  - This is not the Agent. It is the agent's runtime state.
  - Created fresh per run (or restored from checkpoint on resume).
  - Passed by reference to every subsystem that needs runtime state.
  - Serializable so workflow checkpoints work.
  - Thread/coroutine safe: all mutations go through methods, not
    direct attribute assignment.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from helix.config import (
    AgentConfig,
    AuditEventType,
    BudgetStrategy,
    ContextMessage,
    ContextMessageRole,
    ModelResponse,
    MemoryEntry,
    TokenUsage,
    ToolCallRecord,
)


class CostLedger:
    """
    Tracks cost for a single run. All mutations are synchronous and
    protected by an asyncio Lock for concurrent safety.
    """

    def __init__(self, budget_usd: Optional[float]) -> None:
        self._budget_usd = budget_usd
        self._spent_usd: float = 0.0
        self._calls: int = 0
        self._lock = asyncio.Lock()

    async def record(self, cost_usd: float) -> None:
        async with self._lock:
            self._spent_usd += cost_usd
            self._calls += 1

    async def check_gate(self, estimated_cost_usd: float) -> None:
        """
        Called BEFORE every LLM call. Raises BudgetExceededError if
        this call would push spending over budget.
        """
        from helix.errors import BudgetExceededError

        if self._budget_usd is None:
            return
        async with self._lock:
            if self._spent_usd + estimated_cost_usd > self._budget_usd:
                raise BudgetExceededError(
                    agent_id="",  # Filled by caller
                    budget_usd=self._budget_usd,
                    spent_usd=self._spent_usd,
                    attempted_usd=estimated_cost_usd,
                )

    @property
    def spent_usd(self) -> float:
        return self._spent_usd

    @property
    def budget_usd(self) -> Optional[float]:
        return self._budget_usd

    @property
    def budget_pct(self) -> Optional[float]:
        if self._budget_usd is None or self._budget_usd == 0:
            return None
        return self._spent_usd / self._budget_usd

    @property
    def calls(self) -> int:
        return self._calls

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spent_usd": self._spent_usd,
            "budget_usd": self._budget_usd,
            "calls": self._calls,
        }


class ContextWindow:
    """
    Manages the active message list for an agent run.

    Responsibilities:
      - Add and evict messages
      - Track relevance decay
      - Detect when compaction is needed
      - Produce the final message list for LLM calls
      - Provide a stable context_hash for cache lookups
    """

    def __init__(self, limit_tokens: int) -> None:
        self._messages: List[ContextMessage] = []
        self._limit_tokens = limit_tokens
        self._step = 0
        self._lock = asyncio.Lock()

    async def add(self, message: ContextMessage) -> None:
        async with self._lock:
            message.step_added = self._step
            self._messages.append(message)

    async def add_system(self, content: str, pinned: bool = True) -> None:
        await self.add(ContextMessage(
            role=ContextMessageRole.SYSTEM,
            content=content,
            pinned=pinned,
        ))

    async def add_user(self, content: str) -> None:
        await self.add(ContextMessage(role=ContextMessageRole.USER, content=content))

    async def add_assistant(self, content: str) -> None:
        await self.add(ContextMessage(role=ContextMessageRole.ASSISTANT, content=content))

    async def add_tool_result(self, tool_name: str, content: str) -> None:
        await self.add(ContextMessage(
            role=ContextMessageRole.TOOL,
            content=content,
            tool_name=tool_name,
        ))

    async def inject_step_output(self, step: int, output: Any) -> None:
        """Used by FailureReplay to override a step's output in context."""
        async with self._lock:
            content = json.dumps(output) if not isinstance(output, str) else output
            msg = ContextMessage(
                role=ContextMessageRole.ASSISTANT,
                content=f"[REPLAY OVERRIDE step={step}] {content}",
                step_added=step,
                pinned=False,
            )
            self._messages.append(msg)

    def tick(self) -> None:
        """Advance step counter. Called after each agent reasoning step."""
        self._step += 1

    @property
    def step(self) -> int:
        return self._step

    def messages(self) -> List[ContextMessage]:
        """Return a snapshot of the current message list (sorted by step)."""
        return list(self._messages)

    def as_llm_messages(self) -> List[Dict[str, Any]]:
        """
        Convert to the list[dict] format expected by LLM providers.
        Tool messages are mapped to the provider-agnostic format here;
        provider adapters do final normalization.
        """
        result = []
        for msg in self._messages:
            if msg.role == ContextMessageRole.TOOL:
                result.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_name": msg.tool_name or "",
                })
            else:
                result.append({"role": msg.role.value, "content": msg.content})
        return result

    def total_tokens(self) -> int:
        """Estimate total tokens in context (rough: chars / 4)."""
        total = 0
        for msg in self._messages:
            if msg.token_count is not None:
                total += msg.token_count
            else:
                total += len(msg.content) // 4
        return total

    def needs_compaction(self, threshold: float = 0.7) -> bool:
        return self.total_tokens() > self._limit_tokens * threshold

    def context_hash(self) -> str:
        """
        Stable hash of the current context for cache key construction.
        Only hashes pinned messages + last 3 non-pinned to stay stable
        across minor context changes.
        """
        pinned = [m for m in self._messages if m.pinned]
        recent = [m for m in self._messages if not m.pinned][-3:]
        canonical = json.dumps(
            [{"role": m.role.value, "content": m.content} for m in pinned + recent],
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def evict_below_relevance(self, threshold: float = 0.2) -> int:
        """Remove non-pinned messages with relevance below threshold."""
        before = len(self._messages)
        self._messages = [
            m for m in self._messages
            if m.pinned or m.relevance >= threshold
        ]
        return before - len(self._messages)

    def replace_with_summary(
        self,
        to_replace: List[str],  # message ids
        summary_content: str,
    ) -> None:
        """Replace a group of messages with a single summary message."""
        ids = set(to_replace)
        self._messages = [m for m in self._messages if m.id not in ids]
        self._messages.append(ContextMessage(
            role=ContextMessageRole.SYSTEM,
            content=f"[CONTEXT SUMMARY] {summary_content}",
            pinned=False,
            relevance=0.9,
        ))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self._step,
            "limit_tokens": self._limit_tokens,
            "messages": [m.model_dump() for m in self._messages],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextWindow":
        obj = cls(limit_tokens=data["limit_tokens"])
        obj._step = data["step"]
        obj._messages = [ContextMessage(**m) for m in data["messages"]]
        return obj


class LoopGuard:
    """
    Detects pathological patterns in agent behavior.
    Checked after every step; raises LoopDetectedError on trigger.
    """

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._tool_call_history: List[Tuple[str, str]] = []  # (name, args_hash)
        self._output_history: List[str] = []

    def record_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        args_hash = hashlib.md5(
            json.dumps(arguments, sort_keys=True).encode()
        ).hexdigest()[:8]
        self._tool_call_history.append((tool_name, args_hash))

    def record_output(self, output: str) -> None:
        self._output_history.append(output[:200])  # Truncate for comparison

    def check(self, step: int, agent_id: str) -> None:
        from helix.errors import LoopDetectedError

        if step >= self._limit:
            raise LoopDetectedError(
                agent_id=agent_id,
                signal="step_limit",
                step_count=step,
            )

        # Detect same tool+args called 3+ times consecutively
        if len(self._tool_call_history) >= 3:
            last3 = self._tool_call_history[-3:]
            if len(set(last3)) == 1:
                raise LoopDetectedError(
                    agent_id=agent_id,
                    signal="repeated_tool_call",
                    step_count=step,
                    details={"repeated_call": last3[0]},
                )

        # Detect oscillating outputs
        if len(self._output_history) >= 4:
            last4 = self._output_history[-4:]
            if last4[0] == last4[2] and last4[1] == last4[3]:
                raise LoopDetectedError(
                    agent_id=agent_id,
                    signal="oscillating_state",
                    step_count=step,
                )


class ExecutionContext:
    """
    The complete runtime state of a single agent run.

    Created by Agent.run() at the start of each invocation.
    Passed to every subsystem: context engine, memory, cache,
    safety layer, tracer, and tool runtime.

    Can be serialized to dict for:
      - Workflow checkpointing (resume after interruption)
      - FailureReplay (step override and re-run)
      - HITL pause/resume
    """

    def __init__(
        self,
        config: AgentConfig,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
        parent_run_id: Optional[str] = None,
    ) -> None:
        self.run_id: str = run_id or str(uuid.uuid4())
        self.session_id: Optional[str] = session_id
        self.parent_run_id: Optional[str] = parent_run_id
        self.config: AgentConfig = config
        self.started_at: float = time.time()

        # Sub-components
        self.window = ContextWindow(limit_tokens=config.context_limit_tokens)
        self.cost = CostLedger(
            budget_usd=config.budget.budget_usd if config.budget else None
        )
        self.loop_guard = LoopGuard(limit=config.loop_limit)

        # Accumulated outputs
        self.tool_calls: List[ToolCallRecord] = []
        self.step_outputs: Dict[int, Any] = {}
        self.final_output: Optional[str] = None
        self.error: Optional[Exception] = None

        # HITL state
        self.hitl_pending: bool = False
        self.hitl_request_id: Optional[str] = None

        # Memory entries retrieved for this run (for tracing)
        self.recalled_memories: List[MemoryEntry] = []

        # Cache stats for this run
        self.cache_hits: int = 0
        self.cache_savings_usd: float = 0.0

        # Model used per step (may change with degradation)
        self.model_per_step: List[str] = []

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def record_tool_call(self, record: ToolCallRecord) -> None:
        self.tool_calls.append(record)
        self.loop_guard.record_tool_call(record.tool_name, record.arguments)

    def record_step_output(self, step: int, output: Any) -> None:
        self.step_outputs[step] = output
        if isinstance(output, str):
            self.loop_guard.record_output(output)

    def record_cache_hit(self, saved_usd: float) -> None:
        self.cache_hits += 1
        self.cache_savings_usd += saved_usd

    def check_loop(self) -> None:
        """Called after each step. Raises LoopDetectedError if pattern detected."""
        self.loop_guard.check(
            step=self.window.step,
            agent_id=self.config.agent_id,
        )

    def effective_model(self) -> str:
        """
        Return the model to use for the next call, respecting budget degradation.
        """
        if (
            self.config.budget
            and self.config.budget.strategy == BudgetStrategy.DEGRADE
            and self.cost.budget_pct is not None
        ):
            pct = self.cost.budget_pct
            for step in sorted(
                self.config.budget.degradation_steps,
                key=lambda s: s.at_pct,
                reverse=True,
            ):
                if pct >= step.at_pct and step.switch_to_model:
                    return step.switch_to_model
        return self.config.model.primary

    # ------------------------------------------------------------------
    # Serialization (for checkpointing and replay)
    # ------------------------------------------------------------------

    def to_checkpoint(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "parent_run_id": self.parent_run_id,
            "agent_id": self.config.agent_id,
            "started_at": self.started_at,
            "window": self.window.to_dict(),
            "cost": self.cost.to_dict(),
            "step_outputs": {str(k): v for k, v in self.step_outputs.items()},
            "tool_calls": [tc.model_dump() for tc in self.tool_calls],
            "final_output": self.final_output,
            "hitl_pending": self.hitl_pending,
            "hitl_request_id": self.hitl_request_id,
            "cache_hits": self.cache_hits,
            "cache_savings_usd": self.cache_savings_usd,
        }

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        config: AgentConfig,
    ) -> "ExecutionContext":
        ctx = cls(
            config=config,
            run_id=checkpoint["run_id"],
            session_id=checkpoint.get("session_id"),
            parent_run_id=checkpoint.get("parent_run_id"),
        )
        ctx.started_at = checkpoint["started_at"]
        ctx.window = ContextWindow.from_dict(checkpoint["window"])
        ctx.step_outputs = {int(k): v for k, v in checkpoint["step_outputs"].items()}
        ctx.tool_calls = [ToolCallRecord(**tc) for tc in checkpoint["tool_calls"]]
        ctx.final_output = checkpoint.get("final_output")
        ctx.hitl_pending = checkpoint.get("hitl_pending", False)
        ctx.hitl_request_id = checkpoint.get("hitl_request_id")
        ctx.cache_hits = checkpoint.get("cache_hits", 0)
        ctx.cache_savings_usd = checkpoint.get("cache_savings_usd", 0.0)
        return ctx

    # ------------------------------------------------------------------
    # Summary for tracing
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "agent_id": self.config.agent_id,
            "agent_name": self.config.name,
            "duration_s": round(time.time() - self.started_at, 3),
            "steps": self.window.step,
            "cost_usd": round(self.cost.spent_usd, 6),
            "budget_usd": self.cost.budget_usd,
            "tool_calls": len(self.tool_calls),
            "cache_hits": self.cache_hits,
            "cache_savings_usd": round(self.cache_savings_usd, 6),
            "context_tokens": self.window.total_tokens(),
            "error": str(self.error) if self.error else None,
        }
