"""
tests/test_basic.py — Smoke tests for the Helix framework.
Run with: pytest tests/ -v
"""

import asyncio
import pytest

import helix
from helix.config import AgentMode, BudgetConfig, ModelConfig
from helix.errors import BudgetExceededError
from helix.core.tool import tool, ToolRegistry
from helix.core.workflow import Workflow, step
from helix.core.team import Team
from helix.context import ExecutionContext, CostLedger, ContextWindow
from helix.config import AgentConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_config():
    return AgentConfig(
        name="test-agent",
        role="tester",
        goal="run tests",
        mode=AgentMode.EXPLORE,
    )

@pytest.fixture
def budget_config():
    return AgentConfig(
        name="test-agent",
        role="tester",
        goal="run tests",
        budget=BudgetConfig(budget_usd=1.00),
        mode=AgentMode.PRODUCTION,
    )


# ── Config tests ───────────────────────────────────────────────────────────────

class TestConfig:
    def test_agent_config_defaults(self, minimal_config):
        assert minimal_config.name == "test-agent"
        assert minimal_config.loop_limit == 50       # actual default
        assert minimal_config.agent_id is not None

    def test_budget_config_validation(self):
        with pytest.raises(Exception):
            BudgetConfig(budget_usd=-1.0)

    def test_production_requires_budget(self):
        with pytest.raises(Exception):
            AgentConfig(
                name="x", role="x", goal="x",
                mode=AgentMode.PRODUCTION,
                budget=None,
            )

    def test_model_config_defaults(self):
        cfg = ModelConfig()
        # primary is "" — auto-detected at runtime from available keys
        assert cfg.primary == ""
        assert cfg.temperature == 0.7

    def test_model_config_explicit(self):
        cfg = ModelConfig(primary="gemini-2.0-flash")
        assert cfg.primary == "gemini-2.0-flash"

    def test_model_config_temperature_bounds(self):
        with pytest.raises(Exception):
            ModelConfig(temperature=3.0)  # max is 2.0


# ── Context tests ──────────────────────────────────────────────────────────────

class TestContext:
    @pytest.mark.asyncio
    async def test_cost_ledger_gate(self):
        ledger = CostLedger(budget_usd=1.00)
        await ledger.record(0.50)
        assert ledger.budget_pct == pytest.approx(0.50)

        with pytest.raises(BudgetExceededError):
            await ledger.check_gate(0.60)   # must await — it's async

    @pytest.mark.asyncio
    async def test_cost_ledger_no_budget(self):
        """CostLedger with no budget never raises."""
        ledger = CostLedger(budget_usd=None)
        await ledger.record(999.99)
        await ledger.check_gate(999.99)  # should not raise

    @pytest.mark.asyncio
    async def test_context_window_messages(self, minimal_config):
        ctx = ExecutionContext(config=minimal_config)
        await ctx.window.add_user("Hello, agent.")
        msgs = ctx.window.messages()
        assert len(msgs) == 1
        assert msgs[0].content == "Hello, agent."

    def test_execution_context_summary(self, minimal_config):
        ctx = ExecutionContext(config=minimal_config)
        summary = ctx.summary()
        assert "run_id" in summary
        assert "steps" in summary


# ── Tool tests ─────────────────────────────────────────────────────────────────

class TestTools:
    def test_tool_decorator(self):
        @tool(description="Add two numbers.")
        async def add(a: int, b: int) -> int:
            return a + b

        assert add.name == "add"
        assert "Add two numbers" in add.description

    @pytest.mark.asyncio
    async def test_tool_execution(self):
        @tool(description="Return hello.")
        async def hello() -> str:
            return "hello"

        result = await hello()
        assert result == "hello"

    def test_tool_registry(self):
        reg = ToolRegistry()

        @tool(description="Test tool.")
        async def test_fn() -> str:
            return "ok"

        reg.register(test_fn)
        assert reg.has("test_fn")
        assert len(reg) >= 1

    @pytest.mark.asyncio
    async def test_tool_timeout(self):
        @tool(description="Slow tool.", timeout=0.01)
        async def slow() -> str:
            await asyncio.sleep(10)
            return "done"

        from helix.errors import ToolTimeoutError
        with pytest.raises(ToolTimeoutError):
            await slow()

    def test_tool_schema_generation(self):
        @tool(description="Compute something.")
        async def compute(value: int, label: str = "default") -> dict:
            return {"value": value}

        schema = compute.to_llm_schema()
        assert schema["name"] == "compute"
        assert "value" in schema["parameters"]["properties"]


# ── Workflow tests ─────────────────────────────────────────────────────────────

class TestWorkflow:
    @pytest.mark.asyncio
    async def test_sequential_workflow(self):
        @step(name="double")
        async def double(x):
            return x * 2

        @step(name="add_one")
        async def add_one(x):
            return x + 1

        wf = Workflow("test").then(double).then(add_one)
        result = await wf.run(5)
        assert result.final_output == 11  # (5 * 2) + 1
        assert result.error is None

    @pytest.mark.asyncio
    async def test_parallel_workflow(self):
        @step(name="plus_one")
        async def plus_one(x):
            return x + 1

        @step(name="times_two")
        async def times_two(x):
            return x * 2

        wf = Workflow("test").parallel(plus_one, times_two)
        result = await wf.run(3)
        assert set(result.final_output) == {4, 6}

    @pytest.mark.asyncio
    async def test_workflow_on_failure_continue(self):
        @step(name="fails")
        async def fails(x):
            raise ValueError("intentional failure")

        @step(name="ok")
        async def ok(x):
            return "ok"

        wf = Workflow("test").then(fails).then(ok).on_failure("continue")
        result = await wf.run("input")
        assert result.final_output == "ok"

    def test_workflow_run_sync(self):
        @step(name="identity")
        async def identity(x):
            return x

        wf = Workflow("test").then(identity)
        result = wf.run_sync("hello")
        assert result.final_output == "hello"


# ── Team tests ─────────────────────────────────────────────────────────────────

class TestTeam:
    @pytest.mark.asyncio
    async def test_team_parallel_mock(self):
        """Team parallel strategy — mocked agents."""
        from unittest.mock import AsyncMock, MagicMock
        from helix.core.agent import AgentResult

        mock_result = AgentResult(
            output="test output",
            steps=1,
            cost_usd=0.01,
            run_id="r1",
            agent_id="a1",
            agent_name="mock",
            duration_s=0.1,
            tool_calls=0,            # int, not list
            cache_hits=0,
            cache_savings_usd=0.0,
            episodes_used=0,
        )

        agent1 = MagicMock()
        agent1.run = AsyncMock(return_value=mock_result)
        agent1.name = "agent1"

        agent2 = MagicMock()
        agent2.run = AsyncMock(return_value=mock_result)
        agent2.name = "agent2"

        team = Team(name="test-team", agents=[agent1, agent2], strategy="parallel")
        result = await team.run("test task")

        assert result.error is None
        assert len(result.agent_results) == 2
        assert result.total_cost_usd == pytest.approx(0.02)


# ── helix.run() sync helper ───────────────────────────────────────────────────

class TestSyncRun:
    def test_helix_run_is_callable(self):
        assert callable(helix.run)

    def test_public_api_exports(self):
        assert hasattr(helix, "Agent")
        assert hasattr(helix, "run")
        assert hasattr(helix, "run_async")
        assert hasattr(helix, "tool")
        assert hasattr(helix, "Workflow")
        assert hasattr(helix, "Team")
        assert hasattr(helix, "Session")
        assert hasattr(helix, "BudgetConfig")
        assert hasattr(helix, "AgentMode")
        assert hasattr(helix, "BudgetExceededError")
        assert isinstance(helix.__version__, str) and helix.__version__


# ── Memory tests ──────────────────────────────────────────────────────────────

class TestMemory:
    @pytest.mark.asyncio
    async def test_inmemory_backend_upsert_search(self):
        from helix.memory.backends.inmemory import InMemoryBackend
        from helix.config import MemoryEntry, MemoryKind

        backend = InMemoryBackend()
        entry = MemoryEntry(
            content="Paris is the capital of France",
            kind=MemoryKind.FACT,
            importance=0.9,
            embedding=[1.0, 0.0, 0.0],
        )
        await backend.upsert(entry)
        results = await backend.search(query_embedding=[1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].content == "Paris is the capital of France"

    @pytest.mark.asyncio
    async def test_inmemory_backend_delete(self):
        from helix.memory.backends.inmemory import InMemoryBackend
        from helix.config import MemoryEntry, MemoryKind

        backend = InMemoryBackend()
        entry = MemoryEntry(
            content="to delete",
            kind=MemoryKind.FACT,
            importance=0.5,
            embedding=[0.0, 1.0, 0.0],
        )
        await backend.upsert(entry)
        await backend.delete(entry.id)
        results = await backend.search(query_embedding=[0.0, 1.0, 0.0], top_k=1)
        assert len(results) == 0


# ── Cache tests ────────────────────────────────────────────────────────────────

class TestCache:
    @pytest.mark.asyncio
    async def test_semantic_cache_miss(self):
        from helix.cache.semantic import SemanticCache, InMemoryCacheBackend
        from helix.config import CacheConfig
        from helix.models.embedder import NullEmbedder

        cache = SemanticCache(
            config=CacheConfig(semantic_threshold=0.99),  # high threshold → miss
            backend=InMemoryCacheBackend(),
        )
        await cache.initialize(embedder=NullEmbedder())
        result = await cache.get("what is AI?", context_hash="abc123")
        assert result is None

    @pytest.mark.asyncio
    async def test_semantic_cache_hit(self):
        """
        With threshold=0.0 and NullEmbedder (returns all zeros),
        cosine_similarity returns 0.0 and 0.0 >= 0.0 is True → cache hit.
        """
        from helix.cache.semantic import SemanticCache, InMemoryCacheBackend
        from helix.config import CacheConfig
        from helix.models.embedder import NullEmbedder

        cache = SemanticCache(
            config=CacheConfig(semantic_threshold=0.0),  # zero → always match
            backend=InMemoryCacheBackend(),
        )
        await cache.initialize(embedder=NullEmbedder())

        await cache.set("what is AI?", "abc123", "AI is artificial intelligence.", 0.01)
        result = await cache.get("what is AI?", "abc123")
        assert result is not None
        assert result.response == "AI is artificial intelligence."

    @pytest.mark.asyncio
    async def test_semantic_cache_context_isolation(self):
        """Different context_hash → cache miss even for identical query."""
        from helix.cache.semantic import SemanticCache, InMemoryCacheBackend
        from helix.config import CacheConfig
        from helix.models.embedder import NullEmbedder

        cache = SemanticCache(
            config=CacheConfig(semantic_threshold=0.0),
            backend=InMemoryCacheBackend(),
        )
        await cache.initialize(embedder=NullEmbedder())

        await cache.set("hello?", "ctx-A", "Hi!", 0.01)
        result = await cache.get("hello?", "ctx-B")  # different context
        assert result is None


# ── Safety tests ──────────────────────────────────────────────────────────────

class TestSafety:
    def test_permission_scope_allow(self):
        from helix.safety.permissions import PermissionScope
        from helix.config import PermissionConfig

        scope = PermissionScope(
            PermissionConfig(allowed_tools=["search", "calculator"]),
            agent_id="test",
        )
        scope.check_tool("search")
        scope.check_tool("calculator")

        from helix.errors import PermissionDeniedError
        with pytest.raises(PermissionDeniedError):
            scope.check_tool("write_file")

    def test_permission_scope_deny(self):
        from helix.safety.permissions import PermissionScope
        from helix.config import PermissionConfig
        from helix.errors import PermissionDeniedError

        scope = PermissionScope(
            PermissionConfig(denied_tools=["write_file"]),
            agent_id="test",
        )
        scope.check_tool("search")  # allowed

        with pytest.raises(PermissionDeniedError):
            scope.check_tool("write_file")

    @pytest.mark.asyncio
    async def test_pii_redactor(self):
        from helix.safety.guardrails import PIIRedactor

        guard = PIIRedactor()
        result = await guard.check(
            "Contact me at alice@example.com for details.",
            context=None,
        )
        assert result.passed
        assert "EMAIL_REDACTED" in result.modified_content

    @pytest.mark.asyncio
    async def test_keyword_block_guard(self):
        from helix.safety.guardrails import KeywordBlockGuard

        guard = KeywordBlockGuard(blocked_keywords=["forbidden", "blocked"])
        ok = await guard.check("This is fine.", context=None)
        assert ok.passed

        blocked = await guard.check("This is forbidden content.", context=None)
        assert not blocked.passed

    @pytest.mark.asyncio
    async def test_cost_governor_gate(self):
        from helix.safety.governor import CostGovernor

        gov = CostGovernor(BudgetConfig(budget_usd=0.10), agent_id="test")
        await gov.record(
            helix.config.TokenUsage(prompt_tokens=1000, completion_tokens=500),
            model="gpt-4o-mini",
        )
        assert gov.pct_used() > 0

        with pytest.raises(BudgetExceededError):
            await gov.check_gate("gpt-4o-mini", estimated_tokens=10_000_000)

    @pytest.mark.asyncio
    async def test_audit_log_hash_chain(self, tmp_path):
        from helix.safety.audit import LocalFileAuditLog
        from helix.config import AuditEntry, AuditEventType

        log = LocalFileAuditLog(agent_id="test", log_dir=str(tmp_path))

        # Use actual enum values from AuditEventType
        for event_type in [
            AuditEventType.SESSION_START,
            AuditEventType.LLM_CALL,
            AuditEventType.SESSION_END,
        ]:
            entry = AuditEntry(event_type=event_type, agent_id="test")
            await log.append(entry)

        ok, broken = await log.verify_chain()
        assert ok
        assert broken is None


# ── Eval tests ────────────────────────────────────────────────────────────────

class TestEval:
    @pytest.mark.asyncio
    async def test_tool_selection_scorer(self):
        from helix.eval.scoring import ToolSelectionScorer
        from helix.config import EvalCase, ToolCallRecord

        scorer = ToolSelectionScorer()
        case = EvalCase(
            name="test",
            input="test",
            expected_tools=["search", "calculator"],
        )
        tc = [
            ToolCallRecord(tool_name="search", arguments={}, step=1),
            ToolCallRecord(tool_name="calculator", arguments={}, step=2),
        ]
        score = await scorer.score(case, "output", tc, 0.01, 2)
        assert score == 1.0

    @pytest.mark.asyncio
    async def test_trajectory_scorer_forbidden_tool(self):
        from helix.eval.trajectory import TrajectoryScorer
        from helix.config import EvalCase, ExpectedTrajectory, ToolCallRecord

        scorer = TrajectoryScorer()
        case = EvalCase(
            name="test",
            input="test",
            expected_trajectory=ExpectedTrajectory(
                tool_sequence=["search"],
                must_not_call=["write_file"],
            ),
        )
        tc = [
            ToolCallRecord(tool_name="search", arguments={}, step=1),
            ToolCallRecord(tool_name="write_file", arguments={}, step=2),
        ]
        score = await scorer.score(case, "output", tc, 0.01, 2)
        assert score == 0.0
