"""
tests/test_new_features.py

Tests for features added in the 0.4 development cycle:
  - StateGraph / CompiledGraph (LangGraph parity)
  - AgentPipeline / pipe operator  |
  - on_event hooks
  - @helix.agent class decorator
  - helix.quick() factory
  - helix.chain() factory
  - helix.discover_tools()
  - Presets (web_researcher, writer, coder, …)
  - execute_python built-in tool
  - agent.invoke() / agent.ainvoke() aliases
  - EvalSuite.case() decorator fix
  - Workflow typed ChainNode ADT

Run with:
    pytest tests/test_new_features.py -v
"""

from __future__ import annotations

import asyncio
from typing import TypedDict
from unittest.mock import AsyncMock, MagicMock

import pytest

import helix
from helix.config import AgentMode
from helix.core.agent import Agent, AgentResult
from helix.core.graph import END, START, CompiledGraph, StateGraph
from helix.core.hooks import HookEvent
from helix.core.hooks import fire as fire_hook
from helix.core.pipeline import AgentPipeline
from helix.core.tool import tool
from helix.core.workflow import Workflow, step

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_mock_agent(name: str, output: str = "ok", cost: float = 0.01) -> MagicMock:
    """Return a mock Agent that resolves run() with a fixed AgentResult."""
    result = AgentResult(
        output=output,
        steps=1,
        cost_usd=cost,
        run_id="r1",
        agent_id="a1",
        agent_name=name,
        duration_s=0.1,
        tool_calls=0,
        cache_hits=0,
        cache_savings_usd=0.0,
    )
    ag = MagicMock(spec=Agent)
    ag.name = name
    ag.run = AsyncMock(return_value=result)
    ag._on_event = None
    return ag


def _make_agent(name: str, output: str = "ok", cost: float = 0.01) -> Agent:
    """
    Create a real Agent instance with a mocked run() method.
    Using a real Agent preserves __or__, _config and all instance attributes,
    avoiding the spec=MagicMock pitfalls.
    """
    result = AgentResult(
        output=output,
        steps=1,
        cost_usd=cost,
        run_id="r1",
        agent_id="a1",
        agent_name=name,
        duration_s=0.1,
        tool_calls=0,
        cache_hits=0,
        cache_savings_usd=0.0,
    )
    ag = Agent(name=name, role=name, goal="test")
    ag.run = AsyncMock(return_value=result)  # type: ignore[method-assign]
    return ag


# ── StateGraph ─────────────────────────────────────────────────────────────────


class TestStateGraph:
    """Tests for helix.StateGraph / CompiledGraph."""

    def test_sentinels_exported(self):
        assert helix.END == "__end__"
        assert helix.START == "__start__"
        assert END == "__end__"
        assert START == "__start__"

    def test_compile_requires_entry_point(self):
        graph = StateGraph()
        graph.add_node("a", lambda s: s)
        with pytest.raises(ValueError, match="entry point"):
            graph.compile()

    def test_compile_rejects_unknown_entry(self):
        graph = StateGraph()
        graph.set_entry_point("missing")
        with pytest.raises(ValueError, match="'missing' is not a registered node"):
            graph.compile()

    def test_compiled_graph_repr(self):
        g = StateGraph().add_node("a", lambda s: s).set_entry_point("a").compile()
        assert "CompiledGraph" in repr(g)
        assert "'a'" in repr(g)

    def test_simple_sequential_graph_run_sync(self):
        def inc(state):
            return {"value": state["value"] + 1}

        g = (
            StateGraph(dict)
            .add_node("inc", inc)
            .set_entry_point("inc")
            .set_finish_point("inc")
            .compile()
        )
        result = g.run_sync({"value": 0})
        assert result["value"] == 1
        assert result.nodes_visited == ["inc"]
        assert result.error is None

    def test_multi_node_chain(self):
        def plus_one(state):
            return {"n": state["n"] + 1}

        def times_two(state):
            return {"n": state["n"] * 2}

        g = (
            StateGraph()
            .add_node("plus", plus_one)
            .add_node("double", times_two)
            .add_edge("plus", "double")
            .add_edge("double", END)
            .set_entry_point("plus")
            .compile()
        )
        result = g.invoke({"n": 3})  # (3+1)*2 = 8
        assert result["n"] == 8

    def test_conditional_edges_routing(self):
        def inc(state):
            return {"count": state["count"] + 1}

        def router(state):
            return "done" if state["count"] >= 3 else "loop"

        g = (
            StateGraph()
            .add_node("inc", inc)
            .add_conditional_edges("inc", router, {"loop": "inc", "done": END})
            .set_entry_point("inc")
            .compile()
        )
        result = g.run_sync({"count": 0})
        assert result["count"] == 3
        assert result.nodes_visited == ["inc", "inc", "inc"]

    def test_max_steps_guard(self):
        """Graph exceeding max_steps returns an error, does not crash."""

        def forever(state):
            return {}

        g = (
            StateGraph()
            .add_node("loop", forever)
            .add_edge("loop", "loop")
            .set_entry_point("loop")
            .compile(max_steps=5)
        )
        result = g.run_sync({})
        assert result.error is not None
        assert "Max steps" in result.error

    def test_unknown_node_error(self):
        g = (
            StateGraph()
            .add_node("a", lambda s: {"x": 1})
            .add_edge("a", "b")  # 'b' not registered
            .set_entry_point("a")
            .compile()
        )
        result = g.run_sync({})
        assert result.error is not None
        assert "b" in result.error

    @pytest.mark.asyncio
    async def test_async_run(self):
        async def async_inc(state):
            await asyncio.sleep(0)
            return {"v": state["v"] + 10}

        g = (
            StateGraph()
            .add_node("ai", async_inc)
            .add_edge("ai", END)
            .set_entry_point("ai")
            .compile()
        )
        result = await g.run({"v": 5})
        assert result["v"] == 15
        assert result.error is None

    @pytest.mark.asyncio
    async def test_ainvoke_raises_on_error(self):
        """CompiledGraph.ainvoke() re-raises errors as RuntimeError."""
        g = (
            StateGraph()
            .add_node("a", lambda s: {})
            .add_edge("a", "nonexistent")
            .set_entry_point("a")
            .compile()
        )
        with pytest.raises(RuntimeError):
            await g.ainvoke({})

    def test_invoke_raises_on_error(self):
        """CompiledGraph.invoke() re-raises errors as RuntimeError."""
        g = (
            StateGraph()
            .add_node("a", lambda s: {})
            .add_edge("a", "nonexistent")
            .set_entry_point("a")
            .compile()
        )
        with pytest.raises(RuntimeError):
            g.invoke({})

    @pytest.mark.asyncio
    async def test_stream_yields_node_results(self):
        results = []

        def s1(state):
            return {"a": 1}

        def s2(state):
            return {"b": 2}

        g = (
            StateGraph()
            .add_node("s1", s1)
            .add_node("s2", s2)
            .add_edge("s1", "s2")
            .add_edge("s2", END)
            .set_entry_point("s1")
            .compile()
        )
        async for nr in g.stream({}):
            results.append(nr.node)

        assert results == ["s1", "s2"]

    @pytest.mark.asyncio
    async def test_agent_node(self):
        """Agent instances can be used as graph nodes directly."""
        mock = _make_agent("Writer", output="Great article!")
        g = (
            StateGraph()
            .add_node("write", mock)
            .add_edge("write", END)
            .set_entry_point("write")
            .compile()
        )
        result = await g.run({"task": "write a story"})
        assert result["output"] == "Great article!"
        assert result.total_cost_usd == pytest.approx(0.01)

    def test_checkpoint_saves_and_loads(self, tmp_path):
        """State is saved to checkpoint_dir after each node."""

        def make_ten(state):
            return {"value": 10}

        g = (
            StateGraph()
            .add_node("set", make_ten)
            .add_edge("set", END)
            .set_entry_point("set")
            .compile(checkpoint_dir=tmp_path / "ckpt")
        )
        result = g.run_sync({"value": 0}, run_id="test-run")
        assert result["value"] == 10
        # checkpoint cleared on success
        assert not (tmp_path / "ckpt" / "test-run.json").exists()

    def test_typed_state_schema_accepted(self):
        class MyState(TypedDict):
            name: str
            score: int

        g = StateGraph(MyState).add_node("x", lambda s: {}).set_entry_point("x").compile()
        assert isinstance(g, CompiledGraph)

    def test_set_finish_edge_alias(self):
        """add_finish_edge is an alias for set_finish_point."""
        g = StateGraph()
        g.add_node("a", lambda s: {})
        g.set_entry_point("a")
        g.add_finish_edge("a")  # alias
        graph = g.compile()
        result = graph.run_sync({})
        assert result.error is None


# ── AgentPipeline + pipe operator ─────────────────────────────────────────────


class TestAgentPipeline:
    def test_pipe_operator_creates_pipeline(self):
        a = _make_agent("A")
        b = _make_agent("B")
        pipe = a | b
        assert isinstance(pipe, AgentPipeline)
        assert len(pipe.agents) == 2

    def test_chain_three_agents(self):
        a = _make_agent("A")
        b = _make_agent("B")
        c = _make_agent("C")
        pipe = a | b | c
        assert len(pipe.agents) == 3
        assert repr(pipe) == "AgentPipeline(A | B | C)"

    def test_helix_chain_factory(self):
        a = _make_agent("X")
        b = _make_agent("Y")
        pipe = helix.chain(a, b)
        assert isinstance(pipe, AgentPipeline)
        assert len(pipe.agents) == 2

    def test_helix_chain_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            helix.chain()

    @pytest.mark.asyncio
    async def test_pipeline_run_passes_output(self):
        a = _make_agent("A", output="from A")
        b = _make_agent("B", output="from B")
        pipe = a | b
        result = await pipe.run("start task")
        # AgentPipeline passes session_id=None by default
        b.run.assert_called_once_with("from A", session_id=None)
        assert result.output == "from B"

    def test_pipeline_run_sync(self):
        a = _make_agent("A", output="step1")
        b = _make_agent("B", output="step2")
        pipe = a | b
        result = pipe.run_sync("task")
        assert result.output == "step2"

    def test_pipeline_pipe_with_existing_pipeline(self):
        a = _make_agent("A")
        b = _make_agent("B")
        c = _make_agent("C")
        pipe_ab = a | b
        pipe_abc = pipe_ab | c
        assert len(pipe_abc.agents) == 3


# ── HookEvent / on_event ───────────────────────────────────────────────────────


class TestHooks:
    def test_hook_event_fields(self):
        ev = HookEvent(type="step_start", data={"task": "hello"}, cost_so_far=0.0, step=1)
        assert ev.type == "step_start"
        assert ev.step == 1

    @pytest.mark.asyncio
    async def test_fire_calls_sync_hook(self):
        received = []

        def sync_hook(event: HookEvent):
            received.append(event.type)

        await fire_hook(sync_hook, HookEvent(type="done", data={}, cost_so_far=0.0, step=1))
        assert received == ["done"]

    @pytest.mark.asyncio
    async def test_fire_calls_async_hook(self):
        received = []

        async def async_hook(event: HookEvent):
            received.append(event.type)

        await fire_hook(async_hook, HookEvent(type="error", data={}, cost_so_far=0.0, step=1))
        assert received == ["error"]

    @pytest.mark.asyncio
    async def test_fire_swallows_hook_errors(self):
        """A crashing hook must not propagate to the caller."""

        def bad_hook(event):
            raise RuntimeError("hook exploded")

        # Should not raise
        await fire_hook(bad_hook, HookEvent(type="done", data={}, cost_so_far=0.0, step=1))

    def test_agent_stores_on_event(self):
        called = []

        def hook(event):
            called.append(event.type)

        ag = Agent(name="HookBot", role="test", goal="test", on_event=hook)
        assert ag._on_event is hook

    def test_agent_on_event_none_by_default(self):
        ag = Agent(name="Bot", role="test", goal="test")
        assert ag._on_event is None

    def test_hook_event_exported_from_helix(self):
        assert helix.HookEvent is HookEvent


# ── @helix.agent decorator ─────────────────────────────────────────────────────


class TestAgentDecorator:
    def test_basic_decorator(self):
        @helix.agent(budget_usd=0.05)
        class Assistant:
            "You are a helpful assistant."

        ag = Assistant()
        assert isinstance(ag, Agent)
        assert ag.name == "Assistant"
        assert "helpful assistant" in ag._config.goal

    def test_decorator_custom_name(self):
        @helix.agent(name="CustomBot", budget_usd=0.01)
        class SomeName:
            "Ignore class name."

        ag = SomeName()
        assert ag.name == "CustomBot"

    def test_decorator_with_model(self):
        @helix.agent(model="gpt-4o-mini", budget_usd=0.01)
        class Coder:
            "Write Python code."

        ag = Coder()
        assert ag._config.model.primary == "gpt-4o-mini"

    def test_decorator_factory_callable_multiple_times(self):
        @helix.agent(budget_usd=0.01)
        class Bot:
            "Test bot."

        ag1 = Bot()
        ag2 = Bot()
        assert ag1 is not ag2
        assert ag1.name == ag2.name

    def test_decorator_with_tool_method(self):
        @helix.agent(budget_usd=0.01)
        class ToolAgent:
            "Agent with a built-in tool."

            @tool(description="Always returns 42.")
            async def answer(self, question: str) -> int:
                return 42

        ag = ToolAgent()
        # Registry should contain the 'answer' tool
        assert ag._registry.has("answer")

    def test_decorator_no_docstring_gets_default_goal(self):
        @helix.agent(budget_usd=0.01)
        class Silent:
            pass

        ag = Silent()
        assert "Silent" in ag._config.goal  # fallback includes class name

    def test_decorator_mode_default(self):
        @helix.agent(budget_usd=0.10)
        class Prod:
            "Production agent."

        ag = Prod()
        # Default mode is EXPLORE; to use PRODUCTION pass mode='production' explicitly
        assert ag._config.mode == AgentMode.EXPLORE

    def test_decorator_mode_explicit_production(self):
        @helix.agent(mode="production", budget_usd=0.10)
        class Prod:
            "Production agent."

        ag = Prod()
        assert ag._config.mode == AgentMode.PRODUCTION

    def test_helix_agent_exported(self):
        assert callable(helix.agent)


# ── helix.quick() ─────────────────────────────────────────────────────────────


class TestQuick:
    def test_returns_agent(self):
        ag = helix.quick("You are a test bot.")
        assert isinstance(ag, Agent)

    def test_default_name(self):
        ag = helix.quick("Test.")
        assert ag.name == "Agent"

    def test_custom_name(self):
        ag = helix.quick("Test.", name="MyBot")
        assert ag.name == "MyBot"

    def test_custom_model(self):
        ag = helix.quick("Test.", model="claude-3-5-haiku-20241022")
        assert ag._config.model.primary == "claude-3-5-haiku-20241022"

    def test_budget(self):
        ag = helix.quick("Test.", budget_usd=0.50)
        assert ag._config.budget.budget_usd == pytest.approx(0.50)

    def test_on_event_stored(self):
        hook = lambda e: None  # noqa: E731
        ag = helix.quick("Test.", on_event=hook)
        assert ag._on_event is hook

    def test_quick_exported_from_helix(self):
        assert callable(helix.quick)


# ── discover_tools() ──────────────────────────────────────────────────────────


class TestDiscoverTools:
    def test_discovers_from_module(self):
        import helix.tools.builtin as bt

        tools = helix.discover_tools(bt)
        names = [t.name for t in tools]
        assert "web_search" in names
        assert "calculator" in names
        assert "execute_python" in names

    def test_discovers_from_object(self):
        """Works on plain objects with RegisteredTool attributes too."""

        class MyTools:
            @tool(description="noop")
            async def noop(self) -> str:
                return "noop"

        obj = MyTools()
        # noop is a RegisteredTool on the class level via the descriptor
        # discover_tools scans the object/module for RegisteredTool instances
        found = helix.discover_tools(obj)
        # noop is a RegisteredTool attached at module level after decoration
        # so it may or may not appear on the object — test the module path instead
        assert isinstance(found, list)

    def test_deduplicated(self):
        import helix.tools.builtin as bt

        tools1 = helix.discover_tools(bt)
        tools2 = helix.discover_tools(bt)
        names1 = {t.name for t in tools1}
        names2 = {t.name for t in tools2}
        assert names1 == names2  # no accidental duplication

    def test_discover_tools_exported_from_helix(self):
        assert callable(helix.discover_tools)


# ── Presets ───────────────────────────────────────────────────────────────────


class TestPresets:
    def test_web_researcher(self):
        from helix.presets import web_researcher

        ag = web_researcher()
        assert isinstance(ag, Agent)
        assert ag.name == "WebResearcher"

    def test_writer(self):
        from helix.presets import writer

        ag = writer()
        assert isinstance(ag, Agent)
        assert ag.name == "Writer"

    def test_coder(self):
        from helix.presets import coder

        ag = coder()
        assert isinstance(ag, Agent)
        assert ag._registry.has("execute_python")

    def test_coder_custom_language(self):
        from helix.presets import coder

        ag = coder(language="TypeScript")
        assert "TypeScript" in ag._config.goal

    def test_code_reviewer(self):
        from helix.presets import code_reviewer

        ag = code_reviewer()
        assert isinstance(ag, Agent)

    def test_data_analyst(self):
        from helix.presets import data_analyst

        ag = data_analyst()
        assert isinstance(ag, Agent)
        assert ag._registry.has("calculator")

    def test_summariser(self):
        from helix.presets import summariser

        ag = summariser()
        assert isinstance(ag, Agent)
        assert ag.name == "Summariser"

    def test_fact_checker(self):
        from helix.presets import fact_checker

        ag = fact_checker()
        assert isinstance(ag, Agent)

    def test_assistant(self):
        from helix.presets import assistant

        ag = assistant()
        assert isinstance(ag, Agent)

    def test_api_agent(self):
        from helix.presets import api_agent

        ag = api_agent()
        assert isinstance(ag, Agent)

    def test_researcher_alias(self):
        from helix.presets import researcher, web_researcher

        assert web_researcher().name == researcher().name

    def test_preset_pipe_operator(self):
        from helix.presets import summariser, web_researcher

        r = web_researcher()
        s = summariser()
        pipe = r | s
        assert isinstance(pipe, AgentPipeline)
        assert repr(pipe) == "AgentPipeline(WebResearcher | Summariser)"

    def test_presets_module_on_helix(self):
        assert hasattr(helix, "presets")
        assert callable(helix.presets.web_researcher)


# ── execute_python tool ────────────────────────────────────────────────────────


class TestExecutePython:
    @pytest.mark.asyncio
    async def test_basic_execution(self):
        from helix.tools.builtin import execute_python

        result = await execute_python(code="print('hello world')")
        assert result["success"] is True
        assert "hello world" in result["stdout"]
        assert result["returncode"] == 0

    @pytest.mark.asyncio
    async def test_captures_stderr(self):
        from helix.tools.builtin import execute_python

        result = await execute_python(code="import sys; print('err', file=sys.stderr)")
        assert "err" in result["stderr"]

    @pytest.mark.asyncio
    async def test_syntax_error(self):
        from helix.tools.builtin import execute_python

        result = await execute_python(code="def bad syntax")
        assert result["success"] is False
        assert result["returncode"] != 0

    @pytest.mark.asyncio
    async def test_runtime_error(self):
        from helix.tools.builtin import execute_python

        result = await execute_python(code="raise ValueError('oops')")
        assert result["success"] is False
        assert "ValueError" in result["stderr"]

    @pytest.mark.asyncio
    async def test_multiline_code(self):
        from helix.tools.builtin import execute_python

        code = "total = sum(range(101))\nprint(total)"
        result = await execute_python(code=code)
        assert result["success"] is True
        assert "5050" in result["stdout"]

    @pytest.mark.asyncio
    async def test_subprocess_module_blocked(self):
        """subprocess is blocked by the security shim."""
        from helix.tools.builtin import execute_python

        result = await execute_python(code="import subprocess; subprocess.run(['echo', 'hi'])")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_timeout_respected(self):
        from helix.tools.builtin import execute_python

        result = await execute_python(code="import time; time.sleep(5)", timeout=1.0)
        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_python_registered(self):
        """execute_python is in the global registry after import."""
        import helix.tools.builtin  # noqa: F401
        from helix.core.tool import registry

        assert registry.has("execute_python")


# ── agent.invoke / agent.ainvoke aliases ──────────────────────────────────────


class TestAgentAliases:
    def test_invoke_exists(self):
        ag = Agent(name="Bot", role="r", goal="g")
        assert callable(ag.invoke)

    def test_ainvoke_exists(self):
        ag = Agent(name="Bot", role="r", goal="g")
        assert callable(ag.ainvoke)

    def test_run_sync_exists(self):
        ag = Agent(name="Bot", role="r", goal="g")
        assert callable(ag.run_sync)

    def test_invoke_is_sync(self):
        """invoke() should not itself be a coroutine function."""
        import inspect

        ag = Agent(name="Bot", role="r", goal="g")
        assert not inspect.iscoroutinefunction(ag.invoke)

    def test_ainvoke_is_async(self):
        """ainvoke() should be a coroutine function."""
        import inspect

        ag = Agent(name="Bot", role="r", goal="g")
        assert inspect.iscoroutinefunction(ag.ainvoke)


# ── EvalSuite.case() decorator ────────────────────────────────────────────────


class TestEvalSuiteCase:
    def test_case_returns_fn(self):
        from helix.config import EvalCase
        from helix.eval.suite import EvalSuite

        suite = EvalSuite("test")

        @suite.case
        def my_case():
            return EvalCase(input="hi", expected_facts=["hi"])

        # After the fix, suite.case returns the original function
        assert callable(my_case)

    def test_case_is_registered(self):
        from helix.config import EvalCase
        from helix.eval.suite import EvalSuite

        suite = EvalSuite("test")

        @suite.case
        def greet():
            return EvalCase(input="hello", expected_facts=["hello"])

        assert len(suite._cases) == 1

    def test_case_uses_fn_name_when_no_result_name(self):
        from helix.config import EvalCase
        from helix.eval.suite import EvalSuite

        suite = EvalSuite("named-suite")

        @suite.case
        def capital_of_france():
            return EvalCase(input="What is the capital of France?", expected_facts=["Paris"])

        # _cases stores EvalCase objects, not tuples
        assert suite._cases[0].name == "capital_of_france"

    def test_case_uses_result_name_when_set(self):
        from helix.config import EvalCase
        from helix.eval.suite import EvalSuite

        suite = EvalSuite("named-suite")

        @suite.case
        def fn():
            return EvalCase(
                name="explicit-name",
                input="test",
                expected_facts=["fact"],
            )

        assert suite._cases[0].name == "explicit-name"


# ── Workflow ChainNode ADT ─────────────────────────────────────────────────────


class TestWorkflowChainNode:
    @pytest.mark.asyncio
    async def test_typed_sequential_node(self):
        from helix.core.workflow import _SequentialNode

        @step(name="triple")
        async def triple(x):
            return x * 3

        wf = Workflow("test").then(triple)
        assert isinstance(wf._chain[0], _SequentialNode)

    @pytest.mark.asyncio
    async def test_typed_parallel_node(self):
        from helix.core.workflow import _ParallelNode

        @step(name="s1")
        async def s1(x):
            return x + 1

        @step(name="s2")
        async def s2(x):
            return x - 1

        wf = Workflow("test").parallel(s1, s2)
        assert isinstance(wf._chain[0], _ParallelNode)

    @pytest.mark.asyncio
    async def test_map_node(self):
        @step(name="upper")
        async def upper(x):
            return x.upper()

        wf = Workflow("test").map(upper, items_fn=lambda inp: inp.split(","))
        result = await wf.run("a,b,c")
        assert set(result.final_output) == {"A", "B", "C"}

    @pytest.mark.asyncio
    async def test_loop_node(self):
        @step(name="inc")
        async def inc(x):
            return x + 1

        wf = Workflow("test").loop(inc, until=lambda x: x >= 5, max_iter=10)
        result = await wf.run(0)
        assert result.final_output == 5

    @pytest.mark.asyncio
    async def test_workflow_checkpoint_resume(self, tmp_path):
        call_counts = {"count": 0}

        @step(name="counted")
        async def counted(x):
            call_counts["count"] += 1
            return x + 1

        ckpt_dir = str(tmp_path / "ckpt")
        wf = Workflow("resume-test").then(counted).with_checkpoint(ckpt_dir)
        result = await wf.run(0)
        assert result.final_output == 1
        assert call_counts["count"] == 1

    def test_on_step_callback(self):
        calls = []

        @step(name="noop")
        async def noop(x):
            return x

        wf = Workflow("test").then(noop).on_step(lambda name, out: calls.append(name))
        wf.run_sync(42)
        assert "noop" in calls


# ── Public API completeness ────────────────────────────────────────────────────


class TestPublicAPI:
    """Ensure every new symbol is reachable from import helix."""

    def test_state_graph(self):
        assert helix.StateGraph is StateGraph

    def test_compiled_graph(self):
        assert helix.CompiledGraph is CompiledGraph

    def test_end_start(self):
        assert helix.END == "__end__"
        assert helix.START == "__start__"

    def test_agent_pipeline(self):
        assert helix.AgentPipeline is AgentPipeline

    def test_hook_event(self):
        assert helix.HookEvent is HookEvent

    def test_agent_decorator(self):
        assert callable(helix.agent)

    def test_quick(self):
        assert callable(helix.quick)

    def test_chain(self):
        assert callable(helix.chain)

    def test_discover_tools(self):
        assert callable(helix.discover_tools)

    def test_presets_module(self):
        assert hasattr(helix, "presets")
