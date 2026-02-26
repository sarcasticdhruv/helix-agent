"""
helix/core/graph.py

StateGraph — LangGraph-compatible directed graph with typed shared state.

Design goals:
  - Drop-in familiar API for LangGraph users (add_node, add_edge,
    add_conditional_edges, set_entry_point, compile, invoke).
  - Nodes can be plain async/sync callables, Helix Agent instances,
    or any object with a ``run`` coroutine.
  - State is a shared dict (or TypedDict / Pydantic model) that every
    node can read from and write partial updates back to.
  - Full cycle support: a node may route back to a previously-visited
    node, enabling while-loop patterns without arbitrary depth limits.
  - Checkpointing: pass ``checkpoint_dir`` to ``compile()`` and the
    graph saves/loads state after every node for resume-after-crash.

Quickstart::

    import helix
    from typing import TypedDict

    class State(TypedDict):
        topic: str
        draft: str
        ready: bool

    researcher = helix.presets.web_researcher()
    writer     = helix.presets.writer()

    async def research_node(state: State) -> dict:
        result = await researcher.run(state["topic"])
        return {"draft": result.output}

    async def write_node(state: State) -> dict:
        result = await writer.run(state["draft"])
        return {"draft": result.output, "ready": True}

    def router(state: State) -> str:
        return "done" if state.get("ready") else "write"

    graph = (
        helix.StateGraph(State)
        .add_node("research", research_node)
        .add_node("write", write_node)
        .add_edge("research", "write")
        .add_conditional_edges("write", router, {"write": "write", "done": helix.END})
        .set_entry_point("research")
        .compile()
    )

    result = graph.run_sync({"topic": "Quantum computing in 2026", "draft": "", "ready": False})
    print(result["draft"])
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import time
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Sentinel constants (LangGraph-compatible names)
# ---------------------------------------------------------------------------

END = "__end__"
START = "__start__"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class NodeResult:
    """Result of a single node execution."""

    node: str
    state_update: dict[str, Any]
    duration_s: float = 0.0
    error: str | None = None


@dataclass
class GraphResult:
    """Final result of a compiled graph run."""

    state: dict[str, Any]
    nodes_visited: list[str] = field(default_factory=list)
    total_cost_usd: float = 0.0
    duration_s: float = 0.0
    error: str | None = None

    # Convenience: act like a dict for state access
    def __getitem__(self, key: str) -> Any:
        return self.state[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.state


# ---------------------------------------------------------------------------
# StateGraph
# ---------------------------------------------------------------------------


class StateGraph:
    """
    A directed graph of nodes that share a typed state dict.

    Each node receives the current full state and returns a *partial*
    dict of keys to update (similar to LangGraph's reducer pattern).
    Edges control which node runs next; conditional edges call a
    router function and map its return value to a node name.

    Nodes may be:
    - ``async def fn(state: dict) -> dict``  — async function
    - ``def fn(state: dict) -> dict``         — sync function (run in executor)
    - A :class:`helix.Agent` instance         — run with ``state["task"]``

    Example::

        graph = (
            helix.StateGraph(State)
            .add_node("a", fn_a)
            .add_node("b", fn_b)
            .add_edge("a", "b")
            .set_entry_point("a")
            .set_finish_point("b")
            .compile()
        )
        result = graph.run_sync(initial_state)
    """

    def __init__(self, state_schema: type | None = None) -> None:
        """
        Args:
            state_schema: Optional TypedDict class, Pydantic model, or ``dict``.
                          Used for documentation and runtime type hints only;
                          the graph does not enforce types at runtime.
        """
        self._state_schema = state_schema
        self._nodes: dict[str, Any] = {}
        self._edges: dict[str, str | dict[str, Any]] = {}
        self._entry_point: str | None = None
        self._finish_points: set[str] = set()

    # ------------------------------------------------------------------
    # Builder API (all methods return self for chaining)
    # ------------------------------------------------------------------

    def add_node(self, name: str, fn: Any) -> StateGraph:
        """
        Register a node.

        Args:
            name: Unique node identifier used in edges.
            fn:   Callable ``(state) -> dict`` or a :class:`helix.Agent`.

        Returns:
            self (for chaining)
        """
        self._nodes[name] = fn
        return self

    def add_edge(self, source: str, destination: str) -> StateGraph:
        """
        Add an unconditional edge from *source* to *destination*.

        Use the sentinel :data:`helix.END` as *destination* to mark a
        terminal edge.

        Returns:
            self (for chaining)
        """
        self._edges[source] = destination
        return self

    def add_conditional_edges(
        self,
        source: str,
        condition: Callable[[dict[str, Any]], str],
        edge_map: dict[str, str],
    ) -> StateGraph:
        """
        Add conditional routing from *source*.

        After *source* executes, ``condition(state)`` is called.
        Its return value is looked up in *edge_map* to find the
        next node name (or :data:`helix.END`).

        Args:
            source:    Name of the source node.
            condition: ``(state) -> str`` router function.
            edge_map:  Mapping of router return values to node names.

        Returns:
            self (for chaining)

        Example::

            def router(state):
                return "done" if state["quality"] >= 0.8 else "revise"

            graph.add_conditional_edges(
                "grade",
                router,
                {"done": END, "revise": "write"},
            )
        """
        self._edges[source] = {"condition": condition, "map": edge_map}
        return self

    def set_entry_point(self, node: str) -> StateGraph:
        """Set the first node to execute. Required before ``compile()``."""
        self._entry_point = node
        return self

    def set_finish_point(self, node: str) -> StateGraph:
        """Mark *node* as a terminal node (implicitly adds edge to END)."""
        self._finish_points.add(node)
        return self

    # Alias to match LangGraph naming
    add_finish_edge = set_finish_point

    def compile(
        self,
        checkpoint_dir: str | Path | None = None,
        max_steps: int = 100,
    ) -> CompiledGraph:
        """
        Validate and return a runnable :class:`CompiledGraph`.

        Args:
            checkpoint_dir: If set, state is persisted as JSON after each
                            node and can be resumed with ``run(resume=True)``.
            max_steps:      Hard cap on node executions to prevent infinite loops.

        Raises:
            ValueError: If entry point is not set or references an unknown node.
        """
        if not self._entry_point:
            raise ValueError(
                "StateGraph has no entry point. Call .set_entry_point('node_name') first."
            )
        if self._entry_point not in self._nodes:
            raise ValueError(
                f"Entry point '{self._entry_point}' is not a registered node. "
                f"Registered nodes: {list(self._nodes)}"
            )
        return CompiledGraph(
            graph=self,
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
            max_steps=max_steps,
        )

    def __repr__(self) -> str:
        nodes = list(self._nodes)
        return (
            f"StateGraph(nodes={nodes!r}, "
            f"entry={self._entry_point!r}, "
            f"finish={sorted(self._finish_points)!r})"
        )


# ---------------------------------------------------------------------------
# CompiledGraph
# ---------------------------------------------------------------------------


class CompiledGraph:
    """
    Runnable compiled graph. Obtained via :meth:`StateGraph.compile`.

    Use :meth:`run` (async) or :meth:`run_sync` / :meth:`invoke` (sync).
    """

    def __init__(
        self,
        graph: StateGraph,
        checkpoint_dir: Path | None,
        max_steps: int,
    ) -> None:
        self._graph = graph
        self._checkpoint_dir = checkpoint_dir
        self._max_steps = max_steps

    # ------------------------------------------------------------------
    # Public run API
    # ------------------------------------------------------------------

    async def run(
        self,
        initial_state: dict[str, Any],
        *,
        run_id: str | None = None,
        resume: bool = False,
    ) -> GraphResult:
        """
        Execute the graph asynchronously.

        Args:
            initial_state: Starting state dict.
            run_id:        Optional identifier for checkpointing.
            resume:        If True and a checkpoint exists, resume from it.

        Returns:
            :class:`GraphResult` with ``state``, ``nodes_visited``, and cost info.
        """
        start = time.perf_counter()
        state = copy.deepcopy(initial_state)
        nodes_visited: list[str] = []
        total_cost = 0.0

        # Resume from checkpoint
        checkpoint_start_node: str | None = None
        if resume and self._checkpoint_dir and run_id:
            saved = self._load_checkpoint(run_id)
            if saved:
                state = saved["state"]
                nodes_visited = saved["nodes_visited"]
                checkpoint_start_node = saved.get("next_node")

        current = checkpoint_start_node or self._graph._entry_point
        steps = 0

        try:
            while current and current != END:
                if steps >= self._max_steps:
                    return GraphResult(
                        state=state,
                        nodes_visited=nodes_visited,
                        total_cost_usd=total_cost,
                        duration_s=time.perf_counter() - start,
                        error=f"Max steps ({self._max_steps}) reached. "
                        f"Last node: '{current}'. Use StateGraph.compile(max_steps=N) to raise the limit.",
                    )

                if current not in self._graph._nodes:
                    return GraphResult(
                        state=state,
                        nodes_visited=nodes_visited,
                        total_cost_usd=total_cost,
                        duration_s=time.perf_counter() - start,
                        error=f"Node '{current}' not found. Registered: {list(self._graph._nodes)}",
                    )

                nodes_visited.append(current)
                fn = self._graph._nodes[current]

                # Execute node
                update, cost = await self._run_node(fn, state)
                total_cost += cost

                # Merge partial state update
                if isinstance(update, dict):
                    state.update(update)

                # Save checkpoint
                if self._checkpoint_dir and run_id:
                    self._save_checkpoint(run_id, state, nodes_visited, nxt=None)

                # Determine next node
                edge = self._graph._edges.get(current)
                if edge is None:
                    # No explicit edge: check finish points
                    if current in self._graph._finish_points:
                        current = END
                    else:
                        # Implicit end if no edge defined
                        current = END
                elif isinstance(edge, str):
                    current = edge
                elif isinstance(edge, dict):
                    condition_fn = edge["condition"]
                    edge_map = edge["map"]
                    key = (
                        await asyncio.get_event_loop().run_in_executor(None, condition_fn, state)
                        if not asyncio.iscoroutinefunction(condition_fn)
                        else await condition_fn(state)
                    )
                    current = edge_map.get(key, END)
                else:
                    current = END

                steps += 1

        except Exception as exc:
            return GraphResult(
                state=state,
                nodes_visited=nodes_visited,
                total_cost_usd=total_cost,
                duration_s=time.perf_counter() - start,
                error=str(exc),
            )

        # Clear checkpoint on success
        if self._checkpoint_dir and run_id:
            self._clear_checkpoint(run_id)

        return GraphResult(
            state=state,
            nodes_visited=nodes_visited,
            total_cost_usd=total_cost,
            duration_s=time.perf_counter() - start,
        )

    def run_sync(
        self,
        initial_state: dict[str, Any],
        *,
        run_id: str | None = None,
        resume: bool = False,
    ) -> GraphResult:
        """
        Synchronous wrapper around :meth:`run`.

        Safe to call from plain scripts and notebooks (handles any
        existing event loop automatically).

        Example::

            result = graph.run_sync({"topic": "AI safety"})
            print(result["output"])
        """
        coro = self.run(initial_state, run_id=run_id, resume=resume)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, coro).result()
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    # LangGraph-compatible alias
    async def ainvoke(
        self,
        initial_state: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async run returning raw state dict (LangGraph ``ainvoke`` compat)."""
        result = await self.run(initial_state, **kwargs)
        if result.error:
            raise RuntimeError(result.error)
        return result.state

    def invoke(
        self,
        initial_state: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Sync run returning raw state dict (LangGraph ``invoke`` compat)."""
        result = self.run_sync(initial_state, **kwargs)
        if result.error:
            raise RuntimeError(result.error)
        return result.state

    async def stream(
        self,
        initial_state: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> AsyncIterator[NodeResult]:
        """
        Async generator that yields a :class:`NodeResult` after every node.

        Example::

            async for node_result in graph.stream(state):
                print(f"  [{node_result.node}] → {node_result.state_update}")
        """
        state = copy.deepcopy(initial_state)
        current: str | None = self._graph._entry_point
        steps = 0

        while current and current != END and steps < self._max_steps:
            if current not in self._graph._nodes:
                break

            fn = self._graph._nodes[current]
            t0 = time.perf_counter()
            try:
                update, _ = await self._run_node(fn, state)
                if isinstance(update, dict):
                    state.update(update)
                yield NodeResult(
                    node=current,
                    state_update=update if isinstance(update, dict) else {},
                    duration_s=time.perf_counter() - t0,
                )
            except Exception as exc:
                yield NodeResult(
                    node=current,
                    state_update={},
                    duration_s=time.perf_counter() - t0,
                    error=str(exc),
                )
                return

            # Advance
            edge = self._graph._edges.get(current)
            if edge is None:
                current = END
            elif isinstance(edge, str):
                current = edge
            elif isinstance(edge, dict):
                condition_fn = edge["condition"]
                edge_map = edge["map"]
                key = (
                    await asyncio.get_event_loop().run_in_executor(None, condition_fn, state)
                    if not asyncio.iscoroutinefunction(condition_fn)
                    else await condition_fn(state)
                )
                current = edge_map.get(key, END)
            else:
                current = END
            steps += 1

    # ------------------------------------------------------------------
    # Node dispatcher — handles callables and Agent instances
    # ------------------------------------------------------------------

    async def _run_node(
        self,
        fn: Any,
        state: dict[str, Any],
    ) -> tuple[dict[str, Any], float]:
        """Run *fn* with *state* and return (state_update, cost_usd)."""
        # Helix Agent duck-typed check
        if hasattr(fn, "run") and hasattr(fn, "_config"):
            task = state.get("task") or state.get("input") or state.get("messages", [""])[-1] or ""
            result = await fn.run(str(task))
            return {"output": result.output, "last_agent": fn.name}, result.cost_usd

        # Async callable
        if asyncio.iscoroutinefunction(fn):
            update = await fn(state)
            return (update or {}), 0.0

        # Sync callable — run in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        update = await loop.run_in_executor(None, fn, state)
        return (update or {}), 0.0

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _checkpoint_path(self, run_id: str) -> Path:
        assert self._checkpoint_dir is not None
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self._checkpoint_dir / f"{run_id}.json"

    def _save_checkpoint(
        self,
        run_id: str,
        state: dict[str, Any],
        nodes_visited: list[str],
        nxt: str | None,
    ) -> None:
        try:
            path = self._checkpoint_path(run_id)
            path.write_text(
                json.dumps(
                    {"state": state, "nodes_visited": nodes_visited, "next_node": nxt},
                    default=str,
                )
            )
        except Exception:
            pass

    def _load_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        try:
            path = self._checkpoint_path(run_id)
            if path.exists():
                return json.loads(path.read_text())
        except Exception:
            pass
        return None

    def _clear_checkpoint(self, run_id: str) -> None:
        with contextlib.suppress(Exception):
            self._checkpoint_path(run_id).unlink(missing_ok=True)

    def __repr__(self) -> str:
        g = self._graph
        return f"CompiledGraph(nodes={list(g._nodes)!r}, entry={g._entry_point!r})"
