"""
helix/runtime/engine.py

Runtime — the top-level execution environment.

Manages:
  - Worker pool for concurrent agent runs
  - Event/webhook routing
  - Scheduled tasks (cron)
  - Queue-based task ingestion
  - Health monitoring
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from collections.abc import Callable
from typing import Any

from helix.config import RuntimeConfig


class Runtime:
    """
    The Helix runtime. Run agents and workflows in a managed environment.

    Usage::

        rt = Runtime(config=RuntimeConfig(workers=4))

        rt.register(my_agent, name="researcher")
        rt.on_event("report.requested", handler=my_agent.run)
        rt.on_schedule("0 9 * * MON", handler=weekly_summary)

        await rt.start()
    """

    def __init__(self, config: RuntimeConfig | None = None) -> None:
        self._config = config or RuntimeConfig()
        self._agents: dict[str, Any] = {}
        self._workflows: dict[str, Any] = {}
        self._event_handlers: dict[str, list[Callable]] = {}
        self._schedule_handlers: list[tuple] = []  # (cron, handler)
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self._config.queue_max_size)
        self._workers: list[asyncio.Task] = []
        self._running: bool = False
        self._run_count: int = 0
        self._total_cost: float = 0.0
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, obj: Any, name: str | None = None) -> Runtime:
        """Register an Agent or Workflow with the runtime."""
        obj_name = name or getattr(obj, "name", str(uuid.uuid4())[:8])
        if hasattr(obj, "run") and hasattr(obj, "config"):
            self._agents[obj_name] = obj
        else:
            self._workflows[obj_name] = obj
        return self

    def on_event(self, event_name: str, handler: Callable) -> Runtime:
        """Register a handler for a named event."""
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)
        return self

    def on_schedule(self, cron: str, handler: Callable) -> Runtime:
        """Register a handler on a cron schedule (requires croniter)."""
        self._schedule_handlers.append((cron, handler))
        return self

    def subscribe(self, topic: str, handler: Callable) -> Runtime:
        """Alias for on_event — pub/sub style."""
        return self.on_event(topic, handler)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def run(self, name: str, task: str) -> Any:
        """Run a named agent or workflow and return result."""
        if name in self._agents:
            result = await self._agents[name].run(task)
            self._run_count += 1
            self._total_cost += getattr(result, "cost_usd", 0.0)
            return result
        if name in self._workflows:
            return await self._workflows[name].run(task)
        raise KeyError(f"No agent or workflow registered with name '{name}'")

    def run_sync(self, name: str, task: str) -> Any:
        return asyncio.run(self.run(name, task))

    async def emit(self, event_name: str, payload: Any) -> None:
        """Emit a named event and invoke all registered handlers."""
        handlers = self._event_handlers.get(event_name, [])
        if not handlers:
            return
        coros = []
        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                coros.append(handler(payload))
            else:
                coros.append(asyncio.to_thread(handler, payload))
        await asyncio.gather(*coros, return_exceptions=True)

    async def emit_webhook(self, payload: dict) -> None:
        """Route a webhook payload to the correct event handler."""
        event_name = payload.get("event") or payload.get("type") or "unknown"
        await self.emit(event_name, payload)

    async def enqueue(self, name: str, task: str) -> None:
        """Add a task to the async work queue."""
        await self._queue.put((name, task))

    def publish(self, topic: str, payload: Any) -> None:
        """Publish synchronously to a topic."""
        asyncio.create_task(self.emit(topic, payload))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the runtime — spawns workers and blocks until stop() is called."""
        self._running = True
        self._start_time = time.time()

        # Spawn worker pool
        for _ in range(self._config.workers):
            task = asyncio.create_task(self._worker())
            self._workers.append(task)

        # Spawn schedule runner
        if self._schedule_handlers:
            asyncio.create_task(self._schedule_runner())

        # Block
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*self._workers)

    async def stop(self) -> None:
        """Gracefully stop the runtime."""
        self._running = False
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _worker(self) -> None:
        """Worker coroutine — pulls tasks from queue and executes them."""
        while self._running:
            try:
                name, task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                try:
                    await self.run(name, task)
                except Exception:
                    pass
                finally:
                    self._queue.task_done()
            except TimeoutError:
                continue
            except Exception:
                continue

    async def _schedule_runner(self) -> None:
        """Checks scheduled handlers every minute."""
        try:
            from croniter import croniter
        except ImportError:
            return  # croniter not installed — skip schedules

        while self._running:
            now = time.time()
            for cron_expr, handler in self._schedule_handlers:
                try:
                    cron = croniter(cron_expr)
                    prev = cron.get_prev(float)
                    if abs(now - prev) < 60:  # Within the last minute
                        if asyncio.iscoroutinefunction(handler):
                            asyncio.create_task(handler())
                        else:
                            asyncio.create_task(asyncio.to_thread(handler))
                except Exception:
                    continue
            await asyncio.sleep(60)

    def status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "workers_alive": sum(1 for w in self._workers if not w.done()),
            "queue_depth": self._queue.qsize(),
            "registered_agents": list(self._agents.keys()),
            "registered_workflows": list(self._workflows.keys()),
            "total_runs": self._run_count,
            "total_cost_usd": round(self._total_cost, 4),
            "uptime_s": round(time.time() - self._start_time, 1) if self._start_time else 0,
        }
