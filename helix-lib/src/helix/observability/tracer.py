"""
helix/observability/tracer.py

Tracer — span-based execution tracing for every agent run.

All spans are in-memory during the run, then exported
to the configured backend (local JSON, OTel, S3) on finalization.
Never blocks the agent execution path.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from helix.context import ExecutionContext


class Span:
    """A single unit of traced work."""

    def __init__(self, name: str, parent_id: Optional[str] = None, **meta: Any) -> None:
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.parent_id = parent_id
        self.meta: Dict[str, Any] = dict(meta)
        self.start_time = time.monotonic()
        self.end_time: Optional[float] = None
        self.error: Optional[str] = None

    def finish(self, error: Optional[str] = None) -> None:
        self.end_time = time.monotonic()
        if error:
            self.error = error

    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "parent_id": self.parent_id,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "meta": self.meta,
        }


class Tracer:
    """
    Collects spans during an agent run and exports them on completion.
    """

    def __init__(self, run_id: str, agent_id: str, agent_name: str) -> None:
        self._run_id = run_id
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._spans: List[Span] = []
        self._active_span: Optional[Span] = None
        self._start_time = time.time()

    @contextlib.contextmanager
    def span(self, name: str, **meta: Any) -> Generator[Span, None, None]:
        """Context manager for a traced block."""
        s = Span(
            name=name,
            parent_id=self._active_span.id if self._active_span else None,
            **meta,
        )
        prev = self._active_span
        self._active_span = s
        self._spans.append(s)
        try:
            yield s
        except Exception as e:
            s.finish(error=str(e))
            raise
        else:
            s.finish()
        finally:
            self._active_span = prev

    def log_step(self, step: int, response_content: str, model: str) -> None:
        self._spans.append(Span(
            name=f"step.{step}",
            model=model,
            content_preview=response_content[:100],
        ))
        self._spans[-1].finish()

    def log_llm_call(self, model: str, tokens: Dict, cost_usd: float) -> None:
        s = Span("llm.call", model=model, tokens=tokens, cost_usd=cost_usd)
        s.finish()
        self._spans.append(s)

    def finalize(self, ctx: ExecutionContext) -> None:
        """Export spans to configured backend after run completes."""
        asyncio.create_task(self._export(ctx))

    async def _export(self, ctx: ExecutionContext) -> None:
        """Write trace to local JSON file."""
        try:
            trace_dir = Path(".helix/traces")
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = trace_dir / f"{ctx.run_id}.json"
            trace = self.export()
            trace_path.write_text(json.dumps(trace, indent=2, default=str))
        except Exception:
            pass

    def export(self) -> Dict[str, Any]:
        return {
            "run_id": self._run_id,
            "agent_id": self._agent_id,
            "agent_name": self._agent_name,
            "started_at": self._start_time,
            "duration_s": time.time() - self._start_time,
            "spans": [s.to_dict() for s in self._spans],
            "span_count": len(self._spans),
        }

    def export_json(self) -> str:
        return json.dumps(self.export(), indent=2, default=str)
