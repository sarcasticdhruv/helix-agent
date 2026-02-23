"""
helix/safety/hitl.py

Human-in-the-loop transport implementations.

Three transports:
  CLITransport   — blocks stdin for local dev/testing
  WebhookTransport — fires HTTP POST, polls for response (staging)
  QueueTransport — async message queue backed by Redis/SQS (production)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from helix.config import HITLConfig, HITLDecision, HITLRequest, HITLResponse
from helix.interfaces import HITLTransport


class CLITransport(HITLTransport):
    """
    Blocks stdin and waits for terminal input.
    For local development only — never use in production.
    """

    async def send_request(self, request: HITLRequest) -> HITLResponse:
        print("\n" + "=" * 60)
        print("[HELIX HITL] Human approval required")
        print(f"Agent:   {request.agent_id}")
        print(f"Risk:    {request.risk_level}")
        print(f"Prompt:  {request.prompt}")
        if request.proposed_action:
            print(f"Action:  {request.proposed_action}")
        print("=" * 60)
        print("Options: [a]pprove / [r]eject / [e]scalate")

        try:
            loop = asyncio.get_event_loop()
            raw = await asyncio.wait_for(
                loop.run_in_executor(None, input, "Decision > "),
                timeout=request.timeout_seconds,
            )
            raw = raw.strip().lower()
            if raw in ("a", "approve", "yes", "y"):
                decision = HITLDecision.APPROVE
            elif raw in ("r", "reject", "no", "n"):
                decision = HITLDecision.REJECT
            else:
                decision = HITLDecision.ESCALATE
        except TimeoutError:
            print(f"\n[HELIX HITL] Timeout after {request.timeout_seconds}s — escalating")
            decision = HITLDecision.ESCALATE

        return HITLResponse(request_id=request.id, decision=decision)

    async def health(self) -> bool:
        return True


class WebhookTransport(HITLTransport):
    """
    Fires a POST to a webhook URL and polls for a response.
    Suitable for staging environments with a simple approval service.

    Production use QueueTransport instead — polling is fragile.
    """

    def __init__(
        self,
        webhook_url: str,
        response_url: str,
        poll_interval_s: float = 5.0,
    ) -> None:
        self._webhook_url = webhook_url
        self._response_url = response_url
        self._poll_interval = poll_interval_s

    async def send_request(self, request: HITLRequest) -> HITLResponse:
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required for WebhookTransport: pip install httpx")

        async with httpx.AsyncClient() as client:
            # Fire the request
            try:
                await client.post(
                    self._webhook_url,
                    json=request.model_dump(),
                    timeout=10.0,
                )
            except Exception as e:
                return HITLResponse(
                    request_id=request.id,
                    decision=HITLDecision.ESCALATE,
                    note=f"Webhook delivery failed: {e}",
                )

            # Poll for response
            deadline = time.time() + request.timeout_seconds
            while time.time() < deadline:
                await asyncio.sleep(self._poll_interval)
                try:
                    resp = await client.get(
                        f"{self._response_url}/{request.id}",
                        timeout=5.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        return HITLResponse(
                            request_id=request.id,
                            decision=HITLDecision(data.get("decision", "escalate")),
                            reviewer_id=data.get("reviewer_id"),
                            note=data.get("note"),
                        )
                except Exception:
                    continue

        return HITLResponse(request_id=request.id, decision=HITLDecision.ESCALATE)

    async def health(self) -> bool:
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                resp = await client.get(self._response_url, timeout=5.0)
                return resp.status_code < 500
        except Exception:
            return False


class QueueTransport(HITLTransport):
    """
    Async message queue transport backed by Redis.

    Agent publishes request to helix:hitl:requests.
    Approval service publishes response to helix:hitl:responses:{request_id}.
    Agent awaits response — no polling.

    This is the production transport.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379") -> None:
        self._redis_url = redis_url
        self._redis: Any | None = None

    async def _get_redis(self) -> Any:
        if self._redis is None:
            try:
                import redis.asyncio as aioredis

                self._redis = await aioredis.from_url(self._redis_url)
            except ImportError:
                raise ImportError("redis package required: pip install redis")
        return self._redis

    async def send_request(self, request: HITLRequest) -> HITLResponse:
        try:
            r = await self._get_redis()
            # Publish request
            await r.lpush(
                "helix:hitl:requests",
                json.dumps(request.model_dump(), default=str),
            )
            # Wait for response on per-request channel
            response_key = f"helix:hitl:responses:{request.id}"
            result = await r.brpop(
                response_key,
                timeout=int(request.timeout_seconds),
            )
            if result:
                _, raw = result
                data = json.loads(raw)
                return HITLResponse(
                    request_id=request.id,
                    decision=HITLDecision(data.get("decision", "escalate")),
                    reviewer_id=data.get("reviewer_id"),
                    note=data.get("note"),
                )
        except ImportError:
            raise
        except Exception:
            pass  # Fall through to timeout response

        return HITLResponse(request_id=request.id, decision=HITLDecision.ESCALATE)

    async def health(self) -> bool:
        try:
            r = await self._get_redis()
            await r.ping()
            return True
        except Exception:
            return False


class HITLController:
    """
    Façade that selects and delegates to the correct HITLTransport.
    Constructed from HITLConfig.
    """

    def __init__(self, config: HITLConfig) -> None:
        self._config = config
        self._transport = self._build_transport()

    def _build_transport(self) -> HITLTransport:
        transport_name = self._config.transport
        tc = self._config.transport_config
        if transport_name == "cli":
            return CLITransport()
        if transport_name == "webhook":
            return WebhookTransport(
                webhook_url=tc.get("webhook_url", ""),
                response_url=tc.get("response_url", ""),
                poll_interval_s=tc.get("poll_interval_s", 5.0),
            )
        if transport_name == "queue":
            return QueueTransport(redis_url=tc.get("redis_url", "redis://localhost:6379"))
        return CLITransport()

    def should_trigger(
        self,
        confidence: float | None = None,
        tool_name: str | None = None,
        cost_usd: float | None = None,
    ) -> bool:
        """Return True if HITL should be triggered for this action."""
        if not self._config.enabled:
            return False
        if confidence is not None and self._config.on_confidence_below:
            if confidence < self._config.on_confidence_below:
                return True
        if tool_name and tool_name in self._config.on_tool_risk:
            return True
        if cost_usd is not None and self._config.on_cost_above_usd:
            if cost_usd > self._config.on_cost_above_usd:
                return True
        return False

    async def send_request(self, request: HITLRequest) -> HITLResponse:
        return await self._transport.send_request(request)

    async def health(self) -> bool:
        return await self._transport.health()
