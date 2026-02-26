"""
helix/models/providers/gemini.py

Google Gemini provider.
Uses the new google-genai SDK (replaces the deprecated google-generativeai).

Install:  pip install google-genai
Models:   gemini-2.5-flash (default), gemini-2.0-flash, gemini-2.5-pro
Env var:  GOOGLE_API_KEY  or  GEMINI_API_KEY

Migration note:
  Old: google-generativeai  →  import google.generativeai as genai
  New: google-genai          →  from google import genai; client = genai.Client(...)
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from typing import Any

from helix.config import ModelResponse, TokenUsage
from helix.errors import HelixProviderError
from helix.interfaces import LLMProvider


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = (
            api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        )

    def _get_client(self):
        """Return an authenticated google.genai Client."""
        try:
            from google import genai
        except ImportError as err:
            raise ImportError("pip install google-genai") from err
        if not self._api_key:
            raise HelixProviderError(
                provider="google",
                model="gemini",
                reason="GOOGLE_API_KEY not set. Run: helix config set GOOGLE_API_KEY your-key",
                retryable=False,
            )
        return genai.Client(api_key=self._api_key)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        model: str = "gemini-2.5-flash",
        tools: list[dict] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> ModelResponse:
        try:
            from google.genai import types

            client = self._get_client()

            system_text = self._extract_system(messages)
            contents = self._build_contents(messages)

            cfg = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_text if system_text else None,
            )

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=cfg,
                ),
            )

            content = response.text or ""

            meta = getattr(response, "usage_metadata", None)
            usage = TokenUsage(
                prompt_tokens=getattr(meta, "prompt_token_count", 0) or 0,
                completion_tokens=getattr(meta, "candidates_token_count", 0) or 0,
            )

            return ModelResponse(
                content=content,
                usage=usage,
                model=model,
                provider="google",
                finish_reason="stop",
            )

        except HelixProviderError:
            raise
        except Exception as e:
            err_str = str(e).lower()
            retryable = any(
                k in err_str
                for k in ("rate", "quota", "timeout", "503", "429", "resource exhausted")
            )
            raise HelixProviderError(
                model=model,
                provider="google",
                original=e,
                retryable=retryable,
            ) from e

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str = "gemini-2.5-flash",
        **kwargs,
    ) -> AsyncIterator[str]:
        try:
            from google.genai import types

            client = self._get_client()
            system_text = self._extract_system(messages)
            contents = self._build_contents(messages)

            cfg = types.GenerateContentConfig(
                system_instruction=system_text if system_text else None,
            )

            loop = asyncio.get_event_loop()
            response_iter = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content_stream(
                    model=model,
                    contents=contents,
                    config=cfg,
                ),
            )
            for chunk in response_iter:
                text = getattr(chunk, "text", None) or ""
                if text:
                    yield text
        except HelixProviderError:
            raise
        except Exception as e:
            raise HelixProviderError(model=model, provider="google", original=e) from e

    async def count_tokens(self, messages: list[dict], model: str) -> int:
        try:
            client = self._get_client()
            contents = self._build_contents(messages)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: client.models.count_tokens(model=model, contents=contents),
            )
            return getattr(result, "total_tokens", None) or max(1, len(str(contents)) // 4)
        except Exception:
            # Fall back to character estimate if API call fails
            text = " ".join(str(m.get("content", "")) for m in messages)
            return max(1, len(text) // 4)

    def supported_models(self) -> list[str]:
        return [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ]

    async def health(self) -> bool:
        try:
            client = self._get_client()
            loop = asyncio.get_event_loop()
            # list() forces the generator to run so we actually hit the API
            await loop.run_in_executor(None, lambda: list(client.models.list()))
            return True
        except Exception:
            return False

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _extract_system(self, messages: list[dict]) -> str:
        """Collect all system-role messages into one string."""
        parts = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "") or ""
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") if isinstance(c, dict) else str(c) for c in content
                    )
                parts.append(content)
        return "\n\n".join(parts)

    def _build_contents(self, messages: list[dict]) -> list[dict]:
        """
        Convert Helix message dicts to the google-genai `contents` format:
        [{"role": "user"|"model", "parts": [{"text": "..."}]}, ...]
        System messages are handled separately via GenerateContentConfig.
        """
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                continue  # injected via system_instruction
            # Normalise role: assistant → model
            genai_role = "model" if role == "assistant" else "user"
            content = msg.get("content", "") or ""
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c) for c in content
                )
            contents.append({"role": genai_role, "parts": [{"text": content}]})
        # google-genai requires the last message to be from 'user'
        if not contents:
            contents = [{"role": "user", "parts": [{"text": "Hello"}]}]
        return contents
