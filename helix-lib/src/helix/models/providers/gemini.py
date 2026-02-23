"""
helix/models/providers/gemini.py

Google Gemini provider.
Compatible with google-generativeai 0.7.x and 0.8.x.

Install:  pip install google-generativeai
Models:   gemini-2.5-flash (default), gemini-2.0-flash, gemini-2.5-pro
Env var:  GOOGLE_API_KEY  or  GEMINI_API_KEY
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

    def _get_genai(self):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("pip install google-generativeai")
        if not self._api_key:
            raise HelixProviderError(
                provider="google",
                model="gemini",
                reason="GOOGLE_API_KEY not set. Run: helix config set GOOGLE_API_KEY your-key",
                retryable=False,
            )
        genai.configure(api_key=self._api_key)
        return genai

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
            genai = self._get_genai()

            # Build a single prompt string from messages (most compatible path)
            prompt = self._build_prompt(messages)

            gen_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            # Get system instruction if present
            system_text = self._extract_system(messages)

            # Create model — system_instruction supported in 0.7+
            try:
                client = genai.GenerativeModel(
                    model_name=model,
                    system_instruction=system_text if system_text else None,
                    generation_config=gen_config,
                )
            except TypeError:
                # Very old SDK — no system_instruction param
                client = genai.GenerativeModel(
                    model_name=model,
                    generation_config=gen_config,
                )
                if system_text:
                    prompt = f"[Instructions: {system_text}]\n\n{prompt}"

            # Check if we have a multi-turn conversation
            history = self._build_history(messages)
            if history:
                # Multi-turn: use chat session
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: self._chat_complete(client, history, prompt)
                )
            else:
                # Single turn: simplest path — just pass the string
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, lambda: client.generate_content(prompt))

            # Extract text
            content = self._extract_text(response)

            # Usage metadata
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
            )

    def _chat_complete(self, client, history: list[dict], last_msg: str):
        """Use ChatSession for multi-turn conversations."""
        chat = client.start_chat(history=history)
        return chat.send_message(last_msg)

    def _extract_text(self, response) -> str:
        """Safely extract text from any response format."""
        # Try .text property first
        try:
            t = response.text
            if t:
                return t
        except Exception:
            pass
        # Try candidates
        try:
            for candidate in response.candidates:
                parts = getattr(getattr(candidate, "content", None), "parts", [])
                texts = [getattr(p, "text", "") for p in parts if getattr(p, "text", "")]
                if texts:
                    return "".join(texts)
        except Exception:
            pass
        return ""

    def _extract_system(self, messages: list[dict]) -> str:
        """Extract system message text."""
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

    def _build_prompt(self, messages: list[dict]) -> str:
        """Build the final user turn as a plain string."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "") or ""
                if isinstance(content, list):
                    content = " ".join(
                        c.get("text", "") if isinstance(c, dict) else str(c) for c in content
                    )
                return content
        return "Hello"

    def _build_history(self, messages: list[dict]) -> list[dict]:
        """
        Build chat history for multi-turn conversations.
        Returns list of {"role": "user"|"model", "parts": [{"text": "..."}]}
        Only includes prior turns — the final user message is handled separately.
        """
        turns = []
        user_msgs = []
        non_system = [m for m in messages if m.get("role") != "system"]

        # If there's only one user message and no assistant turns, no history needed
        if len(non_system) <= 1:
            return []

        for i, msg in enumerate(non_system):
            role = msg.get("role", "user")
            content = msg.get("content", "") or ""
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c) for c in content
                )

            is_last = i == len(non_system) - 1
            if role == "user":
                if is_last:
                    break  # final user turn handled outside
                turns.append({"role": "user", "parts": [{"text": content}]})
            elif role == "assistant":
                turns.append({"role": "model", "parts": [{"text": content}]})

        return turns

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str = "gemini-2.5-flash",
        **kwargs,
    ) -> AsyncIterator[str]:
        try:
            genai = self._get_genai()
            prompt = self._build_prompt(messages)
            client = genai.GenerativeModel(model_name=model)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: client.generate_content(prompt, stream=True)
            )
            for chunk in response:
                text = self._extract_text(chunk)
                if text:
                    yield text
        except HelixProviderError:
            raise
        except Exception as e:
            raise HelixProviderError(model=model, provider="google", original=e)

    async def count_tokens(self, messages: list[dict], model: str) -> int:
        text = self._build_prompt(messages)
        return max(1, len(text) // 4)

    def supported_models(self) -> list[str]:
        return [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-1.5-flash",  # may be unavailable in some regions
            "gemini-1.5-pro",  # may be unavailable in some regions
        ]

    async def health(self) -> bool:
        try:
            genai = self._get_genai()
            list(genai.list_models())
            return True
        except Exception:
            return False
