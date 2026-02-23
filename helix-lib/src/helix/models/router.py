"""
helix/models/router.py

ModelRouter — intelligent LLM call routing across all supported providers.

Supported providers (auto-detected from model name prefix):
  OpenAI      gpt-*, o1, o3-*
  Anthropic   claude-*
  Google      gemini-*
  Groq        llama-*, mixtral-*, gemma-* (via Groq)  OR  groq/*
  Mistral     mistral-*, codestral-*, pixtral-*, open-*
  Cohere      command-*
  Together    */* (slash in name) or together/*
  Ollama      ollama/* or local/*
  Azure       azure/*
  OpenRouter  openrouter/*
  DeepSeek    deepseek-*
  xAI         grok-*
  Generic     Any base_url via OpenAICompatProvider
"""

from __future__ import annotations

import os
from typing import Any, AsyncIterator, Dict, List, Optional

from helix.config import ComplexityTier, ModelResponse, TokenUsage
from helix.errors import AllModelsExhaustedError, HelixProviderError


# ---------------------------------------------------------------------------
# Pricing table (USD per 1K tokens)
# ---------------------------------------------------------------------------

MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # ── OpenAI ──────────────────────────────────────────────────────────────
    "gpt-4o":                    {"prompt": 0.0025,  "completion": 0.010,   "cached": 0.00125},
    "gpt-4o-mini":               {"prompt": 0.00015, "completion": 0.0006,  "cached": 0.000075},
    "gpt-4-turbo":               {"prompt": 0.010,   "completion": 0.030,   "cached": 0.005},
    "gpt-4":                     {"prompt": 0.030,   "completion": 0.060,   "cached": 0.0},
    "gpt-3.5-turbo":             {"prompt": 0.0005,  "completion": 0.0015,  "cached": 0.0},
    "o1":                        {"prompt": 0.015,   "completion": 0.060,   "cached": 0.0075},
    "o1-mini":                   {"prompt": 0.003,   "completion": 0.012,   "cached": 0.0015},
    "o1-preview":                {"prompt": 0.015,   "completion": 0.060,   "cached": 0.0075},
    "o3":                        {"prompt": 0.010,   "completion": 0.040,   "cached": 0.0025},
    "o3-mini":                   {"prompt": 0.0011,  "completion": 0.0044,  "cached": 0.00055},
    # ── Anthropic ───────────────────────────────────────────────────────────
    "claude-opus-4-6":           {"prompt": 0.015,   "completion": 0.075,   "cached": 0.0015},
    "claude-sonnet-4-6":         {"prompt": 0.003,   "completion": 0.015,   "cached": 0.0003},
    "claude-haiku-4-5-20251001": {"prompt": 0.0008,  "completion": 0.004,   "cached": 0.00008},
    "claude-3-5-sonnet-20241022":{"prompt": 0.003,   "completion": 0.015,   "cached": 0.0003},
    "claude-3-5-haiku-20241022": {"prompt": 0.0008,  "completion": 0.004,   "cached": 0.00008},
    "claude-3-opus-20240229":    {"prompt": 0.015,   "completion": 0.075,   "cached": 0.0015},
    # ── Google Gemini ────────────────────────────────────────────────────────
    "gemini-2.5-flash":          {"prompt": 0.000075,"completion": 0.0003,  "cached": 0.0},
    "gemini-2.5-pro":            {"prompt": 0.00125, "completion": 0.010,   "cached": 0.0003125},
    "models/gemini-2.5-flash":   {"prompt": 0.000075,"completion": 0.0003,  "cached": 0.0},
    "models/gemini-2.5-pro":     {"prompt": 0.00125, "completion": 0.010,   "cached": 0.0003125},
    "gemini-2.0-flash":          {"prompt": 0.000075,"completion": 0.0003,  "cached": 0.0},
    "gemini-2.0-flash-lite":     {"prompt": 0.000038,"completion": 0.00015, "cached": 0.0},
    "gemini-1.5-pro":            {"prompt": 0.00125, "completion": 0.005,   "cached": 0.0003125},
    "gemini-1.5-flash":          {"prompt": 0.000075,"completion": 0.0003,  "cached": 0.0000188},
    "gemini-1.0-pro":            {"prompt": 0.0005,  "completion": 0.0015,  "cached": 0.0},
    # ── Groq (hosted open models) ────────────────────────────────────────────
    "llama-3.3-70b-versatile":   {"prompt": 0.00059, "completion": 0.00079, "cached": 0.0},
    "llama-3.1-8b-instant":      {"prompt": 0.00005, "completion": 0.00008, "cached": 0.0},
    "mixtral-8x7b-32768":        {"prompt": 0.00024, "completion": 0.00024, "cached": 0.0},
    "gemma2-9b-it":              {"prompt": 0.0002,  "completion": 0.0002,  "cached": 0.0},
    # ── Mistral ─────────────────────────────────────────────────────────────
    "mistral-large-latest":      {"prompt": 0.002,   "completion": 0.006,   "cached": 0.0},
    "mistral-small-latest":      {"prompt": 0.0002,  "completion": 0.0006,  "cached": 0.0},
    "open-mistral-nemo":         {"prompt": 0.00015, "completion": 0.00015, "cached": 0.0},
    "codestral-latest":          {"prompt": 0.001,   "completion": 0.003,   "cached": 0.0},
    "pixtral-large-latest":      {"prompt": 0.002,   "completion": 0.006,   "cached": 0.0},
    # ── Cohere ──────────────────────────────────────────────────────────────
    "command-r-plus-08-2024":    {"prompt": 0.0025,  "completion": 0.010,   "cached": 0.0},
    "command-r-08-2024":         {"prompt": 0.00015, "completion": 0.0006,  "cached": 0.0},
    "command-r":                 {"prompt": 0.00015, "completion": 0.0006,  "cached": 0.0},
    # ── Together AI ─────────────────────────────────────────────────────────
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo":  {"prompt": 0.00088, "completion": 0.00088, "cached": 0.0},
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo":   {"prompt": 0.00018, "completion": 0.00018, "cached": 0.0},
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": {"prompt": 0.0035,  "completion": 0.0035,  "cached": 0.0},
    "Qwen/Qwen2.5-72B-Instruct-Turbo":               {"prompt": 0.00072, "completion": 0.00072, "cached": 0.0},
    "deepseek-ai/DeepSeek-R1":                        {"prompt": 0.003,   "completion": 0.007,   "cached": 0.0},
    # ── DeepSeek (direct) ────────────────────────────────────────────────────
    "deepseek-chat":             {"prompt": 0.00014, "completion": 0.00028, "cached": 0.000014},
    "deepseek-reasoner":         {"prompt": 0.00055, "completion": 0.00219, "cached": 0.000055},
    # ── xAI Grok ─────────────────────────────────────────────────────────────
    "grok-2-latest":             {"prompt": 0.002,   "completion": 0.010,   "cached": 0.0},
    "grok-2-vision-1212":        {"prompt": 0.002,   "completion": 0.010,   "cached": 0.0},
    # ── Perplexity ───────────────────────────────────────────────────────────
    "llama-3.1-sonar-large-128k-online": {"prompt": 0.001, "completion": 0.001, "cached": 0.0},
    "llama-3.1-sonar-small-128k-online": {"prompt": 0.0002,"completion": 0.0002,"cached": 0.0},
    # ── Ollama (local — zero cost) ───────────────────────────────────────────
    "ollama/llama3.2":           {"prompt": 0.0, "completion": 0.0, "cached": 0.0},
    "ollama/mistral":            {"prompt": 0.0, "completion": 0.0, "cached": 0.0},
    "ollama/qwen2.5":            {"prompt": 0.0, "completion": 0.0, "cached": 0.0},
}

FALLBACK_CHAINS: Dict[str, List[str]] = {
    # OpenAI
    "gpt-4o":               ["gpt-4o-mini", "claude-sonnet-4-6", "gemini-1.5-pro"],
    "gpt-4o-mini":          ["gemini-2.0-flash", "claude-haiku-4-5-20251001"],
    "o1":                   ["gpt-4o", "claude-opus-4-6"],
    "o3-mini":              ["gpt-4o-mini", "gemini-2.0-flash"],
    # Anthropic
    "claude-opus-4-6":      ["claude-sonnet-4-6", "gpt-4o"],
    "claude-sonnet-4-6":    ["claude-haiku-4-5-20251001", "gpt-4o-mini"],
    "claude-haiku-4-5-20251001": ["gemini-2.0-flash", "gpt-4o-mini"],
    # Gemini
    "gemini-2.5-flash":     ["gemini-2.0-flash", "gpt-4o-mini"],
    "gemini-2.5-pro":       ["gemini-2.5-flash", "gpt-4o", "claude-sonnet-4-6"],
    "models/gemini-2.5-flash": ["gemini-2.0-flash", "gpt-4o-mini"],
    "gemini-1.5-pro":       ["gpt-4o", "claude-sonnet-4-6"],
    "gemini-2.0-flash":     ["gemini-2.5-flash", "gpt-4o-mini"],
    # Groq
    "llama-3.3-70b-versatile": ["gpt-4o-mini", "gemini-2.0-flash"],
    # Mistral
    "mistral-large-latest": ["gpt-4o", "claude-sonnet-4-6"],
}

COMPLEXITY_TO_MODEL: Dict[ComplexityTier, str] = {
    ComplexityTier.LOW:    "gpt-4o-mini",
    ComplexityTier.MEDIUM: "gpt-4o",
    ComplexityTier.HIGH:   "claude-sonnet-4-6",
    ComplexityTier.MAX:    "claude-opus-4-6",
}


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _detect_provider(model: str) -> str:
    """Return a provider key from a model name."""
    m = model.lower()
    # Google's full model path format: "models/gemini-*"
    if m.startswith("models/gemini"):
        return "gemini"
    # Explicit prefixes
    if m.startswith("ollama/") or m.startswith("local/"):
        return "ollama"
    if m.startswith("azure/"):
        return "azure"
    if m.startswith("openrouter/"):
        return "openrouter"
    if m.startswith("groq/"):
        return "groq"
    if m.startswith("together/") or "/" in model:
        # together.ai uses "org/model" format
        return "together"
    # Model name patterns
    if m.startswith("gpt-") or m.startswith("o1") or m.startswith("o3"):
        return "openai"
    if m.startswith("claude-"):
        return "anthropic"
    if m.startswith("gemini-"):
        return "gemini"
    if m.startswith("llama-") or m.startswith("mixtral-") or m.startswith("gemma"):
        return "groq"
    if m.startswith("mistral-") or m.startswith("open-") or m.startswith("codestral") or m.startswith("pixtral"):
        return "mistral"
    if m.startswith("command-"):
        return "cohere"
    if m.startswith("deepseek-"):
        return "deepseek"
    if m.startswith("grok-"):
        return "xai"
    if m.startswith("llama-3.1-sonar"):
        return "perplexity"
    # Check env for custom endpoint
    if os.environ.get("OPENAI_COMPAT_BASE_URL"):
        return "compat"
    return "openai"  # default


class ModelRouter:
    """
    Routes LLM calls to the correct provider, handling fallbacks.

    Usage::

        router = ModelRouter(primary_model="gpt-4o")
        response = await router.complete(messages, model="gemini-2.0-flash")

        # Or let auto-routing pick the model based on complexity:
        router = ModelRouter(auto_route=True)
        response = await router.complete(messages)
    """

    def __init__(
        self,
        primary_model: str = "gpt-4o",
        fallback_chain: Optional[List[str]] = None,
        auto_route: bool = True,
    ) -> None:
        self._primary = primary_model
        self._fallback_chain = fallback_chain or FALLBACK_CHAINS.get(primary_model, [])
        self._auto_route = auto_route
        self._providers: Dict[str, Any] = {}

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict[str, Any]] = None,
        complexity_hint: Optional[str] = None,
    ) -> ModelResponse:
        target = self._select_model(model, messages, tools, complexity_hint)
        # Build fallback chain from the SELECTED model (not primary).
        # If a custom fallback_chain was provided, use it; otherwise use the
        # selected model's chain from FALLBACK_CHAINS.
        if self._fallback_chain and self._primary == target:
            tail = [m for m in self._fallback_chain if m != target]
        else:
            tail = [m for m in FALLBACK_CHAINS.get(target, []) if m != target]
        chain = [target] + tail
        last_exc: Optional[Exception] = None
        errors: Dict[str, str] = {}  # model -> error reason

        for attempt_model in chain:
            try:
                provider = self._get_provider(attempt_model)
                return await provider.complete(
                    messages=messages, model=attempt_model, tools=tools,
                    temperature=temperature, max_tokens=max_tokens,
                    response_format=response_format,
                )
            except HelixProviderError as e:
                last_exc = e
                errors[attempt_model] = str(e.original or e)
                if not e.retryable:
                    break
            except Exception as e:
                last_exc = e
                errors[attempt_model] = f"{type(e).__name__}: {e}"

        raise AllModelsExhaustedError(attempted=chain, errors=errors) from last_exc

    async def stream(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        target = self._select_model(model, messages, tools)
        provider = self._get_provider(target)
        async for chunk in provider.stream(
            messages=messages, model=target,
            tools=tools, temperature=temperature, max_tokens=max_tokens,
        ):
            yield chunk

    def calculate_cost(self, usage: TokenUsage, model: str) -> float:
        pricing = MODEL_PRICING.get(model, {"prompt": 0.002, "completion": 0.008, "cached": 0.0})
        cost = (
            max(0, usage.prompt_tokens - usage.cached_tokens) / 1000 * pricing["prompt"]
            + usage.cached_tokens / 1000 * pricing.get("cached", 0.0)
            + usage.completion_tokens / 1000 * pricing["completion"]
        )
        return round(cost, 8)

    def list_models(self, provider: Optional[str] = None) -> List[str]:
        models = list(MODEL_PRICING.keys())
        if provider:
            return [m for m in models if _detect_provider(m) == provider]
        return models

    def _select_model(self, override, messages, tools, complexity_hint=None) -> str:
        if override:
            return override
        if self._auto_route:
            from helix.models.complexity import ComplexityEstimator
            tier = ComplexityEstimator.estimate(messages=messages, tools=tools or [], hint=complexity_hint)
            return COMPLEXITY_TO_MODEL.get(tier, self._primary)
        return self._primary

    def _get_provider(self, model: str) -> Any:
        provider_key = _detect_provider(model)
        cache_key = f"{provider_key}:{model}"
        if cache_key in self._providers:
            return self._providers[cache_key]

        provider = self._build_provider(provider_key, model)
        self._providers[cache_key] = provider
        return provider

    def _build_provider(self, provider_key: str, model: str) -> Any:
        if provider_key == "openai":
            from helix.models.providers.openai import OpenAIProvider
            return OpenAIProvider()

        if provider_key == "anthropic":
            from helix.models.providers.anthropic import AnthropicProvider
            return AnthropicProvider()

        if provider_key == "gemini":
            from helix.models.providers.gemini import GeminiProvider
            return GeminiProvider()

        if provider_key == "groq":
            from helix.models.providers.groq import GroqProvider
            return GroqProvider()

        if provider_key == "mistral":
            from helix.models.providers.mistral import MistralProvider
            return MistralProvider()

        if provider_key == "cohere":
            from helix.models.providers.cohere import CohereProvider
            return CohereProvider()

        if provider_key == "together":
            from helix.models.providers.together import TogetherProvider
            return TogetherProvider()

        if provider_key == "ollama":
            from helix.models.providers.ollama import OllamaProvider
            return OllamaProvider()

        if provider_key == "azure":
            from helix.models.providers.azure import AzureOpenAIProvider
            return AzureOpenAIProvider()

        if provider_key == "openrouter":
            from helix.models.providers.openai_compat import OpenAICompatProvider
            return OpenAICompatProvider.from_preset("openrouter")

        if provider_key == "deepseek":
            from helix.models.providers.openai_compat import OpenAICompatProvider
            return OpenAICompatProvider.from_preset("deepseek")

        if provider_key == "xai":
            from helix.models.providers.openai_compat import OpenAICompatProvider
            return OpenAICompatProvider.from_preset("xai")

        if provider_key == "perplexity":
            from helix.models.providers.openai_compat import OpenAICompatProvider
            return OpenAICompatProvider.from_preset("perplexity")

        if provider_key == "compat":
            from helix.models.providers.openai_compat import OpenAICompatProvider
            base_url = os.environ.get("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
            return OpenAICompatProvider(base_url=base_url)

        # Default: try OpenAI
        from helix.models.providers.openai import OpenAIProvider
        return OpenAIProvider()
