"""
helix/runtime/health.py  —  powers `helix doctor`
"""

from __future__ import annotations

import os
from typing import Any

# ── Provider registry ─────────────────────────────────────────────────────────
# (name, import_name, pip_install_name, env_vars, description)
# import_name uses dots for subpackages (e.g. "google.generativeai")
# env_vars: empty list = no key needed (Ollama)
# env_vars: multiple entries = ANY one of them satisfies the check

_PROVIDERS: list[tuple[str, str, str, list[str], str]] = [
    ("openai", "openai", "openai", ["OPENAI_API_KEY"], "GPT-4o, GPT-4o-mini, o1, o3-mini"),
    ("anthropic", "anthropic", "anthropic", ["ANTHROPIC_API_KEY"], "Claude Opus / Sonnet / Haiku"),
    (
        "gemini",
        "google.genai",
        "google-genai",
        ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        "Gemini 2.5 Flash, 2.0 Flash, 2.5 Pro",
    ),
    ("groq", "groq", "groq", ["GROQ_API_KEY"], "Llama, Mixtral, Gemma (ultra-fast)"),
    ("mistral", "mistralai", "mistralai", ["MISTRAL_API_KEY"], "Mistral Large / Small / Codestral"),
    ("cohere", "cohere", "cohere", ["COHERE_API_KEY"], "Command R+, Command R"),
    ("together", "together", "together", ["TOGETHER_API_KEY"], "200+ open models via Together AI"),
    ("ollama", "httpx", "httpx", [], "Local models — no key needed"),
    (
        "azure",
        "openai",
        "openai",
        ["AZURE_OPENAI_API_KEY"],
        "Azure OpenAI (needs AZURE_OPENAI_ENDPOINT too)",
    ),
    ("openrouter", "openai", "openai", ["OPENROUTER_API_KEY"], "100+ models via OpenRouter"),
    ("deepseek", "openai", "openai", ["DEEPSEEK_API_KEY"], "DeepSeek Chat / Reasoner"),
    ("xai", "openai", "openai", ["XAI_API_KEY"], "xAI Grok"),
    ("perplexity", "openai", "openai", ["PERPLEXITY_API_KEY"], "Perplexity Sonar (online search)"),
    ("fireworks", "openai", "openai", ["FIREWORKS_API_KEY"], "Fireworks AI (fast open models)"),
]

_OTHER_PACKAGES = [
    ("tiktoken", "tiktoken", "accurate token counting"),
    ("httpx", "httpx", "fetch_url tool + Ollama + WebhookTransport"),
    ("ddgs", "ddgs", "web_search built-in tool"),
    ("redis", "redis", "Redis cache/session backend, QueueTransport"),
    ("pydantic", "pydantic>=2.0", "core — required"),
]


def _pkg_installed(import_name: str) -> bool:
    """
    Check whether a package is importable.
    Uses importlib.import_module directly so dotted names like
    'google.genai' work correctly as namespace packages.
    """
    try:
        import importlib

        importlib.import_module(import_name)
        return True
    except ImportError:
        return False


def _any_key_set(env_vars: list[str]) -> bool:
    return any(bool(os.environ.get(v, "").strip()) for v in env_vars)


def _key_set_names(env_vars: list[str]) -> list[str]:
    return [v for v in env_vars if os.environ.get(v, "").strip()]


def _key_missing_names(env_vars: list[str]) -> list[str]:
    return [v for v in env_vars if not os.environ.get(v, "").strip()]


def _ollama_running() -> bool:
    try:
        import socket

        s = socket.create_connection(("localhost", 11434), timeout=1.0)
        s.close()
        return True
    except Exception:
        return False


async def environment_doctor() -> dict[str, Any]:
    results: dict[str, Any] = {}
    errors: list[str] = []
    warnings: list[str] = []

    provider_results: dict[str, Any] = {}
    available_count = 0

    for name, import_name, pip_name, env_vars, description in _PROVIDERS:
        # ── Ollama: special-case (local server, no key) ──────────────────────
        if name == "ollama":
            running = _ollama_running()
            provider_results[name] = {
                "status": "ready (local)" if running else "not running",
                "description": description,
                "note": "Start with: ollama serve  |  Pull model: ollama pull llama3.2",
            }
            if running:
                available_count += 1
            continue

        # ── All other providers: check key AND package independently ─────────
        pkg_ok = _pkg_installed(import_name)
        key_ok = _any_key_set(env_vars) if env_vars else True
        set_keys = _key_set_names(env_vars)
        miss_keys = _key_missing_names(env_vars)

        if key_ok and pkg_ok:
            # Fully ready
            available_count += 1
            shown_key = set_keys[0] if set_keys else "built-in"
            provider_results[name] = {
                "status": "ready",
                "description": description,
                "via_key": shown_key,
            }

        elif key_ok and not pkg_ok:
            # Key is set but SDK not installed
            provider_results[name] = {
                "status": "key set — install SDK",
                "description": description,
                "fix": f"pip install {pip_name}",
                "set_keys": set_keys,
            }
            warnings.append(
                f"{name}: pip install {pip_name}  (key already set: {', '.join(set_keys)})"
            )

        elif not key_ok and pkg_ok:
            # SDK installed but no key
            provider_results[name] = {
                "status": "missing key",
                "description": description,
                "missing_keys": miss_keys,
            }
            warnings.append(f"{name}: set {' or '.join(miss_keys)}")

        else:
            # Neither key nor SDK
            provider_results[name] = {
                "status": "not configured",
                "description": description,
                "missing_keys": miss_keys,
                "fix": f"pip install {pip_name}",
            }
            warnings.append(f"{name}: set {' or '.join(miss_keys)}  then pip install {pip_name}")

    results["providers"] = provider_results
    results["providers_available"] = available_count

    if available_count == 0:
        errors.append(
            "No LLM providers available. "
            "Set at least one API key (e.g. GOOGLE_API_KEY, OPENAI_API_KEY) "
            "and install its SDK."
        )

    # ── Other packages ────────────────────────────────────────────────────────
    pkg_results: dict[str, Any] = {}
    for import_name, pip_name, desc in _OTHER_PACKAGES:
        try:
            import importlib

            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "installed")
            pkg_results[import_name] = {"status": "ok", "version": version}
        except ImportError:
            level = "required" if import_name == "pydantic" else "optional"
            pkg_results[import_name] = {
                "status": "missing",
                "fix": f"pip install {pip_name}",
                "used_for": desc,
            }
            if level == "required":
                errors.append(f"Required package missing: {pip_name}")
            # optional packages are silent — no warning spam

    results["packages"] = pkg_results

    # ── Local directories ─────────────────────────────────────────────────────
    dirs = [".helix/traces", ".helix/audit", ".helix/wal", ".helix/eval_results"]
    dir_results: dict[str, str] = {}
    for d in dirs:
        try:
            os.makedirs(d, exist_ok=True)
            dir_results[d] = "ok"
        except Exception as e:
            dir_results[d] = f"error: {e}"
            errors.append(f"Cannot create {d}: {e}")
    results["directories"] = dir_results

    results["errors"] = errors
    results["warnings"] = warnings
    results["overall"] = "healthy" if not errors else "unhealthy"
    return results


async def health_check(rt: Any) -> dict[str, Any]:
    return {
        "runtime": {
            "status": "healthy" if rt._running else "stopped",
            "workers_alive": sum(1 for w in rt._workers if not w.done()),
            "queue_depth": rt._queue.qsize(),
        },
        "agents": list(rt._agents.keys()),
        "workflows": list(rt._workflows.keys()),
        "stats": {
            "total_runs": rt._run_count,
            "total_cost_usd": round(rt._total_cost, 4),
        },
    }
