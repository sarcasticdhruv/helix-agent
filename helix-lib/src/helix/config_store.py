"""
helix/config_store.py

Persistent API key and settings storage for Helix.

Keys are stored in ~/.helix/config.json (user home directory).
They are loaded automatically when helix is imported.

Priority order (highest wins):
  1. Explicitly passed in code:  Agent(..., api_keys={"GOOGLE_API_KEY": "..."})
  2. Environment variable:       $env:GOOGLE_API_KEY = "..."
  3. .env file in working dir:   GOOGLE_API_KEY=... in .env
  4. Helix config file:          helix config set GOOGLE_API_KEY ...
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


# Where helix stores its config
_CONFIG_DIR = Path.home() / ".helix"
_CONFIG_FILE = _CONFIG_DIR / "config.json"

# Keys that should be masked in output
_SENSITIVE_KEYS = {
    "GOOGLE_API_KEY", "GEMINI_API_KEY",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "GROQ_API_KEY", "MISTRAL_API_KEY", "COHERE_API_KEY",
    "TOGETHER_API_KEY", "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY", "XAI_API_KEY",
    "PERPLEXITY_API_KEY", "FIREWORKS_API_KEY",
    "AZURE_OPENAI_API_KEY",
}


def _load_config() -> Dict[str, str]:
    """Load saved config from ~/.helix/config.json."""
    if not _CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_config(data: Dict[str, str]) -> None:
    """Save config to ~/.helix/config.json."""
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _CONFIG_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    # Restrict permissions on non-Windows (contains secrets)
    if os.name != "nt":
        _CONFIG_FILE.chmod(0o600)


def _load_dotenv(path: Optional[Path] = None) -> Dict[str, str]:
    """
    Parse a .env file. Returns dict of key=value pairs.
    Does not require python-dotenv — we parse it ourselves.
    """
    env_file = path or (Path.cwd() / ".env")
    if not env_file.exists():
        return {}
    result: Dict[str, str] = {}
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                result[key] = value
    except Exception:
        pass
    return result


def apply_saved_config() -> None:
    """
    Load keys from config file and .env into os.environ.
    Called automatically when helix is imported.
    Priority: existing env vars > .env file > saved config.
    """
    # Load saved config (lowest priority of the three)
    saved = _load_config()
    for k, v in saved.items():
        if not os.environ.get(k):
            os.environ[k] = v

    # Load .env file (medium priority)
    dotenv = _load_dotenv()
    for k, v in dotenv.items():
        if not os.environ.get(k):
            os.environ[k] = v


def set_key(name: str, value: str) -> None:
    """
    Save an API key to ~/.helix/config.json and set it in the current process.
    Called by: helix config set KEY value
    """
    data = _load_config()
    data[name] = value
    _save_config(data)
    os.environ[name] = value


def get_key(name: str) -> Optional[str]:
    """Get a key — checks env first, then saved config."""
    return os.environ.get(name) or _load_config().get(name)


def delete_key(name: str) -> bool:
    """Remove a key from saved config. Returns True if it existed."""
    data = _load_config()
    if name in data:
        del data[name]
        _save_config(data)
        os.environ.pop(name, None)
        return True
    return False


def list_keys() -> Dict[str, str]:
    """
    List all saved keys with values masked.
    Also shows keys set via environment variables.
    """
    result: Dict[str, str] = {}
    saved = _load_config()

    # Show all known provider keys
    for key in sorted(_SENSITIVE_KEYS):
        value = os.environ.get(key) or saved.get(key)
        if value:
            masked = value[:6] + "..." + value[-3:] if len(value) > 12 else "***"
            source = "env" if os.environ.get(key) else "saved"
            result[key] = f"{masked}  ({source})"

    # Show any extra saved keys not in the known list
    for key, value in saved.items():
        if key not in _SENSITIVE_KEYS and key not in result:
            masked = value[:4] + "..." if len(value) > 8 else "***"
            result[key] = f"{masked}  (saved)"

    return result


def config_path() -> Path:
    return _CONFIG_FILE


# ---------------------------------------------------------------------------
# Smart default model selection
# ---------------------------------------------------------------------------

# Priority-ordered list: (env_var_or_vars, model_name)
_PROVIDER_MODEL_PRIORITY = [
    (["OPENAI_API_KEY"],                   "gpt-4o-mini"),
    (["ANTHROPIC_API_KEY"],                "claude-haiku-4-5-20251001"),
    (["GOOGLE_API_KEY", "GEMINI_API_KEY"], "models/gemini-2.5-flash"),
    (["GROQ_API_KEY"],                     "llama-3.3-70b-versatile"),
    (["MISTRAL_API_KEY"],                  "mistral-small-latest"),
    (["TOGETHER_API_KEY"],                 "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
    (["COHERE_API_KEY"],                   "command-r-08-2024"),
    (["OPENROUTER_API_KEY"],               "openrouter/openai/gpt-4o-mini"),
    (["DEEPSEEK_API_KEY"],                 "deepseek-chat"),
    (["XAI_API_KEY"],                      "grok-2-latest"),
    (["PERPLEXITY_API_KEY"],               "llama-3.1-sonar-small-128k-online"),
    (["FIREWORKS_API_KEY"],                "accounts/fireworks/models/llama-v3p1-8b-instruct"),
]


def best_available_model() -> str:
    """
    Return the model to use, in this priority order:
      1. HELIX_DEFAULT_MODEL env var / saved config key
      2. `helix config set HELIX_DEFAULT_MODEL gemini-2.0-flash`
      3. First provider in priority list that has a key configured
      4. 'gpt-4o' fallback (will fail with a clear error if no key is set)

    Configure once:
        helix config set HELIX_DEFAULT_MODEL gemini-2.0-flash
    Or per-session:
        $env:HELIX_DEFAULT_MODEL = "gemini-2.0-flash"
    Or in .env:
        HELIX_DEFAULT_MODEL=gemini-2.0-flash
    """
    # Explicit override wins (env > saved config)
    explicit = os.environ.get("HELIX_DEFAULT_MODEL", "").strip()
    if not explicit:
        explicit = _load_config().get("HELIX_DEFAULT_MODEL", "").strip()
    if explicit:
        return explicit

    # Auto-detect from available keys
    for env_vars, model in _PROVIDER_MODEL_PRIORITY:
        if any(os.environ.get(v, "").strip() for v in env_vars):
            return model

    return "gpt-4o"  # will fail clearly if no key is set


def available_providers() -> list:
    """Return list of provider names that have a key configured."""
    result = []
    for env_vars, model in _PROVIDER_MODEL_PRIORITY:
        if any(os.environ.get(v, "").strip() for v in env_vars):
            result.append(model.split("/")[0] if "/" not in model[:8] else model)
    return result
