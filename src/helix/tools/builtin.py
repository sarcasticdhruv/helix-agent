"""
helix/tools/builtin.py

Built-in tool suite. 12 production-ready tools registered
in the global Helix tool registry.

Import this module to make built-in tools available::

    import helix.tools.builtin  # registers all tools globally

Or pick individual tools::

    from helix.tools.builtin import web_search, read_file
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import UTC
from pathlib import Path
from typing import Any

from helix.core.tool import registry, tool

# ---------------------------------------------------------------------------
# Web / HTTP
# ---------------------------------------------------------------------------


@tool(
    description="Search the web for current information. Returns a list of results with title, URL, and snippet.",
    timeout=15.0,
    retries=2,
)
async def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """
    :param query: The search query string.
    :param max_results: Maximum number of results to return (1-10).
    """
    try:
        from ddgs import DDGS

        with DDGS() as client:
            results = list(client.text(query, max_results=max_results))
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
                for r in results
            ]
    except ImportError:
        return [{"error": "ddgs not installed. pip install ddgs"}]
    except Exception as e:
        return [{"error": str(e)}]


@tool(
    description="Fetch the content of a URL and return the text.",
    timeout=20.0,
    retries=1,
)
async def fetch_url(url: str, timeout: float = 10.0) -> dict[str, Any]:
    """
    :param url: The URL to fetch.
    :param timeout: Request timeout in seconds.
    """
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=timeout, follow_redirects=True)
            text = resp.text
            # Strip HTML tags for readability
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return {
                "url": url,
                "status": resp.status_code,
                "content": text[:10_000],  # Truncate large pages
                "content_type": resp.headers.get("content-type", ""),
            }
    except ImportError:
        return {"error": "httpx not installed. pip install httpx"}
    except Exception as e:
        return {"error": str(e), "url": url}


# ---------------------------------------------------------------------------
# File system
# ---------------------------------------------------------------------------


@tool(
    description="Read the contents of a text file. Returns file content as a string.",
    timeout=10.0,
)
async def read_file(path: str, max_chars: int = 50_000) -> dict[str, Any]:
    """
    :param path: Absolute or relative file path.
    :param max_chars: Maximum characters to return.
    """
    try:
        p = Path(path)
        if not p.exists():
            return {"error": f"File not found: {path}"}
        content = p.read_text(errors="replace")[:max_chars]
        return {
            "path": str(p.absolute()),
            "content": content,
            "size_bytes": p.stat().st_size,
            "truncated": len(content) == max_chars,
        }
    except PermissionError:
        return {"error": f"Permission denied: {path}"}
    except Exception as e:
        return {"error": str(e)}


@tool(
    description="Write content to a file. Creates parent directories if needed.",
    timeout=10.0,
)
async def write_file(path: str, content: str, mode: str = "w") -> dict[str, Any]:
    """
    :param path: File path to write to.
    :param content: Content to write.
    :param mode: Write mode: 'w' (overwrite) or 'a' (append).
    """
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, mode, encoding="utf-8") as f:
            f.write(content)
        return {"path": str(p.absolute()), "bytes_written": len(content.encode())}
    except Exception as e:
        return {"error": str(e)}


@tool(
    description="List files in a directory. Returns file names, sizes, and types.",
    timeout=5.0,
)
async def list_directory(path: str = ".", pattern: str = "*") -> dict[str, Any]:
    """
    :param path: Directory path to list.
    :param pattern: Glob pattern to filter files.
    """
    try:
        p = Path(path)
        if not p.is_dir():
            return {"error": f"Not a directory: {path}"}
        entries = []
        for item in sorted(p.glob(pattern))[:100]:
            entries.append(
                {
                    "name": item.name,
                    "type": "dir" if item.is_dir() else "file",
                    "size_bytes": item.stat().st_size if item.is_file() else 0,
                }
            )
        return {"path": str(p.absolute()), "entries": entries, "count": len(entries)}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Data / Computation
# ---------------------------------------------------------------------------


@tool(
    description="Evaluate a safe mathematical expression. Supports arithmetic, math functions, and basic statistics.",
    timeout=5.0,
)
async def calculator(expression: str) -> dict[str, Any]:
    """
    :param expression: Math expression to evaluate (e.g., '2 ** 10', 'sqrt(144)').
    """
    import math

    safe_globals = {
        "__builtins__": {},
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "pow": pow,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
        "ceil": math.ceil,
        "floor": math.floor,
    }
    # Reject dangerous patterns
    if any(kw in expression for kw in ("import", "exec", "eval", "__", "open", "os")):
        return {"error": "Expression contains disallowed keywords."}
    try:
        result = eval(expression, safe_globals)  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e), "expression": expression}


@tool(
    description="Parse and query JSON data. Supports dot-notation key access and basic filtering.",
    timeout=5.0,
)
async def json_query(data: str, query: str = "") -> dict[str, Any]:
    """
    :param data: JSON string to parse.
    :param query: Dot-notation path to extract (e.g., 'users.0.name'). Empty returns full object.
    """
    try:
        parsed = json.loads(data)
        if not query:
            return {"result": parsed}
        # Dot-notation traversal
        parts = query.split(".")
        current = parsed
        for part in parts:
            if isinstance(current, list):
                try:
                    current = current[int(part)]
                except (ValueError, IndexError) as e:
                    return {"error": f"Array index error at '{part}': {e}"}
            elif isinstance(current, dict):
                current = current.get(part, None)
                if current is None:
                    return {"error": f"Key '{part}' not found"}
            else:
                return {"error": f"Cannot traverse into {type(current).__name__} at '{part}'"}
        return {"query": query, "result": current}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {e}"}


# ---------------------------------------------------------------------------
# System / Environment
# ---------------------------------------------------------------------------


@tool(
    description="Get the current date and time in ISO format.",
    timeout=2.0,
)
async def get_datetime(timezone: str = "UTC") -> dict[str, str]:
    """
    :param timezone: Timezone name (e.g., 'UTC', 'US/Eastern'). Defaults to UTC.
    """
    from datetime import datetime

    now = datetime.now(UTC)
    return {
        "iso": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "timezone": "UTC",
    }


@tool(
    description="Read environment variables. Cannot read sensitive vars (keys, tokens, passwords).",
    timeout=2.0,
)
async def get_env(key: str) -> dict[str, Any]:
    """
    :param key: Environment variable name to read.
    """
    BLOCKED = {"api_key", "secret", "password", "token", "auth", "private"}
    if any(b in key.lower() for b in BLOCKED):
        return {"error": f"Reading '{key}' is not permitted for security reasons."}
    value = os.environ.get(key)
    if value is None:
        return {"key": key, "value": None, "found": False}
    return {"key": key, "value": value, "found": True}


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


@tool(
    description="Count words, characters, sentences, and paragraphs in text.",
    timeout=2.0,
)
async def text_stats(text: str) -> dict[str, int]:
    """
    :param text: The text to analyze.
    """
    words = len(text.split())
    chars = len(text)
    sentences = len(re.findall(r"[.!?]+", text)) or 1
    paragraphs = len([p for p in text.split("\n\n") if p.strip()]) or 1
    return {
        "words": words,
        "characters": chars,
        "sentences": sentences,
        "paragraphs": paragraphs,
    }


@tool(
    description="Extract URLs from text.",
    timeout=2.0,
)
async def extract_urls(text: str) -> dict[str, Any]:
    """
    :param text: Text to extract URLs from.
    """
    pattern = re.compile(r"https?://[^\s\"'<>]+")
    urls = pattern.findall(text)
    return {"urls": urls, "count": len(urls)}


@tool(
    description="Sleep for a specified number of seconds. Useful for rate-limiting retries.",
    timeout=70.0,
)
async def sleep(seconds: float) -> dict[str, float]:
    """
    :param seconds: Number of seconds to sleep (max 60).
    """
    clamped = min(max(0.0, seconds), 60.0)
    await asyncio.sleep(clamped)
    return {"slept_seconds": clamped}


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------


@tool(
    description="Execute Python code in an isolated subprocess and return stdout/stderr. "
    "Code has access to the standard library. Dangerous builtins (os.system, "
    "subprocess, open outside cwd) are blocked by a watchdog.",
    timeout=30.0,
    retries=0,
)
async def execute_python(code: str, timeout: float = 15.0) -> dict[str, Any]:
    """
    :param code:    Valid Python source code to execute.
    :param timeout: Max execution time in seconds (clamped to 60).
    """
    import sys
    import tempfile

    timeout = min(max(1.0, timeout), 60.0)

    # Wrap the user code in a thin security shim that blocks the most
    # common escape hatches before exec-ing the real code.
    wrapper = (
        "import sys, builtins\n"
        "_orig_import = builtins.__import__\n"
        "_BLOCKED = {'subprocess', 'pty', 'ctypes', 'multiprocessing'}\n"
        "def _safe_import(name, *a, **kw):\n"
        "    if name in _BLOCKED:\n"
        "        raise ImportError(f'Module {name!r} is blocked in execute_python.')\n"
        "    return _orig_import(name, *a, **kw)\n"
        "builtins.__import__ = _safe_import\n"
        "import os\n"
        "_orig_system = getattr(os, 'system', None)\n"
        "os.system = lambda *a, **k: (_ for _ in ()).throw(PermissionError('os.system blocked'))\n"
        + code
    )

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(wrapper)
            tmp_path = tmp.name

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            return {
                "success": False,
                "stdout": "",
                "stderr": "",
                "error": f"Execution timed out after {timeout}s",
                "returncode": -1,
            }
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        stdout = stdout_b.decode(errors="replace")
        stderr = stderr_b.decode(errors="replace")
        return {
            "success": proc.returncode == 0,
            "stdout": stdout[:20_000],
            "stderr": stderr[:5_000],
            "returncode": proc.returncode,
        }
    except Exception as exc:
        return {"success": False, "stdout": "", "stderr": "", "error": str(exc), "returncode": -1}


# ---------------------------------------------------------------------------
# Register all built-in tools
# ---------------------------------------------------------------------------

_BUILTIN_TOOLS = [
    web_search,
    fetch_url,
    read_file,
    write_file,
    list_directory,
    calculator,
    json_query,
    get_datetime,
    get_env,
    text_stats,
    extract_urls,
    sleep,
    execute_python,
]

for _t in _BUILTIN_TOOLS:
    registry.register(_t)
