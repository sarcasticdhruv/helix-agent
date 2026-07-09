"""
helix/tools/mcp.py

MCP (Model Context Protocol) client integration — connects to any MCP
server over stdio and exposes its tools as Helix RegisteredTools, so an
Agent can use them exactly like a built-in tool.

Install:  pip install mcp

Usage::

    from helix.tools.mcp import MCPToolSource

    async with MCPToolSource(command="npx", args=["-y", "@some/mcp-server"]) as tools:
        agent = helix.Agent(name="Bot", role="...", goal="...", tools=tools)
        result = await agent.run("...")

    # Or without the context manager, when the source must outlive one call:
    source = MCPToolSource(command="python", args=["-m", "my_mcp_server"])
    tools = await source.connect()
    ...
    await source.close()

The MCP connection is a long-lived subprocess/pipe pair (the stdio
transport), not a one-shot request — connect() once and reuse the
returned tools for as long as the source stays open.
"""

from __future__ import annotations

import contextlib
from typing import Any

from helix.core.tool import RegisteredTool
from helix.errors import ToolError


class MCPToolSource:
    """Wraps one MCP server connection and exposes its tools as RegisteredTools."""

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        tool_timeout_s: float = 30.0,
    ) -> None:
        self._command = command
        self._args = args or []
        self._env = env
        self._tool_timeout_s = tool_timeout_s
        self._session: Any | None = None
        self._exit_stack: contextlib.AsyncExitStack | None = None

    async def connect(self) -> list[RegisteredTool]:
        """Start the MCP server and return its tools as Helix RegisteredTools."""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as err:
            raise ImportError("mcp package required. Install with: pip install mcp") from err

        self._exit_stack = contextlib.AsyncExitStack()
        try:
            server_params = StdioServerParameters(
                command=self._command, args=self._args, env=self._env
            )
            read, write = await self._exit_stack.enter_async_context(stdio_client(server_params))
            self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))
            await self._session.initialize()

            listed = await self._session.list_tools()
            return [self._wrap_tool(t) for t in listed.tools]
        except Exception:
            await self.close()
            raise

    def _wrap_tool(self, mcp_tool: Any) -> RegisteredTool:
        tool_name = mcp_tool.name

        async def _call(**kwargs: Any) -> Any:
            if self._session is None:
                raise ToolError(
                    tool_name=tool_name,
                    message="MCP session is closed — call connect() again before use.",
                )
            result = await self._session.call_tool(tool_name, arguments=kwargs)
            text_parts = [
                block.text for block in result.content if getattr(block, "text", None) is not None
            ]
            output = "\n".join(text_parts) if text_parts else str(result.content)
            if result.isError:
                raise ToolError(tool_name=tool_name, message=output)
            return output

        return RegisteredTool(
            fn=_call,
            name=tool_name,
            description=mcp_tool.description or f"MCP tool '{tool_name}'",
            timeout_s=self._tool_timeout_s,
            retries=0,
            on_error="raise",
            fallback_fn=None,
            parameters_schema=mcp_tool.inputSchema or {"type": "object", "properties": {}},
        )

    async def close(self) -> None:
        if self._exit_stack is not None:
            with contextlib.suppress(Exception):
                await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None

    async def __aenter__(self) -> list[RegisteredTool]:
        return await self.connect()

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()
