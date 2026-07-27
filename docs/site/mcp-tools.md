# MCP Tools

Connect to any [Model Context Protocol](https://modelcontextprotocol.io) server and use its tools like any other Helix tool. Requires `pip install "helix-framework[mcp]"`.

```python
import helix
from helix.tools.mcp import MCPToolSource

async def main():
    async with MCPToolSource(command="npx", args=["-y", "@some/mcp-server"]) as tools:
        agent = helix.Agent(
            name="Bot",
            role="Assistant",
            goal="Use the connected MCP tools to help the user.",
            tools=tools,
        )
        result = await agent.run("...")
        print(result.output)
```

Without the context manager (when the connection needs to outlive one call):

```python
source = MCPToolSource(command="python", args=["-m", "my_mcp_server"])
tools = await source.connect()
# ... use tools across multiple agent runs ...
await source.close()
```
