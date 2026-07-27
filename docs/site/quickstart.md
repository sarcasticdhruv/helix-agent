# Quickstart

Install Helix from PyPI:

```bash
pip install helix-framework
```

```python
import helix

agent = helix.Agent(
    name="Researcher",
    role="Research analyst",
    goal="Find accurate, cited answers.",
)

result = helix.run(agent, "What is quantum entanglement?")
print(result.output)
print(f"Cost:  ${result.cost_usd:.4f}")
print(f"Steps: {result.steps}")
```

For the fastest possible start, use `helix.quick()` — no config objects needed:

```python
import helix

agent = helix.quick("You are a concise Python tutor.", budget_usd=0.10)
result = helix.run(agent, "Explain list comprehensions.")
print(result.output)
```

Inside an async function, call `run_async` or `agent.run` directly:

```python
import asyncio
import helix

async def main():
    agent = helix.Agent(
        name="Researcher",
        role="Research analyst",
        goal="Find accurate answers.",
    )
    result = await agent.run("What is quantum entanglement?")
    print(result.output)

asyncio.run(main())
```

`helix.quick()` parameters:

| Parameter | Description |
|---|---|
| `system_prompt` | The agent's purpose as plain instructions |
| `name` | Agent name shown in traces (default `"Agent"`) |
| `model` | Model string, e.g. `"gpt-4o"`. Auto-detected if omitted |
| `tools` | List of `@helix.tool`-decorated functions |
| `budget_usd` | Hard spend cap per run (default `0.10`) |
| `on_event` | Optional async/sync event callback (see [Event Hooks](https://github.com/sarcasticdhruv/helix-agent#event-hooks)) |
