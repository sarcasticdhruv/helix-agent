"""
examples/with_tools.py  —  agent with custom tools and budget.

Run: python examples/with_tools.py
"""

import helix


@helix.tool(description="Evaluate a safe mathematical expression.")
async def calculator(expression: str) -> dict:
    """
    :param expression: e.g. '2 ** 10' or 'sqrt(144)'
    """
    import math
    safe = {
        "__builtins__": {},
        "sqrt": math.sqrt, "log": math.log, "sin": math.sin,
        "cos": math.cos, "pi": math.pi, "e": math.e,
        "abs": abs, "round": round, "pow": pow,
    }
    try:
        result = eval(expression, safe)  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as exc:
        return {"error": str(exc)}


@helix.tool(description="Get the current UTC date and time.")
async def get_time() -> dict:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    return {"utc": now.isoformat(), "date": now.strftime("%Y-%m-%d")}


agent = helix.Agent(
    name="MathBot",
    role="Mathematical assistant",
    goal="Solve problems using the calculator tool.",
    tools=[calculator, get_time],
    budget=helix.BudgetConfig(budget_usd=0.10),
)

result = helix.run(agent, "What is 2^10 + sqrt(144)? Show your work.")

if result.error:
    print(f"Error: {result.error}")
    print("Run 'helix doctor' to check your setup.")
else:
    print(f"Answer:      {result.output}")
    print(f"Tool calls:  {result.tool_calls}")
    print(f"Cost:        ${result.cost_usd:.6f}")
