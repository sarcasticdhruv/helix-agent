"""
examples/basic_async.py  —  async usage pattern.
Run: python examples/basic_async.py
"""

import asyncio

import helix


async def main():
    agent = helix.Agent(
        name="Assistant",
        role="General assistant",
        goal="Answer questions clearly.",
    )

    result = await agent.run("Explain what a neural network is in 2 sentences.")

    if result.error:
        print(f"Error: {result.error}")
        print()
        print("Troubleshooting:")
        print("  1. helix doctor")
        print("  2. python examples/debug_provider.py")
    else:
        print(f"Output:  {result.output}")
        print(f"Model:   {result.model_used}")
        print(f"Cost:    ${result.cost_usd:.6f}")


asyncio.run(main())
