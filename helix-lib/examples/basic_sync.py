"""
examples/basic_sync.py  —  simplest possible Helix usage.
Run: python examples/basic_sync.py
"""

import helix

agent = helix.Agent(
    name="Assistant",
    role="General assistant",
    goal="Answer questions clearly and concisely.",
)

result = helix.run(agent, "What is the capital of Japan? Answer in one sentence.")

if result.error:
    print(f"Error: {result.error}")
    print()
    print("Troubleshooting:")
    print("  1. helix doctor          — check which providers are ready")
    print("  2. helix config list     — see your saved API keys")
    print("  3. helix config set GOOGLE_API_KEY your-key-here")
    print("  4. python examples/debug_provider.py   — test Gemini directly")
else:
    print(f"Output:    {result.output}")
    print(f"Model:     {result.model_used}")
    print(f"Cost:      ${result.cost_usd:.6f}")
    print(f"Steps:     {result.steps}")
