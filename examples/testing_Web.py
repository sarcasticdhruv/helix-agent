"""
testing_Web.py  —  quick smoke-test for web search + other built-in tools

Run: python testing_Web.py
"""

import asyncio

import helix
import helix.tools.builtin
from helix.tools.builtin import calculator, fetch_url, get_datetime, web_search

# ── 1. Direct tool smoke-tests (no agent / no LLM cost) ──────────────────────


async def smoke_tests():
    print("── web_search ──────────────────────────────────────────────────")
    results = await web_search(query="latest AI news 2026", max_results=3)
    for r in results:
        if "error" in r:
            print(f"  [ERROR] {r['error']}")
        else:
            print(f"  ✓ {r['title']}")
            print(f"    {r['url']}")

    print("\n── fetch_url ───────────────────────────────────────────────────")
    page = await fetch_url(url="https://httpbin.org/get")
    if "error" in page:
        print(f"  [ERROR] {page['error']}")
    else:
        print(f"  ✓ status={page['status']}  chars={len(page['content'])}")

    print("\n── calculator ──────────────────────────────────────────────────")
    r = await calculator(expression="2**10 + sqrt(144)")
    print(f"  ✓ {r}")

    print("\n── get_datetime ────────────────────────────────────────────────")
    r = await get_datetime()
    print(f"  ✓ {r['date']}  {r['time']}")


asyncio.run(smoke_tests())


# ── 2. Agent with web search (uses LLM — needs API key) ──────────────────────

agent = helix.Agent(
    name="Researcher",
    role="Research analyst",
    goal="Answer questions using web search.",
    tools=[web_search, fetch_url, calculator],
)

print("\n── Agent run ───────────────────────────────────────────────────────")
result = helix.run(agent, "What are the top 3 AI news stories today?")

if result.error:
    print(f"[error] {result.error}")
else:
    print(result.output)
    print(f"\ntool_calls : {result.tool_calls}")
    print(f"cost       : ${result.cost_usd:.6f}")

#    sleep(seconds)                            → {slept_seconds}
#
# ── Custom tool cheat-sheet ──────────────────────────────────────────────────
#
#  @helix.tool(
#      description="...",   # shown to the LLM — be specific
#      timeout=10.0,        # seconds (optional, default 30)
#      retries=2,           # auto-retry count (optional, default 0)
#  )
#  async def my_tool(param: str, optional: int = 5) -> dict:
#      """
#      :param param:    Describe this for the LLM.
#      :param optional: Describe this for the LLM.
#      """
#      return {"result": param}
#
# ──────────────────────────────────────────────────────────────────────────────
