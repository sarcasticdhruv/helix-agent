# FAQ

**Is Helix free?**
Yes. Apache 2.0 license, no paid tier, no usage limits imposed by Helix itself (only your own budget config and provider costs apply).

**Does Helix support MCP servers?**
Yes — Helix ships a native MCP client. Connect to any MCP server and its tools become available to your agents automatically. See [MCP Tools](mcp-tools.md).

**Can I use Helix with LangChain, CrewAI, or AutoGen agents I already built?**
Yes. `helix.wrap_llm()`, `helix.from_langchain()`, `helix.from_crewai()`, and `helix.from_autogen()` wrap existing code from those frameworks with Helix's cost governance, caching, and observability — no rewrite required. See [Framework Adapters](https://github.com/sarcasticdhruv/helix-agent#framework-adapters).

**Does Helix work with local models?**
Yes, via Ollama (`ollama/*` or `local/*` model strings) or any OpenAI-compatible endpoint (Azure, custom base URLs), with no environment variable requirements.

**How is Helix different from LangGraph?**
Helix ships a LangGraph-compatible `StateGraph` (same `START`/`END` API) plus things LangGraph doesn't provide out of the box: hard budget limits, semantic caching, multi-tier memory, and a built-in eval suite. See the [Framework Comparison](framework-comparison.md) table above.

**What does "hard budget limits" actually mean?**
Every agent run can be capped with `BudgetConfig(budget_usd=...)`. If a run would exceed that cap, Helix raises `BudgetExceededError` instead of continuing to spend — no silent overage.

**Does Helix require an API key to try it?**
No — Google Gemini and Groq both have usable free tiers, and `helix doctor` will tell you which providers are configured.

**Can a multi-agent crew still blow through budget if a tool call keeps failing and retrying?**
Not if you wrap it with Helix: `helix.from_crewai(crew, budget_usd=5.00)` (or `from_langchain`/`from_autogen`) puts a hard `BudgetConfig`-backed cap on top of the existing crew and raises `BudgetExceededError` the moment spend would cross that line, no matter how many times a failing tool call retries. See [Framework Adapters](https://github.com/sarcasticdhruv/helix-agent#framework-adapters).

**Does a multi-agent Group Chat need a manual reply cap to stop token spend from multiplying?**
No — `helix.GroupChat` takes `max_rounds`, `termination_keyword`, and `termination_fn` directly, so you can bound a conversation without hand-rolling an AutoGen-style `max_consecutive_auto_reply` limit. See [Group Chat](https://github.com/sarcasticdhruv/helix-agent#group-chat).

**How would I notice an agent getting expensive before the bill arrives, not after?**
Every agent can emit live cost telemetry through `on_event` — the `step_end` event carries `cost_so_far` — and `BudgetConfig(warn_at_pct=0.8)` fires a warning at 80% of budget well before `BudgetExceededError` would trigger. See [Event Hooks](https://github.com/sarcasticdhruv/helix-agent#event-hooks) and [Budget Enforcement](https://github.com/sarcasticdhruv/helix-agent#budget-enforcement).

**How do I keep a human in the loop instead of letting agents run fully autonomously?**
Add a `helix.HumanAgent` to any `GroupChat` — it prompts the terminal for your input each turn just like any other agent in the rotation, giving you a manual checkpoint instead of a fully autonomous loop. Pair it with Helix's guardrails (prompt injection blocking, audit log) for a record of what ran. See [Group Chat](https://github.com/sarcasticdhruv/helix-agent#group-chat) and [Guardrails](https://github.com/sarcasticdhruv/helix-agent#guardrails).
