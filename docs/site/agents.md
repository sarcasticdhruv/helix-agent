# Agents

```python
import helix

agent = helix.Agent(
    name="Analyst",
    role="Senior data analyst",
    goal="Analyze datasets and produce concise summaries.",

    # Optional: rich background context that shapes agent behaviour
    backstory=(
        "You have 8 years of experience in financial data analysis. "
        "You prefer bullet-point summaries over long prose."
    ),

    # Model selection with automatic fallback
    model=helix.ModelConfig(
        primary="gpt-4o",
        fallback_chain=["gpt-4o-mini", "gemini-2.0-flash"],
        temperature=0.3,
    ),

    # Hard cost limit
    budget=helix.BudgetConfig(budget_usd=1.00),
    mode=helix.AgentMode.PRODUCTION,

    # Memory
    memory=helix.MemoryConfig(short_term_limit=20),

    # Semantic caching (cost reduction on repeated/similar queries)
    cache=helix.CacheConfig(enabled=True, semantic_threshold=0.92),
)

result = helix.run(agent, "Summarize last quarter's sales trends.")
```

`AgentResult` fields: `output`, `cost_usd`, `steps`, `model_used`, `cache_hits`, `cache_savings_usd`, `tool_calls`, `run_id`, `duration_s`, `trace`.

Agents also expose LangChain-compatible aliases: `agent.invoke(task)` (sync) and `await agent.ainvoke(task)` (async), both equivalent to `helix.run()` / `await agent.run()`.
