# External Growth Content — Drafts Only

Nothing in this file gets posted automatically. Review and post manually.

## Show HN

**Title:** Show HN: Helix – a Python AI agent framework with hard budget limits

**Body:**

Hi HN — I built Helix after reading one too many stories like the one on the
CrewAI community forum ("How are you handling agent spending limits in
production crews?"): an agency's crew hit a tool error, kept retrying, and
racked up an overnight $2,400 API bill — with no native way to cap spend
before it happened.

Helix is a production-grade Python agent framework: hard per-run budget caps
(`BudgetConfig(budget_usd=...)` raises before an overspend, not after), a
semantic cache that cuts cost on repeated queries, multi-tier memory,
native MCP tool support, and a 6-scorer eval suite with regression gates.
It wraps existing LangChain/CrewAI/AutoGen agents rather than requiring a
rewrite.

Docs: https://sarcasticdhruv.github.io/helix-agent/
Repo: https://github.com/sarcasticdhruv/helix-agent
PyPI: pip install helix-framework

Demo (budget cap stopping a runaway agent, no API key needed to reproduce):
[link the demo.gif or a short screen recording]

Would love feedback, especially from anyone who's been burned by
runaway agent costs.

## Reddit — r/LocalLLaMA

**Title:** Built a Python agent framework with hard budget limits + semantic caching (Apache 2.0)

**Body:**

Saw a CrewAI forum thread recently where a team's crew hit a tool error,
kept retrying, and ran up a $2,400 bill overnight because there was no
native way to cap spend before it ran — that's basically why I built this.
Helix adds a hard per-run budget cap (`BudgetConfig(budget_usd=...)`) that
raises before an overspend instead of after, plus a semantic cache so
repeated/similar queries don't re-hit the model — sharing in case it's
useful to others running multi-step agents against local or paid models.
Works with Ollama and any OpenAI-compatible endpoint alongside the usual
cloud providers.
Repo: https://github.com/sarcasticdhruv/helix-agent — happy to answer
questions or take feedback on the API.

## Reddit — r/MachineLearning

**Title:** [P] Helix: cost-governed multi-agent framework (budget caps, semantic caching, built-in eval)

**Body:**

Posting for feedback, not promotion — built this after hitting cost/observability
gaps in CrewAI/AutoGen for production use. Core ideas: hard budget enforcement
per run, an embedding-based semantic cache for repeated queries, and a 6-scorer
eval suite with regression gates so agent behavior changes are caught in CI.
Apache 2.0. Comparison table against CrewAI/AutoGen/LangGraph:
https://github.com/sarcasticdhruv/helix-agent#framework-comparison

## Awesome-list PR descriptions

Use for PRs to repos like `awesome-ai-agents`, `awesome-llm-apps`, or similar
curated lists — check each list's own contribution guidelines for exact
format (usually a single line addition to a markdown table/list) before
opening the PR.

**One-line entry (typical awesome-list format):**
```
- [Helix](https://github.com/sarcasticdhruv/helix-agent) - Production-grade Python AI agent framework with hard budget limits, semantic caching, multi-agent teams, and native MCP support.
```

**PR description:**

Adding Helix, a Python AI agent framework focused on production
concerns most agent frameworks leave to the user: hard per-run cost caps,
semantic caching, multi-tier memory, and a built-in eval suite. Apache 2.0
licensed. Happy to adjust the entry format/wording to match this list's
conventions.
