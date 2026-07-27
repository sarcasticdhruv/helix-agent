# Framework Comparison

| Feature | Helix | AutoGen | CrewAI | LangGraph |
|---------|:---:|:---:|:---:|:---:|
| **Cost Governance** | ✓ Hard budget limits + semantic caching | ✗ | ✗ | ✗ |
| **Memory** | ✓ Multi-tier (short-term, episodic, WAL-backed) | Partial | Partial | ✗ |
| **Semantic Caching** | ✓ Tier 1 embedding-based cache | ✗ | ✗ | ✗ |
| **Multi-Agent Teams** | ✓ Handoffs + group chat | ✓ | ✓ | ✗ |
| **YAML Pipelines** | ✓ Task + workflow YAML | ✗ | ✓ | ✗ |
| **Built-in Eval Suite** | ✓ 6 scorers + regression gate | ✓ | ✗ | ✗ |
| **MCP Tools** | ✓ Native MCP client | ✗ | ✗ | ✗ |
| **Guardrails** | ✓ Prompt injection, HITL, audit log | Partial | Partial | ✗ |
| **LangGraph Compat** | ✓ StateGraph, `StateGraph.START` | ✗ | ✗ | Native |
| **Framework Wrappers** | ✓ LangChain, CrewAI, AutoGen | N/A | ✓ | ✗ |

**When to choose Helix:**
- You need **hard cost control** over multi-step agents (budget overruns are expensive in production)
- **Caching matters** — repeating queries with semantic similarity wastes money
- You want **production observability** (traces, eval gates, failure replay)
- Your team already uses **CrewAI/AutoGen/LangChain** and needs governance overlay
