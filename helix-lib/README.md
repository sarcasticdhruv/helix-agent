# Helix

> **Production-grade AI agent framework** — cost governance, memory, semantic caching, multi-agent teams, and built-in evaluation

![PyPI](https://img.shields.io/pypi/v/helix-agent)
![Python](https://img.shields.io/pypi/pyversions/helix-agent)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

Helix builds on everything AutoGen and CrewAI gave us, then adds the production layer they left out: **budget enforcement**, **3-tier semantic caching**, **WAL-backed memory**, **HITL gates**, **5-scorer eval**, and a full **observability** suite — all wired together by default.

## Installation

```bash
pip install helix-agent                      # core only
pip install "helix-agent[gemini]"            # + Gemini (free tier ✓)
pip install "helix-agent[openai,anthropic]"  # + OpenAI + Anthropic
pip install "helix-agent[all]"              # every provider
```

Or from source:
```bash
git clone https://github.com/your-org/helix-agent
cd helix-agent/helix-lib
pip install -e ".[all]"
```

### Provider setup

**Option A — persistent config (recommended):**

```bash
helix config set GOOGLE_API_KEY   "AIza..."   # Gemini — free tier available
helix config set OPENAI_API_KEY   "sk-..."
helix config set ANTHROPIC_API_KEY "sk-ant-..."
```

Helix stores keys in `~/.helix/config.json` and picks the best available model automatically.

**Option B — environment variables:**

```bash
# Windows PowerShell
$env:GOOGLE_API_KEY = "AIza..."

# macOS / Linux
export GOOGLE_API_KEY="AIza..."
```

---

## Quickstart

```python
import helix

agent = helix.Agent(
    name="Researcher",
    role="Research analyst",
    goal="Find accurate, cited answers.",
)

# Synchronous — works in plain scripts, no asyncio needed
result = helix.run(agent, "What is quantum entanglement?")

print(result.output)
print(f"Cost:  ${result.cost_usd:.4f}")
print(f"Steps: {result.steps}")
```

**If you are inside an `async` function**, use `await` directly:

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

> **Common mistake:** `await agent.run(...)` must be inside an `async def` function.
> If you are writing a plain script, use `helix.run(agent, task)` instead — it handles the event loop for you.

---

## Core concepts

### Agent

```python
import helix

agent = helix.Agent(
    name="Analyst",
    role="Data analyst",
    goal="Analyze data and produce clear summaries.",

    # Model selection
    model=helix.ModelConfig(
        primary="gpt-4o",
        fallback_chain=["gpt-4o-mini", "claude-sonnet-4-6"],
        temperature=0.3,
    ),

    # Budget enforcement
    budget=helix.BudgetConfig(budget_usd=1.00),
    mode=helix.AgentMode.PRODUCTION,

    # Memory
    memory=helix.MemoryConfig(short_term_limit=20),

    # Caching (40-70% cost reduction on repeated queries)
    cache=helix.CacheConfig(enabled=True, semantic_threshold=0.92),
)

result = helix.run(agent, "Summarise last quarter's sales trends.")
```

### Tools

```python
import helix

@helix.tool(
    description="Search the web for current information.",
    timeout=15.0,
    retries=2,
)
async def web_search(query: str, max_results: int = 5) -> list:
    # Your implementation
    return [{"title": "...", "url": "...", "snippet": "..."}]


@helix.tool(description="Read a file from disk.")
async def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()


agent = helix.Agent(
    name="Researcher",
    role="Research analyst",
    goal="Find accurate answers using web search.",
    tools=[web_search, read_file],
)

result = helix.run(agent, "What are the latest AI news headlines?")
```

### Built-in tools

Import built-ins to register them globally:

```python
import helix.tools.builtin  # registers 12 tools in the global registry

# Available: web_search, fetch_url, read_file, write_file, list_directory,
#            calculator, json_query, get_datetime, get_env,
#            text_stats, extract_urls, sleep
```

### Budget enforcement

```python
import helix

agent = helix.Agent(
    name="Bot",
    role="Assistant",
    goal="Help users.",
    budget=helix.BudgetConfig(
        budget_usd=0.50,
        warn_at_pct=0.8,              # warn at 80% spent
        strategy=helix.BudgetStrategy.DEGRADE,  # switch to cheaper model instead of stopping
    ),
    mode=helix.AgentMode.PRODUCTION,  # budget required in PRODUCTION mode
)

try:
    result = helix.run(agent, "Write a 10,000 word essay...")
except helix.BudgetExceededError as e:
    print(f"Budget exceeded: spent ${e.spent_usd:.4f} of ${e.budget_usd:.4f}")
```

### Workflows

```python
import helix

@helix.step(name="search", retry=2)
async def search_step(query: str) -> list:
    # search implementation
    return []

@helix.step(name="summarise")
async def summarise_step(results: list) -> str:
    return "\n".join(str(r) for r in results)

pipeline = (
    helix.Workflow("research-pipeline")
    .then(search_step)
    .then(summarise_step)
    .with_budget(2.00)
)

result = pipeline.run_sync("quantum computing trends 2025")
print(result.final_output)
```

### Teams

```python
import helix

searcher = helix.Agent(name="Searcher", role="Web researcher", goal="Find sources.")
analyst  = helix.Agent(name="Analyst",  role="Data analyst",   goal="Analyse data.")
writer   = helix.Agent(name="Writer",   role="Technical writer", goal="Write reports.")

team = helix.Team(
    name="research-team",
    agents=[searcher, analyst, writer],
    strategy="sequential",   # searcher → analyst → writer
    budget_usd=5.00,
)

result = team.run_sync("Write a report on renewable energy trends.")
print(result.final_output)
print(f"Total cost: ${result.total_cost_usd:.4f}")
```

### Multi-turn sessions

```python
import asyncio
import helix

async def main():
    agent = helix.Agent(name="Bot", role="Assistant", goal="Help users.")
    session = helix.Session(agent=agent)
    await session.start()

    r1 = await session.send("My name is Alice.")
    r2 = await session.send("What is my name?")   # Agent remembers: Alice
    print(r2.output)

    await session.end()

asyncio.run(main())
```

### Framework adapters (LangChain, CrewAI, AutoGen)

Add Helix governance to existing code with one line:

```python
from langchain_openai import ChatOpenAI
import helix

# Before: llm = ChatOpenAI(model="gpt-4o")
llm = helix.wrap_llm(ChatOpenAI(model="gpt-4o"), budget_usd=2.00)

# Everything else stays identical — Helix adds:
#   ✓ Budget gate (raises BudgetExceededError before the call)
#   ✓ Cost tracking
#   ✓ Execution tracing
#   ✓ Audit log
```

```python
from crewai import Crew, Agent, Task
import helix

crew = Crew(agents=[...], tasks=[...])
wrapped = helix.from_crewai(crew, budget_usd=5.00)
result = await wrapped.run(inputs={"topic": "AI trends"})
print(f"Cost: ${wrapped.cost_usd:.4f}")
```

### Evaluation

```python
import asyncio
import helix
from helix.eval.suite import EvalSuite
from helix.config import EvalCase

suite = EvalSuite("my-agent-suite")

suite.add_cases([
    EvalCase(
        name="capital_cities",
        input="What is the capital of France?",
        expected_facts=["Paris"],
        max_cost_usd=0.05,
    ),
    EvalCase(
        name="math",
        input="What is 15% of 240?",
        expected_facts=["36"],
        max_cost_usd=0.05,
    ),
])

async def main():
    agent = helix.Agent(name="Bot", role="Assistant", goal="Answer questions accurately.")
    results = await suite.run(agent, verbose=True)
    print(f"Pass rate: {results.pass_rate:.0%}")
    print(f"Total cost: ${results.total_cost_usd:.4f}")
    suite.assert_pass_rate(0.90)   # Raises AssertionError if < 90%

asyncio.run(main())
```

---

## CLI

```bash
# Check environment
helix doctor

# View a trace
helix trace <run-id>

# Compare two runs for divergence
helix trace <run-id-a> --diff <run-id-b>

# Interactive failure replay
helix replay <run-id>

# List models with pricing
helix models

# Cost report
helix cost --all
```

---

## Architecture overview

```
helix/
├── core/           Agent, Workflow, Team, Session, Tool
├── memory/         Short-term buffer, WAL-backed long-term, episodic recall
├── cache/          Semantic cache (tier 1), Plan/APC cache (tier 2), Prefix (tier 3)
├── models/         Router, complexity estimator, 12 provider backends
├── safety/         Cost governor, permissions, guardrails, HITL, audit log
├── context_engine/ Multi-factor decay, intelligent compactor, preflight estimator
├── eval/           EvalSuite, 5 scorers, trajectory eval, regression gate, monitor
├── observability/  Tracer, ghost debug resolver, failure replay
├── adapters/       LangChain, CrewAI, AutoGen, LangGraph + universal LLM shim
├── runtime/        Event loop, worker pool, health checks
└── cli/            10 commands — doctor, models, cost, trace, replay, config …
```

---

## Environment variables

| Variable | Provider | Free tier |
|---|---|:---:|
| `GOOGLE_API_KEY` | Gemini 2.5 Flash/Pro, 2.0 Flash | ✓ |
| `OPENAI_API_KEY` | GPT-4o, GPT-4o-mini, o1, o3 | ✗ |
| `ANTHROPIC_API_KEY` | Claude Opus/Sonnet/Haiku | ✗ |
| `GROQ_API_KEY` | Llama 3, Mixtral, Gemma (ultra-fast) | ✓ |
| `MISTRAL_API_KEY` | Mistral Large/Small, Codestral | partial |
| `COHERE_API_KEY` | Command R+ | partial |
| `TOGETHER_API_KEY` | 200+ open-source models | ✗ |
| `OPENROUTER_API_KEY` | 100+ models via OpenRouter | partial |
| `DEEPSEEK_API_KEY` | DeepSeek V3 / R1 | ✗ |
| `XAI_API_KEY` | Grok models | ✗ |
| `PERPLEXITY_API_KEY` | Perplexity online models | ✗ |
| `FIREWORKS_API_KEY` | Fireworks fast inference | ✗ |

> Helix uses `helix config` or env variables — whichever is set. Multiple keys = automatic fallback routing.

---

## License

MIT

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.
