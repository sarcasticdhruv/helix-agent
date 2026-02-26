# Helix — Complete Usage Reference

Everything you need to build anything with Helix. Every API, every config option, every pattern — no details left out.

**Version:** 0.3.4 · **Python:** 3.10+

---

## Table of Contents

1. [Installation](#installation)
2. [API Key Setup](#api-key-setup)
3. [Core Concepts](#core-concepts)
4. [Quickstart Patterns](#quickstart-patterns)
5. [Agent — Full Reference](#agent--full-reference)
6. [Class-Based Agents (`@helix.agent`)](#class-based-agents-helixagent)
7. [Preset Agents (`helix.presets`)](#preset-agents-helixpresets)
8. [Agent Pipelines (`|` operator)](#agent-pipelines--operator)
9. [Tools — Full Reference](#tools--full-reference)
10. [Built-in Tools (13)](#built-in-tools-13)
11. [Tasks and Pipelines](#tasks-and-pipelines)
12. [YAML Configuration](#yaml-configuration)
13. [Multi-Agent Teams](#multi-agent-teams)
14. [Group Chat](#group-chat)
15. [Workflows](#workflows)
16. [StateGraph (LangGraph-compatible)](#stategraph-langgraph-compatible)
17. [Sessions](#sessions)
18. [Memory System](#memory-system)
19. [3-Tier Semantic Cache](#3-tier-semantic-cache)
20. [Budget Enforcement](#budget-enforcement)
21. [Event Hooks](#event-hooks)
22. [Human-in-the-Loop (HITL)](#human-in-the-loop-hitl)
23. [Safety — Permissions & Guardrails](#safety--permissions--guardrails)
24. [Structured Output](#structured-output)
25. [Observability — Tracing & Audit](#observability--tracing--audit)
26. [Context Engine](#context-engine)
27. [Evaluation Suite](#evaluation-suite)
28. [Framework Adapters](#framework-adapters)
29. [CLI — Full Reference](#cli--full-reference)
30. [Error Reference](#error-reference)
31. [Configuration Model Reference](#configuration-model-reference)
32. [Provider Routing Rules](#provider-routing-rules)

---

## Installation

```bash
# Minimal (requires pydantic)
pip install helix-framework

# Single provider
pip install "helix-framework[gemini]"        # Google Gemini (free tier)
pip install "helix-framework[openai]"        # OpenAI
pip install "helix-framework[anthropic]"     # Anthropic Claude

# Multiple providers
pip install "helix-framework[openai,anthropic,gemini]"

# All providers
pip install "helix-framework[all]"

# From source (editable, recommended for development)
git clone https://github.com/sarcasticdhruv/helix-agent
cd helix-agent
pip install -e ".[all]"
```

**Available extras:** `openai`, `anthropic`, `gemini`, `groq`, `mistral`, `cohere`, `together`,
`azure`, `openrouter`, `deepseek`, `ollama`, `dev`, `all`

---

## API Key Setup

### Persistent config store (recommended)

Keys saved to `~/.helix/config.json`, auto-loaded on every `import helix`:

```bash
helix config set GOOGLE_API_KEY    "AIza..."
helix config set OPENAI_API_KEY    "sk-..."
helix config set ANTHROPIC_API_KEY "sk-ant-..."
helix config set GROQ_API_KEY      "gsk_..."
helix config set MISTRAL_API_KEY   "..."
helix config set COHERE_API_KEY    "..."
helix config set TOGETHER_API_KEY  "..."
helix config set DEEPSEEK_API_KEY  "..."
helix config set XAI_API_KEY       "..."
helix config set PERPLEXITY_API_KEY "..."
helix config set FIREWORKS_API_KEY "..."
helix config set OPENROUTER_API_KEY "..."
helix config set AZURE_OPENAI_API_KEY "..."
helix config set AZURE_OPENAI_ENDPOINT "https://..."

# List saved keys (values masked)
helix config list

# Delete a key
helix config delete OPENAI_API_KEY

# Show config file path
helix config path
```

### Environment variables

```bash
# Linux / macOS
export GOOGLE_API_KEY="AIza..."

# Windows PowerShell
$env:GOOGLE_API_KEY = "AIza..."
```

### In code

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
import helix  # keys applied automatically if already in env
```

### Provider auto-detection priority

When `ModelConfig()` has no `primary` set, Helix selects the cheapest available model from:

1. Google Gemini (checks `GOOGLE_API_KEY`)
2. OpenAI (checks `OPENAI_API_KEY`)
3. Anthropic (checks `ANTHROPIC_API_KEY`)
4. Groq (checks `GROQ_API_KEY`)
5. Mistral, Cohere, Together, DeepSeek, xAI, Perplexity, Fireworks, OpenRouter

---

## Core Concepts

| Concept | Class | Purpose |
|---|---|---|
| **Agent** | `helix.Agent` | Single LLM-powered reasoning unit |
| **Tool** | `@helix.tool` | Function callable by an agent |
| **Task** | `helix.Task` | Declarative work unit assigned to an agent |
| **Pipeline** | `helix.Pipeline` | Ordered sequence of Tasks |
| **AgentPipeline** | `helix.AgentPipeline` | Chain agents with `\|` operator |
| **Team** | `helix.Team` | Multiple agents with shared strategy |
| **GroupChat** | `helix.GroupChat` | Multi-turn N-agent conversation |
| **Workflow** | `helix.Workflow` | Step-based DAG with branching/loops |
| **StateGraph** | `helix.StateGraph` | LangGraph-compatible graph engine |
| **Session** | `helix.Session` | Persistent multi-turn agent memory |
| **EvalSuite** | `helix.EvalSuite` | Automated agent evaluation |

---

## Quickstart Patterns

### Plain script (sync)

```python
import helix

agent = helix.Agent(name="Bot", role="Assistant", goal="Help users.")
result = helix.run(agent, "What is 2 + 2?")
print(result.output)
print(f"Cost: ${result.cost_usd:.6f}")
```

### Async function

```python
import asyncio, helix

async def main():
    agent = helix.Agent(name="Bot", role="Assistant", goal="Help users.")
    result = await agent.run("Explain transformers in one paragraph.")
    print(result.output)

asyncio.run(main())
```

### One-liner with `helix.quick()`

```python
import helix

agent = helix.quick("You are a concise Python tutor.", budget_usd=0.05)
result = helix.run(agent, "Explain list comprehensions.")
print(result.output)
```

### Jupyter notebook

```python
import helix

# helix.run() handles nested event loops automatically
agent = helix.quick("You are a helpful assistant.")
result = helix.run(agent, "What is quantum entanglement?")
print(result.output)
```

### With structured output

```python
from pydantic import BaseModel
import helix

class Summary(BaseModel):
    title: str
    key_points: list[str]
    word_count: int

agent = helix.Agent(name="Summariser", role="Editor", goal="Summarise text.")
result = helix.run(agent, "Summarise the theory of relativity.", output_schema=Summary)
summary: Summary = result.output
print(summary.title)
print(summary.key_points)
```

---

## Agent — Full Reference

### Constructor

```python
agent = helix.Agent(
    # Required
    name="Analyst",                    # Shown in traces and logs
    role="Senior data analyst",        # Role context in system prompt
    goal="Analyze data and summarize.", # Task purpose

    # Optional identity
    backstory=(                        # Rich background injected into system prompt
        "You have 8 years of experience in financial data analysis. "
        "You prefer bullet-point summaries."
    ),

    # Model selection
    model=helix.ModelConfig(
        primary="gpt-4o",                         # Primary model
        fallback_chain=["gpt-4o-mini", "gemini-2.0-flash"],  # Tried in order on failure
        temperature=0.3,
        max_tokens=4096,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        timeout_s=60.0,
    ),

    # Cost control
    budget=helix.BudgetConfig(
        budget_usd=1.00,               # Hard spend cap
        warn_at_pct=0.8,               # Warn at 80% used
        strategy=helix.BudgetStrategy.STOP,   # STOP or DEGRADE
    ),

    # Execution mode
    mode=helix.AgentMode.PRODUCTION,   # EXPLORE (default) or PRODUCTION

    # Tools
    tools=[my_search_fn, my_calc_fn],  # @helix.tool-decorated functions

    # Memory
    memory=helix.MemoryConfig(
        backend="inmemory",            # "inmemory" | "pinecone" | "qdrant" | "chroma"
        short_term_limit=20,
        auto_promote=True,             # Auto-promote important entries to long-term
        importance_threshold=0.7,
        embedding_model="text-embedding-3-small",
    ),

    # Semantic cache
    cache=helix.CacheConfig(
        enabled=True,
        semantic_threshold=0.92,       # Similarity threshold for cache hit
        ttl_seconds=3600,
        max_entries=10_000,
        exclude_patterns=["today", "now", "current", "latest"],  # Never cache these
        plan_cache_enabled=True,       # Cache successful agent plans (tier 2)
        plan_match_threshold=0.85,
    ),

    # Permissions
    permissions=helix.PermissionConfig(
        allowed_tools=None,            # None = all tools allowed
        denied_tools=["write_file"],   # Blacklist specific tools
        allowed_domains=["*.github.com", "arxiv.org"],  # For web tools
        max_file_size_mb=10.0,
    ),

    # Human-in-the-loop
    # (pass HITLConfig to agent via AgentConfig / extra_config)

    # Observability
    observability=helix.ObservabilityConfig(
        trace_enabled=True,
        trace_backend="local",         # "local" | "s3" | "otel"
        audit_enabled=True,
        audit_backend="local",
        metrics_enabled=True,
    ),

    # System prompt override (replaces auto-generated prompt)
    system_prompt="You are an expert at summarising financial data.",

    # Live event callback
    on_event=my_hook_fn,               # async or sync (HookFn)
)
```

### `AgentMode`

| Value | Description |
|---|---|
| `AgentMode.EXPLORE` | Relaxed limits, suitable for experimentation |
| `AgentMode.PRODUCTION` | Requires `budget`, hard caps `loop_limit` to 20, strict defaults |

### Running an agent

```python
# Synchronous — works everywhere
result = helix.run(agent, "task string")
result = agent.run_sync("task string")
result = agent.invoke("task string")           # LangChain alias (sync)

# Asynchronous
result = await agent.run("task string")
result = await helix.run_async(agent, "task string")
result = await agent.ainvoke("task string")    # LangChain alias (async)

# With options
result = await agent.run(
    "task string",
    session_id="my-session-123",     # Tie to persistent session
    parent_run_id="workflow-run-id", # For nested agents
    output_schema=MyPydanticModel,   # Structured output
)

# Streaming (tokens as they arrive, no tool calls)
async for chunk in agent.stream("task string"):
    print(chunk, end="", flush=True)
```

### `AgentResult` fields

```python
result.output             # str or Pydantic model (when output_schema set)
result.cost_usd           # Total USD spent
result.steps              # Number of reasoning loop iterations
result.run_id             # Unique run identifier
result.agent_id           # Agent's UUID
result.agent_name         # Agent name
result.duration_s         # Wall-clock seconds
result.tool_calls         # Number of tool invocations
result.cache_hits         # Cache hits this run
result.cache_savings_usd  # Cost saved by cache
result.episodes_used      # Long-term memory episodes recalled
result.model_used         # Actual model that handled the run
result.error              # Error string if run failed (never raises by default)
result.trace              # Full trace dict (when observability enabled)
```

### Agent utilities

```python
# Add a tool after construction
agent.add_tool(my_new_tool)              # Returns self (chainable)

# Clone with overrides (for A/B testing)
fast_agent = agent.clone(model=helix.ModelConfig(primary="gpt-4o-mini"))

# Pipe to another agent
pipeline = agent | analyst_agent | writer_agent
result = pipeline.run_sync("task")

# Access config
print(agent.name)
print(agent.agent_id)
print(agent.config.mode)
```

---

## Class-Based Agents (`@helix.agent`)

Define agents as annotated classes. The class docstring is the system prompt/goal; methods decorated with `@helix.tool` become the agent's tools.

```python
import helix

@helix.agent(
    model="claude-sonnet-4-6",
    budget_usd=2.00,
    mode="production",            # "explore" or "production"
    name="WebResearcher",         # Defaults to class name
    backstory="You are a thorough investigative researcher.",
)
class WebResearcher:
    """
    You are an expert web researcher.
    Find accurate, up-to-date information and always cite sources.
    """

    @helix.tool(description="Search the web for recent information.")
    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        from helix.tools.builtin import web_search
        return await web_search(query, max_results=max_results)

    @helix.tool(description="Fetch and read a URL.", timeout=20.0, retries=1)
    async def fetch(self, url: str) -> str:
        from helix.tools.builtin import fetch_url
        result = await fetch_url(url)
        return result.get("content", "")

# The decorator returns a factory — call it to create an Agent instance
researcher = WebResearcher()
result = helix.run(researcher, "Latest AI safety research 2026")
print(result.output)
```

**`@helix.agent` parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str \| None` | auto-detect | LLM model string |
| `budget_usd` | `float` | `0.50` | Hard spend cap |
| `mode` | `str` | `"explore"` | `"explore"` or `"production"` |
| `name` | `str \| None` | class name | Override agent name |
| `backstory` | `str` | `""` | Background context |
| `**agent_kwargs` | `Any` | — | Forwarded to `Agent.__init__` |

---

## Preset Agents (`helix.presets`)

Nine ready-made agents. Import from `helix.presets` or use via `helix.presets.<name>()`.

```python
from helix.presets import (
    web_researcher, writer, summariser, fact_checker,
    coder, code_reviewer, data_analyst, api_agent, assistant
)
import helix
```

### `web_researcher()`

```python
researcher = web_researcher(
    name="WebResearcher",    # default
    budget_usd=0.50,
    model=None,              # auto-detect
    max_results=5,           # max web search results
)
result = helix.run(researcher, "Top AI papers this week")
```

Tools: `web_search`, `fetch_url`

### `writer()`

```python
w = writer(
    name="Writer",
    style="clear, engaging prose",   # style hint injected into goal
    budget_usd=0.30,
    model=None,
)
```

### `summariser()`

```python
s = summariser(
    name="Summariser",
    style="concise bullet points",
    budget_usd=0.20,
)
```

### `fact_checker()`

```python
fc = fact_checker(budget_usd=0.50)
```

Tools: `web_search`

### `coder()`

```python
c = coder(
    name="Coder",
    language="Python",        # injected into role and goal
    budget_usd=1.00,
    allow_file_io=True,       # adds read_file + write_file tools
)
```

### `code_reviewer()`

```python
cr = code_reviewer(language="TypeScript", budget_usd=0.50)
```

Tools: `read_file`

### `data_analyst()`

```python
da = data_analyst(budget_usd=1.00)
```

Tools: `calculator`, `read_file`

### `api_agent()`

```python
api = api_agent(
    base_url="https://api.github.com",
    auth_token=TOKEN,
    budget_usd=0.50,
)
```

Tools: `fetch_url`

### `assistant()`

```python
a = assistant(domain="Python programming", budget_usd=0.25)
```

### `researcher` alias

```python
from helix.presets import researcher  # alias for web_researcher
```

---

## Agent Pipelines (`|` operator)

Chain agents so each agent's output becomes the next agent's input.

```python
# Build with | operator
pipeline = web_researcher() | summariser() | writer(style="blog post")

# Synchronous run
result = pipeline.run_sync("Quantum computing advances in 2026")

# Async run
result = await pipeline.run("Quantum computing advances in 2026")

# With session
result = pipeline.run_sync("task", session_id="session-123")

print(result.output)
print(f"Cost: ${result.cost_usd:.4f}")
```

### `helix.chain()` helper

```python
pipeline = helix.chain(web_researcher(), summariser(), writer())
result = pipeline.run_sync("task")
```

### `AgentPipeline` API

```python
from helix.core.pipeline import AgentPipeline

pipe = AgentPipeline([researcher, analyst, writer])

pipe.agents          # list of Agent instances
pipe.add(another)    # append an agent, returns self

# Compose pipelines
combined = pipe1 | pipe2
```

---

## Tools — Full Reference

### Defining a tool

```python
import helix

@helix.tool(
    description="Search the web for current information.",  # Required: shown to the LLM
    timeout=15.0,       # Seconds before ToolTimeoutError (default: no timeout)
    retries=2,          # Auto-retry on transient failure (default: 0)
    on_error="raise",   # "raise" | "return_empty" | callable
)
async def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Optional docstring — also included in the tool description."""
    # ... implementation
    return [{"title": "...", "url": "...", "snippet": "..."}]

# Sync functions work too
@helix.tool(description="Add two numbers.")
def add(a: float, b: float) -> float:
    return a + b
```

**`@helix.tool` parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `description` | `str` | required | What the tool does (shown to the LLM) |
| `timeout` | `float \| None` | `None` | Seconds before timeout |
| `retries` | `int` | `0` | Retry count on failure |
| `on_error` | `str \| callable` | `"raise"` | `"raise"`, `"return_empty"`, or a fallback callable |

### Tool registry

```python
from helix.core.tool import registry, ToolRegistry

# Global registry — all @helix.tool functions auto-register here
all_tools = registry.all()          # list[RegisteredTool]
tool = registry.get("web_search")   # RegisteredTool | None

# Per-agent registry (agent gets a copy of global + agent-specific tools)
# registered via Agent(tools=[...])

# Discover all registered tools
tools = helix.discover_tools()      # list[RegisteredTool]
```

### Accessing tool metadata

```python
t = registry.get("web_search")
t.name          # "web_search"
t.description   # The description string
t._fn           # The underlying function
t._timeout_s    # Timeout in seconds
t._retries      # Retry count
```

---

## Built-in Tools (13)

Import to register globally:

```python
import helix.tools.builtin
# or pick individual tools:
from helix.tools.builtin import web_search, fetch_url, execute_python
```

| Tool | Signature | Description |
|---|---|---|
| `web_search` | `(query: str, max_results: int = 5) -> list[dict]` | DuckDuckGo search. Requires `ddgs` package. |
| `fetch_url` | `(url: str, timeout: float = 10.0) -> dict` | HTTP GET, strips HTML tags, returns first 10K chars |
| `read_file` | `(path: str) -> str` | Read text file from disk |
| `write_file` | `(path: str, content: str) -> dict` | Write text to disk |
| `list_directory` | `(path: str = ".") -> list[str]` | List directory contents |
| `calculator` | `(expression: str) -> dict` | Safe math eval (`ast`-based, no `eval`) |
| `json_query` | `(data: str, query: str) -> Any` | JMESPath query over a JSON string |
| `get_datetime` | `(timezone: str = "UTC") -> dict` | Current date/time in any timezone |
| `get_env` | `(key: str, default: str = "") -> str` | Read environment variable |
| `text_stats` | `(text: str) -> dict` | Character / word / sentence counts |
| `extract_urls` | `(text: str) -> list[str]` | Regex-extract all URLs from text |
| `sleep` | `(seconds: float) -> dict` | Non-blocking async sleep |
| `execute_python` | `(code: str, timeout: float = 15.0) -> dict` | Sandboxed subprocess Python execution |

### `execute_python` details

Runs arbitrary Python code in an isolated subprocess. Returns:

```python
{
    "success": bool,
    "stdout": str,
    "stderr": str,
    "returncode": int,
}
```

**Blocked modules:** `subprocess`, `pty`, `ctypes`, `multiprocessing`, `os.system`.

```python
from helix.tools.builtin import execute_python

result = await execute_python("""
import math
print(math.pi * 4**2)
""", timeout=10.0)

print(result["stdout"])   # "50.26548245743669"
```

---

## Tasks and Pipelines

Tasks are declarative work units. They chain outputs, support guardrails, and save results to files.

### Defining Tasks and a Pipeline

```python
import helix

researcher = helix.Agent(name="Researcher", role="Research analyst", goal="Research {topic}.")
writer     = helix.Agent(name="Writer",     role="Technical writer",  goal="Write about {topic}.")

research = helix.Task(
    description="Research the latest advances in {topic}.",
    expected_output="A list of 5 key findings with sources.",
    agent=researcher,
)

article = helix.Task(
    description="Write a 3-paragraph article based on the research.",
    expected_output="A well-structured article, no jargon.",
    agent=writer,
    context=[research],          # Receives research.output automatically
    output_file="article.md",    # Written to disk on completion
)

pipeline = helix.Pipeline(tasks=[research, article])
result = pipeline.kickoff(inputs={"topic": "quantum computing"})

print(result.final_output)
print(f"Total cost: ${result.total_cost_usd:.4f}")

# Inspect per-task outputs
for task_output in result.task_outputs:
    print(f"Task:  {task_output.summary}")
    print(f"Raw:   {task_output.raw}")
    if task_output.pydantic:
        print(f"Model: {task_output.pydantic}")
    if task_output.json_dict:
        print(f"JSON:  {task_output.json_dict}")
```

### All `Task` options

| Parameter | Type | Description |
|---|---|---|
| `description` | `str` | Task instruction (supports `{variable}` templates) |
| `expected_output` | `str` | Describes what a good result looks like |
| `agent` | `Agent` | The agent that runs this task |
| `context` | `list[Task]` | Tasks whose outputs are passed as context |
| `output_schema` | `type[BaseModel]` | Pydantic model for structured output |
| `output_file` | `str` | Write raw output to this path on completion |
| `guardrail` | `callable \| str` | Single validation function or LLM-eval string |
| `guardrails` | `list[callable]` | Chained validation functions |
| `guardrail_max_retries` | `int` | Retries on guardrail failure (default `3`) |
| `async_execution` | `bool` | Run concurrently with other async tasks |
| `callback` | `callable` | Called with `TaskOutput` after completion |
| `markdown` | `bool` | Instruct agent to format output as Markdown |

### Guardrails on Tasks

**Callable guardrail:**

```python
from helix import Task, TaskOutput

def must_be_under_300_words(result: TaskOutput):
    words = len(result.raw.split())
    if words > 300:
        return False, f"Too long: {words} words (max 300)"
    return True, result.raw

task = helix.Task(
    description="Write a short summary.",
    expected_output="Under 300 words.",
    agent=writer,
    guardrail=must_be_under_300_words,
    guardrail_max_retries=2,
)
```

**LLM-validated string guardrail:**

```python
task = helix.Task(
    description="Write a product description for {product}.",
    expected_output="A professional product description.",
    agent=writer,
    guardrail="Must be professional, under 100 words, and avoid superlatives.",
)
```

**Multiple chained guardrails:**

```python
task = helix.Task(
    ...
    guardrails=[no_pii_check, word_count_check, format_check],
    guardrail_max_retries=3,
)
```

### `TaskOutput` fields

```python
task_output.raw           # Raw string output
task_output.pydantic      # Pydantic model instance (if output_schema set)
task_output.json_dict     # dict (if output_schema set)
task_output.summary       # Short summary string
task_output.to_dict()     # Full dict representation
```

### `PipelineResult` fields

```python
result.final_output       # str — output of the last task
result.task_outputs       # list[TaskOutput]
result.total_cost_usd     # total across all tasks
```

---

## YAML Configuration

### `agents.yaml`

```yaml
researcher:
  role: Senior Research Analyst
  goal: Find cutting-edge developments in {topic}.
  backstory: You work at a leading tech think tank with access to academic databases.

writer:
  role: Content Strategist
  goal: Write engaging, accurate articles about {topic}.
  backstory: You have 5 years of experience writing technical content for developers.
```

### `tasks.yaml`

```yaml
research_task:
  description: Research the latest developments in {topic}.
  expected_output: A structured report with at least 5 key findings.
  agent: researcher

write_task:
  description: Write a concise article based on the research.
  expected_output: A 3-paragraph article for a developer audience.
  agent: writer
  context: [research_task]
  output_file: output/article.md
```

### Loading in code

```python
import helix

# High-level (returns Pipeline ready to run)
pipeline = helix.from_yaml(
    "agents.yaml",
    "tasks.yaml",
    inputs={"topic": "large language models"},
)
result = pipeline.kickoff()
print(result.final_output)
```

**Low-level helpers:**

```python
from helix.core.yaml_config import load_agents, load_tasks, load_pipeline

agents   = load_agents("agents.yaml", inputs={"topic": "LLMs"})
tasks    = load_tasks("tasks.yaml", agents, inputs={"topic": "LLMs"})
pipeline = load_pipeline(tasks)
result   = pipeline.kickoff()
```

---

## Multi-Agent Teams

Teams coordinate multiple agents with shared budget and one of three execution strategies.

```python
import helix

searcher = helix.Agent(name="Searcher", role="Web researcher",   goal="Find sources.")
analyst  = helix.Agent(name="Analyst",  role="Data analyst",     goal="Analyze data.")
writer   = helix.Agent(name="Writer",   role="Technical writer", goal="Write reports.")
```

### Sequential strategy

Each agent receives the previous agent's full output as its input:

```python
team = helix.Team(
    name="research-team",
    agents=[searcher, analyst, writer],
    strategy="sequential",
    budget_usd=5.00,
)

result = team.run_sync("Write a report on renewable energy trends.")
print(result.final_output)
print(f"Total cost: ${result.total_cost_usd:.4f}")
```

### Parallel strategy

All agents run on the same input concurrently; outputs returned as a list:

```python
team = helix.Team(
    name="parallel-reviewers",
    agents=[security_reviewer, perf_reviewer, style_reviewer],
    strategy="parallel",
    budget_usd=3.00,
)
result = await team.run("Review this pull request: ...")
# result.final_output is a list of each agent's output
```

### Hierarchical strategy

A lead agent decomposes the task and delegates subtasks to specialists:

```python
lead = helix.Agent(name="Lead", role="Project lead", goal="Decompose and delegate tasks.")

team = helix.Team(
    name="product-team",
    agents=[searcher, analyst, writer],
    strategy="hierarchical",
    lead=lead,
    budget_usd=10.00,
    shared_memory=True,    # Agents share a common memory store
)
result = await team.run("Create a comprehensive market analysis for EVs.")
```

### `Team` parameters

| Parameter | Type | Description |
|---|---|---|
| `name` | `str` | Team name for traces |
| `agents` | `list[Agent]` | Participating agents |
| `strategy` | `str` | `"sequential"`, `"parallel"`, `"hierarchical"` |
| `lead` | `Agent \| None` | Lead agent (hierarchical only) |
| `budget_usd` | `float \| None` | Shared budget across all agents |
| `shared_memory` | `bool` | Share a common memory backend (default `True`) |

### Team methods

```python
result = team.run_sync("task")          # sync
result = await team.run("task")         # async
```

---

## Group Chat

N agents in a shared multi-turn conversation. Helix's equivalent of AutoGen's `GroupChat`.

```python
import asyncio, helix

ceo    = helix.ConversableAgent(name="CEO",    role="CEO",    goal="Make strategic decisions.")
cto    = helix.ConversableAgent(name="CTO",    role="CTO",    goal="Assess technical risk.")
lawyer = helix.ConversableAgent(name="Lawyer", role="Lawyer", goal="Flag compliance issues.")

chat = helix.GroupChat(
    agents=[ceo, cto, lawyer],
    max_rounds=6,
    speaker_selection="round_robin",   # see table below
    termination_keyword="AGREED",
)

async def main():
    result = await chat.run("Should we migrate our core product to microservices?")
    print(result.transcript())
    print(f"Rounds: {result.rounds}")
    print(f"Cost: ${result.total_cost_usd:.4f}")
    print(f"Terminated: {result.termination_reason}")

asyncio.run(main())
```

### Speaker selection

| Value | Behavior |
|---|---|
| `"round_robin"` | Agents speak in order (default) |
| `"auto"` | A coordinator LLM picks the most relevant next speaker |
| `"random"` | Random selection each round |
| `callable` | `fn(agents, history) -> Agent` |

### Termination options

```python
chat = helix.GroupChat(
    agents=[...],
    max_rounds=10,                          # Stop after N rounds
    termination_keyword="FINAL ANSWER",     # Stop when any message contains this
    termination_fn=lambda msgs: len(msgs) > 8,  # Custom predicate
)
```

### Human in the loop

```python
human = helix.HumanAgent(name="You")   # prompts the terminal each turn

chat = helix.GroupChat(
    agents=[agent1, agent2, human],
    max_rounds=5,
)
```

### `ConversableAgent` options

```python
agent = helix.ConversableAgent(
    name="CEO",
    role="CEO",
    goal="Make strategic decisions.",
    human_input=False,              # True = prompts human terminal on every turn
    max_consecutive_replies=3,      # Prevent one agent dominating
    # All standard Agent kwargs also accepted
    model=helix.ModelConfig(primary="gpt-4o"),
    budget=helix.BudgetConfig(budget_usd=2.00),
)
```

### `GroupChatResult` fields

```python
result.messages         # list[ChatMessage]
result.rounds           # number of completed rounds
result.total_cost_usd   # total spend
result.termination_reason  # "keyword" | "max_rounds" | "custom_fn" | "error"
result.transcript()     # formatted string of the full conversation
```

### `ChatMessage` fields

```python
msg.agent_name   # who spoke
msg.content      # message text
msg.round        # round number
msg.timestamp    # float (unix time)
```

---

## Workflows

Step-based directed pipelines with retry, timeout, fallback, branching, loops, map/reduce, and
HITL checkpoints.

### Basic sequential workflow

```python
import helix

@helix.step(name="search", retry=2, timeout_s=10.0)
async def search_step(query: str) -> list:
    return []  # your implementation

@helix.step(name="summarise", retry=1)
async def summarise_step(results: list) -> str:
    return "\n".join(str(r) for r in results)

wf = (
    helix.Workflow("research-pipeline")
    .then(search_step)
    .then(summarise_step)
    .with_budget(2.00)
)

result = wf.run_sync("quantum computing trends 2026")
print(result.final_output)
print(f"Cost: ${result.total_cost_usd:.4f}")
```

### All workflow builder methods

```python
wf = helix.Workflow("my-pipeline")

# Sequential step
.then(step_fn)

# Concurrent steps — outputs collected as a list
.parallel(step_a, step_b, step_c)

# Map: apply step to each item produced by items_fn(input)
.map(process_item_step, items_fn=lambda inp: inp["items"])

# Reduce: collapse a list to a single value
.reduce(lambda acc, x: acc + x, initial="")

# Branch: run if_true or if_false based on condition
.branch(
    condition=lambda x: len(x) > 100,
    if_true=long_handler_step,
    if_false=short_handler_step,
)

# Loop: repeat until predicate is true (or max_iter)
.loop(refine_step, until=lambda x: "FINAL" in x, max_iter=5)

# HITL checkpoint: pause and wait for human approval
.human_review(prompt="Is this output acceptable?", risk_level="high")

# Cost cap
.with_budget(5.00)

# Failure strategy: "fail" | "continue" | "fallback"
.on_failure("continue")

# Checkpoint directory (enables resume-after-crash)
.with_checkpoint(".helix/checkpoints/my-pipeline")

# Per-step callback
.on_step(lambda name, output: print(f"Step {name} done"))
```

### `@helix.step` decorator options

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str \| None` | `fn.__name__` | Step name in traces |
| `retry` | `int` | `0` | Retry count on failure |
| `fallback` | `callable \| None` | `None` | Called when all retries exhausted |
| `timeout_s` | `float \| None` | `None` | Per-step timeout |

### Checkpoint / resume

```python
wf = (
    helix.Workflow("etl-job")
    .then(extract_step)
    .then(transform_step)
    .then(load_step)
    .with_checkpoint(".helix/checkpoints/etl-job")
)

# First run — runs all steps, saves state after each
result = wf.run_sync(input_data)

# Subsequent run — skips already-completed steps
result = wf.run_sync(input_data, resume=True)
```

### Async run

```python
result = await wf.run("task input")
result = wf.run_sync("task input")   # sync wrapper
```

### `WorkflowResult` fields

```python
result.workflow_name    # str
result.final_output     # Any
result.steps            # list[StepResult]
result.total_cost_usd   # float
result.duration_s       # float
result.error            # str | None
```

---

## StateGraph (LangGraph-compatible)

Full directed graph engine with typed shared state, cycles, conditional edges, and checkpoint
persistence. Drop-in familiar for LangGraph users.

### Basic example

```python
import helix
from typing import TypedDict

class State(TypedDict):
    topic: str
    draft: str
    review_count: int
    approved: bool

researcher = helix.presets.web_researcher()
writer     = helix.presets.writer()
reviewer   = helix.presets.code_reviewer()

async def research_node(state: State) -> dict:
    result = await researcher.run(state["topic"])
    return {"draft": result.output}

async def write_node(state: State) -> dict:
    result = await writer.run(state["draft"])
    return {"draft": result.output}

async def review_node(state: State) -> dict:
    result = await reviewer.run(state["draft"])
    approved = "LGTM" in result.output or "approve" in result.output.lower()
    return {"approved": approved, "review_count": state["review_count"] + 1}

def route_after_review(state: State) -> str:
    if state["approved"]:
        return helix.END
    if state["review_count"] >= 3:
        return helix.END   # give up after 3 reviews
    return "write"         # cycle back to write

graph = (
    helix.StateGraph(State)
    .add_node("research", research_node)
    .add_node("write", write_node)
    .add_node("review", review_node)
    .add_edge("research", "write")
    .add_edge("write", "review")
    .add_conditional_edges(
        "review",
        route_after_review,
        {
            "write": "write",
            helix.END: helix.END,
        },
    )
    .set_entry_point("research")
    .compile(
        checkpoint_dir=".helix/checkpoints",   # optional: resume on crash
        max_steps=50,                          # safety guard (default 100)
    )
)

result = graph.run_sync({
    "topic": "Quantum computing in 2026",
    "draft": "",
    "review_count": 0,
    "approved": False,
})
print(result["draft"])
```

### `StateGraph` API

```python
g = helix.StateGraph(StateType)

# Add nodes — async or sync callables, or Agent instances
g.add_node("node_name", async_fn)
g.add_node("agent_node", helix_agent_instance)   # .run() called automatically

# Unconditional edges
g.add_edge("node_a", "node_b")
g.add_edge("node_a", helix.END)

# Conditional edges
g.add_conditional_edges(
    "source_node",
    routing_fn,          # fn(state: State) -> str
    {"route_a": "node_x", "route_b": "node_y"},
)

# Entry and finish points
g.set_entry_point("first_node")
g.set_finish_point("last_node")   # optional; END sentinel also works

# Compile
compiled = g.compile(
    checkpoint_dir=".helix/checkpoints/graph-name",  # None = no checkpointing
    max_steps=100,
)
```

### `CompiledGraph` execution

```python
# Synchronous
result_state = compiled.run_sync(initial_state)

# Asynchronous
result_state = await compiled.run(initial_state)

# LangChain-compatible aliases
result_state = await compiled.ainvoke(initial_state)
result_state = compiled.invoke(initial_state)   # sync

# Step-by-step streaming (async generator)
async for partial_state in compiled.stream(initial_state):
    print(partial_state)
```

### `GraphResult` fields (internal; `.run()` returns the final state dict)

```python
result["draft"]              # State field values
result.nodes_visited         # list[str]  (GraphResult only)
result.total_cost_usd        # float
result.duration_s            # float
result.error                 # str | None
```

### Sentinel constants

```python
helix.END    # "__end__"  — route here to terminate the graph
helix.START  # "__start__" — internal entry point sentinel
```

---

## Sessions

Sessions give an agent persistent memory across multiple turns.

```python
import asyncio, helix

async def main():
    agent = helix.Agent(name="Bot", role="Assistant", goal="Help users.")
    session = helix.Session(agent=agent)
    await session.start()

    r1 = await session.send("My name is Alice.")
    r2 = await session.send("What is my name?")   # remembers: Alice
    r3 = await session.send("What did I say first?")

    print(r2.output)
    print(r3.output)

    await session.end()

asyncio.run(main())
```

### `SessionConfig` options

```python
config = helix.SessionConfig(
    session_id="my-session-123",     # auto-generated UUID if omitted
    agent_id=agent.agent_id,
    tenant_id="org-456",             # optional multi-tenancy
    idle_timeout_s=1800.0,           # 30 minutes idle timeout
    max_duration_s=86400.0,          # 24 hour maximum
    store="inmemory",                # "inmemory" | "redis"
    store_config={},                 # redis URL, etc.
)

session = helix.Session(agent=agent, config=config)
```

### Passing `session_id` directly

```python
session_id = "user-session-abc123"

r1 = helix.run(agent, "My name is Bob.", session_id=session_id)
r2 = helix.run(agent, "What is my name?", session_id=session_id)
```

---

## Memory System

Helix uses a three-layer memory architecture managed automatically. You can also interact with it directly.

### Architecture

| Layer | Storage | Notes |
|---|---|---|
| Short-term buffer | In-process `list` | Rolling buffer, evicts lowest-importance when full |
| Long-term store | Pluggable backend | Vec­tor search, WAL-backed promotion |
| Episodic store | Long-term backend | Completed run summaries with success/failure info |

### `MemoryConfig` options

```python
memory = helix.MemoryConfig(
    backend="inmemory",            # "inmemory" | "pinecone" | "qdrant" | "chroma"
    short_term_limit=20,           # Max messages in rolling buffer
    auto_promote=True,             # Auto-promote important entries to long-term
    importance_threshold=0.7,      # Minimum importance score for auto-promotion
    embedding_model="text-embedding-3-small",  # Model used for semantic search
)
```

### `MemoryEntry` fields

```python
from helix.config import MemoryEntry, MemoryKind

entry = MemoryEntry(
    content="User prefers bullet-point summaries",
    kind=MemoryKind.PREFERENCE,    # FACT | PREFERENCE | TOOL_RESULT | REASONING | EPISODE_REF
    importance=0.9,                # 0.0–1.0; higher = survives eviction longer
    metadata={"source": "user_turn_3"},
    agent_id=agent.agent_id,
)
```

### `MemoryKind` enum

| Value | Use |
|---|---|
| `FACT` | Factual information recalled |
| `PREFERENCE` | User or task preferences |
| `TOOL_RESULT` | Cached tool output |
| `REASONING` | Agent's reasoning steps |
| `EPISODE_REF` | Reference to a completed episode |

### `Episode` fields (episodic memory records)

```python
from helix.config import Episode, EpisodeOutcome

# Episodes are stored automatically after each run
episode.id
episode.agent_id
episode.task               # The task string
episode.outcome            # EpisodeOutcome.SUCCESS | FAILURE | PARTIAL
episode.summary            # Human-readable summary
episode.steps              # Step count
episode.cost_usd
episode.tools_used         # list[str] of tool names
episode.failure_reason     # str | None
episode.learned_strategy   # Post-mortem LLM recommendation | None
```

### WAL-backed promotion

Helix uses a write-ahead log at `.helix/wal/memory_wal.json`. Memory entries that fail to persist to the backend are replayed on next startup — ensuring at-least-once delivery even across crashes.

---

## 3-Tier Semantic Cache

Helix reduces API costs by 40–70% using a three-tier cache hierarchy applied automatically before
every LLM call.

| Tier | Name | Mechanism | Typical hit rate |
|---|---|---|---|
| 1 | Semantic cache | Embedding similarity ≥ threshold | 40–70% on repeated queries |
| 2 | Plan cache (APC) | Reuse successful agent plans | Varies by task domain |
| 3 | Prefix cache | Common prompt-prefix deduplication | Provider-dependent |

### `CacheConfig` options

```python
cache = helix.CacheConfig(
    enabled=True,
    semantic_threshold=0.92,           # Cosine similarity must exceed this
    ttl_seconds=3600,                  # Cache entry lifetime (seconds)
    max_entries=10_000,                # Max entries before LRU eviction
    exclude_patterns=[                 # Never cache queries containing these strings
        "today", "now", "current", "latest", "price",
    ],
    plan_cache_enabled=True,           # Enable tier-2 plan caching
    plan_match_threshold=0.85,         # Similarity for plan template reuse
)
```

### Cache statistics via CLI

```bash
helix cache stats    # hit rates, savings, entry counts
helix cache clear    # flush all cached entries
```

### Monitoring cache savings in results

```python
result = helix.run(agent, "What is quantum entanglement?")
print(f"Cache hits:      {result.cache_hits}")
print(f"Savings:         ${result.cache_savings_usd:.4f}")
```

---

## Budget Enforcement

Helix enforces budgets **before** every LLM call — budget is a gate, not a report.

### `BudgetConfig` options

```python
budget = helix.BudgetConfig(
    budget_usd=1.00,                     # Hard spend cap (required)
    warn_at_pct=0.8,                     # Fire BudgetWarning event at 80%
    strategy=helix.BudgetStrategy.STOP,  # STOP or DEGRADE
    degradation_steps=[                  # Only used with DEGRADE
        helix.BudgetDegradationStep(
            at_pct=0.7,                  # At 70% spent...
            action="switch_model",
            switch_to_model="gpt-4o-mini",
        ),
        helix.BudgetDegradationStep(
            at_pct=0.9,
            action="summarize_conclude", # Force agent to conclude
        ),
    ],
)
```

### `BudgetStrategy` enum

| Value | Behavior |
|---|---|
| `STOP` | Raise `BudgetExceededError` (default) |
| `DEGRADE` | Step down through fallback chain / summarize as budget depletes |

### Catching budget errors

```python
import helix

try:
    result = helix.run(agent, "Write a 10,000-word essay on climate change...")
except helix.BudgetExceededError as e:
    print(f"Budget hit: ${e.spent_usd:.4f} / ${e.budget_usd:.4f}")
    print(f"Attempted:  ${e.attempted_usd:.4f}")
    print(f"Agent ID:   {e.agent_id}")
```

### Team / Workflow budgets

```python
team = helix.Team(agents=[...], strategy="sequential", budget_usd=5.00)
wf   = helix.Workflow("pipeline").then(s1).then(s2).with_budget(3.00)
```

---

## Event Hooks

Attach an `on_event` callback to any agent for live telemetry. Works alongside (or instead of) full
observability traces. Errors in hooks are silently swallowed — they never affect agent execution.

```python
import helix
from helix.core.hooks import HookEvent

async def my_hook(event: HookEvent) -> None:
    match event.type:
        case "step_start":
            print(f"[Step {event.step}] starting")
        case "step_end":
            preview = event.data["output_preview"]
            print(f"[Step {event.step}] done — {preview[:60]}")
        case "llm_call":
            print(f"[LLM] calling {event.data['model']} ({event.data['messages']} messages)")
        case "llm_response":
            print(f"[LLM] {event.data['tokens']} tokens, finish={event.data['finish_reason']}")
        case "tool_call":
            print(f"[TOOL] {event.data['tool_name']}({event.data['args']})")
        case "tool_result":
            print(f"[TOOL] result: {event.data['result_preview'][:60]}")
        case "tool_error":
            print(f"[TOOL ERROR] {event.data['tool_name']}: {event.data['error']}")
        case "cache_hit":
            print(f"[CACHE] similarity={event.data['similarity']:.3f} saved=${event.data['saved_usd']:.4f}")
        case "done":
            print(f"[DONE] {event.data['steps']} steps, ${event.data['cost_usd']:.4f}")
        case "error":
            print(f"[ERROR] {event.data['error']}")

agent = helix.Agent(
    name="Bot", role="Assistant", goal="Help users.",
    on_event=my_hook,    # async or sync callable
)
```

### `HookEvent` fields

| Field | Type | Description |
|---|---|---|
| `type` | `str` | Event identifier (see table below) |
| `data` | `dict` | Event-specific payload |
| `cost_so_far` | `float` | USD spent in this run so far |
| `step` | `int` | Current reasoning loop step (0-based) |

### All event types and payloads

| `event.type` | `event.data` keys |
|---|---|
| `"step_start"` | `step` |
| `"step_end"` | `step`, `output_preview` |
| `"llm_call"` | `model`, `messages` |
| `"llm_response"` | `model`, `tokens`, `finish_reason` |
| `"tool_call"` | `tool_name`, `args` |
| `"tool_result"` | `tool_name`, `result_preview` |
| `"tool_error"` | `tool_name`, `error` |
| `"cache_hit"` | `similarity`, `saved_usd` |
| `"done"` | `output_preview`, `steps`, `cost_usd` |
| `"error"` | `error` |

---

## Human-in-the-Loop (HITL)

Pause agent execution and request human approval before high-risk actions.

### `HITLConfig` options

```python
hitl = helix.HITLConfig(
    enabled=True,
    on_confidence_below=0.5,           # Trigger when model confidence < 0.5
    on_tool_risk=["write_file", "execute_python"],  # Always approve these tools
    on_cost_above_usd=0.50,            # Trigger if single LLM call > $0.50
    transport="cli",                   # "cli" | "webhook" | "queue"
    transport_config={},               # Transport-specific config (see below)
    timeout_seconds=300.0,             # Decision timeout (default 5 min)
)

agent = helix.Agent(
    name="Bot", role="Assistant", goal="...",
    # Pass via extra_config — HITLConfig is part of AgentConfig
)
```

### HITL transports

**CLI (local dev):**

```python
hitl = helix.HITLConfig(enabled=True, transport="cli")
# Blocks stdin and prompts: [a]pprove / [r]eject / [e]scalate
```

**Webhook (staging):**

```python
hitl = helix.HITLConfig(
    enabled=True,
    transport="webhook",
    transport_config={
        "webhook_url": "https://my-service.com/hitl/request",
        "response_url": "https://my-service.com/hitl/response",
        "poll_interval_s": 5.0,
    },
)
```

**Queue (production — Redis/SQS):**

```python
hitl = helix.HITLConfig(
    enabled=True,
    transport="queue",
    transport_config={
        "request_queue": "hitl-requests",
        "response_queue": "hitl-responses",
        "broker_url": "redis://localhost:6379/0",
    },
)
```

### `HITLDecision` enum

| Value | Meaning |
|---|---|
| `APPROVE` | Agent continues as planned |
| `REJECT` | Action is blocked |
| `ESCALATE` | Forwarded to a higher authority |
| `MODIFY` | Reviewer supplied a modified action |

### HITL in workflows

```python
wf = (
    helix.Workflow("deployment-pipeline")
    .then(build_step)
    .then(test_step)
    .human_review(prompt="Review test results before deploying.", risk_level="high")
    .then(deploy_step)
)
```

---

## Safety — Permissions & Guardrails

### `PermissionConfig`

```python
permissions = helix.PermissionConfig(
    allowed_tools=["web_search", "calculator"],  # Whitelist (None = all allowed)
    denied_tools=["write_file", "execute_python"],   # Blacklist
    allowed_domains=["*.arxiv.org", "*.github.com"], # For fetch_url
    max_file_size_mb=10.0,                           # Max readable file size
)
```

### Built-in guardrails

Import and pass to any task or wire into agent guardrail chain:

```python
from helix.safety.guardrails import PIIRedactor, LengthGuard, KeywordBlockGuard
```

**`PIIRedactor`** — detects and redacts PII patterns (email, phone, SSN, credit card, IP):

```python
guardrail = PIIRedactor(patterns=["email", "phone"])   # None = all patterns
# Passes but modifies content: replaces matches with [EMAIL_REDACTED], etc.
```

**`LengthGuard`** — min/max character enforcement:

```python
guardrail = LengthGuard(
    min_chars=10,
    max_chars=5000,
    on_fail="block",    # "block" | "truncate"
)
```

**`KeywordBlockGuard`** — block responses containing specific words:

```python
guardrail = KeywordBlockGuard(
    blocked_keywords=["competitor_name", "confidential"],
    case_sensitive=False,
)
```

### Safety audit log

Every tool call, HITL decision, guardrail pass/block, and budget event is recorded as an `AuditEntry` with tamper-evident chaining (each entry hashes the previous). Stored at `.helix/audit/`.

```python
from helix.config import AuditEntry, AuditEventType

# Verify a chain of audit entries
for i, entry in enumerate(entries[1:], 1):
    assert entry.verify(entries[i - 1]), f"Audit chain broken at entry {i}"
```

---

## Structured Output

Force agents to return typed Pydantic models instead of raw strings.

### Via `run()` / `run_sync()`

```python
from pydantic import BaseModel
import helix

class ResearchReport(BaseModel):
    title: str
    key_findings: list[str]
    sources: list[str]
    confidence: float

agent = helix.Agent(name="Researcher", role="Analyst", goal="Research topics.")

result = helix.run(
    agent,
    "Research the current state of quantum computing.",
    output_schema=ResearchReport,
)

report: ResearchReport = result.output
print(report.title)
print(report.key_findings)
```

### Via `StructuredOutputConfig` on the agent

```python
from helix.config import StructuredOutputConfig

agent = helix.Agent(
    name="Extractor",
    role="Data extractor",
    goal="Extract structured data.",
    structured_output=StructuredOutputConfig(
        enabled=True,
        schema=MyPydanticModel,
        strict=True,          # Raise on schema mismatch (default: retry)
        max_retries=3,
    ),
)
```

### Via `Task.output_schema`

```python
task = helix.Task(
    description="Extract key facts from this article: {text}",
    expected_output="Structured JSON with title, facts, and sources.",
    agent=agent,
    output_schema=ResearchReport,
)
```

---

## Observability — Tracing & Audit

### `ObservabilityConfig`

```python
obs = helix.ObservabilityConfig(
    trace_enabled=True,
    trace_backend="local",        # "local" | "s3" | "otel"
    trace_config={},              # Backend-specific config
    audit_enabled=True,
    audit_backend="local",
    audit_config={},
    metrics_enabled=True,
)
```

Traces are written to `.helix/traces/<run_id>.json`.

### CLI trace commands

```bash
# View a trace
helix trace <run-id>

# Diff two runs — find exact divergence point
helix trace <run-id-a> --diff <run-id-b>
# Output: diverged at step N, span X, likely cause: ..., recommendation: ...

# Interactive failure replay
helix replay <run-id>
```

### Accessing the trace in code

```python
obs_config = helix.ObservabilityConfig(trace_enabled=True)
agent = helix.Agent(..., observability=obs_config)

result = helix.run(agent, "task")
if result.trace:
    print(result.trace["spans"])
    print(result.trace["total_cost_usd"])
```

### Ghost Debug Resolver

Compares two run traces to find where (and why) they diverged:

```python
from helix.observability.ghost_debug import GhostDebugResolver

resolver = GhostDebugResolver(trace_dir=".helix/traces")
report = await resolver.compare(run_id_a, run_id_b)

if not report.identical:
    print(f"Diverged at step {report.diverged_at_step}, span: {report.diverged_at_span}")
    print(f"Likely cause: {report.likely_cause}")
    print(f"Recommendation: {report.recommendation}")
```

---

## Context Engine

Helix's context engine manages the full context window lifecycle automatically. You rarely need to
interact with it directly, but understanding it helps you tune agent behaviour.

### Multi-factor relevance scoring

Each message in the context window gets a relevance score computed as:

```
relevance = α × time_decay + β × reference_score + γ × semantic_similarity + δ × role_weight
```

Default weights: **α=0.3** (time), **β=0.4** (cited by LLM), **γ=0.2** (semantic), **δ=0.1**
(role).

**Role weights:** `SYSTEM=1.0`, `TOOL=0.8`, `USER=0.7`, `ASSISTANT=0.5`

**Time decay:** exponential. Rates: `SYSTEM=0.01`, `USER=0.03`, `TOOL=0.05`, `ASSISTANT=0.08`.
System context stays relevant longest; intermediate agent reasoning decays fastest.

### Pinned messages

```python
from helix.config import ContextMessage, ContextMessageRole

# Pinned messages are never evicted from the context window
msg = ContextMessage(
    role=ContextMessageRole.SYSTEM,
    content="Always respond in Spanish.",
    pinned=True,
)
```

### Intelligent compaction

When the context window approaches the model's token limit, the `IntelligentCompactor` uses a
cheap model (`gpt-4o-mini` by default) to summarize low-relevance messages before evicting them,
preserving meaning at a fraction of the token cost.

### Pre-flight estimation

Before executing an agent run, the `PreflightEstimator` estimates the token count and cost of
the current context. This feeds the budget gate — the run is blocked before it starts if the
estimated cost would exceed the budget.

### `AgentConfig` context options

```python
agent_config_extras = {
    "loop_limit": 50,                # Max reasoning loop iterations (default 50; 20 in production)
    "context_limit_tokens": 128_000, # Context window limit
}
```

---

## Evaluation Suite

Automated testing for agent correctness, cost efficiency, and behaviour.

### Basic setup

```python
import asyncio, helix
from helix.eval.suite import EvalSuite
from helix.config import EvalCase, ExpectedTrajectory

suite = EvalSuite(
    name="my-suite",
    results_dir=".helix/eval_results",   # Where to persist results
)
```

### Adding cases

**Programmatically:**

```python
suite.add_case(EvalCase(
    name="capitals",
    input="What is the capital of France?",
    expected_facts=["Paris"],
    max_cost_usd=0.05,
))

suite.add_cases([case1, case2, case3])
```

**Via `@suite.case` decorator:**

```python
@suite.case
def capitals():
    return EvalCase(
        input="What is the capital of Germany?",
        expected_facts=["Berlin"],
        max_cost_usd=0.05,
    )

@suite.case
def trajectory_check():
    return EvalCase(
        input="Search for Python tutorials and summarise them.",
        expected_tools=["web_search"],
        expected_trajectory=ExpectedTrajectory(
            tool_sequence=["web_search"],
            max_steps=5,
            must_not_call=["write_file"],
        ),
        max_cost_usd=0.10,
    )
```

### Running the suite

```python
async def main():
    agent = helix.Agent(name="Bot", role="Assistant", goal="Answer questions accurately.")
    results = await suite.run(
        agent,
        subset=["capitals", "trajectory_check"],   # None = run all
        verbose=True,                              # Print per-case output
    )

    print(f"Pass rate:  {results.pass_rate:.0%}")    # e.g. "80%"
    print(f"Total cost: ${results.total_cost_usd:.4f}")
    print(f"Duration:   {results.duration_s:.1f}s")

    # Per-case breakdown
    for r in results.results:
        status = "✓" if r.passed else "✗"
        print(f"  {status} {r.case_name}: {r.overall:.2f} ({r.cost_usd:.4f} USD)")
        for scorer, score in r.scores.items():
            print(f"       {scorer}: {score:.2f}")

    # Regression gate — raises AssertionError if below threshold
    suite.assert_pass_rate(0.90)

asyncio.run(main())
```

### 5 built-in scorers

| Scorer | What it measures |
|---|---|
| Factual accuracy | Checks `expected_facts` appear in the output |
| Tool usage | Validates `expected_tools` were called |
| Trajectory adherence | Checks `expected_trajectory` (sequence + forbidden calls) |
| Cost efficiency | Score = 1.0 if cost ≤ `max_cost_usd`, else decays linearly |
| Output format | Checks expected structure, length, and coherence |

### `EvalCase` options

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `""` (→ `fn.__name__`) | Case identifier |
| `input` | `str` | required | Task string sent to the agent |
| `expected_facts` | `list[str]` | `[]` | Strings that must appear in output |
| `expected_tools` | `list[str]` | `[]` | Tool names agent must call |
| `expected_trajectory` | `ExpectedTrajectory \| None` | `None` | Sequence/constraint checker |
| `max_steps` | `int` | `10` | Max reasoning steps |
| `max_cost_usd` | `float` | `1.00` | Cost cap per case |
| `pass_threshold` | `float` | `0.70` | Min overall score to pass |
| `tags` | `list[str]` | `[]` | Labels for filtering subsets |

### `ExpectedTrajectory` options

```python
trajectory = ExpectedTrajectory(
    tool_sequence=["web_search", "fetch_url"],  # Must be called in this order
    max_steps=8,                               # Fail if more steps used
    must_not_call=["write_file", "execute_python"],  # Forbidden tools
)
```

### `EvalRunResult` fields

```python
results.id                  # UUID
results.suite_name          # str
results.pass_count          # int
results.fail_count          # int
results.results             # list[EvalCaseResult]
results.total_cost_usd      # float
results.duration_s          # float
results.pass_rate           # property: pass_count / total
results.scores_by_case      # property: dict[case_name, overall_score]
```

### Regression gate in CI

```bash
# Run all EvalSuites in a file and fail CI if pass rate < 0.90
helix test my_eval_suite.py --gate 0.90 --verbose
```

---

## Framework Adapters

Wrap existing LangChain, CrewAI, or AutoGen objects with Helix cost governance.

### LangChain LLM wrapper

```python
from langchain_openai import ChatOpenAI
import helix

llm = helix.wrap_llm(
    ChatOpenAI(model="gpt-4o"),
    budget_usd=2.00,
)
# Adds: budget gate, cost tracking, distributed tracing, audit log
# Use llm as you normally would with LangChain
```

### LangChain chain

```python
from langchain.chains import LLMChain
import helix

wrapped = helix.from_langchain(
    LLMChain(llm=..., prompt=...),
    budget_usd=3.00,
)
result = await wrapped.run(inputs={"input": "Summarise this document..."})
print(f"Cost: ${wrapped.cost_usd:.4f}")
```

### CrewAI crew

```python
from crewai import Crew, Agent as CrewAgent, Task as CrewTask
import helix

crew = Crew(agents=[...], tasks=[...])
wrapped = helix.from_crewai(crew, budget_usd=5.00)

result = await wrapped.run(inputs={"topic": "AI trends 2026"})
print(f"Output: {result.output}")
print(f"Cost:   ${wrapped.cost_usd:.4f}")
```

### AutoGen agent

```python
from autogen import AssistantAgent
import helix

ag_agent = AssistantAgent("assistant", llm_config={"model": "gpt-4o"})
wrapped = helix.from_autogen(ag_agent, budget_usd=2.00)

result = await wrapped.run(inputs={"message": "Explain transformers"})
```

### Universal wrapper (any OpenAI-compatible client)

```python
any_client = SomeCompatibleLLMClient(...)
wrapped = helix.wrap_llm(any_client, budget_usd=1.00)
```

---

## CLI — Full Reference

### `helix doctor`

Check environment, provider keys, and subsystem health:

```bash
helix doctor
```

### `helix run`

Execute an agent Python file:

```bash
helix run my_agent.py
helix run my_agent.py --dry-run     # Estimate cost only, don't call LLMs
```

### `helix serve`

Expose an agent as a FastAPI HTTP service:

```bash
helix serve my_agent.py                         # Default: 0.0.0.0:8080
helix serve my_agent.py --host 127.0.0.1 --port 9000
helix serve my_agent.py --object agent          # Variable name in file (default: "agent")
helix serve my_agent.py --reload                # Auto-reload on file change
```

Endpoints:
- `POST /run` — `{"task": "...", "session_id": "optional"}`
- `GET  /health`

### `helix new`

Scaffold a new Helix agent project:

```bash
helix new my-project                              # Basic template
helix new my-project --template web-researcher   # Web research agent
helix new my-project --template workflow          # Workflow pipeline
helix new my-project --template team             # Multi-agent team
helix new my-project --template rag              # RAG agent
```

### `helix test`

Run `EvalSuite` objects from a Python file:

```bash
helix test my_eval.py
helix test my_eval.py --gate 0.90          # Fail if pass rate < 90%
helix test my_eval.py --verbose            # Show per-case results
```

### `helix trace`

Inspect run traces:

```bash
helix trace <run-id>                       # View full trace
helix trace <run-id> --diff <run-id-b>    # Diff two runs (ghost debug)
```

### `helix eval`

```bash
helix eval run                             # Run registered eval suite
helix eval compare <run-a> <run-b>        # Compare two eval run results
```

### `helix cache`

```bash
helix cache stats                          # Hit rates, savings, entry counts
helix cache clear                          # Flush all cached entries
```

### `helix cost`

```bash
helix cost                                 # Cost for current run
helix cost --all                           # Cost report across all runs
```

### `helix models`

```bash
helix models                               # List all available models with pricing
```

### `helix replay`

Interactive failure replay from a stored trace:

```bash
helix replay <run-id>
```

### `helix config`

```bash
helix config set OPENAI_API_KEY "sk-..."   # Save an API key
helix config list                          # List saved keys (masked)
helix config delete OPENAI_API_KEY         # Remove a key
helix config path                          # Show config file path (~/.helix/config.json)
```

---

## Error Reference

All errors inherit from `helix.HelixError`.

| Exception | Status | When raised |
|---|---|---|
| `HelixError` | 500 | Base class — all Helix exceptions |
| `HelixConfigError` | 422 | Invalid config (e.g. `production` mode without budget) |
| `BudgetExceededError` | 402 | Agent would exceed its budget |
| `ToolError` | 502 | Base class for all tool errors |
| `ToolTimeoutError` | 502 | Tool exceeded timeout |
| `ToolAuthError` | 401 | Tool credentials missing or invalid |
| `ToolSchemaMismatchError` | 422 | Arguments don't match tool schema |
| `ToolRateLimitError` | 429 | Upstream API quota hit |
| `ToolNotFoundError` | 404 | Target resource doesn't exist |
| `ToolPermissionError` | 403 | Agent lacks permission to call the tool |
| `ToolHallucinatedError` | 422 | Model called a nonexistent tool |
| `ToolNetworkError` | 502 | Transient connectivity failure |
| `ToolValidationError` | 502 | Tool output failed output schema |
| `ContextLimitError` | 413 | Context window exceeded and uncompactable |
| `MemoryBackendError` | 503 | Memory backend unavailable |
| `MemoryConflictError` | 409 | Optimistic lock conflict (shared memory) |
| `LoopDetectedError` | 508 | Agent stuck in a reasoning loop |
| `WorkflowError` | 500 | Workflow step failure |
| `AllModelsExhaustedError` | 503 | All models in fallback chain failed |
| `HelixProviderError` | 502 | Provider-specific error |

### Catching errors

```python
import helix

try:
    result = helix.run(agent, "task")
except helix.BudgetExceededError as e:
    print(e.spent_usd, e.budget_usd)
except helix.ToolPermissionError as e:
    print(e.tool_name)
except helix.LoopDetectedError:
    print("Agent got stuck — increase loop_limit or check your tools")
except helix.HelixError as e:
    print(f"Helix error: {e.message}")
    print(e.details)
```

---

## Configuration Model Reference

### `ModelConfig`

```python
helix.ModelConfig(
    primary=None,                            # Model string; None = auto-detect
    fallback_chain=[],                       # list[str] tried in order on failure
    temperature=0.7,
    max_tokens=None,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    timeout_s=60.0,
)
```

### `AgentConfig` (internal, auto-built by Agent.__init__)

```python
# All Agent.__init__ kwargs map directly to AgentConfig fields
helix.AgentConfig(
    name=..., role=..., goal=...,
    backstory="",
    mode=AgentMode.EXPLORE,
    model=ModelConfig(),
    budget=None,                           # Required in PRODUCTION mode
    memory=MemoryConfig(),
    cache=CacheConfig(),
    permissions=PermissionConfig(),
    hitl=HITLConfig(),
    guardrails=[],                         # list of guardrail names
    loop_limit=50,                         # 20 in PRODUCTION mode
    context_limit_tokens=128_000,
    structured_output=StructuredOutputConfig(),
    observability=ObservabilityConfig(),
    system_prompt_override=None,
    agent_id=...,                          # auto-generated UUID
)
```

**`production` mode rules:**
- `budget` is required or `HelixConfigError` is raised
- `loop_limit` is silently capped to `20`

### `WorkflowConfig`

```python
helix.WorkflowConfig(
    name="my-workflow",
    mode=helix.WorkflowMode.SEQUENTIAL,    # SEQUENTIAL | PARALLEL | CONDITIONAL | LOOP | MAP | REDUCE | HUMAN_REVIEW
    budget_usd=None,
    max_parallel=10,
    loop_limit=50,
    on_failure="fail",                     # "fail" | "continue" | "fallback"
)
```

### `TeamConfig`

```python
helix.TeamConfig(
    name="my-team",
    strategy="sequential",                 # "sequential" | "parallel" | "hierarchical"
    lead_agent_id=None,                    # Required for hierarchical
    shared_memory=True,
    budget_usd=None,
)
```

### `SessionConfig`

```python
helix.SessionConfig(
    session_id="...",                      # auto-generated UUID
    agent_id="...",
    tenant_id=None,
    idle_timeout_s=1800.0,                 # 30 min
    max_duration_s=86400.0,                # 24 hours
    store="inmemory",                      # "inmemory" | "redis"
    store_config={},
)
```

### `RuntimeConfig`

```python
helix.RuntimeConfig(
    workers=4,
    queue_max_size=1000,
    default_mode=helix.AgentMode.EXPLORE,
    trace_dir=".helix/traces",
    wal_dir=".helix/wal",
    health_check_interval_s=30.0,
    shutdown_timeout_s=30.0,
)
```

### `StructuredOutputConfig`

```python
helix.StructuredOutputConfig(
    enabled=False,
    schema=None,             # Pydantic model class or JSON Schema dict
    strict=False,            # True = raise on mismatch; False = retry
    max_retries=3,
)
```

---

## Provider Routing Rules

The `ModelRouter` determines which provider to call based on the model name:

| Pattern | Provider |
|---|---|
| `gpt-*`, `o1`, `o1-*`, `o3`, `o3-*` | OpenAI |
| `claude-*` | Anthropic |
| `gemini-*`, `models/gemini-*` | Google Gemini |
| `groq/*`, `llama-*`, `mixtral-*`, `gemma-*` | Groq |
| `mistral-*`, `codestral-*`, `pixtral-*`, `open-*` | Mistral |
| `command-*` | Cohere |
| `*/…` (slash in name), `together/*` | Together AI |
| `ollama/*`, `local/*` | Ollama |
| `azure/*` | Azure OpenAI |
| `openrouter/*` | OpenRouter |
| `deepseek-*` | DeepSeek |
| `grok-*` | xAI |
| Any model with explicit `base_url` | OpenAI-compatible generic |

### Fallback behaviour

When a model call fails (network, rate limit, auth), Helix automatically tries the next model in
`ModelConfig.fallback_chain`. When `BudgetStrategy.DEGRADE` is set, cheaper models are used
proactively as the budget depletes.

### Using Ollama (local models)

```python
agent = helix.Agent(
    name="LocalBot",
    role="Assistant",
    goal="Help users.",
    model=helix.ModelConfig(primary="ollama/llama3"),
)
```

Requires Ollama running at `http://localhost:11434`.

### Using any OpenAI-compatible endpoint

```python
agent = helix.Agent(
    name="CustomBot",
    role="Assistant",
    goal="Help users.",
    model=helix.ModelConfig(
        primary="my-custom-model",
        # Pass base_url via environment: OPENAI_BASE_URL=https://my-endpoint
    ),
)
```

---

## Complete Example: Production agent with everything

```python
"""
A production-grade agent demonstrating:
- Full AgentConfig
- Custom tools
- Event hooks
- Structured output
- Session memory
- Error handling
"""
import asyncio
import helix
from helix.core.hooks import HookEvent
from pydantic import BaseModel


class ResearchReport(BaseModel):
    title: str
    key_findings: list[str]
    sources: list[str]
    confidence: float


async def my_hook(event: HookEvent) -> None:
    if event.type in ("tool_call", "cache_hit", "done"):
        print(f"[{event.type}] {event.data}")


@helix.tool(description="Fetch news headlines about a topic.", timeout=10.0, retries=2)
async def get_headlines(topic: str) -> list[dict]:
    import helix.tools.builtin as bt
    return await bt.web_search(f"{topic} 2026 news", max_results=5)


async def main():
    agent = helix.Agent(
        name="ProductionResearcher",
        role="Senior research analyst",
        goal="Produce comprehensive, cited research reports.",
        backstory=(
            "You have 10 years of investigative journalism experience. "
            "You cite every claim with a URL and assess confidence honestly."
        ),
        model=helix.ModelConfig(
            primary="claude-sonnet-4-6",
            fallback_chain=["gpt-4o-mini", "gemini-2.0-flash"],
            temperature=0.2,
        ),
        budget=helix.BudgetConfig(
            budget_usd=2.00,
            warn_at_pct=0.7,
            strategy=helix.BudgetStrategy.DEGRADE,
        ),
        mode=helix.AgentMode.PRODUCTION,
        tools=[get_headlines],
        memory=helix.MemoryConfig(short_term_limit=20, auto_promote=True),
        cache=helix.CacheConfig(enabled=True, semantic_threshold=0.90),
        permissions=helix.PermissionConfig(denied_tools=["write_file"]),
        on_event=my_hook,
    )

    # Multi-turn session
    session_id = "user-42-session-1"

    # Run with structured output
    result = await agent.run(
        "Research the latest advances in protein folding AI.",
        session_id=session_id,
        output_schema=ResearchReport,
    )

    if result.error:
        print(f"Error: {result.error}")
    else:
        report: ResearchReport = result.output
        print(f"Title:      {report.title}")
        print(f"Confidence: {report.confidence:.0%}")
        for finding in report.key_findings:
            print(f"  • {finding}")
        print(f"\nCost: ${result.cost_usd:.4f} | Steps: {result.steps} | Cache hits: {result.cache_hits}")


asyncio.run(main())
```

---

## File Layout Reference

```
~/.helix/
├── config.json           # API keys saved by `helix config set`
└── ...

.helix/                   # Project-level runtime data (gitignore this)
├── traces/               # Run traces (JSON)
├── audit/                # Tamper-evident audit log
├── eval_results/         # EvalSuite run results
├── wal/                  # Write-ahead log for memory promotion
└── checkpoints/          # Workflow / StateGraph step checkpoints
```

---

*Helix v0.3.4 — Apache License 2.0 — Copyright 2026 Dhruv Choudhary*
