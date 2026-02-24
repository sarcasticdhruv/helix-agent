# Helix

**A Python framework for building production AI agents.**

[![PyPI](https://img.shields.io/pypi/v/helix-framework)](https://pypi.org/project/helix-framework/)
[![Python](https://img.shields.io/pypi/pyversions/helix-framework)](https://pypi.org/project/helix-framework/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/sarcasticdhruv/helix-agent/actions)

Helix gives you agents that actually behave in production: hard budget limits, semantic caching that cuts API costs by 40-70%, persistent memory, multi-agent teams, YAML-based task pipelines, and a 5-scorer eval suite. It works out of the box with OpenAI, Anthropic, Gemini, Groq, Mistral, and 8 other providers.

The `import helix` API is intentionally close to what you already know from AutoGen and CrewAI, but with the production layer those frameworks leave to you: cost governance, caching, memory, observability, and safety controls.

## Table of Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Agents](#agents)
- [Tools](#tools)
- [Tasks and Pipelines](#tasks-and-pipelines)
- [YAML Configuration](#yaml-configuration)
- [Multi-Agent Teams](#multi-agent-teams)
- [Group Chat](#group-chat)
- [Workflows](#workflows)
- [Sessions](#sessions)
- [Budget Enforcement](#budget-enforcement)
- [Evaluation](#evaluation)
- [Framework Adapters](#framework-adapters)
- [CLI](#cli)
- [Architecture](#architecture)
- [Supported Providers](#supported-providers)
- [Contributing](#contributing)

---

## Installation

```bash
pip install helix-framework                        # core only (pydantic required)
pip install "helix-framework[gemini]"              # + Google Gemini (free tier available)
pip install "helix-framework[openai,anthropic]"    # + OpenAI and Anthropic
pip install "helix-framework[all]"                 # all providers
```

From source:

```bash
git clone https://github.com/sarcasticdhruv/helix-agent
cd helix-agent
pip install -e ".[all]"
```

### API key setup

The easiest way is the persistent config store:

```bash
helix config set GOOGLE_API_KEY    "AIza..."    # Gemini, free tier works fine
helix config set OPENAI_API_KEY    "sk-..."
helix config set ANTHROPIC_API_KEY "sk-ant-..."
```

Keys are saved to `~/.helix/config.json`. Helix picks the best available model automatically when multiple keys are set.

Or use environment variables directly:

```bash
# Linux / macOS
export GOOGLE_API_KEY="AIza..."

# Windows PowerShell
$env:GOOGLE_API_KEY = "AIza..."
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

result = helix.run(agent, "What is quantum entanglement?")
print(result.output)
print(f"Cost:  ${result.cost_usd:.4f}")
print(f"Steps: {result.steps}")
```

Inside an async function, call `run_async` or `agent.run` directly:

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

---

## Agents

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

    # Semantic caching (40-70% cost reduction on repeated queries)
    cache=helix.CacheConfig(enabled=True, semantic_threshold=0.92),
)

result = helix.run(agent, "Summarize last quarter's sales trends.")
```

`AgentResult` fields: `output`, `cost_usd`, `steps`, `model_used`, `cache_hits`, `cache_savings_usd`, `tool_calls`, `run_id`, `duration_s`, `trace`.

---

## Tools

```python
import helix

@helix.tool(
    description="Search the web for current information.",
    timeout=15.0,
    retries=2,
)
async def web_search(query: str, max_results: int = 5) -> list:
    # your implementation here
    return [{"title": "...", "url": "...", "snippet": "..."}]


@helix.tool(description="Read a file from disk.")
async def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()


agent = helix.Agent(
    name="Researcher",
    role="Research analyst",
    goal="Find answers using web search.",
    tools=[web_search, read_file],
)

result = helix.run(agent, "What are the latest AI headlines?")
```

**Built-in tools** (12 included):

```python
import helix.tools.builtin  # registers tools globally

# web_search, fetch_url, read_file, write_file, list_directory,
# calculator, json_query, get_datetime, get_env,
# text_stats, extract_urls, sleep
```

---

## Tasks and Pipelines

Tasks are first-class declarative units of work. They chain outputs together, support output validation with guardrails, and can write results to files. This is the Helix equivalent of CrewAI's Task + crew.kickoff().

```python
import helix

researcher = helix.Agent(
    name="Researcher",
    role="Research analyst",
    goal="Find accurate information on {topic}.",
    backstory="You specialize in academic and technical research.",
)
writer = helix.Agent(
    name="Writer",
    role="Technical writer",
    goal="Write clear articles on {topic}.",
)

research = helix.Task(
    description="Research the latest advances in {topic}.",
    expected_output="A list of 5 key findings with sources.",
    agent=researcher,
)
article = helix.Task(
    description="Write a 3-paragraph article based on the research.",
    expected_output="A well-structured article, no jargon.",
    agent=writer,
    context=[research],        # automatically receives research output
    output_file="article.md",  # saved to disk when done
)

pipeline = helix.Pipeline(tasks=[research, article])
result = pipeline.kickoff(inputs={"topic": "quantum computing"})
print(result.final_output)
print(f"Total cost: ${result.total_cost_usd:.4f}")
```

**Task options:**

| Parameter | Description |
|---|---|
| `context` | List of Tasks whose outputs are passed as context |
| `output_schema` | Pydantic model for structured output |
| `guardrail` | Validation function or string description |
| `guardrails` | List of validation functions (chained) |
| `guardrail_max_retries` | How many times to retry on validation failure (default 3) |
| `output_file` | Path to write the task output |
| `async_execution` | Run this task concurrently with others |
| `callback` | Called with `TaskOutput` after completion |
| `markdown` | Instruct the agent to format output as Markdown |

**Validation with guardrails:**

```python
from helix import Task, TaskOutput

def must_be_under_300_words(result: TaskOutput):
    words = len(result.raw.split())
    if words > 300:
        return False, f"Too long: {words} words (max 300)"
    return True, result.raw

task = helix.Task(
    description="Write a short summary of {topic}.",
    expected_output="A summary under 300 words.",
    agent=writer,
    guardrail=must_be_under_300_words,
    guardrail_max_retries=2,
)
```

You can also pass a plain string and Helix uses the agent's own LLM to validate:

```python
task = helix.Task(
    description="Write a product description for {product}.",
    expected_output="A concise, professional product description.",
    agent=writer,
    guardrail="Must be professional, under 100 words, and avoid superlatives.",
)
```

**Accessing task output:**

```python
result = pipeline.kickoff(inputs={"topic": "AI safety"})

for task_output in result.task_outputs:
    print(f"Task:  {task_output.summary}")
    print(f"Raw:   {task_output.raw}")
    if task_output.pydantic:
        print(f"Model: {task_output.pydantic}")
```

---

## YAML Configuration

Define agents and tasks in YAML files for cleaner project structure:

```yaml
# agents.yaml
researcher:
  role: Senior Research Analyst
  goal: Find cutting-edge developments in {topic}.
  backstory: You work at a leading tech think tank with access to academic databases.

writer:
  role: Content Strategist
  goal: Write engaging, accurate articles about {topic}.
  backstory: You have 5 years of experience writing technical content for developers.
```

```yaml
# tasks.yaml
research_task:
  description: Research the latest developments in {topic}.
  expected_output: A structured report with at least 5 key findings.
  agent: researcher

write_task:
  description: Write a concise article based on the research.
  expected_output: A 3-paragraph article written for a developer audience.
  agent: writer
  context: [research_task]
  output_file: output/article.md
```

```python
import helix

pipeline = helix.from_yaml(
    "agents.yaml",
    "tasks.yaml",
    inputs={"topic": "large language models"},
)
result = pipeline.kickoff()
print(result.final_output)
```

Or use the lower-level helpers:

```python
from helix.core.yaml_config import load_agents, load_tasks, load_pipeline

agents   = load_agents("agents.yaml", inputs={"topic": "LLMs"})
tasks    = load_tasks("tasks.yaml", agents, inputs={"topic": "LLMs"})
pipeline = load_pipeline(tasks)
result   = pipeline.kickoff()
```

---

## Multi-Agent Teams

Teams coordinate multiple agents with three execution strategies.

```python
import helix

searcher = helix.Agent(name="Searcher", role="Web researcher",   goal="Find sources.")
analyst  = helix.Agent(name="Analyst",  role="Data analyst",     goal="Analyze data.")
writer   = helix.Agent(name="Writer",   role="Technical writer", goal="Write reports.")

# sequential: searcher output feeds into analyst, then into writer
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

**Strategies:**

- `sequential` - each agent receives the previous agent's output as its input
- `parallel` - all agents run on the same input concurrently, outputs returned as a list
- `hierarchical` - a lead agent decomposes the task and delegates subtasks to specialists

```python
lead = helix.Agent(name="Lead", role="Project lead", goal="Decompose and delegate tasks.")

team = helix.Team(
    name="product-team",
    agents=[searcher, analyst, writer],
    strategy="hierarchical",
    lead=lead,
)
```

---

## Group Chat

Group chat puts multiple agents in a shared multi-turn conversation. This is Helix's equivalent of AutoGen's `GroupChat`.

```python
import asyncio
import helix

ceo    = helix.ConversableAgent(name="CEO",    role="CEO",    goal="Make strategic decisions.")
cto    = helix.ConversableAgent(name="CTO",    role="CTO",    goal="Assess technical risk.")
lawyer = helix.ConversableAgent(name="Lawyer", role="Lawyer", goal="Flag compliance issues.")

chat = helix.GroupChat(
    agents=[ceo, cto, lawyer],
    max_rounds=6,
    speaker_selection="round_robin",  # or "auto", "random", or a callable
    termination_keyword="AGREED",
)

async def main():
    result = await chat.run("Should we migrate our core product to microservices?")
    print(result.transcript())
    print(f"Rounds: {result.rounds}, Cost: ${result.total_cost_usd:.4f}")

asyncio.run(main())
```

**Speaker selection:**

| Value | Behavior |
|---|---|
| `round_robin` | Agents speak in order (default) |
| `auto` | A coordinator LLM picks the most relevant next speaker |
| `random` | Random selection each round |
| `callable` | `fn(agents, history) -> Agent` |

**Termination:**

```python
chat = helix.GroupChat(
    agents=[...],
    max_rounds=10,
    termination_keyword="FINAL ANSWER",
    termination_fn=lambda msgs: len(msgs) > 8,
)
```

**Human in the loop:**

```python
human = helix.HumanAgent(name="You")   # prompts the terminal each turn

chat = helix.GroupChat(
    agents=[agent1, agent2, human],
    max_rounds=5,
)
```

---

## Workflows

Workflows are step-based directed pipelines with retry, timeout, fallback, and branching.

```python
import helix

@helix.step(name="search", retry=2, timeout_s=10.0)
async def search_step(query: str) -> list:
    return []  # your search implementation

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

---

## Sessions

Sessions give an agent persistent memory across multiple turns.

```python
import asyncio
import helix

async def main():
    agent = helix.Agent(name="Bot", role="Assistant", goal="Help users.")
    session = helix.Session(agent=agent)
    await session.start()

    r1 = await session.send("My name is Alice.")
    r2 = await session.send("What is my name?")   # remembers: Alice
    print(r2.output)

    await session.end()

asyncio.run(main())
```

---

## Budget Enforcement

```python
import helix

agent = helix.Agent(
    name="Bot",
    role="Assistant",
    goal="Help users.",
    budget=helix.BudgetConfig(
        budget_usd=0.50,
        warn_at_pct=0.8,
        strategy=helix.BudgetStrategy.DEGRADE,  # step down to cheaper model instead of stopping
    ),
    mode=helix.AgentMode.PRODUCTION,
)

try:
    result = helix.run(agent, "Write a 10,000 word essay on climate change...")
except helix.BudgetExceededError as e:
    print(f"Budget hit: ${e.spent_usd:.4f} of ${e.budget_usd:.4f}")
```

With `BudgetStrategy.DEGRADE`, Helix steps down through the fallback chain as the budget depletes rather than stopping outright.

---

## Evaluation

```python
import asyncio
import helix
from helix.eval.suite import EvalSuite
from helix.config import EvalCase

suite = EvalSuite("qa-suite")
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
    print(f"Pass rate:  {results.pass_rate:.0%}")
    print(f"Total cost: ${results.total_cost_usd:.4f}")
    suite.assert_pass_rate(0.90)   # raises AssertionError if below 90%

asyncio.run(main())
```

The eval suite runs 5 scorers per case: factual accuracy, tool usage, trajectory adherence, cost efficiency, and output format.

---

## Framework Adapters

Wrap existing LangChain, CrewAI, or AutoGen code with Helix cost governance:

```python
from langchain_openai import ChatOpenAI
import helix

llm = helix.wrap_llm(ChatOpenAI(model="gpt-4o"), budget_usd=2.00)
# adds budget gate, cost tracking, tracing, and audit log to any LangChain LLM
```

```python
from crewai import Crew
import helix

crew = Crew(agents=[...], tasks=[...])
wrapped = helix.from_crewai(crew, budget_usd=5.00)
result = await wrapped.run(inputs={"topic": "AI trends"})
print(f"Cost: ${wrapped.cost_usd:.4f}")
```

---

## CLI

```bash
helix doctor                          # check environment and provider keys
helix models                          # list available models with pricing
helix cost --all                      # cost report across all runs
helix trace <run-id>                  # view a run trace
helix trace <run-id> --diff <run-id>  # compare two runs for divergence
helix replay <run-id>                 # interactive failure replay
helix config set KEY value            # set a provider API key
```

---

## Architecture

```
helix/
├── core/            Agent, ConversableAgent, GroupChat, Task, Pipeline,
│                    Workflow, Team, Session, Tool
├── memory/          Short-term buffer, WAL-backed long-term store, episodic recall
├── cache/           Semantic cache (tier 1), plan cache (tier 2), prefix cache (tier 3)
├── models/          Router, complexity estimator, 12 provider backends
├── safety/          Cost governor, permission model, guardrails, HITL, audit log
├── context_engine/  Multi-factor token decay, context compactor, preflight estimator
├── eval/            EvalSuite, 5 scorers, trajectory eval, regression gate, monitor
├── observability/   Tracer, ghost debug resolver, failure replay
├── adapters/        LangChain, CrewAI, AutoGen + universal LLM wrapper
├── runtime/         Event loop, worker pool, health checks
└── cli/             doctor, models, cost, trace, replay, config, ...
```

---

## Supported Providers

| Environment variable | Provider | Models | Free tier |
|---|---|---|:---:|
| `GOOGLE_API_KEY` | Google Gemini | Gemini 2.5 Flash/Pro, 2.0 Flash | Yes |
| `OPENAI_API_KEY` | OpenAI | GPT-4o, GPT-4o-mini, o1, o3 | No |
| `ANTHROPIC_API_KEY` | Anthropic | Claude Opus/Sonnet/Haiku | No |
| `GROQ_API_KEY` | Groq | Llama 3, Mixtral, Gemma | Yes |
| `MISTRAL_API_KEY` | Mistral AI | Mistral Large/Small, Codestral | Partial |
| `COHERE_API_KEY` | Cohere | Command R+ | Partial |
| `TOGETHER_API_KEY` | Together AI | 200+ open-source models | No |
| `OPENROUTER_API_KEY` | OpenRouter | 100+ models | Partial |
| `DEEPSEEK_API_KEY` | DeepSeek | DeepSeek V3, R1 | No |
| `XAI_API_KEY` | xAI | Grok | No |
| `PERPLEXITY_API_KEY` | Perplexity | Online search models | No |
| `FIREWORKS_API_KEY` | Fireworks | Fast open-source inference | No |

Set multiple keys and Helix automatically falls back to the next available provider on failure.

---

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

```bash
git clone https://github.com/YOUR_USERNAME/helix-agent
cd helix-agent
pip install -e ".[dev,gemini]"
pytest tests/
```

---

## Contributors

| Name | Role |
|:---|:---|
| [Dhruv Choudhary](https://github.com/sarcasticdhruv) | Author and maintainer |

---

## License

[Apache License 2.0](LICENSE). Copyright 2026 Dhruv Choudhary.

See [CHANGELOG.md](CHANGELOG.md) for release history.
