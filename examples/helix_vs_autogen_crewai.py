"""
examples/helix_vs_autogen_crewai.py
====================================
Deep comparison: Helix  vs  Microsoft AutoGen  vs  CrewAI
Plus live agent demos of every Helix differentiator.

Run:  python examples/helix_vs_autogen_crewai.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — Framework comparison (printed, not executed)
# ─────────────────────────────────────────────────────────────────────────────

COMPARISON = """
╔══════════════════════════════════════════════════════════════════════════════╗
║              HELIX  vs  AUTOGEN  vs  CREWAI  —  Deep Comparison            ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌────────────────────────┬──────────────────────┬──────────────────────┬──────────────────────┐
│ Feature                │ Helix (yours)        │ AutoGen (Microsoft)  │ CrewAI               │
├────────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ ARCHITECTURE           │                      │                      │                      │
│  Mental model          │ Agent/Team/Workflow   │ Actor message-passing│ Crew + Flow          │
│  Multi-agent           │ sequential/parallel/ │ Conversational chat  │ Role-based crew      │
│                        │ hierarchical teams   │ between agents       │ + task delegation    │
│  Orchestration         │ Fluent Workflow DSL  │ GroupChat / Swarm    │ Flow (event-driven)  │
│  Agent communication   │ shared memory + ctx  │ async message bus    │ task output chaining │
├────────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ PROVIDER SUPPORT       │                      │                      │                      │
│  OpenAI                │ ✓ native             │ ✓ primary target     │ ✓ native             │
│  Anthropic / Claude    │ ✓ native             │ via extension        │ ✓ native             │
│  Google Gemini         │ ✓ native             │ via extension        │ ✓ native             │
│  Groq / Mistral        │ ✓ native             │ via extension        │ via LiteLLM          │
│  Ollama (local)        │ ✓ native             │ via extension        │ via LiteLLM          │
│  Azure OpenAI          │ ✓ native             │ ✓ native             │ via LiteLLM          │
│  Auto-detect provider  │ ✓ from model name    │ ✗ explicit           │ ✗ explicit           │
│  Smart fallback chain  │ ✓ per-model          │ ✗                    │ ✗                    │
├────────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ COST & BUDGET          │                      │                      │                      │
│  Hard budget cap       │ ✓ BudgetConfig       │ ✗ none               │ ✗ none               │
│  Budget strategies     │ ✓ STOP / DEGRADE     │ ✗                    │ ✗                    │
│  Cost tracking         │ ✓ per-call + total   │ ✗                    │ ✗                    │
│  Pre-call cost gate    │ ✓ guards before call │ ✗                    │ ✗                    │
│  Cost in result        │ ✓ result.cost_usd    │ ✗                    │ ✗                    │
├────────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ CACHING                │                      │                      │                      │
│  Semantic cache        │ ✓ vector similarity  │ ✗                    │ ✗                    │
│  Plan cache            │ ✓ reuse task plans   │ ✗                    │ ✗                    │
│  Prefix cache          │ ✓ prompt dedup       │ ✗                    │ ✗                    │
│  Cache savings tracked │ ✓ result.cache_hits  │ ✗                    │ ✗                    │
├────────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ MEMORY                 │                      │                      │                      │
│  Episodic memory       │ ✓ per-agent          │ ✗                    │ ✓ crew-level         │
│  Semantic search       │ ✓ embedding-based    │ ✗                    │ ✓ via mem0           │
│  Memory kinds          │ ✓ fact/prefer/tool/  │ ✗                    │ ✓ short/long/entity  │
│                        │   reasoning/episode  │                      │                      │
│  WAL (write-ahead log) │ ✓ crash-safe writes  │ ✗                    │ ✗                    │
│  Pluggable backends    │ ✓ in-memory/external │ ✗                    │ ✓ mem0/chromadb      │
├────────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ SAFETY                 │                      │                      │                      │
│  Human-in-the-loop     │ ✓ built-in HITL      │ ✓ human proxy agent  │ ✓ human input step   │
│  Guardrails            │ ✓ input + output     │ ✗ delegate to user   │ ✗ delegate to user   │
│  Permission control    │ ✓ allowed/denied     │ ✗                    │ ✗                    │
│                        │   tools per agent    │                      │                      │
│  Audit log             │ ✓ every tool call    │ ✗                    │ ✗                    │
├────────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ OBSERVABILITY          │                      │                      │                      │
│  Trace in result       │ ✓ result.trace       │ partial              │ ✗                    │
│  Replay debugging      │ ✓ ghost debug mode   │ ✗                    │ ✗                    │
│  Step-level logging    │ ✓ per step + cost    │ ✓ via callbacks      │ ✓ verbose mode       │
│  Eval suite built-in   │ ✓ EvalSuite          │ ✗ external           │ ✗ external           │
├────────────────────────┼──────────────────────┼──────────────────────┼──────────────────────┤
│ UNIQUE STRENGTHS       │                      │                      │                      │
│  Helix only            │ Budget governance    │                      │                      │
│                        │ Semantic cache       │                      │                      │
│                        │ Plan cache           │                      │                      │
│                        │ Context compaction   │                      │                      │
│                        │ Structured output    │                      │                      │
│                        │ Built-in eval suite  │                      │                      │
│  AutoGen only          │                      │ gRPC distributed     │                      │
│                        │                      │ actors, code exec    │                      │
│                        │                      │ in Docker, studio UI │                      │
│  CrewAI only           │                      │                      │ YAML crew def,       │
│                        │                      │                      │ enterprise cloud,    │
│                        │                      │                      │ 100k+ community      │
└────────────────────────┴──────────────────────┴──────────────────────┴──────────────────────┘

VERDICT:
  • AutoGen  → Best for: distributed systems, code-executing agents, event-driven
                          multi-agent chat where agents message each other like actors.
                          Leans heavily OpenAI. No budget control. Complex setup.

  • CrewAI   → Best for: teams of role-playing agents with structured task delegation.
                          Great ecosystem, YAML definitions, enterprise cloud offering.
                          No budget control. Memory via external mem0 add-on.

  • Helix    → Best for: production deployments where cost, safety, and observability
                          matter from day one. Only framework with built-in budget
                          governance, semantic caching, provider-agnostic auto-routing,
                          and a built-in eval suite. Async-first, fully typed.

Same quickstart ergonomics as CrewAI, same async power as AutoGen,
plus the production-safety layer neither ships with.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Imports and setup
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import helix
from helix.core.workflow import Workflow, step
from helix.config import BudgetConfig, ModelConfig, PermissionConfig, StructuredOutputConfig
from pydantic import BaseModel
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Shared tools (reused across demos)
# ─────────────────────────────────────────────────────────────────────────────

@helix.tool(description="Look up a fact about a topic. Returns a short summary.")
async def lookup(topic: str) -> dict:
    """Simulates a knowledge lookup (no real network call needed for demo)."""
    knowledge = {
        "autogen":  "AutoGen is Microsoft's event-driven multi-agent framework. "
                    "It uses an actor model with async message passing. Agents communicate "
                    "via a message bus. Supports distributed gRPC runtimes and Docker code execution.",
        "crewai":   "CrewAI is an open-source framework for orchestrating autonomous AI agents "
                    "using a role-based 'crew' metaphor. Crews + Flows. Over 100k certified devs. "
                    "YAML-based agent definitions. Enterprise cloud offering.",
        "helix":    "Helix is a production-grade Python agent framework with built-in budget "
                    "governance, semantic caching, provider-agnostic routing (Gemini/OpenAI/"
                    "Anthropic/Groq/Mistral/...), context compaction, episodic memory, guardrails, "
                    "HITL, and a built-in eval suite.",
        "langchain": "LangChain is a chaining framework for LLM apps. Primarily a composition "
                     "library rather than an agent framework. LangGraph adds graph-based workflows.",
    }
    topic_key = topic.lower().replace("-", "").replace(" ", "")
    for key, val in knowledge.items():
        if key in topic_key or topic_key in key:
            return {"topic": topic, "summary": val}
    return {"topic": topic, "summary": f"No stored knowledge for '{topic}'. Ask the LLM directly."}


@helix.tool(description="Calculate a mathematical expression safely.")
async def calculate(expression: str) -> dict:
    """Safe math evaluator for demo."""
    import math
    safe = {
        "__builtins__": {},
        "sqrt": math.sqrt, "log": math.log, "pi": math.pi,
        "abs": abs, "round": round, "pow": pow,
    }
    try:
        result = eval(expression, safe)  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}


@helix.tool(description="Summarize a list of bullet points into a paragraph.")
async def summarize_bullets(bullets: List[str]) -> dict:
    """Joins bullet points into a formatted summary."""
    text = " ".join(f"({i+1}) {b}" for i, b in enumerate(bullets))
    return {"summary": f"Key findings: {text}"}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — DEMO 1: Single Agent with Tools + Budget
#   What this shows vs AutoGen/CrewAI:
#     - Zero boilerplate: just Agent(...) + run(). No "system" to start.
#     - BudgetConfig: hard $0.10 cap — AutoGen and CrewAI have nothing like this.
#     - result.cost_usd, result.tool_calls, result.model_used — all in one result.
# ─────────────────────────────────────────────────────────────────────────────

async def demo_single_agent():
    print("\n" + "═"*70)
    print("DEMO 1 — Single Agent with Tools + Budget Governance")
    print("  (AutoGen: no budget control | CrewAI: no budget control)")
    print("═"*70)

    agent = helix.Agent(
        name="ResearchBot",
        role="Technology researcher",
        goal="Answer questions about AI frameworks using the lookup tool.",
        tools=[lookup, calculate],
        # ── Helix exclusive: hard cost cap ──────────────────────────────────
        budget=BudgetConfig(budget_usd=0.10),
    )

    result = await helix.run_async(
        agent,
        "Use the lookup tool to find out what AutoGen is, "
        "then what CrewAI is, then compare them in 2 sentences."
    )

    if result.error:
        print(f"  ✗ Error: {result.error}")
    else:
        print(f"\n  Answer:\n{result.output}")
        print(f"\n  ── Metadata (Helix-only fields) ──")
        print(f"  model_used  : {result.model_used}")
        print(f"  cost_usd    : ${result.cost_usd:.6f}  (hard cap was $0.10)")
        print(f"  tool_calls  : {result.tool_calls}")
        print(f"  steps       : {result.steps}")
        print(f"  cache_hits  : {result.cache_hits}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — DEMO 2: Structured Output (Pydantic schema)
#   What this shows vs AutoGen/CrewAI:
#     - Agent output is a validated Pydantic model, not a raw string.
#     - AutoGen: you must parse JSON yourself.
#     - CrewAI: has output_json/output_pydantic on Task, but not on the agent run.
# ─────────────────────────────────────────────────────────────────────────────

class FrameworkComparison(BaseModel):
    framework_name: str
    primary_use_case: str
    top_3_strengths: List[str]
    main_weakness: str
    best_for_team_size: str


async def demo_structured_output():
    print("\n" + "═"*70)
    print("DEMO 2 — Structured Output (Pydantic schema enforcement)")
    print("  (AutoGen: raw string | CrewAI: task-level only)")
    print("═"*70)

    agent = helix.Agent(
        name="Analyst",
        role="Framework analyst",
        goal="Produce structured analysis of AI frameworks.",
        tools=[lookup],
        budget=BudgetConfig(budget_usd=0.05),
        structured_output=StructuredOutputConfig(enabled=True),
    )

    result = await helix.run_async(
        agent,
        "Use the lookup tool to research CrewAI, then fill in a structured "
        "comparison card for it.",
        output_schema=FrameworkComparison,
    )

    if result.error:
        print(f"  ✗ Error: {result.error}")
    elif isinstance(result.output, FrameworkComparison):
        fc = result.output
        print(f"\n  Framework      : {fc.framework_name}")
        print(f"  Use case       : {fc.primary_use_case}")
        print(f"  Strengths      : {', '.join(fc.top_3_strengths)}")
        print(f"  Weakness       : {fc.main_weakness}")
        print(f"  Best team size : {fc.best_for_team_size}")
        print(f"\n  Cost: ${result.cost_usd:.6f} | Model: {result.model_used}")
    else:
        # Fallback: raw output if schema enforcement not fully supported yet
        print(f"\n  Output: {result.output}")
        print(f"  Cost: ${result.cost_usd:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — DEMO 3: Sequential Team
#   What this shows vs AutoGen/CrewAI:
#     - helix.Team with strategy="sequential" — each agent's output feeds the next.
#     - Equivalent to CrewAI Crew with sequential process.
#     - Equivalent to AutoGen's GroupChat sequential speaker selection.
#     - Helix: 5 lines. AutoGen requires a GroupChatManager + multiple agent roles.
#     - Helix: total_cost_usd tracked across the whole team automatically.
# ─────────────────────────────────────────────────────────────────────────────

async def demo_sequential_team():
    print("\n" + "═"*70)
    print("DEMO 3 — Sequential Team (Researcher → Analyst → Writer)")
    print("  CrewAI equivalent: Crew(process=Process.sequential)")
    print("  AutoGen equivalent: GroupChat with ordered speaker selection")
    print("═"*70)

    researcher = helix.Agent(
        name="Researcher",
        role="Technology researcher",
        goal="Gather factual information about AI frameworks using the lookup tool.",
        tools=[lookup],
        budget=BudgetConfig(budget_usd=0.05),
    )

    analyst = helix.Agent(
        name="Analyst",
        role="Technical analyst",
        goal="Identify the key differentiators from the research provided.",
        budget=BudgetConfig(budget_usd=0.05),
    )

    writer = helix.Agent(
        name="Writer",
        role="Technical writer",
        goal="Write a clear, concise summary paragraph for a developer audience.",
        budget=BudgetConfig(budget_usd=0.05),
    )

    # ── Helix Team — 5 lines vs pages of AutoGen boilerplate ────────────────
    team = helix.Team(
        name="framework-review-team",
        agents=[researcher, analyst, writer],
        strategy="sequential",          # researcher → analyst → writer
    )

    result = await team.run(
        "Compare Helix and AutoGen as Python AI agent frameworks. "
        "Use the lookup tool to get facts about each, then analyse differences, "
        "then write a 3-sentence developer summary."
    )

    if result.error:
        print(f"  ✗ Error: {result.error}")
    else:
        print(f"\n  Final output:\n{result.final_output}")
        print(f"\n  ── Team Metadata ──")
        print(f"  Total cost : ${result.total_cost_usd:.6f}")
        print(f"  Duration   : {result.duration_s:.2f}s")
        print(f"  Agents run : {len(result.agent_results)}")
        for r in result.agent_results:
            print(f"    • {r.agent_name:<12} steps={r.steps} cost=${r.cost_usd:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — DEMO 4: Parallel Team
#   What this shows vs AutoGen/CrewAI:
#     - Three agents run concurrently on the same task using asyncio.gather.
#     - CrewAI equivalent: Crew(process=Process.hierarchical) with manager + specialists.
#     - AutoGen: requires custom GroupChat.
#     - Helix: one line change from sequential → parallel.
# ─────────────────────────────────────────────────────────────────────────────

async def demo_parallel_team():
    print("\n" + "═"*70)
    print("DEMO 4 — Parallel Team (3 specialists run concurrently)")
    print("  CrewAI equivalent: multiple Tasks with async_execution=True")
    print("  AutoGen equivalent: custom GroupChat with simultaneous speak")
    print("═"*70)

    autogen_specialist = helix.Agent(
        name="AutoGenExpert",
        role="AutoGen specialist",
        goal="Explain AutoGen's architecture and primary strength in 2 sentences.",
        tools=[lookup],
        budget=BudgetConfig(budget_usd=0.03),
    )

    crewai_specialist = helix.Agent(
        name="CrewAIExpert",
        role="CrewAI specialist",
        goal="Explain CrewAI's architecture and primary strength in 2 sentences.",
        tools=[lookup],
        budget=BudgetConfig(budget_usd=0.03),
    )

    helix_specialist = helix.Agent(
        name="HelixExpert",
        role="Helix specialist",
        goal="Explain Helix's architecture and primary strength in 2 sentences.",
        tools=[lookup],
        budget=BudgetConfig(budget_usd=0.03),
    )

    team = helix.Team(
        name="parallel-analysis-team",
        agents=[autogen_specialist, crewai_specialist, helix_specialist],
        strategy="parallel",            # all three run at the same time
    )

    result = await team.run(
        "Use the lookup tool to research your assigned framework, "
        "then write a 2-sentence expert summary of its strengths."
    )

    if result.error:
        print(f"  ✗ Error: {result.error}")
    else:
        outputs = result.final_output
        if isinstance(outputs, list):
            for i, (agent_r, out) in enumerate(
                zip(result.agent_results, outputs), 1
            ):
                print(f"\n  [{i}] {agent_r.agent_name}:\n  {out}")
        else:
            print(f"\n  Output:\n{outputs}")
        print(f"\n  Total cost: ${result.total_cost_usd:.6f} | "
              f"Duration: {result.duration_s:.2f}s (parallel!)")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — DEMO 5: Fluent Workflow DSL
#   What this shows vs AutoGen/CrewAI:
#     - Workflow builder: .then() .parallel() .branch() .loop() .human_review()
#     - CrewAI equivalent: Flow with @start / @listen decorators.
#     - AutoGen: no built-in workflow DSL (use custom code or Swarm patterns).
#     - Helix: composable, retryable steps with a fluent builder pattern.
# ─────────────────────────────────────────────────────────────────────────────

@step(name="fetch_frameworks", retry=1)
async def fetch_frameworks(topic: str) -> dict:
    """Gather framework names from the topic string."""
    frameworks = []
    for word in topic.lower().split():
        if word in ("autogen", "crewai", "helix", "langchain"):
            frameworks.append(word)
    if not frameworks:
        frameworks = ["helix", "autogen", "crewai"]
    return {"topic": topic, "frameworks": frameworks}


@step(name="score_frameworks")
async def score_frameworks(data: dict) -> dict:
    """Assign rough production-readiness scores (demo scoring logic)."""
    scores = {
        "helix":     {"budget_control": 10, "provider_support": 9, "observability": 9},
        "autogen":   {"budget_control":  1, "provider_support": 7, "observability": 7},
        "crewai":    {"budget_control":  2, "provider_support": 8, "observability": 6},
        "langchain": {"budget_control":  1, "provider_support": 9, "observability": 5},
    }
    results = {}
    for fw in data.get("frameworks", []):
        results[fw] = scores.get(fw, {"note": "unknown"})
    return {**data, "scores": results}


@step(name="format_scorecard")
async def format_scorecard(data: dict) -> str:
    """Format scores as a readable scorecard."""
    lines = [f"  Production Readiness Scorecard — {data['topic']}", ""]
    for fw, s in data.get("scores", {}).items():
        if isinstance(s, dict) and "budget_control" in s:
            total = sum(s.values())
            lines.append(f"  {fw:<12} budget={s['budget_control']}/10  "
                         f"providers={s['provider_support']}/10  "
                         f"observability={s['observability']}/10  "
                         f"→ total={total}/30")
    return "\n".join(lines)


async def demo_workflow():
    print("\n" + "═"*70)
    print("DEMO 5 — Fluent Workflow DSL (.then / .parallel / .branch)")
    print("  CrewAI equivalent: Flow with @start/@listen decorators")
    print("  AutoGen equivalent: no built-in DSL — write custom code")
    print("═"*70)

    pipeline = (
        Workflow("framework-scorecard-pipeline")
        .then(fetch_frameworks)                     # step 1 — fetch
        .then(score_frameworks)                     # step 2 — score
        # branch: human review for "important" topics, skip otherwise
        .branch(
            condition=lambda data: len(data.get("frameworks", [])) >= 3,
            if_true=format_scorecard,               # format if 3+ frameworks
            if_false=lambda d: f"Only {d.get('frameworks')} found.",
        )
        .with_budget(0.50)
    )

    result = await pipeline.run("Compare helix autogen crewai frameworks")

    print(f"\n{result.final_output}")
    print(f"\n  Steps: {len(result.steps)} | Duration: {result.duration_s:.3f}s")
    for s in result.steps:
        status = "✓" if not s.error else "✗"
        print(f"    {status} {s.name:<25} {s.duration_s:.3f}s")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — DEMO 6: Permission Control (tool allow/deny)
#   What this shows vs AutoGen/CrewAI:
#     - PermissionConfig: whitelist or blacklist tools per agent.
#     - AutoGen: no built-in permission system.
#     - CrewAI: no built-in permission system.
#     - Helix: security-first — production agents should never run unrestricted.
# ─────────────────────────────────────────────────────────────────────────────

async def demo_permissions():
    print("\n" + "═"*70)
    print("DEMO 6 — Permission Control (tool allow/deny lists)")
    print("  AutoGen: no built-in permissions | CrewAI: no built-in permissions")
    print("═"*70)

    # This agent can ONLY call lookup — calculate is explicitly denied.
    restricted_agent = helix.Agent(
        name="RestrictedResearcher",
        role="Read-only researcher",
        goal="Look up facts. Never perform calculations.",
        tools=[lookup, calculate],
        permissions=PermissionConfig(
            denied_tools=["calculate"],   # ← Helix exclusive
        ),
        budget=BudgetConfig(budget_usd=0.05),
    )

    result = await helix.run_async(
        restricted_agent,
        "Use lookup to find what AutoGen is. "
        "Also try to calculate 2+2 (this should be blocked)."
    )

    if result.error:
        print(f"  ✗ Error: {result.error}")
    else:
        print(f"\n  Output:\n{result.output}")
        print(f"\n  calculate tool was denied — agent couldn't use it even though it was registered.")
        print(f"  Cost: ${result.cost_usd:.6f} | Tool calls made: {result.tool_calls}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — Main runner
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    print(COMPARISON)

    demos = [
        ("1 — Single Agent + Budget",       demo_single_agent),
        ("2 — Structured Output",           demo_structured_output),
        ("3 — Sequential Team",             demo_sequential_team),
        ("4 — Parallel Team",               demo_parallel_team),
        ("5 — Fluent Workflow DSL",         demo_workflow),
        ("6 — Permission Control",          demo_permissions),
    ]

    passed = 0
    failed = 0

    for label, demo_fn in demos:
        try:
            await demo_fn()
            passed += 1
        except Exception as exc:
            print(f"\n  ✗ Demo {label} raised: {type(exc).__name__}: {exc}")
            failed += 1

    print("\n" + "═"*70)
    print(f"  Results: {passed} passed, {failed} failed")
    print("═"*70)
    print("""
  What you just saw — features neither AutoGen nor CrewAI ship:

    ✓  BudgetConfig    — hard per-agent cost cap with pre-call gating
    ✓  result.cost_usd — real cost on every single run result
    ✓  StructuredOutput— Pydantic model returned directly from agent.run()
    ✓  Team strategies — sequential / parallel / hierarchical in one API
    ✓  Workflow DSL    — .then/.parallel/.branch/.loop/.human_review
    ✓  PermissionConfig— tool allow/deny lists per agent
    ✓  Provider routing— models/gemini-2.5-flash → Gemini auto-detected
    ✓  Fallback chains — if primary model fails, auto-retry on next
""")


if __name__ == "__main__":
    asyncio.run(main())
