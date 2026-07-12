# Changelog

All notable changes to **helix-framework** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.5.4] — 2026-07-12

### Added
- **CITATION.cff**: Machine-readable citation metadata for academic/research reference
- **FAQ section in README**: Direct Q&A on licensing, MCP support, framework interop, local models, and budget enforcement — formatted for LLM answer extraction
- **llms-full.txt**: Full README + USAGE.md concatenated into one file for RAG ingestion and deep LLM Q&A
- **pepy.tech download badge**: Added to README badge row

### Improved
- **GEO**: llms.txt now links to llms-full.txt; FAQ content is structured for direct LLM quoting

## [0.5.3] — 2026-07-12

### Fixed
- **PyPI publish**: v0.5.2 publish was blocked by pre-existing wheel. Bumped to v0.5.3 to release with corrected URLs and SEO improvements.

## [0.5.2] — 2026-07-12

### Added
- **PyPI project metadata**: Fixed broken repository/homepage URLs (replace `your-org` placeholder with actual GitHub org)
- **Framework comparison table**: Added to README showing Helix vs CrewAI, AutoGen, LangGraph — highlights cost governance, semantic caching, and multi-tier memory as unique strengths
- **llms.txt discovery file**: Added for AI crawler discovery (Perplexity, Claude, etc.) — documents key features, providers, use cases, and framework integrations

### Improved
- **SEO/GEO**: PyPI links now correct; README comparison table improves LLM-based recommendations; llms.txt follows emerging AI crawler convention

## [0.5.1] — 2026-07-10

### Fixed
- README: the "Contributing" section's `git clone` example still pointed at
  the `YOUR_USERNAME` placeholder instead of this repo.

## [0.5.0] — 2026-07-10

A correctness-and-honesty pass, driven by an independent code audit plus live
end-to-end testing against a real provider. The headline: the tool-calling
loop — the single most important path in the framework — was broken against
every real provider, and several advertised "AI-native" subsystems were wired
up in name but never actually invoked. Every item below was verified either
by a targeted unit test or live against the Groq API (noted per-item).

### Fixed — the tool-calling loop (previously broken end to end)
- **Every agent silently inherited the entire global tool registry**, even
  when constructed with `tools=[]` or no tools at all — including
  `execute_python`, `write_file`, and `get_env`, registered globally by a bare
  `import helix`. Tool inheritance is now opt-in via `Agent(..., inherit_global_tools=True)`;
  the default is no inherited tools. **This is a breaking change**: agents
  that relied on implicit global tool access must now pass `tools=[...]`
  explicitly or set `inherit_global_tools=True`.
- **The assistant's tool-call turn was never persisted** back into
  conversation history — only its text was — so the follow-up request to the
  provider was missing the very message the tool result was supposed to
  respond to.
- **Tool-result messages were missing `tool_call_id`** (and carried an
  unrecognized `tool_name` field some providers reject outright), so
  OpenAI/Groq/OpenAI-compatible endpoints 400'd on the second turn of any
  tool round-trip, and Anthropic received a fabricated `tool_use_id="unknown"`.
- **Gemini silently dropped the `tools` parameter** — a tool-equipped agent on
  Gemini could never actually call a tool, with no error raised. Gemini now
  forwards tool schemas and parses `function_call`/`function_response` parts
  correctly for multi-turn tool conversations.
- **Provider-exhaustion failures looked like successful output.**
  `AllModelsExhaustedError` was converted to a plain `"[ERROR] ..."` string in
  `AgentResult.output` with no other signal — `Team`, `AgentPipeline`, and
  `GroupChat` would cascade that string into the next agent's input as if it
  were real work. Added `AgentResult.success`; `Team`/`AgentPipeline`/`GroupChat`
  now stop and surface the failure instead of propagating it.
- **`GroupChat`'s round-robin speaker selection skipped the first agent**
  (off-by-one); `ConversableAgent.reply()` hardcoded `cost=0.0` regardless of
  actual spend; the `auto` speaker-selection coordinator's own LLM call was
  never counted toward total cost. All three fixed.

### Fixed — making the existing feature claims true
- **The context engine's "multi-factor relevance decay" was dead code** —
  `update_relevance()` had zero callers anywhere, so every message's
  relevance stayed at its default and compaction could never find anything
  eligible to compress. Now called every reasoning step, with a real
  semantic-similarity term computed from the configured embedder.
- **Context compaction was a guaranteed no-op** in the live agent path — the
  call site never passed an embedder or LLM router, so it always fell back to
  naive truncation. Now passes both; also fixed a second silent no-op where a
  zero-vector embedder (no embedding key configured) caused clustering to
  degenerate into one message per cluster — it now falls back to a single
  real LLM-summarized group instead.
- **The observability tracer recorded zero spans.** `span()`, `log_step()`,
  and `log_llm_call()` were fully implemented but never called — every
  exported trace had `spans: []`, so replay and ghost-debug had nothing to
  work with. The tracer is now instantiated fresh per run (it was previously
  shared across an agent's entire lifetime, leaking spans between runs and
  never getting the correct `run_id`) and fed real LLM-call and tool-call
  spans with timing, tokens, and cost.
- **The default memory backend never persisted anything** — `InMemoryBackend`
  is plain Python dicts, wiped on every restart, and the `qdrant`/`pinecone`/
  `chroma` backends referenced in code didn't exist as files (a
  `ModuleNotFoundError`, not the documented behavior). Added `SQLiteBackend`
  (`backend="sqlite"`, stdlib-only, zero new dependency) — verified to
  survive a simulated process restart. The three unimplemented backends now
  raise a clear `NotImplementedError` instead of a confusing import crash.
- **The eval suite's tool-selection and trajectory scorers always scored
  against a hardcoded empty list** — `AgentResult` never exposed the
  underlying `ToolCallRecord`s. Added `AgentResult.tool_call_records`; both
  scorers now score real tool-call data (verified live: a case with
  `expected_tools=["add"]` now correctly scores `1.0` instead of always `0.0`
  whenever tools were expected).
- **The pre-call budget gate estimated cost with a single flat rate**
  (`$0.005/1k` tokens) regardless of model, ignoring the ~800x price spread
  between the cheapest and most expensive supported models. Now uses the same
  per-model pricing table as the real post-call cost calculation.
- **The plan cache's "savings" were a hardcoded guess** (`avg_cost_usd * 0.5`)
  — and the retrieved plan was never actually used, so a cache "hit" changed
  nothing about agent behavior. The cached plan is now injected as a real
  hint into the reasoning loop, and savings are computed from the actual
  measured cost delta of the run that used it.
- **`AgentConfig.guardrails` was completely unreachable** from the public
  `Agent(...)` constructor — the field existed, but `Agent.__init__`'s
  `**extra_config` catch-all was never forwarded to `AgentConfig`, so
  anything passed that way (including `guardrails=[...]`, `hitl=...`,
  `tenant_id=...`) was silently dropped. Now forwarded. Also fixed a latent
  crash this exposed: `KeywordBlockGuard` and `SchemaGuard` required a
  positional constructor argument, so naming them by string (the only
  supported configuration path) would have raised `TypeError` the moment
  anyone used it — both now have safe defaults.
- **Corrected the README's eval-suite scorer count** (was described as "5
  scorers" naming one, "output format", that doesn't exist; it's actually 6:
  factual accuracy, tool selection, trajectory adherence, cost efficiency,
  step efficiency, and output quality).

### Added
- **`Agent.__init__(handoffs=[...])`** — a first-class handoff primitive.
  Each target agent gets a `transfer_to_<name>` tool the model can call
  directly (matching the OpenAI Agents SDK pattern); on a handoff, control
  and the conversation genuinely transfer to the target agent, whose
  `AgentResult` is returned with `handoff_chain` recording the path and cost
  from every agent in the chain summed. Verified live: a triage agent handed
  a billing question to a `BillingAgent`, and the final response was
  genuinely produced by the target, not the triage agent.
- **`helix.tools.mcp.MCPToolSource`** — MCP (Model Context Protocol) client
  support (`pip install "helix-framework[mcp]"`). Connects to any MCP server
  over stdio and exposes its tools as ordinary `RegisteredTool`s usable
  directly in `Agent(tools=[...])`. Verified live end-to-end against a real
  MCP server subprocess: tool discovery, schema translation, and tool
  execution through the actual wire protocol, then through a full agent run
  against a live LLM.
- **`safety.guardrails.PromptInjectionGuard`** — heuristic jailbreak/
  prompt-injection detector (instruction override, persona hijack,
  restriction-bypass requests, "developer mode", system-prompt exfiltration
  attempts). Pattern-based, not a trained classifier — a real first line of
  defense where previously there was none. Guardrails were also only ever
  checked on model *output*; the incoming task is now checked too, before it
  ever reaches context, memory, or the LLM.
- **Native structured-output mode.** `StructuredOutputConfig.use_native` was
  declared but never read; the JSON-parse-and-retry path is now backed by
  `response_format={"type": "json_object"}` on retry (OpenAI/Azure/
  OpenAI-compatible providers honor it; others accept and ignore it, no
  regression). Also fixed the correction prompt itself, which was
  interpolating a Python class repr (`"<class '...'>"`) instead of the
  model's real JSON schema.
- **`AgentResult.tool_call_records`, `.handoff_chain`, `.success`** — new
  fields/property for programmatic access to full tool-call records, the
  agent handoff path, and a clean success check that doesn't require string-
  matching `"[ERROR]"` in `.output`.

### Changed
- **Breaking:** agents no longer inherit the global tool registry by
  default (see above). Pass `tools=[...]` explicitly, or set
  `inherit_global_tools=True` to restore the old behavior.
- **Breaking:** `ConversableAgent.reply()` now returns `(content, cost_usd)`
  instead of just `content` — the old signature made it structurally
  impossible to report real cost.
- 34 new regression tests (`tests/test_v05_fixes.py`), each tied to a
  specific fix above.

---

## [0.3.4] — 2026-02-26

### Added
- **`StateGraph` / `CompiledGraph`** (`helix.core.graph`) — full LangGraph-compatible graph
  execution engine. `StateGraph` supports `.add_node()`, `.add_edge()`,
  `.add_conditional_edges()`, `.set_entry_point()`, `.set_finish_point()`, and `.compile()`.
  `CompiledGraph` exposes `.run()`, `.run_sync()`, `.invoke()`, `.ainvoke()`, and `.stream()`
  (async generator). Supports cycles, conditional branching, checkpoint persistence, and a
  configurable `max_steps` guard. `END` and `START` sentinels exported from top-level `helix`.
- **`execute_python` builtin tool** — sandboxed Python code execution in an isolated
  subprocess. Blocks `subprocess`, `pty`, `ctypes`, `multiprocessing`, and `os.system`.
  Returns `{"success", "stdout", "stderr", "returncode"}` with configurable timeout
  (default 15 s). Included in `discover_tools()` output as the 13th builtin.
- **`Agent.invoke()` / `Agent.ainvoke()`** — LangChain-compatible aliases for
  `Agent.run_sync()` and `Agent.run()` respectively.
- **`AgentPipeline`** (`helix.core.pipeline`) — sequential multi-agent pipe created with the
  `|` operator (`agent_a | agent_b | agent_c`). Each agent's output feeds the next as input.
  Exposes `.run()` (async) and `.run_sync()`.
- **`@helix.agent` class decorator** (`helix.core.agent_decorator`) — decorate any class with
  `@helix.agent(model=..., budget_usd=..., mode=...)` to automatically construct an `Agent`
  from its method-level `@helix.tool` members.
- **`helix.presets`** — 9 ready-made agent factory functions: `web_researcher()`, `writer()`,
  `coder(language)`, `code_reviewer(language)`, `data_analyst()`, `api_agent()`,
  `assistant()`, `summariser()`, and `fact_checker()`. `researcher` alias also exported.
- **`HookEvent` / `HookFn`** (`helix.core.hooks`) — lightweight event hook system.
  `Agent` accepts `on_event: HookFn | None` for per-call telemetry callbacks. Hook errors
  are silenced so they never interrupt the agent loop.
- **97 new pytest tests** (`tests/test_new_features.py`) — full coverage of
  `StateGraph`, `AgentPipeline`, `HookEvent`, `@helix.agent` decorator, `quick()`,
  `discover_tools`, presets, `execute_python`, `invoke`/`ainvoke`, `EvalSuite.case()`,
  workflow `ChainNode`, and public API surface.

### Fixed
- **`EvalCase.name` UUID default** — `EvalCase.name` previously defaulted to a random UUID
  fragment, making the `name or fn.__name__` fallback in `EvalSuite.case()` unreachable.
  Default is now `""` so the decorator correctly uses the function name when no name is given.
- **`contextlib.suppress` cleanup** — replaced all bare `try/except: pass` blocks in
  `__init__.py`, `workflow.py`, and `core/graph.py` with `contextlib.suppress(Exception)`
  (ruff SIM105).

### Changed
- **Ruff coverage extended to `tests/`** — CI lint and format steps now cover `src/ tests/`
  (previously `src/` only).
- **`Union[X, Y]` → `X | Y`** syntax modernisation in `workflow.py` (ruff UP007).
- **`import helix.tools`** import-shadowing bug fixed in `__init__.py` — was silently
  shadowing the `helix.tools` subpackage; corrected to import-with-alias pattern.

---

## [0.3.3] — 2026-02-25

### Changed
- **Version bump only** — `0.3.2` wheel was already immutably stored on pypi
  pypi forbids file reuse so a new version is required.
  No functional changes from `0.3.2`.

---

## [0.3.2] — 2026-02-24

### Added
- **`Task` first-class object** — declarative unit of work assigned to an Agent,
  with `expected_output`, `output_file`, `output_schema`, `callback`, `context`
  (dependency chaining), `async_execution`, and `guardrails` chain with auto-retry.
  Inspired by CrewAI Tasks; extended with Pydantic output validation and
  both callable and LLM-string guardrails.
- **`Pipeline`** — runs an ordered list of Tasks in sequence (or concurrently
  for `async_execution=True` tasks), passing outputs forward as context.
  `pipeline.kickoff(inputs={...})` mirrors the CrewAI API exactly.
- **`TaskOutput`** — structured result from a Task run, with `.raw`, `.pydantic`,
  `.json_dict`, `.summary`, and `.to_dict()` accessors.
- **`ConversableAgent`** — AutoGen-style agent capable of multi-turn conversation,
  with `human_input=True` for terminal-based HITL and `max_consecutive_replies`
  to prevent one agent dominating a group chat.
- **`HumanAgent`** — a `ConversableAgent` that always prompts the human terminal.
- **`GroupChat`** — N agents in a shared multi-turn conversation. Speaker selection
  strategies: `round_robin`, `auto` (LLM picks), `random`, or any callable.
  Termination by `max_rounds`, `termination_keyword`, or custom `termination_fn`.
- **`GroupChatResult`** — full message history, transcript, cost, and termination reason.
- **`backstory` field on `Agent`** — rich character/background context injected into
  the system prompt, matching the CrewAI `backstory` parameter.
- **YAML config loader** (`helix.core.yaml_config`) — load agents and tasks from
  `agents.yaml` / `tasks.yaml` with `{variable}` template substitution.
  `helix.from_yaml("agents.yaml", "tasks.yaml", inputs={...})` returns a
  ready-to-run `Pipeline`.
- **`helix.from_yaml()`** — top-level convenience function for YAML-driven pipelines.

### Changed
- `Agent.__init__` now accepts `backstory` parameter (default `""`).
- `__all__` updated with all new public classes.

---


### Changed
- **PyPI package renamed to `helix-framework`** — install with `pip install helix-framework`;
  `import helix` is unchanged.
- **Optimized provider error handling** — all providers now use `raise ... from err` for
  cleaner tracebacks and better debuggability.
- **Enum modernization** — all enums migrated to `StrEnum`, removing redundant `(str, Enum)`
  double-inheritance.
- **Reduced boilerplate** — `try/except/pass` blocks replaced with `contextlib.suppress()`
  across caching, eval, memory, and runtime modules.
- **Dict literals** — `dict()` constructor calls replaced with `{}` literals in all
  provider files for consistency and minor performance gain.

---

## [0.3.0] — 2026-02-23

### Added
- First public release
- **Gemini 2.5 Flash/Pro support** — `models/gemini-2.5-flash` and `models/gemini-2.5-pro`
  added to pricing table, fallback chains, and `config_store` provider priority list.
- **`models/` prefix routing** — `_detect_provider()` in `router.py` now handles the
  full `models/gemini-*` path format that the Google SDK returns, preventing incorrect
  routing to the Together AI provider.
- **`helix config set` documented** in README as the recommended provider setup method.
- **`examples/helix_vs_autogen_crewai.py`** — competitive showcase demonstrating 6 live
  demos covering budget enforcement, structured output, sequential teams, parallel teams,
  workflow DSL, and permission enforcement.
- **Python 3.13 support** — CI matrix and pyproject classifiers updated.
- **Extras `azure`, `openrouter`, `deepseek`** added to `pyproject.toml`
  (all three reuse the `openai` package).
- **`[tool.ruff.lint]`**, **`[tool.coverage.run/report]`**, and pytest
  `filterwarnings` config added to `pyproject.toml`.
- **Complete provider env vars table** added to README (12 providers with free-tier
  indicators).
- **`CHANGELOG.md`** (this file).

### Fixed
- **`model_used` always blank in `AgentResult`** — `_initialize_subsystems()` in
  `agent.py` now writes the resolved model name back to `self._config.model.primary`
  so that `ExecutionContext.effective_model()` returns the real model instead of `""`.
- **`helix.run()` dangling coroutine** — the sync wrapper in `__init__.py` previously
  created a coroutine upfront (without `output_schema`) then created a second coroutine
  in the thread-pool branch, leaving the first one unawaited. Both are now created at
  point-of-use with full parameters.
- **`output_schema` silently dropped** in `helix.run()` thread-pool branch — fixed by
  the same change above.
- **Gemini 1.5-flash 404 errors** — `gemini.py` defaults updated from the deprecated
  `gemini-1.5-flash` to `gemini-2.0-flash`; `supported_models()` refreshed to current
  Gemini lineup.
- **`asyncio.run()` in running event loop** — showcase demos now use
  `await helix.run_async()` / `await team.run()` / `await pipeline.run()` instead of
  their sync wrappers when called from inside `asyncio.run(main())`.

### Changed
- **`__version__`** now resolved dynamically via `importlib.metadata.version("helix-framework")`
  with a source-install fallback, replacing the hardcoded string.
- **`requirements.txt`** reordered — `google-generativeai` is now the first (un-commented)
  provider; `openai` and `anthropic` moved to commented section.
- **`pyproject.toml`** version bumped `0.2.1 → 0.3.0`; classifiers expanded; GitHub URLs
  updated; `google-generativeai` minimum bumped to `>=0.8`.

---

## [0.2.1] — 2026-02-12

### Fixed
- Minor packaging fixes; `entry_points` corrected for the `helix` CLI command.
- Pydantic v2 model validators updated after `@validator` → `@field_validator` migration.

---

## [0.2.0] — 2026-02-12

### Added
- **3-tier semantic cache** (semantic similarity → plan/APC → prefix matching).
- **`EvalSuite`** with 5 built-in scorers and regression gate.
- **`GhostDebugResolver`** for silent-failure diagnosis.
- **Failure replay** from stored traces.
- **HITL gate** — pause execution and request human approval before high-risk actions.
- **Framework adapters** — `helix.wrap_llm()`, `helix.from_crewai()`, `helix.from_autogen()`.

### Changed
- `Team.run_sync()` and `Workflow.run_sync()` introduced alongside `await`-based `run()`.
- `BudgetStrategy.DEGRADE` added (switch to cheaper model instead of hard-stop).

---

## [0.1.0] — 2026-02-03

### Added
- `Agent`, `Team`, `Workflow`, `Session` core primitives.
- `MemoryConfig` / short-term buffer + WAL-backed long-term store.
- `CacheConfig` / semantic cache (tier 1).
- `BudgetConfig` / cost governor with hard-stop and warn threshold.
- `PermissionConfig` / allowed + denied tool lists.
- Providers: OpenAI, Anthropic, Gemini, Groq, Mistral, Cohere, Together, Ollama,
  OpenAI-compatible (Azure, OpenRouter, DeepSeek, xAI, Perplexity, Fireworks).
- `helix doctor`, `helix models`, `helix cost`, `helix trace`, `helix replay` CLI commands.
