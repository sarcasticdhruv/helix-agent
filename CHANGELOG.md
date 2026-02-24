# Changelog

All notable changes to **helix-framework** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
