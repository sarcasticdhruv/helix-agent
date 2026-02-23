# Changelog

All notable changes to **helix-agent** are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **`__version__`** now resolved dynamically via `importlib.metadata.version("helix-agent")`
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
