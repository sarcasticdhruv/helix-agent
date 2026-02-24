# Contributing to Helix

Thank you for your interest in contributing! This guide covers everything you need to get started.

---

## Table of contents

- [Code of Conduct](#code-of-conduct)
- [How to contribute](#how-to-contribute)
- [Development setup](#development-setup)
- [Running tests](#running-tests)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Commit style](#commit-style)
- [Reporting bugs](#reporting-bugs)
- [Requesting features](#requesting-features)

---

## Code of Conduct

By participating in this project you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## How to contribute

- **Bug fixes** — open an issue first describing the bug, then submit a PR that references it.
- **New features** — open a Feature Request issue and discuss before implementing large changes.
- **Provider support** — add a new backend in `src/helix/models/providers/`, model it after `gemini.py`.
- **Documentation** — typos, clarity, examples — all welcome, no issue required.
- **Tests** — improve coverage for any module in `tests/`.

---

## Development setup

```bash
# 1. Fork + clone
git clone https://github.com/YOUR_USERNAME/helix-agent.git
cd helix-agent

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\Activate.ps1       # Windows PowerShell

# 3. Install in editable mode with all dev + provider deps
pip install -e ".[dev,gemini,openai,anthropic]"

# 4. Set up at least one provider key
helix config set GOOGLE_API_KEY "AIza..."
```

---

## Running tests

```bash
# All tests (fast, no real LLM calls needed for most)
pytest tests/ -q

# With coverage
pytest tests/ --cov=helix --cov-report=term-missing

# Lint
ruff check src/
ruff format src/ --check
```

All tests must pass and `ruff check` must produce no errors before a PR is merged.

---

## Submitting a Pull Request

1. Create a branch from `main`: `git checkout -b feat/my-feature`
2. Make your changes + add/update tests.
3. Run `pytest` and `ruff check src/` — both must be clean.
4. Update `CHANGELOG.md` under `[Unreleased]` describing your change.
5. Push and open a PR against `main`.
6. Fill in the PR template — describe *what* and *why*.

PRs that change public API must update the corresponding section in `README.md`.

---

## Commit style

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Fireworks AI provider
fix: correct model_used blank in AgentResult
docs: add structured-output example
chore: bump google-generativeai to 0.9
refactor: extract _detect_provider into router module
test: add coverage for BudgetStrategy.DEGRADE path
```

---

## Reporting bugs

Use the [Bug Report](.github/ISSUE_TEMPLATE/bug_report.yml) template.

Include:
- Helix version (`python -c "import helix; print(helix.__version__)"`)
- Python version + OS
- Minimal reproducible example
- Full traceback

---

## Requesting features

Use the [Feature Request](.github/ISSUE_TEMPLATE/feature_request.yml) template.

Describe the use case first — *why* you need it matters more than *how* it should work.
