# Helix SEO/GEO + GitHub Growth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Helix (`helix-framework`) discoverable and compelling — to search engines, to LLMs answering "what agent framework should I use", and to a human landing on the GitHub repo cold — without touching the `helix` package's runtime code.

**Architecture:** Twelve independent, sequential tasks grouped into five phases (research → visual identity → GitHub metadata → GEO content → SEO docs site → external growth drafts). Each task produces one committable, independently-verifiable deliverable: a file, an image, or a doc. No task depends on unmerged state from a later task.

**Tech Stack:** Python 3.11 (repo's floor), Pillow + pyfiglet (banner/social-preview image generation, both pure-Python, no system deps), VHS (Homebrew, terminal-recording), mkdocs + mkdocs-material (docs site), GitHub Actions (Pages deploy).

## Global Constraints

- No code changes to `src/helix/**` — this plan only touches README.md, llms.txt/llms-full.txt, new `assets/`, `scripts/growth/`, `docs/growth/`, `docs/site/`, `mkdocs.yml`, and one new GitHub Actions workflow.
- No fabricated metrics, badges, or trophies (e.g. no "#1 Trending" claim) — every badge/claim must be independently verifiable (shields.io pulling real PyPI/test data, or plainly labeled as a manual step).
- Discord/community server setup is explicitly out of scope (deferred by the user) — do not add a Discord link or badge anywhere.
- Repo is `sarcasticdhruv/helix-agent` on GitHub, package is `helix-framework` on PyPI, import name is `helix`. Use these exact names everywhere (no placeholder org/username).
- Python floor for any new script that imports `helix` is **3.11** (repo requires `>=3.11` per `pyproject.toml`; this environment's default `python3` is 3.9, so use `python3.11` explicitly in every command).
- GitHub repo Settings changes (About description, topics, social preview upload) cannot be made via API/CLI in this environment (no `gh` CLI available) — Task 6 produces exact copy + image and a manual checklist, not a live change.

---

### Task 1: Research findings note

**Files:**
- Create: `docs/growth/research-findings.md`

**Interfaces:**
- Produces: a markdown file with three required `##` sections — `## Keywords`, `## Common Complaints`, `## Structural Patterns` — consumed by Tasks 3, 8, and 12 as the source of framing/keywords/copy.

- [ ] **Step 1: Run the research skill**

Invoke the `last30days` skill with this query: *"AI agent frameworks last 30 days — CrewAI, AutoGen, LangGraph complaints and pain points, and reception of developer tools with ASCII-banner READMEs like Headroom"*. Capture whatever concrete keywords, complaints, and phrasings it returns (e.g. "CrewAI has no cost control", "AutoGen setup is too complex", specific search phrasings people use).

- [ ] **Step 2: Write the findings note**

Create `docs/growth/research-findings.md`:

```markdown
# Research Findings — SEO/GEO/Growth

Source: last30days skill run, YYYY-MM-DD. Feeds README hero copy (Task 3), FAQ
expansion (Task 8), and external growth drafts (Task 12).

## Keywords

<!-- 8-12 phrases people actually search/ask, pulled from the last30days run,
     e.g. "AI agent framework python", "CrewAI alternative", "cost control
     LLM agents". One per line. -->

## Common Complaints

<!-- 5-8 specific, sourced complaints about CrewAI/AutoGen/LangGraph pulled
     from the run, each one sentence, e.g. "No built-in way to cap spend
     per run — found out the hard way after a $400 AutoGen bill." -->

## Structural Patterns

Borrowed from Headroom's README (github.com/... — the "Headroom" context
compression layer for AI agents), used as a structural reference, not copied
verbatim:
- Big block-letter ASCII banner image directly under the H1, no other text above it.
- One-line bold tagline immediately under the banner.
- Badge row (PyPI/npm version, CI status, license, docs) directly under the tagline.
- A single link row right after badges: `Docs · Install · Examples · llms.txt · Contributing`.
- A terminal-recording GIF embedded within the first screen of scroll, showing one
  concrete before/after moment.
```

Fill in the `<!-- -->` placeholders with real content from the Step 1 run before committing — do not leave HTML comments in the committed file.

- [ ] **Step 3: Verify required sections exist**

Run: `grep -c "^## " docs/growth/research-findings.md`
Expected: `3` (Keywords, Common Complaints, Structural Patterns)

Run: `grep -c "<!--" docs/growth/research-findings.md`
Expected: `0` (no leftover placeholder comments)

- [ ] **Step 4: Commit**

```bash
git add docs/growth/research-findings.md
git commit -m "docs(growth): add research findings for SEO/GEO copy"
```

---

### Task 2: ASCII banner image generator

**Files:**
- Create: `scripts/growth/generate_banner.py`
- Create: `assets/banner.png` (generated, not hand-written)

**Interfaces:**
- Produces: `render_banner(text: str, tagline: str, out_path: str, font_size: int = 22, canvas_size: tuple[int, int] | None = None) -> None` — reused by Task 6 (social preview image) with a larger `font_size` and fixed `canvas_size=(1280, 640)`.

- [ ] **Step 1: Install dependencies**

```bash
python3.11 -m pip install pyfiglet
python3.11 -c "import PIL; print(PIL.__version__)"
```
Expected: pyfiglet installs cleanly; PIL prints a version (Pillow is already present in this environment — if the second command fails elsewhere, run `python3.11 -m pip install pillow` first).

- [ ] **Step 2: Write the generator script**

Create `scripts/growth/generate_banner.py`:

```python
"""Generate the HELIX ASCII-art banner and social preview image.

Usage:
    python3.11 scripts/growth/generate_banner.py
"""
import pyfiglet
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/System/Library/Fonts/Menlo.ttc"
BG_COLOR = (13, 17, 23)      # GitHub dark background
FG_COLOR = (230, 237, 243)   # near-white


def render_banner(
    text: str,
    tagline: str,
    out_path: str,
    font_size: int = 22,
    canvas_size: tuple[int, int] | None = None,
) -> None:
    art_lines = pyfiglet.figlet_format(text, font="block").rstrip("\n").split("\n")
    font = ImageFont.truetype(FONT_PATH, font_size)
    tagline_font = ImageFont.truetype(FONT_PATH, font_size // 2)

    line_width = max(font.getbbox(line)[2] for line in art_lines)
    line_height = font.getbbox("Xg")[3] + 6
    padding = font_size * 2

    content_width = line_width + padding * 2
    content_height = line_height * len(art_lines) + padding * 3 + font_size

    width, height = canvas_size if canvas_size else (content_width, content_height)
    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)

    x_offset = (width - line_width) // 2
    y = (height - content_height) // 2 + padding if canvas_size else padding
    for line in art_lines:
        draw.text((x_offset, y), line, font=font, fill=FG_COLOR)
        y += line_height

    tagline_bbox = draw.textbbox((0, 0), tagline, font=tagline_font)
    tagline_width = tagline_bbox[2] - tagline_bbox[0]
    draw.text(((width - tagline_width) // 2, y + padding // 2), tagline, font=tagline_font, fill=FG_COLOR)

    img.save(out_path)


if __name__ == "__main__":
    render_banner(
        "HELIX",
        "Production AI agents: budget limits, semantic caching, multi-agent teams",
        "assets/banner.png",
    )
```

- [ ] **Step 3: Run it and verify the output**

```bash
mkdir -p assets
python3.11 scripts/growth/generate_banner.py
python3.11 -c "from PIL import Image; im = Image.open('assets/banner.png'); print(im.size, im.mode)"
```
Expected: prints a size like `(730, 398) RGB` and `assets/banner.png` exists. Open the file with the Read tool to confirm the text reads "HELIX" clearly and the tagline is centered and legible.

- [ ] **Step 4: Commit**

```bash
git add scripts/growth/generate_banner.py assets/banner.png
git commit -m "feat(growth): generate ASCII-art HELIX banner image"
```

---

### Task 3: README hero section rewrite

**Files:**
- Modify: `README.md:1-9` (current H1 through badge row)

**Interfaces:**
- Consumes: `assets/banner.png` (Task 2), `## Structural Patterns` from `docs/growth/research-findings.md` (Task 1).
- Produces: new README hero block — consumed visually by Task 4 (demo GIF gets embedded directly below this block).

- [ ] **Step 1: Replace the hero block**

In `README.md`, replace lines 1-9 (from `# Helix` through the `Tests` badge line) with:

```markdown
<p align="center">
  <img src="assets/banner.png" alt="Helix — production AI agent framework" width="600">
</p>

<p align="center"><b>Production AI agents: hard budget limits, semantic caching, multi-agent teams, MCP tools.</b></p>

<p align="center">
[![PyPI](https://img.shields.io/pypi/v/helix-framework)](https://pypi.org/project/helix-framework/)
[![Downloads](https://static.pepy.tech/badge/helix-framework)](https://pepy.tech/project/helix-framework)
[![Python](https://img.shields.io/pypi/pyversions/helix-framework)](https://pypi.org/project/helix-framework/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/sarcasticdhruv/helix-agent/actions)
</p>

<p align="center">
<a href="#installation">Install</a> ·
<a href="#quickstart">Quickstart</a> ·
<a href="USAGE.md">Docs</a> ·
<a href="examples/">Examples</a> ·
<a href="llms.txt">llms.txt</a> ·
<a href="CONTRIBUTING.md">Contributing</a>
</p>
```

Keep everything from the old line 10 onward (`Helix gives you agents that actually behave in production: ...`) unchanged directly below this block.

- [ ] **Step 2: Verify the banner reference resolves**

```bash
grep -n "assets/banner.png" README.md
test -f assets/banner.png && echo "banner exists"
```
Expected: one match in README.md, and `banner exists` printed.

- [ ] **Step 3: Verify markdown renders without broken links**

```bash
grep -oE '\]\(#[a-z0-9-]+\)' README.md | sort -u | head -20
```
Expected: each anchor (e.g. `#installation`, `#quickstart`) corresponds to an existing `##` heading (spot-check 2-3 against `grep -n "^## " README.md`).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "feat(growth): rework README hero with banner, tagline, link row"
```

---

### Task 4: Terminal demo GIF (budget enforcement)

**Files:**
- Create: `scripts/growth/demo_budget_stop.py`
- Create: `assets/demo.tape`
- Create: `assets/demo.gif` (generated, not hand-written)
- Modify: `README.md` (insert GIF embed directly below the hero block added in Task 3)

**Interfaces:**
- Consumes: nothing from other tasks (self-contained; uses a fake in-process `LLMProvider`, no real API key, no network).
- Produces: `assets/demo.gif`, embedded in README — no later task depends on its internal code.

This task's script deliberately reaches into `Agent` internals (`_llm_router._providers`) to inject a fake, zero-cost provider so the recording is fully deterministic and requires no API key. This is a demo-recording utility, not a public usage example — keep it out of `examples/`.

- [ ] **Step 1: Install VHS and its dependencies**

```bash
brew install vhs
vhs --version
```
Expected: prints a version like `vhs version 0.11.0`.

- [ ] **Step 2: Install helix in editable mode with Python 3.11**

```bash
python3.11 -m pip install -e .
```
Expected: completes without error (only `pydantic>=2.0` is required for this demo — no provider extras needed since the fake provider is used).

- [ ] **Step 3: Write the demo script**

Create `scripts/growth/demo_budget_stop.py`:

```python
"""Deterministic, offline demo: Helix stopping a runaway agent at its budget cap.

No API key or network access required — a fake in-process LLMProvider stands
in for a real one. Run with:

    PYTHONPATH=src python3.11 scripts/growth/demo_budget_stop.py
"""
import asyncio

from helix.core.agent import Agent
from helix.core.tool import tool
from helix.config import ModelConfig, BudgetConfig, ModelResponse, TokenUsage, ToolCallRecord
from helix.interfaces import LLMProvider


@tool(description="Keep researching.")
async def keep_going(note: str) -> str:
    return f"noted: {note}"


class FakeProvider(LLMProvider):
    """Always asks for another tool call, at a fixed simulated cost per step."""

    async def complete(self, messages, model, tools=None, temperature=0.7, max_tokens=4096, response_format=None):
        return ModelResponse(
            content="Still working...",
            tool_calls=[ToolCallRecord(tool_name="keep_going", arguments={"note": "x"})],
            usage=TokenUsage(prompt_tokens=1000, completion_tokens=1000),
            model=model,
            provider="fake",
            finish_reason="tool_calls",
        )

    async def stream(self, messages, model, **kw):
        yield "fake"

    def count_tokens(self, messages, model):
        return 100

    def supported_models(self):
        return ["gpt-4o"]

    async def health(self):
        return True


async def main():
    agent = Agent(
        name="Demo",
        role="tester",
        goal="loop until budget stops me",
        model=ModelConfig(primary="gpt-4o", auto_route=False, max_tokens=1000),
        budget=BudgetConfig(budget_usd=0.03),
        tools=[keep_going],
    )
    await agent._ensure_initialized()
    agent._llm_router._providers["openai:gpt-4o"] = FakeProvider()

    print("Agent budget: $0.03 — asking it to run forever...\n")
    result = await agent.run("Keep going forever")
    print(f"steps={result.steps} cost=${result.cost_usd:.4f}")
    print("OUTPUT:", result.output)


asyncio.run(main())
```

- [ ] **Step 4: Run it manually to confirm deterministic output before recording**

```bash
PYTHONPATH=src python3.11 scripts/growth/demo_budget_stop.py
```
Expected output (values are deterministic given the fixed costs above):
```
Agent budget: $0.03 — asking it to run forever...

steps=3 cost=$0.0250
OUTPUT: [ERROR] Agent '' budget $0.0300 exceeded. Spent $0.0250, attempted $0.0101.
```

- [ ] **Step 5: Write the VHS tape script**

Create `assets/demo.tape`:

```
Output assets/demo.gif

Set Shell "bash"
Set FontSize 16
Set Width 900
Set Height 500
Set Theme "Dracula"
Set Padding 20

Type "PYTHONPATH=src python3.11 scripts/growth/demo_budget_stop.py"
Sleep 500ms
Enter
Sleep 4s
```

- [ ] **Step 6: Record the GIF**

```bash
cd /Users/apple/developer/personal/helix-agent
vhs assets/demo.tape
```
Expected: `assets/demo.gif` is created. Verify with:
```bash
python3.11 -c "from PIL import Image; im = Image.open('assets/demo.gif'); print(im.size, im.n_frames)"
```
Expected: prints a size and a frame count greater than 1.

- [ ] **Step 7: Embed the GIF in the README**

In `README.md`, directly below the link row added in Task 3 (the `<a href="CONTRIBUTING.md">Contributing</a></p>` line), insert:

```markdown
<p align="center">
  <img src="assets/demo.gif" alt="Helix stopping a runaway agent at its budget cap" width="700">
</p>
```

- [ ] **Step 8: Verify the embed**

```bash
grep -n "assets/demo.gif" README.md
test -f assets/demo.gif && echo "gif exists"
```
Expected: one match in README.md, `gif exists` printed.

- [ ] **Step 9: Commit**

```bash
git add scripts/growth/demo_budget_stop.py assets/demo.tape assets/demo.gif README.md
git commit -m "feat(growth): add budget-enforcement demo GIF to README"
```

---

### Task 5: Social preview image

**Files:**
- Modify: `scripts/growth/generate_banner.py` (add a second call, no signature change — `render_banner` already supports `canvas_size`)
- Create: `assets/social-preview.png`

**Interfaces:**
- Consumes: `render_banner()` from Task 2 (unchanged signature).
- Produces: `assets/social-preview.png`, referenced by the manual checklist in Task 6.

- [ ] **Step 1: Add the social preview generation call**

In `scripts/growth/generate_banner.py`, change the `if __name__ == "__main__":` block to:

```python
if __name__ == "__main__":
    render_banner(
        "HELIX",
        "Production AI agents: budget limits, semantic caching, multi-agent teams",
        "assets/banner.png",
    )
    render_banner(
        "HELIX",
        "Production AI agents: budget limits, semantic caching, multi-agent teams",
        "assets/social-preview.png",
        font_size=36,
        canvas_size=(1280, 640),
    )
```

- [ ] **Step 2: Run and verify dimensions**

```bash
python3.11 scripts/growth/generate_banner.py
python3.11 -c "from PIL import Image; im = Image.open('assets/social-preview.png'); assert im.size == (1280, 640), im.size; print('OK', im.size)"
```
Expected: `OK (1280, 640)`. Open the file with the Read tool to confirm the text is centered and not clipped at this larger font size — if clipped, lower `font_size` to 30 and re-run.

- [ ] **Step 3: Commit**

```bash
git add scripts/growth/generate_banner.py assets/banner.png assets/social-preview.png
git commit -m "feat(growth): generate GitHub social preview image"
```

---

### Task 6: GitHub repo metadata doc + manual checklist

**Files:**
- Create: `docs/growth/github-metadata.md`

**Interfaces:**
- Consumes: `assets/social-preview.png` (Task 5).
- Produces: a doc the user follows manually in GitHub Settings — no other task depends on this file's content.

- [ ] **Step 1: Write the metadata doc**

Create `docs/growth/github-metadata.md`:

```markdown
# GitHub Repo Metadata — Manual Setup Checklist

These are GitHub Settings-page changes, not repo files — no CLI/API access
(`gh`) is available in this environment, so apply them by hand at
`https://github.com/sarcasticdhruv/helix-agent/settings`.

## About description

Set the repo "About" description (top-right gear icon on the repo homepage) to:

> Production-grade Python framework for AI agents — hard budget limits, semantic caching, multi-agent teams, MCP tools. Drop-in cost governance for LangChain/CrewAI/AutoGen.

## Topics

Add these topics (same gear icon, "Topics" field):

```
ai-agents, llm, agent-framework, multi-agent, python, mcp, langchain-alternative, crewai-alternative, autogen-alternative, llm-agents
```

## Social preview image

Settings → General → scroll to "Social preview" → Upload `assets/social-preview.png`.
This is the image shown when the repo link is shared on X/Slack/LinkedIn/Discord.

## Verification

After applying: share the repo URL in a private Slack DM to yourself or use
https://cards-dev.twitter.com/validator (or similar preview tool) to confirm
the social preview image renders instead of the default blank GitHub card.
```

- [ ] **Step 2: Verify the doc references an existing image**

```bash
grep -n "social-preview.png" docs/growth/github-metadata.md
test -f assets/social-preview.png && echo "referenced image exists"
```
Expected: one match, `referenced image exists` printed.

- [ ] **Step 3: Commit**

```bash
git add docs/growth/github-metadata.md
git commit -m "docs(growth): add GitHub repo metadata manual checklist"
```

---

### Task 7: Tighten llms.txt / llms-full.txt for quotability

**Files:**
- Modify: `llms.txt`
- Modify: `llms-full.txt`

**Interfaces:**
- Consumes: `## Common Complaints` from `docs/growth/research-findings.md` (Task 1).
- Produces: updated GEO files — no later task depends on exact wording, only that both files stay internally consistent with each other and with the README's Framework Comparison table.

- [ ] **Step 1: Add directly-quotable comparison sentences to llms.txt**

In `llms.txt`, after the existing `## Frameworks Helix Integrates With` section, add:

```markdown

## Direct Comparisons

- Helix vs CrewAI: Helix adds hard budget limits and semantic caching that CrewAI does not provide out of the box.
- Helix vs AutoGen: Helix adds built-in cost governance and a 6-scorer eval suite; AutoGen has neither natively.
- Helix vs LangGraph: Helix ships a LangGraph-compatible `StateGraph` plus hard budget limits, semantic caching, and multi-tier memory that LangGraph's core does not include.
- Helix is a good fit when: an existing CrewAI/AutoGen/LangChain team needs a cost-governance and observability layer without rewriting agent logic (see Framework Adapters).
```

- [ ] **Step 2: Mirror the same section into llms-full.txt**

In `llms-full.txt`, find the section that mirrors `llms.txt`'s content (concatenated near the top, before the full README/USAGE.md dump) and insert the identical `## Direct Comparisons` block from Step 1 in the same relative position.

- [ ] **Step 3: Verify both files stay in sync on this section**

```bash
grep -A6 "^## Direct Comparisons" llms.txt > /tmp/llms_section.txt
grep -A6 "^## Direct Comparisons" llms-full.txt > /tmp/llms_full_section.txt
diff /tmp/llms_section.txt /tmp/llms_full_section.txt
```
Expected: no output (files identical for this section).

- [ ] **Step 4: Commit**

```bash
git add llms.txt llms-full.txt
git commit -m "docs(geo): add directly-quotable framework comparison sentences"
```

---

### Task 8: Expand README FAQ

**Files:**
- Modify: `README.md` (FAQ section, currently `README.md:1107-1131` per the pre-Task-3/4 line numbers — re-locate by heading text, not line number, since earlier tasks shift line numbers)

**Interfaces:**
- Consumes: `## Common Complaints` from `docs/growth/research-findings.md` (Task 1).
- Produces: expanded FAQ — no later task depends on its content.

- [ ] **Step 1: Locate the FAQ section**

```bash
grep -n "^## FAQ" README.md
```
Note the line number for the next step (insert new Q&A pairs before the closing `---` that ends the FAQ section, i.e. immediately after the last existing `**Does Helix require an API key to try it?**` answer).

- [ ] **Step 2: Add new Q&A pairs sourced from research findings**

Immediately after the existing "Does Helix require an API key to try it?" answer and before the section's closing `---`, insert 3-4 new pairs, each following the exact format of the existing ones (`**Question?**` on its own line, answer directly below). Draft each pair from a specific line in `docs/growth/research-findings.md`'s `## Common Complaints` section — for example, if that section says "found out the hard way after a $400 AutoGen bill", add:

```markdown
**Can an agent run away and rack up a huge bill?**
Not with Helix — `BudgetConfig(budget_usd=...)` is enforced per run and raises `BudgetExceededError` before the next LLM call would exceed it, not after the bill arrives.
```

Write one such pair per distinct complaint found in Task 1's research — do not invent complaints that weren't in that file.

- [ ] **Step 3: Verify FAQ structure is still well-formed**

```bash
grep -c "^\*\*.*\?\*\*$" README.md
```
Expected: a count higher than before Task 8 (baseline was 6 question lines; confirm the new count is `6 + <number of pairs you added>`).

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(geo): expand FAQ with research-sourced questions"
```

---

### Task 9: mkdocs scaffold + content migration

**Files:**
- Create: `mkdocs.yml`
- Create: `docs/site/index.md`
- Create: `docs/site/quickstart.md`
- Create: `docs/site/agents.md`
- Create: `docs/site/multi-agent-teams.md`
- Create: `docs/site/mcp-tools.md`
- Create: `docs/site/evaluation.md`
- Create: `docs/site/framework-comparison.md`
- Create: `docs/site/faq.md`

**Interfaces:**
- Consumes: final README.md content from Tasks 3, 4, 7, 8 (split, not rewritten).
- Produces: `mkdocs.yml` with `docs_dir: docs/site`, consumed by Task 10 (CI workflow) and Task 11 (theme override path).

Use `docs/site/` (not bare `docs/`) as the mkdocs source directory so it doesn't collide with the existing `docs/superpowers/` and `docs/growth/` directories already in this repo.

- [ ] **Step 1: Install mkdocs and the Material theme**

```bash
python3.11 -m pip install mkdocs mkdocs-material
mkdocs --version
```
Expected: prints an mkdocs version.

- [ ] **Step 2: Write mkdocs.yml**

Create `mkdocs.yml` at the repo root:

```yaml
site_name: Helix
site_description: Production-grade Python framework for AI agents — cost governance, memory, caching, multi-agent teams, and built-in eval.
site_url: https://sarcasticdhruv.github.io/helix-agent/
repo_url: https://github.com/sarcasticdhruv/helix-agent
repo_name: sarcasticdhruv/helix-agent
docs_dir: docs/site

theme:
  name: material
  custom_dir: docs/overrides
  palette:
    scheme: slate
    primary: black
  features:
    - navigation.instant
    - navigation.top
    - content.code.copy

nav:
  - Home: index.md
  - Quickstart: quickstart.md
  - Agents: agents.md
  - Multi-Agent Teams: multi-agent-teams.md
  - MCP Tools: mcp-tools.md
  - Evaluation: evaluation.md
  - Framework Comparison: framework-comparison.md
  - FAQ: faq.md

markdown_extensions:
  - tables
  - fenced_code
  - admonition
```

Note: `custom_dir: docs/overrides` points to Task 11's override directory, which doesn't exist yet — mkdocs only errors on a missing `custom_dir` at build time, not at config-parse time, so this is safe to leave referenced now and created in Task 11. If you want to build before Task 11, temporarily remove the `custom_dir` line.

- [ ] **Step 3: Migrate content into per-page files**

Create each `docs/site/*.md` file by copying the corresponding section's content from the current `README.md`/`USAGE.md` (use the section headings already in `README.md` — e.g. `## Quickstart`, `## Agents`, `## Multi-Agent Teams`, `## MCP Tools`, `## Evaluation`, `## Framework Comparison`, `## FAQ` — as the exact boundaries to copy between). `docs/site/index.md` gets the hero block (banner, tagline, badges, link row) plus the introductory paragraph, mirroring the top of `README.md`. Add one `# <Title>` H1 at the top of each file matching its nav label above.

- [ ] **Step 4: Build and verify**

```bash
mkdocs build --strict
```
Expected: exits 0 with no warnings (with the `custom_dir` line temporarily removed if Task 11 hasn't run yet). Then:
```bash
ls site/index.html site/quickstart/index.html site/faq/index.html
```
Expected: all three files exist.

- [ ] **Step 5: Add generated site/ to .gitignore**

Append to `.gitignore`:
```
site/
```

- [ ] **Step 6: Commit**

```bash
git add mkdocs.yml docs/site .gitignore
git commit -m "feat(seo): scaffold mkdocs docs site split from README"
```

---

### Task 10: GitHub Pages deploy workflow

**Files:**
- Create: `.github/workflows/docs.yml`

**Interfaces:**
- Consumes: `mkdocs.yml` (Task 9).
- Produces: a live docs URL at `https://sarcasticdhruv.github.io/helix-agent/`, linked into the README hero's link row (Task 3's `Docs` link, updated here to point externally instead of to `USAGE.md`).

- [ ] **Step 1: Write the workflow**

Create `.github/workflows/docs.yml`:

```yaml
name: docs

on:
  push:
    branches: [main]
    paths:
      - "docs/site/**"
      - "mkdocs.yml"
      - ".github/workflows/docs.yml"
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install mkdocs mkdocs-material
      - run: mkdocs build --strict
      - uses: actions/upload-pages-artifact@v3
        with:
          path: site

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: Verify workflow YAML is well-formed**

```bash
python3.11 -c "import yaml; yaml.safe_load(open('.github/workflows/docs.yml'))" && echo "valid yaml"
```
Expected: `valid yaml` (install with `python3.11 -m pip install pyyaml` first if `yaml` isn't available).

- [ ] **Step 3: Update the README's Docs link**

In `README.md`'s link row (added in Task 3), change:
```markdown
<a href="USAGE.md">Docs</a> ·
```
to:
```markdown
<a href="https://sarcasticdhruv.github.io/helix-agent/">Docs</a> ·
```

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/docs.yml README.md
git commit -m "feat(seo): add GitHub Pages deploy workflow for docs site"
```

Note: GitHub Pages must be enabled once, manually, at `Settings → Pages → Source: GitHub Actions` — same manual-step constraint as Task 6, since it's a repo setting, not a file.

---

### Task 11: JSON-LD structured data on the docs site

**Files:**
- Create: `docs/overrides/main.html`

**Interfaces:**
- Consumes: `mkdocs.yml`'s `theme.custom_dir: docs/overrides` (Task 9).
- Produces: nothing consumed by later tasks — this is the last docs-site task.

- [ ] **Step 1: Find the mkdocs-material base template to extend**

```bash
python3.11 -c "import material, os; print(os.path.dirname(material.__file__))"
```
Note the printed path — confirms `{% extends "base.html" %}` is the correct pattern for this installed Material version (Material's `main.html` has extended `base.html` across all recent releases).

- [ ] **Step 2: Write the override**

Create `docs/overrides/main.html`:

```html
{% extends "base.html" %}

{% block htmltitle %}
<title>{{ page.title | default("Helix") }} — Helix</title>
{% endblock %}

{% block extrahead %}
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "Helix",
  "applicationCategory": "DeveloperApplication",
  "operatingSystem": "Cross-platform",
  "description": "Production-grade Python framework for AI agents — hard budget limits, semantic caching, multi-agent teams, MCP tools.",
  "url": "https://sarcasticdhruv.github.io/helix-agent/",
  "codeRepository": "https://github.com/sarcasticdhruv/helix-agent",
  "license": "https://www.apache.org/licenses/LICENSE-2.0",
  "programmingLanguage": "Python",
  "offers": {
    "@type": "Offer",
    "price": "0",
    "priceCurrency": "USD"
  }
}
</script>
{% endblock %}
```

- [ ] **Step 3: Rebuild with the override active and verify**

```bash
mkdocs build --strict
grep -o 'application/ld+json' site/index.html
```
Expected: `mkdocs build --strict` exits 0, and the grep prints `application/ld+json` (confirms the script tag made it into the built HTML).

- [ ] **Step 4: Commit**

```bash
git add docs/overrides/main.html
git commit -m "feat(seo): add SoftwareApplication JSON-LD to docs site"
```

---

### Task 12: External growth content drafts

**Files:**
- Create: `docs/growth/external-posts.md`

**Interfaces:**
- Consumes: `## Keywords` and `## Common Complaints` from `docs/growth/research-findings.md` (Task 1), the live docs URL from Task 10, and `assets/demo.gif` from Task 4.
- Produces: draft text only — the user posts these manually; nothing here is submitted automatically by any task.

- [ ] **Step 1: Draft the Show HN post**

Create `docs/growth/external-posts.md` starting with:

```markdown
# External Growth Content — Drafts Only

Nothing in this file gets posted automatically. Review and post manually.

## Show HN

**Title:** Show HN: Helix – a Python AI agent framework with hard budget limits

**Body:**

Hi HN — I built Helix because [pull the sharpest, most specific complaint from
docs/growth/research-findings.md's Common Complaints section here, e.g. "I lost
track of spend running AutoGen agents and got a surprise bill" — do not
invent one, use what the research turned up].

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
```

- [ ] **Step 2: Draft the Reddit posts**

Append to the same file:

```markdown
## Reddit — r/LocalLLaMA

**Title:** Built a Python agent framework with hard budget limits + semantic caching (Apache 2.0)

**Body:**

[2-3 sentences on the specific pain point from research findings, framed
conversationally, not as an ad] — sharing in case it's useful to others
running multi-step agents against local or paid models. Works with Ollama
and any OpenAI-compatible endpoint alongside the usual cloud providers.
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
```

- [ ] **Step 3: Draft awesome-list PR text**

Append:

```markdown
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
```

- [ ] **Step 4: Verify no placeholder brackets remain**

```bash
grep -n "\[.*research-findings.md\|invent one\|do not\]" docs/growth/external-posts.md
```
Expected: this should find the bracketed instruction in the Show HN draft — replace it with the actual complaint text pulled from Task 1's file before committing, then re-run and confirm no output.

- [ ] **Step 5: Commit**

```bash
git add docs/growth/external-posts.md
git commit -m "docs(growth): draft external growth posts (Show HN, Reddit, awesome-lists)"
```
