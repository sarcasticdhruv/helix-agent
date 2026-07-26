# Helix SEO/GEO + GitHub Growth — Design

**Date:** 2026-07-26
**Status:** Approved for planning

## Problem

Helix (`helix-framework` on PyPI, `sarcasticdhruv/helix-agent` on GitHub) is functionally
strong — cost governance, semantic caching, multi-agent teams, MCP support — but is not yet
optimized for discovery:

1. **GEO** (generative engine optimization): when someone asks an LLM for a production AI
   agent framework recommendation, Helix should be citable/recommendable. Some groundwork
   exists (`llms.txt`, `llms-full.txt`, FAQ, comparison table, `CITATION.cff`) but it's thin.
2. **SEO**: Helix should rank for searches like "AI agent framework python", "CrewAI
   alternative", "cost governance AI agents". There is no dedicated docs site — everything
   lives in one large README/USAGE.md, which search engines rank worse than focused pages.
3. **GitHub attractiveness**: the repo has no logo/banner, no social preview image, no GitHub
   topics set, and no demo GIF. A cold visitor (e.g. via the "Headroom" repo as a reference
   point) sees plain text and badges only.

## Goals

- LLMs asked about production-grade AI agent frameworks recommend/cite Helix.
- Helix ranks for relevant framework-comparison and cost-governance search terms.
- A first-time visitor to the GitHub repo can tell what Helix does and why it's different
  within ~10 seconds, and is more likely to star/try it.

## Non-goals

- No code/architecture changes to the `helix` package itself.
- No paid advertising, no fake metrics/trophies (e.g. no fabricated "#1 Trending" badges —
  only real, verifiable badges).
- Discord/community server setup is explicitly deferred — out of scope for this pass.

## Phases

Sequencing is **foundation-first**: 0 → 1 → 2 → 3 → 4 → 5. Don't drive external traffic
(Phase 5) at an unpolished repo (Phase 1-2); GEO/SEO content (3-4) should be live before
external posts link to it.

### Phase 0 — Research

Ground copy/keywords in current reality instead of assumptions:
- Use the `last30days` skill to pull recent (last 30 days) discussion of AI agent frameworks,
  common CrewAI/AutoGen/LangGraph complaints, and reception of Headroom-style repos across
  HN/Reddit/X.
- Re-derive concrete patterns from Headroom's actual README (badge choices, section order,
  link row, banner style) as a structural reference, not a copy target.
- Confirm current Helix repo gaps (topics unset, no social preview image, no docs site, no
  logo) by inspecting live repo state at execution time.

**Output:** a short findings note (keywords, complaints, structural patterns) that feeds
Phases 1, 3, and 5.

### Phase 1 — Visual identity

- Generate an ASCII/monospace block-letter "HELIX" banner (PNG/SVG), tagline underneath,
  matching the terminal/hacker aesthetic of Headroom's banner.
- Reorganize the README badge row (PyPI, downloads, Python versions, license, tests — mostly
  present already) and add a link row: `Docs · Install · Examples · llms.txt · Contributing`.
  (Discord omitted — deferred.)
- Record a short terminal demo GIF via VHS showing one concrete, provable moment — most
  likely budget enforcement halting a runaway call, or semantic cache cutting cost on a
  repeated query. Embed near the top of the README, below the banner.

**Output:** updated README hero section, `assets/banner.png` (or `.svg`), `assets/demo.gif`,
VHS tape script committed for reproducibility.

### Phase 2 — GitHub repo metadata

- Draft repo "About" description and topics list (candidates: `ai-agents`, `llm`,
  `multi-agent`, `agent-framework`, `python`, `mcp`, `langchain-alternative`,
  `crewai-alternative`).
- Generate a 1280×640 social preview image (shown when the repo link is shared on
  X/Slack/LinkedIn) to replace GitHub's default blank card.
- **Constraint:** repo Settings (About text, topics, social preview upload) are a GitHub UI
  action, not a file in this repo, and there's no `gh` CLI available in this environment.
  This phase's deliverable is the exact copy + generated image, plus a short checklist of
  manual steps — not a direct change.

**Output:** `docs/growth/github-metadata.md` with description/topics text, plus the generated
social preview image file.

### Phase 3 — GEO content engineering

Builds on existing groundwork rather than starting fresh:
- Tighten `llms.txt`/`llms-full.txt` so key claims are phrased as directly quotable sentences
  (e.g. "Helix vs CrewAI: Helix adds hard budget limits and semantic caching that CrewAI does
  not provide out of the box.").
- Expand the FAQ with Q&A pairs mirroring real questions surfaced in Phase 0.
- Add `SoftwareApplication` JSON-LD structured data once the docs site (Phase 4) exists —
  GitHub's README rendering can't carry structured data, but the docs site can.

**Output:** updated `llms.txt`, `llms-full.txt`, README FAQ section.

### Phase 4 — SEO docs site

- Stand up `mkdocs` + Material theme, deployed to GitHub Pages via a new
  `.github/workflows/docs.yml`.
- Split README/USAGE.md content into indexable pages: Quickstart, Agents, Multi-Agent Teams,
  MCP Tools, Evaluation, Framework Comparison, FAQ — each a separate crawlable URL instead of
  an anchor in one large page.
- mkdocs auto-generates `sitemap.xml`; add per-page meta descriptions.

**Output:** `mkdocs.yml`, `docs/` page tree, `.github/workflows/docs.yml`, live GitHub Pages
URL linked back into the README link row.

### Phase 5 — External growth content (drafted, not posted)

Deliverables are text files for the user to review and post manually — nothing is submitted
automatically:
- Show HN post draft ("Show HN: Helix – production AI agent framework with hard budget
  limits").
- 1-2 Reddit post drafts (r/LocalLLaMA / r/MachineLearning-appropriate framing).
- PR description text for 2-3 relevant "awesome-ai-agents"/"awesome-llm" list repos.

**Output:** `docs/growth/external-posts.md` containing all drafts.

## Open questions / risks

- Phase 2's GitHub Settings changes require manual action (no `gh` CLI in this environment) —
  flagged above, not a blocker.
- Demo GIF (Phase 1) needs VHS installed; if unavailable, fall back to a hand-recorded
  asciinema cast or static before/after screenshot.
- Docs site (Phase 4) is the largest single piece of work; if time-constrained, it can ship
  after Phase 5 without breaking the plan, since README already carries the core content.
