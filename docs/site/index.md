# Home

<p align="center">
  <img src="assets/banner.png" alt="Helix — production AI agent framework" width="600">
</p>

<p align="center"><b>Production AI agents: hard budget limits, semantic caching, multi-agent teams, MCP tools.</b></p>

<p align="center">
<a href="https://pypi.org/project/helix-framework/"><img src="https://img.shields.io/pypi/v/helix-framework" alt="PyPI"></a>
<a href="https://pepy.tech/project/helix-framework"><img src="https://static.pepy.tech/badge/helix-framework" alt="Downloads"></a>
<a href="https://pypi.org/project/helix-framework/"><img src="https://img.shields.io/pypi/pyversions/helix-framework" alt="Python"></a>
<a href="https://github.com/sarcasticdhruv/helix-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License"></a>
</p>

<p align="center">
<a href="quickstart/">Install</a> ·
<a href="quickstart/">Quickstart</a> ·
<a href="https://github.com/sarcasticdhruv/helix-agent/tree/main/examples">Examples</a> ·
<a href="https://github.com/sarcasticdhruv/helix-agent/blob/main/llms.txt">llms.txt</a> ·
<a href="https://github.com/sarcasticdhruv/helix-agent/blob/main/CONTRIBUTING.md">Contributing</a>
</p>

Helix gives you agents that actually behave in production: hard budget limits, semantic caching for repeated queries, opt-in persistent memory (SQLite, zero extra dependency), MCP tool support, agent handoffs, multi-agent teams, YAML-based task pipelines, a LangGraph-compatible `StateGraph`, and a 6-scorer eval suite. It works out of the box with OpenAI, Anthropic, Gemini, Groq, Mistral, and 8 other providers.

The `import helix` API is intentionally close to what you already know from AutoGen, CrewAI, and LangGraph, but with the production layer those frameworks leave to you: cost governance, caching, memory, observability, and safety controls.
