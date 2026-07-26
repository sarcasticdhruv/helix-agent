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
