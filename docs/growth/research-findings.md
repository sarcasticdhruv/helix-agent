# Research Findings — SEO/GEO/Growth

Source: last30days skill run, 2026-07-27. Feeds README hero copy (Task 3), FAQ
expansion (Task 8), and external growth drafts (Task 12).

Run details: engine pass covered Reddit (27 threads, r/AI_Agents, r/LocalLLaMA,
r/LangChain, r/crewai, r/MachineLearning) and Hacker News (24 stories) for the
2026-06-26 to 2026-07-26 window, plus targeted WebSearch supplements (Step 2
of the skill) covering CrewAI/AutoGen/LangGraph comparison articles, the
CrewAI community forum, and Headroom's README/reception. X and YouTube were
unavailable in this environment (no X auth, no yt-dlp installed), so the pull
is Reddit + Hacker News + web-supplement only — see Concerns in the task
report.

## Keywords

AI agent framework python
CrewAI alternative
AutoGen alternative
LangGraph vs CrewAI vs AutoGen
cost control LLM agents
agent spending limits
which agent framework should I use
best open source agent framework 2026
AutoGen maintenance mode
token cost AI agents
multi-agent framework comparison
how to cap AI agent spend

## Common Complaints

- CrewAI has no native way to cap what a crew spends before it runs — one agency reported an overnight $2,400 API bill from a single crew that hit a tool error and kept retrying (per the CrewAI community forum thread "How are you handling agent spending limits in production crews?", community.crewai.com).
- AutoGen's default agent-to-agent conversation pattern multiplies token spend — a three-agent crew where each agent gets full context can run 3x baseline token cost, and production AutoGen deployments typically hard-cap `max_consecutive_auto_reply=3` to survive it (per 2026 framework comparison writeups, e.g. pooya.blog and Towards AI's LangGraph/CrewAI/AutoGen breakdown).
- Switching frameworks can quietly cost 4x–6x more per agent decision than the closest competitor, with teams "not finding out until the third invoice" — a framing that shows up repeatedly in 2026 framework-decision-matrix content.
- Microsoft put AutoGen into maintenance mode in late 2025, and the 2026 developer shorthand is blunt: "LangGraph for production, CrewAI for prototypes, AutoGen is dead."
- General framework-complexity complaint, recurring across dev blogs: "Frameworks are powerful and can do amazing things, but in practice, you end up only using 10% and then you realize that it's too complex to do the simple, specific things you need it to do."
- Builders describe uncontrolled agentic loops as "an expensive nightmare" — constant file-editing errors, infinite loops burning tens of thousands of tokens, and "phantom executions" where the orchestrator marks a task complete without doing the work.
- On r/AI_Agents (23 upvotes, 31 comments, 2026-07-26), a thread titled "The more I learn about AI automation, the less control I want to give the AI" reflects a live trust/control anxiety around autonomous agents, not just a cost concern.
- A same-day r/AI_Agents thread, "The AI agent market is about to discover that 'autonomous' and 'unsupervised' are not the same thing" (16 comments), signals rising skepticism toward marketing claims of agent autonomy — relevant framing for any positioning that emphasizes safe, bounded execution.

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
