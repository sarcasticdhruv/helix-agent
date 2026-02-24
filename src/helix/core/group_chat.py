"""
helix/core/group_chat.py

ConversableAgent and GroupChat — AutoGen-style multi-turn conversational agents.

Key features:
  - ConversableAgent: an Agent that can converse with humans or other agents
  - GroupChat: N agents in a shared conversation with pluggable speaker selection
  - Speaker selection strategies: round_robin, auto (LLM picks next), random, custom
  - Termination conditions: max_rounds, keyword, custom callable
  - Human-in-the-loop: inject a HumanAgent anywhere in the group
  - Full message history shared across all agents

Usage::

    from helix import ConversableAgent, GroupChat

    ceo    = ConversableAgent(name="CEO",    role="CEO",    goal="Make decisions.")
    cto    = ConversableAgent(name="CTO",    role="CTO",    goal="Assess tech risk.")
    lawyer = ConversableAgent(name="Lawyer", role="Lawyer", goal="Check compliance.")

    chat = GroupChat(
        agents=[ceo, cto, lawyer],
        max_rounds=6,
        speaker_selection="auto",   # LLM decides who speaks next
        termination_keyword="DONE",
    )
    result = await chat.run("Should we adopt LLMs in our core product?")
    for msg in result.messages:
        print(f"[{msg.speaker}]: {msg.content}")
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any

from helix.core.agent import Agent, AgentResult

# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------


@dataclass
class ChatMessage:
    """A single message in a GroupChat conversation."""
    speaker: str
    content: str
    step: int = 0
    cost_usd: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return f"[{self.speaker}]: {self.content}"


# ---------------------------------------------------------------------------
# ConversableAgent
# ---------------------------------------------------------------------------


class ConversableAgent(Agent):
    """
    An Agent capable of multi-turn conversation.

    Extends :class:`helix.Agent` with:
      - ``reply()``      — generate one conversational turn
      - ``human_input``  — if True, prompts the terminal for a human reply
      - ``max_consecutive_replies`` — prevent one agent dominating

    In most cases you can use :class:`helix.Agent` directly inside a
    :class:`GroupChat`; ConversableAgent adds the human-input shortcut.
    """

    def __init__(
        self,
        name: str,
        role: str,
        goal: str,
        backstory: str = "",
        human_input: bool = False,
        max_consecutive_replies: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, role=role, goal=goal, backstory=backstory, **kwargs)
        self.human_input = human_input
        self.max_consecutive_replies = max_consecutive_replies
        self._consecutive_count = 0

    async def reply(
        self,
        conversation_history: list[ChatMessage],
        task: str,
    ) -> str:
        """Generate a reply based on the full conversation history."""
        if self.human_input:
            # Human-in-the-loop: read from stdin
            history_text = "\n".join(str(m) for m in conversation_history)
            print(f"\n--- Conversation so far ---\n{history_text}\n")
            return input(f"[{self.name} — YOUR REPLY]: ").strip()

        # Build a conversation-aware prompt
        history_text = "\n".join(str(m) for m in conversation_history[-10:])  # last 10
        prompt = (
            f"You are participating in a group discussion about:\n{task}\n\n"
            f"Conversation so far:\n{history_text}\n\n"
            f"As {self.name} ({self.config.role}), provide your contribution. "
            f"Be concise and focused."
        )
        result: AgentResult = await self.run(prompt)
        return str(result.output)


# ---------------------------------------------------------------------------
# HumanAgent — always prompts the terminal
# ---------------------------------------------------------------------------


class HumanAgent(ConversableAgent):
    """Represents a human participant in a GroupChat."""

    def __init__(self, name: str = "Human") -> None:
        super().__init__(
            name=name,
            role="Human participant",
            goal="Provide human judgment",
            human_input=True,
        )


# ---------------------------------------------------------------------------
# GroupChatResult
# ---------------------------------------------------------------------------


@dataclass
class GroupChatResult:
    """Result of running a GroupChat."""
    messages: list[ChatMessage] = field(default_factory=list)
    final_output: str = ""
    total_cost_usd: float = 0.0
    rounds: int = 0
    duration_s: float = 0.0
    terminated_by: str = ""  # "max_rounds" | "keyword" | "custom" | "error"
    error: str | None = None

    def transcript(self) -> str:
        """Full conversation as a readable string."""
        return "\n".join(str(m) for m in self.messages)

    def __str__(self) -> str:
        return self.final_output or self.transcript()


# ---------------------------------------------------------------------------
# GroupChat
# ---------------------------------------------------------------------------


_SPEAKER_STRATEGIES = ("round_robin", "auto", "random")


class GroupChat:
    """
    N agents in a shared multi-turn conversation.

    Speaker selection strategies:
      ``round_robin``  — agents speak in order (default)
      ``auto``         — a coordinator LLM picks the next speaker
      ``random``       — random selection each round
      ``callable``     — pass ``speaker_fn(agents, history) -> Agent``

    Termination:
      ``max_rounds``           — hard cap on conversation turns
      ``termination_keyword``  — any agent saying this word ends the chat
      ``termination_fn``       — ``(messages) -> bool``

    Usage::

        chat = GroupChat(
            agents=[agent1, agent2, agent3],
            max_rounds=10,
            speaker_selection="round_robin",
        )
        result = await chat.run("Debate the pros and cons of microservices.")
    """

    def __init__(
        self,
        agents: list[Agent | ConversableAgent],
        max_rounds: int = 10,
        speaker_selection: str | Any = "round_robin",
        termination_keyword: str | None = None,
        termination_fn: Any | None = None,
        allow_repeat_speaker: bool = True,
        intro_message: str | None = None,
    ) -> None:
        if not agents:
            raise ValueError("GroupChat requires at least one agent.")
        self._agents = agents
        self._max_rounds = max_rounds
        self._speaker_selection = speaker_selection
        self._termination_keyword = termination_keyword
        self._termination_fn = termination_fn
        self._allow_repeat_speaker = allow_repeat_speaker
        self._intro_message = intro_message

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, task: str) -> GroupChatResult:
        """Run the group conversation on `task`."""
        start = time.time()
        messages: list[ChatMessage] = []
        total_cost = 0.0

        # Optional intro
        if self._intro_message:
            messages.append(ChatMessage(speaker="System", content=self._intro_message, step=0))

        last_speaker_idx = -1

        for round_num in range(1, self._max_rounds + 1):
            # Pick next speaker
            speaker = await self._pick_speaker(round_num, last_speaker_idx, messages, task)
            last_speaker_idx = self._agents.index(speaker)

            # Generate reply
            try:
                if isinstance(speaker, ConversableAgent):
                    content = await speaker.reply(messages, task)
                    cost = 0.0
                else:
                    # Plain Agent — build contextual prompt
                    history_text = "\n".join(str(m) for m in messages[-10:])
                    prompt = (
                        f"Group discussion topic: {task}\n\n"
                        f"Conversation so far:\n{history_text}\n\n"
                        f"Provide your perspective as {speaker.name}."
                    )
                    result: AgentResult = await speaker.run(prompt)
                    content = str(result.output)
                    cost = result.cost_usd
                    total_cost += cost
            except Exception as exc:
                return GroupChatResult(
                    messages=messages,
                    total_cost_usd=total_cost,
                    rounds=round_num,
                    duration_s=time.time() - start,
                    terminated_by="error",
                    error=str(exc),
                )

            msg = ChatMessage(speaker=speaker.name, content=content, step=round_num, cost_usd=cost)
            messages.append(msg)

            # Check termination keyword
            if self._termination_keyword and self._termination_keyword.lower() in content.lower():
                return GroupChatResult(
                    messages=messages,
                    final_output=content,
                    total_cost_usd=total_cost,
                    rounds=round_num,
                    duration_s=time.time() - start,
                    terminated_by="keyword",
                )

            # Check custom termination fn
            if self._termination_fn and self._termination_fn(messages):
                return GroupChatResult(
                    messages=messages,
                    final_output=content,
                    total_cost_usd=total_cost,
                    rounds=round_num,
                    duration_s=time.time() - start,
                    terminated_by="custom",
                )

        # Max rounds reached
        final = messages[-1].content if messages else ""
        return GroupChatResult(
            messages=messages,
            final_output=final,
            total_cost_usd=total_cost,
            rounds=self._max_rounds,
            duration_s=time.time() - start,
            terminated_by="max_rounds",
        )

    def run_sync(self, task: str) -> GroupChatResult:
        """Synchronous convenience wrapper."""
        return asyncio.run(self.run(task))

    # ------------------------------------------------------------------
    # Speaker selection
    # ------------------------------------------------------------------

    async def _pick_speaker(
        self,
        round_num: int,
        last_idx: int,
        messages: list[ChatMessage],
        task: str,
    ) -> Agent | ConversableAgent:
        strategy = self._speaker_selection

        if strategy == "round_robin" or strategy is None:
            return self._agents[round_num % len(self._agents)]

        if strategy == "random":
            return random.choice(self._agents)

        if strategy == "auto":
            return await self._auto_pick(messages, task)

        if callable(strategy):
            return strategy(self._agents, messages)

        # Fallback
        return self._agents[round_num % len(self._agents)]

    async def _auto_pick(
        self,
        messages: list[ChatMessage],
        task: str,
    ) -> Agent | ConversableAgent:
        """Use the first available agent as a coordinator to pick the next speaker."""
        coordinator = self._agents[0]
        names = [a.name for a in self._agents]
        recent = "\n".join(str(m) for m in messages[-5:])

        prompt = (
            f"Discussion topic: {task}\n\n"
            f"Recent messages:\n{recent}\n\n"
            f"Available speakers: {', '.join(names)}\n"
            f"Who should speak next to best advance the conversation? "
            f"Reply with ONLY the speaker name (exact match)."
        )
        result: AgentResult = await coordinator.run(prompt)
        picked_name = str(result.output).strip().strip('"').strip("'")

        for agent in self._agents:
            if agent.name.lower() == picked_name.lower():
                return agent

        # Fallback: next in round-robin
        last_idx = 0
        if messages:
            for i, a in enumerate(self._agents):
                if a.name == messages[-1].speaker:
                    last_idx = i
                    break
        return self._agents[(last_idx + 1) % len(self._agents)]
