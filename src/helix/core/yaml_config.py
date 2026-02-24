"""
helix/core/yaml_config.py

YAML-based agent and task configuration loader.

Inspired by CrewAI's @CrewBase / agents.yaml / tasks.yaml pattern.
Lets teams define agents and tasks in YAML, then load them in Python.

Usage::

    # agents.yaml
    # -----------
    # researcher:
    #   role: Senior Research Analyst
    #   goal: Uncover cutting-edge developments in {topic}.
    #   backstory: You work at a leading tech think tank.
    #
    # writer:
    #   role: Tech Content Strategist
    #   goal: Craft compelling content about {topic}.
    #   backstory: You are a renowned content writer.

    # tasks.yaml
    # ----------
    # research_task:
    #   description: Research the latest advances in {topic}.
    #   expected_output: A structured report with 5 key findings.
    #   agent: researcher
    #
    # write_task:
    #   description: Write a concise article based on the research.
    #   expected_output: A 3-paragraph article.
    #   agent: writer

    # Python
    # ------
    # from helix.core.yaml_config import load_agents, load_tasks, load_pipeline
    #
    # agents = load_agents("agents.yaml", inputs={"topic": "quantum computing"})
    # tasks  = load_tasks("tasks.yaml", agents, inputs={"topic": "quantum computing"})
    # pipeline = load_pipeline(tasks)
    # result = pipeline.kickoff()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from helix.config import BudgetConfig, ModelConfig
from helix.core.agent import Agent
from helix.core.task import Pipeline, Task

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _substitute(value: Any, inputs: dict[str, Any]) -> Any:
    """Recursively substitute {var} placeholders in strings and nested structures."""
    if isinstance(value, str):
        for k, v in inputs.items():
            value = value.replace(f"{{{k}}}", str(v))
        return value
    if isinstance(value, dict):
        return {k: _substitute(v, inputs) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(item, inputs) for item in value]
    return value


def _load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file, returning a dict. Requires PyYAML."""
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for YAML config loading. "
            "Install it with: pip install pyyaml"
        ) from exc

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_agents(
    path: str | Path,
    inputs: dict[str, Any] | None = None,
    *,
    model: ModelConfig | None = None,
    budget_usd: float | None = None,
) -> dict[str, Agent]:
    """
    Load agents from a YAML file.

    YAML format::

        researcher:
          role: Senior Research Analyst
          goal: Uncover developments in {topic}.
          backstory: You are an expert researcher with 10 years experience.
          model: gemini-2.0-flash      # optional — overrides global default
          budget_usd: 0.50             # optional

        writer:
          role: Content Strategist
          goal: Write engaging articles about {topic}.

    Returns a dict of ``{agent_key: Agent}``.
    """
    raw = _load_yaml(path)
    _inputs = inputs or {}
    agents: dict[str, Agent] = {}

    for key, cfg in raw.items():
        if not isinstance(cfg, dict):
            continue
        cfg = _substitute(cfg, _inputs)

        agent_model = None
        if "model" in cfg:
            agent_model = ModelConfig(primary=cfg["model"])
        elif model:
            agent_model = model

        agent_budget = None
        raw_budget = cfg.get("budget_usd") or budget_usd
        if raw_budget:
            agent_budget = BudgetConfig(budget_usd=float(raw_budget))

        agents[key] = Agent(
            name=cfg.get("name", key),
            role=cfg["role"],
            goal=cfg["goal"],
            backstory=cfg.get("backstory", ""),
            model=agent_model,
            budget=agent_budget,
        )

    return agents


def load_tasks(
    path: str | Path,
    agents: dict[str, Agent],
    inputs: dict[str, Any] | None = None,
) -> list[Task]:
    """
    Load tasks from a YAML file, linking them to loaded agents.

    YAML format::

        research_task:
          description: Research the latest advances in {topic}.
          expected_output: A structured report with 5 key findings.
          agent: researcher          # must match a key in agents dict
          async_execution: false
          output_file: research.md   # optional

        write_task:
          description: Write an article based on the research.
          expected_output: A 3-paragraph article.
          agent: writer
          context: [research_task]   # wait for research_task output

    Returns an ordered list of Tasks (preserving YAML order).
    """
    raw = _load_yaml(path)
    _inputs = inputs or {}
    task_map: dict[str, Task] = {}
    task_list: list[Task] = []

    # First pass: create Task objects (without context resolution)
    for key, cfg in raw.items():
        if not isinstance(cfg, dict):
            continue
        cfg = _substitute(cfg, _inputs)

        agent_key = cfg.get("agent")
        if agent_key and agent_key not in agents:
            raise KeyError(
                f"Task '{key}' references agent '{agent_key}' which was not found. "
                f"Available agents: {list(agents.keys())}"
            )

        task = Task(
            description=cfg.get("description", key),
            expected_output=cfg.get("expected_output", ""),
            agent=agents.get(agent_key) if agent_key else None,
            async_execution=bool(cfg.get("async_execution", False)),
            output_file=cfg.get("output_file"),
            markdown=bool(cfg.get("markdown", False)),
            name=cfg.get("name", key),
        )
        task_map[key] = task
        task_list.append(task)

    # Second pass: resolve context dependencies
    __raw_list = list(raw.items())
    for i, (_key, cfg) in enumerate(__raw_list):
        if not isinstance(cfg, dict):
            continue
        context_keys: list[str] = cfg.get("context", []) or []
        for ctx_key in context_keys:
            if ctx_key in task_map:
                task_list[i].context.append(task_map[ctx_key])

    return task_list


def load_pipeline(
    tasks: list[Task],
    *,
    inputs: dict[str, Any] | None = None,
) -> Pipeline:
    """
    Create a Pipeline from a list of Tasks.

    This is the Helix equivalent of::

        crew = Crew(agents=[...], tasks=[...], process=Process.sequential)

    Returns a :class:`helix.Pipeline` ready for ``.kickoff(inputs=...)`` or
    ``await .run(inputs=...)``.
    """
    return Pipeline(tasks=tasks)


def from_yaml(
    agents_yaml: str | Path,
    tasks_yaml: str | Path,
    inputs: dict[str, Any] | None = None,
    *,
    model: ModelConfig | None = None,
    budget_usd: float | None = None,
) -> Pipeline:
    """
    One-shot helper: load agents + tasks + build pipeline from two YAML files.

    Example::

        from helix.core.yaml_config import from_yaml

        pipeline = from_yaml("agents.yaml", "tasks.yaml", inputs={"topic": "AI"})
        result = pipeline.kickoff()
        print(result.final_output)
    """
    agents = load_agents(agents_yaml, inputs=inputs, model=model, budget_usd=budget_usd)
    tasks = load_tasks(tasks_yaml, agents, inputs=inputs)
    return load_pipeline(tasks, inputs=inputs)
