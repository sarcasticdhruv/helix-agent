"""
helix/adapters/langchain.py

LangChain-specific adapter.
Re-exports from universal.py for clean import paths.

Usage::

    from helix.adapters.langchain import from_langchain, from_langgraph
"""

from helix.adapters.universal import LangChainWrapper, from_langchain

__all__ = ["from_langchain", "LangChainWrapper"]


async def from_langgraph(
    compiled_graph: object,
    budget_usd: float,
    loop_limit: int = 50,
) -> "LangGraphWrapper":
    """
    Run a compiled LangGraph graph under Helix governance.
    Patches all LLMs inside the graph's nodes.
    """
    from helix.adapters.universal import HelixLLMShim, _guess_model_name
    from helix.config import AgentConfig, AgentMode, BudgetConfig
    from helix.context import ExecutionContext

    config = AgentConfig(
        name="langgraph",
        role="graph",
        goal="Execute LangGraph graph under Helix governance",
        mode=AgentMode.PRODUCTION,
        budget=BudgetConfig(budget_usd=budget_usd),
        loop_limit=loop_limit,
    )
    ctx = ExecutionContext(config=config)

    # Patch nodes if accessible
    nodes = getattr(compiled_graph, "nodes", {})
    for _, node in nodes.items():
        if hasattr(node, "llm"):
            model_name = _guess_model_name(node.llm)
            node.llm = HelixLLMShim(underlying=node.llm, context=ctx, model_name=model_name)

    return LangGraphWrapper(graph=compiled_graph, context=ctx)


class LangGraphWrapper:
    def __init__(self, graph: object, context: object) -> None:
        self._graph = graph
        self._context = context

    async def run(self, input: object) -> object:
        import asyncio

        try:
            invoke_fn = getattr(self._graph, "ainvoke", None) or getattr(
                self._graph, "invoke", None
            )
            if invoke_fn is None:
                raise AttributeError("Graph has no invoke method")
            if asyncio.iscoroutinefunction(invoke_fn):
                return await invoke_fn(input)
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: invoke_fn(input))
        except Exception as e:
            from helix.errors import AdapterError

            raise AdapterError(framework="langgraph", reason=str(e)) from e

    @property
    def cost_usd(self) -> float:
        return self._context.cost.spent_usd  # type: ignore[attr-defined]
