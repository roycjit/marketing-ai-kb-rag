"""LangGraph agent orchestration — frameworks layer.

This module wires the use cases into a stateful graph.
It is the outermost layer: it knows about use cases, but use cases
know nothing about LangGraph.
"""

from typing import TypedDict

from langgraph.graph import END, StateGraph

from domain.models import PsychographicProfile, SearchResult, StrategyResponse
from use_cases.classify_intent import ClassifyIntentUseCase
from use_cases.decompose_queries import DecomposeQueriesUseCase
from use_cases.generate_strategy import GenerateStrategyUseCase
from use_cases.search_chunks import HybridSearchUseCase


class AgentState(TypedDict, total=False):
    """Shared state passed between graph nodes."""

    user_input: str
    profile: PsychographicProfile | None
    sub_queries: list[str]
    search_results: list[SearchResult]
    strategy: StrategyResponse | None
    error: str | None


class FunnelAgentGraph:
    """LangGraph-based agent for funnel strategy generation.

    Graph structure:
        classify → decompose → retrieve → generate → END
    """

    def __init__(
        self,
        classify_use_case: ClassifyIntentUseCase,
        decompose_use_case: DecomposeQueriesUseCase,
        search_use_case: HybridSearchUseCase,
        generate_use_case: GenerateStrategyUseCase,
    ) -> None:
        self._classify = classify_use_case
        self._decompose = decompose_use_case
        self._search = search_use_case
        self._generate = generate_use_case

        # Build graph
        graph = StateGraph(AgentState)
        graph.add_node("classify", self._classify_node)
        graph.add_node("decompose", self._decompose_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate", self._generate_node)

        graph.set_entry_point("classify")
        graph.add_edge("classify", "decompose")
        graph.add_edge("decompose", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)

        self._compiled = graph.compile()

    # ------------------------------------------------------------------ #
    # Node implementations
    # ------------------------------------------------------------------ #

    def _classify_node(self, state: AgentState) -> AgentState:
        """Infer psychographic profile from user input."""
        profile = self._classify.execute(state["user_input"])
        return {
            **state,
            "profile": profile,
        }

    def _decompose_node(self, state: AgentState) -> AgentState:
        """Break brief into sub-queries using profile."""
        profile = state.get("profile")
        if profile is None:
            return {**state, "sub_queries": [state["user_input"]], "error": "missing profile"}

        sub_queries = self._decompose.execute(
            brief=state["user_input"],
            profile=profile,
        )
        return {
            **state,
            "sub_queries": sub_queries,
        }

    def _retrieve_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant chunks for each sub-query."""
        sub_queries = state.get("sub_queries", [state["user_input"]])
        all_results: list[SearchResult] = []
        seen_ids: set[str] = set()

        for q in sub_queries:
            results = self._search.execute(q, top_k=5)
            for r in results:
                if r.chunk.chunk_id not in seen_ids:
                    all_results.append(r)
                    seen_ids.add(r.chunk.chunk_id)

        return {
            **state,
            "search_results": all_results,
        }

    def _generate_node(self, state: AgentState) -> AgentState:
        """Generate strategy from retrieved context."""
        profile = state.get("profile")
        if profile is None:
            profile = PsychographicProfile(
                risk_tolerance="medium",
                purchase_cycle="medium",
                tech_savviness="medium",
                age_bracket="35-54",
                price_sensitivity="medium",
            )

        strategy = self._generate.execute(
            brief=state["user_input"],
            profile=profile,
            search_results=state.get("search_results", []),
        )
        return {
            **state,
            "strategy": strategy,
        }

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self, user_input: str) -> AgentState:
        """Execute the full agent pipeline.

        Args:
            user_input: Raw user brief.

        Returns:
            Final state containing strategy, profile, sub_queries, and sources.
        """
        return self._compiled.invoke({"user_input": user_input})
