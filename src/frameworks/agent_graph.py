"""LangGraph agent orchestration — frameworks layer.

This module wires the use cases into a stateful graph.
It is the outermost layer: it knows about use cases, but use cases
know nothing about LangGraph.
"""

from __future__ import annotations

from typing import TypedDict

import structlog
from langgraph.graph import END, StateGraph

from domain.models import PsychographicProfile, SearchResult, StrategyResponse
from use_cases.classify_intent import ClassifyIntentUseCase
from use_cases.decompose_queries import DecomposeQueriesUseCase
from use_cases.generate_strategy import GenerateStrategyUseCase
from use_cases.search_chunks import HybridSearchUseCase

logger = structlog.get_logger(__name__)


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
        """Initialize the agent graph with injected use cases."""
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
    # Node implementations — each wrapped in try/except for resilience
    # ------------------------------------------------------------------ #

    def _classify_node(self, state: AgentState) -> AgentState:
        """Infer psychographic profile from user input."""
        try:
            profile = self._classify.execute(state["user_input"])
            return {**state, "profile": profile}
        except Exception as exc:
            logger.warning("classify_node_failed", error=str(exc))
            return {
                **state,
                "profile": None,
                "error": f"Classification failed: {exc}",
            }

    def _decompose_node(self, state: AgentState) -> AgentState:
        """Break brief into sub-queries using profile."""
        profile = state.get("profile")
        if profile is None:
            logger.info("decompose_node_fallback_no_profile")
            return {
                **state,
                "sub_queries": [state["user_input"]],
                "error": (state.get("error") or "") + "; missing profile, using raw input",
            }

        try:
            sub_queries = self._decompose.execute(
                brief=state["user_input"],
                profile=profile,
            )
            return {**state, "sub_queries": sub_queries}
        except Exception as exc:
            logger.warning("decompose_node_failed", error=str(exc))
            return {
                **state,
                "sub_queries": [state["user_input"]],
                "error": f"Decomposition failed: {exc}",
            }

    def _retrieve_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant chunks for each sub-query in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        sub_queries = state.get("sub_queries", [state["user_input"]])
        all_results: list[SearchResult] = []
        seen_ids: set[str] = set()

        def _search_one(q: str) -> list[SearchResult]:
            try:
                result: list[SearchResult] = self._search.execute(q, top_k=5)
                return result
            except Exception as exc:
                logger.warning("retrieve_subquery_failed", query=q, error=str(exc))
                return []

        # Parallelize independent sub-query searches
        with ThreadPoolExecutor(max_workers=min(len(sub_queries), 4)) as executor:
            futures = {executor.submit(_search_one, q): q for q in sub_queries}
            for future in as_completed(futures):
                results = future.result()
                for r in results:
                    if r.chunk.chunk_id not in seen_ids:
                        all_results.append(r)
                        seen_ids.add(r.chunk.chunk_id)

        if not all_results:
            logger.warning("retrieve_node_no_results", sub_queries=sub_queries)

        return {**state, "search_results": all_results}

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

        search_results = state.get("search_results", [])
        if not search_results:
            logger.warning("generate_node_no_context")
            fallback = StrategyResponse(
                strategy_name="Insufficient context",
                target_audience="Unable to determine",
                recommended_funnel_type="TBD",
                key_steps=["No relevant documents were retrieved for this brief."],
                rationale="The retrieval step returned no results.",
                citations=[],
                confidence="low",
            )
            return {**state, "strategy": fallback}

        try:
            strategy = self._generate.execute(
                brief=state["user_input"],
                profile=profile,
                search_results=search_results,
            )
            return {**state, "strategy": strategy}
        except Exception as exc:
            logger.error("generate_node_failed", error=str(exc))
            fallback = StrategyResponse(
                strategy_name="Generation failed",
                target_audience="Unable to determine",
                recommended_funnel_type="TBD",
                key_steps=["An error occurred during strategy generation."],
                rationale=f"Error: {exc}",
                citations=[],
                confidence="low",
            )
            return {
                **state,
                "strategy": fallback,
                "error": f"Generation failed: {exc}",
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
        from typing import cast

        result = cast(AgentState, self._compiled.invoke({"user_input": user_input}))
        return result
