"""Streamlit UI — thin presentation layer over the agent graph.

This module contains NO business logic. It only renders widgets,
calls the agent graph, and displays results.
"""

import streamlit as st

from frameworks.config import ENVIRONMENT
from frameworks.database import SessionLocal
from interface_adapters.embeddings.sentence_transformer_client import (
    SentenceTransformerEmbedder,
)
from interface_adapters.llm.ollama_client import OllamaClient
from interface_adapters.repositories.pgvector_search_repo import (
    PgVectorSearchRepository,
)
from interface_adapters.repositories.sqlalchemy_chunk_repo import (
    SQLAlchemyChunkRepository,
)
from use_cases.classify_intent import ClassifyIntentUseCase
from use_cases.decompose_queries import DecomposeQueriesUseCase
from use_cases.generate_strategy import GenerateStrategyUseCase
from use_cases.search_chunks import HybridSearchUseCase
from frameworks.agent_graph import FunnelAgentGraph


@st.cache_resource
def _build_agent() -> FunnelAgentGraph:
    """Wire dependencies and build the agent graph.

    Cached across Streamlit reruns to avoid re-creating sessions.
    """
    db = SessionLocal()
    search_repo = PgVectorSearchRepository(db)
    chunk_repo = SQLAlchemyChunkRepository(db)
    embedder = SentenceTransformerEmbedder()
    llm = OllamaClient()

    classify_uc = ClassifyIntentUseCase(llm)
    decompose_uc = DecomposeQueriesUseCase(llm)
    search_uc = HybridSearchUseCase(search_repo, embedder)
    generate_uc = GenerateStrategyUseCase(llm)

    return FunnelAgentGraph(classify_uc, decompose_uc, search_uc, generate_uc)


def render_app() -> None:
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Funnel Intelligence",
        page_icon="🎯",
        layout="wide",
    )

    st.title("🎯 Funnel Intelligence RAG")
    st.markdown(
        "Enter a client website and campaign brief to generate a data-backed "
        "funnel strategy with citations from proven case studies."
    )

    # Input section
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            url = st.text_input(
                "Client Website (optional)",
                placeholder="https://example.com",
                help="Used for context; the strategy is generated from the knowledge base.",
            )
        with col2:
            brief = st.text_area(
                "Campaign Brief",
                placeholder=(
                    "e.g., 30-day campaign to sell PV & Wärmepumpe Komplettsystem "
                    "to homeowners in rural Germany"
                ),
                height=100,
            )

        generate_clicked = st.button("Generate Strategy", type="primary", use_container_width=True)

    if not generate_clicked:
        st.info("👆 Enter a brief and click **Generate Strategy** to begin.")
        return

    if not brief.strip():
        st.warning("Please enter a campaign brief.")
        return

    # Build agent (cached)
    try:
        agent = _build_agent()
    except Exception as exc:
        st.error(f"Failed to initialize agent: {exc}")
        if ENVIRONMENT == "development":
            st.exception(exc)
        return

    # Run pipeline with progress
    with st.status("🔍 Analyzing brief...", expanded=True) as status:
        try:
            full_input = f"URL: {url}\nBrief: {brief}" if url else brief
            result = agent.run(full_input)
            status.update(label="✅ Strategy generated", state="complete")
        except Exception as exc:
            status.update(label="❌ Generation failed", state="error")
            st.error(f"Pipeline error: {exc}")
            if ENVIRONMENT == "development":
                st.exception(exc)
            return

    # Display results
    profile = result.get("profile")
    strategy = result.get("strategy")
    search_results = result.get("search_results", [])
    sub_queries = result.get("sub_queries", [])

    if profile:
        with st.expander("📊 Inferred Audience Profile", expanded=False):
            cols = st.columns(5)
            dims = [
                ("Risk Tolerance", profile.risk_tolerance),
                ("Purchase Cycle", profile.purchase_cycle),
                ("Tech Savviness", profile.tech_savviness),
                ("Age Bracket", profile.age_bracket),
                ("Price Sensitivity", profile.price_sensitivity),
            ]
            for col, (label, value) in zip(cols, dims):
                col.metric(label, value.title())

    if sub_queries:
        with st.expander("🔎 Retrieval Sub-Queries", expanded=False):
            for i, q in enumerate(sub_queries, 1):
                st.write(f"{i}. {q}")

    if strategy:
        st.divider()
        st.header(strategy.strategy_name)

        confidence_color = {
            "high": "green",
            "medium": "orange",
            "low": "red",
        }.get(strategy.confidence, "gray")
        st.markdown(f"**Confidence:** :{confidence_color}[{strategy.confidence.upper()}]")

        st.subheader("Target Audience")
        st.write(strategy.target_audience)

        st.subheader("Recommended Funnel Type")
        st.write(strategy.recommended_funnel_type)

        st.subheader("Key Steps")
        for step in strategy.key_steps:
            st.write(f"- {step}")

        st.subheader("Rationale")
        st.write(strategy.rationale)

        if strategy.citations:
            st.subheader("Sources")
            for citation in strategy.citations:
                st.write(f"- `{citation}`")

    if search_results:
        with st.expander("📚 Retrieved Context", expanded=False):
            for i, sr in enumerate(search_results[:10], 1):
                chunk = sr.chunk
                st.write(f"**{i}. {chunk.source_doc}** ({chunk.doc_subtype})")
                st.caption(f"{chunk.section_path} | Score: {sr.similarity_score:.3f}")
                st.write(chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content)
                st.divider()
