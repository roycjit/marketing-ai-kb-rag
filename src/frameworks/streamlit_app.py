"""Streamlit UI — thin presentation layer over the agent graph.

This module contains NO business logic. It only renders widgets,
calls the agent graph, and displays results.
"""

from __future__ import annotations

import contextlib
import time
import uuid

import streamlit as st
import structlog

from frameworks.agent_graph import FunnelAgentGraph
from frameworks.config import ENVIRONMENT
from frameworks.database import get_db
from interface_adapters.embeddings.cross_encoder_client import CrossEncoderReranker
from interface_adapters.embeddings.sentence_transformer_client import (
    SentenceTransformerEmbedder,
)
from interface_adapters.llm.ollama_client import OllamaClient
from interface_adapters.repositories.pgvector_search_repo import (
    PgVectorSearchRepository,
)
from use_cases.classify_intent import ClassifyIntentUseCase
from use_cases.decompose_queries import DecomposeQueriesUseCase
from use_cases.generate_strategy import GenerateStrategyUseCase
from use_cases.search_chunks import HybridSearchUseCase

logger = structlog.get_logger(__name__)

# Rate-limiting constants
_MIN_REQUEST_INTERVAL_SECONDS = 30


def _get_agent_for_session() -> FunnelAgentGraph:
    """Build the agent graph with a fresh database session.

    Unlike the previous @st.cache_resource approach that leaked sessions,
    this creates a new session per Streamlit session and closes it on cleanup.
    """
    db = next(get_db())

    # Ensure session is closed when the Streamlit script run ends
    # by registering it in session state for manual cleanup on rerun
    if "db_session" in st.session_state:
        with contextlib.suppress(Exception):
            st.session_state["db_session"].close()

    st.session_state["db_session"] = db

    search_repo = PgVectorSearchRepository(db)
    embedder = SentenceTransformerEmbedder()
    reranker = CrossEncoderReranker()
    llm = OllamaClient()

    classify_uc = ClassifyIntentUseCase(llm)
    decompose_uc = DecomposeQueriesUseCase(llm)
    search_uc = HybridSearchUseCase(search_repo, embedder, reranker=reranker)
    generate_uc = GenerateStrategyUseCase(llm)

    return FunnelAgentGraph(classify_uc, decompose_uc, search_uc, generate_uc)


def _check_rate_limit() -> bool:
    """Return True if the user is allowed to make a request."""
    last_request = st.session_state.get("last_request_time", 0)
    now = time.time()
    if now - last_request < _MIN_REQUEST_INTERVAL_SECONDS:
        return False
    st.session_state["last_request_time"] = now
    return True


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
                max_chars=10_000,
            )

        generate_clicked = st.button("Generate Strategy", type="primary", use_container_width=True)

    if not generate_clicked:
        st.info("👆 Enter a brief and click **Generate Strategy** to begin.")
        return

    if not brief.strip():
        st.warning("Please enter a campaign brief.")
        return

    # Rate limiting
    if not _check_rate_limit():
        st.warning(f"⏱️ Please wait {_MIN_REQUEST_INTERVAL_SECONDS} seconds between requests.")
        return

    # Generate correlation ID for tracing this request
    request_id = str(uuid.uuid4())[:8]
    log = logger.bind(request_id=request_id)

    # Build agent with fresh DB session (NOT cached)
    try:
        agent = _get_agent_for_session()
    except Exception as exc:
        log.error("agent_initialization_failed", error=str(exc))
        st.error(f"Failed to initialize agent: {exc}")
        if ENVIRONMENT == "development":
            st.exception(exc)
        return

    # Run pipeline with progress
    with st.status("🔍 Analyzing brief...", expanded=True) as status:
        try:
            full_input = f"URL: {url}\nBrief: {brief}" if url else brief
            log.info("pipeline_started", brief_preview=brief[:80])
            result = agent.run(full_input)
            status.update(label="✅ Strategy generated", state="complete")
            log.info("pipeline_complete", has_strategy=result.get("strategy") is not None)
        except Exception as exc:
            status.update(label="❌ Generation failed", state="error")
            log.error("pipeline_failed", error=str(exc))
            st.error(f"Pipeline error: {exc}")
            if ENVIRONMENT == "development":
                st.exception(exc)
            return

    # Display results
    profile = result.get("profile")
    strategy = result.get("strategy")
    search_results = result.get("search_results", [])
    sub_queries = result.get("sub_queries", [])
    error = result.get("error")

    if error:
        st.warning(f"Partial result: {error}")

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
            for col, (label, value) in zip(cols, dims, strict=True):
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
