"""Tests for FunnelAgentGraph with mocked use cases."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from domain.models import Chunk, PsychographicProfile, SearchResult, StrategyResponse
from frameworks.agent_graph import FunnelAgentGraph


@pytest.fixture
def mock_classify():
    uc = MagicMock()
    uc.execute.return_value = PsychographicProfile(
        risk_tolerance="low",
        purchase_cycle="long",
        tech_savviness="low",
        age_bracket="55+",
        price_sensitivity="high",
    )
    return uc


@pytest.fixture
def mock_decompose():
    uc = MagicMock()
    uc.execute.return_value = ["query a", "query b"]
    return uc


@pytest.fixture
def mock_search():
    uc = MagicMock()
    chunk = Chunk(
        chunk_id="c1",
        content="Solar funnels work well for retirees",
        source_doc="solar-guide.md",
        doc_type="guide",
        doc_subtype="guide",
        last_updated=datetime.now(timezone.utc),
    )
    uc.execute.return_value = [SearchResult(chunk=chunk, similarity_score=0.9)]
    return uc


@pytest.fixture
def mock_generate():
    uc = MagicMock()
    uc.execute.return_value = StrategyResponse(
        strategy_name="Test Strategy",
        target_audience="Test Audience",
        recommended_funnel_type="landing_page",
        key_steps=["Step 1"],
        rationale="Test rationale",
        citations=["x.md"],
        confidence="high",
    )
    return uc


class TestRun:
    def test_full_pipeline(self, mock_classify, mock_decompose, mock_search, mock_generate):
        graph = FunnelAgentGraph(mock_classify, mock_decompose, mock_search, mock_generate)
        result = graph.run("Sell PV to cautious retirees")

        assert result["profile"] is not None
        assert result["profile"].age_bracket == "55+"
        assert result["sub_queries"] == ["query a", "query b"]
        assert len(result["search_results"]) > 0
        assert result["strategy"] is not None
        assert isinstance(result["strategy"], StrategyResponse)

    def test_classify_called_with_input(self, mock_classify, mock_decompose, mock_search, mock_generate):
        graph = FunnelAgentGraph(mock_classify, mock_decompose, mock_search, mock_generate)
        graph.run("My brief")
        mock_classify.execute.assert_called_once_with("My brief")

    def test_decompose_called_with_profile(self, mock_classify, mock_decompose, mock_search, mock_generate):
        graph = FunnelAgentGraph(mock_classify, mock_decompose, mock_search, mock_generate)
        graph.run("brief")

        call_args = mock_decompose.execute.call_args[1]
        assert call_args["brief"] == "brief"
        assert call_args["profile"].risk_tolerance == "low"
