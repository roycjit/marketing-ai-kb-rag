"""Tests for GenerateStrategyUseCase with mocked LLM."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from domain.models import Chunk, PsychographicProfile, SearchResult, StrategyResponse
from use_cases.generate_strategy import GenerateStrategyUseCase


@pytest.fixture
def mock_llm():
    client = MagicMock()
    client.generate.side_effect = [
        # First generate call
        json.dumps({
            "strategy_name": "Solar Retiree Funnel",
            "target_audience": "Risk-averse homeowners 55+",
            "recommended_funnel_type": "Multi-step qualification",
            "key_steps": ["ROI calculator", "Testimonials", "Financing options"],
            "rationale": "Retirees need trust signals before high-ticket purchases.",
            "citations": ["solar-guide.md"],
            "confidence": "high",
        }),
        # Validation call
        json.dumps({"faithful": True, "unsupported_claims": [], "suggested_fix": ""}),
    ]
    return client


@pytest.fixture
def profile():
    return PsychographicProfile(
        risk_tolerance="low",
        purchase_cycle="long",
        tech_savviness="low",
        age_bracket="55+",
        price_sensitivity="high",
    )


@pytest.fixture
def search_results():
    chunk = Chunk(
        chunk_id="c1",
        content="Solar funnels for retirees need trust signals.",
        source_doc="solar-guide.md",
        doc_type="guide",
        doc_subtype="guide",
        last_updated=datetime.now(timezone.utc),
        summary="Guide to solar funnels for older homeowners",
    )
    return [SearchResult(chunk=chunk, similarity_score=0.9)]


class TestExecute:
    def test_generates_strategy(self, mock_llm, profile, search_results):
        use_case = GenerateStrategyUseCase(mock_llm)
        strategy = use_case.execute("Sell PV to retirees", profile, search_results)

        assert strategy.strategy_name == "Solar Retiree Funnel"
        assert strategy.confidence == "high"
        assert "solar-guide.md" in strategy.citations

    def test_validates_citations(self, mock_llm, profile, search_results):
        use_case = GenerateStrategyUseCase(mock_llm)
        strategy = use_case.execute("Sell PV to retirees", profile, search_results)

        # LLM should have been called twice: generate + validate
        assert mock_llm.generate.call_count == 2

    def test_retries_on_validation_failure(self, mock_llm, profile, search_results):
        # First generate OK, validation fails, retry generate OK, validation passes
        mock_llm.generate.side_effect = [
            json.dumps({
                "strategy_name": "X",
                "target_audience": "Y",
                "recommended_funnel_type": "Z",
                "key_steps": [],
                "rationale": "Bad claim.",
                "citations": ["solar-guide.md"],
                "confidence": "high",
            }),
            json.dumps({"faithful": False, "unsupported_claims": ["Bad claim."], "suggested_fix": "Remove it"}),
            json.dumps({
                "strategy_name": "Fixed",
                "target_audience": "Y",
                "recommended_funnel_type": "Z",
                "key_steps": [],
                "rationale": "Good claim.",
                "citations": ["solar-guide.md"],
                "confidence": "high",
            }),
            json.dumps({"faithful": True, "unsupported_claims": [], "suggested_fix": ""}),
        ]

        use_case = GenerateStrategyUseCase(mock_llm)
        strategy = use_case.execute("brief", profile, search_results)

        assert strategy.strategy_name == "Fixed"
        assert mock_llm.generate.call_count == 4

    def test_fallback_on_invalid_json(self, mock_llm, profile, search_results):
        mock_llm.generate.side_effect = [
            "not json",
            json.dumps({"faithful": True, "unsupported_claims": [], "suggested_fix": ""}),
        ]
        use_case = GenerateStrategyUseCase(mock_llm)
        strategy = use_case.execute("brief", profile, search_results)

        assert strategy.confidence == "low"
        assert "Unable to generate" in strategy.strategy_name
