"""Tests for DecomposeQueriesUseCase with mocked LLM."""

import json
from unittest.mock import MagicMock

import pytest

from domain.models import PsychographicProfile
from use_cases.decompose_queries import DecomposeQueriesUseCase


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
def mock_llm():
    client = MagicMock()
    client.generate.return_value = json.dumps([
        "low-risk solar funnel for retirees",
        "high-trust lead qualification for cautious buyers",
        "long-cycle nurture strategy for expensive purchases",
    ])
    return client


class TestExecute:
    def test_returns_sub_queries(self, mock_llm, profile):
        use_case = DecomposeQueriesUseCase(mock_llm)
        queries = use_case.execute("Sell PV to retirees", profile)

        assert len(queries) == 3
        assert any("retire" in q.lower() for q in queries)
        assert any("trust" in q.lower() for q in queries)

    def test_includes_profile_in_prompt(self, mock_llm, profile):
        use_case = DecomposeQueriesUseCase(mock_llm)
        use_case.execute("brief", profile)

        prompt = mock_llm.generate.call_args[1]["prompt"]
        assert "low" in prompt  # risk_tolerance
        assert "55+" in prompt  # age_bracket

    def test_fallback_on_invalid_json(self, mock_llm, profile):
        mock_llm.generate.return_value = "bad response"
        use_case = DecomposeQueriesUseCase(mock_llm)
        queries = use_case.execute("brief", profile)

        assert queries == ["brief"]  # fallback to original brief
