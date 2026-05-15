"""Tests for ClassifyIntentUseCase with mocked LLM."""

import json
from unittest.mock import MagicMock

import pytest

from domain.models import PsychographicProfile
from use_cases.classify_intent import ClassifyIntentUseCase


@pytest.fixture
def mock_llm():
    client = MagicMock()
    client.generate.return_value = json.dumps({
        "risk_tolerance": "low",
        "purchase_cycle": "long",
        "tech_savviness": "low",
        "age_bracket": "55+",
        "price_sensitivity": "high",
    })
    return client


class TestExecute:
    def test_parses_valid_json(self, mock_llm):
        use_case = ClassifyIntentUseCase(mock_llm)
        profile = use_case.execute("Sell PV systems to cautious retirees")

        assert profile.risk_tolerance == "low"
        assert profile.purchase_cycle == "long"
        assert profile.age_bracket == "55+"
        assert profile.price_sensitivity == "high"

    def test_calls_llm_with_json_mode(self, mock_llm):
        use_case = ClassifyIntentUseCase(mock_llm)
        use_case.execute("brief")

        call_kwargs = mock_llm.generate.call_args[1]
        assert call_kwargs["json_mode"] is True
        assert "marketing strategy analyst" in call_kwargs["system"]

    def test_fallback_on_invalid_json(self, mock_llm):
        mock_llm.generate.return_value = "not json"
        use_case = ClassifyIntentUseCase(mock_llm)
        profile = use_case.execute("brief")

        # Should return generic fallback profile
        assert profile.risk_tolerance == "medium"
        assert profile.purchase_cycle == "medium"
