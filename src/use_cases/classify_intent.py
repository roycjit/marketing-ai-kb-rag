"""Intent classification use case — infers psychographic profile from brief."""

import json
from typing import Tuple

import structlog

from domain.models import PsychographicProfile
from frameworks.config import OLLAMA_CLASSIFICATION_MODEL
from interface_adapters.llm.ollama_client import OllamaClient
from interface_adapters.llm import prompt_templates as prompts

logger = structlog.get_logger(__name__)


class ClassifyIntentUseCase:
    """Classify psychographic intent from a user brief."""

    def __init__(self, llm_client: OllamaClient) -> None:
        self._llm = llm_client

    def execute(self, brief: str) -> PsychographicProfile:
        """Infer audience psychographics from campaign brief.

        Args:
            brief: User's campaign description (may include URL and goals).

        Returns:
            PsychographicProfile with 5 dimensions.
        """
        logger.info("intent_classification_started", brief_preview=brief[:80])

        prompt = prompts.INTENT_CLASSIFICATION_PROMPT.format(brief=brief)
        raw = self._llm.generate(
            model=OLLAMA_CLASSIFICATION_MODEL,
            prompt=prompt,
            system=prompts.INTENT_CLASSIFICATION_SYSTEM,
            temperature=0.3,
            max_tokens=256,
            json_mode=True,
        )

        try:
            data = json.loads(raw)
            profile = PsychographicProfile(**data)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("intent_classification_parse_failed", raw=raw, error=str(exc))
            # Fallback to generic profile
            profile = PsychographicProfile(
                risk_tolerance="medium",
                purchase_cycle="medium",
                tech_savviness="medium",
                age_bracket="35-54",
                price_sensitivity="medium",
            )

        logger.info("intent_classification_complete", profile=profile.model_dump())
        return profile
