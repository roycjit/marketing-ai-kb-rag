"""Query decomposition use case — breaks a brief into retrieval sub-queries."""

from __future__ import annotations

import json

import structlog

from domain.models import PsychographicProfile
from domain.validation import sanitize_brief
from frameworks.config import OLLAMA_CLASSIFICATION_MODEL
from interface_adapters.llm import prompt_templates as prompts
from interface_adapters.llm.ollama_client import OllamaClient

logger = structlog.get_logger(__name__)


class DecomposeQueriesUseCase:
    """Decompose a campaign brief into focused retrieval sub-queries."""

    def __init__(self, llm_client: OllamaClient) -> None:
        """Initialize with an LLM client."""
        self._llm = llm_client

    def execute(
        self,
        brief: str,
        profile: PsychographicProfile,
    ) -> list[str]:
        """Generate 2–4 sub-queries tailored to the audience profile.

        Args:
            brief: Original user brief.
            profile: Inferred psychographic profile.

        Returns:
            List of sub-query strings.
        """
        safe_brief = sanitize_brief(brief)
        logger.info("query_decomposition_started")

        prompt = prompts.QUERY_DECOMPOSITION_PROMPT.format(
            brief=safe_brief,
            risk_tolerance=profile.risk_tolerance,
            purchase_cycle=profile.purchase_cycle,
            tech_savviness=profile.tech_savviness,
            age_bracket=profile.age_bracket,
            price_sensitivity=profile.price_sensitivity,
        )
        raw = self._llm.generate(
            model=OLLAMA_CLASSIFICATION_MODEL,
            prompt=prompt,
            system=prompts.QUERY_DECOMPOSITION_SYSTEM,
            temperature=0.4,
            max_tokens=256,
            json_mode=True,
        )

        try:
            data = json.loads(raw)
            if isinstance(data, list):
                sub_queries = [str(q) for q in data]
            elif isinstance(data, dict) and "queries" in data:
                sub_queries = [str(q) for q in data["queries"]]
            else:
                sub_queries = []
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("query_decomposition_parse_failed", raw=raw, error=str(exc))
            sub_queries = []

        if not sub_queries:
            # Fallback: use the brief itself as a single query
            sub_queries = [safe_brief]

        logger.info("query_decomposition_complete", count=len(sub_queries), queries=sub_queries)
        return sub_queries
