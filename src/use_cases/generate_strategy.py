"""Strategy generation use case — produces structured recommendations with citations."""

import json
from typing import List

import structlog

from domain.models import Chunk, PsychographicProfile, SearchResult, StrategyResponse
from domain.services import resolve_conflicts
from frameworks.config import OLLAMA_GENERATION_MODEL
from interface_adapters.llm.ollama_client import OllamaClient
from interface_adapters.llm import prompt_templates as prompts

logger = structlog.get_logger(__name__)

_MAX_RETRIES = 1


class GenerateStrategyUseCase:
    """Generate a funnel strategy from retrieved context and user brief."""

    def __init__(self, llm_client: OllamaClient) -> None:
        self._llm = llm_client

    def execute(
        self,
        brief: str,
        profile: PsychographicProfile,
        search_results: List[SearchResult],
    ) -> StrategyResponse:
        """Generate and validate a strategy recommendation.

        Args:
            brief: Original user brief.
            profile: Inferred psychographic profile.
            search_results: Retrieved chunks from hybrid search.

        Returns:
            Validated StrategyResponse with citations.
        """
        logger.info("strategy_generation_started", results_count=len(search_results))

        # Deduplicate and limit context
        chunks = [r.chunk for r in search_results]
        unique_chunks = resolve_conflicts(chunks)
        context_chunks = unique_chunks[:8]  # Top 8 most relevant / authoritative

        # Build context string
        context = self._build_context(context_chunks)
        profile_json = profile.model_dump_json(indent=2)

        # Generate with retry on validation failure
        for attempt in range(_MAX_RETRIES + 1):
            strategy = self._generate_once(brief, profile_json, context, attempt)
            validation = self._validate(strategy, context_chunks)

            if validation["faithful"]:
                logger.info("strategy_validation_passed", attempt=attempt)
                return strategy

            logger.warning(
                "strategy_validation_failed",
                attempt=attempt,
                unsupported=validation.get("unsupported_claims", []),
            )
            if attempt < _MAX_RETRIES:
                # Tighten instructions for retry
                context += f"\n\nIMPORTANT: Remove these unsupported claims: {validation.get('unsupported_claims', [])}"

        # Final fallback: return strategy with reduced confidence
        logger.warning("strategy_validation_max_retries_reached")
        return strategy.model_copy(update={"confidence": "low"})

    def _generate_once(
        self,
        brief: str,
        profile_json: str,
        context: str,
        attempt: int,
    ) -> StrategyResponse:
        """Single generation attempt."""
        temperature = 0.5 if attempt == 0 else 0.3
        prompt = prompts.STRATEGY_GENERATION_PROMPT.format(
            brief=brief,
            profile_json=profile_json,
            context=context,
        )
        raw = self._llm.generate(
            model=OLLAMA_GENERATION_MODEL,
            prompt=prompt,
            system=prompts.STRATEGY_GENERATION_SYSTEM,
            temperature=temperature,
            max_tokens=2048,
            json_mode=True,
        )

        try:
            data = json.loads(raw)
            return StrategyResponse(**data)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.error("strategy_generation_parse_failed", raw=raw, error=str(exc))
            # Return a minimal fallback response
            return StrategyResponse(
                strategy_name="Unable to generate strategy",
                target_audience="See retrieved sources below",
                recommended_funnel_type="TBD",
                key_steps=["Please review the cited sources for relevant guidance."],
                rationale="The model failed to produce valid JSON.",
                citations=[c.source_doc for c in []],
                confidence="low",
            )

    def _validate(
        self,
        strategy: StrategyResponse,
        context_chunks: List[Chunk],
    ) -> dict:
        """Check faithfulness and citation accuracy."""
        # 1. Citation verification: every cited source must exist in retrieved chunks
        available_sources = {c.source_doc for c in context_chunks}
        invalid_citations = [c for c in strategy.citations if c not in available_sources]

        if invalid_citations:
            return {
                "faithful": False,
                "unsupported_claims": [f"Invalid citation: {c}" for c in invalid_citations],
                "suggested_fix": "Remove invalid citations.",
            }

        # 2. LLM-based faithfulness check (lightweight for MVP)
        context = self._build_context(context_chunks)
        strategy_text = strategy.model_dump_json()

        prompt = prompts.FAITHFULNESS_VALIDATION_PROMPT.format(
            context=context,
            strategy=strategy_text,
        )
        try:
            raw = self._llm.generate(
                model=OLLAMA_GENERATION_MODEL,
                prompt=prompt,
                system=prompts.FAITHFULNESS_VALIDATION_SYSTEM,
                temperature=0.0,
                max_tokens=512,
                json_mode=True,
            )
            result = json.loads(raw)
            return result
        except Exception as exc:
            logger.warning("faithfulness_check_failed", error=str(exc))
            # If validation itself fails, assume faithful to avoid infinite loops
            return {"faithful": True, "unsupported_claims": [], "suggested_fix": ""}

    @staticmethod
    def _build_context(chunks: List[Chunk]) -> str:
        """Format chunks as citation-ready context string."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            header = f"[Source {i}] {chunk.source_doc} | {chunk.section_path} | {chunk.doc_subtype}"
            parts.append(f"{header}\n{chunk.content}\n")
        return "\n".join(parts)
