"""Domain entities — pure data models with no framework dependencies."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """A semantically self-contained document segment.

    Represents the core entity of the RAG system. Every chunk carries
    provenance, structural context, and quality signals for retrieval.
    """

    chunk_id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Original text content")
    embedding: list[float] = Field(default_factory=list, description="Dense vector representation")
    source_doc: str = Field(..., description="Traceability to origin document")
    doc_version: str = Field(default="1.0", description="Version disambiguation")
    section_path: str = Field(
        default="", description="Hierarchical location, e.g. 'Benefits > Eligibility'"
    )
    doc_type: str = Field(
        ..., description="Content classification: policy, procedure, case_study, guide, etc."
    )
    doc_subtype: str = Field(
        ..., description="Granular type: case_study | guide | comparison | explainer"
    )
    last_updated: datetime = Field(..., description="Freshness tracking")
    language: str = Field(default="en", description="ISO-639-1 language code")
    outcome_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Provenance quality signal from case-study metrics"
    )
    summary: str | None = Field(
        default=None, description="Concise summary for preview / BM25 boost"
    )
    question_variants: list[str] = Field(
        default_factory=list, description="Anticipated user questions this chunk answers"
    )

    model_config = ConfigDict(frozen=True)


class PsychographicProfile(BaseModel):
    """Inferred audience dimensions used to route retrieval."""

    risk_tolerance: Literal["low", "medium", "high"] = Field(
        ..., description="Audience risk tolerance level"
    )
    purchase_cycle: Literal["impulse", "short", "medium", "long"] = Field(
        ..., description="Typical purchase decision duration"
    )
    tech_savviness: Literal["low", "medium", "high"] = Field(
        ..., description="Technical sophistication level"
    )
    age_bracket: Literal["18-34", "35-54", "55+"] = Field(..., description="Target age demographic")
    price_sensitivity: Literal["low", "medium", "high"] = Field(
        ..., description="Price sensitivity level"
    )

    def to_filter_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for metadata filtering."""
        return self.model_dump()


class StrategyResponse(BaseModel):
    """Structured output from the generation layer."""

    strategy_name: str
    target_audience: str
    recommended_funnel_type: str
    key_steps: list[str]
    rationale: str
    citations: list[str] = Field(default_factory=list, description="Source document references")
    confidence: Literal["low", "medium", "high"] = Field(
        default="medium", description="Confidence level in the recommendation"
    )


class SearchResult(BaseModel):
    """Wrapper attaching retrieval scores to a Chunk."""

    chunk: Chunk
    similarity_score: float = Field(default=0.0, description="Cosine similarity or RRF score")
    keyword_score: float | None = Field(default=None, description="BM25 score when available")
