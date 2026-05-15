"""Domain entities — pure data models with no framework dependencies."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A semantically self-contained document segment.

    Represents the core entity of the RAG system. Every chunk carries
    provenance, structural context, and quality signals for retrieval.
    """

    chunk_id: str = Field(..., description="UUID v4")
    content: str = Field(..., description="Original text content")
    embedding: List[float] = Field(default_factory=list, description="Dense vector representation")
    source_doc: str = Field(..., description="Traceability to origin document")
    doc_version: str = Field(default="1.0", description="Version disambiguation")
    section_path: str = Field(default="", description="Hierarchical location, e.g. 'Benefits > Eligibility'")
    doc_type: str = Field(..., description="Content classification: policy, procedure, case_study, guide, etc.")
    doc_subtype: str = Field(..., description="Granular type: case_study | guide | comparison | explainer")
    last_updated: datetime = Field(..., description="Freshness tracking")
    language: str = Field(default="en", description="ISO-639-1 language code")
    outcome_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Provenance quality signal from case-study metrics")
    summary: Optional[str] = Field(default=None, description="Concise summary for preview / BM25 boost")
    question_variants: List[str] = Field(default_factory=list, description="Anticipated user questions this chunk answers")

    class Config:
        frozen = True


class PsychographicProfile(BaseModel):
    """Inferred audience dimensions used to route retrieval."""

    risk_tolerance: str = Field(..., description="low | medium | high")
    purchase_cycle: str = Field(..., description="impulse | short | medium | long")
    tech_savviness: str = Field(..., description="low | medium | high")
    age_bracket: str = Field(..., description="18-34 | 35-54 | 55+")
    price_sensitivity: str = Field(..., description="low | medium | high")

    def to_filter_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for metadata filtering."""
        return self.model_dump()


class StrategyResponse(BaseModel):
    """Structured output from the generation layer."""

    strategy_name: str
    target_audience: str
    recommended_funnel_type: str
    key_steps: List[str]
    rationale: str
    citations: List[str] = Field(default_factory=list, description="Source document references")
    confidence: str = Field(default="medium", description="low | medium | high")


class SearchResult(BaseModel):
    """Wrapper attaching retrieval scores to a Chunk."""

    chunk: Chunk
    similarity_score: float = Field(default=0.0, description="Cosine similarity or RRF score")
    keyword_score: Optional[float] = Field(default=None, description="BM25 score when available")
