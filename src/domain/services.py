"""Pure domain services — no I/O, no framework dependencies.

These functions encapsulate business rules that don't naturally belong
to a single entity.
"""

from __future__ import annotations

from datetime import UTC, datetime

from domain.chunking import chunk_document  # noqa: F401  # re-export for backward compat
from domain.models import Chunk, SearchResult

# Document subtype hierarchy for conflict resolution.
# Higher index = higher authority.
_SUBTYPE_PRECEDENCE = ["explainer", "comparison", "guide", "case_study"]


def compute_outcome_score(chunk: Chunk) -> float:
    """Derive a quality signal from document metadata.

    Case studies with explicit metrics score highest.
    Guides and comparisons score moderately.
    Generic explainers score lowest.

    Returns:
        Float in [0.0, 1.0].
    """
    base_score = {
        "case_study": 0.9,
        "guide": 0.6,
        "comparison": 0.5,
        "explainer": 0.3,
    }.get(chunk.doc_subtype, 0.3)

    # Small recency bonus: documents from the last 12 months get +0.1
    if chunk.last_updated:
        now = datetime.now(UTC)
        # Ensure chunk.last_updated is timezone-aware for subtraction
        chunk_dt = chunk.last_updated
        if chunk_dt.tzinfo is None:
            chunk_dt = chunk_dt.replace(tzinfo=UTC)
        age_days = (now - chunk_dt).days
        if age_days < 365:
            base_score = min(1.0, base_score + 0.1)

    return round(base_score, 2)


def resolve_conflicts(chunks: list[Chunk]) -> list[Chunk]:
    """When multiple chunks cover the same topic, prefer authoritative sources.

    Resolution rules (in order):
    1. Higher subtype precedence (case_study > guide > comparison > explainer)
    2. More recent last_updated
    3. Higher outcome_score

    Args:
        chunks: Potentially overlapping chunks from different documents.

    Returns:
        Deduplicated list with conflicts resolved.
    """
    if not chunks:
        return []

    def _sort_key(c: Chunk) -> tuple:
        precedence = (
            _SUBTYPE_PRECEDENCE.index(c.doc_subtype) if c.doc_subtype in _SUBTYPE_PRECEDENCE else -1
        )
        return (
            precedence,
            c.last_updated.timestamp() if c.last_updated else 0,
            c.outcome_score,
        )

    # Group by source_doc and section_path to detect overlap
    seen: dict[str, Chunk] = {}
    for chunk in chunks:
        key = f"{chunk.source_doc}::{chunk.section_path}"
        if key not in seen:
            seen[key] = chunk
        else:
            existing = seen[key]
            if _sort_key(chunk) > _sort_key(existing):
                seen[key] = chunk

    return list(seen.values())


# --------------------------------------------------------------------------- #
# Retrieval Services
# --------------------------------------------------------------------------- #


def rrf_fuse(
    semantic_results: list[SearchResult],
    keyword_results: list[SearchResult],
    k: int = 60,
) -> list[SearchResult]:
    """Fuse semantic and keyword results via Reciprocal Rank Fusion.

    RRF_score(d) = sum_m 1 / (k + rank_m(d))

    Documents present in both result sets receive boosted scores.

    Args:
        semantic_results: Ranked list from vector similarity search.
        keyword_results: Ranked list from full-text keyword search.
        k: RRF constant (standard value = 60).

    Returns:
        Fused and re-ranked list of search results.
    """
    from domain.models import SearchResult

    scores: dict[str, float] = {}
    result_map: dict[str, SearchResult] = {}

    for rank, sr in enumerate(semantic_results, start=1):
        cid = sr.chunk.chunk_id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        result_map[cid] = sr

    for rank, sr in enumerate(keyword_results, start=1):
        cid = sr.chunk.chunk_id
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in result_map:
            result_map[cid] = sr

    # Sort by RRF score descending
    sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [
        SearchResult(
            chunk=result_map[cid].chunk,
            similarity_score=scores[cid],
            keyword_score=result_map[cid].keyword_score,
        )
        for cid in sorted_ids
    ]


def outcome_rerank(results: list[SearchResult], top_k: int = 10) -> list[SearchResult]:
    """Re-rank fused results using document-encoded outcome signals.

    Boosts documents by:
    1. Outcome score (case studies with metrics)
    2. Recency (documents from last 12 months)
    3. Document subtype precedence (case_study > guide > comparison > explainer)

    The boost is additive to the RRF score to preserve the fused ranking
    while surfacing higher-quality content.
    """

    def _boost(sr: SearchResult) -> float:
        base: float = sr.similarity_score
        chunk = sr.chunk

        # Outcome score boost: up to +0.15
        base += chunk.outcome_score * 0.15

        # Recency boost: +0.05 for documents < 365 days old
        if chunk.last_updated:
            now = datetime.now(UTC)
            chunk_dt = chunk.last_updated
            if chunk_dt.tzinfo is None:
                chunk_dt = chunk_dt.replace(tzinfo=UTC)
            age_days = (now - chunk_dt).days
            if age_days < 365:
                base += 0.05

        # Subtype boost
        subtype_boost: float = {
            "case_study": 0.10,
            "guide": 0.05,
            "comparison": 0.02,
            "explainer": 0.0,
        }.get(chunk.doc_subtype, 0.0)
        base += subtype_boost

        return base

    boosted = sorted(results, key=_boost, reverse=True)
    return boosted[:top_k]



