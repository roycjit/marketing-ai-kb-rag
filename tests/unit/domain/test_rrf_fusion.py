"""Tests for RRF fusion and outcome re-ranking — pure functions, no DB needed."""

from datetime import datetime, timezone

from domain.models import Chunk, SearchResult
from domain.services import outcome_rerank, rrf_fuse


def _make_chunk(chunk_id: str, subtype: str = "guide", outcome: float = 0.5, age_days: int = 400) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        content="...",
        source_doc="x.md",
        doc_type=subtype,
        doc_subtype=subtype,
        last_updated=datetime.now(timezone.utc) - __import__("datetime").timedelta(days=age_days),
        outcome_score=outcome,
    )


class TestRrfFuse:
    def test_both_lists_empty(self):
        assert rrf_fuse([], []) == []

    def test_single_list_only(self):
        chunks = [_make_chunk("a")]
        semantic = [SearchResult(chunk=chunks[0], similarity_score=0.9)]
        result = rrf_fuse(semantic, [])
        assert len(result) == 1
        assert result[0].chunk.chunk_id == "a"

    def test_document_in_both_gets_boosted(self):
        a = _make_chunk("a")
        b = _make_chunk("b")
        semantic = [SearchResult(chunk=a, similarity_score=0.9), SearchResult(chunk=b, similarity_score=0.8)]
        keyword = [SearchResult(chunk=b, similarity_score=0.0, keyword_score=0.7), SearchResult(chunk=a, similarity_score=0.0, keyword_score=0.6)]

        result = rrf_fuse(semantic, keyword)
        # a and b have identical RRF scores: 1/61 + 1/62 == 1/62 + 1/61
        # Both documents should appear and have higher scores than single-list minimum
        assert len(result) == 2
        assert result[0].similarity_score > 1 / 60  # more than single-list minimum
        assert result[1].similarity_score > 1 / 60

    def test_deduplicates_by_chunk_id(self):
        a = _make_chunk("a")
        semantic = [SearchResult(chunk=a, similarity_score=0.9)]
        keyword = [SearchResult(chunk=a, similarity_score=0.0, keyword_score=0.8)]

        result = rrf_fuse(semantic, keyword)
        assert len(result) == 1


class TestOutcomeRerank:
    def test_prefers_case_study(self):
        case_study = _make_chunk("cs", subtype="case_study", outcome=0.9)
        guide = _make_chunk("gd", subtype="guide", outcome=0.6)
        results = [
            SearchResult(chunk=guide, similarity_score=0.5),
            SearchResult(chunk=case_study, similarity_score=0.5),
        ]
        reranked = outcome_rerank(results, top_k=2)
        assert reranked[0].chunk.chunk_id == "cs"

    def test_prefers_recent(self):
        old = _make_chunk("old", outcome=0.9, age_days=400)
        new = _make_chunk("new", outcome=0.9, age_days=30)
        results = [
            SearchResult(chunk=old, similarity_score=0.5),
            SearchResult(chunk=new, similarity_score=0.5),
        ]
        reranked = outcome_rerank(results, top_k=2)
        assert reranked[0].chunk.chunk_id == "new"

    def test_respects_top_k(self):
        chunks = [_make_chunk(f"c{i}") for i in range(10)]
        results = [SearchResult(chunk=c, similarity_score=0.5) for c in chunks]
        reranked = outcome_rerank(results, top_k=3)
        assert len(reranked) == 3
