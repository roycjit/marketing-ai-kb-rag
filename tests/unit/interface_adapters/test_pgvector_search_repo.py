"""Unit tests for PgVectorSearchRepository.

Requires a running PostgreSQL with pgvector.
"""

from datetime import datetime, timezone

import pytest

from domain.models import Chunk
from interface_adapters.repositories.pgvector_search_repo import (
    PgVectorSearchRepository,
)
from interface_adapters.repositories.sqlalchemy_chunk_repo import (
    SQLAlchemyChunkRepository,
)


@pytest.fixture
def seeded_db(db_session):
    """Seed the database with chunks that have distinct embeddings for search testing."""
    chunks = [
        Chunk(
            chunk_id="search-001",
            content="Solar calculator funnels achieve 150% conversion lift",
            embedding=[1.0] + [0.0] * 383,
            source_doc="solar-case-study.md",
            doc_version="1.0",
            section_path="Results",
            doc_type="case_study",
            doc_subtype="case_study",
            last_updated=datetime(2024, 12, 16, tzinfo=timezone.utc),
            language="en",
            outcome_score=0.95,
            summary="Solar case study with high conversion metrics",
            question_variants=["What conversion rate do solar calculators get?"],
        ),
        Chunk(
            chunk_id="search-002",
            content="Insurance quiz funnels need TCPA compliance checkboxes",
            embedding=[0.0, 1.0] + [0.0] * 382,
            source_doc="insurance-guide.md",
            doc_version="1.0",
            section_path="Compliance",
            doc_type="guide",
            doc_subtype="guide",
            last_updated=datetime(2025, 1, 10, tzinfo=timezone.utc),
            language="en",
            outcome_score=0.5,
            summary="Guide to insurance funnel compliance",
            question_variants=["What compliance does insurance need?"],
        ),
        Chunk(
            chunk_id="search-003",
            content="Mobile-first design loads under 2 seconds for solar funnels",
            embedding=[0.5] * 384,
            source_doc="solar-guide.md",
            doc_version="1.0",
            section_path="Design",
            doc_type="guide",
            doc_subtype="guide",
            last_updated=datetime(2026, 3, 10, tzinfo=timezone.utc),
            language="en",
            outcome_score=0.6,
            summary="Mobile design best practices for solar",
            question_variants=["How fast should a solar funnel load?"],
        ),
    ]
    SQLAlchemyChunkRepository(db_session).save_all(chunks)
    return db_session


class TestSimilaritySearch:
    def test_returns_ranked_results(self, seeded_db):
        repo = PgVectorSearchRepository(seeded_db)
        # Query embedding close to search-001
        query = [0.9] + [0.1] * 383
        results = repo.similarity_search(query, top_k=3)

        assert len(results) == 3
        # search-001 should be first (closest to [1.0, 0.0, ...])
        assert results[0].chunk.chunk_id == "search-001"
        assert results[0].similarity_score > 0.8

    def test_applies_doc_type_filter(self, seeded_db):
        repo = PgVectorSearchRepository(seeded_db)
        query = [0.5] * 384  # Close to all three
        results = repo.similarity_search(
            query, top_k=10, filters={"doc_type": "case_study"}
        )

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "search-001"


class TestKeywordSearch:
    def test_finds_by_content_keyword(self, seeded_db):
        repo = PgVectorSearchRepository(seeded_db)
        results = repo.keyword_search("solar calculator", top_k=5)

        assert len(results) >= 1
        chunk_ids = {r.chunk.chunk_id for r in results}
        assert "search-001" in chunk_ids

    def test_finds_by_summary_keyword(self, seeded_db):
        repo = PgVectorSearchRepository(seeded_db)
        results = repo.keyword_search("compliance", top_k=5)

        assert len(results) >= 1
        chunk_ids = {r.chunk.chunk_id for r in results}
        assert "search-002" in chunk_ids

    def test_respects_top_k(self, seeded_db):
        repo = PgVectorSearchRepository(seeded_db)
        results = repo.keyword_search("funnel", top_k=1)
        assert len(results) == 1
