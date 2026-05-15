"""Unit tests for SQLAlchemyChunkRepository.

Requires a running PostgreSQL with pgvector (see docker-compose.yml).
Tests use transactional rollback — no persistent side effects.
"""

from datetime import datetime, timezone

import pytest

from domain.models import Chunk
from domain.services import compute_outcome_score
from interface_adapters.repositories.orm_models import ChunkORM
from interface_adapters.repositories.sqlalchemy_chunk_repo import (
    SQLAlchemyChunkRepository,
)


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="chunk-001",
            content="Solar savings calculators convert 3x better than static forms.",
            embedding=[0.1] * 384,
            source_doc="case-study-1komma5grad.md",
            doc_version="1.0",
            section_path="Results",
            doc_type="case_study",
            doc_subtype="case_study",
            last_updated=datetime(2024, 12, 16, tzinfo=timezone.utc),
            language="en",
            outcome_score=0.95,
            summary="1KOMMA5° achieved 150% conversion increase with Heyflow.",
            question_variants=["How did 1KOMMA5° improve conversions?"],
        ),
        Chunk(
            chunk_id="chunk-002",
            content="Multi-step forms reduce dropout rates by 40%.",
            embedding=[0.2] * 384,
            source_doc="funnel-builder-renewable-energy-sector.md",
            doc_version="1.0",
            section_path="Best Practices",
            doc_type="guide",
            doc_subtype="guide",
            last_updated=datetime(2026, 3, 10, tzinfo=timezone.utc),
            language="en",
            outcome_score=0.6,
            summary="Guide to building renewable energy funnels.",
            question_variants=["What funnel type works for solar?"],
        ),
    ]


class TestSaveAndRetrieve:
    def test_save_all_persists_chunks(self, db_session, sample_chunks):
        repo = SQLAlchemyChunkRepository(db_session)
        repo.save_all(sample_chunks)

        orms = db_session.query(ChunkORM).all()
        assert len(orms) == 2
        assert {orm.chunk_id for orm in orms} == {"chunk-001", "chunk-002"}

    def test_get_by_id_returns_chunk(self, db_session, sample_chunks):
        repo = SQLAlchemyChunkRepository(db_session)
        repo.save_all(sample_chunks)

        result = repo.get_by_id("chunk-001")
        assert result is not None
        assert result.chunk_id == "chunk-001"
        assert result.doc_subtype == "case_study"

    def test_get_by_id_returns_none_for_missing(self, db_session):
        repo = SQLAlchemyChunkRepository(db_session)
        assert repo.get_by_id("nonexistent") is None

    def test_list_by_source_filters_correctly(self, db_session, sample_chunks):
        repo = SQLAlchemyChunkRepository(db_session)
        repo.save_all(sample_chunks)

        results = repo.list_by_source("case-study-1komma5grad.md")
        assert len(results) == 1
        assert results[0].chunk_id == "chunk-001"
