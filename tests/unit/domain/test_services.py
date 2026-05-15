"""Unit tests for pure domain services — no database required."""

from datetime import datetime, timezone

from domain.models import Chunk
from domain.services import compute_outcome_score, resolve_conflicts


class TestComputeOutcomeScore:
    def test_case_study_scores_highest(self):
        chunk = Chunk(
            chunk_id="c1",
            content="...",
            source_doc="x.md",
            doc_type="case_study",
            doc_subtype="case_study",
            last_updated=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert compute_outcome_score(chunk) == 0.9

    def test_guide_scores_lower(self):
        chunk = Chunk(
            chunk_id="c1",
            content="...",
            source_doc="x.md",
            doc_type="guide",
            doc_subtype="guide",
            last_updated=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert compute_outcome_score(chunk) == 0.6

    def test_recency_boosts_score(self):
        chunk = Chunk(
            chunk_id="c1",
            content="...",
            source_doc="x.md",
            doc_type="case_study",
            doc_subtype="case_study",
            last_updated=datetime.now(timezone.utc),
        )
        assert compute_outcome_score(chunk) == 1.0


class TestResolveConflicts:
    def test_prefers_case_study_over_guide(self):
        case_study = Chunk(
            chunk_id="cs",
            content="Case study data",
            source_doc="same.md",
            doc_type="case_study",
            doc_subtype="case_study",
            last_updated=datetime(2024, 1, 1, tzinfo=timezone.utc),
            section_path="Section A",
        )
        guide = Chunk(
            chunk_id="gd",
            content="Guide data",
            source_doc="same.md",
            doc_type="guide",
            doc_subtype="guide",
            last_updated=datetime(2025, 1, 1, tzinfo=timezone.utc),
            section_path="Section A",
        )
        result = resolve_conflicts([guide, case_study])
        assert len(result) == 1
        assert result[0].chunk_id == "cs"  # case_study wins despite being older

    def test_prefers_newer_when_same_subtype(self):
        old = Chunk(
            chunk_id="old",
            content="Old data",
            source_doc="same.md",
            doc_type="guide",
            doc_subtype="guide",
            last_updated=datetime(2024, 1, 1, tzinfo=timezone.utc),
            section_path="Section A",
        )
        new = Chunk(
            chunk_id="new",
            content="New data",
            source_doc="same.md",
            doc_type="guide",
            doc_subtype="guide",
            last_updated=datetime(2025, 1, 1, tzinfo=timezone.utc),
            section_path="Section A",
        )
        result = resolve_conflicts([old, new])
        assert len(result) == 1
        assert result[0].chunk_id == "new"
