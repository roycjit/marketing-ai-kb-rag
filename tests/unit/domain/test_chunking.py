"""Tests for chunk_document and helpers."""

from datetime import datetime, timezone

import pytest

from domain.models import Chunk
from domain.services import chunk_document, _infer_doc_subtype, _infer_language


class TestInferDocSubtype:
    def test_case_study_from_filename(self):
        assert _infer_doc_subtype("case-study-1komma5grad.md", {}) == "case_study"

    def test_case_study_from_category(self):
        assert _infer_doc_subtype("x.md", {"category": "Customer stories"}) == "explainer"
        assert _infer_doc_subtype("x.md", {"category": "Case Study"}) == "case_study"

    def test_guide_from_filename(self):
        assert _infer_doc_subtype("best-practices-guide.md", {}) == "guide"

    def test_comparison_from_filename(self):
        assert _infer_doc_subtype("clickfunnels-alternatives.md", {}) == "comparison"

    def test_default_explainer(self):
        assert _infer_doc_subtype("random-post.md", {}) == "explainer"


class TestInferLanguage:
    def test_english_content(self):
        text = "Solar panels are a great investment for homeowners."
        assert _infer_language(text) == "en"

    def test_german_content(self):
        text = "Die Photovoltaik ist eine gute Investition für Hausbesitzer."
        assert _infer_language(text) == "de"

    def test_mixed_defaults_to_english(self):
        text = "Solar panels are great and die Sonne scheint."
        assert _infer_language(text) == "en"


class TestChunkDocument:
    def test_splits_by_headers(self):
        body = "## Section A\n\nParagraph one.\n\n## Section B\n\nParagraph two."
        chunks = chunk_document(
            source_doc="test.md",
            title="Test Doc",
            body=body,
            metadata={},
        )

        assert len(chunks) >= 2
        section_paths = [c.section_path for c in chunks]
        assert any("Section A" in sp for sp in section_paths)
        assert any("Section B" in sp for sp in section_paths)

    def test_preserves_table(self):
        body = "## Comparison\n\n| Feature | A | B |\n| --- | --- | --- |\n| X | 1 | 2 |"
        chunks = chunk_document(
            source_doc="comparison.md",
            title="Comparison",
            body=body,
            metadata={},
        )

        assert len(chunks) == 1
        assert "| Feature |" in chunks[0].content

    def test_creates_unique_chunk_ids(self):
        body = "## Sec A\n\nText.\n\n## Sec B\n\nMore text."
        chunks = chunk_document(
            source_doc="doc.md",
            title="Doc",
            body=body,
            metadata={},
        )

        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_assigns_correct_subtype(self):
        chunks = chunk_document(
            source_doc="case-study-xyz.md",
            title="Case Study",
            body="## Results\n\nGreat results.",
            metadata={},
        )
        assert all(c.doc_subtype == "case_study" for c in chunks)

    def test_empty_body_returns_empty(self):
        chunks = chunk_document(
            source_doc="empty.md",
            title="Empty",
            body="",
            metadata={},
        )
        assert chunks == []
