"""Tests for IngestDocumentsUseCase with mocked dependencies."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from domain.models import Chunk
from use_cases.ingest_documents import IngestDocumentsUseCase


@pytest.fixture
def mock_parser():
    parser = MagicMock()
    from interface_adapters.parsers.markdown_parser import ParsedDocument

    parser.parse_directory.return_value = [
        ParsedDocument(
            source_doc="doc1.md",
            title="Doc One",
            body="## Section A\n\nContent here.",
            metadata={"date": "2024-01-01"},
        ),
    ]
    return parser


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1] * 384]
    return embedder


@pytest.fixture
def mock_repository():
    repo = MagicMock()
    repo.save_all.return_value = None
    return repo


class TestExecute:
    def test_full_pipeline(self, mock_parser, mock_embedder, mock_repository, tmp_path: Path):
        use_case = IngestDocumentsUseCase(
            parser=mock_parser,
            embedder=mock_embedder,
            repository=mock_repository,
        )
        count = use_case.execute(tmp_path)

        assert count > 0
        mock_parser.parse_directory.assert_called_once_with(tmp_path)
        mock_embedder.embed.assert_called_once()
        mock_repository.save_all.assert_called_once()

        # Verify saved chunks have embeddings and scores
        saved_chunks = mock_repository.save_all.call_args[0][0]
        assert all(len(c.embedding) == 384 for c in saved_chunks)
        assert all(c.outcome_score > 0 for c in saved_chunks)

    def test_empty_directory(self, mock_parser, mock_embedder, mock_repository, tmp_path: Path):
        mock_parser.parse_directory.return_value = []

        use_case = IngestDocumentsUseCase(
            parser=mock_parser,
            embedder=mock_embedder,
            repository=mock_repository,
        )
        count = use_case.execute(tmp_path)

        assert count == 0
        mock_embedder.embed.assert_not_called()
        mock_repository.save_all.assert_not_called()
