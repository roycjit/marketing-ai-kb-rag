"""Tests for HybridSearchUseCase with mocked repository."""

from unittest.mock import MagicMock

import pytest

from domain.models import Chunk, SearchResult
from use_cases.search_chunks import HybridSearchUseCase


@pytest.fixture
def mock_repo():
    repo = MagicMock()
    a = Chunk(chunk_id="a", content="...", source_doc="x.md", doc_type="guide", doc_subtype="guide", last_updated=__import__("datetime").datetime.now(__import__("datetime").timezone.utc))
    b = Chunk(chunk_id="b", content="...", source_doc="y.md", doc_type="case_study", doc_subtype="case_study", last_updated=__import__("datetime").datetime.now(__import__("datetime").timezone.utc))
    repo.similarity_search.return_value = [SearchResult(chunk=a, similarity_score=0.9)]
    repo.keyword_search.return_value = [SearchResult(chunk=b, similarity_score=0.0, keyword_score=0.8)]
    return repo


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed_single.return_value = [0.1] * 384
    return embedder


class TestExecute:
    def test_calls_both_searches(self, mock_repo, mock_embedder):
        use_case = HybridSearchUseCase(mock_repo, mock_embedder)
        results = use_case.execute("solar funnel", top_k=5)

        mock_repo.similarity_search.assert_called_once()
        mock_repo.keyword_search.assert_called_once()
        mock_embedder.embed_single.assert_called_once_with("solar funnel")

    def test_returns_reranked_results(self, mock_repo, mock_embedder):
        use_case = HybridSearchUseCase(mock_repo, mock_embedder)
        results = use_case.execute("solar funnel", top_k=5)

        assert len(results) == 2
        # Both a and b should be present (from different searches)
        ids = {r.chunk.chunk_id for r in results}
        assert ids == {"a", "b"}

    def test_applies_filters(self, mock_repo, mock_embedder):
        use_case = HybridSearchUseCase(mock_repo, mock_embedder)
        use_case.execute("query", top_k=5, filters={"language": "en"})

        call_kwargs = mock_repo.similarity_search.call_args[1]
        assert call_kwargs["filters"] == {"language": "en"}
